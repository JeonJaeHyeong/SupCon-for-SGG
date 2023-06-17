# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)

import argparse
import datetime
import os
import sys
sys.path.append(os.getcwd())
import random
import time
import threading
import json

import gpustat
import numpy as np
import torch
import torch.nn.functional as F

from pysgg.config import cfg
from pysgg.data import make_data_loader, make_contrastive_loader
from pysgg.engine.inference import inference
from pysgg.engine.trainer import reduce_loss_dict
from pysgg.modeling.detector import build_detection_model
from pysgg.solver import make_lr_scheduler
from pysgg.solver import make_optimizer
from pysgg.utils.checkpoint import DetectronCheckpointer
from pysgg.utils.checkpoint import clip_grad_norm
from pysgg.utils import visualize_graph as vis_graph
from pysgg.utils.collect_env import collect_env_info
from pysgg.utils.comm import synchronize, get_rank, all_gather
from pysgg.utils.logger import setup_logger, debug_print, TFBoardHandler_LEVEL
from pysgg.utils.metric_logger import MetricLogger
from pysgg.utils.miscellaneous import mkdir, save_config
from pysgg.utils.global_buffer import save_buffer
from pysgg.utils.comm import get_world_size

from pysgg.modeling.roi_heads.relation_head.model_contrastive import ResNetSimCLR
# See if we can use apex.DistributedDataParallel instead of the torch default,
# and enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError("Use APEX for multi-precision via apex.amp")

SEED = 666

torch.cuda.manual_seed(SEED)  # 현재 GPU에 대한 임의 시드 설정
torch.cuda.manual_seed_all(SEED)  # 모든 GPU에 대해 임의 시드 설정
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

torch.backends.cudnn.enabled = True  # 기본값
torch.backends.cudnn.benchmark = True  # 기본값은 False
torch.backends.cudnn.deterministic = True  # 기본값은 False. 벤치마크가 True일 때 임의성을 제외하려면 y가 True여야 합니다.


torch.autograd.set_detect_anomaly(True)

SHOW_COMP_GRAPH = False


def show_params_status(model):
    """
    Prints parameters of a model
    """
    st = {}
    strings = []
    total_params = 0
    trainable_params = 0
    for p_name, p in model.named_parameters():

        if not ("bias" in p_name.split(".")[-1] or "bn" in p_name.split(".")[-1]):
            st[p_name] = ([str(x) for x in p.size()], np.prod(p.size()), p.requires_grad)
        total_params += np.prod(p.size())
        if p.requires_grad:
            trainable_params += np.prod(p.size())
    for p_name, (size, prod, p_req_grad) in st.items():
        strings.append(
            "{:<80s}: {:<16s}({:8d}) ({})".format(
                p_name, "[{}]".format(",".join(size)), prod, "grad" if p_req_grad else "    "
            )
        )
    strings = "\n".join(strings)
    return (
        f"\n{strings}\n ----- \n \n"
        f"      trainable parameters:  {trainable_params/ 1e6:.3f}/{total_params / 1e6:.3f} M \n "
    )


def train(
    cfg,
    local_rank,
    distributed,
    logger,
):
    global SHOW_COMP_GRAPH

    debug_print(logger, "Start initializing dataset & dataloader")

    arguments = {}
    arguments["iteration"] = 0
    output_dir = cfg.OUTPUT_DIR
    contrastive = cfg.MODEL.CONTRASTIVE.LEARNING
    epochs = cfg.MODEL.CONTRASTIVE.EPOCH
    '''
    train_data_loader = make_data_loader(
        cfg,
        mode="train",
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    val_data_loaders = make_data_loader(
        cfg,
        mode="val",
        is_distributed=distributed,5
    )
    '''
    train_data_loader = make_contrastive_loader(
        cfg,
        mode="train",
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    debug_print(logger, "end dataloader")
    
    debug_print(logger, "prepare training")
    out_dim = cfg.MODEL.CONTRASTIVE.OUT_DIM
    lr = cfg.MODEL.CONTRASTIVE.LEARNING_RATE
    wd = cfg.MODEL.CONTRASTIVE.WEIGHT_DECAY
    model = ResNetSimCLR(cfg=cfg)
    model.train()
    debug_print(logger, "end model construction")
    logger.info(str(model))
    # modules that should be always set in eval mode
    # their eval() method should be called after model.train() is called
    
    eval_modules = (
        #model.rpn,
        #model.backbone,
        #model.box_feature_extractor,
    )
    
    fix_eval_modules(eval_modules)

    logger.info("trainable models:")
    logger.info(show_params_status(model))


    # load pretrain layers to new layers
    load_mapping = {
        "box_feature_extractor": "roi_heads.box.feature_extractor",
    }

    print("load model to GPU")
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=wd)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_data_loader), eta_min=0,
                                                           last_epoch=-1)
    debug_print(logger, "end optimizer and schedule")
    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = "O1" if use_mixed_precision else "O0"
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    )
    # todo, unless mark as resume, otherwise load from the pretrained checkpoint
    if cfg.MODEL.PRETRAINED_DETECTOR_CKPT != "":
        checkpointer.load(
            cfg.MODEL.PRETRAINED_DETECTOR_CKPT, with_optim=False, load_mapping=load_mapping
        )
    else:
        checkpointer.load(
            cfg.MODEL.WEIGHT,
            with_optim=False,
        )

    checkpoint_period = cfg.MODEL.CONTRASTIVE.CHECKPOINT_PERIOD
    debug_print(logger, "end load checkpointer")

    #if cfg.MODEL.ROI_RELATION_HEAD.RE_INITIALIZE_CLASSIFIER:
    #    model.roi_heads.relation.predictor.init_classifier_weight()

    # preserve a reference for logging
    #rel_model_ref = model.roi_heads.relation

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    debug_print(logger, "end distributed")

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()

    model.train()
    temperature = cfg.MODEL.CONTRASTIVE.TEMPERATURE
    n_iter = 0
    
    def sup_contrast_loss(ori_feats, pos_feats, neg_feats):
        
        num_pos = cfg.MODEL.CONTRASTIVE.NUM_POS_SAMPLES
        num_neg = cfg.MODEL.CONTRASTIVE.NUM_NEG_SAMPLES
        batch_size = len(ori_feats)
        
        add_losses = {}
        for b in range(batch_size):
            ori_feats[b] = F.normalize(ori_feats[b], dim=1)
            pos_feats[b] = F.normalize(pos_feats[b], dim=1)
            neg_feats[b] = F.normalize(neg_feats[b], dim=1)
            
            sim_matrix_pos = torch.div(
                torch.matmul(ori_feats[b], pos_feats[b].T), 
                temperature)
            sim_matrix_neg = torch.div(
                torch.matmul(ori_feats[b], neg_feats[b].T), 
                temperature)
            #logits_max_pos, _ = torch.max(sim_matrix_pos, dim=1, keepdim=True) 
            #logits_max_neg, _ = torch.max(sim_matrix_neg, dim=1, keepdim=True)
            
            #sim_matrix_pos = torch.exp(sim_matrix_pos - logits_max_pos.detach())
            #sim_matrix_neg = torch.exp(sim_matrix_neg - logits_max_neg.detach())
            
            exp_pos = torch.exp(sim_matrix_pos)
            exp_neg = torch.exp(sim_matrix_neg)
            
            for rel in range(len(ori_feats[b])):
                positive = sim_matrix_pos[rel, rel*num_pos:(rel+1)*num_pos]
                log_sum_n = torch.log(torch.sum(exp_neg[rel, rel*num_neg:(rel+1)*num_neg]) +
                                        torch.sum(exp_pos[rel, rel*num_pos:(rel+1)*num_pos])+1e-6)
                loss = torch.sum(positive - log_sum_n)
                loss = torch.div(loss, num_pos)
                loss = torch.div(loss, len(ori_feats[b]))
                loss = torch.div(loss, batch_size)
                add_losses[f"loss for batch : {b}, relation : {rel}"] = - loss
                
        return add_losses

    print_first_grad = True
    
    debug_print(logger, f"Start SimCLR training for {cfg.MODEL.CONTRASTIVE.EPOCH} epochs.")
    for e in range(epochs):
        if e != 0:
            train_data_loader = make_contrastive_loader(
                cfg,
                mode="train",
                is_distributed=distributed,
                start_iter=arguments["iteration"],
            )
        average_loss = 0
        count = 0
        
        for iteration, (oris, poss, negs) in enumerate(train_data_loader):
        #for iteration, (poss, negs) in enumerate(train_data_loader):
            iteration = iteration + 1
            data_time = time.time() - end  
                      
                      
            '''
            for b in range(len(poss)):
                
                use_object = cfg.MODEL.CONTRASTIVE.USE_OBJECT_INFO
                
                if use_object:
                    sub_pos, obj_pos = poss[b][0], poss[b][1]
                    sub_neg, obj_neg = negs[b][0], negs[b][1]
                
                    for r in range(len(poss[b][0])):
                        assert len(sub_pos) == cfg.MODEL.CONTRASTIVE.MAX_REL_NUM
                        
                        for ep in range(len(sub_pos[r])):
                            assert len(sub_pos[r][0]) == cfg.MODEL.CONTRASTIVE.NUM_POS_SAMPLES
                            imgs, targets = sub_pos[r]
                            imgs = [ img.to(device) for img in imgs]
                            result = model(imgs, targets)
                            
                            
                        for en in range(len(sub_neg[r])):
                            assert len(sub_neg[r]) == cfg.MODEL.CONTRASTIVE.NUM_NEG_SAMPLES
            '''
                      
            Doris = [[[ rel[0].to(device), rel[1].to(device) ] for rel in batch ] for batch in oris ] 
            Dposs = [[[ rel[0].to(device), rel[1].to(device) ] for rel in batch ] for batch in poss ] 
            Dnegs = [[[ rel[0].to(device), rel[1].to(device) ] for rel in batch ] for batch in negs ] 
            
            ori_rel_feats = model(Doris) # batch size list & each torch.Size([5, 128])
            pos_rel_feats = model(Dposs)
            neg_rel_feats = model(Dnegs)
            
            loss_dict = sup_contrast_loss(ori_rel_feats, pos_rel_feats, neg_rel_feats)
            
            losses = sum(loss for loss in loss_dict.values())
            
            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            average_loss += losses.item()
            count += 1
            # losses_reduced : tensor(2.3199)

            meters.update(loss=losses_reduced, **loss_dict_reduced)
            optimizer.zero_grad()
            # Note: If mixed precision is not used, this ends up doing nothing
            # Otherwise apply loss scaling for mixed-precision recipe
            # try:
            with amp.scale_loss(losses, optimizer) as scaled_losses:
                scaled_losses.backward()

            if not SHOW_COMP_GRAPH and get_rank() == 0:
                try:
                    g = vis_graph.visual_computation_graph(
                        losses, model.named_parameters(), cfg.OUTPUT_DIR, "total_loss-graph"
                    )
                    g.render()
                    for name, ls in loss_dict_reduced.items():
                        g = vis_graph.visual_computation_graph(
                            losses, model.named_parameters(), cfg.OUTPUT_DIR, f"{name}-graph"
                        )
                        g.render()
                except:
                    logger.info("print computational graph failed")

                SHOW_COMP_GRAPH = True

            # add clip_grad_norm from MOTIFS, tracking gradient, used for debug
            verbose = (
                iteration % cfg.SOLVER.PRINT_GRAD_FREQ
            ) == 0 or print_first_grad  # print grad or not
            print_first_grad = False
            clip_grad_norm(
                [(n, p) for n, p in model.named_parameters() if p.requires_grad],
                max_norm=cfg.SOLVER.GRAD_NORM_CLIP,
                logger=logger,
                verbose=verbose,
                clip=True,
            )

            optimizer.step()

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            elapsed_time = str(datetime.timedelta(seconds=int(end - start_training_time)))

        
            if iteration % 30 == 0:
                logger.log(TFBoardHandler_LEVEL, (meters.meters, iteration))

                logger.log(
                    TFBoardHandler_LEVEL,
                    ({"curr_lr": float(optimizer.param_groups[0]["lr"])}, iteration),
                )
                # save_buffer(output_dir)

            if iteration % 10 == 0 or iteration == max_iter:
                logger.info(
                    meters.delimiter.join(
                        [
                            "\ninstance name: {instance_name}\n" "elapsed time: {elapsed_time}\n",
                            "eta: {eta}\n",
                            "iter: {iter}/{max_iter}\n",
                            "avg_loss: {avg_loss:.6f}",
                            "{meters}",
                            "lr: {lr:.6f}\n",
                            "max mem: {memory:.0f}\n",
                        ]
                    ).format(
                        instance_name=cfg.OUTPUT_DIR[len("checkpoints/") :],
                        eta=eta_string,
                        elapsed_time=elapsed_time,
                        iter=iteration,
                        avg_loss = average_loss / count,
                        meters=str(meters),
                        lr=optimizer.param_groups[0]["lr"],
                        max_iter=max_iter,
                        memory=torch.cuda.memory_allocated() / 1024.0 / 1024.0,
                    )
                )
                average_loss = 0
                count = 0
                #if pre_clser_pretrain_on:
                #    logger.info("relness module pretraining..")

            if iteration % checkpoint_period == 0:
                checkpointer.save("model_{:07d}".format(iteration), **arguments)
            if iteration == max_iter:
                checkpointer.save("model_final", **arguments)

            val_result_value = None  # used for scheduler updating
            #if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0:
            #    logger.info("Start validating")
            #    val_result = run_val(cfg, model, val_data_loaders, distributed, logger)
            #    val_result_value = val_result[1]
            #    if get_rank() == 0:
            #        for each_ds_eval in val_result[0]:w
            #            for each_evalator_res in each_ds_eval[1]:
            #                logger.log(TFBoardHandler_LEVEL, (each_evalator_res, iteration))
            # scheduler should be called after optimizer.step() in pytorch>=1.1.0
            # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
            if cfg.SOLVER.SCHEDULE.TYPE == "WarmupReduceLROnPlateau":
                scheduler.step(val_result_value, epoch=iteration)
                if scheduler.stage_count >= cfg.SOLVER.SCHEDULE.MAX_DECAY_STEP:
                    logger.info("Trigger MAX_DECAY_STEP at iteration {}.".format(iteration))
                    break
            else:
                scheduler.step()

        total_training_time = time.time() - start_training_time
        total_time_str = str(datetime.timedelta(seconds=total_training_time))
        logger.info(
            "Total training time: {} ({:.4f} s / it)".format(
                total_time_str, total_training_time / (max_iter)
            )
        )
    return model




def fix_eval_modules(eval_modules):
    for module in eval_modules:
        if module is None:
            continue

        for _, param in module.named_parameters():
            param.requires_grad = False
        # DO NOT use module.eval(), otherwise the module will be in the test mode, i.e., all self.training condition is set to False


def set_train_modules(modules):
    for module in modules:
        for _, param in module.named_parameters():
            param.requires_grad = True


def run_val(cfg, model, val_data_loaders, distributed, logger):
    if distributed:
        model = model.module
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations",)
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes",)

    dataset_names = cfg.DATASETS.VAL
    val_result = []
    for dataset_name, val_data_loader in zip(dataset_names, val_data_loaders):
        dataset_result = inference(
            cfg,
            model,
            val_data_loader,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=None,
            logger=logger,
        )
        synchronize()
        val_result.append(dataset_result)

    val_values = []
    for each in val_result:
        if isinstance(each, tuple):
            val_values.append(each[0])
    # support for multi gpu distributed testing
    # send evaluation results to each process
    gathered_result = all_gather(torch.tensor(val_values).cpu())
    gathered_result = [t.view(-1) for t in gathered_result]
    gathered_result = torch.cat(gathered_result, dim=-1).view(-1)
    valid_result = gathered_result[gathered_result >= 0]
    val_result_val = float(valid_result.mean())

    del gathered_result, valid_result
    return val_result, val_result_val


def run_test(cfg, model, distributed, logger):
    if distributed:
        model = model.module
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations",)
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, mode="test", is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(
        output_folders, dataset_names, data_loaders_val
    ):
        inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
            logger=logger,
        )
        synchronize()


def main():
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument(
        "--config-file",
        default="configs/e2e_relBGNN_vg_contrastive.yaml",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # mode
    mode = "contrastive learning"

    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H-%M")

    cfg.OUTPUT_DIR = os.path.join(
        cfg.OUTPUT_DIR,
        f"{mode}",
        f"({time_str}){cfg.EXPERIMENT_NAME}"
        + f"{'(debug)' if cfg.DEBUG else ''}",
    )

    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("pysgg", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)
    # if cfg.DEBUG:
    #     logger.info("Collecting env info (might take some time)")
    #     logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    
    ## 중복된 config print
    #with open(args.config_file, "r") as cf:
    #    config_str = "\n" + cf.read()
    #    logger.info(config_str)
    
    #logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os.path.join(cfg.OUTPUT_DIR, "config.yml")
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    model = train(cfg, args.local_rank, args.distributed, logger)

    #if not args.skip_test:
    #    run_test(cfg, model, args.distributed, logger)


if __name__ == "__main__":
    os.environ["OMP_NUM_THREADS"] = "8"

    main()
