# Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved. This source code is licensed under the BSD-style license found in the LICENSE file in the root directory of this source tree.
import os
import math
import logging
import random
import argparse
from pprint import pformat

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR

from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint
from ignite.metrics import Loss, MetricsLambda, RunningAverage
from ignite.contrib.handlers import ProgressBar, PiecewiseLinear, LRScheduler
from ignite.contrib.handlers.tensorboard_logger import TensorboardLogger, OutputHandler, OptimizerParamsHandler
from transformers import GPT2LMHeadModel, BertTokenizer, AdamW, WEIGHTS_NAME, CONFIG_NAME, GPT2Tokenizer, GPT2Config
from torch.utils.data import DataLoader
# from data_utils import load_data, NEW_ADD_TOKENS
from data_utils import load_data, NEW_ADD_TOKENS
from dataset_gpt2 import GPT2Dataset, IGNORE_INDEX
import pickle
from prefix_tuning import PrefixTuningTemplate

import torch.nn as nn
from model import PromptGPT2

logger = logging.getLogger(__file__)
torch.set_num_threads(8)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def read_binary_file(file_path):
    with open(file_path,'rb') as f:
        data = pickle.load(f)
    return data

def get_action_topic_vocab(all_actions, all_topics, tokenizer):
    vocab = np.zeros(len(tokenizer))
    for action in all_actions:
        tokens = tokenizer.tokenize(action)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)  
        vocab[token_ids] = 1.0
    
    for topic in all_topics:
        tokens = tokenizer.tokenize(topic)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)  
        vocab[token_ids] = 1.0
    
    for token in tokenizer.all_special_tokens:
        id = tokenizer.convert_tokens_to_ids(token)
        vocab[id] = 1
    
    return vocab


def str2bool(v):
    if v.lower() in ('true', 'yes', 't', 'y', '1'):
        return True
    elif v.lower() in ('false',' no', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")

def average_distributed_scalar(scalar, args):
    """ Average a scalar over the nodes if we are in distributed training. We use this for distributed evaluation. """
    if args.local_rank == -1:
        return scalar
    scalar_t = torch.tensor(scalar, dtype=torch.float, device=args.device) / torch.distributed.get_world_size()
    torch.distributed.all_reduce(scalar_t, op=torch.distributed.ReduceOp.SUM)
    return scalar_t.item()



def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_tcp', type=str2bool, default="False", help="Whether or not use TCP-enhanced generation")

    parser.add_argument("--train_path", type=str, default=None,
                        help="Path of the train dataset for dist dataset. ")
    parser.add_argument("--valid_path", type=str, default=None,
                        help="Path of the valid dataset for dist dataset. ")
    parser.add_argument("--cache_dir", type=str, default="dataset_cache",
                        help="Path or url of the dataset cache dir.")

    parser.add_argument('--log_dir', type=str, default="", help="Output logs to a dir")
    parser.add_argument("--model_checkpoint", type=str, default="gpt2", help="Path or URL of the model")
    parser.add_argument("--from_step", type=int, default=-1, help="Init learning rate from this step")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of subprocesses for data loading")
    parser.add_argument("--n_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=2, help="Batch size for validation")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--scheduler", type=str, default="linear", choices=['noam', 'linear'], help="method of optim")
    parser.add_argument("--n_emd", type=int, default=768, help="Number of n_emd in config file (for noam)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--eval_before_start", action='store_true',
                        help="If true start with a first evaluation before training")

    parser.add_argument("--warmup_steps", type=int, default=5000, help="Warm up steps")
    parser.add_argument("--valid_steps", type=int, default=5000, help="Perfom validation every X steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2,
                        help="Accumulate gradients on several steps")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--fp16", type=str, default="",
                        help="Set to O0, O1, O2 or O3 for fp16 training (see apex documentation)")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (-1: not distributed)")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")               
    parser.add_argument("--contrain_vocab", type=int, default=1, help="Contrain the vocabulary") 

    parser.add_argument("--use_goal_topic", type=int, default=1, help="Number of tokens used for optimizing the prompt")
    parser.add_argument("--num_tokens", type=int, default=50, help="Number of tokens used for optimizing the prompt")
    parser.add_argument("--n_action_toks", type=int, default=5, help="number of soft tokens for the action prompt")
    parser.add_argument("--n_topic_toks", type=int, default=10, help="number of soft tokens for the topic prompt")
    parser.add_argument("--freeze_plm", type=int, default=1, help="freeze the parameters of the pretrained language model")

    args = parser.parse_args()
    setup_seed(args.random_seed)

    # logging is set to INFO (resp. WARN) for main (resp. auxiliary) process.
    # logger.info => log main process only, logger.warning => log all processes
    logging.basicConfig(level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.info("Arguments: %s", pformat(args))

    # Initialize distributed training if needed
    args.distributed = (args.local_rank != -1)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_checkpoint)
    plm = GPT2LMHeadModel.from_pretrained(args.model_checkpoint)
    config = GPT2Config.from_pretrained(args.model_checkpoint)

    special_tokens_dict = {'additional_special_tokens': NEW_ADD_TOKENS}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info("We have added {} special tokens".format(num_added_toks))
    # # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
    plm.resize_token_embeddings(len(tokenizer))

    prefix_model = PrefixTuningTemplate(
        config=config,
        num_token = args.num_tokens,  
        n_action_toks=args.n_action_toks,
        n_topic_toks= args.n_topic_toks,
        use_goal_topic=args.use_goal_topic
    )
    model = PromptGPT2(plm = plm, prefix_model = prefix_model, freeze_plm= args.freeze_plm)
    model.to(args.device)

    # Prepare datasets
    logger.info("Prepare datasets")
    train_data = load_data(tokenizer, logger, args.train_path, args.cache_dir, 
        data_partition="train", use_tcp=args.use_tcp)
    valid_data = load_data(tokenizer, logger, args.valid_path, args.cache_dir, 
        data_partition="valid", use_tcp=args.use_tcp)
    
    train_dataset = GPT2Dataset(train_data, tokenizer, max_seq_len=args.max_seq_len)
    valid_dataset = GPT2Dataset(valid_data, tokenizer, max_seq_len=args.max_seq_len)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
    
    logger.info("Build train and val dataloaders...")
    train_loader = DataLoader(train_dataset,
                            sampler=train_sampler,
                            collate_fn=train_dataset.collate,
                            num_workers=args.num_workers,
                            batch_size=args.train_batch_size,
                            shuffle=(not args.distributed))
    val_loader = DataLoader(valid_dataset,
                            sampler=valid_sampler,
                            collate_fn=valid_dataset.collate,
                            num_workers=args.num_workers,
                            batch_size=args.valid_batch_size,
                            shuffle=False)

    # Prepare model for FP16 and distributed training if needed (order is important, distributed should be the last)
    optimizer = AdamW([{'params': prefix_model.parameters(), 'initial_lr': args.lr}], lr=args.lr, correct_bias=True)
    if args.fp16:
        from apex import amp  # Apex is only required if we use fp16 training
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16)
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Training function and trainer
    def update(engine, batch):
        batch = {key: value.to(args.device) for key, value in batch.items()}
        # input_ids, lm_labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
        model.train()
        lm_output = model(batch)
        lm_loss = lm_output["loss"]
        loss = lm_loss / args.gradient_accumulation_steps
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_norm)
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        if engine.state.iteration % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        return loss.item(), optimizer.param_groups[0]['lr']

    trainer = Engine(update)

    # Evaluation function and evaluator (evaluator output is the input of the metrics)
    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch = {key: value.to(args.device) for key, value in batch.items()}
            lm_output = model(batch)
            lm_logits = lm_output["logits"]
            lm_labels = batch["labels"]
            lm_logits_flat_shifted = lm_logits[..., :-1, :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)
            return lm_logits_flat_shifted, lm_labels_flat_shifted

    evaluator = Engine(inference)

    # Attach evaluation to trainer: we evaluate when we start the training and at the end of each epoch
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda _: evaluator.run(val_loader))
    if args.n_epochs < 1:
        trainer.add_event_handler(Events.COMPLETED, lambda _: evaluator.run(val_loader))
    if args.eval_before_start:
        trainer.add_event_handler(Events.STARTED, lambda _: evaluator.run(val_loader))

    # Evaluation during training
    @trainer.on(Events.ITERATION_STARTED)
    def log_iterations(engine):
        # if engine.state.iteration % max(int(0.1 * len(train_loader)), 1) == 0:
        if engine.state.iteration % args.valid_steps == 0:
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics

    # Make sure distributed data samplers split the dataset nicely between the distributed processes
    if args.distributed:
        trainer.add_event_handler(Events.EPOCH_STARTED, lambda engine: train_sampler.set_epoch(engine.state.epoch))
        evaluator.add_event_handler(Events.EPOCH_STARTED, lambda engine: valid_sampler.set_epoch(engine.state.epoch))

    # noam decrease the learning rate
    # model_size = model.config.n_embd
    model_size = args.n_emd
    noam_lambda = lambda step: (
            model_size ** (-0.5) * min((step + 1) ** (-0.5), (step + 1) * args.warmup_steps ** (-1.5)))
    noam_scheduler = LambdaLR(optimizer, lr_lambda=noam_lambda, last_epoch=args.from_step)
    scheduler = LRScheduler(noam_scheduler)
    if args.scheduler == "linear":
        scheduler = PiecewiseLinear(optimizer, "lr", [(0, args.lr), (args.n_epochs * len(train_loader), 0.0)])
    trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    # Prepare metrics - note how we compute distributed metrics
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, "loss")
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, "lr")
    metrics = {"nll": Loss(torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX), output_transform=lambda x: (x[0], x[1]))}
    metrics.update({"average_nll": MetricsLambda(average_distributed_scalar, metrics["nll"], args)})
    metrics["average_ppl"] = MetricsLambda(math.exp, metrics["average_nll"])
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    # On the main process: add progress bar, tensorboard, checkpoints
    # And save model, configuration and tokenizer before we start to train
    if args.local_rank in [-1, 0]:
        pbar = ProgressBar(persist=True, mininterval=2)
        pbar.attach(trainer, metric_names=["loss", "lr"])
        evaluator.add_event_handler(Events.COMPLETED,
                                    lambda _: pbar.log_message("Validation: %s" % pformat(evaluator.state.metrics)))
        # set log dir
        log_dir = args.log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        tb_logger = TensorboardLogger(log_dir=log_dir)
        tb_logger.attach(trainer, log_handler=OutputHandler(tag="training", metric_names=["loss"]),
                         event_name=Events.ITERATION_COMPLETED)
        tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer), event_name=Events.ITERATION_STARTED)
        tb_logger.attach(evaluator, log_handler=OutputHandler(tag="validation", metric_names=list(metrics.keys()),
                        another_engine=trainer),
                        event_name=Events.EPOCH_COMPLETED)

        checkpoint_handler = ModelCheckpoint(tb_logger.writer.logdir, 'checkpoint', save_interval=1, n_saved=3)
        # save model after evaluation
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'mymodel': getattr(model, 'module', model)})
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpoint_handler, {
            'mymodel': getattr(model, 'module', model)})  # "getattr" take care of distributed encapsulation

        torch.save(args, tb_logger.writer.logdir + '/model_training_args.bin')
        # getattr(model, 'module', model).config.to_json_file(os.path.join(tb_logger.writer.logdir, CONFIG_NAME))
        tokenizer.save_vocabulary(tb_logger.writer.logdir)

    # Run the training
    trainer.run(train_loader, max_epochs=args.n_epochs)

    # On the main process: close tensorboard logger and rename the last checkpoint
    # (for easy re-loading with GPT2LMHeadModel.from_pretrained method)
    if args.local_rank in [-1, 0] and args.n_epochs > 0:
        os.rename(checkpoint_handler._saved[-1][1][-1],
                  os.path.join(tb_logger.writer.logdir,
                               WEIGHTS_NAME))  # TODO: PR in ignite to have better access to saved file paths (cleaner)
        tb_logger.close()


if __name__ == "__main__":
    train()
