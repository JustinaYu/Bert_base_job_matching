import sys
import os

import torch
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertForPreTraining, BertConfig, AdamW, get_scheduler

from train.trainer import Trainer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from argparse import ArgumentParser
from io_new.bert_processor import BertProcessor
from config.config import config
from torch.utils.data import DataLoader, random_split
from common.tools import logger, init_logger
from common.tools import seed_everything


def run_train(args, writer, accelerator):
    # get train dataset
    processor = BertProcessor(vocab_path=config['bert_vocab_path'], args=args)
    documents = processor.get_documents(config['data_dir'] / args.input_file) if args.do_data else [[]]
    print("examples")
    examples = processor.create_examples(documents, config['data_dir'] / args.cached_examples_file)
    print("dataset")
    # 生成tensordataset
    total_dataset = processor.create_dataset(examples)
    # 固定数量的测试集，例如 1000 个样本作为测试集
    eval_size = args.eval_size
    train_size = len(total_dataset) - eval_size
    # 使用 random_split 划分数据集
    train_dataset, eval_dataset = random_split(total_dataset, [train_size, eval_size])
    print("dataloader")
    # 生成dataset
    # train_dataset = processor.create_train_dataset(examples)
    # shuffle
    # train_sampler = RandomSampler(train_dataset)
    """
    num_workers 控制数据加载时使用的子进程数量，可以设置为 CPU 核心数的一半或等于核心数（物理）
    """
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.train_batch_size, num_workers=args.num_workers)
    eval_dataloader = DataLoader(eval_dataset, shuffle=True, batch_size=args.eval_batch_size, num_workers=args.num_workers)
    print("finished")
    # model
    logger.info("initializing model")
    bert_config = BertConfig.from_json_file(config['bert_config_file'])
    model = BertForPreTraining(config=bert_config)

    device = accelerator.device
    model.to(device) 

    
    # the total update times
    update_total = int(len(train_dataloader) / args.gradient_accumulation_steps * args.epochs)
    warmup_steps = int(update_total * args.warmup_proportion)
    params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=warmup_steps, num_training_steps=update_total)

    # load_state 断点续训
    if args.from_trained:  # 加载之前保存的训练状态
        logger.info(f"Loading trained model : args.ckpt_dir")
        ckpt = torch.load(args.ckpt_dir)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        lr_scheduler.load_state_dict(ckpt['lr_state'])



    train_dataloader, eval_dataloader, model, optimizer,lr_scheduler = accelerator.prepare(train_dataloader, eval_dataloader, model, optimizer,lr_scheduler)
    # mixed precision fp16现在可以通过设置training_args，accelerator或者pytorch内置的amp实现，不需要apex
    # callback
    """
    断点续训可以通过Trainer实现
    """
    logger.info("initializing callbacks")

    # train
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(examples))
    logger.info("  Num Epochs = %d", args.epochs)
    # 分布式训练也可以通过高级API Trainer来实现
    # logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
    #             args.train_batch_size * args.gradient_accumulation_steps * (
    #                 torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", update_total)
    """
    Trainer API
    存在data不匹配的问题，应该需要自定义data collator
    """
    # train_args = TrainingArguments(output_dir=args.output_path,
    #                                num_train_epochs=3,  # total # of training epochs 训练总批次
    #                                per_device_train_batch_size=8,
    #                                max_grad_norm=args.grad_clip, # 梯度裁剪
    #                                save_steps=args.save_step,
    #                                save_total_limit=args.save_total_limit,
    #                                fp16=args.fp16, # 混合精度
    #                                dataloader_drop_last=False, #是否抛弃dataloader之后的,
    #                                dataloader_num_workers=args.num_workers,
    #                                )
    # trainer = Trainer(model=model,
    #                   args=train_args,
    #                   train_dataset=train_dataset,
    #                   optimizers=(optimizer, lr_scheduler))
    # trainer.train()
    """
    自定义train的过程
    """
    trainer = Trainer(model=model,
                      device=device,
                      checkpoint_name=args.checkpoint_name,
                      save_step=args.save_step,
                      writer=writer,
                      accelerator=accelerator,
                      epochs=args.epochs,
                      logger=logger,
                      optimizer=optimizer,
                      lr_scheduler=lr_scheduler,
                      grad_clip=args.grad_clip,
                      eval_step=args.eval_step,
                      gradient_accumulation_steps=args.gradient_accumulation_steps)
    trainer.train(train_data=train_dataloader, eval_data=eval_dataloader, seed=args.seed)


def main():
    parser = ArgumentParser()
    parser.add_argument("--input_file", default="total_resume_data.tokens", type=str, help="the input files to train")
    parser.add_argument("--cached_examples_file", default="cached_examples_file_bert.pt", type=str, help="the cached file to store examples to train")
    parser.add_argument("--do_data", action='store_true', help="whether to process document file(documents)")
    parser.add_argument("--do_train", action='store_true', help="whether to train")
    parser.add_argument("--epochs", default=6, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help='Number of steps to accumulate before update')
    # data
    parser.add_argument('--num_workers', type=int, default=5, help='Number of threads to load dataset')
    parser.add_argument("--train_batch_size", default=2, type=int)
    parser.add_argument("--eval_batch_size", default=2, type=int)
    parser.add_argument("--train_max_seq_len", default=2048, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=int, help="for optimizer")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="avoid overfit, for optimizer")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="for optimizer")
    parser.add_argument("--grad_clip", default=1.0, type=float, help="gradient clip, for optimizer")
    parser.add_argument("--learning_rate", default=2e-5, type=float, help="basic learning rate, for optimizer")
    parser.add_argument('--seed', type=int, default=42, help="global random seed")
    parser.add_argument('--dupe_factor', type=int, default=2,
                        help="Number of times to duplicate the input data (with different masks).")
    parser.add_argument('--masked_lm_prob', type=float, default=0.15, help="Masked LM probability.")
    parser.add_argument('--short_seq_prob', type=float, default=0.1, help="Probability of creating sequences which are shorter than the maximum length.")
    parser.add_argument('--max_predictions_per_seq', type=int, default=200, help="Maximum number of masked LM predictions per sequence.")
    # store path
    parser.add_argument("--checkpoint_name", default="bert_checkpoint.pth", type=str, help="the store path of outputs of training.")
    parser.add_argument("--save_step", default=100, type=int, help="Number of updates steps before two checkpoint saves if save_strategy=steps")
    parser.add_argument("--save_total_limit", default=10, type=int,
                        help="The value limit the total amount of checkpoints. ")
    # eval
    parser.add_argument('--eval_size', type=int, default=1000, help='eval data size')
    parser.add_argument('--eval_step', type=int, default=5, help='Validation interval steps')

    # 断点续训
    parser.add_argument('--from_trained', action='store_true', help='whether to train from pretrained ckpt')
    parser.add_argument('--ckpt_dir', type=str, default='model/checkpoints/xx.ckpt')

    args = parser.parse_args()
    seed_everything(args.seed)
    # init_logger(log_file=config['log_dir'] / "train.log")
    init_logger()
    # 启用混合精度
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    writer = SummaryWriter(config["writer_dir"])

    # arg.do_data用于判断是不是要从.tokens文件生成examples
    if args.do_train:  # conduct dataset generation
        run_train(args, writer, accelerator)



if __name__ == '__main__':
    main()