import torch
import sys
import os

from torch.nn.utils import clip_grad_norm_

from config.config import config

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from callback.progressbar import ProgressBar
from common.tools import seed_everything


class Trainer(object):
    def __init__(self, model,
                 output_path,
                 save_step,
                 writer,
                 accelerator,
                 epochs,
                 logger,
                 optimizer,
                 lr_scheduler,
                 gradient_accumulation_steps,
                 grad_clip=1.0,
                 ):
        self.model = model
        self.epochs = epochs
        self.logger = logger
        self.grad_clip = grad_clip
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.accelerator = accelerator
        self.writer = writer
        self.output_path = output_path
        self.save_step = save_step

    def save_info(self, epoch, best):
        if hasattr(self.model, 'module'):
            print("has attr: module")
        model_save = self.model.module if hasattr(self.model, 'module') else self.model
        state = {"model": model_save,
                 'epoch': epoch,
                 'best': best}
        return state

    def train_epoch(self, train_data, epoch, update_step):
        pbar = ProgressBar(n_total=len(train_data), disable=not self.accelerator.is_main_process)
        for step, batch in enumerate(train_data):
            with self.accelerator.accumulate(self.model):
                input_ids, attention_mask, token_type_ids,labels,next_sentence_labels = batch
                # print(type(input_ids))
                outputs = self.model(input_ids=input_ids,attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, next_sentence_label=next_sentence_labels)
                loss = outputs.loss
                # if step % 100 == 0:
                # self.logger.info(f"EPOCH {epoch+1}  -- STEP {step} : {loss:.4f} -- Loss value")
                self.writer.add_scalar('Loss/train', loss.item(), global_step=epoch * len(train_data) + step)
                self.accelerator.backward(loss)
                # 梯度裁剪
                clip_grad_norm_(self.model.parameters(), self.grad_clip)
                # 梯度更新
                self.optimizer.step()
                # 更新学习率
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                update_step += 1
                # save
                if update_step % 100== 0:
                    torch.save(self.model, config["checkpoint_dir"] / f"{str(update_step*100)}-{self.output_path}")
                pbar.batch_step(step=epoch, info={}, bar_type='train a batch')
        return update_step

    def train(self, train_data, seed):
        seed_everything(seed)
        # train cycle
        update_step = 0
        for epoch in range(self.epochs):
            self.logger.info(f"Epoch {epoch}/{self.epochs}")
            update_step = self.train_epoch(train_data, epoch, update_step)
