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
                 device,
                 checkpoint_name,
                 save_step,
                 writer,
                 accelerator,
                 epochs,
                 logger,
                 optimizer,
                 lr_scheduler,
                 eval_step,
                 gradient_accumulation_steps,
                 grad_clip=1.0,
                 ):
        self.model = model
        self.epochs = epochs
        self.logger = logger
        self.grad_clip = grad_clip
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator
        self.writer = writer
        self.checkpoint_name = checkpoint_name
        self.save_step = save_step
        self.eval_step = eval_step
        self.gradient_accumulation_steps=gradient_accumulation_steps
        self.device=device

    def save_info(self, epoch, best):
        if hasattr(self.model, 'module'):
            print("has attr: module")
        model_save = self.model.module if hasattr(self.model, 'module') else self.model
        state = {"model": model_save,
                 'epoch': epoch,
                 'best': best}
        return state

    def train_epoch(self, train_data, eval_data, epoch, update_step):
        # pbar = ProgressBar(n_total=len(train_data), disable=not self.accelerator.is_main_process)
        for step, batch in enumerate(train_data):
            self.model.train()
            with self.accelerator.accumulate(self.model):
                input_ids, attention_mask, token_type_ids,labels,next_sentence_labels = batch
                # print(type(input_ids))
                outputs = self.model(input_ids=input_ids,attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, next_sentence_label=next_sentence_labels)
                loss = outputs.loss
                # if step % 100 == 0:
                self.writer.add_scalar('Loss/train', loss.item(), global_step=epoch * len(train_data) + step)
                self.accelerator.backward(loss)
                # 梯度裁剪
                clip_grad_norm_(self.model.parameters(), self.grad_clip)
                # 梯度更新
                self.optimizer.step()
                # 更新学习率
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                update_step = (epoch * len(train_data) + step + 1) / self.gradient_accumulation_steps
                # save
                self.accelerator.wait_for_everyone()
                if update_step % (self.eval_step) == 0:
                        # 每eval_step做验证
                        self.validate(eval_data, update_step)
                if self.accelerator.is_main_process:
                    self.logger.info(f"EPOCH {epoch+1}  -- STEP {step} : {loss:.4f} -- Loss value")
                    if update_step % self.save_step == 0:
                        # 每save_step保存状态
                        unwrap_model = self.accelerator.unwrap_model(self.model)
                        unwrap_optim = self.accelerator.unwrap_model(self.optimizer)
                        unwrap_lr = self.accelerator.unwrap_model(self.lr_scheduler)
                        torch.save({
                            'model_state': unwrap_model.state_dict(),
                            'optim_state': unwrap_optim.state_dict(),
                            'lr_state': unwrap_lr.state_dict()},
                            config["checkpoint_dir"] / f"{str(update_step)}-ckpt.pt")
                        # logger.info(f'checkpoint ckpt_{epoch + 1}.pt is saved...')
                # pbar.batch_step(step=epoch, info={}, bar_type='train a batch')
        return update_step

    def validate(self, eval_data, update_step):
        # self.logger.info(f"Start validation{str(update_step/self.eval_step)}.")
        # print("1")
        self.model.eval()
        # print("2")
        losses = []
        # print("3")
        # forbid gradient computation
        with torch.no_grad():
            # print("4")
            # eval_pbar = ProgressBar(n_total=len(eval_data), disable=not self.accelerator.is_main_process)
            for step, batch in enumerate(eval_data):
                input_ids, attention_mask, token_type_ids,labels,next_sentence_labels = batch
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                                     labels=labels, next_sentence_label=next_sentence_labels)
                loss = outputs.loss
                print(f"LOSS:{str(loss.item())}-STEP:{step}")
                # eval_pbar.batch_step(step=step, info={}, bar_type='eval a batch')
                all_losses = self.accelerator.gather(loss)
                if self.accelerator.is_main_process:
                    print("yes!")
                    # all_losses = [torch.zeros(1, dtype=loss.dtype, device=self.device) for _ in range(2)]
                    # torch.distributed.all_gather(all_losses, loss)
                    # print(str(len(losses)))
                    losses.extend(all_losses.cpu().numpy())
                    # losses.append(loss.cpu().numpy())
                    # losses.extend([l.cpu().numpy() for l in all_losses])
                    print(str(len(losses)))
            # eval_pbar.close()
            print("finished")
            if self.accelerator.is_main_process:
                print("1")
                loss_mean = sum(losses) / len(losses)
                print("2")
                self.writer.add_scalar('Loss/eval', loss_mean, global_step=update_step/self.eval_step)
                print("3")
                self.logger.info(f"Finished validation{str(update_step / self.eval_step)}. LOSS - {str(loss_mean)}")
                print("4")
        # transfer to train mode
        self.model.train()



    def train(self, train_data, eval_data, seed):
        seed_everything(seed)
        # train cycle
        update_step = 0
        for epoch in range(self.epochs):
            self.logger.info(f"Epoch {epoch}/{self.epochs}")
            update_step = self.train_epoch(train_data, eval_data, epoch, update_step)
