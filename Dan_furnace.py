# wheels
from dynaconf import Dynaconf
from abc import ABC, abstractmethod
import torch
from torch import optim
import torch.amp
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
import time
# our libs
import datasets
# import datasets.gen_rand
import datasets.CCPD
import datasets.gen_rand
import models_cit
import models_cit.Loss
import models_cit.ALPR

class BaseTainer_a_config:
    def __init__(self,conf_file:str):
        '''conf_file in yaml format'''
        args = self.args = Dynaconf(settings_files=[conf_file])
        self.train_loader=self.build_dataloader(args.dataset)
        self.model=self.build_model(args.model)
        self.criterion=self.build_lossFun(args.criterion) #loss
        self.optimizer=self.build_optim(self.model,self.args.optim)
        self.scheduler=self.build_lr_scheduler(self.optimizer,self.args.lr_scheduler)
        self.logger=self.build_logger(args.LOGGER)
        return
    
    @staticmethod
    @abstractmethod
    def build_dataloader(args):
        return
    @staticmethod
    @abstractmethod
    def build_model(args):
        return
    @staticmethod
    @abstractmethod
    def build_lossFun(args):
        return
    @staticmethod
    @abstractmethod
    def build_optim(model,args):
        return
    @staticmethod
    @abstractmethod
    def build_lr_scheduler(optimizer,args):
        return
    @staticmethod
    @abstractmethod
    def build_logger(args):
        return
    def train(self):
        '''train the hole config file'''
        self.train_an_epoch()
        return
    @abstractmethod
    def train_an_epoch(self):
        self.train_a_step()
        return
    @abstractmethod
    def train_a_step(self):
        return
    pass
# import torch.amp.GradScaler
from torch.amp import autocast
# GradScaler()
# torch.amp.GradScaler('cuda', args...)
from torch.cuda.amp import GradScaler
class Trainer_a_conf(BaseTainer_a_config):
    def __init__(self, conf_file):
        super().__init__(conf_file)
        # int AMP grad scaler
        self.scaler=self.build_scaler()
        self.amp_cfg = self.build_amp_cfg(self.args)
        if "val_set" in self.args:
            self.eval_loader = self.build_dataloader(self.args.val_set)
            self.evaluator:models_cit.Loss.IoU_LPs_evaluator = getattr(models_cit.Loss, self.args.val_set.evaluator.name)()
        else:
            self.eval_loader =  None
        return

    def build_scaler(self):
        scaler = GradScaler(self.args.device,enabled=self.args.GradScaler_enable)
        # scaler = GradScaler(enabled=self.args.GradScaler_enable)
        return scaler

    @staticmethod
    def build_dataloader(args):
        dataset_name=args.name
        if dataset_name==None:
            dataset=datasets.gen_rand.Dataset_rand(20)
        else:
            # datasets.CCPD.CCPD_base()
            dataset=getattr(datasets.CCPD,dataset_name)(args.csvPath)
        dataloader=datasets.gen_rand.DataLoader(dataset,args.batch_size,shuffle=True,num_workers=args.n_worker)
        return dataloader
    @staticmethod
    def build_model(args):
        model_name=args.name
        model_type=getattr(models_cit.ALPR,model_name)
        model=model_type()
        return model
    @staticmethod
    def build_lossFun(args):
        lossName=args.name
        lossType=getattr(models_cit.Loss,lossName)
        criterion= lossType(args)
        # models.Loss.Hungarian_loss()
        return  criterion
    @staticmethod
    def build_optim(model:nn.Module, args):

        optim_type=getattr(optim,args.name)
        optimizer=optim_type(model.parameters(),lr=args.lr,betas=args.betas,weight_decay=args.weight_decay)
        # optimizer = optim.Adam(
        #     [
        #         {
        #             "params": model.bone.parameters(),
        #             "lr": args.backbone_lr,
        #             'betas': args.adam_betas,
        #             'weight_decay': args.weight_decay,
        #         },
        #         {
        #             "params": model.neck.parameters(),
        #             "lr": args.learning_rate,
        #             'betas': args.adam_betas,
        #             'weight_decay': args.weight_decay,
        #         },
        #         {
        #             "params": model.head.parameters(),
        #             "lr": args.learning_rate,
        #             'betas': args.adam_betas,
        #             'weight_decay': args.weight_decay,
        #         },
        #     ]
        # )
        return optimizer
    @staticmethod
    def build_lr_scheduler(optimizer, args):
        # optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_schedule_step_size, gamma=args.gamma)
        # optim.lr_scheduler.ConstantLR()
        if args.name == 'LambdaLR':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
        elif args.name=='StepLR' :
            scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=args.step_size,gamma=args.gamma)
        else:
            scheduler = getattr(torch.optim.lr_scheduler, args.name)(optimizer)
        # lr_scheduler_type=getattr(optim.lr_scheduler,args.name)
        return scheduler
    @staticmethod
    def build_logger(args):
        # return torch.utils.tensorboard.SummaryWriter(args.log_dir)
        return SummaryWriter(args.LOG_DIR)
    def train(self):
        cfg=self.args
        self.global_step,start_epoch, best_acc = self.resume_ckpt(cfg)
        # eval_results=self.eval_dateset(cfg)
        # print(eval_results)
        # exit()
        # 训练循环
        for epoch in range(start_epoch, cfg.num_epochs):
            self.train_a_epoch(cfg, best_acc, epoch)
        return 

    def train_a_epoch(self, cfg, best_acc, epoch):
        self.model.train()
        train_loss = 0.0
        progressBar={'now':0,'end':len(self.train_loader)} 
        batch_start = time.time()
        for imgs, targets in self.train_loader:
            progressBar['now']+=1
            self.global_step+=1

            imgs, targets = self._move_batch_2_device(cfg, imgs, targets)   

            self.optimizer.zero_grad()
            
            # AMP前向传播
            with autocast(cfg.device,**self.amp_cfg):
                outputs = self.model(imgs)
                loss = self.criterion(outputs, targets)

                # AMP反向传播
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_loss += loss.item()
            if (self.global_step %100)==0:
                with autocast(cfg.device, **self.amp_cfg), torch.no_grad():
                    loss_dict = self.criterion(outputs, targets, details=True)
                for key, value in loss_dict.items():
                    self.logger.add_scalar(f'step/{key}', value, self.global_step)
                pass
            batch_end=time.time()
            batch_time =  batch_end- batch_start
            batch_start=batch_end
            print(f'{progressBar["now"]}/{progressBar["end"]} | Batch time: {batch_time:.2f}s', end='\r')

        # 学习率调整
        self.scheduler.step()

        # eval model
        eval_results=self.eval_dateset(cfg)
        # 记录日志
        self.write_log(epoch, train_loss, eval_results)

        # 保存检查点
        val_acc=eval_results['lp_acc']
        best_acc=self.save_checkpoint(cfg, best_acc, epoch, val_acc)

        print(f'Epoch {epoch}/{cfg.num_epochs} | '
                f'Train Loss: {train_loss/len(self.train_loader):.4f} | '
                # f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%'
                )
        return

    def write_log(self, epoch, train_loss, eval_results):
        # eval_results:{'lp_acc','mean_iou','mean_iou1'}
        self.logger.add_scalar('epoch/Loss', train_loss/len(self.train_loader), epoch)
        self.logger.add_scalar('epoch/LP_acc', eval_results['lp_acc'], epoch)
        self.logger.add_scalar('epoch/mean_iou', eval_results['mean_iou'], epoch)
        self.logger.add_scalar('epoch/mean_iou1', eval_results['mean_iou1'], epoch)
        self.logger.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)

    def eval_dateset(self, cfg):
        if self.eval_loader==None:
            return None
        self.model.eval()
        progessBarLen=len(self.eval_loader)
        with torch.no_grad():
            for i,(imgs, targets) in enumerate(self.eval_loader):
                print(f'eval {i}/{progessBarLen}',end='\r')
                imgs, targets = self._move_batch_2_device(cfg, imgs, targets) 
                with autocast(cfg.device,**self.amp_cfg):
                    outputs = self.model(imgs,denoise=False)
                self.evaluator.forward_batch(outputs,targets)
                pass
        eval_result=self.evaluator.statistic_Dataset()
        # print(eval_result)
        return eval_result

    def build_amp_cfg(self, cfg):
        if cfg.amp_dtype=='bf16':
            amp_cfg={'enabled':True,'dtype':torch.bfloat16}
        elif cfg.amp_dtype=='fp16':
            amp_cfg={'enabled':True,'dtype':torch.float16}
        else:
            amp_cfg={'enabled':False}
        return amp_cfg

    def _move_batch_2_device(self, cfg, imgs, targets):
        if isinstance(imgs, dict):
            imgs = {k: v.to(cfg.device) if isinstance(v, torch.Tensor) else v for k, v in imgs.items()}
        elif isinstance(imgs, torch.Tensor):
            imgs = imgs.to(cfg.device)
        targets = {k: v.to(cfg.device) if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
        return imgs,targets

    def save_checkpoint(self, cfg, best_acc, epoch, val_acc):
        state = {
                'global_step':self.global_step,
                'epoch': epoch,
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'scaler': self.scaler.state_dict(),
                'best_acc': best_acc
            }
        os.makedirs(cfg.checkpoint_dir,exist_ok=True)
        # best
        if val_acc > best_acc:
            best_acc = val_acc
            state['best_acc']=best_acc
            torch.save(state, os.path.join(cfg.checkpoint_dir, 'best.pth'))
            pass
        # name_step
        torch.save(state, os.path.join(cfg.checkpoint_dir, f'{self.args.model.name}_{self.global_step}.pth'))
        # latest
        torch.save(state, os.path.join(cfg.checkpoint_dir, f'latest.pth'))

        return best_acc

    def resume_ckpt(self, cfg):
        global_step=0
        start_epoch = 0
        best_acc = 0.0
        self.model.to(cfg.device)
        if cfg.resume:
            ckpt_path = os.path.join(cfg.checkpoint_dir, 'latest.pth')
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, weights_only=False)
                self.model.load_state_dict(checkpoint['model'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                self.scheduler.load_state_dict(checkpoint['scheduler'])
                self.scaler.load_state_dict(checkpoint['scaler'])
                global_step=checkpoint['global_step']+1
                start_epoch = checkpoint['epoch'] + 1
                best_acc = checkpoint['best_acc']
                print(f"Resuming from epoch {start_epoch}")
            else:
                print(f'file not exist:{ckpt_path}')
                pass
        return global_step,start_epoch,best_acc

    pass

class Eval_a_conf(Trainer_a_conf):
    def __init__(self, conf_file):
        '''conf_file in yaml format'''
        args = self.args = Dynaconf(settings_files=[conf_file])
        # self.train_loader=self.build_dataloader(args.dataset)
        self.eval_loader = self.build_dataloader(self.args.val_set)
        self.model=self.build_model(args.model)
        # self.criterion=self.build_lossFun(args.criterion) #loss
        self.evaluator:models_cit.Loss.IoU_LPs_evaluator = getattr(models_cit.Loss, self.args.val_set.evaluator.name)()
        self.amp_cfg = self.build_amp_cfg(self.args)
        return
    def change_dataset(self,csvPath:str):
        self.args.val_set.csvPath=csvPath
        self.eval_loader = self.build_dataloader(self.args.val_set)
        return
    def test(self):
        cfg=self.args
        self.global_step,start_epoch, best_acc = self.resume_ckpt(cfg)
        eval_results=self.eval_dateset(cfg)
        print({k: v.item() for k, v in eval_results.items()})
        # exit()
        return eval_results
    def eval_batch(self, cfg,batch_input:dict):
        if len(batch_input[0]['imgs'].shape)==3:
            batch_input[0]['imgs']=batch_input[0]['imgs'].unsqueeze(0)
        imgs,targets=batch_input
        self.model.eval()
        with torch.no_grad():
            imgs, targets = self._move_batch_2_device(cfg, imgs, targets) 
            with autocast(cfg.device,**self.amp_cfg):
                outputs = self.model(imgs,denoise=False)
            self.evaluator.forward_batch(outputs,targets)
            pass
        # eval_result=self.evaluator.statistic_Dataset()
        # print(eval_result)
        return outputs #eval_result
    def resume_ckpt(self, cfg):
        global_step=0
        start_epoch = 0
        best_acc = 0.0
        self.model.to(cfg.device)
        if cfg.resume:

            ckpt_path = (
                os.path.join(cfg.checkpoint_dir, cfg.ckpt_file)
                if "ckpt_file" in cfg
                else os.path.join(cfg.checkpoint_dir, "best.pth")
            )
            # 'latest.pth'
            if os.path.exists(ckpt_path):
                checkpoint = torch.load(ckpt_path, weights_only=False)
                self.model.load_state_dict(checkpoint['model'])
                global_step=checkpoint['global_step']+1
                start_epoch = checkpoint['epoch'] + 1
                best_acc = checkpoint['best_acc']
                print(f"Resuming from epoch {start_epoch}")
                pass
        return global_step,start_epoch,best_acc
