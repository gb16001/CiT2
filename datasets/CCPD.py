import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import cv2
from jpeg4py import JPEG

import pandas as pd
import os
from datasets.chars import CHARS_DICT
import numpy as np
from tools.box_ops import validate_xyxy_bbox,box_xyxy_to_cxcywh
import albumentations as A

class PreprocFuns:
    @staticmethod
    def img_color_affine_aug_cv2():
        strong_augment = A.Compose(
            [
                
                A.ColorJitter(
                    brightness=(0.8, 1.2),
                    contrast=(0.8, 1.2),
                    saturation=(0.8, 1.2),
                    hue=(-0.1, 0.1),
                    p=0.5,
                ),
                # A.ToGray(p=0.01),
                A.Affine(
                    scale=(0.8, 1.2),
                    rotate=(-30, 30),
                    shear=(-30, 30),
                    translate_percent=(-0.3, 0.3),
                    p=0.8,
                ),
                A.Normalize(mean=0, std=1),
                A.pytorch.ToTensorV2(),
            ],
            # bbox_params=A.BboxParams(format='pascal_voc'),
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )
        return strong_augment
    @staticmethod
    def img_resize_color_affine_aug_cv2(imgSize):
        strong_augment = A.Compose(
            [
                A.Resize(height=imgSize[0], width=imgSize[1]),
                A.ColorJitter(
                    brightness=(0.8, 1.2),
                    contrast=(0.8, 1.2),
                    saturation=(0.8, 1.2),
                    hue=(-0.1, 0.1),
                    p=0.5,
                ),
                # A.ToGray(p=0.01),
                A.Affine(
                    scale=(0.8, 1.2),
                    rotate=(-30, 30),
                    shear=(-30, 30),
                    translate_percent=(-0.3, 0.3),
                    p=0.8,
                ),
                A.Normalize(mean=0, std=1),
                A.pytorch.ToTensorV2(),
            ],
            # bbox_params=A.BboxParams(format='pascal_voc'),
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
        )
        # A.Compose(
        #     [
        #         A.Resize(height=imgSize[0], width=imgSize[1]),
        #         A.ColorJitter(p=0.5),
        #         A.ToGray(p=0.01),
        #         A.Normalize(mean=0, std=1),
        #         A.pytorch.ToTensorV2(),
        #     ]
        # )
        return strong_augment

    @staticmethod
    def img_augment_no_norm_cv2(imgSize):
        # mean=(0.485, 0.456, 0.406)
        # std=(0.229, 0.224, 0.225)
        strong_augment = A.Compose(
            [
                A.Resize(height=imgSize[0], width=imgSize[1]),
                A.ColorJitter(
                    p=0.5
                ),
                A.ToGray(p=0.01),
                A.Normalize(
                    mean=0,std=1
                ),
                A.pytorch.ToTensorV2(),
            ]
        )
        return strong_augment
    @staticmethod
    def img_augment_cv2(imgSize):
        mean=(0.485, 0.456, 0.406)
        std=(0.229, 0.224, 0.225)
        strong_augment = A.Compose(
            [
                A.Resize(height=imgSize[0], width=imgSize[1]),
                A.ColorJitter(),
                A.ToGray(p=0.01),
                A.Normalize(
                    mean=mean,  
                    std=std,
                ),
                A.pytorch.ToTensorV2(),
            ]
        )
        return strong_augment

    @staticmethod
    def strong_augment(imgSize):
        '''imgSize:(y,x)
        '''
        # 定义增强管道
        strong_augment = transforms.Compose(
            [
                transforms.Resize(imgSize),
                # color augment
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),  # 调整色彩
                transforms.RandomGrayscale(p=0.01),
                # shape augment
                # transforms.RandomHorizontalFlip(p=0.1),  # 随机水平翻转
                # transforms.RandomVerticalFlip(p=0.1),
                # transforms.RandomRotation(degrees=30),  # 随机旋转
                # transforms.RandomResizedCrop(size=(224, 224)), # 随机裁剪后调整大小
                # transforms.RandomApply(
                #     [transforms.RandomAffine(
                #         degrees=(-30, 30),  # 随机旋转角度范围
                #         translate=(0.3, 0.3),  # 随机平移范围（宽度、高度的比例）
                #         scale=(0.8, 1.2),  # 随机缩放范围
                #         shear=(-30, 30),  # 随机剪切角度范围
                #         fill=0,  # 填充像素值（0 表示黑色）
                #     )],
                #     p=0.8  # 应用 RandomAffine 的概率
                # ),
                # transforms.RandomPerspective(
                #     distortion_scale=0.5, p=0.5
                # ),  # 随机透视变换
                # tensor augment
                transforms.ToTensor(),  # 转为张量
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),  # 标准化imagenet distribution
                # transforms.RandomErasing(p=0.5),               # 随机擦除
            ]
        )
        return strong_augment
    @staticmethod
    def resize(imgSize):
        '''imgSize:(y,x)
        '''
        return transforms.Compose(
                [
                    transforms.Resize(imgSize),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 标准化图像
                ]
            )
    @staticmethod
    def resize_cv2(img_size):
        """img_size: (height, width)"""
        return A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(mean=0, std=1),
            A.pytorch.ToTensorV2(),
        ])
    @staticmethod
    def resize_cv2_A(img_size):
        """img_size: (height, width)"""
        return A.Compose([
            A.Resize(height=img_size[0], width=img_size[1]),
            A.Normalize(mean=0, std=1),
            A.pytorch.ToTensorV2(),
        ],keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),)
    
    # A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),  # 如果需要标准化，可取消注释
    pass


class CCPD_base(Dataset):
    r"""
    load CCPD img and tgt, return list:  
    imgs_tensor[3,y,x],
    labels[1],
    boxes[1,4],
    LPs[8]
    """
    defaltOutImgSize = (290, 720) #y,x
    imgSize=(720,1160) # used to calcu relative pixel location

    @staticmethod
    def rescale(img_int, size):
        return img_int.resize(size)

    def __init__(
        self,
        csvFile,
        lpr_max_len=8,
        PreprocFun=None,
        shuffle=False,
        imgSize=defaltOutImgSize,
    ):
        self.df = pd.read_csv(csvFile)
        self.batch_name_space = ["imgs", "labels", "boxes", "LPs"]
        # 获取列的位置

        keys = [
            "filename",
            "CCPD_path",
            "license_plate",
            "bounding_box_1_x",
            "bounding_box_1_y",
            "bounding_box_2_x",
            "bounding_box_2_y",
        ]
        self.col_indexes = [self.df.columns.get_loc(key) for key in keys]
        # get dirpath of ccpd
        self.CCPD_dir = os.path.dirname(csvFile)
        # shuffle self.anno_csv
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

        self.lp_max_len = lpr_max_len
        # PreprocFun. default resize img only. def other fun if need data augment.
        self.PreprocFun = (
            PreprocFuns.resize_cv2(imgSize) if PreprocFun is None else PreprocFun
        )
        
        return

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        r"""
        imgs_tensor[3,y,x],
        labels[1],
        boxes[1,4],
        LPs[8],
        """
        filename, CCPD_path, license_plate, bbox1x, bbox1y, bbox2x, bbox2y = (
            self.df.iloc[index, self.col_indexes]
        )
        imgs_tensor = self.read_imgs_tensor_cv2(filename, CCPD_path)
        LPs = self.read_LPs(license_plate)
        labels = torch.ones(1, dtype=int)
        boxes = self.gen_bbox(bbox1x, bbox1y, bbox2x, bbox2y)
        return (
            imgs_tensor,
            labels,
            boxes,
            LPs,
        )
    @staticmethod
    def gen_bbox( bbox1x, bbox1y, bbox2x, bbox2y):
        boxes = torch.tensor((bbox1x/CCPD_base.imgSize[0], bbox1y/CCPD_base.imgSize[1], bbox2x/CCPD_base.imgSize[0], bbox2y/CCPD_base.imgSize[1]),dtype=torch.float32).unsqueeze(0)
        _,boxes=validate_xyxy_bbox(boxes)
        return boxes

    def read_LPs(self, license_plate, allow_worning=False):
        license_plate = license_plate.ljust(
            self.lp_max_len, "-"
        )  # license_plate.len = 7 or 8
        try:
            LPs = torch.tensor([CHARS_DICT[c] for c in license_plate])
        except KeyError as e:
            import warnings

            warnings.warn(
                f"Character {e.args[0]} not found in CHARS_DICT. Assigning default value 0."
            ) if allow_worning else None
            LPs = torch.tensor([CHARS_DICT.get(c, 0) for c in license_plate])
            pass
        return LPs

    def read_imgs_tensor_PIL(self, filename, CCPD_path):
        filePath = f"{self.CCPD_dir}/{CCPD_path}/{filename}"  # os.path.join(self.CCPD_dir, CCPD_path, filename)
        img_int = Image.open(filePath)
        imgs_tensor = self.PreprocFun(img_int)
        return imgs_tensor
    def read_imgs_tensor_cv2(self, filename, CCPD_path):
        filePath = f"{self.CCPD_dir}/{CCPD_path}/{filename}"
        # image = JPEG('your_image.jpg').decode()
        # 用 cv2 加载图像（BGR）
        img_bgr = cv2.imread(filePath)
        if img_bgr is None:
            raise FileNotFoundError(f"Image not found: {filePath}")
        
        # 转换为 RGB（cv2 默认是 BGR）
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # # 如果你的 PreprocFun 是基于 PIL 的，这里要转回 Image
        # img_rgb_pil = Image.fromarray(img_rgb)
        
        imgs_tensor = self.PreprocFun(image=img_rgb)['image']
        return imgs_tensor
    def read_imgs_tensor_jpeg4py(self, filename, CCPD_path):
        '''jpeg4py load'''
        filePath = f"{self.CCPD_dir}/{CCPD_path}/{filename}"
        image = JPEG(filePath).decode()
        
        imgs_tensor = self.PreprocFun(image=image)['image']
        return imgs_tensor

    pass
class CCPD_4pBbox(CCPD_base):
    def __init__(self, csvFile, lpr_max_len=8, PreprocFun=None, shuffle=False, imgSize=CCPD_base.defaltOutImgSize):
        super().__init__(csvFile, lpr_max_len, PreprocFun, shuffle, imgSize)
        self.batch_name_space = ["imgs", "labels", "boxes","boxes1", "LPs"]
        keys = [
            "filename",
            "CCPD_path",
            "license_plate",
            'vertex_1_x',
            'vertex_1_y',
            'vertex_3_x',
            'vertex_3_y',
            'vertex_2_x',
            'vertex_2_y',
            'vertex_4_x',
            'vertex_4_y',
            # "bounding_box_1_x",
            # "bounding_box_1_y",
            # "bounding_box_2_x",
            # "bounding_box_2_y",
        ]
        self.col_indexes = [self.df.columns.get_loc(key) for key in keys]
        return
    def __getitem__(self, index):
        r"""
        imgs_tensor[3,y,x],
        tgt:['labels':1=lp, 'boxes':xyxy, 'boxes_1', 'LPs']

        labels[1],
        boxes[1,4],
        boxes1[1,4],
        LPs[8],
        """
        filename, CCPD_path, license_plate, bboxx1, bboxy1, bboxx2, bboxy2,box1x1,boxy1,box1x2,box1y2 = (
            self.df.iloc[index, self.col_indexes]
        )
        imgs_tensor = self.read_imgs_tensor_jpeg4py(filename, CCPD_path)
        LPs = self.read_LPs(license_plate)
        labels = torch.ones(1, dtype=int)
        boxes=self.gen_bbox(bboxx1, bboxy1, bboxx2, bboxy2)
        boxes1=self.gen_bbox(box1x1,boxy1,box1x2,box1y2)
        target={'labels':labels, 'boxes':boxes, 'boxes_1':boxes1, 'LPs':LPs}
        inputs={'imgs':imgs_tensor}
        return (
            inputs,
            target
        )
    def _test_get4vertex(self,index):
        filename, CCPD_path, license_plate, bboxx1, bboxy1, bboxx2, bboxy2,box1x1,boxy1,box1x2,box1y2 = (
            self.df.iloc[index, self.col_indexes]
        )
        return bboxx1, bboxy1, bboxx2, bboxy2,box1x1,boxy1,box1x2,box1y2
    pass 
class CCPD_4pBbox_0size(CCPD_4pBbox):
    def __init__(self, csvFile, lpr_max_len=8, PreprocFun=None, shuffle=False, imgSize=CCPD_base.imgSize[::-1]):
        super().__init__(csvFile, lpr_max_len, PreprocFun, shuffle, imgSize)
        return
    pass
class CCPD_4pBbox_CDN(CCPD_4pBbox):
    def __init__(self, csvFile, lpr_max_len=8, PreprocFun=None, shuffle=False, imgSize=CCPD_base.defaltOutImgSize,noise_level=0.05,max_neg_noise=0.2,LP_s0_index:int=0):
        super().__init__(csvFile, lpr_max_len, PreprocFun, shuffle, imgSize)
        
        self.noise_level,self.max_neg_noise,self.LP_s0_index=noise_level,max_neg_noise,LP_s0_index
        return
    def __getitem__(self, index):
        r"""
        inputs:{'imgs':[3,y,x],'pos_bbox':xywhxywh,'neg_bbox':xywhxywh,'LPs_delay':[8]}
        tgt:['labels':1=lp, 'boxes':xyxy, 'boxes_1', 'LPs']

        labels[1],
        boxes[1,4],
        boxes1[1,4],
        LPs[8],
        """
        inputs,targets=super().__getitem__(index)
        imgs_tensor=inputs['imgs']
        boxes=targets['boxes']
        boxes1=targets['boxes_1']
        LPs=targets['LPs']
        P_noise_box=self.bbox_add_noise(boxes,self.noise_level,'pos',self.max_neg_noise)
        N_noise_box=self.bbox_add_noise(boxes,self.noise_level,'neg',self.max_neg_noise)
        P_noise_box1=self.bbox_add_noise(boxes1,self.noise_level,'pos',self.max_neg_noise)
        N_noise_box1=self.bbox_add_noise(boxes1,self.noise_level,'neg',self.max_neg_noise)
        pos_bbox=torch.cat((P_noise_box,P_noise_box1),dim=-1)
        neg_bbox=torch.cat((N_noise_box,N_noise_box1),dim=-1)
        

        LPs_delay = torch.cat([torch.tensor([self.LP_s0_index]), LPs[:-1]])
        # LPs_delay=LPs[1:]
        
        inputs={'imgs':imgs_tensor,'pos_bbox':pos_bbox,'neg_bbox':neg_bbox,'LPs_delay':LPs_delay}
        return inputs,targets
    @staticmethod
    def bbox_add_noise(bbox, noise_level=0.05, mode='pos', max_neg_noise=0.2,eps=1e-4):
        """
        参数:
            bbox: Tensor([x1, y1, x2, y2]) 归一化坐标
            mode: 'pos' 或 'neg'
        返回:
            Tensor([x1', y1', x2', y2']) 带噪声的新 bbox
        """
        # 转中心宽高格式
        x, y, w, h = box_xyxy_to_cxcywh(bbox).unbind(-1)
        # w,h maybe 0
        def rand_uniform(min_val, max_val):
            if isinstance(min_val, torch.Tensor):
                min_val = min_val.item()
            if isinstance(max_val, torch.Tensor):
                max_val = max_val.item()
            return torch.empty(1).uniform_(min_val, max_val).item()

        def random_sign():
            return 1.0 if torch.rand(1).item() > 0.5 else -1.0

        if mode == 'pos':
            dx = rand_uniform(-w * noise_level, w * noise_level)
            dy = rand_uniform(-h * noise_level, h * noise_level)
            dw = rand_uniform(-w * noise_level, w * noise_level)
            dh = rand_uniform(-h * noise_level, h * noise_level)
        elif mode == 'neg':
            dx = rand_uniform(w * noise_level, w * max_neg_noise) * random_sign()
            dy = rand_uniform(h * noise_level, h * max_neg_noise) * random_sign()
            dw = rand_uniform(w * noise_level, w * max_neg_noise) * random_sign()
            dh = rand_uniform(h * noise_level, h * max_neg_noise) * random_sign()
        else:
            raise ValueError("mode must be 'pos' or 'neg'")

        # 加扰动
        new_x = x + dx
        new_y = y + dy
        new_w = w + dw
        new_h = h + dh

        # 限制宽高最小为一个阈值（避免空bbox）
        
        eps_t = torch.tensor([eps-4])
        # if (new_h<eps or new_h<eps).item():
        #     print(new_w , w , dw,new_h , h , dh)
        new_w = max(new_w, eps_t)
        new_h = max(new_h, eps_t)

        return torch.stack((new_x,new_y,new_w,new_h),dim=-1)
    pass

class CCPD_4pBbox_CDN_0size(CCPD_4pBbox_CDN):
    def __init__(self, csvFile, lpr_max_len=8, PreprocFun=None, shuffle=False, imgSize=CCPD_base.imgSize[::-1], noise_level=0.05, max_neg_noise=0.2, LP_s0_index = 0):
        super().__init__(csvFile, lpr_max_len, PreprocFun, shuffle, imgSize, noise_level, max_neg_noise, LP_s0_index)
        return
class CCPD_4pBbox_CDN_augment_0size(CCPD_4pBbox_CDN):
    '''
    img augment and not resize img.
    '''
    def __init__(self, csvFile, lpr_max_len=8, PreprocFun=None, shuffle=False, imgSize=CCPD_base.imgSize[::-1], noise_level=0.05, max_neg_noise=0.2, LP_s0_index = 0):
        if PreprocFun is None:
            PreprocFun=PreprocFuns.img_augment_cv2(imgSize)
        super().__init__(csvFile, lpr_max_len, PreprocFun, shuffle, imgSize, noise_level, max_neg_noise, LP_s0_index)
    pass
class CCPD_4pBbox_CDN_augment_noNorm_0size(CCPD_4pBbox_CDN):
    '''
    img augment and not resize img.
    without imgnet norm
    '''
    def __init__(self, csvFile, lpr_max_len=8, PreprocFun=None, shuffle=False, imgSize=CCPD_base.imgSize[::-1], noise_level=0.05, max_neg_noise=0.2, LP_s0_index = 0):
        if PreprocFun is None:
            PreprocFun=PreprocFuns.img_augment_no_norm_cv2(imgSize)
        super().__init__(csvFile, lpr_max_len, PreprocFun, shuffle, imgSize, noise_level, max_neg_noise, LP_s0_index)
    pass

class CCPD_4pBbox_CDN_augment(CCPD_4pBbox_CDN_augment_0size):
    def __init__(self, csvFile, lpr_max_len=8, PreprocFun=None, shuffle=False, imgSize=CCPD_base.defaltOutImgSize, noise_level=0.05, max_neg_noise=0.2, LP_s0_index=0):
        super().__init__(csvFile, lpr_max_len, PreprocFun, shuffle, imgSize, noise_level, max_neg_noise, LP_s0_index)
        return

class CCPD_4pBbox_affin_noNormAug_CDN_0size(Dataset):
    '''
    color and affine augment and not resize img.
    without imgnet norm
    '''
    def __init__(self, csvFile, lpr_max_len=8, PreprocFun=None, shuffle=False, imgSize=CCPD_base.imgSize[::-1], noise_level=0.05, max_neg_noise=0.2, LP_s0_index = 0):
        if PreprocFun is None:
            PreprocFun=PreprocFuns.img_resize_color_affine_aug_cv2(imgSize)
            pass 
        self.df = pd.read_csv(csvFile)
        # get dirpath of ccpd
        self.CCPD_dir = os.path.dirname(csvFile)
        # shuffle self.anno_csv
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

        self.lp_max_len = lpr_max_len
        # PreprocFun. default resize img only. def other fun if need data augment.
        self.PreprocFun = PreprocFun
        self.batch_name_space = ["imgs", "labels", "boxes","boxes1", "LPs"]
        keys = [
            "filename",
            "CCPD_path",
            "license_plate",
            'vertex_1_x',
            'vertex_1_y',
            'vertex_3_x',
            'vertex_3_y',
            'vertex_2_x',
            'vertex_2_y',
            'vertex_4_x',
            'vertex_4_y',
        ]
        self.col_indexes = [self.df.columns.get_loc(key) for key in keys]
        self.noise_level,self.max_neg_noise,self.LP_s0_index=noise_level,max_neg_noise,LP_s0_index
        return
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        r"""
        inputs:{'imgs':[3,y,x],'pos_bbox':xywhxywh,'neg_bbox':xywhxywh,'LPs_delay':[8]}
        tgt:['labels':1=lp, 'boxes':xyxy, 'boxes_1', 'LPs']

        labels[1],
        boxes[1,4],
        boxes1[1,4],
        LPs[8],
        """
        filename, CCPD_path, license_plate, bboxx1, bboxy1, bboxx2, bboxy2,box1x1,box1y1,box1x2,box1y2 = (
            self.df.iloc[index, self.col_indexes]
        )
        keypoints = [(bboxx1, bboxy1), (bboxx2, bboxy2), (box1x1, box1y1), (box1x2, box1y2)]
        filePath = f"{self.CCPD_dir}/{CCPD_path}/{filename}"
        image = JPEG(filePath).decode()
        
        transformed_dict = self.PreprocFun(image=image,keypoints=keypoints)
        imgs_tensor = transformed_dict['image']
        keypoints = transformed_dict['keypoints']
        # imgs_tensor = self.read_imgs_tensor_jpeg4py(filename, CCPD_path)
        boxes=CCPD_base.gen_bbox(*keypoints[0], *keypoints[1])
        boxes1=CCPD_base.gen_bbox(*keypoints[2], *keypoints[3])

        LPs = CCPD_base.read_LPs(self,license_plate)
        labels = torch.ones(1, dtype=int)
        targets={'labels':labels, 'boxes':boxes, 'boxes_1':boxes1, 'LPs':LPs}
        
        boxes=targets['boxes']
        boxes1=targets['boxes_1']
        LPs=targets['LPs']

        P_noise_box=CCPD_4pBbox_CDN.bbox_add_noise(boxes,self.noise_level,'pos',self.max_neg_noise)
        N_noise_box=CCPD_4pBbox_CDN.bbox_add_noise(boxes,self.noise_level,'neg',self.max_neg_noise)
        P_noise_box1=CCPD_4pBbox_CDN.bbox_add_noise(boxes1,self.noise_level,'pos',self.max_neg_noise)
        N_noise_box1=CCPD_4pBbox_CDN.bbox_add_noise(boxes1,self.noise_level,'neg',self.max_neg_noise)
        pos_bbox=torch.cat((P_noise_box,P_noise_box1),dim=-1)
        neg_bbox=torch.cat((N_noise_box,N_noise_box1),dim=-1)
        

        LPs_delay = torch.cat([torch.tensor([self.LP_s0_index]), LPs[:-1]])
        # LPs_delay=LPs[1:]
        
        inputs={'imgs':imgs_tensor,'pos_bbox':pos_bbox,'neg_bbox':neg_bbox,'LPs_delay':LPs_delay}
        return inputs,targets
    pass

class CLPD_4pBbox_affin_noNormAug_CDN_0size(Dataset):
    def __init__(self, csvFile, lpr_max_len=8, PreprocFun=None, imgSize=CCPD_base.imgSize[::-1],shuffle=False, noise_level=0.05, max_neg_noise=0.2, LP_s0_index = 0):
        if PreprocFun is None:
            PreprocFun=PreprocFuns.img_resize_color_affine_aug_cv2(imgSize)
            pass 
        self.df = pd.read_csv(csvFile)
        # get dirpath of ccpd
        self.CCPD_dir = os.path.dirname(csvFile)
        # shuffle self.anno_csv
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)

        self.lp_max_len = lpr_max_len
        # PreprocFun. default resize img only. def other fun if need data augment.
        self.PreprocFun = PreprocFun
        self.batch_name_space = ["imgs", "labels", "boxes","boxes1", "LPs"]
        keys = [
            "filename",
            "CCPD_path",
            "label",
            'x1',
            'y1',
            'x3',
            'y3',
            'x2',
            'y2',
            'x4',
            'y4',
        ]
        self.col_indexes = [self.df.columns.get_loc(key) for key in keys]
        self.noise_level,self.max_neg_noise,self.LP_s0_index=noise_level,max_neg_noise,LP_s0_index
        return
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        r"""
        inputs:{'imgs':[3,y,x],'pos_bbox':xywhxywh,'neg_bbox':xywhxywh,'LPs_delay':[8]}
        tgt:['labels':1=lp, 'boxes':xyxy, 'boxes_1', 'LPs']

        labels[1],
        boxes[1,4],
        boxes1[1,4],
        LPs[8],
        """
        filename, CCPD_path, license_plate, bboxx1, bboxy1, bboxx2, bboxy2,box1x1,box1y1,box1x2,box1y2 = (
            self.df.iloc[index, self.col_indexes]
        )
        keypoints = [(bboxx1, bboxy1), (bboxx2, bboxy2), (box1x1, box1y1), (box1x2, box1y2)]
        filePath = f"{self.CCPD_dir}/{CCPD_path}/{filename}"
        image = JPEG(filePath).decode()
        
        transformed_dict = self.PreprocFun(image=image,keypoints=keypoints)
        imgs_tensor = transformed_dict['image']
        keypoints = transformed_dict['keypoints']
        boxes = CCPD_base.gen_bbox( *keypoints[0], *keypoints[1])
        boxes1=CCPD_base.gen_bbox(*keypoints[2], *keypoints[3])


        LPs = CCPD_base.read_LPs(self,license_plate)
        labels = torch.ones(1, dtype=int)
        targets={'labels':labels, 'boxes':boxes, 'boxes_1':boxes1, 'LPs':LPs}
        
        boxes=targets['boxes']
        boxes1=targets['boxes_1']
        LPs=targets['LPs']

        P_noise_box=CCPD_4pBbox_CDN.bbox_add_noise(boxes,self.noise_level,'pos',self.max_neg_noise)
        N_noise_box=CCPD_4pBbox_CDN.bbox_add_noise(boxes,self.noise_level,'neg',self.max_neg_noise)
        P_noise_box1=CCPD_4pBbox_CDN.bbox_add_noise(boxes1,self.noise_level,'pos',self.max_neg_noise)
        N_noise_box1=CCPD_4pBbox_CDN.bbox_add_noise(boxes1,self.noise_level,'neg',self.max_neg_noise)
        pos_bbox=torch.cat((P_noise_box,P_noise_box1),dim=-1)
        neg_bbox=torch.cat((N_noise_box,N_noise_box1),dim=-1)
        

        LPs_delay = torch.cat([torch.tensor([self.LP_s0_index]), LPs[:-1]])
        # LPs_delay=LPs[1:]
        
        inputs={'imgs':imgs_tensor,'pos_bbox':pos_bbox,'neg_bbox':neg_bbox,'LPs_delay':LPs_delay}
        return inputs,targets

    def gen_bbox(self, height, width, bbox1x, bbox1y, bbox2x, bbox2y):
        boxes = torch.tensor((bbox1x/width, bbox1y/height, bbox2x/width, bbox2y/height),dtype=torch.float32).unsqueeze(0)
        _,boxes=validate_xyxy_bbox(boxes)
        return boxes
class CLPD_4pBbox_0size(CCPD_base):
    def __init__(self, csvFile, lpr_max_len=8, PreprocFun=None, shuffle=False, imgSize=CCPD_base.imgSize[:: -1]):
        if PreprocFun is None:
            PreprocFun=PreprocFuns.resize_cv2_A(imgSize)
            pass 
        self.df = pd.read_csv(csvFile)
        # get dirpath of ccpd
        self.CCPD_dir = os.path.dirname(csvFile)
        # shuffle self.anno_csv
        if shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.lp_max_len = lpr_max_len
        self.PreprocFun = PreprocFun
        self.batch_name_space = ["imgs", "labels", "boxes","boxes1", "LPs"]
        keys = [
            "filename",
            "CCPD_path",
            "label",
            'x1',
            'y1',
            'x3',
            'y3',
            'x2',
            'y2',
            'x4',
            'y4',
        ]
        self.col_indexes = [self.df.columns.get_loc(key) for key in keys]
        return
    def __len__(self):
        return len(self.df)
    def __getitem__(self, index):
        r"""
        inputs:{'imgs':[3,y,x],'pos_bbox':xywhxywh,'neg_bbox':xywhxywh,'LPs_delay':[8]}
        tgt:['labels':1=lp, 'boxes':xyxy, 'boxes_1', 'LPs']

        labels[1],
        boxes[1,4],
        boxes1[1,4],
        LPs[8],
        """
        filename, CCPD_path, license_plate, bboxx1, bboxy1, bboxx2, bboxy2,box1x1,box1y1,box1x2,box1y2 = (
            self.df.iloc[index, self.col_indexes]
        )
        keypoints = [(bboxx1, bboxy1), (bboxx2, bboxy2), (box1x1, box1y1), (box1x2, box1y2)]
        filePath = f"{self.CCPD_dir}/{CCPD_path}/{filename}"
        image = JPEG(filePath).decode()
        
        transformed_dict = self.PreprocFun(image=image,keypoints=keypoints)
        imgs_tensor = transformed_dict['image']
        keypoints = transformed_dict['keypoints']
        boxes = CCPD_base.gen_bbox( *keypoints[0], *keypoints[1])
        boxes1=CCPD_base.gen_bbox(*keypoints[2], *keypoints[3])


        LPs = CCPD_base.read_LPs(self,license_plate)
        labels = torch.ones(1, dtype=int)
        targets={'labels':labels, 'boxes':boxes, 'boxes_1':boxes1, 'LPs':LPs}

        
        inputs={'imgs':imgs_tensor}
        return inputs,targets
    pass
from torch.utils.data._utils.collate import default_collate

def collate_fn_CLPD(batch):
    inputs_list, targets_list = zip(*batch)  # 拆成两个 tuple

    # 提取 imgs 作为 list 保留
    imgs = [x['imgs'] for x in inputs_list]

    # 剩下的字段用 default_collate
    other_inputs = {
        key: default_collate([x[key] for x in inputs_list])
        for key in inputs_list[0]
        if key != 'imgs'
    }

    # 组装最终 batch
    inputs_batch = {'imgs': imgs, **other_inputs}
    return inputs_batch, default_collate(targets_list)
def dataset2loader(
    dataset: Dataset,
    batch_size=16,
    shuffle=True,
    num_workers=8,
    collate_fn=None,
    pin_memory=True
):
    return DataLoader(
        dataset, batch_size, shuffle, num_workers=num_workers, collate_fn=collate_fn,pin_memory=pin_memory
    )
