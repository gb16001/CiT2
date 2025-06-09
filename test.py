# test model
import torch
import torch.optim as optim
import datasets.CCPD
import models_cit.ALPR
from datasets import gen_rand
from models_cit import Loss
from models_cit.Loss import Matcher
from tools.box_ops import validate_xyxy_bbox

import Dan_furnace
def train_Danlu_script():
    conf_file= 'configs/args-CCPD18-affine-res50-stn-2en-CDNv1_1.yaml'
    # 'configs/finetune-CLPD-args-CCPD18-affine-res50-stn-2en-CDNv1_1.yaml'
    # 'configs/args-CCPD20-affine-res50-stn-2en-CDNv1_1.yaml'
    # 'configs/args-CCPD18-affine-res50-stn-2en-CDNv1_1.yaml'
    trainer=Dan_furnace.Trainer_a_conf(conf_file)
    trainer.train()
train_Danlu_script()

def eval_model():
    conf_file='configs/test-args-CCPD20-affine-res50-stn-2en-CDNv1_1.yaml'
    # 'configs/test-args-CCPD18-affine-res50-stn-2en-CDNv1_1.yaml'
    results=[]
    evaluator=Dan_furnace.Eval_a_conf(conf_file)
    evaluator.test()
    for testSet in evaluator.args.val_set.csvPaths:
        print(f'testing {testSet}')
        evaluator.change_dataset(testSet)
        results.append(evaluator.test()) 
    print(results)
    return
# eval_model()

def eval_dataset():
    batchSize=64
    csvPath="datasets/CCPD/ccpd_tilt_test.csv"
    data=datasets.CCPD.CCPD_4pBbox_CDN(csvPath,8)
    batch_iterator=datasets.CCPD.dataset2loader(data,batch_size=batchSize,num_workers=8)
    datasetLen=len(batch_iterator)

    evaluator=Loss.IoU_LPs_evaluator()

    model=models_cit.ALPR.CiT_CND().cuda()
    model.eval()
    # with torch.no_grad():
    
    for idx,(inputs,targets) in enumerate(batch_iterator):
        inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()} 
        targets = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in targets.items()} 
        print(f'{idx}/{datasetLen}',end='\r')
        outputs = model(inputs)
        eval_batch=evaluator.forward_batch(outputs,targets)
        pass
    eval_result=evaluator.statistic_Dataset()
    print(eval_result)
    return
# eval_dataset()

def mini_forBackward():
    batchSize=2
    model=models_cit.ALPR.ALPR_res18_fm32_STN_CDNv1_1()
    # imgs,targets=gen_rand.batch_rand_4pBBox(batchSize)
    csvPath="datasets/CLPD/CLPD-train.csv"
    data=datasets.CCPD.CLPD_4pBbox_0size(csvPath,8)
    sample=data[10]
    batch_iterator=datasets.CCPD.dataset2loader(data,batch_size=batchSize,num_workers=1)
    imgs,targets=next(iter(batch_iterator))
    print(f"img={imgs}\ntargets={targets.keys()}\nbbox={targets['boxes']}")

    # model forward
    model.train()
    # model.eval()
    outputs = model(imgs)
    logits = outputs['pred_logits']
    boxes = outputs['pred_boxes']
    print(f"shapes={logits.shape, boxes.shape}")
    criterion=Loss.Infer_DenoiseLoss(None)
    # matcher = Matcher.single_4p_string_Matcher()
    # criterion=Loss.SetLoss_4p_string()
    # indexes = matcher.forward(outputs, targets)
    # print(f"indexes={indexes}")
    loss=criterion.forward(outputs,targets,details=False)
    
    #backprop
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()  # 清空历史梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新模型参数
    print(f"loss={loss}")
    return
# mini_forBackward()




# test dataset loader
import datasets
import time

from torchvision.transforms.functional import to_pil_image

def test_datasetLoader():
    csvPath='datasets/CCPD2018/ccpd_B+O_test.csv'
    "datasets/CCPD2018/ccpd_others_5%val.csv"
    data=datasets.CCPD.CCPD_4pBbox(csvPath,8,imgSize=[1160,720])
    good,bad=0,0
    for i in range(len(data)):
        vertexes=data._test_get4vertex(i)#b0x,b0y,b1x,b1y,b2x,b2y,b3x,b3y,
        if __goodBox(*vertexes):
            good+=1
        else:
            bad+=1
        pass
    print(f'{good},{bad}')

    None
    aSample=data[10]
    # save img
    # img = to_pil_image(aSample[0]['imgs'])
    # img.save("tensor_image_augment.png")
    batch_iterator=datasets.CCPD.dataset2loader(data,batch_size=64,num_workers=1)
    iterLen=len(batch_iterator)
    start_time = time.time()  # 记录开始时间
    for i, batch in enumerate(batch_iterator):
        print(f"{i}/{iterLen}",end='\r')
        # print(batch[0]['LPs_delay'].shape)
        # _,bbox=validate_xyxy_bbox(batch[3])
        pass
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算消耗的时间
    print(f"\nElapsed time: {elapsed_time} seconds")
# test_datasetLoader()

def __goodBox(b0x, b0y, b1x, b1y, b2x, b2y, b3x, b3y):
    good02_x,good02_y=b2x>b0x,b2y>b0y
    good02=good02_x==good02_y
    good13_x,good13_y=b1x>b3x,b3y>b1y
    good13 = good13_x ^ good13_y 
    goodBox=good02 and good13
    return goodBox


def test_randLoader():
    from datasets.gen_rand import Dataset_rand
    data=Dataset_rand(length=10000)
    batch_iterator=datasets.CCPD.dataset2loader(data,batch_size=32,num_workers=11)
    start_time = time.time()  # 记录开始时间
    for i, batch in enumerate(batch_iterator):
        print(i,end='\r')
        pass
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算消耗的时间
    print(f"\nElapsed time: {elapsed_time} seconds")
    return
def test_model():
    images = torch.randn(2, 3, 1140, 720)
    model=models_cit.ALPR.CiT()
    output=model(images)
    # print(output.shape)
    for key, value in output.items():
        print(f"{key}: {value.shape}")
    return
if __name__=="__main__":
    pass
def official_detr_demo():
    import torch
    import torchvision.transforms as T
    from PIL import Image
    import matplotlib.pyplot as plt
    import requests

    # 加载模型
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=True)
    model.eval()

    # 图像预处理
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 下载测试图像（你也可以替换为本地路径）
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)

    # 推理
    with torch.no_grad():
        outputs = model(img_tensor)

    # COCO 类别标签
    CLASSES = [
        'N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
        'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
        'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
        'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
        'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    # 输出预测结果
    probabilities = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    boxes = outputs['pred_boxes'][0]

    # 只显示置信度高于 0.9 的目标
    keep = probabilities.max(-1).values > 0.9

    def plot_results(pil_img, prob, boxes):
        plt.imshow(pil_img)
        ax = plt.gca()
        colors = ['r', 'g', 'b', 'y', 'm']
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes, colors):
            cl = p.argmax()
            label = f'{CLASSES[cl]}: {p[cl]:0.2f}'
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
            ax.text(xmin, ymin, label, fontsize=12,
                    bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        plt.show()

    # 将 [cx, cy, w, h] 转换为 [xmin, ymin, xmax, ymax]
    def rescale_bboxes(out_bbox, size):
        img_w, img_h = size
        b = out_bbox.cpu()
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        cx, cy, w, h = b.unbind(1)
        return torch.stack([cx - 0.5 * w,
                            cy - 0.5 * h,
                            cx + 0.5 * w,
                            cy + 0.5 * h], dim=1)

    plot_results(image, probabilities[keep], rescale_bboxes(boxes[keep], image.size))
    return


def test_Bbox_fix_func():
    bboxes = torch.tensor([
    [50, 30, 100, 80],   # 合法
    [120, 60, 110, 90],  # x0 > x1，需要修正
    [40, 100, 70, 90],   # y0 > y1，需要修正
])
    is_valid, corrected_bboxes = validate_xyxy_bbox(bboxes)

    print("原始是否合法:", is_valid)
    print("修正后的 bbox:\n", corrected_bboxes)

# test_Bbox_fix_func()
