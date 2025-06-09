import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

defalt_imgSize = (720, 1160)


def img_randn(B:int, C:int=3, W:int=defalt_imgSize[0], H:int=defalt_imgSize[1]):
    img = torch.randn(B, C, W, H)
    return img

def gen_label(B: int):
    '''0: nothing;1: vehical'''
    labels = torch.ones((B, 1), dtype=int)
    return labels


def bbox_rand(B: int, n_points: int = 4):
    bbox = torch.rand(B, 1, 2 * n_points)
    return bbox


def category_rand(B: int, nClass: int):
    l_class = torch.randint(0, nClass, (B, 1))
    return l_class


def LP_rand(B: int, nChars: int, N: int = 8):
    """
    nChars: num of chars in diction
    N: LP length
    """
    lp = torch.randint(0, nChars, (B, N))
    return lp


def batch_rand(batchSize=2):
    """return imgs and targets for LPD model forward"""
    imgs = img_randn(B=batchSize)
    labels = gen_label(batchSize)
    bboxes = bbox_rand(batchSize, 2)
    LPs = LP_rand(batchSize, nChars=75, N=8)
    targets = {"labels": labels, "boxes": bboxes, "LPs": LPs}
    # targets=[{"labels":labels[i],"boxes":bboxes[i],"LP":LPs[i]} for i in range(batchSize)]
    return imgs, targets


from tools.box_ops import validate_xyxy_bbox
def batch_rand_4pBBox(batchSize=2):
    """return imgs and targets for LPD model forward"""
    imgs = img_randn(B=batchSize)
    labels = gen_label(batchSize)
    bboxes = bbox_rand(batchSize, 2)
    _,bboxes=validate_xyxy_bbox(bboxes)
    bboxes_1 = bbox_rand(batchSize, 2)
    _,bboxes_1=validate_xyxy_bbox(bboxes_1)
    LPs = LP_rand(batchSize, nChars=75, N=8)
    targets = {"labels": labels, "boxes": bboxes,"boxes_1":bboxes_1, "LPs": LPs}
    # targets=[{"labels":labels[i],"boxes":bboxes[i],"LP":LPs[i]} for i in range(batchSize)]
    return imgs, targets


def batch_rand_tgtList(batchSize=2):
    """return imgs and targets for LPD model forward"""
    imgs = img_randn(B=batchSize)
    labels = gen_label(batchSize)
    bboxes = bbox_rand(batchSize, 2)
    LPs = LP_rand(batchSize, nChars=75, N=8)
    targets = [
        {"labels": labels[i], "boxes": bboxes[i], "LPs": LPs[i]}
        for i in range(batchSize)
    ]
    return imgs, targets



class Dataset_rand(Dataset):
    def __init__(self, length=10):
        self.length = length
        self.batch_name_space = ["imgs", "labels", "boxes", "LPs"]
        super().__init__()
        return

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img, tgt = batch_rand(batchSize=1)
        return (
            img.squeeze(0),
            tgt["labels"].squeeze(0),
            tgt["boxes"].squeeze(0),
            tgt["LPs"].squeeze(0),
        )


def collate_fn(batch):# TODO not work now
    imgs,labels,boxes,LPs=[torch.stack(tensor,0) for tensor in batch]
    return {"imgs":imgs,"labels":labels,'boxes':boxes,'LPs':LPs}
    # imgs=torch.stack(batch)
    imgs = []
    labels = []
    lengths = []
    lp_classes = []
    for _, sample in enumerate(batch):
        img, label, length, lp_class = sample
        imgs.append(img)
        labels.extend(label)
        lengths.append(length)
        lp_classes.append(lp_class)
    labels = np.asarray(labels).flatten().astype(int)

    return torch.stack(imgs, 0), torch.from_numpy(labels), lengths, lp_classes
if __name__=="__main__":
    batch_iterator=DataLoader(
        Dataset_rand(10), 2, True, num_workers=1,
    )#collate_fn=collate_fn,
    for i, batch in enumerate(batch_iterator):
        print(batch)
        pass
