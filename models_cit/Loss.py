import torch
from torch import nn
import torch.nn.functional as F
import torchvision.ops as ops
from scipy.optimize import linear_sum_assignment
from tools.box_ops import box_cxcywh_to_xyxy
class ClassifyLoss:
    
    CEloss = nn.CrossEntropyLoss # input=[B,C,d_1, ...]

    class FocalLoss(nn.Module):
        def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs, targets):
            # logits =>log_softmax
            log_pt = F.log_softmax(inputs, dim=1)
            pt = torch.exp(log_pt)  
            
            # gather log_pt
            log_pt = log_pt.gather(1, targets.view(-1, 1)).squeeze()
            pt = pt.gather(1, targets.view(-1, 1)).squeeze()

            ######FL(pt​)=−α(1−pt​)γlog(pt​)######
            loss = -self.alpha * (1 - pt) ** self.gamma * log_pt
            
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss

    pass
class BboxLoss:
    '''
    L1 loss
    L2 loss
    '''
    L1_loss = nn.L1Loss
    L2_loss = nn.MSELoss
    GIoULoss=ops.generalized_box_iou_loss
    pass

class LP_Loss:
    """
    ctc 
    focal 
    ce 
    """
    CEloss = nn.CrossEntropyLoss # input=[B,C,d_1, ...]
    FocalLoss=ClassifyLoss.FocalLoss # same as CE loss
    CTCloss=nn.CTCLoss # log prob, tgt, len prob, len tgt
    pass


class Matcher:
    class HungarianMatcher(nn.Module):
        """
        from DETR repo \n
        This class computes an assignment between the targets and the predictions of the network

        For efficiency reasons, the targets don't include the no_object. Because of this, in general,
        there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
        while the others are un-matched (and thus treated as non-objects).
        """

        def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
            """Creates the matcher

            Params:
                cost_class: This is the relative weight of the classification error in the matching cost
                cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
                cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
            """
            super().__init__()
            self.cost_class = cost_class
            self.cost_bbox = cost_bbox
            self.cost_giou = cost_giou
            assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        @torch.no_grad()
        def forward(self, outputs, targets):
            """ Performs the matching

            Params:
                outputs: This is a dict that contains at least these entries:
                    "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                    "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

                targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                    "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                            objects in the target) containing the class labels
                    "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

            Returns:
                A list of size batch_size, containing tuples of (index_i, index_j) where:
                    - index_i is the indices of the selected predictions (in order)
                    - index_j is the indices of the corresponding selected targets (in order)
                For each batch element, it holds:
                    len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
            """
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
            out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets])
            tgt_bbox = torch.cat([v["boxes"] for v in targets])

            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -out_prob[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

            # Compute the giou cost betwen boxes
            cost_giou = -ops.generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()#C: batchsize, n_quairy, sum(n_i). too big

            sizes = [len(v["boxes"]) for v in targets]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    class RecessionaryMatcher(nn.Module):
        '''recession when ground truth sample for each img is ALWAYS 1'''
        def __init__(self, coef_class: float = 1, coef_bbox: float = 5, coef_giou: float = 2):
            assert coef_class != 0 or coef_bbox != 0 or coef_giou != 0, "all coefficients cant be 0"
            super().__init__()
            self.p_class, self.p_bbox, self.p_giou = coef_class, coef_bbox, coef_giou
            # self.cost_class = lambda out_prob, tgt_ids: -out_prob[:, tgt_ids]
            # self.cost_bbox = lambda out_bbox, tgt_bbox: torch.cdist(out_bbox, tgt_bbox, p=1)
            # self.cost_giou = lambda out_bbox, tgt_bbox: -ops.generalized_box_iou(
            #     box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
            # )
            return

        @torch.no_grad()
        def forward(self, outputs, targets):
            bs, num_queries = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].softmax(-1)  # [batch_size , num_queries, num_classes]
            out_bbox = outputs["pred_boxes"]  # [batch_size , num_queries, 4]

            # Also concat the target labels and boxes
            tgt_ids=targets["labels"]
            tgt_bbox=targets["boxes"]

            # tgt_ids, tgt_bbox = self.tgt_List2Tensor(targets)
            device=tgt_bbox.device
            cost_class=torch.zeros(bs,num_queries,1,device=device)
            cost_bbox=torch.zeros(bs,num_queries,1,device=device)
            cost_giou=torch.zeros(bs,num_queries,1,device=device)
            for i in range(bs):
                cost_class[i]=-out_prob[i,:,tgt_ids[i]]
                cost_bbox[i]=torch.cdist(out_bbox[i], tgt_bbox[i], p=1)
                cost_giou[i]=-ops.generalized_box_iou(box_cxcywh_to_xyxy(out_bbox[i]), box_cxcywh_to_xyxy(tgt_bbox[i]))
                pass
            # coefficients = torch.tensor([self.p_bbox, self.p_class, self.p_giou],device=device)
            # cost_matrix = torch.stack([cost_bbox, cost_class, cost_giou], dim=0)
            # C = torch.matmul(coefficients, cost_matrix)
            # Final cost matrix
            C = self.p_bbox * cost_bbox + self.p_class * cost_class + self.p_giou * cost_giou
            
            C = C.view(bs, num_queries, -1)
            indices=torch.argmin(C,1)
            return indices
            indices = [linear_sum_assignment(C[i]) for i in range(bs)]
            # sizes = [len(v["boxes"]) for v in targets]
            # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

        def tgt_List2Tensor(self, targets):
            tgt_ids = torch.cat([v["labels"] for v in targets])#B
            tgt_bbox = torch.cat([v["boxes"] for v in targets]).unsqueeze(1)#B,1,4
            return tgt_ids,tgt_bbox
    class single_4p_Matcher(nn.Module):
        '''
        1 sample per img, 4 vertex bbox
        '''
        def __init__(self, k_class: float = 1, k_bbox: float = 5, k_giou: float = 2,k_bbox1: float = 5, k_giou1: float = 2):
            super().__init__()
            self.k_class,self.k_bbox,self.k_giou,self.k_bbox1,self.k_giou1=k_class,k_bbox,k_giou,k_bbox1,k_giou1
            self.costMatrix=self.calcu_costMatrix # not used
            return
        @torch.no_grad()
        def forward(self, outputs, targets):
            '''
            outputs:'pred_logits', 'pred_boxes','pred_string_logits'
            [B,N,C]
            tgt:['labels':1=lp, 'boxes':xyxy, 'boxes_1', 'LPs']
            '''
            bs, n_predict, C = self.calcu_costMatrix(outputs, targets)
            C = C.view(bs, n_predict, -1)
            indices=torch.argmin(C,1)
            return indices
            
        @torch.no_grad()
        def calcu_costMatrix(self, outputs, targets):
            bs,n_predict  = outputs["pred_logits"].shape[:2]

            # We flatten to compute the cost matrices in a batch
            out_prob = outputs["pred_logits"].softmax(-1)  
            out_bbox = outputs["pred_boxes"][...,:4]
            out_bbox1=outputs['pred_boxes'][...,4:]

            # Also concat the target labels and boxes
            tgt_ids=targets["labels"]
            tgt_bbox=targets["boxes"]
            tgt_bbox1=targets["boxes_1"]

            # tgt_ids, tgt_bbox = self.tgt_List2Tensor(targets)
            device=tgt_bbox.device
            cost_class=torch.zeros(bs,n_predict,1,device=device)
            cost_bbox=torch.zeros(bs,n_predict,1,device=device)
            cost_giou=torch.zeros(bs,n_predict,1,device=device)
            cost_bbox1=torch.zeros(bs,n_predict,1,device=device)
            cost_giou1=torch.zeros(bs,n_predict,1,device=device)
            for i in range(bs):
                cost_class[i]=-out_prob[i,:,tgt_ids[i]]
                cost_bbox[i]=torch.cdist(out_bbox[i], tgt_bbox[i], p=1)
                cost_giou[i]=-ops.generalized_box_iou(box_cxcywh_to_xyxy(out_bbox[i]), (tgt_bbox[i]))
                cost_bbox1[i]=torch.cdist(out_bbox1[i], tgt_bbox1[i], p=1)
                cost_giou1[i]=-ops.generalized_box_iou(box_cxcywh_to_xyxy(out_bbox1[i]), (tgt_bbox1[i]))
                pass
            C =  self.k_class * cost_class +self.k_bbox * cost_bbox + self.k_giou * cost_giou+self.k_bbox1 * cost_bbox1 + self.k_giou1 * cost_giou1
            return bs,n_predict,C
        pass
    class single_4p_string_Matcher(single_4p_Matcher):
        @staticmethod
        def read_cfg(kwargs):
            return InferLoss_4p_string.read_cfg(kwargs)
        def __init__(self, kwargs):
            '''kwargs: from yml.criterion'''
            k_ce,k_l1,k_giou,n_class,k_l11,k_giou1,k_string,void_class_idx,void_class_weight= self.read_cfg(kwargs)
            # k_ce = 1, k_l1 = 5, k_giou = 2, k_l11 = 5, k_giou1 = 2,k_string=1
            super().__init__(k_ce, k_l1, k_giou, k_l11, k_giou1)
            self.k_string=k_string
            self.CEloss=nn.CrossEntropyLoss(reduction='none')
            return
        @torch.no_grad()
        def calcu_costMatrix(self, outputs, targets):
            '''
            outputs:'pred_logits', 'pred_boxes','pred_string_logits'
            [B,N,C]
            tgt:['labels':1=lp, 'boxes':xyxy, 'boxes_1', 'LPs']
            '''
            bs,n_predict,C=super().calcu_costMatrix(outputs, targets)
            in_string=outputs['pred_string_logits']
            tgt_string=targets['LPs'].unsqueeze(1).repeat(1,n_predict,1)
            
            Cost_string=self.CEloss.forward(in_string,tgt_string).mean(dim=-1,keepdim=True)
            C=C+self.k_string*Cost_string
            return bs,n_predict,C
    pass


class InferLoss_base(nn.Module):
    '''pred output, targets, match index
    return loss'''
    def __init__(self,k_ce:float=1,k_l1:float=5,k_giou:float=2, n_class:int=2,void_class_idx:int=0,void_class_weight:float = 0.1):
        super().__init__()
        self.k_ce,self.k_l1,self.k_giou=k_ce,k_l1,k_giou

        class_weights = torch.ones(n_class)
        class_weights[void_class_idx] = void_class_weight 
        self.classLoss=ClassifyLoss.CEloss(weight=class_weights)
        self.L1_loss=BboxLoss.L1_loss()
        self.GIoUloss=BboxLoss.GIoULoss
        return
    @staticmethod
    def one_hot_expand(indices: torch.Tensor, n: int) -> torch.Tensor:

        indices = indices.squeeze(dim=-1)  # 从 [B, 1] → [B]

        return torch.nn.functional.one_hot(indices, num_classes=n)
    
    def genPredSet_class(self, pred_logits_shape, indexes, targets_labels):
        """indexes=[B,1],targets_label=[B,1].
        Void class index=0"""
        B, N, C = pred_logits_shape
        device=targets_labels.device
        predSet_labels = torch.zeros(B, N, dtype=int,device=device)
        batch_idces = torch.arange(B,device=device)
        predSet_labels[batch_idces,indexes[:, 0]]=targets_labels[:, 0]
        # for i in range(B):
        #     predSet_labels[i, indexes[i, 0]] = targets_labels[i, 0]
        #     pass
        # predSet_labels2=self.one_hot_expand(indexes,n=N)
        return predSet_labels  # B,N

    def forward(self,outputs,targets,indexes):
        r'''outputs:{'pred_logits'=[B,N,C],'pred_boxes'=[B,N,points]}
        targets:{'labels'=int[B,1],'boxes'=[B,1,points],'LPs'=[B,8]}
        indexes:[B,1]'''
        
        B, CEloss = self.calcu_classLoss(outputs, targets, indexes)

        batch_ids = torch.arange(B, device=outputs['pred_logits'].device)
        predBbox_nonVoid = outputs['pred_boxes'][batch_ids, indexes[:, 0]]  # shape [B, 4]
        tgt_bbox=targets['boxes']
        l1_loss, GiouLoss = self.cal_bboxLoss(predBbox_nonVoid, tgt_bbox)
        
        loss=self.k_ce*CEloss+self.k_l1*l1_loss+self.k_giou*GiouLoss
        
        return loss

    def cal_bboxLoss(self, predBbox_nonVoid, tgt_bbox):
        predBbox_nonVoid_xyxy=box_cxcywh_to_xyxy(predBbox_nonVoid).unsqueeze(1)#cxcywh->xyxy
        l1_loss=self.L1_loss.forward(predBbox_nonVoid_xyxy,tgt_bbox)
        GiouLoss=self.GIoUloss(predBbox_nonVoid_xyxy,tgt_bbox,reduction='mean')
        return l1_loss,GiouLoss

    def calcu_classLoss(self, outputs, targets, indexes):
        B,N,C = outputs['pred_logits'].shape
        device=indexes.device
        self.classLoss.to(device=device)
        predSet_labels=self.genPredSet_class([B,N,C],indexes,targets['labels']) # B,N
        CEloss=self.classLoss.forward(outputs['pred_logits'].permute(0,2,1),predSet_labels)
        return B,CEloss
class InferLoss_4p_string(InferLoss_base):
    def __init__(self, kwargs):
        '''kwargs: from yml.criterion'''
        k_ce, k_l1, k_giou, n_class, k_l11, k_giou1, k_string, void_class_idx, void_class_weight = self.read_cfg(kwargs)
        super().__init__(k_ce, k_l1, k_giou, n_class, void_class_idx, void_class_weight)
        self.k_l11,self.k_giou1,self.k_string=k_l11,k_giou1,k_string
        self.stringLoss=nn.CrossEntropyLoss()
        return
    @staticmethod
    def read_cfg(kwargs):
        try:
            kwargs = kwargs.infer_cfg
        except AttributeError:
            kwargs = {}
        k_ce = kwargs.get('k_ce', 1)
        k_l1 = kwargs.get('k_l1', 5)
        k_giou = kwargs.get('k_giou', 2)
        n_class = kwargs.get('n_class', 2)
        k_l11 = kwargs.get('k_l11', 5)
        k_giou1 = kwargs.get('k_giou1', 2)
        k_string = kwargs.get('k_string', 10)
        void_class_idx = kwargs.get('void_class_idx', 0)
        void_class_weight = kwargs.get('void_class_weight', 0.1)
        return k_ce,k_l1,k_giou,n_class,k_l11,k_giou1,k_string,void_class_idx,void_class_weight
    def forward(self, outputs, targets, indexes,details:bool=False):
        r'''outputs:{'pred_logits'=[B,N,C],'pred_boxes'=[B,N,8],'pred_string_logits'=[B,C,pred,len]}
        targets:{'labels'=int[B,1],'boxes'=[B,1,4],'boxes1'=[B,1,4],'LPs'=[B,8]}
        indexes:[B,1]'''

        B, CEloss = self.calcu_classLoss(outputs, targets, indexes)

        batch_ids = torch.arange(B, device=outputs['pred_logits'].device)
        predBbox_nonVoid = outputs['pred_boxes'][batch_ids, indexes[:, 0]]  # shape [B, 8]
        predbbox=predBbox_nonVoid[...,:4]
        predbbox1=predBbox_nonVoid[...,4:]
        tgt_bbox=targets['boxes']
        tgt_bbox1=targets['boxes_1']
        l1_loss, GiouLoss = self.cal_bboxLoss(predbbox, tgt_bbox)
        l11_loss, Giou1Loss = self.cal_bboxLoss(predbbox1, tgt_bbox1)
        # string loss
        predString=outputs['pred_string_logits'][batch_ids,:, indexes[:, 0]] 
        stringLoss=self.stringLoss.forward(predString,targets['LPs'])
        #
        loss = (
            self.k_ce * CEloss
            + self.k_l1 * l1_loss
            + self.k_giou * GiouLoss
            + self.k_l11 * l11_loss
            + self.k_giou1 * Giou1Loss
            + self.k_string * stringLoss
        )
        if details:
            loss_box = {
                "infer_loss": loss,
                "infer_CEloss": CEloss,
                "infer_l1_loss": l1_loss,
                "infer_GiouLoss": GiouLoss,
                "infer_l11_loss": l11_loss,
                "infer_Giou1Loss": Giou1Loss,
                "infer_stringLoss": stringLoss,
            }
            return loss_box
        return loss


class HungarianLoss(nn.Module):
    '''
    HungarianLoss=match, then calcu loss noVoid.
    '''
    def __init__(self, matcher=None, setLoss=None, args=None):
        super().__init__()
        if args is not None:
            # 初始化方式 2：从配置 args 构造 matcher 和 criterion
            self.matcher = Matcher.single_4p_string_Matcher(args)
            self.criterion = InferLoss_4p_string(args)
        elif matcher is not None and setLoss is not None:
            # 初始化方式 1：直接提供 matcher 和 loss
            self.matcher = matcher
            self.criterion = setLoss
        else:
            self.matcher = Matcher.single_4p_string_Matcher(None)
            self.criterion = InferLoss_4p_string(None)
            # raise ValueError("You must provide either (matcher and setLoss) or args.")
        return
    def forward(self,outputs,targets, details:bool=False):
        indexes = self.matcher.forward(outputs, targets)
        if details:
            loss_dict=self.criterion.forward(outputs,targets,indexes,details)
            return loss_dict
        loss=self.criterion.forward(outputs,targets,indexes)
        return loss
    pass

class DenoiseLoss(nn.Module):
    '''
    contact denoise calcu for class, bbox
    and autoregressive LP string
    '''
    def __init__(self, kwargs):
        '''kwargs: from yml'''
        try:
            kwargs = kwargs.CDN_cfg
        except:
            kwargs = {}

        # 从 kwargs 中读取参数，如果没有就使用默认值
        self.k_ce = kwargs.get('k_ce', 1)
        self.k_l1 = kwargs.get('k_l1', 5)
        self.k_giou = kwargs.get('k_giou', 2)
        self.k_l11 = kwargs.get('k_l11', 5)
        self.k_giou1 = kwargs.get('k_giou1', 2)
        self.k_string = kwargs.get('k_string', 10)

        super().__init__()
        self.classLoss=ClassifyLoss.CEloss()
        self.l1Loss=BboxLoss.L1_loss()
        self.giouLoss=BboxLoss.GIoULoss
        self.stringLoss=nn.CrossEntropyLoss()
        self.groundClass=torch.tensor([[1,0]],dtype=int)
        return

    def forward(self, outputs, targets,details:bool=False):
        stringLoss = self.stringLoss.forward(outputs["LP_dn_logits"], targets["LPs"])
        p_box_l1, p_box1_l1, p_box_giou, p_box1_giou = self.gen_bbox_loss(outputs, targets)

        class_logits = torch.stack(
            [outputs["pos_class_logit"], outputs["neg_class_logit"]], dim=-1
        )  # B,C,2
        B = class_logits.size(0)
        groundClassIdx = self.groundClass.to(class_logits.device).repeat(B, 1)
        classLoss = self.classLoss.forward(class_logits, groundClassIdx)

        # sum
        DN_loss = (
            self.k_ce * classLoss
            + self.k_l1 * p_box_l1
            + self.k_giou * p_box_giou
            + self.k_l11 * p_box1_l1
            + self.k_giou1 * p_box1_giou
            + self.k_string * stringLoss
        )
        if details:
            DN_loss,classLoss,p_box_l1,p_box_giou,p_box1_l1,p_box1_giou,stringLoss
            loss_dict = {
                'DN_loss': DN_loss,
                'DN_classLoss': classLoss,
                'DN_p_box_l1': p_box_l1,
                'DN_p_box_giou': p_box_giou,
                'DN_p_box1_l1': p_box1_l1,
                'DN_p_box1_giou': p_box1_giou,
                'DN_stringLoss': stringLoss,
            }
            return loss_dict
        return DN_loss

    def gen_bbox_loss(self, outputs, targets):
        p_box = box_cxcywh_to_xyxy(outputs["pos_bbox"][..., :4])  # xyxy
        p_box1 = box_cxcywh_to_xyxy(outputs["pos_bbox"][..., 4:])
        tgtbox = targets["boxes"].squeeze(1)
        tgtbox1 = targets["boxes_1"].squeeze(1)
        p_box_l1 = self.l1Loss.forward(p_box, tgtbox)
        p_box1_l1 = self.l1Loss.forward(p_box1, tgtbox1)
        p_box_giou = self.giouLoss(p_box, tgtbox, reduction="mean")
        p_box1_giou = self.giouLoss(p_box1, tgtbox1, reduction="mean")
        return p_box_l1,p_box1_l1,p_box_giou,p_box1_giou

    pass
class Infer_DenoiseLoss(nn.Module):
    '''combine HungarianLoss and denoise loss'''
    def __init__(self,args):
        '''args: from yml.criterion'''
        super().__init__()
        self.infer_loss=HungarianLoss(args=args)
        self.denoise_loss=DenoiseLoss(args)
        return
    def forward(self,outputs, targets, details:bool=False):
        if details:
            infer_loss_dict=self.infer_loss(outputs, targets,details)
            DN_loss_dict= self.denoise_loss(outputs, targets,details)
            sumLoss=infer_loss_dict['infer_loss']+DN_loss_dict['DN_loss']
            total_loss_dict = {
                'sumLoss': sumLoss,
                **infer_loss_dict,
                **DN_loss_dict,
            }
            return total_loss_dict
        infer_loss=self.infer_loss(outputs, targets)
        DN_loss= self.denoise_loss(outputs, targets)
        return DN_loss+infer_loss


def build_Hungarian_loss(args):
    """
    k_ce:float=1,k_l1:float=5,k_giou:float=2, n_class:int=2,void_class_idx:int=0,void_class_weight:float = 0.1
    """
    return HungarianLoss(
        Matcher.RecessionaryMatcher(args.k_ce, args.k_l1, args.k_giou),
        InferLoss_base(
            args.k_ce,
            args.k_l1,
            args.k_giou,
            args.n_class,
            args.void_class_idx,
            args.void_class_weight,
        ),
    )
def build_Hungarian_loss_4p_LPstring(args=None):
    if args==None:
        return HungarianLoss(Matcher.single_4p_string_Matcher(),InferLoss_4p_string())
    return HungarianLoss(Matcher.single_4p_string_Matcher(args),InferLoss_4p_string(args))

# evaluate functions here
class IoU_LPs_evaluator():
    '''eval LP string acc. bbox iou'''
    def __init__(self):
        self._reset()
        return

    def _reset(self):
        self.LP_matches=[]
        self.bbox_iou=[]
        self.bbox1_iou=[]
        return
    
    @torch.no_grad()
    def forward_batch(self,predicts,targets):
        '''
        predicts={
            "pred_logits": Tensor [B, pred, 2]
            "pred_boxes": Tensor [B, pred, 8]
            "pred_string_logits": Tensor [B, C, pred, L]
        }
        targets={'labels':1=lp, 'boxes':xyxy, 'boxes_1', 'LPs'}
        return {'LP_matches','bbox_iou','bbox1_iou'} #[B]
        '''

        logits, top_indices = self.find_index(predicts)  

        batch_indices = torch.arange(logits.size(0))

        # fetch data
        boxes_predicts = predicts['pred_boxes']    # [B, pred, 8]
        string_predicts=predicts['pred_string_logits']
        boxes = boxes_predicts[batch_indices, top_indices]  # [B, 8]
        string=string_predicts[batch_indices,:,top_indices,:]#B,C,L

        ious, ious1 = self.calcu_iou(targets, boxes)

        LP_matches = self.calcu_LP_match(targets, string)  

        
        self._update(ious, ious1, LP_matches)
        eval_result = {"LP_matches": LP_matches, "bbox_iou": ious, "bbox1_iou": ious1}
        return eval_result
    def statistic_Dataset(self,reset:bool=True,iou_threshold=0.7):
        '''statistic acc on the hole dataset
        return {'lp_acc','mean_iou','mean_iou1'}'''
        LP_matches=torch.cat(self.LP_matches)
        bbox_iou=torch.cat(self.bbox_iou)
        bbox1_iou=torch.cat(self.bbox1_iou)
        lp_acc = LP_matches.float().mean() 
        mean_iou,mean_iou1=bbox_iou.mean(),bbox1_iou.mean()

        tp = (bbox_iou >= iou_threshold).float() # true positive
        AP= tp.mean()  
        
        result={'lp_acc':lp_acc,'mean_iou':mean_iou,'mean_iou1':mean_iou1,'AP':AP }

        self._reset() if reset else None
        return result

    def _update(self, ious, ious1, LP_matches):
        self.LP_matches.append(LP_matches)
        self.bbox_iou.append(ious)
        self.bbox1_iou.append(ious1)

    def calcu_LP_match(self, targets, string):
        string_gt=targets['LPs']
        string_pred = string.argmax(dim=1)  # [B, N]

        # 判断每个样本的预测字符串是否完全等于 GT
        matches = (string_pred == string_gt).all(dim=1)
        return matches# [B]，每个元素是 True/False

    def calcu_iou(self, targets, boxes):
        box=box_cxcywh_to_xyxy(boxes[...,:4]) #B,4
        box1=box_cxcywh_to_xyxy(boxes[...,4:])

        box_gt,box1_gt=targets['boxes'].squeeze(1),targets['boxes_1'].squeeze(1)
        ious = ops.box_iou(box, box_gt).diag()
        ious1 = ops.box_iou(box1, box1_gt).diag()
        return ious,ious1

    def find_index(self, predicts):
        logits = predicts['pred_logits']  # [B, pred, 2]
        probs = F.softmax(logits, dim=-1)  # [B, pred, 2]
        confidences = probs[:, :, 1]  # [B, pred]
        top_indices = confidences.argmax(dim=1)
        return logits,top_indices# [B]
