import torch
from torch import nn

# import torchvision.models
from torchvision.models import resnet50, resnet18
from .basic_block import Neck
from .basic_block import MLP
import models_detr.transformer

class DETR_my(nn.Module):
    """TODO: remove this"""

    def __init__(
        self,
        num_classes=2,
        hidden_dim=256,
        nheads=8,
        num_encoder_layers=2,
        num_decoder_layers=2,
        n_query=100,
    ):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet18(pretrained=True)
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(512, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers
        )

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(n_query, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))

    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)

        # construct positional encodings
        B, C, H, W = h.shape
        pos = (
            torch.cat(
                [
                    self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                    self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
                ],
                dim=-1,
            )
            .flatten(0, 1)
            .unsqueeze(1)
        )

        # propagate through the transformer
        h = self.transformer(
            pos + 0.1 * h.flatten(2).permute(2, 0, 1),
            self.query_pos.unsqueeze(1).repeat(1, B, 1),
        ).transpose(0, 1)

        # finally project transformer outputs to class labels and bounding boxes
        return {
            "pred_logits": self.linear_class(h),
            "pred_boxes": self.linear_bbox(h).sigmoid(),
        }

    pass


from util.misc import nested_tensor_from_tensor_list, NestedTensor


class Encoder:
    """input dict->src,pos,mask"""
    @staticmethod
    def unfreeze_model(model):
        for param in model.parameters():
            param.requires_grad_(True)
            pass
        return
    @staticmethod
    def replace_backbone(model, d_model=256, name_backbone="resnet18",unfreeze:bool=False):
        backbone_res = models_detr.backbone.Backbone(
            name_backbone,
            train_backbone=True,
            return_interm_layers=False,
            dilation=False,
        )
        if unfreeze:
            Encoder.unfreeze_model(backbone_res)
        model.backbone[0] = backbone_res
        model.input_proj = nn.Conv2d(
            backbone_res.num_channels, d_model, kernel_size=1
        )
        return

    @staticmethod
    def replace_backbone_layer3(model, d_model=256, name_backbone="resnet18",unfreeze:bool=False):
        backbone_res = models_detr.backbone.Backbone(
            name_backbone,
            train_backbone=True,
            return_interm_layers="layer3",
            dilation=False,
        )
        if unfreeze:
            Encoder.unfreeze_model(backbone_res)
        model.backbone[0] = backbone_res
        model.input_proj = nn.Conv2d(
            backbone_res.num_channels // 2, d_model, kernel_size=1
        )
        return

    class DETR_base(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            model = torch.hub.load(
                "facebookresearch/detr:main", "detr_resnet50", pretrained=True
            )
            self.backbone = model.backbone
            self.encoder = model.transformer.encoder
            self.input_proj = model.input_proj
            return

        def forward(self, samples: NestedTensor):
            if isinstance(samples, (list, torch.Tensor)):
                samples = nested_tensor_from_tensor_list(samples)
            features, pos = self.backbone(samples)
            src, mask = features[-1].decompose()
            pos_embed = pos[-1]
            # Flatten
            src = self.input_proj(src)
            src = src.flatten(2).permute(2, 0, 1)  # [HW, B, C]
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            mask = mask.flatten(1)
            # Encode
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
            return {"memory": memory, "mask": mask, "en_pos": pos_embed}

        pass

    class DETR_res18(DETR_base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.replace_backbone()
            return

        def replace_backbone(self, d_model=256, name_backbone="resnet18"):
            backbone_res18 = models_detr.backbone.Backbone(
                name_backbone,
                train_backbone=True,
                return_interm_layers=False,
                dilation=False,
            )
            self.backbone[0] = backbone_res18
            self.input_proj = nn.Conv2d(
                backbone_res18.num_channels, d_model, kernel_size=1
            )
            return
        pass
    @staticmethod
    def build_2layer_TRen(d_model=256,nhead=8):
        enLayer=models_detr.transformer.TransformerEncoderLayer(d_model=d_model,nhead=nhead)
        encoder=models_detr.transformer.TransformerEncoder(enLayer,num_layers=2)
        return encoder
    class DETR_2en_res18(DETR_base):
        def __init__(self, ):
            super().__init__()
            enLayer=models_detr.transformer.TransformerEncoderLayer(d_model=256,nhead=8)
            encoder=models_detr.transformer.TransformerEncoder(enLayer,num_layers=2)
            self.encoder=encoder
            Encoder.replace_backbone(self)
            # Encoder.unfreeze_model(self.backbone[0])
            return
        pass

    class DETR_res18_fm16(DETR_base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.replace_backbone_layer3
            return

        def replace_backbone_layer3(self, d_model=256, name_backbone="resnet18"):
            backbone_res18 = models_detr.backbone.Backbone(
                name_backbone,
                train_backbone=True,
                return_interm_layers="layer3",
                dilation=False,
            )
            self.backbone[0] = backbone_res18
            self.input_proj = nn.Conv2d(
                backbone_res18.num_channels // 2, d_model, kernel_size=1
            )
            return

    class DETR_STN_res18_fm16(nn.Module):
        """STN both fm and pos embedding"""

        def __init__(
            self,
        ):
            super().__init__()
            model = torch.hub.load(
                "facebookresearch/detr:main", "detr_resnet50", pretrained=True
            )
            self.backbone = model.backbone
            self.encoder = model.transformer.encoder
            self.input_proj = model.input_proj
            self.STN = Neck.STN_s16g270()
            CiT_CDN_res18_fm16.replace_backbone(self)
            return

        def forward(self, samples: NestedTensor):
            if isinstance(samples, (list, torch.Tensor)):
                samples = nested_tensor_from_tensor_list(samples)
            features, pos = self.backbone(samples)
            src, mask = features[-1].decompose()
            pos_embed = pos[-1]

            src, pos_embed, mask = self.STN.forward_img_pos_mask(src, pos_embed, mask)
            # Flatten
            src = self.input_proj(src)
            src = src.flatten(2).permute(2, 0, 1)  # [HW, B, C]
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
            mask = mask.flatten(1)
            # Encode
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
            return {"memory": memory, "mask": mask, "en_pos": pos_embed}

        pass

    class DETR_STN_res18_fm32(nn.Module):
        def __init__(
            self,
        ):
            super().__init__()
            model = torch.hub.load(
                "facebookresearch/detr:main", "detr_resnet50", pretrained=True
            )
            self.backbone = model.backbone
            self.encoder = model.transformer.encoder
            self.input_proj = model.input_proj
            Encoder.replace_backbone(self)
            self.STN = Neck.STN_s32g276()
            return

        def forward(self, samples: NestedTensor):
            return Encoder.DETR_STN_res18_fm16.forward(self, samples)
    class DETR_2en_STN_res50_fm32(nn.Module):
        '''unfreeze res50 backbone'''
        def __init__(
            self,
        ):
            super().__init__()
            model = torch.hub.load(
                "facebookresearch/detr:main", "detr_resnet50", pretrained=True
            )
            self.backbone = model.backbone
            Encoder.unfreeze_model(self.backbone[0])
            self.input_proj = model.input_proj
            self.STN = Neck.STN_s32g276(d_fm=2048)
            self.encoder = Encoder.build_2layer_TRen()
            return

        def forward(self, samples: NestedTensor):
            return Encoder.DETR_STN_res18_fm16.forward(self, samples)
        pass
    class DETR_2en_STN_res18_fm32(nn.Module):
        '''unfreeze res50 backbone'''
        def __init__(
            self,
        ):
            super().__init__()
            model = torch.hub.load(
                "facebookresearch/detr:main", "detr_resnet50", pretrained=True
            )
            self.backbone = model.backbone
            self.input_proj = None
            Encoder.replace_backbone(self,unfreeze=True)
            self.STN = Neck.STN_s32g276()
            self.encoder = Encoder.build_2layer_TRen()
            return

        def forward(self, samples: NestedTensor):
            return Encoder.DETR_STN_res18_fm16.forward(self, samples)
        pass

pass




# decoders
class Decoder:
    """src,pos,mask->predict dict"""

    class CiT(nn.Module):
        """
        version2. 10 pos query, 1 content query
        4 point vertex bbox and LP string predicte
        """

        def __init__(
            self,
            num_predict=10,
            Lgroup=9,
            nlayers=2,
            nhead=8,
            d_ffn=512,
            d_model=256,
            n_class=2,
            n_bbox_vertex=4,
            n_character=75,
        ):
            """Lgroup: how many tokens each prediction."""
            super().__init__()
            self.num_predict, self.Lgroup, self.n_character, self.n_class = (
                num_predict,
                Lgroup,
                n_character,
                n_class,
            )
            self.tgt_infer = nn.Embedding(Lgroup, d_model)
            self.de_pos_infer = nn.Embedding(num_predict * Lgroup, d_model)
            delayer = models_detr.transformer.TransformerDecoderLayer(
                d_model, nhead, d_ffn
            )
            self.TRdecoder = models_detr.transformer.TransformerDecoder(
                delayer, nlayers
            )
            self.class_proj = nn.Linear(d_model, n_class)
            self.bbox_proj = MLP(d_model, d_model, n_bbox_vertex * 2, 3)
            self.character_proj = nn.Linear(d_model, n_character)
            return

        def forward(self, en_outputs: dict):
            # =super().forward(imgs)
            """en_outputs: dict{src,pos,mask}"""
            memory, mask, en_pos = self.read_memory(en_outputs)
            _, B, C = memory.shape
            tgt_infer, pos_infer = self.prepare_query(B)
            output = self.TRdecoder.forward(
                tgt_infer,
                memory,
                memory_key_padding_mask=mask,
                pos=en_pos,
                query_pos=pos_infer,
            )
            output = output.squeeze(0)
            # seperate tokens
            out = self.seg_prediction(B, output)
            return out

        def seg_prediction(self, B, output):
            seq_len = output.size(0)
            mask = torch.ones(seq_len, dtype=torch.bool, device=output.device)
            mask[:: self.Lgroup] = False  # this all tokens_detection
            tokens_detection = output[~mask, ...]
            tokens_recognition = output[mask, ...]
            outputs_class = self.class_proj(tokens_detection).permute(1, 0, 2)
            outputs_coord = self.bbox_proj(tokens_detection).sigmoid().permute(1, 0, 2)
            outputs_string = self.character_proj(tokens_recognition)
            outputs_string = outputs_string.view(
                self.num_predict, self.Lgroup - 1, B, self.n_character
            ).permute(
                2, 3, 0, 1
            )  # [B,C,pred,len]:[2, 75, 10, 8]
            out = {
                "pred_logits": outputs_class,
                "pred_boxes": outputs_coord,
                "pred_string_logits": outputs_string,
            }
            return out

        def prepare_query(self, B):
            tgt_infer = self.tgt_infer.weight.unsqueeze(1).repeat(
                self.num_predict, B, 1
            )
            pos_infer = self.de_pos_infer.weight.unsqueeze(1).repeat(1, B, 1)
            return tgt_infer, pos_infer  # [B,N,C]

        def read_memory(self, en_outputs):
            memory = en_outputs["memory"]
            mask = en_outputs["mask"]
            en_pos = en_outputs["en_pos"]
            return memory, mask, en_pos

        pass

    class CiT_v1(nn.Module):
        """
        4 point vertex bbox and LP string predicte
        """

        def __init__(
            self,
            num_predict=10,
            Lgroup=9,
            nlayers=2,
            nhead=8,
            d_ffn=512,
            d_model=256,
            n_class=2,
            n_bbox_vertex=4,
            n_character=75,
        ):
            """Lgroup: how many tokens each prediction."""
            super().__init__()
            self.num_predict, self.Lgroup, self.n_character,self.n_class = (
                num_predict,
                Lgroup,
                n_character,
                n_class,
            )
            self.tgt_infer = nn.Embedding(num_predict * Lgroup, d_model)
            self.de_pos_infer = nn.Embedding(Lgroup, d_model)
            delayer = models_detr.transformer.TransformerDecoderLayer(
                d_model, nhead, d_ffn
            )
            self.TRdecoder = models_detr.transformer.TransformerDecoder(
                delayer, nlayers
            )
            self.class_proj = nn.Linear(d_model, n_class)
            self.bbox_proj = MLP(d_model, d_model, n_bbox_vertex * 2, 3)
            self.character_proj = nn.Linear(d_model, n_character)
            return

        def forward(self, en_outputs: dict):
            # =super().forward(imgs)
            """en_outputs: dict{src,pos,mask}"""
            memory, mask, en_pos = self.read_memory(en_outputs)
            _, B, C = memory.shape
            tgt_infer, pos_infer = self.prepare_query(B)
            output = self.TRdecoder.forward(
                tgt_infer,
                memory,
                memory_key_padding_mask=mask,
                pos=en_pos,
                query_pos=pos_infer,
            )
            output = output.squeeze(0)
            # seperate tokens
            out = self.seg_prediction(B, output)
            return out

        def seg_prediction(self, B, output):
            seq_len = output.size(0)
            mask = torch.ones(seq_len, dtype=torch.bool, device=output.device)
            mask[:: self.Lgroup] = False  # this all tokens_detection
            tokens_detection = output[~mask, ...]
            tokens_recognition = output[mask, ...]
            outputs_class = self.class_proj(tokens_detection).permute(1, 0, 2)
            outputs_coord = self.bbox_proj(tokens_detection).sigmoid().permute(1, 0, 2)
            outputs_string = self.character_proj(tokens_recognition)
            outputs_string = outputs_string.view(
                self.num_predict, self.Lgroup - 1, B, self.n_character
            ).permute(
                2, 3, 0, 1
            )  # [B,C,pred,len]:[2, 75, 10, 8]
            out = {
                "pred_logits": outputs_class,
                "pred_boxes": outputs_coord,
                "pred_string_logits": outputs_string,
            }
            return out

        def prepare_query(self, B):
            tgt_infer = self.tgt_infer.weight.unsqueeze(1).repeat(1, B, 1)
            pos_infer = self.de_pos_infer.weight.unsqueeze(1).repeat(
                self.num_predict, B, 1
            )
            return tgt_infer, pos_infer  # [B,N,C]

        def read_memory(self, en_outputs):
            memory = en_outputs["memory"]
            mask = en_outputs["mask"]
            en_pos = en_outputs["en_pos"]
            return memory, mask, en_pos

        pass
    class CiT_CNDv3(nn.Module):
        """version3. cdn content query=[class,LP] pos=[bbox,mean_infer_pos]"""

        def __init__(
            self, decoder: nn.Module, d_model=256, n_bbox_vertex=4, n_character=75
        ):
            """decoder: CiT infer decoder"""
            super().__init__()
            self.decoder:Decoder.CiT = decoder
            self.d_model = d_model
            self.classEmbed = nn.Embedding(self.decoder.n_class, d_model)
            self.lps_embed = nn.Embedding(n_character, d_model)
            self.bbox_embed = MLP(n_bbox_vertex * 2, d_model // 2, d_model, 3)
            self.register_buffer("tgt_mask", self.build_qk_mask())
            return

        def build_qk_mask(self):
            """build self attn qk mask = [a,b;c,d]"""
            L_infer: int = self.decoder.num_predict * self.decoder.Lgroup
            L_denoise: int = 2 * self.decoder.Lgroup
            # Part a: 90x90 all zeros
            a = torch.zeros((L_infer, L_infer))

            # Part b: 90x18 all -inf
            b = torch.full((L_infer, L_denoise), float("-inf"))

            # Part c: 18x90 all zeros
            c = torch.zeros((L_denoise, L_infer))

            # Part d: 18x18 composed of two 9x9 causal masks
            def causal_mask(size):
                return torch.triu(torch.full((size, size), float("-inf")), diagonal=1)

            d = torch.full((L_denoise, L_denoise), float("-inf"))
            d[: self.decoder.Lgroup, : self.decoder.Lgroup] = causal_mask(
                self.decoder.Lgroup
            )
            d[self.decoder.Lgroup :, self.decoder.Lgroup :] = causal_mask(
                self.decoder.Lgroup
            )

            # Assemble the full mask
            top = torch.cat([a, b], dim=1)  # (90, 108)
            bottom = torch.cat([c, d], dim=1)  # (18, 108)
            tgt_mask = torch.cat([top, bottom], dim=0)  # (108, 108)
            return tgt_mask

        def forward(self, inputs: dict, denoise: bool = True):
            """
            inputs: {
                'en_outputs': dict{src,pos,mask}
                'pos_bbox': Tensor [B, 8]           # xywhxywh
                'neg_bbox': Tensor [B, 8]           # xywhxywh
                'LPs_delay': Tensor [B, 8]
            }

            return: {
                "pred_logits": Tensor [B,pred, 2]
                "pred_boxes": Tensor [B,pred, 8] #xywh xywh
                "pred_string_logits": Tensor [B, C, pred, L]

                "pos_class_logit": Tensor [B, 2]
                "neg_class_logit": Tensor [B, 2]
                "pos_bbox": Tensor [B, 8]
                "neg_bbox": Tensor [B, 8]
                "LP_dn_logits": Tensor [B, C, N]
            }
            """

            # imgs=inputs['imgs']
            en_outputs: dict = inputs["en_outputs"]
            if not denoise:
                return super().forward(en_outputs)

            memory, mask, en_pos = self.decoder.read_memory(en_outputs)
            _, B, C = memory.shape

            tgt_infer_CDN, pos_infer_CDN, tgt_mask = self.prepare_query_denoise(
                B, inputs
            )
            output = self.decoder.TRdecoder.forward(
                tgt_infer_CDN,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=mask,
                pos=en_pos,
                query_pos=pos_infer_CDN,
            )
            output = output.squeeze(0)

            # segment predicted tokens
            out = self.seg_prediction_denoise(B, output)

            return out

        def seg_prediction_denoise(self, B, output):
            out_infer = self.decoder.seg_prediction(
                B, output[: self.decoder.num_predict * self.decoder.Lgroup]
            )
            tokens_denoise = output[self.decoder.num_predict * self.decoder.Lgroup :]
            pos_neg_bbox_token = tokens_denoise[[0, self.decoder.Lgroup]]

            pos_neg_class_logits = self.decoder.class_proj.forward(pos_neg_bbox_token)
            pos_neg_bbox = self.decoder.bbox_proj.forward(pos_neg_bbox_token).sigmoid()
            LP_dn = self.decoder.character_proj.forward(
                tokens_denoise[1 : self.decoder.Lgroup]
            )  # N,B,C
            pos_class_logit, neg_class_logit = pos_neg_class_logits.unbind()
            pos_bbox, neg_bbox = pos_neg_bbox.unbind()
            LP_dn_logits = LP_dn.permute(1, 2, 0)  # B,C,N
            out_denoise = {
                "pos_class_logit": pos_class_logit,  # B,2
                "neg_class_logit": neg_class_logit,  # B,2
                "pos_bbox": pos_bbox,  # B,8
                "neg_bbox": neg_bbox,  # B,8
                "LP_dn_logits": LP_dn_logits,  # B,C,N
            }
            out = {**out_infer, **out_denoise}
            return out

        def prepare_query_denoise(self, B, inputs: dict):
            """
            inputs:
                'imgs'      : [B, 3, H, W]
                'pos_bbox'  : [B, 8]  # xywhxywh
                'neg_bbox'  : [B, 8]
                'LPs_delay' : [B, 8]  # IDs
            """
            tgt_infer, pos_infer = self.decoder.prepare_query(B)

            # Embed LPs_delay
            lps_delay_ids = inputs["LPs_delay"].permute(1, 0)  # [B, 8]
            lps_embed = self.lps_embed(lps_delay_ids)  # [B, 8, 256]

            # Project pos and neg bbox
            pos_bbox = inputs["pos_bbox"].permute(1,0,2)# [1,B,8]
            neg_bbox = inputs["neg_bbox"].permute(1,0,2)# [1,B,8]
            p_pos_embed_1=self.bbox_embed(pos_bbox) # [1,B,256]
            n_pos_embed_1=self.bbox_embed(neg_bbox)
            pos_infer_LP=pos_infer.view(self.decoder.num_predict,self.decoder.Lgroup,B,self.d_model)# 10,9,B,256
            mean_pos_infer_LP=pos_infer_LP.mean(0)# 8,B,256
            p_pos_embed=n_pos_embed=mean_pos_infer_LP
            p_pos_embed[0]+=p_pos_embed_1[0]
            n_pos_embed[0]+=n_pos_embed_1[0]
            # content query
            class_ids = torch.tensor([1, 0],device=tgt_infer.device).unsqueeze(1)
            class_embed = self.classEmbed(class_ids) # 2,1,C
            class_embed=class_embed.repeat(1,B,1)
            tgt = torch.cat([tgt_infer, class_embed[[0]], lps_embed, class_embed[[1]], lps_embed], dim=0)
            pos = torch.cat([pos_infer, p_pos_embed,n_pos_embed], dim=0)

            tgt_mask = self.tgt_mask

            return tgt, pos, tgt_mask

    class CiT_CNDv2(nn.Module):
        """version2. cdn content query=[class,LP] pos=[bbox,bbox+Δ]"""

        def __init__(
            self, decoder: nn.Module, d_model=256, n_bbox_vertex=4, n_character=75
        ):
            """decoder: CiT infer decoder"""
            super().__init__()
            self.decoder:Decoder.CiT = decoder
            self.d_model = d_model
            self.classEmbed = nn.Embedding(self.decoder.n_class, d_model)
            self.lps_embed = nn.Embedding(n_character, d_model)
            self.bbox_embed = MLP(n_bbox_vertex * 2, d_model // 2, d_model, 3)
            self.deltas = nn.Parameter(
                torch.zeros((self.decoder.Lgroup, n_bbox_vertex * 2))
            )  # bbox+Δ
            self.register_buffer("tgt_mask", self.build_qk_mask())
            return

        def build_qk_mask(self):
            """build self attn qk mask = [a,b;c,d]"""
            L_infer: int = self.decoder.num_predict * self.decoder.Lgroup
            L_denoise: int = 2 * self.decoder.Lgroup
            # Part a: 90x90 all zeros
            a = torch.zeros((L_infer, L_infer))

            # Part b: 90x18 all -inf
            b = torch.full((L_infer, L_denoise), float("-inf"))

            # Part c: 18x90 all zeros
            c = torch.zeros((L_denoise, L_infer))

            # Part d: 18x18 composed of two 9x9 causal masks
            def causal_mask(size):
                return torch.triu(torch.full((size, size), float("-inf")), diagonal=1)

            d = torch.full((L_denoise, L_denoise), float("-inf"))
            d[: self.decoder.Lgroup, : self.decoder.Lgroup] = causal_mask(
                self.decoder.Lgroup
            )
            d[self.decoder.Lgroup :, self.decoder.Lgroup :] = causal_mask(
                self.decoder.Lgroup
            )

            # Assemble the full mask
            top = torch.cat([a, b], dim=1)  # (90, 108)
            bottom = torch.cat([c, d], dim=1)  # (18, 108)
            tgt_mask = torch.cat([top, bottom], dim=0)  # (108, 108)
            return tgt_mask

        def forward(self, inputs: dict, denoise: bool = True):
            """
            inputs: {
                'en_outputs': dict{src,pos,mask}
                'pos_bbox': Tensor [B, 8]           # xywhxywh
                'neg_bbox': Tensor [B, 8]           # xywhxywh
                'LPs_delay': Tensor [B, 8]
            }

            return: {
                "pred_logits": Tensor [B,pred, 2]
                "pred_boxes": Tensor [B,pred, 8] #xywh xywh
                "pred_string_logits": Tensor [B, C, pred, L]

                "pos_class_logit": Tensor [B, 2]
                "neg_class_logit": Tensor [B, 2]
                "pos_bbox": Tensor [B, 8]
                "neg_bbox": Tensor [B, 8]
                "LP_dn_logits": Tensor [B, C, N]
            }
            """

            # imgs=inputs['imgs']
            en_outputs: dict = inputs["en_outputs"]
            if not denoise:
                return super().forward(en_outputs)

            memory, mask, en_pos = self.decoder.read_memory(en_outputs)
            _, B, C = memory.shape

            tgt_infer_CDN, pos_infer_CDN, tgt_mask = self.prepare_query_denoise(
                B, inputs
            )
            output = self.decoder.TRdecoder.forward(
                tgt_infer_CDN,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=mask,
                pos=en_pos,
                query_pos=pos_infer_CDN,
            )
            output = output.squeeze(0)

            # segment predicted tokens
            out = self.seg_prediction_denoise(B, output)

            return out

        def seg_prediction_denoise(self, B, output):
            out_infer = self.decoder.seg_prediction(
                B, output[: self.decoder.num_predict * self.decoder.Lgroup]
            )
            tokens_denoise = output[self.decoder.num_predict * self.decoder.Lgroup :]
            pos_neg_bbox_token = tokens_denoise[[0, self.decoder.Lgroup]]

            pos_neg_class_logits = self.decoder.class_proj.forward(pos_neg_bbox_token)
            pos_neg_bbox = self.decoder.bbox_proj.forward(pos_neg_bbox_token).sigmoid()
            LP_dn = self.decoder.character_proj.forward(
                tokens_denoise[1 : self.decoder.Lgroup]
            )  # N,B,C
            pos_class_logit, neg_class_logit = pos_neg_class_logits.unbind()
            pos_bbox, neg_bbox = pos_neg_bbox.unbind()
            LP_dn_logits = LP_dn.permute(1, 2, 0)  # B,C,N
            out_denoise = {
                "pos_class_logit": pos_class_logit,  # B,2
                "neg_class_logit": neg_class_logit,  # B,2
                "pos_bbox": pos_bbox,  # B,8
                "neg_bbox": neg_bbox,  # B,8
                "LP_dn_logits": LP_dn_logits,  # B,C,N
            }
            out = {**out_infer, **out_denoise}
            return out

        def prepare_query_denoise(self, B, inputs: dict):
            """
            inputs:
                'imgs'      : [B, 3, H, W]
                'pos_bbox'  : [B, 8]  # xywhxywh
                'neg_bbox'  : [B, 8]
                'LPs_delay' : [B, 8]  # IDs
            """
            tgt_infer, pos_infer = self.decoder.prepare_query(B)

            # Embed LPs_delay
            lps_delay_ids = inputs["LPs_delay"].permute(1, 0)  # [B, 8]
            lps_embed = self.lps_embed(lps_delay_ids)  # [B, 8, 256]

            # Project pos and neg bbox
            pos_bbox = inputs["pos_bbox"].permute(1,0,2).repeat(self.decoder.Lgroup,1,1)# [L,B,8]
            neg_bbox = inputs["neg_bbox"].permute(1,0,2).repeat(self.decoder.Lgroup,1,1)# [L,B,8]
            pos_bboxes=pos_bbox+self.deltas.unsqueeze(1)
            neg_bboxes=neg_bbox+self.deltas.unsqueeze(1)
            p_pos_embed=self.bbox_embed(pos_bboxes)
            n_pos_embed=self.bbox_embed(neg_bboxes)

            # content query
            class_ids = torch.tensor([1, 0],device=tgt_infer.device).unsqueeze(1)
            class_embed = self.classEmbed(class_ids) # 2,1,C
            class_embed=class_embed.repeat(1,B,1)
            tgt = torch.cat([tgt_infer, class_embed[[0]], lps_embed, class_embed[[1]], lps_embed], dim=0)
            pos = torch.cat([pos_infer, p_pos_embed,n_pos_embed], dim=0)

            tgt_mask = self.tgt_mask

            return tgt, pos, tgt_mask
    class CiT_CNDv1_1(nn.Module):
        '''content q=[class,LP]
        pos q=[pos[0]+bbox,pos[1:]]'''
        def __init__(
            self, decoder: nn.Module, d_model=256, n_bbox_vertex=4, n_character=75
        ):
            # num_predict=10, Lgroup=9, nlayers=2, nhead=8, d_ffn=512,n_class=2,
            super().__init__()
            self.decoder = decoder
            self.d_model:Decoder.CiT_v1 = d_model
            self.classEmbed = nn.Embedding(self.decoder.n_class, d_model)
            self.lps_embed = nn.Embedding(n_character, d_model)
            self.bbox_embed = MLP(n_bbox_vertex * 2, d_model // 2, d_model, 3)
            self.register_buffer("tgt_mask", self.build_qk_mask())
            return

        def build_qk_mask(self):
            """build self attn qk mask = [a,b;c,d]"""
            L_infer: int = self.decoder.num_predict * self.decoder.Lgroup
            L_denoise: int = 2 * self.decoder.Lgroup
            # Part a: 90x90 all zeros
            a = torch.zeros((L_infer, L_infer))

            # Part b: 90x18 all -inf
            b = torch.full((L_infer, L_denoise), float("-inf"))

            # Part c: 18x90 all zeros
            c = torch.zeros((L_denoise, L_infer))

            # Part d: 18x18 composed of two 9x9 causal masks
            def causal_mask(size):
                return torch.triu(torch.full((size, size), float("-inf")), diagonal=1)

            d = torch.full((L_denoise, L_denoise), float("-inf"))
            d[: self.decoder.Lgroup, : self.decoder.Lgroup] = causal_mask(
                self.decoder.Lgroup
            )
            d[self.decoder.Lgroup :, self.decoder.Lgroup :] = causal_mask(
                self.decoder.Lgroup
            )

            # Assemble the full mask
            top = torch.cat([a, b], dim=1)  # (90, 108)
            bottom = torch.cat([c, d], dim=1)  # (18, 108)
            tgt_mask = torch.cat([top, bottom], dim=0)  # (108, 108)
            return tgt_mask

        def forward(self, inputs: dict, denoise: bool = True):
            """
            inputs: {
                'en_outputs': dict{src,pos,mask}
                'pos_bbox': Tensor of shape [B, 8]           # xywhxywh
                'neg_bbox': Tensor of shape [B, 8]           # xywhxywh
                'LPs_delay': Tensor of shape [B, 8]
            }

            return: {
                "pred_logits": Tensor of shape [B,pred, 2]
                "pred_boxes": Tensor of shape [B,pred, 8] #xywh xywh
                "pred_string_logits": Tensor of shape [B, C, pred, L]

                "pos_class_logit": Tensor of shape [B, 2]
                "neg_class_logit": Tensor of shape [B, 2]
                "pos_bbox": Tensor of shape [B, 8]
                "neg_bbox": Tensor of shape [B, 8]
                "LP_dn_logits": Tensor of shape [B, C, N]
            }
            """

            # imgs=inputs['imgs']
            en_outputs: dict = inputs["en_outputs"]
            if not denoise:
                return self.decoder.forward(en_outputs)

            memory, mask, en_pos = self.decoder.read_memory(en_outputs)
            _, B, C = memory.shape

            tgt_infer_CDN, pos_infer_CDN, tgt_mask = self.prepare_query_denoise(
                B, inputs
            )
            output = self.decoder.TRdecoder.forward(
                tgt_infer_CDN,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=mask,
                pos=en_pos,
                query_pos=pos_infer_CDN,
            )
            output = output.squeeze(0)

            # segment predicted tokens
            out = self.seg_prediction_denoise(B, output)

            return out

        def seg_prediction_denoise(self, B, output):
            out_infer = self.decoder.seg_prediction(
                B, output[: self.decoder.num_predict * self.decoder.Lgroup]
            )
            tokens_denoise = output[self.decoder.num_predict * self.decoder.Lgroup :]
            pos_neg_bbox_token = tokens_denoise[[0, self.decoder.Lgroup]]

            pos_neg_class_logits = self.decoder.class_proj.forward(pos_neg_bbox_token)
            pos_neg_bbox = self.decoder.bbox_proj.forward(pos_neg_bbox_token).sigmoid()
            LP_dn = self.decoder.character_proj.forward(
                tokens_denoise[1 : self.decoder.Lgroup]
            )  # N,B,C
            pos_class_logit, neg_class_logit = pos_neg_class_logits.unbind()
            pos_bbox, neg_bbox = pos_neg_bbox.unbind()
            LP_dn_logits = LP_dn.permute(1, 2, 0)  # B,C,N
            out_denoise = {
                "pos_class_logit": pos_class_logit,  # B,2
                "neg_class_logit": neg_class_logit,  # B,2
                "pos_bbox": pos_bbox,  # B,8
                "neg_bbox": neg_bbox,  # B,8
                "LP_dn_logits": LP_dn_logits,  # B,C,N
            }
            out = {**out_infer, **out_denoise}
            return out

        def prepare_query_denoise(self, B, inputs: dict):
            """
            inputs:
                'imgs'      : [B, 3, H, W]
                'pos_bbox'  : [B, 8]  # xywhxywh
                'neg_bbox'  : [B, 8]
                'LPs_delay' : [B, 8]  # IDs
            """
            tgt_infer, pos_infer = self.decoder.prepare_query(B)

            # Project pos and neg bbox
            pos_bbox = inputs["pos_bbox"].squeeze(1)  # [B, 8]
            neg_bbox = inputs["neg_bbox"].squeeze(1)  # [B, 8]
            pos_bbox_embed = self.bbox_embed(pos_bbox)  # [B, 256]
            neg_bbox_embed = self.bbox_embed(neg_bbox)  # [B, 256]

            noise_pos = self.decoder.de_pos_infer.weight.unsqueeze(1).repeat(2, B, 1)
            noise_pos[0] += pos_bbox_embed
            noise_pos[self.decoder.Lgroup] += neg_bbox_embed
            

            # content query
            # Embed LPs_delay
            lps_delay_ids = inputs["LPs_delay"].permute(1, 0)  # [B, 8]
            lps_embed = self.lps_embed(lps_delay_ids)  # [B, 8, 256]
            class_ids = torch.tensor([1, 0],device=tgt_infer.device).unsqueeze(1)
            class_embed = self.classEmbed(class_ids) # 2,1,C
            class_embed=class_embed.repeat(1,B,1)

            tgt = torch.cat([tgt_infer, class_embed[[0]], lps_embed, class_embed[[1]], lps_embed], dim=0)
            pos = torch.cat([pos_infer, noise_pos], dim=0)

            tgt_mask = self.tgt_mask

            return tgt, pos, tgt_mask

    class CiT_CND_v1(nn.Module):
        def __init__(
            self, decoder: nn.Module, d_model=256, n_bbox_vertex=4, n_character=75
        ):
            # num_predict=10, Lgroup=9, nlayers=2, nhead=8, d_ffn=512,n_class=2,
            super().__init__()
            self.decoder:Decoder.CiT_v1 = decoder
            self.d_model = d_model
            self.lps_embed = nn.Embedding(n_character, d_model)
            self.bbox_embed = MLP(n_bbox_vertex * 2, d_model // 2, d_model, 3)
            self.register_buffer("tgt_mask", self.build_qk_mask())
            return

        def build_qk_mask(self):
            """build self attn qk mask = [a,b;c,d]"""
            L_infer: int = self.decoder.num_predict * self.decoder.Lgroup
            L_denoise: int = 2 * self.decoder.Lgroup
            # Part a: 90x90 all zeros
            a = torch.zeros((L_infer, L_infer))

            # Part b: 90x18 all -inf
            b = torch.full((L_infer, L_denoise), float("-inf"))

            # Part c: 18x90 all zeros
            c = torch.zeros((L_denoise, L_infer))

            # Part d: 18x18 composed of two 9x9 causal masks
            def causal_mask(size):
                return torch.triu(torch.full((size, size), float("-inf")), diagonal=1)

            d = torch.full((L_denoise, L_denoise), float("-inf"))
            d[: self.decoder.Lgroup, : self.decoder.Lgroup] = causal_mask(
                self.decoder.Lgroup
            )
            d[self.decoder.Lgroup :, self.decoder.Lgroup :] = causal_mask(
                self.decoder.Lgroup
            )

            # Assemble the full mask
            top = torch.cat([a, b], dim=1)  # (90, 108)
            bottom = torch.cat([c, d], dim=1)  # (18, 108)
            tgt_mask = torch.cat([top, bottom], dim=0)  # (108, 108)
            return tgt_mask

        def forward(self, inputs: dict, denoise: bool = True):
            """
            inputs: {
                'en_outputs': dict{src,pos,mask}
                'pos_bbox': Tensor of shape [B, 8]           # xywhxywh
                'neg_bbox': Tensor of shape [B, 8]           # xywhxywh
                'LPs_delay': Tensor of shape [B, 8]
            }

            return: {
                "pred_logits": Tensor of shape [B,pred, 2]
                "pred_boxes": Tensor of shape [B,pred, 8] #xywh xywh
                "pred_string_logits": Tensor of shape [B, C, pred, L]

                "pos_class_logit": Tensor of shape [B, 2]
                "neg_class_logit": Tensor of shape [B, 2]
                "pos_bbox": Tensor of shape [B, 8]
                "neg_bbox": Tensor of shape [B, 8]
                "LP_dn_logits": Tensor of shape [B, C, N]
            }
            """

            # imgs=inputs['imgs']
            en_outputs: dict = inputs["en_outputs"]
            if not denoise:
                return super().forward(en_outputs)

            memory, mask, en_pos = self.decoder.read_memory(en_outputs)
            _, B, C = memory.shape

            tgt_infer_CDN, pos_infer_CDN, tgt_mask = self.prepare_query_denoise(
                B, inputs
            )
            output = self.decoder.TRdecoder.forward(
                tgt_infer_CDN,
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=mask,
                pos=en_pos,
                query_pos=pos_infer_CDN,
            )
            output = output.squeeze(0)

            # segment predicted tokens
            out = self.seg_prediction_denoise(B, output)

            return out

        def seg_prediction_denoise(self, B, output):
            out_infer = self.decoder.seg_prediction(
                B, output[: self.decoder.num_predict * self.decoder.Lgroup]
            )
            tokens_denoise = output[self.decoder.num_predict * self.decoder.Lgroup :]
            pos_neg_bbox_token = tokens_denoise[[0, self.decoder.Lgroup]]

            pos_neg_class_logits = self.decoder.class_proj.forward(pos_neg_bbox_token)
            pos_neg_bbox = self.decoder.bbox_proj.forward(pos_neg_bbox_token).sigmoid()
            LP_dn = self.decoder.character_proj.forward(
                tokens_denoise[1 : self.decoder.Lgroup]
            )  # N,B,C
            pos_class_logit, neg_class_logit = pos_neg_class_logits.unbind()
            pos_bbox, neg_bbox = pos_neg_bbox.unbind()
            LP_dn_logits = LP_dn.permute(1, 2, 0)  # B,C,N
            out_denoise = {
                "pos_class_logit": pos_class_logit,  # B,2
                "neg_class_logit": neg_class_logit,  # B,2
                "pos_bbox": pos_bbox,  # B,8
                "neg_bbox": neg_bbox,  # B,8
                "LP_dn_logits": LP_dn_logits,  # B,C,N
            }
            out = {**out_infer, **out_denoise}
            return out

        def prepare_query_denoise(self, B, inputs: dict):
            """
            inputs:
                'imgs'      : [B, 3, H, W]
                'pos_bbox'  : [B, 8]  # xywhxywh
                'neg_bbox'  : [B, 8]
                'LPs_delay' : [B, 8]  # IDs
            """
            tgt_infer, pos_infer = self.decoder.prepare_query(B)

            # Embed LPs_delay
            lps_delay_ids = inputs["LPs_delay"].permute(1, 0)  # [B, 8]
            lps_embed = self.lps_embed(lps_delay_ids)  # [B, 8, 256]

            # Project pos and neg bbox
            pos_bbox = inputs["pos_bbox"].squeeze(1)  # [B, 8]
            neg_bbox = inputs["neg_bbox"].squeeze(1)  # [B, 8]
            pos_bbox_embed = self.bbox_embed(pos_bbox)  # [B, 256]
            neg_bbox_embed = self.bbox_embed(neg_bbox)  # [B, 256]

            noise_pos = self.decoder.de_pos_infer.weight.unsqueeze(1).repeat(2, B, 1)
            noise_pos[0] = noise_pos[0] + pos_bbox_embed
            noise_pos[self.decoder.Lgroup] = (
                noise_pos[self.decoder.Lgroup] + neg_bbox_embed
            )

            pad_0 = torch.zeros([1, B, self.d_model], device=tgt_infer.device)

            tgt = torch.cat([tgt_infer, pad_0, lps_embed, pad_0, lps_embed], dim=0)
            pos = torch.cat([pos_infer, noise_pos], dim=0)

            tgt_mask = self.tgt_mask

            return tgt, pos, tgt_mask

    pass


# DETR resnet18
# from torchvision.models._utils import IntermediateLayerGetter
import models_detr.backbone

class ALPR_res18_fm32_stn_2en_CDNv1_1(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.encoder = Encoder.DETR_2en_STN_res18_fm32()
        self.decoder = Decoder.CiT_CNDv1_1(decoder=Decoder.CiT_v1())
        return
    def forward(self, inputs: dict, denoise: bool = True):
        return ALPR_res18_fm16_STN_cdn.forward(self, inputs, denoise)
    pass 

class ALPR_res50_fm32_stn_2en_CDNv1_1(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.encoder = Encoder.DETR_2en_STN_res50_fm32()
        self.decoder = Decoder.CiT_CNDv1_1(decoder=Decoder.CiT_v1())
        return
    def forward(self, inputs: dict, denoise: bool = True):
        return ALPR_res18_fm16_STN_cdn.forward(self, inputs, denoise)
    pass 
class ALPR_res18_fm32_2en_CDNv1_1(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.encoder = Encoder.DETR_2en_res18()
        self.decoder = Decoder.CiT_CNDv1_1(decoder=Decoder.CiT_v1())
        return
    def forward(self, inputs: dict, denoise: bool = True):
        return ALPR_res18_fm16_STN_cdn.forward(self, inputs, denoise)
    pass 

class ALPR_res18_fm32_CDNv1_1(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.encoder = Encoder.DETR_res18()
        self.decoder = Decoder.CiT_CNDv1_1(decoder=Decoder.CiT_v1())
        return
    def forward(self, inputs: dict, denoise: bool = True):
        return ALPR_res18_fm16_STN_cdn.forward(self, inputs, denoise)
    pass 
class ALPR_res18_fm32_STN_CDNv1_1(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.encoder = Encoder.DETR_STN_res18_fm32()
        self.decoder = Decoder.CiT_CNDv1_1(decoder=Decoder.CiT_v1())
        return
    def forward(self, inputs: dict, denoise: bool = True):
        return ALPR_res18_fm16_STN_cdn.forward(self, inputs, denoise)
    pass 

class CiT_CDN_res18(Decoder.CiT_CND_v1):
    """TODO del this"""

    def __init__(
        self,
        num_predict=10,
        Lgroup=9,
        nlayers=2,
        nhead=8,
        d_ffn=512,
        d_model=256,
        n_class=2,
        n_bbox_vertex=4,
        n_character=75,
    ):
        super().__init__(
            num_predict,
            Lgroup,
            nlayers,
            nhead,
            d_ffn,
            d_model,
            n_class,
            n_bbox_vertex,
            n_character,
        )
        self.replace_backbone(d_model)
        return

    def replace_backbone(self, d_model=256, name_backbone="resnet18"):
        backbone_res18 = models_detr.backbone.Backbone(
            name_backbone,
            train_backbone=True,
            return_interm_layers=False,
            dilation=False,
        )
        self.backbone[0] = backbone_res18
        self.input_proj = nn.Conv2d(backbone_res18.num_channels, d_model, kernel_size=1)

    pass


class CiT_CDN_res18_fm16(Decoder.CiT_CND_v1):
    """TODO del this"""

    def __init__(
        self,
        num_predict=10,
        Lgroup=9,
        nlayers=2,
        nhead=8,
        d_ffn=512,
        d_model=256,
        n_class=2,
        n_bbox_vertex=4,
        n_character=75,
    ):
        super().__init__(
            num_predict,
            Lgroup,
            nlayers,
            nhead,
            d_ffn,
            d_model,
            n_class,
            n_bbox_vertex,
            n_character,
        )
        self.replace_backbone(d_model)
        return

    def replace_backbone(self, d_model=256, name_backbone="resnet18"):
        backbone_res18 = models_detr.backbone.Backbone(
            name_backbone,
            train_backbone=True,
            return_interm_layers="layer3",
            dilation=False,
        )
        self.backbone[0] = backbone_res18
        self.input_proj = nn.Conv2d(
            backbone_res18.num_channels // 2, d_model, kernel_size=1
        )

    pass


class ALPR_res18_fm16_STN_cdn(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.encoder = Encoder.DETR_STN_res18_fm16()
        self.decoder = Decoder.CiT_CND_v1(decoder=Decoder.CiT_v1())
        return

    def forward(self, inputs: dict, denoise: bool = True):
        """
        inputs: {
            'imgs': Tensor of shape [B, 3, H, W]
            'pos_bbox': Tensor of shape [B, 8]           # xywhxywh
            'neg_bbox': Tensor of shape [B, 8]           # xywhxywh
            'LPs_delay': Tensor of shape [B, 8]
        }

        return: {
            "pred_logits": Tensor of shape [B,pred, 2]
            "pred_boxes": Tensor of shape [B,pred, 8] #xywh xywh
            "pred_string_logits": Tensor of shape [B, C, pred, L]

            "pos_class_logit": Tensor of shape [B, 2]
            "neg_class_logit": Tensor of shape [B, 2]
            "pos_bbox": Tensor of shape [B, 8]
            "neg_bbox": Tensor of shape [B, 8]
            "LP_dn_logits": Tensor of shape [B, C, N]
        }
        """
        en_outputs = self.encoder.forward(inputs["imgs"])
        inputs["en_outputs"] = en_outputs
        outputs = self.decoder.forward(inputs, denoise)
        return outputs

    pass


class ALPR_res18_fm32_STN_cdn(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.encoder = Encoder.DETR_STN_res18_fm32()
        self.decoder = Decoder.CiT_CND_v1(decoder=Decoder.CiT_v1())
        return

    def forward(self, inputs: dict, denoise: bool = True):
        return ALPR_res18_fm16_STN_cdn.forward(self, inputs, denoise)


class ALPR_res18_fm32_cdn(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.encoder = Encoder.DETR_res18()
        self.decoder = Decoder.CiT_CND_v1(decoder=Decoder.CiT_v1())
        return

    def forward(self, inputs: dict, denoise: bool = True):
        return ALPR_res18_fm16_STN_cdn.forward(self, inputs, denoise)
    pass 
class ALPR_res18_fm32_CDNv1_1(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.encoder = Encoder.DETR_res18()
        self.decoder = Decoder.CiT_CNDv1_1(decoder=Decoder.CiT_v1())
        return
    def forward(self, inputs: dict, denoise: bool = True):
        return ALPR_res18_fm16_STN_cdn.forward(self, inputs, denoise)
    pass 
class ALPR_res18_fm32_STN_CDNv1_1(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.encoder = Encoder.DETR_STN_res18_fm32()
        self.decoder = Decoder.CiT_CNDv1_1(decoder=Decoder.CiT_v1())
        return
    def forward(self, inputs: dict, denoise: bool = True):
        return ALPR_res18_fm16_STN_cdn.forward(self, inputs, denoise)
    pass 
class ALPR_res18_fm32_CDNv2(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.encoder = Encoder.DETR_res18()
        self.decoder = Decoder.CiT_CNDv2(decoder=Decoder.CiT())
        return

    def forward(self, inputs: dict, denoise: bool = True):
        return ALPR_res18_fm16_STN_cdn.forward(self, inputs, denoise)
    pass 
class ALPR_res18_fm32_CDNv3(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.encoder = Encoder.DETR_res18()
        self.decoder = Decoder.CiT_CNDv3(decoder=Decoder.CiT())
        return

    def forward(self, inputs: dict, denoise: bool = True):
        return ALPR_res18_fm16_STN_cdn.forward(self, inputs, denoise)
