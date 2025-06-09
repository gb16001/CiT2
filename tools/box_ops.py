import torch
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def validate_xyxy_bbox(bboxes: torch.Tensor):
    """
    检查 xyxy 格式的 bbox 是否合法，如果不合法就进行修正。

    参数:
        bboxes (Tensor): 形状为 [..., 4] 的张量，每个 bbox 是 [x0, y0, x1, y1] 格式。

    返回:
        Tuple[Tensor, Tensor]: 
            - 一个 bool 类型的张量，表示每个 bbox 是否原本合法。
            - 一个修正后的 bbox 张量，仍为 xyxy 格式。
    """
    x0, y0, x1, y1 = bboxes.unbind(-1)

    valid_x = x0 <= x1
    valid_y = y0 <= y1
    is_valid = valid_x & valid_y
    if is_valid.all() :
        return is_valid,bboxes
    # 修正非法的 bbox
    corrected_x0 = torch.min(x0, x1)
    corrected_y0 = torch.min(y0, y1)
    corrected_x1 = torch.max(x0, x1)
    corrected_y1 = torch.max(y0, y1)

    corrected = torch.stack([corrected_x0, corrected_y0, corrected_x1, corrected_y1], dim=-1)

    return is_valid, corrected
