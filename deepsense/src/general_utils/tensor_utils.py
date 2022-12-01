import torch


def bisc_to_bcis(loc_mod_features):
    """Convert the shape of loc_mod_features from [b, i, s ,c] to [b, c, i, s].

    Args:
        loc_mod_features (_type_): _description_
    """
    output_loc_mod_features = dict()

    for loc in loc_mod_features:
        output_loc_mod_features[loc] = torch.permute(loc_mod_features[loc], (0, 3, 1, 2))

    return output_loc_mod_features


def bcis_to_bisc(loc_mod_features):
    """Convert the shape of loc_mod_features from [b, c, i ,s] to [b, i, s, c].

    Args:
        loc_mod_features (_type_): _description_
    """
    output_loc_mod_features = dict()

    for loc in loc_mod_features:
        output_loc_mod_features[loc] = torch.permute(loc_mod_features[loc], (0, 2, 3, 1))

    return output_loc_mod_features


def miss_ids_to_masks(miss_ids, target_shape, device):
    """Generate the miss_masks with the same shape as the target_shape.
       Note: 1 means available, 0 means missing.

    Args:
        miss_ids (_type_): [[miss_ids] for each sample]
        sensors (_type_): _description_
        target_shape (_type_): [b, c, i, s]
        device (_type_): _description_
    Return:
        masks: [b, c, i, s]
    """
    b, c, i, s = target_shape
    masks = torch.ones([b, s, c, i]).to(device)

    for sample_id, sample_miss_ids in enumerate(miss_ids):
        masks[sample_id][sample_miss_ids] = 0

    # [b, s, c, i] --> [b, c, i, s]
    masks = torch.permute(masks, (0, 2, 3, 1))

    return masks


def masks_to_miss_ids(miss_masks):
    """Generate the miss sensors ids from the miss masks.
    NOTE: The batch dimension should be 1.

    Args:
        miss_masks (_type_): [b, c, i, s]
    """
    if miss_masks.ndim == 4:
        sample_mask = miss_masks[0]
    else:
        sample_mask = miss_masks

    sample_mask = torch.permute(sample_mask, [2, 0, 1])
    sample_mask = torch.mean(sample_mask, dim=[1, 2])
    avl_ids = torch.nonzero(sample_mask).flatten()
    miss_ids = torch.nonzero(sample_mask == 0).flatten()

    return avl_ids, miss_ids


def manual_mask_select(x, miss_masks):
    """Only preserve the elements with positive mask values.

    Args:
        x (_type_): []b, c, i, s]
        miss_masks (_type_): [b, c, i, s], some modalities can be missing, but each sample have same #missing mods.
    """
    sample_features = torch.split(x, 1)
    out_sample_features = []
    for sample_id, sample_feature in enumerate(sample_features):
        avl_ids, miss_ids = masks_to_miss_ids(miss_masks[sample_id])
        out_sample_feature = torch.index_select(sample_feature, dim=3, index=avl_ids)
        out_sample_features.append(out_sample_feature)

    x_out = torch.cat(out_sample_features, dim=0)

    return x_out
