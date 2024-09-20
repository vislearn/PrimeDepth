import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from contextlib import redirect_stdout

from einops import rearrange

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
    checkpoint
    )
from ldm.modules.attention import SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import (
    UNetModel,
    TimestepEmbedSequential,
    ResBlock as ResBlock_orig,
    Downsample,
    Upsample,
    AttentionBlock,
    TimestepBlock
    )
from ldm.modules.diffusionmodules.model import Decoder
from ldm.util import exists


class PrimeDepthLabeller(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            num_res_blocks,
            attention_resolutions,
            label_decoder_config,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=False,
            use_linear_in_transformer=False,
            infusion2refiner='cat',         # how to infuse intermediate information into the refiner? {'add', 'cat', None}
            refiner_model_ratio=1.0,        # ratio of the refiner size compared to the base model. [0, 1]
            label_mode={
                'depth': 3,
                'segmentation': 20
                },                          # label mode and corresponding number of output channels
            use_self_attn_maps=True,        # use self attention maps from the base model
            use_cross_attn_maps=True,       # use cross attention maps from the base model
            use_feature_maps=True,          # use intermediate features from the base model
            n_ca_maps=77,                   # number of cross-attention maps
            n_sa_maps=64,                   # number of self-attention maps
            channels2predictor=512,
            zero_convs=True,                # use zero-convolutions for the infusion of features from base to refiner
            scale_input_features_with_model=False,  # scale input features from base to refiner with refiner model size
    ):
        assert infusion2refiner in ('cat', 'add', 'att', None), f'infusion2refiner needs to be cat, add, att or None, but not {infusion2refiner}'
        super().__init__()

        self.infusion2refiner = infusion2refiner
        self.in_ch_factor = 1 if infusion2refiner in ('add', 'att') else 2
        self.refiner_model_ratio = refiner_model_ratio
        self.out_channels = out_channels
        self.dims = 2
        self.model_channels = model_channels
        self.label_mode = label_mode
        self.use_self_attn_maps = use_self_attn_maps
        self.use_cross_attn_maps = use_cross_attn_maps
        self.use_feature_maps = use_feature_maps
        self.n_ca_maps = n_ca_maps
        self.n_sa_maps = n_sa_maps
        self.channels2predictor = channels2predictor
        self.zero_convs = zero_convs

        with redirect_stdout(None):
            ################# dummy base model to gather information #################
            base_model = UNetModel(
                image_size=image_size, in_channels=in_channels, model_channels=model_channels,
                out_channels=out_channels, num_res_blocks=num_res_blocks,
                attention_resolutions=attention_resolutions, dropout=dropout, channel_mult=channel_mult,
                conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint,
                use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels,
                num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,
                resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order,
                use_spatial_transformer=use_spatial_transformer, transformer_depth=transformer_depth,
                context_dim=context_dim, n_embed=n_embed, legacy=legacy,
                use_linear_in_transformer=use_linear_in_transformer,
                )

            #####################################################################
            ###     Gathering information about Attention Block Existince     ###
            #####################################################################
            blockwise_attn_dec = [torch.tensor(False)]
            blockwise_attn_enc = []
            heads_enc = []
            heads_dec = [0]
            for module in base_model.input_blocks:
                has_att = False
                heads = 0
                for mod in list(module.children()):
                    if isinstance(mod, SpatialTransformer):
                        has_att = True
                        heads = mod.transformer_blocks[0].attn1.heads
                heads_enc.append(heads)
                blockwise_attn_enc.append(has_att)

            blockwise_middle = False
            for mod in list(base_model.middle_block.children()):
                heads = 0
                if isinstance(mod, SpatialTransformer):
                    blockwise_middle = True
                    heads_middle = mod.transformer_blocks[0].attn1.heads

            for module in base_model.output_blocks:
                has_att = False
                heads = 0
                for mod in list(module.children()):
                    if isinstance(mod, SpatialTransformer):
                        has_att = True
                        heads = mod.transformer_blocks[0].attn1.heads
                heads_dec.append(heads)
                blockwise_attn_dec.append(has_att)

            additional_attention_blocks = torch.tensor(blockwise_attn_enc[::-1]) + torch.tensor(blockwise_attn_dec[:-1])
            additional_attention_blocks[0] = torch.tensor([blockwise_middle]) or additional_attention_blocks[0]

            additional_attention_channels = []
            ch_per_head = n_ca_maps * use_cross_attn_maps + n_sa_maps * use_self_attn_maps
            for h_enc, h_dec in zip(heads_enc[::-1], heads_dec[:-1]):
                additional_attention_channels.append(max(h_enc, h_dec) * ch_per_head)
            additional_attention_channels[0] = max(heads_middle * ch_per_head, additional_attention_channels[0])

            self.refiner = Refiner(
                    image_size=image_size, in_channels=in_channels, model_channels=model_channels,
                    out_channels=channels2predictor, num_res_blocks=num_res_blocks,
                    attention_resolutions=attention_resolutions, dropout=dropout, channel_mult=channel_mult,
                    conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint,
                    use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels,
                    num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,
                    resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order,
                    use_spatial_transformer=use_spatial_transformer, transformer_depth=transformer_depth,
                    context_dim=context_dim, n_embed=n_embed, legacy=legacy,
                    use_linear_in_transformer=use_linear_in_transformer,
                    infusion2refiner=infusion2refiner, refiner_model_ratio=refiner_model_ratio,
                    additional_attention_channels=additional_attention_channels,
                    additional_attention_blocks=additional_attention_blocks,
                    scale_input_features_with_model=scale_input_features_with_model,
                    use_feature_maps=use_feature_maps,
                    )  # initialise pretrained model

            # self.enc_zero_convs_out = nn.ModuleList([])
            self.enc_zero_convs_in = nn.ModuleList([])

            # self.middle_block_out = None
            self.middle_block_in = None

            # self.dec_zero_convs_out = nn.ModuleList([])
            self.dec_zero_convs_in = nn.ModuleList([])

            ch_inout_refiner = {'enc': [], 'mid': [], 'dec': []}
            ch_inout_base = {'enc': [], 'mid': [], 'dec': []}

            ################# Gather Channel Sizes #################

            for module in base_model.input_blocks:
                if isinstance(module[0], nn.Conv2d):
                    ch_inout_base['enc'].append((module[0].in_channels, module[0].out_channels))
                elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                    ch_inout_base['enc'].append((module[0].channels, module[0].out_channels))
                elif isinstance(module[0], Downsample):
                    ch_inout_base['enc'].append((module[0].channels, module[-1].out_channels))

            ch_inout_base['mid'].append((base_model.middle_block[0].channels, base_model.middle_block[-1].out_channels))

            for module in self.refiner.output_blocks:
                if isinstance(module[0], nn.Conv2d):
                    ch_inout_refiner['dec'].append((module[0].in_channels, module[0].out_channels))
                elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                    ch_inout_refiner['dec'].append((module[0].channels, module[0].out_channels))
                elif isinstance(module[-1], Upsample):
                    ch_inout_refiner['dec'].append((module[0].channels, module[-1].out_channels))

            for module in base_model.output_blocks:
                if isinstance(module[0], nn.Conv2d):
                    ch_inout_base['dec'].append((module[0].in_channels, module[0].out_channels))
                elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                    ch_inout_base['dec'].append((module[0].channels, module[0].out_channels))
                elif isinstance(module[-1], Upsample):
                    ch_inout_base['dec'].append((module[0].channels, module[-1].out_channels))

            self.ch_inout_refiner = ch_inout_refiner
            self.ch_inout_base = ch_inout_base

            ########################################
            ###     Build Zero-Convolutions      ###
            ########################################
            # infusion2refiner

            self.convs_features = nn.ModuleList()
            self.convs_attentions = nn.ModuleList()
            self.convs_merge = nn.ModuleList()

            ############## kill ################
            self.kill_blockwise_attn_enc = blockwise_attn_enc[::-1]
            self.kill_blockwise_attn_dec = blockwise_attn_dec
            ####################################

            for io_ch_base_enc, io_ch_base_dec, ch_features, ch_attn, attn_heads_enc, attn_heads_dec in zip(
                ch_inout_base['enc'][::-1],
                ch_inout_base['dec'],
                self.refiner.ch_feature_stream,
                self.refiner.ch_attn_stream,
                heads_enc[::-1],  # blockwise_attn_enc[::-1],
                heads_dec,  # blockwise_attn_dec
            ):

                if self.use_feature_maps:
                    self.convs_features.append(self.make_zero_conv(
                        # in_channels=ch_base_enc[-1] + ch_base_dec[-1], out_channels=ch_features))
                        in_channels=io_ch_base_dec[0], out_channels=ch_features, zero=zero_convs))
                else:
                    self.convs_features.append(None)

                if self.use_self_attn_maps or self.use_cross_attn_maps:
                    ch_attn_proj = ch_attn if ch_attn > 0 else ch_features
                    ch_attn_in = (n_ca_maps * self.use_cross_attn_maps + n_sa_maps * self.use_self_attn_maps)
                    # if attn_enc or attn_dec or blockwise_middle:
                    if attn_heads_enc + attn_heads_dec + heads_middle:
                        self.convs_attentions.append(self.make_zero_conv(
                            in_channels=ch_attn_in * (heads_middle + attn_heads_enc + attn_heads_dec),
                            out_channels=ch_attn_proj, zero=zero_convs))
                        blockwise_middle = False
                        heads_middle = 0
                    else:
                        self.convs_attentions.append(None)
                        ch_attn_proj = 0

                    if ch_features + ch_attn_proj != 0:
                        self.convs_merge.append(self.make_zero_conv(
                            in_channels=ch_features + ch_attn_proj, out_channels=ch_features + ch_attn, zero=zero_convs))
                    else:
                        self.convs_merge.append(None)
                else:
                    if ch_features != 0:
                        self.convs_merge.append(self.make_zero_conv(
                            in_channels=ch_features, out_channels=ch_features + ch_attn, zero=zero_convs))
                    else:
                        self.convs_merge.append(None)
            label_channels = 0
            for key in label_mode:
                label_channels += label_mode[key]
            self.label_channels = label_channels
            self.label_predictor = LabelDecoder(
                                        input_channels=channels2predictor,
                                        channels_out=label_mode['segmentation'],
                                        label_decoder_config=label_decoder_config,
                                        )
            self.depth_predictor = LabelDecoder(
                                        input_channels=channels2predictor,
                                        channels_out=label_mode['depth'],
                                        label_decoder_config=label_decoder_config,
                                        )

    def make_zero_conv(self, in_channels, out_channels=None, zero=True):
        in_channels = in_channels
        out_channels = out_channels or in_channels
        if zero:
            return zero_module(conv_nd(self.dims, in_channels, out_channels, 1, padding=0))
        return conv_nd(self.dims, in_channels, out_channels, 1, padding=0)

    def infuse(self, stream, infusion, mlp, variant, emb, scale=1.0):
        if variant == 'add':
            stream = stream + mlp(infusion, emb) * scale
        elif variant == 'cat':
            stream = torch.cat([stream, mlp(infusion, emb) * scale], dim=1)
        elif variant == 'att':
            stream = mlp(stream, infusion)

        return stream

    def prepare_attn_map_features(self, attn_map, attn_version, x_shape, n_slices=None, patches=None):
        '''
        Preprocesses the attention map features based on the specified attention version.

        Parameters:
            attn_map (ndarray): The attention map tensor of shape (bs, heads, pixels, tokens).
            attn_version (str): The version of attention to be applied. Possible values are 'self' or 'cross'.
            n_slices (int, optional): The number of slices. Default is None.
            patches (int, optional): The number of patches per row and column. Default is None.

        Returns:
            ndarray: The preprocessed attention map tensor.
        '''
        assert n_slices is not None or patches is not None, '!Either <n_slices> or <patches> has to be specified!'
        bs, heads, pixels, tokens = attn_map.shape
        b, c, h, w = x_shape
        # h = w = int(np.sqrt(pixels))
        if attn_version == 'self':
            # normalise first, patch after
            if patches is not None:
                h_p = max(1, h // patches)
                w_p = max(1, w // patches)

                h_mod = 8 - (h % 8)
                w_mod = 8 - (w % 8)

                attn_map = rearrange(attn_map, 'b head p (h w) -> (b head) p h w', h=h, w=w)
                if h_mod != 0 or w_mod != 0:
                    attn_map = F.interpolate(attn_map, size=(h + h_mod, w + w_mod), mode="bilinear")

                attn_map = nn.AvgPool2d(kernel_size=(h_p, w_p), stride=(h_p, w_p))(attn_map)

                if h_mod != 0 or w_mod != 0:
                    attn_map = F.interpolate(attn_map, size=(patches, patches), mode="bilinear")

                # normalisation of self-attention maps across head dimensions
                attn_map = attn_map / attn_map.max(dim=-3, keepdim=True)[0].max(dim=-2, keepdim=True)[0].max(dim=-1, keepdim=True)[0]
                attn_map = rearrange(attn_map, '(b head) (h w) h_p w_p -> b (head h_p w_p) h w', h=h, w=w, b=bs, h_p=patches)

            else:
                attn_map = rearrange(attn_map, 'b head (n slice) p -> b head n slice p', n=n_slices).mean(dim=3)
                attn_map = rearrange(attn_map, 'b head n (h w) -> b (head n) h w', h=h, w=w)

        elif attn_version == 'cross':
            # normalisation of cross-attention maps
            # normalisation not necessary because maximum values for each head are 1 by construction

            attn_map = rearrange(attn_map, 'b head p (n slice) -> b head n slice p', n=n_slices).mean(dim=3)
            attn_map = rearrange(attn_map, 'b head n (h w) -> b (head n) h w', h=h, w=w)

        return attn_map

    def get_used_attn_maps(self, attn_maps):
        if self.use_self_attn_maps and self.use_cross_attn_maps:
            return attn_maps
        elif self.use_self_attn_maps:
            return [attn_maps[0]]
        elif self.use_cross_attn_maps:
            return [attn_maps[1]]
        return None

    def forward(self, x, timesteps, context, base_model, return_labels=False, return_attn=False, exclude_depth=False, exclude_segmentation=False,  **kwargs):
        if not return_labels:
            return base_model(x=x, timesteps=timesteps, context=context, **kwargs)

        #########################
        ###    PREPARATION    ###
        #########################
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = base_model.time_embed(t_emb)

        h_base = x.type(base_model.dtype)
        hs_base = []
        attn_maps_base_enc = []
        it_feature_convs = iter(self.convs_features)
        if self.use_cross_attn_maps or self.use_self_attn_maps:
            it_attn_convs = iter(self.convs_attentions)
        it_merge_convs = iter(self.convs_merge)
        used_attn_maps_base = {'self': {}, 'cross': {}}
        used_attn_maps_refiner = {'self': {}, 'cross': {}}

        #########################
        ###    INPUT BLOCK    ###
        #########################
        for i, module_base in enumerate(base_model.input_blocks):
            shape = h_base.shape
            h_base, attn_maps_base = module_base(h_base, emb, context, return_attn_maps=True)
            hs_base.append(h_base)

            if attn_maps_base is not None:
                attn_maps_base = [
                    self.prepare_attn_map_features(attn_maps_base[0], 'self', patches=int(np.sqrt(self.n_sa_maps)), x_shape=shape),
                    self.prepare_attn_map_features(attn_maps_base[1], 'cross', n_slices=self.n_ca_maps, x_shape=h_base.shape)
                ]
                if return_attn:
                    used_attn_maps_base['self'][f'input_block-{i}'] = attn_maps_base[0]
                    used_attn_maps_base['cross'][f'input_block-{i}'] = attn_maps_base[1]
  
            attn_maps_base_enc.append(attn_maps_base)

        h_base, attn_maps_base = base_model.middle_block(h_base, emb, context, return_attn_maps=True)

        attn_maps_base = [
            self.prepare_attn_map_features(attn_maps_base[0], 'self', patches=int(np.sqrt(self.n_sa_maps)), x_shape=shape),
            self.prepare_attn_map_features(attn_maps_base[1], 'cross', n_slices=self.n_ca_maps, x_shape=h_base.shape)
            ]
        h_refiner = None
        if return_attn:
            used_attn_maps_base['self']['mid_block'] = attn_maps_base[0]
            used_attn_maps_base['cross']['mid_block'] = attn_maps_base[1]

        #######################################
        ###    OUTPUT BLOCK WITH REFINER    ###
        #######################################
        for i, (module_base, module_refiner) in enumerate(zip(
                base_model.output_blocks,
                self.refiner.output_blocks
                )):

            attn_maps_enc = attn_maps_base_enc.pop()

            # prepare attention maps
            if self.use_self_attn_maps or self.use_cross_attn_maps:
                if attn_maps_base is not None:
                    attn_maps_all = torch.cat(self.get_used_attn_maps(attn_maps_base), dim=1)
                    if attn_maps_enc is not None:
                        attn_maps_enc = torch.cat(self.get_used_attn_maps(attn_maps_enc), dim=1)
                        if attn_maps_all.shape[-2:] != attn_maps_enc.shape[-2:]:
                            attn_maps_all = F.interpolate(attn_maps_all, size=attn_maps_enc.shape[-2:], mode="bilinear")
                        attn_maps_all = torch.cat([
                            attn_maps_all,
                            attn_maps_enc
                            ], dim=1)
                elif attn_maps_enc is not None:
                    attn_maps_all = torch.cat(self.get_used_attn_maps(attn_maps_enc), dim=1)

            h_base = th.cat([h_base, hs_base.pop()], dim=1)

            if len(hs_base) != 0:
                output_size = hs_base[-1].shape
            else:
                output_size = None

            # process features and attention maps for Refiner
            # get zero-convolutions
            feature_conv = next(it_feature_convs)
            if self.use_cross_attn_maps or self.use_self_attn_maps:
                attn_conv = next(it_attn_convs)
            else:
                attn_conv = None
            merge_conv = next(it_merge_convs)

            # process inputs from base model
            merge_input = []
            if feature_conv is not None:
                features_base = feature_conv(h_base)
                merge_input.append(features_base)
            if attn_conv is not None:
                features_attn = attn_conv(attn_maps_all)
                merge_input.append(features_attn)
            if merge_conv is not None:
                features_merged = merge_conv(
                    torch.cat(merge_input, dim=1))
            else:
                features_merged = None

            # process with Refiner
            if h_refiner is None:
                # there should always be either attention features of feature maps
                if return_attn:
                    h_refiner, attn_maps_refiner = module_refiner(features_merged, emb, context, return_attn_maps=True, output_size=output_size)
                else:
                    h_refiner = module_refiner(features_merged, emb, context, return_attn_maps=False, output_size=output_size)
            else:
                if features_merged is not None:
                    merged_input = torch.cat([h_refiner, features_merged], dim=1)
                else:
                    merged_input = h_refiner
                if return_attn:
                    h_refiner, attn_maps_refiner = module_refiner(merged_input, emb, context, return_attn_maps=True, output_size=output_size)
                else:
                    h_refiner = module_refiner(merged_input, emb, context, return_attn_maps=False, output_size=output_size)

            ##### compute additional features for refiner #####
            shape = h_base.shape
            h_base, attn_maps_base = module_base(h_base, emb, context, return_attn_maps=True, output_size=output_size)
            if attn_maps_base is not None:
                attn_maps_base = [
                    self.prepare_attn_map_features(attn_maps_base[0], 'self', patches=int(np.sqrt(self.n_sa_maps)), x_shape=shape),
                    self.prepare_attn_map_features(attn_maps_base[1], 'cross', n_slices=self.n_ca_maps, x_shape=shape)
                ]
                if return_attn:

                    attn_maps_refiner = [
                        self.prepare_attn_map_features(attn_maps_refiner[0], 'self', patches=int(np.sqrt(self.n_sa_maps)), x_shape=shape),
                        self.prepare_attn_map_features(attn_maps_refiner[1], 'cross', n_slices=self.n_ca_maps, x_shape=shape)
                    ]
                    used_attn_maps_base['self'][f'out_block-{i}'] = attn_maps_base[0]
                    used_attn_maps_base['cross'][f'out_block-{i}'] = attn_maps_base[1]
                    used_attn_maps_refiner['self'][f'out_block-{i}'] = attn_maps_refiner[0]
                    used_attn_maps_refiner['cross'][f'out_block-{i}'] = attn_maps_refiner[1]

        #########################################################################
        ###    LABEL CALCULATION WITH PREDICTION HEADS FROM REFINER OUTPUT    ###
        #########################################################################
        if return_attn:
            return used_attn_maps_base, used_attn_maps_refiner
        labels = {}

        h_refiner = self.refiner.out(h_refiner)

        if not exclude_depth:
            labels['depth'] = self.depth_predictor(h_refiner)
        if not exclude_segmentation:
            labels['segmentation'] = self.label_predictor(h_refiner)

        return base_model.out(h_base), labels


class Refiner(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        use_linear_in_transformer=False,
        infusion2refiner='cat',         # how to infuse intermediate information into the refiner? {'add', 'cat', None}
        refiner_model_ratio=1.0,
        additional_attention_channels=1128,     # additional features from attention maps
        additional_attention_blocks=None,       # boolean list stating existance of attention layers in blocks
        scale_input_features_with_model=False,  # scale input features with ratio for model channels
        use_feature_maps=True,                  # use intermediate features from the base model
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        self.infusion2refiner = infusion2refiner
        if scale_input_features_with_model:
            infusion_factor = 1
        else:
            infusion_factor = 1 / refiner_model_ratio

        cat_infusion = 1 if infusion2refiner == 'cat' else 0

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None
        time_embed_dim = model_channels * 4
        additional_attention_channels = [int(attn_ch * refiner_model_ratio) for attn_ch in additional_attention_channels]
        self.additional_attention_channels = additional_attention_channels
        self.scale_input_features_with_model = scale_input_features_with_model
        self.use_feature_maps = use_feature_maps

        model_channels = max(1, int(model_channels * refiner_model_ratio))
        self.model_channels = model_channels
        self.refiner_model_ratio = refiner_model_ratio

        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1

        ############# START - getting rid of the encoder initialisation #############
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                ch = mult * model_channels
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                input_block_chans.append(ch)
                ds *= 2

        if num_head_channels == -1:
            dim_head = max(num_heads, ch // num_heads)
        else:
            # custom code for smaller models - start
            num_head_channels = find_denominator(ch, min(ch, self.num_head_channels))
            # custom code for smaller models - end
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels

        ############# END - getting rid of the encoder initialisation #############

        self.ch_label_stream = []
        self.ch_feature_stream = []
        self.ch_attn_stream = []
        self.output_blocks = nn.ModuleList([])
        attentions_exist = iter(additional_attention_blocks)
        additional_attn_ch = iter(additional_attention_channels)
        out_ch = 0

        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                attn_flag = next(attentions_exist)
                additional_attention_features = next(additional_attn_ch)
                ich = input_block_chans.pop()
                if level == len(channel_mult) - 1 and i == 0:
                    self.ch_feature_stream.append(int(ch * cat_infusion * infusion_factor) * 2 if self.use_feature_maps else 0)
                    self.ch_label_stream.append(0)
                else:
                    self.ch_feature_stream.append(int(ch + (ich + ch) * cat_infusion * infusion_factor) - ch if self.use_feature_maps else 0)
                    self.ch_label_stream.append(ich)

                self.ch_attn_stream.append(additional_attention_features * attn_flag)
                layers = [
                    ResBlock(
                        # int(ch * cat_infusion * infusion_factor) * 2 + additional_attention_features * attn_flag if level == len(channel_mult) - 1 and i == 0 else int(
                        #     ch + (ich + ch) * cat_infusion * infusion_factor) + additional_attention_features * attn_flag,
                        int(ch * cat_infusion * infusion_factor) * 2 * self.use_feature_maps + additional_attention_features * attn_flag if level == len(channel_mult) - 1 and i == 0 else int(
                            (ich + ch) * cat_infusion * infusion_factor) * self.use_feature_maps + ch + additional_attention_features * attn_flag,
                            # ch + (ich + ch) * cat_infusion * infusion_factor) + additional_attention_features * attn_flag,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = max(num_heads, ch // num_heads)
                    else:
                        # custom code for smaller models - start
                        num_head_channels = find_denominator(ch, min(ch, self.num_head_channels))
                        # custom code for smaller models - end
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )


class LabelDecoder(Decoder):
    def __init__(self, input_channels, channels_out, label_decoder_config):
        super().__init__(**label_decoder_config)
        self.channels_out = channels_out
        block_in = self.conv_out.in_channels

        self.conv_in = nn.Conv2d(input_channels, 512, 3, 1, 1)
        self.conv_out = nn.Conv2d(block_in, channels_out, 3, 1, 1)


def find_denominator(number, start):
    if start >= number:
        return number
    while (start != 0):
        residual = number % start
        if residual == 0:
            return start
        start -= 1


def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    if find_denominator(channels, 32) < 32:
        print(f'[USING GROUPNORM OVER LESS CHANNELS ({find_denominator(channels, 32)}) FOR {channels} CHANNELS]')
    return GroupNorm_leq32(find_denominator(channels, 32), channels)


class GroupNorm_leq32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h
