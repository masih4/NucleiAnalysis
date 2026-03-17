from transformers import SegformerModel, SegformerConfig
import segmentation_models_pytorch as smp
from Models.ACS.unet_decoder.decoder import UnetDecoder
import matplotlib.pyplot as plt
from segmentation_models_pytorch.decoders.unetplusplus.decoder import UnetPlusPlusDecoder

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class TimmEncoderFixed(nn.Module):
    def __init__(
        self,
        name,
        pretrained=True,
        in_channels=3,
        depth=5,
        output_stride=32,
        drop_rate=0.5,
        drop_path_rate=0.0,
    ):
        super().__init__()
        kwargs = dict(
            in_chans=in_channels,
            features_only=True,
            pretrained=pretrained,
            out_indices=tuple(range(depth)),
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        self.model = timm.create_model(name, **kwargs)

        self._in_channels = in_channels
        self._out_channels = [
            in_channels,
        ] + self.model.feature_info.channels()
        self._depth = depth
        self._output_stride = output_stride

    def forward(self, x):
        features = self.model(x)
        # features = [
        #     x,
        # ] + features
        return features

    @property
    def out_channels(self):
        return self._out_channels

    @property
    def output_stride(self):
        return min(self._output_stride, 2**self._depth)


def get_timm_encoder(
    name,
    in_channels=3,
    depth=5,
    weights=False,
    output_stride=32,
    drop_rate=0.5,
    drop_path_rate=0.25,
):
    encoder = TimmEncoderFixed(
        name, weights, in_channels, depth, output_stride, drop_rate, drop_path_rate
    )
    return encoder





class FusionBlock0(nn.Module):
    """
    simple concatenate
    out = concatenate([s,u])
    """
    def __init__(self):
        super().__init__()
    def forward(self, U, S):
        # Resize S if needed to match U
        if S.shape[2:] != U.shape[2:]:
            S = F.interpolate(S, size=U.shape[2:], mode="bilinear", align_corners=False)
        fused = torch.cat([U, S], dim=1)  # Concatenate along channel dimension
        return fused



class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)

        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)

        scale = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(x_cat))
        return x * scale

#
class FusionBlock1(nn.Module):
    """
    CBAM :
    out = CBAM(x)
    """
    def __init__(self, channels, reduction=16, spatial_kernel=7,u_channels=1):
        super(FusionBlock1, self).__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(spatial_kernel)
        self.LN = nn.LayerNorm([channels])
        self.BN = nn.BatchNorm2d(channels)

    def forward(self, x, S = None):
        # x_max = U.amax(dim=(2,3), keepdim=True)
        # # x_max[-x_min>x_max] = x_min[-x_min>x_max]
        # #
        # U = U/(x_max + 1e-6)

        if S is not None:
            x = torch.cat([x, S], dim=1)

        # x_min = x.amin(dim=(2,3), keepdim=True)
        x = x.permute(0, 2, 3, 1)
        x = self.LN(x)
        x = x.permute(0, 3, 1, 2)

        # x = self.BN(x)


        att = self.channel_att(x)
        att = self.spatial_att(att)
    #
    # def forward(self, U, S = None):
    #     if S is None:
    #         x = U
    #     else:
    #         x = torch.cat([U, S], dim=1)
    #     att = self.channel_att(x)
    #     att = self.spatial_att(att)

        return att   # residual addition

class FusionBlock2(nn.Module):
    """
    Fuse two encoder feature maps using a lightweight ConvAdapter module
    instead of direct concatenation.

    Args:
        in_channels1: number of channels in encoder1 feature map
        in_channels2: number of channels in encoder2 feature map
        out_channels: number of output channels for the fused feature
        reduction: reduction ratio for bottleneck (e.g., 4 → bottleneck_channels = out_channels // 4)
        mode: fusion mode - ['add', 'concat', 'gate']
    """

    def __init__(self, in_channels1, in_channels2, out_channels, reduction=4, mode='concat'):
        super().__init__()
        self.mode = mode

        # Align both encoder feature maps to a shared dimension
        self.proj1 = nn.Conv2d(in_channels1, out_channels, kernel_size=1, bias=False)
        self.proj2 = nn.Conv2d(in_channels2, out_channels, kernel_size=1, bias=False)

        # Bottleneck (ConvAdapter)
        bottleneck_channels = max(out_channels // reduction, 4)
        self.adapter = nn.Sequential(
            nn.Conv2d(2*out_channels, bottleneck_channels, kernel_size=3, padding=1, groups=bottleneck_channels,
                      bias=False),
            # nn.BatchNorm2d(bottleneck_channels),
            # nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False),
            # nn.BatchNorm2d(out_channels)
        )

        # Gating parameter (optional, learnable α)
        self.alpha = nn.Parameter(torch.ones(1))
        self.att = FusionBlock1(channels = in_channels1 + in_channels2 + out_channels)
        # Optional gating mechanism for 'gate' mode
        if mode == 'gate':
            self.gate = nn.Sequential(
                nn.Conv2d(out_channels * 2, out_channels, 1, bias=False),
                nn.Sigmoid()
            )

    def forward(self, feat1, feat2):
        # Ensure same spatial size (ViT might be smaller)
        if feat1.size()[2:] != feat2.size()[2:]:
            feat2 = F.interpolate(feat2, size=feat1.shape[2:], mode='bilinear', align_corners=False)

        f1 = self.proj1(feat1)
        f2 = self.proj2(feat2)

        if self.mode == 'concat':
            fused = torch.cat([f1, f2], dim=1)
            fused = self.adapter(fused)
            fused = torch.cat([fused], dim=1)
            # fused = torch.cat([feat1, feat2, fused], dim=1)
            # fused = self.att(fused)
        elif self.mode == 'gate':
            gate = self.gate(torch.cat([f1, f2], dim=1))
            fused = f1 * gate + f2 * (1 - gate)
            fused = fused + self.alpha * self.adapter(fused)
        else:  # default: 'add'
            fused = (f1 + f2) / 2
            fused = fused + self.alpha * self.adapter(fused)

        return fused



class DualEncoderUNet(nn.Module):
    def __init__(
        self,
        unet_encoder_name="resnet34",
        unet_encoder_weights=None,
        segformer_variant="nvidia/segformer-b2-finetuned-ade-512-512",
        classes=1,
        decoder_channels=(256, 128, 64, 32,16),
        simple_fusion=0,
        regression=False,
        in_channels=3,
        freeze_segformer = False,
        freeze_unet=False,
        input_size=1024,
        decoder_type="unet",
        IgnoreBottleNeck = False,
            cof_seg = 1,
            cof_unet = 1,
            model_depth=5,
            instance_segmentation = False,
    ):
        super().__init__()
        self.classes = classes
        self.cof_seg = cof_seg
        self.cof_unet = cof_unet
        self.freeze_segformer = freeze_segformer
        self.freeze_unet = freeze_unet
        self.model_depth = model_depth
        self.instance_segmentation = instance_segmentation

        ## unet encoder

        if 'convnext' in unet_encoder_name.lower():
            self.unet_encoder = get_timm_encoder(
        name="convnextv2_tiny.fcmae_ft_in22k_in1k",
        in_channels=in_channels,
        depth=model_depth,
        weights=True,
        output_stride=32,
        drop_rate=0.5,
        drop_path_rate=0.25,
    )
        else:
            self.unet_encoder = smp.encoders.get_encoder(
                unet_encoder_name,
                in_channels=in_channels,
                depth=model_depth,
                weights=unet_encoder_weights,
                dropout=0.5,
            )
        u_out_channels = self.unet_encoder.out_channels[1:]  # [64, 64, 128, 256, 512]
        self.IgnoreBottleNeck = IgnoreBottleNeck
        seg_cfg = SegformerConfig.from_pretrained(segformer_variant)
        # seg_cfg.attention_probs_dropout_prob = 0.5
        seg_cfg.output_hidden_states = True

        ## segformer encoder
        self.segformer = SegformerModel.from_pretrained(segformer_variant, config=seg_cfg)
        self.register_buffer('segformer_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('segformer_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        s_expected = list(seg_cfg.hidden_sizes[-(model_depth):])
        # Fusion blocks for first 4 skips (index 0 to 3)
        self.fusions = nn.ModuleList()
        # mid_size = int(input_size/2)
        for i in range(model_depth):  # i = 0,1,2,3 for skips 0..3
            u_ch = u_out_channels[i]
            s_ch = s_expected[i]
            if simple_fusion == 0:
                self.fusions.append(FusionBlock0())
            if simple_fusion == 1:
                self.fusions.append(FusionBlock1(channels = u_ch + s_ch,u_channels=u_ch))
            if simple_fusion == 2:
                self.fusions.append(FusionBlock2(in_channels1=u_ch, in_channels2=s_ch, out_channels=u_ch))

        if simple_fusion == 0:
            encoder_channels_for_decoder = s_expected #+ [u_out_channels[model_depth - 1]]
            for i in range(model_depth):
                encoder_channels_for_decoder[i] = s_expected[i] + u_out_channels[i]
        elif simple_fusion == 1:
            encoder_channels_for_decoder = s_expected #+ [u_out_channels[model_depth - 1]]
            for i in range(model_depth):
                encoder_channels_for_decoder[i] = s_expected[i] + u_out_channels[i]
        elif simple_fusion == 2:
            encoder_channels_for_decoder = s_expected
            for i in range(4):
                # encoder_channels_for_decoder[i] = s_expected[i] + 2*u_out_channels[i]
                encoder_channels_for_decoder[i] = u_out_channels[i]

        # Decoder expects 5 skips: 4 fused + 1 bottleneck (last unet encoder output)

        # ----- choose decoder type -----
        if decoder_type.lower() == "unet":
            DecoderClass = UnetDecoder
        elif decoder_type.lower() in ("unet++", "unetplusplus"):
            DecoderClass = UnetPlusPlusDecoder
        else:
            raise ValueError(f"Unknown decoder_type '{decoder_type}'. Use 'unet' or 'unet++'.")

        if self.IgnoreBottleNeck:
            encoder_channels_for_decoder = encoder_channels_for_decoder  # keep first 4 (skip0..skip3)
            decoder_channels = decoder_channels
            print('skip')
            n_blocks = model_depth
        else:
            n_blocks = model_depth

        self.dropout = nn.Dropout2d(p=0.0)




        if self.instance_segmentation:
            # Regression hv head
            self.segmentation_head1 = smp.base.SegmentationHead(
                in_channels=decoder_channels[-1],
                out_channels=5,  # instance channels
                activation=None,
                kernel_size=1,
            )
            self.decoder1 = DecoderClass(
                encoder_channels=[in_channels] + encoder_channels_for_decoder,
                decoder_channels=decoder_channels[0:model_depth],
                n_blocks=n_blocks,
                use_batchnorm=False,
                IgnoreBottleNeck=self.IgnoreBottleNeck
            )

        self.decoder2 = DecoderClass(
            encoder_channels=[in_channels] + encoder_channels_for_decoder,
            decoder_channels=decoder_channels[0:model_depth],
            n_blocks=n_blocks,
            use_batchnorm=True,
            IgnoreBottleNeck=self.IgnoreBottleNeck
        )

        self.segmentation_head2 = smp.base.SegmentationHead(
            # in_channels=decoder_channels[0:model_depth][-1],
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=None,
            kernel_size=1,
        )

        # self.segmentation_head1 = nn.Sequential(
        #     nn.Conv2d(decoder_channels[0:model_depth][-1], 1, kernel_size=3, padding=1),
        #     nn.ReLU()
        # )
        # # Binary segmentation head
        # self.segmentation_head2 = nn.Conv2d(
        #     decoder_channels[0:model_depth][-1],
        #     5,
        #     kernel_size=3,
        #     padding=1
        # )
        # No activation here — leave logits for the loss function

    def _filter_and_sort_unet_feats(self, u_feats, input_h):
        filtered = [f for f in u_feats if f.shape[2] < input_h]
        filtered = sorted(filtered, key=lambda t: t.shape[2], reverse=True)
        return filtered

    def _sort_segf_feats(self, s_feats):
        return sorted(s_feats, key=lambda t: t.shape[2], reverse=True)

    def forward(self, x, debug_print_shapes=False):
        cof_seg = self.cof_seg
        cof_unet = self.cof_unet

        B, C, H_in, W_in = x.shape


        ### segformer forward pass
        # Normalize first 3 channels for segformer input
        # Handle possible extra channels (like depth or others)
        x_for_segformer = cof_seg*x[:, :3, :, :]
        x_for_segformer = (x_for_segformer - self.segformer_mean) / self.segformer_std
        s_all = self.segformer(pixel_values=cof_seg*x_for_segformer).hidden_states
        s_feats = s_all[-(self.model_depth):]
        s_feats = sorted(s_feats, key=lambda t: t.shape[2], reverse=True)



        # resnet forward pass
        u_feats_all = self.unet_encoder(cof_unet*x_for_segformer)
        u_feats = self._filter_and_sort_unet_feats(u_feats_all, H_in)




        skips = []
        skips.append(x)
        # Fuse SegFormer with first 4 U-Net skips
        for i in range(self.model_depth):
            U = u_feats[i]
            S = s_feats[i]
            # if (S.shape[2] != U.shape[2]) or (S.shape[3] != U.shape[3]):
            # S = F.interpolate(S, size=(2*U.shape[2], 2*U.shape[3]), mode="bilinear", align_corners=False)
            # U = F.interpolate(U, size=(2*U.shape[2], 2*U.shape[3]), mode="bilinear", align_corners=False)
            if debug_print_shapes:
                print(f"Fusing skip {i} shapes: U{tuple(U.shape)} S(resized) {tuple(S.shape)}")
            U = self.dropout(U)
            S = self.dropout(S)
            fused = self.fusions[i](U, S)

            skips.append(fused)

        # Add bottleneck skip (last U-Net encoder output) without fusion
        # bottleneck = cof_unet*u_feats_all[-1]
        # if self.IgnoreBottleNeck:
        #     bottleneck = torch.empty([2,0,bottleneck.shape[2],bottleneck.shape[3]], device=bottleneck.device)
        # skips.append(bottleneck)

        if debug_print_shapes:
            print("== Skip tensors provided to decoder ==")
            for i, s in enumerate(skips, start=1):
                print(f"skip {i}: shape {tuple(s.shape)}")

        dec_out2 = self.decoder2(skips)
        # print(f"Decoder output shape before segmentation_head: {dec_out.shape}")

        out_binary = self.segmentation_head2(dec_out2)

        if self.instance_segmentation:
            dec_out1 = self.decoder1(skips)
            out_hv = self.segmentation_head1(dec_out1)
            out = torch.cat([out_hv, out_binary], dim=1)
            return out
        else:
            return out_binary

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualEncoderUNet(
                            unet_encoder_weights="imagenet",
                            unet_encoder_name='convnext',
                            segformer_variant="nvidia/segformer-b2-finetuned-ade-512-512",
                            model_depth=4,
        cof_seg=0,
                            simple_fusion = 1,
            IgnoreBottleNeck=True,
                            ).to('cpu')

    # print(model)  # print network structure

    x = torch.randn(1, 3, 1024, 1024).to('cpu')

    with torch.no_grad():
        out = model(x, debug_print_shapes=True)
