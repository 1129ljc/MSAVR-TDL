import torch
from torch import nn

RECONSTRUCTION = {
    "av": {"input": "audio", "output": "video"},
    "va": {"input": "video", "output": "audio"},
    "vv": {"input": "video", "output": "video"},
    "aa": {"input": "audio", "output": "audio"},
}


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, timesteps):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.LayerNorm([out_channels, timesteps])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, timesteps):
        super(ConvTransposeBlock, self).__init__()
        self.deconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding=1)
        self.norm = nn.LayerNorm([out_channels, timesteps])
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(x)
        x = self.relu(x)
        return x


class Projection(nn.Module):
    def __init__(self, nlayers, in_channels, out_channels, timesteps, pre):
        super().__init__()
        self.projection = nn.Sequential(
            *nn.ModuleList(
                [
                    ConvBlock(
                        in_channels=in_channels if not pre or (pre and i == 0) else out_channels,
                        out_channels=out_channels if pre or (not pre and i == nlayers - 1) else in_channels,
                        kernel_size=3,
                        stride=1,
                        padding="same",
                        timesteps=timesteps,
                    )
                    for i in range(nlayers)
                ]
            )
        )

    def forward(self, x):
        return self.projection(x)


class ReconstructionModule(nn.Module):
    def __init__(self, input_dimension, d_model, win_size, max_length, nlayers):
        super().__init__()
        self.projection_pre = Projection(
            nlayers=nlayers["pre"],
            in_channels=input_dimension,
            out_channels=d_model,
            timesteps=max_length,
            pre=True,
        )
        self.projection_post = Projection(
            nlayers=nlayers["post"],
            in_channels=d_model,
            out_channels=input_dimension,
            timesteps=max_length,
            pre=False,
        )
        self.encoder = nn.Sequential(
            *nn.ModuleList(
                [
                    ConvBlock(
                        in_channels=d_model,
                        out_channels=d_model,
                        kernel_size=win_size,
                        stride=2,
                        padding=win_size // 2,
                        timesteps=max_length // 2 ** (i + 1),
                    )
                    for i in range(nlayers["downsample"])
                ]
            )
        )
        self.decoder = nn.Sequential(
            *nn.ModuleList(
                [
                    ConvTransposeBlock(
                        in_channels=d_model,
                        out_channels=d_model,
                        kernel_size=win_size,
                        stride=2,
                        padding=win_size // 2,
                        timesteps=max_length // 2 ** (nlayers["upsample"] - i - 1),
                    )
                    for i in range(nlayers["upsample"])
                ]
            )
        )

    def forward(self, x):
        x = self.projection_pre(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.projection_post(x)
        return x


class EncoderModule(nn.Module):
    def __init__(self, input_dimension, d_model, win_size, max_length, encoder_nlayers):
        super().__init__()
        self.model = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=input_dimension if i == 0 else d_model,
                    out_channels=d_model,
                    kernel_size=win_size,
                    stride=1,
                    padding="same",
                    timesteps=max_length
                )
                for i in range(encoder_nlayers["retain"])
            ]
            + [
                ConvBlock(
                    in_channels=d_model,
                    out_channels=d_model,
                    kernel_size=win_size,
                    stride=2,
                    padding=win_size // 2,
                    timesteps=max_length // 2 ** (i + 1)
                )
                for i in range(encoder_nlayers["downsample"])
            ]
        )

    def forward(self, x):
        multi_scale_features = []
        for item_model in self.model:
            x = item_model(x)
            multi_scale_features.append(x)
        return multi_scale_features


class CrossModalReconstructionModel(nn.Module):
    def __init__(self, reconstruction_modality, d_model, win_size, max_length, reconstruction_nlayers, reconstruction_error_encoder_nlayers):
        super().__init__()
        self.reconstruction_modality = reconstruction_modality
        self.reconstruction_model = ReconstructionModule(
            input_dimension=d_model,
            d_model=d_model,
            win_size=win_size,
            max_length=max_length,
            nlayers=reconstruction_nlayers
        )
        self.reconstruction_error_encoder = nn.Sequential(
            *nn.ModuleList(
                [
                    ConvBlock(
                        in_channels=d_model * len(self.reconstruction_modality) if i == 0 else d_model,
                        out_channels=d_model,
                        kernel_size=win_size,
                        stride=1,
                        padding="same",
                        timesteps=max_length
                    )
                    for i in range(reconstruction_error_encoder_nlayers)
                ]
            )
        )

    def forward(self, video_feature, audio_feature):
        reconstruction_error = []
        dissimilarity = []
        for item in range(len(self.reconstruction_modality)):
            modality = dict(self.reconstruction_modality[item])
            if modality["input"] == "audio" and modality["output"] == "audio":
                reconstruction_error_item = audio_feature - self.reconstruction_model(audio_feature)
                reconstruction_error.append(reconstruction_error_item)
                dissimilarity.append(torch.abs(reconstruction_error_item))
            elif modality["input"] == "audio" and modality["output"] == "video":
                reconstruction_error_item = video_feature - self.reconstruction_model(audio_feature)
                reconstruction_error.append(reconstruction_error_item)
                dissimilarity.append(torch.abs(reconstruction_error_item))
            elif modality["input"] == "video" and modality["output"] == "audio":
                reconstruction_error_item = audio_feature - self.reconstruction_model(video_feature)
                reconstruction_error.append(reconstruction_error_item)
                dissimilarity.append(torch.abs(reconstruction_error_item))
            elif modality["input"] == "video" and modality["output"] == "video":
                reconstruction_error_item = video_feature - self.reconstruction_model(video_feature)
                reconstruction_error.append(reconstruction_error_item)
                dissimilarity.append(torch.abs(reconstruction_error_item))

        reconstruction_error = torch.cat(reconstruction_error, dim=1)
        reconstruction_error = self.reconstruction_error_encoder(reconstruction_error)
        dissimilarity = torch.stack(dissimilarity, dim=0).sum(dim=0).mean(dim=1)
        return reconstruction_error, dissimilarity


class MutiScaleCrossModalReconstructionModule(nn.Module):
    def __init__(self, scale_length_list, reconstruction_modality, d_model, win_size, reconstruction_nlayers, reconstruction_error_encoder_nlayers):
        super().__init__()
        self.scale_num = len(scale_length_list)
        self.model = nn.ModuleList(
            [
                CrossModalReconstructionModel(
                    reconstruction_modality=reconstruction_modality,
                    d_model=d_model,
                    win_size=win_size,
                    max_length=scale_length_list[i],
                    reconstruction_nlayers=reconstruction_nlayers,
                    reconstruction_error_encoder_nlayers=reconstruction_error_encoder_nlayers
                )
                for i in range(self.scale_num)
            ]
        )

    def forward(self, multi_scale_video_features, multi_scale_audio_features):
        multi_scale_reconstruction_error_features = []
        multi_scale_dissimilarity_features = []
        for i in range(self.scale_num):
            reconstruction_error, dissimilarity = self.model[i](multi_scale_video_features[i], multi_scale_audio_features[i])
            multi_scale_dissimilarity_features.append(dissimilarity)
            multi_scale_reconstruction_error_features.append(reconstruction_error)
        return multi_scale_reconstruction_error_features, multi_scale_dissimilarity_features


class FeaturePyramid1DModule(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()

        for in_channels in in_channels_list:
            inner_block_module = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0)
            layer_block_module = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

    def get_result_from_inner_blocks(self, x, idx):
        num_blocks = len(self.inner_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.inner_blocks):
            if i == idx:
                out = module(x)
        return out

    def get_result_from_layer_blocks(self, x, idx):
        num_blocks = len(self.layer_blocks)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.layer_blocks):
            if i == idx:
                out = module(x)
        return out

    def forward(self, x):
        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-1]
            inner_top_down = torch.nn.functional.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        return results


class HeadModule(nn.Module):
    def __init__(self, d_model, output_size, timesteps, final_relu):
        super().__init__()
        submodules = [
            ConvBlock(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=3,
                stride=1,
                padding="same",
                timesteps=timesteps
            ),
            ConvBlock(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=3,
                stride=1,
                padding="same",
                timesteps=timesteps
            ),
            nn.Conv1d(
                in_channels=d_model,
                out_channels=output_size,
                kernel_size=3,
                padding="same",
            ),
        ]
        if final_relu:
            submodules.append(nn.ReLU())
        self.module = nn.Sequential(*submodules)

    def forward(self, x):
        return self.module(x)


class HeadModel(nn.Module):
    def __init__(self, d_model, factor, output_size, max_length, final_relu):
        super().__init__()
        self.model = nn.ModuleList(
            [
                HeadModule(
                    d_model=d_model,
                    output_size=output_size,
                    timesteps=max_length // f,
                    final_relu=final_relu,
                )
                for f in factor
            ]
        )

    def forward(self, x):
        return [self.model[i](f) for i, f in enumerate(x)]


class TDL(nn.Module):
    def __init__(self, input_dimension, max_length, d_model, win_size, encoder_nlayers, reconstruction_modality, reconstruction_nlayers, reconstruction_error_encoder_nlayers):
        super().__init__()
        self.input_dimension = input_dimension
        self.max_length = max_length
        self.d_model = d_model
        self.win_size = win_size
        self.encoder_nlayers = encoder_nlayers
        self.reconstruction_modality = reconstruction_modality
        self.reconstruction_nlayers = reconstruction_nlayers

        # EncoderModule
        self.encoder_model = EncoderModule(
            input_dimension=self.input_dimension,
            d_model=self.d_model,
            win_size=self.win_size,
            max_length=self.max_length,
            encoder_nlayers=self.encoder_nlayers
        )

        # MutiScaleReconstructionModule
        self.factor = [1] * self.encoder_nlayers["retain"] + [2 ** (i + 1) for i in range(self.encoder_nlayers["downsample"])]
        self.muti_scale_cross_modal_reconstruction_model = MutiScaleCrossModalReconstructionModule(
            scale_length_list=[int(max_length / i) for i in self.factor],
            reconstruction_modality=[RECONSTRUCTION[m] for m in reconstruction_modality],
            d_model=self.d_model,
            win_size=self.win_size,
            reconstruction_nlayers=self.reconstruction_nlayers,
            reconstruction_error_encoder_nlayers=reconstruction_error_encoder_nlayers,
        )

        # FeaturePyramidNet1D
        self.feature_pyramid_1d_model = FeaturePyramid1DModule(
            in_channels_list=[d_model] * (encoder_nlayers["retain"] + encoder_nlayers["downsample"]),
            out_channels=d_model
        )

        # classification
        self.classification = HeadModel(
            d_model=self.d_model,
            factor=self.factor,
            output_size=1,
            max_length=self.max_length,
            final_relu=False,
        )

        # regression
        self.regression = HeadModel(
            d_model=self.d_model,
            factor=self.factor,
            output_size=2,
            max_length=self.max_length,
            final_relu=True,
        )

    def forward(self, video, audio):
        video = video.permute((0, 2, 1))
        audio = audio.permute((0, 2, 1))
        multi_scale_video_features = self.encoder_model(video)
        multi_scale_audio_features = self.encoder_model(audio)
        multi_scale_reconstruction_error_features, dissimilarities = \
            self.muti_scale_cross_modal_reconstruction_model(multi_scale_video_features, multi_scale_audio_features)
        multi_scale_pyramid_features = self.feature_pyramid_1d_model(multi_scale_reconstruction_error_features)
        logits = self.classification(multi_scale_pyramid_features)
        boundaries = self.regression(multi_scale_pyramid_features)
        outputs = []
        for logit, boundary in zip(logits, boundaries):
            output = torch.cat((logit, boundary), dim=1)
            output = torch.permute(output, (0, 2, 1))
            outputs.append(output)
        return outputs, dissimilarities


if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = TDL(
        input_dimension=768,
        max_length=512,
        d_model=128,
        win_size=15,
        encoder_nlayers={"retain": 2, "downsample": 2},
        reconstruction_modality=["av", "va", "aa", "vv"],
        reconstruction_nlayers={"pre": 2, "downsample": 1, "upsample": 1, "post": 2},
        reconstruction_error_encoder_nlayers=2,
    ).to(device)
    video_features = torch.rand(size=[64, 512, 768]).to(device)
    audio_features = torch.rand(size=[64, 512, 768]).to(device)
    outputs, dissimilarities = model(video_features, audio_features)
    print('outputs[0].shape :', outputs[0].shape)
    print('outputs[1].shape :', outputs[1].shape)
    print('outputs[2].shape :', outputs[2].shape)
    print('outputs[3].shape :', outputs[3].shape)
    print('dissimilarities[0].shape :', dissimilarities[0].shape)
    print('dissimilarities[1].shape :', dissimilarities[1].shape)
    print('dissimilarities[2].shape :', dissimilarities[2].shape)
    print('dissimilarities[3].shape :', dissimilarities[3].shape)
