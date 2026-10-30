from model2 import TDL
from loaders import get_loaders
from eval import Evaluation
from losses import CombinedLoss
import torch


def main():
    device = torch.device("cuda:0")
    ckpt_path = '/data/ljc/deepfake/detection/OWN/ckpt6/avdeepfake1m_base_lrs3_iter4_512_768_128_15_e22_av-va-aa-vv_r2112_r2_focal-diou-rec.pth'
    print(f"Task:tfl Train Dataset:avdeepfake1m Test Dataset:avdeepfake1m")

    test_loader = get_loaders(
        dataset_name='avdeepfake1m',
        av_hubert_name='base_lrs3_iter4',
        max_length=512,
        batch_size=64,
        workers=4,
        splits=["test"],
    )

    model_config = {
        "input_dimension": 768,
        "d_model": 128,
        "win_size": 15,
        "encoder_nlayers": {
            "retain": 2,
            "downsample": 2
        },
        "reconstruction_modality": ["av", "va", "aa", "vv"],
        "reconstruction_nlayers": {"pre": 2, "downsample": 1, "upsample": 1, "post": 2},
        "reconstruction_error_encoder_nlayers": 2
    }

    model = TDL(
        input_dimension=768,
        max_length=512,
        d_model=128,
        win_size=15,
        encoder_nlayers={
            "retain": 2,
            "downsample": 2
        },
        reconstruction_modality=["av", "va", "aa", "vv"],
        reconstruction_nlayers={"pre": 2, "downsample": 1, "upsample": 1, "post": 2},
        reconstruction_error_encoder_nlayers=2,
    )
    model.to(device)

    model.load_state_dict(torch.load(ckpt_path))
    factor = [1] * model_config["encoder_nlayers"]["retain"] + [2 ** (i + 1) for i in range(model_config["encoder_nlayers"]["downsample"])]
    e = Evaluation(
        model=model,
        loader=test_loader['test'],
        criterion=CombinedLoss(composition=["focal", "diou", "rec"], factor=factor),
        device=device,
        factor=factor,
        max_length=512,
        dataset_name='avdeepfake1m',
        task='tfl',
    )
    e.compute_metrics()
    performance = {f"t{x}@{y}": 100 * e.metrics[x][y] for x in ["ap", "ar"] for y in e.metrics[x]}
    print(performance)


if __name__ == '__main__':
    main()
