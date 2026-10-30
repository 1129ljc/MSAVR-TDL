from training import Experiment

DEFAULT_AVDEEPFAKE1M = {
    "device": "cuda:0",
    "seed": 0,
    "epochs": 20,
    "dataset_name": "avdeepfake1m",
    "av_hubert_name": "base_lrs3_iter4",
    "max_length": 512,
    "batch_size": 64,
    "workers": 4,
    "model": {
        "input_dimension": 768,
        "d_model": 128,
        "win_size": 15,
        "encoder_nlayers": {
            "retain": 2,
            "downsample": 2
        },
        "reconstruction_modality": ["av", "va", "aa", "vv"],
        "reconstruction_nlayers": {
            "pre": 2,
            "downsample": 1,
            "upsample": 1,
            "post": 2
        },
        "reconstruction_error_encoder_nlayers": 2
    },
    "criterion": ["focal", "diou", "rec"],
    "learning_rate": 0.001,
    "save_dir": '/data/ljc/deepfake/detection/OWN/ckpt/'
}

e = Experiment(cfg=DEFAULT_AVDEEPFAKE1M)
e.run()
