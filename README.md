# Exposing Speech Coupling Collapse : Multi-Scale Audio-Visual Reconstruction for Temporal Deepfake Localization

[*Weight:https://huggingface.co/1129ljc/MSAVR-TDL/tree/main*](https://huggingface.co/1129ljc/MSAVR-TDL/tree/main) 

> Temporal deepfake localization aims to localize fake temporal segments. 
> Current detectors directly adopt compressed audio-visual embeddings extracted from upstream pre-trained models, lacking an interpretable account of why anomalous representations emerge in temporal deepfake segments. 
> Furthermore, most detectors localize tampering from single-scale features, which exhibit reduced accuracy at strict temporal IoU thresholds. 
> In this paper, we point out a measurable property of real speech. 
> A speaker's voice and face originate from a common source. 
> However, this intrinsic coupling breaks down because generators synthesize audio and face separately in temporal deepfake segments. 
> We define this phenomenon as Speech Coupling Collapse and translate it into a precise localization signal by employing a multi-scale audio-visual reconstruction framework. 
> Specifically, a shared 1D encoder produces multi-scales of audio and visual features. 
> At each scale, four reconstruction mappings (audio→video, video→audio, audio→audio, video→video) trained on real frames only recover the natural coupling pattern, turning each scale into a scale-conditional anomaly detector. 
> A 1D feature pyramid fuses the residuals top-down so that coarse-scale confidence guides fine-scale boundary regression. 
> Finally, per-scale heads jointly predict frame-level forgery probabilities and segment boundaries. 
> Our method achieves state-of-the-art video-level detection and temporal localization performance on LAV-DF, AV-Deepfake1M and AV-Deepfake1M++. 
> In particular, on LAV-DF the average precision at a temporal IoU threshold of 0.95 (AP@0.95) improves by 6.1% over the strongest baseline, and on AV-Deepfake1M++ the AP@0.75 improves by 9.12%. 
> Cross-dataset testing and perturbation experiments further confirm the generalizability and robustness of our method.

## Installation
```bash
conda create -n msavr_tdl python=3.10.19
conda activate msavr_tdl
# Install torch
pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu126
# Install the dependencies
pip install -r requirements.txt
```
##  Preprocess
Download the [AV-Deepfake1m](https://huggingface.co/datasets/ControlNet/AV-Deepfake1M/tree/main) dataset.
```bash
hf download ControlNet/AV-Deepfake1M --repo-type dataset --local-dir /.../AV-Deepfake1M/
```
Extracting audios from speaking videos. FFmpeg needs to be installed.
```bash
cd preprocess
python preprocess_audio.py
```
Extract lip movements from the speaking videos. You need 
[shape_predictor_68_face_landmarks.dat](https://huggingface.co/1129ljc/MSAVR-TDL/tree/main/AV-HuBERT/shape_predictor_68_face_landmarks.dat)
and [20words_mean_face.npy](https://huggingface.co/1129ljc/MSAVR-TDL/tree/main/AV-HuBERT/20words_mean_face.npy).
```bash
cd preprocess
python preprocess_video.py
```
##  AV-HuBERT Embedding
Extract AV-HuBERT embeddings from the audio and lip movements. You need
[base_lrs3_iter4.pt](https://huggingface.co/1129ljc/MSAVR-TDL/tree/main/AV-HuBERT/).
```bash
cd AV-HuBERT
python feature_extraction.py
```
##  Train
Train on AV-Deepfake1m AV-HuBERT embeddings. You need 
[AV-Deepfake1m protocols](https://huggingface.co/1129ljc/MSAVR-TDL/tree/main/protocols/AV-Deepfake1m/).
```bash
cd train
python train_avdeepfake1m.py
```
## Test
Test on AV-Deepfake1m AV-HuBERT embeddings. We have released the training 
[weights](https://huggingface.co/1129ljc/MSAVR-TDL/tree/main/weights/avdeepfake1m/)
for the AV-Deepfake1m dataset.
```bash
cd train
python test_avdeepfake1m.py
```
## Acknowledgements

This repo benefits from 
[LAV-DF](https://github.com/ControlNet/LAV-DF),
[UMMAFormer](https://github.com/ymhzyj/UMMAFormer),
[dimodif](https://github.com/mever-team/dimodif),
[auvire](https://github.com/mever-team/auvire). Thanks for their great works!