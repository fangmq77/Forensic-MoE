# Forensic-MoE
The code implementation of paper **Forensic-MoE: Exploring Comprehensive Synthetic Image Detection Traces with Mixture of Experts**
## Abstract
Recently, synthetic images have evolved incredibly realistic with the development of generative techniques. To avoid the spread of misinformation and identify synthetic content, research on synthetic image detection becomes urgent. Unfortunately, limited to the singular forensic perspective, existing methods struggle to explore sufficient traces encountered with diverse synthetic techniques. In response to this, we argue that different synthetic images encompass a variety of forensic traces, and utilizing multiple experts to explore traces from diverse perspectives will be beneficial. Accordingly, a novel detector with the Mixture of multiple forensic Experts is proposed, named Forensic-MoE. To integrate multiple experts and enhance the knowledge interaction, Forensic-MoE follows an adapter-backbone architecture. Specifically, multiple adapters trained on different synthetic images serve as the trace exploration experts, and they are uniformly integrated into a pretrained backbone model to learn the detection prior and encourage the expert interaction. By guiding multiple experts to align with each other and collaborate together, Forensic-MoE can integrate comprehensive and discriminative detection traces from multiple perspectives. Moreover, for the discrimination improvement of each expert, a multi-stage structure is proposed for efficient trace perception, and a patch decentralization strategy is applied to encourage the model’s attention on every local region.

## Environment Setup
Since the DWT transformation is required in Forensic-MoE, additional data processing tools are necessary.
```down-python
pip install pytorch_wavelets PyWavelets
```

## Get Start

### Download the Pretrained Model
The pretrained checkpoints file are available in [google drive](https://drive.google.com/drive/folders/1DAgT2XgUIERMTPI6YejC0bbP8ZIgk0lr?usp=sharing). Among them, `CLIP.pt` represents the official clip pretrained checkpoint, and `detector.pth` represents the weights of our Forensic-MoE.

Please download both of them and place in the `./checkpoints` folder.

### Inference
You can run the `inference.py` to simply verify the authenticity of the specified image:

```down-python
python inference.py --model-path ./checkpoints/detector.pth --image-path ./examples/midjourney.png
```

after that, the expect output is:

```down-python
This image is FAKE with fakeness score 0.6492539644241333
```

The default threshold is 0.5.

## Results
The provided `detector.pth` is a re-trained version that outperforms the result reported in our original paper, especially in several popular generative models like Midjourney.
| Test Set Acc  | Paper  | This Implementation  | Difference|
|------|------|------|------|
| ProGAN | 99.96 | 99.96 |0 |
| StyleGAN | 91.20 | 94.99 |+3.79 |
| StyleGAN2 | 91.23 | 91.69 |+0.46 |
| BigGAN^1 | 95.35 | 95.7 |+0.35 |
| BigGAN^2 | 99.41 | 99.49 |+0.08 |
| CycleGAN | 99.36 | 99.09 |-0.27 |
| StarGAN | 99.37 | 98.67 |-0.7 |
| GauGAN | 95.35 | 95.59 |+0.24 |
| WFIR | 98.10 | 98.65 |+0.55 |
| Midjourney | 72.63 | 78.05 |+5.42 |
| SDv1.4 | 99.67 | 99.71 |+0.04 |
| SDv1.5| 99.44 | 99.49 |+0.05 |
| ADM | 87.04 | 83.24 |-3.8 |
| GLIDE | 97.97 | 97.22 |-0.75 |
| Wukong | 99.21 | 99.37 |+0.16 |
| VQDM | 96.89 | 99.60 |-0.29 |
| Kandinsky 2 | 88.14 | 90.19 |+2.05 |
| Kandinsky 3 | 79.04 | 85.91 |+6.87 |
| PixArt-alpha | 79.85 | 85.17 |+5.32 |
| Playground 2.5 | 80.54 | 84.46 |+3.92 |
| Wurstchen 2 | 94.36 | 93.15 |-1.21 |
|**Mean**|**92.58**|**93.64**| **+1.06**|

## Citation
If you find Forensic-MoE useful for you, we would greatly appreciate it if you could cite this paper:
```down-python
@InProceedings{Fang_2025_ICCV,
    author    = {Fang, Mingqi and Li, Ziguang and Yu, Lingyun and Yang, Quanwei and Xie, Hongtao and Zhang, Yongdong},
    title     = {Forensic-MoE: Exploring Comprehensive Synthetic Image Detection Traces with Mixture of Experts},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025},
    pages     = {17772-17782}
}
```

