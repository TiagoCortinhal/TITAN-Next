# TITAN-Next
## Depth- and semantics-aware multi-modal domain translation: Generating 3D panoramic color images from LiDAR point clouds

[![YouTube Demo](https://img.shields.io/badge/YouTube-Demo-red)](https://www.youtube.com/watch?v=eV510t29TAc)

**TITAN-Next** builds upon the foundation of our previous work, [TITAN-Net](https://github.com/TiagoCortinhal/TITAN-Net), enhancing its capabilities to achieve state-of-the-art results in multi-modal domain translation by transforming raw 3D LiDAR point clouds into detailed RGB-D panoramic images. This unique framework leverages depth and semantic segmentation, offering groundbreaking applications for autonomous vehicles and beyond.

📄 **[Read the Full Paper](https://pdf.sciencedirectassets.com/271599/1-s2.0-S0921889023X00116/1-s2.0-S0921889023002221)**

## 🌟 Key Features

- **Cross-Modal Translation**: Converts sparse LiDAR data to RGB-D images with added semantic depth and texture.
- **Multi-Scale Feature Aggregation**: Enhances accuracy and robustness by incorporating a feature pyramid module.
- **Depth Head Integration**: Delivers a complete 3D perception by estimating depth from LiDAR input, achieving impressive performance boosts.
- **Fail-Safe Mechanism**: Supports autonomous systems by synthesizing RGB-D images when camera data is unavailable.
- **Benchmark Performance**: Outperforms baselines by 23.7% in mean IoU on the Semantic-KITTI dataset.

## 🔗 Pretrained Models

This project utilizes **SalsaNext** for LiDAR segmentation. You can download the pretrained weights for SalsaNext here:

- [SalsaNext Weights](https://github.com/TiagoCortinhal/SalsaNext)

Additionally, the pretrained weights for **TITAN-Next** are available for download:

- [TITAN-Next Weights](https://drive.google.com/file/d/1NQyvevDiJ1Jo00Ywstgu3YLQ6yfVmUO5/view?usp=sharing)

## 🧠 Methodology

TITAN-Next is a modular framework that combines **semantic segmentation** and **depth estimation**. The key components include:

- **LiDAR to Semantic Segments**: Using SalsaNext, we segment LiDAR point clouds and project them into 2D.
- **Camera Image Segmentation**: SD-Net segments RGB images, aiding in translating LiDAR into the camera domain.
- **RGB-D Generation**: The Vid2Vid model synthesizes high-quality RGB images from the semantic maps.


## 📊 Results

TITAN-Next has been extensively evaluated on the Semantic-KITTI dataset, achieving significant improvements in:

- **Mean Intersection-over-Union (IoU)**: +23.7% over TITAN-Net
- **Fréchet Inception Distance (FID)** and **Sliced Wasserstein Distance (SWD)**: Demonstrates superior image fidelity and depth realism compared to other baselines.

### Sample Results

| Model         | Mean IoU (%) | FID ↓  | SWD ↓ |
|---------------|--------------|--------|-------|
| Pix2Pix       | 12.5         | 261.28 | 2.59  |
| [TITAN-Net](https://github.com/TiagoCortinhal/TITAN-Net) | 31.1         | 61.91  | 2.38  |
| **TITAN-Next** | **54.8**     | **29.56** | **1.82** |

## 🏆 Citation

If you use TITAN-Next in your research, please consider citing:

```bibtex
@article{Cortinhal2024,
  title={Depth- and Semantics-Aware Multi-Modal Domain Translation: Generating 3D Panoramic Color Images from LiDAR Point Clouds},
  author={Tiago Cortinhal and Eren Erdal Aksoy},
  journal={Robotics and Autonomous Systems},
  year={2024},
  volume={171},
  pages={104583},
  doi={10.1016/j.robot.2023.104583}
}
