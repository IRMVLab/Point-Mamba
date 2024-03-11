<div  align="center">    
 <img src="./figure/Arc.png" width = ""  align=center />
</div>


<div align="center">
<h1>Point Mamba</h1>
<h3>A Novel Point Cloud Backbone Based on State Space Model with Octree-Based Ordering Strategy</h3>


## Abstact


<div align="left">
 Recently, state space model (SSM) has gained great attention due to its promising performance, linear complexity, and long sequence modeling ability in both language and image domains. However, it is non-trivial to extend SSM to the point cloud field, because of the causality requirement of SSM and the disorder and irregularity nature of point clouds. In this paper, we propose a novel SSM-based point cloud processing backbone, named Point Mamba, with a causality-aware ordering mechanism. To construct the causal dependency relationship, we design an octree-based ordering strategy on raw irregular points, globally sorting points in a z-order sequence and also retaining their spatial proximity. Our method achieves state-of-the-art performance compared with transformer-based counterparts, with 93.4\% accuracy and 75.7 mIOU respectively on the ModelNet40 classification dataset and ScanNet semantic segmentation dataset. Furthermore, our Point Mamba has linear complexity, which is more efficient than transformer-based methods. Our method demonstrates the great potential that SSM can serve as a generic backbone in point cloud understanding.


<div align="left">

## 1. Environment

The code has been tested on Ubuntu 20.04 with 3 Nvidia 4090 GPUs (24GB memory).

1. Python 3.10.13
    ```bash
    conda create -n your_env_name python=3.10.13
    ```

2. Install torch 2.1.1 + cu118

    ```bash
    pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
    ```

3. Clone this repository and install the requirements.

    ```bash
    pip install -r requirements.txt
    ```

4. Install the library for octree-based depthwise convolution.

    ```bash
    git clone https://github.com/octree-nn/dwconv.git
    pip install ./dwconv
    ```

5. Install ``causal-conv1d`` and ``mamba``
    ```bash
    pip install -e causal-conv1d
    pip install -e mamba
    ```

## 2. ScanNet Segmentation

1. **Data**: Download the data from the
   [ScanNet benchmark](https://kaldir.vc.in.tum.de/scannet_benchmark/).
   Unzip the data and place it to the folder <scannet_folder>. Run the following
   command to prepare the dataset.

    ```bash
    python tools/seg_scannet.py --run process_scannet --path_in scannet
    ```
    The filelist should be like this:
    ```bash

    ├── scannet
    │ ├── scans
    │ │ ├── [scene_id]
    │ │ │ ├── [scene_id].aggregation.json
    │ │ │ ├── [scene_id].txt
    │ │ │ ├── [scene_id]_vh_clean.aggregation.json
    │ │ │ ├── [scene_id]_vh_clean.segs.json
    │ │ │ ├── [scene_id]_vh_clean_2.0.010000.segs.json
    │ │ │ ├── [scene_id]_vh_clean_2.labels.ply
    │ │ │ ├── [scene_id]_vh_clean_2.ply
    │ ├── scans_test
    │ │ ├── [scene_id]
    │ │ │ ├── [scene_id].aggregation.json
    │ │ │ ├── [scene_id].txt
    │ │ │ ├── [scene_id]_vh_clean.aggregation.json
    │ │ │ ├── [scene_id]_vh_clean.segs.json
    │ │ │ ├── [scene_id]_vh_clean_2.0.010000.segs.json
    │ │ │ ├── [scene_id]_vh_clean_2.ply
    │ ├── scannetv2-labels.combined.tsv
    ```

2. **Train**: Run the following command to train the network with 3 GPUs and port 10001. The mIoU on the validation set without voting is 75.0. And the training log and weights can be downloaded
   <!-- [here](https://1drv.ms/u/s!Ago-xIr0OR2-gRrV35QGxnHJR4ku?e=ZXRqV7). -->

    ```bash
    python scripts/run_seg_scannet.py --gpu 0,1,2 --alias scannet --port 10001
    ```

3. **Evaluate**: Run the following command to get the per-point predictions for the validation dataset with a voting strategy. And after voting, the mIoU is 75.7 on the validation dataset.

    ```bash
    python scripts/run_seg_scannet.py --gpu 0 --alias scannet --run validate
    ```
## 3. ModelNet40 Classification (Point Mamba(O))

1. **Data**: Run the following command to prepare the dataset.

    ```bash
    python tools/cls_modelnet.py
    ```

2. **Train**: Run the following command to train the network with 1 GPU. The classification accuracy on the testing set without voting is 92.7%. The code for Point Mamba(C) will be released in another branch later.
   Checkpoints will be released later.
    ```bash
    python classification.py --config configs/cls_m40.yaml SOLVER.gpu 0,
    ```