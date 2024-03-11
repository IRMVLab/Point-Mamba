# dwconv

`dwconv` implements the octree-based depth-wise convolution with CUDA.
It speed up the original PyTorch implementation from
[ocnn](https://ocnn-pytorch.readthedocs.io/en/latest/modules/nn.html#ocnn.nn.OctreeDWConv)
by 2.5 times.

The code has been tested on Ubuntu 20.04 with CUDA 11.2 and PyTorch 12.1. After
install the required packages, run the following command to install dwconv.

``` shell
git clone https://github.com/octree-nn/dwconv.git
pip install ./dwconv
```


<!-- 
## Installation


- Install via the following command:
    ``` shell
    pip install dwconv
    ``` 


- Alternatively, install from source via the following commands.
    ``` shell
    git clone https://github.com/octree-nn/dwconv.git
    pip install ./dwconv
    ```
 -->
