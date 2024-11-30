## Install
### On Linux
We have adapted to different versions of PyTorch, such as PyTorch-1.13.1, PyTorch-2.0, and pre-release PyTorch-2.1. The repository is with PyTorch2.0. 
```
git clone -b release/2.0 https://github.com/pytorch/pytorch
cd pytorch
git submodule sync
git submodule update --init --recursive
cp GCPool/src/CUDACachingAllocator.cpp ~/pytorch/c10/cuda
cp GCPool/include/* ~/pytorch/c10/cuda
vim pytorch/c10/cuda/CMakeLists.txt
change target_link_libraries(c10_cuda PUBLIC c10 torch::cudart) -> target_link_libraries(c10_cuda PUBLIC c10 torch::cudart caffe2::cuda)
TORCH_CUDA_ARCH_LIST="8.0" USE_CUDA=1 python setup.py install
```
### Testing
Because it is already integrated with pytorch, you just need to use pytorch and it will automatically be used