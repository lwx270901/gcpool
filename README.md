## Install
### On Linux
We have adapted to different versions of PyTorch, such as PyTorch-1.13.1, PyTorch-2.0, and pre-release PyTorch-2.1. The repository is with PyTorch2.0. 
```
git clone -b release/2.0 https://github.com/pytorch/pytorch
cd pytorch
git submodule sync
git submodule update --init --recursive
cp ~/GMLake/CUDACachingAllocator.cpp ~/pytorch/c10/cuda
cp ~/GMLake/CUDACachingAllocator.h ~/pytorch/c10/cuda
cp ~/GMLake/cuda_gcp_allocator.h ~/pytorch/c10/cuda
vim pytorch/c10/cuda/CMakeLists.txt
change target_link_libraries(c10_cuda PUBLIC c10 torch::cudart) -> target_link_libraries(c10_cuda PUBLIC c10 torch::cudart caffe2::cuda)
TORCH_CUDA_ARCH_LIST="8.0" USE_CUDA=1 python setup.py install
```
### We create a training scripts
Modify parameter 
BS = Batch size
True = use our method, False = use pytorch caching allocator
```
cd test/DeepSpeed-Chat/training/step1_supervised_finetuning/
vim training_scripts/single_node/benchmark.sh
for BS in 160
do
for GPU_NUM in 4
do
        bash training_scripts/single_node/finetune.sh $GPU_NUM $BS facebook/opt-1.3b True 1 1 0
done
done
```

Run benchmark
```
bash training_scripts/single_node/benchmark.sh
```

You can see the output hear:
```
tail -f output/output_opt-1.3b/training_opt-1.3b_***.log
```
end