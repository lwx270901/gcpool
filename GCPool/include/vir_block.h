#pragma once

#include <memory>
#include "phy_block.h"
#include "vir_dev_ptr.h"

struct VirBlock {
    VirBlock(std::shared_ptr<VirDevPtr> vir_dev_ptr_in, size_t offset_in, size_t blockSize_in,
             std::shared_ptr<PhyBlock> phy_block_in, int device_id = -1);
    ~VirBlock();

    void release_resources();

    std::shared_ptr<VirDevPtr> vir_dev_ptr;
    size_t offset;
    size_t blockSize;
    void* block_ptr;
    std::shared_ptr<PhyBlock> phy_block;
    int device_id;
    CUresult status;
    bool released;
};
