#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include "vir_block.h"

struct VmmSegment {
    VmmSegment();
    VmmSegment(size_t blocks, size_t block_size_in = granularitySize, int device_id_in = -1);
    VmmSegment(std::vector<std::shared_ptr<PhyBlock>>&& phy_blocks_in);
    VmmSegment(std::vector<std::shared_ptr<PhyBlock>> phy_blocks_in, std::vector<std::shared_ptr<VirBlock>> vir_blocks_in);
    virtual ~VmmSegment();

    void allocate_phy_blocks(size_t blocks, size_t block_size_in, int device_id_in);
    void release_resources();
    void* mapVirAddr();
    std::shared_ptr<VmmSegment> split(size_t keep_size);
    bool remerge(VmmSegment& segment);

    std::vector<std::shared_ptr<PhyBlock>> phy_blocks;
    std::vector<std::shared_ptr<VirBlock>> vir_blocks;
    const size_t granul_size;
    void* segment_ptr;
    int device_id;
    CUresult status;
    size_t free_blocks;
    size_t used_blocks;
    bool fused;
    bool released;
};
