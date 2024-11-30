#pragma once

#include <vector>
#include <memory>
#include <cuda.h>
#include "utils.h"
#include "gcpool_logging.h"

struct BlockSegment;

struct PhyBlock {
    PhyBlock(int device_id_in = -1, size_t block_size_in = granularitySize);
    ~PhyBlock();

    void release_resources();

    int device_id;
    const size_t block_size;
    CUmemGenericAllocationHandle alloc_handle;
    CUresult status;

    bool free;
    cudaStream_t owner_stream;
    std::vector<BlockSegment> mapped_blocks;
    bool released;
};
