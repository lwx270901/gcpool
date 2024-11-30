#pragma once

#include <memory>
#include <cuda.h>
#include "utils.h"
#include "gcpool_logging.h"

struct VirDevPtr {
    VirDevPtr(CUdeviceptr addr_in, size_t allocSize_in, int device_id = -1);
    ~VirDevPtr();

    void release_resources();

    void* virAddr;
    const size_t allocSize;
    bool mapped;
    int device_id;
    CUresult status;
    bool released;
};
