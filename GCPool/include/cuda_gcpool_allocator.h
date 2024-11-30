#pragma once

#include "gcpool_logging.h"
#include "utils.h"
#include "phy_block.h"
#include "vir_dev_ptr.h"
#include "vir_block.h"
#include "vmm_segment.h"

#include <typeindex>
#include <typeinfo>
#include <type_traits>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <unordered_map>
#include <utility>
#include <iostream>
#include <sstream>
#include <vector>
#include <c10/util/intrusive_ptr.h>

namespace c10 {
namespace cuda {
namespace CUDACachingAllocator {
namespace Native {
    namespace {
        struct Block;
    }
}
}
}

struct BlockSegment {
    BlockSegment();
    BlockSegment(c10::cuda::CUDACachingAllocator::Native::Block* block, size_t offset);

    c10::cuda::CUDACachingAllocator::Native::Block* block;
    size_t offset;
};
}