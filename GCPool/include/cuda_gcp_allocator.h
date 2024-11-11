// Copyright 2022 The GCPool Authors. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// Modified GLake Memory Allocator with Compaction
// Implements compaction using mark-and-sweep algorithm

#pragma once

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
#include <execinfo.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <sys/syscall.h>
#include <stdio.h>
#include <memory>

#include <c10/util/intrusive_ptr.h>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

typedef unsigned long long CUmemGenericAllocationHandle;

extern int gcpoolInfoLevel = -1;
typedef enum {GCPool_LOG_NONE=0, GCPool_LOG_INFO=1} gcpoolInfoLogLevel;
pthread_mutex_t gcpoolInfoLock = PTHREAD_MUTEX_INITIALIZER;
FILE* gcpoolInfoFile = stdout;

void gcpoolInfoLog(const char* filefunc, int line, const char* format, ...);

#define GCPool_INFO(...) \
    do { \
        gcpoolInfoLog(__func__, __LINE__, __VA_ARGS__); \
    } while(0);

#define gettid() (pid_t) syscall(SYS_gettid)

#define gtrace()  { \
    void *traces[32]; \
    int size = backtrace(traces, 32); \
    char **msgs = backtrace_symbols(traces, size); \
    if (NULL == msgs)  { \
        exit(EXIT_FAILURE); \
    } \
    printf("------------------\n"); \
    for (int i = 0; i < size; i++) { \
        printf("[bt] #%d %s symbol:%p \n", i, msgs[i], traces[i]); \
        fflush(stdout); \
    } \
    printf("------------------\n"); \
    free (msgs); \
    msgs = NULL; \
}

#define LOGE(format, ...) fprintf(stdout, "L%d:" format "\n", __LINE__, ##__VA_ARGS__); fflush(stdout);
#define ASSERT(cond, ...) { if(!(cond)) { LOGE(__VA_ARGS__); assert(0); } }
#define WARN(cond, ...) { if(!(cond)) { LOGE(__VA_ARGS__); } } 

#define DRV_CALL(call)                                                                                  \
    {                                                                                                   \
        CUresult result = (call);                                                                       \
        if (CUDA_SUCCESS != result)                                                                     \
        {                                                                                               \
            const char *errMsg; cuGetErrorString(result, &errMsg);                                      \
            ASSERT(0, "Error when exec " #call " %s-%d code:%d err:%s", __FUNCTION__, __LINE__, result, errMsg); \
        }                                                                                               \
    }

#define DRV_CALL_RET(call, status_val)                                                                  \
    {                                                                                                   \
        if(CUDA_SUCCESS == status_val)                                                                  \
        {                                                                                               \
            CUresult result = (call);                                                                   \
            if (CUDA_SUCCESS != result)                                                                 \
            {                                                                                           \
                const char *errMsg; cuGetErrorString(result, &errMsg);                                  \
                WARN(0, "Error when exec " #call " %s-%d code:%d err:%s", __FUNCTION__, __LINE__, result, errMsg); \
            }                                                                                           \
            status_val = result;                                                                        \
        }                                                                                               \
    }

static constexpr size_t granularitySize = 2097152; // 2 MB

void gcpoolInfoInit() {
    pthread_mutex_lock(&gcpoolInfoLock);
    if (gcpoolInfoLevel != -1) {pthread_mutex_unlock(&gcpoolInfoLock); return;}
    const char* gcpool_info = getenv("GCPool_INFO");
    if (gcpool_info == NULL) {
        gcpoolInfoLevel = GCPool_LOG_NONE;
    } else if (strcasecmp(gcpool_info, "INFO") == 0) {
        gcpoolInfoLevel = GCPool_LOG_INFO;
    }
}

void getHostName(char* hostname, int maxlen, const char delim) {
    if (gethostname(hostname, maxlen) != 0) {
        strncpy(hostname, "unknown", maxlen);
        return;
    }
    int i = 0;
    while ((hostname[i] != delim) && (hostname[i] != '\0') && (i < maxlen - 1)) i++;
    hostname[i] = '\0';
    return; 
}

void gcpoolInfoLog(const char* filefunc, int line, const char* fmt, ...) {
    if (gcpoolInfoLevel == -1) gcpoolInfoInit();
    if (gcpoolInfoLevel == GCPool_LOG_NONE) return;

    char hostname[1024];
    getHostName(hostname, 1024, '.');
    int cudaDev;
    cudaGetDevice(&cudaDev);
    int pid = getpid();
    int tid = gettid();

    char buffer[1024];
    size_t len = 0;
    len = snprintf(buffer, sizeof(buffer), "%s:%d:%d [%d] GCPool_INFO %s():%d ", hostname, pid, tid, cudaDev, filefunc, line);
    if (len) {
        va_list vargs;
        va_start(vargs, fmt);
        (void) vsnprintf(buffer+len, sizeof(buffer)-len, fmt, vargs);
        va_end(vargs);
        fprintf(gcpoolInfoFile, "%s\n", buffer);
        fflush(gcpoolInfoFile);
    }
}

size_t getGranularitySize()
{
    static size_t granularity = -1;
  
    if(granularity == -1) {
        int current_device;
        DRV_CALL(cuCtxGetDevice(&current_device));
      
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = current_device;
      
        DRV_CALL(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
    }

    return granularity;
}

void setMemAccess(void* ptr, size_t size, int current_device_in = -1)
{
    int current_device = current_device_in;
    if(current_device == -1) {
        DRV_CALL(cuCtxGetDevice(&current_device));
    }

    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = current_device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    DRV_CALL(cuMemSetAccess((CUdeviceptr)ptr, size, &accessDesc, 1)); 
}

namespace c10
{
    namespace cuda
    {
        namespace CUDACachingAllocator
        {
            namespace Native
            {   namespace
                {
                    struct Block;
                }
            }
        }
    }
}

// Forward declarations
struct BlockSegment;
struct PhyBlock;
struct VirDevPtr;
struct VirBlock;
struct VmmSegment;

struct BlockSegment
{
    BlockSegment():block(nullptr), offset(0) {}
    BlockSegment(c10::cuda::CUDACachingAllocator::Native::Block* block, size_t offset):block(block), offset(offset) {}
    
    c10::cuda::CUDACachingAllocator::Native::Block* block;
    size_t offset;
};


// Physical memory block
struct PhyBlock
{
    PhyBlock(int device_id_in = -1, size_t block_size_in = granularitySize): 
        device_id(device_id_in), 
        block_size(block_size_in), 
        status(CUDA_SUCCESS), 
        free(true),
        owner_stream(nullptr),
        released(false)
    {
        if(device_id == -1)
        {
            DRV_CALL(cuCtxGetDevice(&device_id));
        }
        
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_id;
        
        DRV_CALL_RET(cuMemCreate(&alloc_handle, block_size, &prop, 0), status); 
    }
    
    void release_resources()
    {
        if(status == CUDA_SUCCESS)
        {
            DRV_CALL(cuMemRelease(alloc_handle)); 
        }
        released = true;
    }  
    
    ~PhyBlock()
    {
        if(!released)
        {
            this->release_resources();
            released = true;
        }
    }
    
    int device_id;
    const size_t block_size;
    CUmemGenericAllocationHandle alloc_handle;
    CUresult status;
    
    bool free;
    cudaStream_t owner_stream;
    std::vector<BlockSegment> mapped_blocks;
    bool released;
};

// Virtual device pointer representing reserved virtual address space
struct VirDevPtr {
    VirDevPtr(size_t allocSize_in, int device_id = -1): allocSize(allocSize_in), mapped(false), device_id(device_id), status(CUDA_SUCCESS), released(false) {
        if(device_id == -1) {
            DRV_CALL(cuCtxGetDevice(&device_id));
        }
      
        CUdeviceptr device_ptr;
        CUdeviceptr request_ptr = 0ULL; // Let CUDA choose the address
      
        DRV_CALL_RET(cuMemAddressReserve(&device_ptr, allocSize, 0ULL, request_ptr, 0ULL), status);
      
        if(status != CUDA_SUCCESS || !device_ptr) {
            GCPool_INFO("Failed to reserve virtual address space of size %zu bytes", allocSize);
            virAddr = nullptr;
            return;
        }
      
        virAddr = (void*)device_ptr;
    }

    void release_resources() {
        if(virAddr) {
            if(mapped) {
                DRV_CALL(cuMemUnmap((CUdeviceptr)virAddr, allocSize));
            }
            DRV_CALL(cuMemAddressFree((CUdeviceptr)virAddr, allocSize)); 
        }
        released = true;
    }
  
    ~VirDevPtr() {
        if(!released) {
            this->release_resources();
            released = true;
        }
    }

    void* virAddr;
    const size_t allocSize;
    bool mapped;
    int device_id;
    CUresult status;
    bool released;
};

// Virtual memory block representing a mapping from virtual to physical memory
struct VirBlock
{
    VirBlock(std::shared_ptr<VirDevPtr> vir_dev_ptr_in, 
             size_t offset_in, 
             size_t blockSize_in, 
             std::shared_ptr<PhyBlock> phy_block_in,
             int device_id = -1):vir_dev_ptr(vir_dev_ptr_in),
                                 offset(offset_in), 
                                 blockSize(blockSize_in), 
                                 phy_block(phy_block_in),
                                 device_id(device_id),
                                 status(CUDA_SUCCESS),
                                 released(false) {
        if(device_id == -1) {
            DRV_CALL(cuCtxGetDevice(&device_id));
        }
        
        block_ptr = (void*) ( ((char*)vir_dev_ptr->virAddr) + offset);
        
        CUdeviceptr device_ptr = (CUdeviceptr)block_ptr;
        
        DRV_CALL_RET(cuMemMap(device_ptr, blockSize, 0ULL, phy_block->alloc_handle, 0ULL), status);
        setMemAccess((void*)device_ptr, blockSize, device_id);
        
        phy_block->mapped_blocks.emplace_back(nullptr, offset); // Keep track of mappings
        
        if(offset == 0) {
            vir_dev_ptr->mapped = true;
        }
    }

    void release_resources() {
        vir_dev_ptr.reset();
        released = true;
    }
    
    ~VirBlock() {
        if(!released) {
            this->release_resources();
            released = true;
        }
    }
    
    std::shared_ptr<VirDevPtr> vir_dev_ptr;
    
    size_t offset;
    size_t blockSize;
    void* block_ptr;
    
    std::shared_ptr<PhyBlock> phy_block;
    
    int device_id;
    CUresult status;
    bool released;
};

// The main memory segment manager
struct VmmSegment
{
    VmmSegment(size_t totalSize_in, int device_id_in = -1):
        totalSize(totalSize_in),
        device_id(device_id_in),
        status(CUDA_SUCCESS),
        released(false)
    {
        if(device_id == -1) {
            DRV_CALL(cuCtxGetDevice(&device_id));
        }

        // Reserve virtual address space
        vir_dev_ptr = std::make_shared<VirDevPtr>(totalSize, device_id);
        if(vir_dev_ptr->status != CUDA_SUCCESS) {
            status = vir_dev_ptr->status;
            return;
        }

        segment_ptr = vir_dev_ptr->virAddr;

        // Initialize the free block list
        auto initialBlock = std::make_shared<MemBlock>();
        initialBlock->offset = 0;
        initialBlock->size = totalSize;
        initialBlock->is_free = true;
        initialBlock->phy_block = nullptr;

        free_blocks.emplace_back(initialBlock);
    }

    void release_resources() {
        // Unmap and release all physical blocks
        for(auto& block : allocated_blocks) {
            unmapBlock(block);
        }
        allocated_blocks.clear();
        free_blocks.clear();

        vir_dev_ptr.reset();
        released = true;
    }

    ~VmmSegment()
    {
        if(!released) {
            this->release_resources();
            released = true;
        }
    }

    // Structure to represent a memory block within the segment
    struct MemBlock {
        size_t offset; // Offset within the virtual address space
        size_t size;
        bool is_free;
        std::shared_ptr<PhyBlock> phy_block;
    };

    // Allocate a block of memory
    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex);

        // Align size to granularity
        size = ((size + granularitySize - 1) / granularitySize) * granularitySize;

        // First, try to find a suitable free block
        for(auto it = free_blocks.begin(); it != free_blocks.end(); ++it) {
            auto block = *it;
            if(block->size >= size) {
                // Allocate physical memory
                auto phy_block = std::make_shared<PhyBlock>(device_id, size);
                if(phy_block->status != CUDA_SUCCESS) {
                    status = phy_block->status;
                    return nullptr;
                }

                // Map the physical block to virtual address space
                CUdeviceptr device_ptr = (CUdeviceptr)((char*)segment_ptr + block->offset);
                DRV_CALL_RET(cuMemMap(device_ptr, size, 0ULL, phy_block->alloc_handle, 0ULL), status);
                setMemAccess((void*)device_ptr, size, device_id);

                // Create a new allocated block
                auto alloc_block = std::make_shared<MemBlock>();
                alloc_block->offset = block->offset;
                alloc_block->size = size;
                alloc_block->is_free = false;
                alloc_block->phy_block = phy_block;

                allocated_blocks.emplace_back(alloc_block);

                // Update the free block
                if(block->size > size) {
                    block->offset += size;
                    block->size -= size;
                } else {
                    free_blocks.erase(it);
                }

                return (void*)device_ptr;
            }
        }

        // If no suitable block found, perform compaction
        compact();

        // Try allocation again after compaction
        return allocate(size);
    }

    // Deallocate a block of memory
    void deallocate(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex);

        size_t offset = (size_t)((char*)ptr - (char*)segment_ptr);

        // Find the allocated block
        auto it = std::find_if(allocated_blocks.begin(), allocated_blocks.end(),
            [offset](const std::shared_ptr<MemBlock>& block) {
                return block->offset == offset;
            });

        if(it != allocated_blocks.end()) {
            auto block = *it;

            // Unmap and release the physical block
            unmapBlock(block);

            // Mark the block as free and add to free list
            block->is_free = true;
            block->phy_block.reset();
            free_blocks.emplace_back(block);

            allocated_blocks.erase(it);

            // Coalesce adjacent free blocks
            coalesceFreeBlocks();
        } else {
            GCPool_INFO_INFO("Attempted to deallocate invalid pointer %p", ptr);
        }
    }

    // Compact the memory to reduce fragmentation
    void compact() {
        GCPool_INFO_INFO("Starting compaction");
        std::lock_guard<std::mutex> lock(mutex);

        if(allocated_blocks.empty() || free_blocks.empty()) {
            // Nothing to compact
            return;
        }

        // Sort allocated blocks by offset
        std::sort(allocated_blocks.begin(), allocated_blocks.end(),
            [](const std::shared_ptr<MemBlock>& a, const std::shared_ptr<MemBlock>& b) {
                return a->offset < b->offset;
            });

        size_t next_offset = 0;
        for(auto& block : allocated_blocks) {
            if(block->offset > next_offset) {
                // Need to move the block
                size_t old_offset = block->offset;
                size_t size = block->size;

                // Unmap the old mapping
                CUdeviceptr old_device_ptr = (CUdeviceptr)((char*)segment_ptr + old_offset);
                DRV_CALL(cuMemUnmap(old_device_ptr, size));

                // Map to new location
                CUdeviceptr new_device_ptr = (CUdeviceptr)((char*)segment_ptr + next_offset);
                DRV_CALL(cuMemMap(new_device_ptr, size, 0ULL, block->phy_block->alloc_handle, 0ULL));
                setMemAccess((void*)new_device_ptr, size, device_id);

                // Copy the data
                DRV_CALL(cuMemcpy((CUdeviceptr)new_device_ptr, (CUdeviceptr)old_device_ptr, size));

                // Update block offset
                block->offset = next_offset;

                // Update pointers in application (if necessary)
                // In practice, you need to provide a mechanism to update pointers in the application.
                // This could be a callback or a mapping from old to new addresses.

                // Update next_offset
                next_offset += size;
            } else {
                // Block is already in the correct position
                next_offset += block->size;
            }
        }

        // Unmap the remaining virtual address space
        size_t remaining_size = totalSize - next_offset;
        if(remaining_size > 0) {
            CUdeviceptr device_ptr = (CUdeviceptr)((char*)segment_ptr + next_offset);
            DRV_CALL(cuMemUnmap(device_ptr, remaining_size));
        }

        // Update free blocks
        free_blocks.clear();

        if(remaining_size > 0) {
            auto free_block = std::make_shared<MemBlock>();
            free_block->offset = next_offset;
            free_block->size = remaining_size;
            free_block->is_free = true;
            free_block->phy_block = nullptr;
            free_blocks.emplace_back(free_block);
        }

        GCPool_INFO_INFO("Compaction completed");
    }

private:
    // Unmap and release a block
    void unmapBlock(const std::shared_ptr<MemBlock>& block) {
        CUdeviceptr device_ptr = (CUdeviceptr)((char*)segment_ptr + block->offset);
        size_t size = block->size;

        DRV_CALL(cuMemUnmap(device_ptr, size));
        block->phy_block->release_resources();
    }

    // Coalesce adjacent free blocks
    void coalesceFreeBlocks() {
        if(free_blocks.size() <= 1) {
            return;
        }

        // Sort free blocks by offset
        std::sort(free_blocks.begin(), free_blocks.end(),
            [](const std::shared_ptr<MemBlock>& a, const std::shared_ptr<MemBlock>& b) {
                return a->offset < b->offset;
            });

        std::vector<std::shared_ptr<MemBlock>> new_free_blocks;
        auto prev_block = free_blocks[0];

        for(size_t i = 1; i < free_blocks.size(); ++i) {
            auto curr_block = free_blocks[i];
            if(prev_block->offset + prev_block->size == curr_block->offset) {
                // Adjacent blocks, coalesce
                prev_block->size += curr_block->size;
            } else {
                new_free_blocks.emplace_back(prev_block);
                prev_block = curr_block;
            }
        }
        new_free_blocks.emplace_back(prev_block);

        free_blocks = std::move(new_free_blocks);
    }

    std::shared_ptr<VirDevPtr> vir_dev_ptr;
    void* segment_ptr;
    size_t totalSize;
    int device_id;
    CUresult status;
    bool released;

    struct MemBlock;

    std::vector<std::shared_ptr<MemBlock>> allocated_blocks;
    std::vector<std::shared_ptr<MemBlock>> free_blocks;

    std::mutex mutex; // For thread safety
};

