#pragma once

#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "gcpool_logging.h"

size_t getGranularitySize();

CUresult setMemAccess(void* ptr, size_t size, int current_device_in = -1);

void gcpoolInfoInit();

void getHostName(char* hostname, int maxlen, const char delim);

#define DRV_CALL(call) \
    { \
        CUresult result = (call); \
        if (CUDA_SUCCESS != result) { \
            const char* errMsg; cuGetErrorString(result, &errMsg); \
            ASSERT(0, "Error when exec " #call " %s-%d code:%d err:%s", __FUNCTION__, __LINE__, result, errMsg); \
        } \
    }

#define DRV_CALL_RET(call, status_val) \
    { \
        if (CUDA_SUCCESS == status_val) { \
            CUresult result = (call); \
            if (CUDA_SUCCESS != result) { \
                const char* errMsg; cuGetErrorString(result, &errMsg); \
                WARN(0, "Error when exec " #call " %s-%d code:%d err:%s", __FUNCTION__, __LINE__, result, errMsg); \
            } \
            status_val = result; \
        } \
    }

static constexpr size_t granularitySize = 2097152;
