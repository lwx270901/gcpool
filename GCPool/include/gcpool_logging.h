#pragma once

#include <execinfo.h>
#include <stdarg.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/syscall.h>

extern int gcpoolInfoLevel;
typedef enum { GCPOOL_LOG_NONE = 0, GCPOOL_LOG_INFO = 1 } gcpooleInfoLogLevel;
extern pthread_mutex_t gcpoolInfoLock;
extern FILE* gcpoolInfoFile;

void gcpoolInfoLog(const char* filefunc, int line, const char* fmt, ...);

#define GCPOOL_INFO(...) \
    do { \
        gcpoolInfoLog(__func__, __LINE__, __VA_ARGS__); \
    } while (0)

#define gettid() (pid_t)syscall(SYS_gettid)

#define gtrace() { \
    void* traces[32]; \
    int size = backtrace(traces, 32); \
    char** msgs = backtrace_symbols(traces, size); \
    if (msgs == NULL) { \
        exit(EXIT_FAILURE); \
    } \
    printf("------------------\n"); \
    for (int i = 0; i < size; i++) { \
        printf("[bt] #%d %s symbol:%p \n", i, msgs[i], traces[i]); \
        fflush(stdout); \
    } \
    printf("------------------\n"); \
    free(msgs); \
}

#define LOGE(format, ...) fprintf(stdout, "L%d:" format "\n", __LINE__, ##__VA_ARGS__); fflush(stdout)
#define ASSERT(cond, ...) { if (!(cond)) { LOGE(__VA_ARGS__); assert(0); } }
#define WARN(cond, ...) { if (!(cond)) { LOGE(__VA_ARGS__); } }
