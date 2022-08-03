#pragma once

#include <string>

#include <thrust/detail/static_assert.h>
#undef THRUST_STATIC_ASSERT
#undef THRUST_STATIC_ASSERT_MSG

#define THRUST_STATIC_ASSERT(B) unittest::assert_static((B), __FILE__, __LINE__);
#define THRUST_STATIC_ASSERT_MSG(B, msg) unittest::assert_static((B), __FILE__, __LINE__);

namespace unittest
{
    __host__ __device__
    void assert_static(bool condition, const char * filename, int lineno);
}

#include <thrust/device_new.h>
#include <thrust/device_delete.h>

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

#define ASSERT_STATIC_ASSERT(X) \
    { \
        bool triggered = false; \
        typedef unittest::static_assert_exception ex_t; \
        thrust::device_ptr<ex_t> device_ptr = thrust::device_new<ex_t>(); \
        ex_t* raw_ptr = thrust::raw_pointer_cast(device_ptr); \
        ::cudaMemcpyToSymbol(unittest::detail::device_exception, &raw_ptr, sizeof(ex_t*)); \
        try { X; } catch (ex_t) { triggered = true; } \
        if (!triggered) { \
            triggered = static_cast<ex_t>(*device_ptr).triggered; \
        } \
        thrust::device_free(device_ptr); \
        raw_ptr = NULL; \
        ::cudaMemcpyToSymbol(unittest::detail::device_exception, &raw_ptr, sizeof(ex_t*)); \
        if (!triggered) { unittest::UnitTestFailure f; f << "[" << __FILE__ << ":" << __LINE__ << "] did not trigger a THRUST_STATIC_ASSERT"; throw f; } \
    }

#else

#define ASSERT_STATIC_ASSERT(X) \
    { \
        bool triggered = false; \
        typedef unittest::static_assert_exception ex_t; \
        try { X; } catch (ex_t) { triggered = true; } \
        if (!triggered) { unittest::UnitTestFailure f; f << "[" << __FILE__ << ":" << __LINE__ << "] did not trigger a THRUST_STATIC_ASSERT"; throw f; } \
    }

#endif

namespace unittest
{
    class static_assert_exception
    {
    public:
        __host__ __device__
        static_assert_exception() : triggered(false)
        {
        }

        __host__ __device__
        static_assert_exception(const char * filename, int lineno)
            : triggered(true), filename(filename), lineno(lineno)
        {
        }

        bool triggered;
        const char * filename;
        int lineno;
    };

    namespace detail
    {
#if THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_GCC || \
    THRUST_HOST_COMPILER == THRUST_HOST_COMPILER_CLANG
        __attribute__((used))
#endif
        __device__ static static_assert_exception* device_exception = NULL;
    }

    __host__ __device__
    void assert_static(bool condition, const char * filename, int lineno)
    {
        if (!condition)
        {
            static_assert_exception ex(filename, lineno);

#ifdef __CUDA_ARCH__
            *detail::device_exception = ex;
#else
            throw ex;
#endif
        }
    }
}

