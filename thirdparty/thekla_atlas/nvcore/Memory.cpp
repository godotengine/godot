// This code is in the public domain -- Ignacio Castaño <castano@gmail.com>

#include "Memory.h"
#include "Debug.h"
#include "Utils.h"

#include <stdlib.h>

#ifdef NV_OS_LINUX
#include <malloc.h>
#endif

#define USE_EFENCE 0

#if USE_EFENCE
extern "C" void *EF_malloc(size_t size);
extern "C" void *EF_realloc(void * oldBuffer, size_t newSize);
extern "C" void EF_free(void * address);
#endif

using namespace nv;

#if NV_OVERRIDE_ALLOC

void * malloc(size_t size)
{
#if USE_EFENCE
    return EF_malloc(size);
#else
    return ::malloc(size);
#endif
}

void * debug_malloc(size_t size, const char * file, int line)
{
    NV_UNUSED(file);
    NV_UNUSED(line);
#if USE_EFENCE
    return EF_malloc(size);
#else
    return ::malloc(size);
#endif
}

void free(void * ptr)
{
#if USE_EFENCE
    return EF_free(const_cast<void *>(ptr));
#else
    ::free(const_cast<void *>(ptr));
#endif
}

void * realloc(void * ptr, size_t size)
{
    nvDebugCheck(ptr != NULL || size != 0); // undefined realloc behavior.
#if USE_EFENCE
    return EF_realloc(ptr, size);
#else
    return ::realloc(ptr, size);
#endif
}


/* No need to override this unless we want line info.
void * operator new (size_t size) throw()
{
    return malloc(size);
}

void operator delete (void *p) throw()
{
    free(p);
}

void * operator new [] (size_t size) throw()
{
    return malloc(size);
}

void operator delete [] (void * p) throw()
{
    free(p);
}
*/

#if 0 // Code from Apple:
void* operator new(std::size_t sz) throw (std::bad_alloc)
{
        void *result = std::malloc (sz == 0 ? 1 : sz);
        if (result == NULL)
                throw std::bad_alloc();
        gNewCounter++;
        return result;
}
void operator delete(void* p) throw()
{
        if (p == NULL)
                return;
        std::free (p);
        gDeleteCounter++;
}

/* These are the 'nothrow' versions of the above operators.
   The system version will try to call a std::new_handler if they
   fail, but your overriding versions are not required to do this.  */
void* operator new(std::size_t sz, const std::nothrow_t&) throw()
{
        try {
                void * result = ::operator new (sz);  // calls our overridden operator new
                return result;
        } catch (std::bad_alloc &) {
          return NULL;
        }
}
void operator delete(void* p, const std::nothrow_t&) throw()
{
        ::operator delete (p);
}

#endif // 0

#endif // NV_OVERRIDE_ALLOC

void * nv::aligned_malloc(size_t size, size_t alignment)
{
    // alignment must be a power of two, multiple of sizeof(void*)
    nvDebugCheck(isPowerOfTwo(alignment));
    nvDebugCheck((alignment & (sizeof(void*) - 1)) == 0);

#if NV_OS_WIN32 || NV_OS_DURANGO
    return _aligned_malloc(size, alignment);
#elif NV_OS_DARWIN && !NV_OS_IOS
    void * ptr = NULL;
    posix_memalign(&ptr, alignment, size);
    return ptr;
#elif NV_OS_LINUX
    return memalign(alignment, size);
#else // NV_OS_ORBIS || NV_OS_IOS
    // @@ IC: iOS appears to be 16 byte aligned, should we check alignment and assert if we request a higher alignment factor?
    return ::malloc(size);
#endif
}

void nv::aligned_free(void * ptr)
{
#if NV_OS_WIN32 || NV_OS_DURANGO
    _aligned_free(ptr);
#else
    ::free(ptr);
#endif
}

