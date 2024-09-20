#pragma once

#include "./defs.h"
#include "./types.h"
#include "core/os/memory.h"


#include <assert.h>
#include <new>
#include <xmemory>

template<typename TYPE> class OffsetPtr;

namespace human_anim
{
namespace memory
{
    // Should be constructed with either:
    //      kMemAnimation
    //      kMemAnimationTemp
    class HeapAllocator 
    {
    public:

        virtual void* Allocate(std::size_t size, std::size_t align)
        {
            void* p = DefaultAllocator::alloc(size);
            return p;
        }

        virtual void Deallocate(void * p)
        {
            DefaultAllocator::free( p);
        }

    };


}
}
template<typename TYPE> class OffsetPtr;

class RuntimeBaseAllocator 
{
public:
    virtual void* Allocate(std::size_t size, std::size_t align) = 0;
    virtual void  Deallocate(void * p) = 0;

    template<typename TYPE>
    TYPE* Construct(std::size_t align = alignof(TYPE))
    {
        return memnew(TYPE);
    }

    template<typename TYPE>
    TYPE* ConstructArray(std::size_t count, const TYPE& val = TYPE(), std::size_t align = alignof(TYPE))
    {
        TYPE* uninitPtr = memnew_arr(TYPE, count);
         std::uninitialized_fill(uninitPtr, uninitPtr + count, val);
         return uninitPtr;
    }

    template<typename TYPE>
    TYPE* ConstructArray(const TYPE* input, std::size_t count, std::size_t align = alignof(TYPE))
    {
        TYPE* uninitPtr = memnew_arr(TYPE, count);
         return uninitPtr;
    }

    void Deallocate(void const * p)
    {
    }

    template<typename TYPE>
    void Deallocate(OffsetPtr<TYPE>& p)
    {
    }

    template<typename TYPE>
    void Deallocate(OffsetPtr<TYPE const>& p)
    {
    }

    static std::size_t AlignAddress(std::size_t aAddr, std::size_t aAlign)
    {
        return aAddr + ((~aAddr + 1U) & (aAlign - 1U));
    }

    template<typename TYPE>
    static std::size_t AlignForAllocate(std::size_t aAddr = 0, std::size_t count = 1, std::size_t aAlign = alignof(TYPE))
    {
        return aAddr;
    }
};
