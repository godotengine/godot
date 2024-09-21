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
};
