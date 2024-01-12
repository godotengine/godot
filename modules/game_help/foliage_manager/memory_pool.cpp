#include "memory_pool.h"
namespace Foliage
{
    MemoryPool::Block* MemoryPool::Block::s_freeBlock = nullptr;
}