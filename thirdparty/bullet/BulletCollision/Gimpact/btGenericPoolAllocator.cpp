/*! \file btGenericPoolAllocator.cpp
\author Francisco Leon Najera. email projectileman@yahoo.com

General purpose allocator class
*/
/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  http://continuousphysics.com/Bullet/

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#include "btGenericPoolAllocator.h"

/// *************** btGenericMemoryPool ******************///////////

size_t btGenericMemoryPool::allocate_from_free_nodes(size_t num_elements)
{
	size_t ptr = BT_UINT_MAX;

	if (m_free_nodes_count == 0) return BT_UINT_MAX;
	// find an avaliable free node with the correct size
	size_t revindex = m_free_nodes_count;

	while (revindex-- && ptr == BT_UINT_MAX)
	{
		if (m_allocated_sizes[m_free_nodes[revindex]] >= num_elements)
		{
			ptr = revindex;
		}
	}
	if (ptr == BT_UINT_MAX) return BT_UINT_MAX;  // not found

	revindex = ptr;
	ptr = m_free_nodes[revindex];
	// post: ptr contains the node index, and revindex the index in m_free_nodes

	size_t finalsize = m_allocated_sizes[ptr];
	finalsize -= num_elements;

	m_allocated_sizes[ptr] = num_elements;

	// post: finalsize>=0, m_allocated_sizes[ptr] has the requested size

	if (finalsize > 0)  // preserve free node, there are some free memory
	{
		m_free_nodes[revindex] = ptr + num_elements;
		m_allocated_sizes[ptr + num_elements] = finalsize;
	}
	else  // delete free node
	{
		// swap with end
		m_free_nodes[revindex] = m_free_nodes[m_free_nodes_count - 1];
		m_free_nodes_count--;
	}

	return ptr;
}

size_t btGenericMemoryPool::allocate_from_pool(size_t num_elements)
{
	if (m_allocated_count + num_elements > m_max_element_count) return BT_UINT_MAX;

	size_t ptr = m_allocated_count;

	m_allocated_sizes[m_allocated_count] = num_elements;
	m_allocated_count += num_elements;

	return ptr;
}

void btGenericMemoryPool::init_pool(size_t element_size, size_t element_count)
{
	m_allocated_count = 0;
	m_free_nodes_count = 0;

	m_element_size = element_size;
	m_max_element_count = element_count;

	m_pool = (unsigned char *)btAlignedAlloc(m_element_size * m_max_element_count, 16);
	m_free_nodes = (size_t *)btAlignedAlloc(sizeof(size_t) * m_max_element_count, 16);
	m_allocated_sizes = (size_t *)btAlignedAlloc(sizeof(size_t) * m_max_element_count, 16);

	for (size_t i = 0; i < m_max_element_count; i++)
	{
		m_allocated_sizes[i] = 0;
	}
}

void btGenericMemoryPool::end_pool()
{
	btAlignedFree(m_pool);
	btAlignedFree(m_free_nodes);
	btAlignedFree(m_allocated_sizes);
	m_allocated_count = 0;
	m_free_nodes_count = 0;
}

//! Allocates memory in pool
/*!
\param size_bytes size in bytes of the buffer
*/
void *btGenericMemoryPool::allocate(size_t size_bytes)
{
	size_t module = size_bytes % m_element_size;
	size_t element_count = size_bytes / m_element_size;
	if (module > 0) element_count++;

	size_t alloc_pos = allocate_from_free_nodes(element_count);
	// a free node is found
	if (alloc_pos != BT_UINT_MAX)
	{
		return get_element_data(alloc_pos);
	}
	// allocate directly on pool
	alloc_pos = allocate_from_pool(element_count);

	if (alloc_pos == BT_UINT_MAX) return NULL;  // not space
	return get_element_data(alloc_pos);
}

bool btGenericMemoryPool::freeMemory(void *pointer)
{
	unsigned char *pointer_pos = (unsigned char *)pointer;
	unsigned char *pool_pos = (unsigned char *)m_pool;
	// calc offset
	if (pointer_pos < pool_pos) return false;  //other pool
	size_t offset = size_t(pointer_pos - pool_pos);
	if (offset >= get_pool_capacity()) return false;  // far away

	// find free position
	m_free_nodes[m_free_nodes_count] = offset / m_element_size;
	m_free_nodes_count++;
	return true;
}

/// *******************! btGenericPoolAllocator *******************!///

btGenericPoolAllocator::~btGenericPoolAllocator()
{
	// destroy pools
	size_t i;
	for (i = 0; i < m_pool_count; i++)
	{
		m_pools[i]->end_pool();
		btAlignedFree(m_pools[i]);
	}
}

// creates a pool
btGenericMemoryPool *btGenericPoolAllocator::push_new_pool()
{
	if (m_pool_count >= BT_DEFAULT_MAX_POOLS) return NULL;

	btGenericMemoryPool *newptr = (btGenericMemoryPool *)btAlignedAlloc(sizeof(btGenericMemoryPool), 16);

	m_pools[m_pool_count] = newptr;

	m_pools[m_pool_count]->init_pool(m_pool_element_size, m_pool_element_count);

	m_pool_count++;
	return newptr;
}

void *btGenericPoolAllocator::failback_alloc(size_t size_bytes)
{
	btGenericMemoryPool *pool = NULL;

	if (size_bytes <= get_pool_capacity())
	{
		pool = push_new_pool();
	}

	if (pool == NULL)  // failback
	{
		return btAlignedAlloc(size_bytes, 16);
	}

	return pool->allocate(size_bytes);
}

bool btGenericPoolAllocator::failback_free(void *pointer)
{
	btAlignedFree(pointer);
	return true;
}

//! Allocates memory in pool
/*!
\param size_bytes size in bytes of the buffer
*/
void *btGenericPoolAllocator::allocate(size_t size_bytes)
{
	void *ptr = NULL;

	size_t i = 0;
	while (i < m_pool_count && ptr == NULL)
	{
		ptr = m_pools[i]->allocate(size_bytes);
		++i;
	}

	if (ptr) return ptr;

	return failback_alloc(size_bytes);
}

bool btGenericPoolAllocator::freeMemory(void *pointer)
{
	bool result = false;

	size_t i = 0;
	while (i < m_pool_count && result == false)
	{
		result = m_pools[i]->freeMemory(pointer);
		++i;
	}

	if (result) return true;

	return failback_free(pointer);
}

/// ************** STANDARD ALLOCATOR ***************************///

#define BT_DEFAULT_POOL_SIZE 32768
#define BT_DEFAULT_POOL_ELEMENT_SIZE 8

// main allocator
class GIM_STANDARD_ALLOCATOR : public btGenericPoolAllocator
{
public:
	GIM_STANDARD_ALLOCATOR() : btGenericPoolAllocator(BT_DEFAULT_POOL_ELEMENT_SIZE, BT_DEFAULT_POOL_SIZE)
	{
	}
};

// global allocator
GIM_STANDARD_ALLOCATOR g_main_allocator;

void *btPoolAlloc(size_t size)
{
	return g_main_allocator.allocate(size);
}

void *btPoolRealloc(void *ptr, size_t oldsize, size_t newsize)
{
	void *newptr = btPoolAlloc(newsize);
	size_t copysize = oldsize < newsize ? oldsize : newsize;
	memcpy(newptr, ptr, copysize);
	btPoolFree(ptr);
	return newptr;
}

void btPoolFree(void *ptr)
{
	g_main_allocator.freeMemory(ptr);
}
