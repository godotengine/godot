/*! \file btGenericPoolAllocator.h
\author Francisco Leon Najera. email projectileman@yahoo.com

General purpose allocator class
*/
/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef BT_GENERIC_POOL_ALLOCATOR_H
#define BT_GENERIC_POOL_ALLOCATOR_H

#include <limits.h>
#include <stdio.h>
#include <string.h>
#include "LinearMath/btAlignedAllocator.h"

#define BT_UINT_MAX UINT_MAX
#define BT_DEFAULT_MAX_POOLS 16

//! Generic Pool class
class btGenericMemoryPool
{
public:
	unsigned char *m_pool;      //[m_element_size*m_max_element_count];
	size_t *m_free_nodes;       //[m_max_element_count];//! free nodes
	size_t *m_allocated_sizes;  //[m_max_element_count];//! Number of elements allocated per node
	size_t m_allocated_count;
	size_t m_free_nodes_count;

protected:
	size_t m_element_size;
	size_t m_max_element_count;

	size_t allocate_from_free_nodes(size_t num_elements);
	size_t allocate_from_pool(size_t num_elements);

public:
	void init_pool(size_t element_size, size_t element_count);

	void end_pool();

	btGenericMemoryPool(size_t element_size, size_t element_count)
	{
		init_pool(element_size, element_count);
	}

	~btGenericMemoryPool()
	{
		end_pool();
	}

	inline size_t get_pool_capacity()
	{
		return m_element_size * m_max_element_count;
	}

	inline size_t gem_element_size()
	{
		return m_element_size;
	}

	inline size_t get_max_element_count()
	{
		return m_max_element_count;
	}

	inline size_t get_allocated_count()
	{
		return m_allocated_count;
	}

	inline size_t get_free_positions_count()
	{
		return m_free_nodes_count;
	}

	inline void *get_element_data(size_t element_index)
	{
		return &m_pool[element_index * m_element_size];
	}

	//! Allocates memory in pool
	/*!
	\param size_bytes size in bytes of the buffer
	*/
	void *allocate(size_t size_bytes);

	bool freeMemory(void *pointer);
};

//! Generic Allocator with pools
/*!
General purpose Allocator which can create Memory Pools dynamiacally as needed.
*/
class btGenericPoolAllocator
{
protected:
	size_t m_pool_element_size;
	size_t m_pool_element_count;

public:
	btGenericMemoryPool *m_pools[BT_DEFAULT_MAX_POOLS];
	size_t m_pool_count;

	inline size_t get_pool_capacity()
	{
		return m_pool_element_size * m_pool_element_count;
	}

protected:
	// creates a pool
	btGenericMemoryPool *push_new_pool();

	void *failback_alloc(size_t size_bytes);

	bool failback_free(void *pointer);

public:
	btGenericPoolAllocator(size_t pool_element_size, size_t pool_element_count)
	{
		m_pool_count = 0;
		m_pool_element_size = pool_element_size;
		m_pool_element_count = pool_element_count;
	}

	virtual ~btGenericPoolAllocator();

	//! Allocates memory in pool
	/*!
	\param size_bytes size in bytes of the buffer
	*/
	void *allocate(size_t size_bytes);

	bool freeMemory(void *pointer);
};

void *btPoolAlloc(size_t size);
void *btPoolRealloc(void *ptr, size_t oldsize, size_t newsize);
void btPoolFree(void *ptr);

#endif
