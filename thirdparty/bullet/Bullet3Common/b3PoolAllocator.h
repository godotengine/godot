/*
Copyright (c) 2003-2013 Gino van den Bergen / Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef _BT_POOL_ALLOCATOR_H
#define _BT_POOL_ALLOCATOR_H

#include "b3Scalar.h"
#include "b3AlignedAllocator.h"

///The b3PoolAllocator class allows to efficiently allocate a large pool of objects, instead of dynamically allocating them separately.
class b3PoolAllocator
{
	int m_elemSize;
	int m_maxElements;
	int m_freeCount;
	void* m_firstFree;
	unsigned char* m_pool;

public:
	b3PoolAllocator(int elemSize, int maxElements)
		: m_elemSize(elemSize),
		  m_maxElements(maxElements)
	{
		m_pool = (unsigned char*)b3AlignedAlloc(static_cast<unsigned int>(m_elemSize * m_maxElements), 16);

		unsigned char* p = m_pool;
		m_firstFree = p;
		m_freeCount = m_maxElements;
		int count = m_maxElements;
		while (--count)
		{
			*(void**)p = (p + m_elemSize);
			p += m_elemSize;
		}
		*(void**)p = 0;
	}

	~b3PoolAllocator()
	{
		b3AlignedFree(m_pool);
	}

	int getFreeCount() const
	{
		return m_freeCount;
	}

	int getUsedCount() const
	{
		return m_maxElements - m_freeCount;
	}

	int getMaxCount() const
	{
		return m_maxElements;
	}

	void* allocate(int size)
	{
		// release mode fix
		(void)size;
		b3Assert(!size || size <= m_elemSize);
		b3Assert(m_freeCount > 0);
		void* result = m_firstFree;
		m_firstFree = *(void**)m_firstFree;
		--m_freeCount;
		return result;
	}

	bool validPtr(void* ptr)
	{
		if (ptr)
		{
			if (((unsigned char*)ptr >= m_pool && (unsigned char*)ptr < m_pool + m_maxElements * m_elemSize))
			{
				return true;
			}
		}
		return false;
	}

	void freeMemory(void* ptr)
	{
		if (ptr)
		{
			b3Assert((unsigned char*)ptr >= m_pool && (unsigned char*)ptr < m_pool + m_maxElements * m_elemSize);

			*(void**)ptr = m_firstFree;
			m_firstFree = ptr;
			++m_freeCount;
		}
	}

	int getElementSize() const
	{
		return m_elemSize;
	}

	unsigned char* getPoolAddress()
	{
		return m_pool;
	}

	const unsigned char* getPoolAddress() const
	{
		return m_pool;
	}
};

#endif  //_BT_POOL_ALLOCATOR_H
