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

#ifndef BT_UNION_FIND_H
#define BT_UNION_FIND_H

#include "LinearMath/btAlignedObjectArray.h"

#define USE_PATH_COMPRESSION 1

///see for discussion of static island optimizations by Vroonsh here: http://code.google.com/p/bullet/issues/detail?id=406
#define STATIC_SIMULATION_ISLAND_OPTIMIZATION 1

struct btElement
{
	int m_id;
	int m_sz;
};

///UnionFind calculates connected subsets
// Implements weighted Quick Union with path compression
// optimization: could use short ints instead of ints (halving memory, would limit the number of rigid bodies to 64k, sounds reasonable)
class btUnionFind
{
private:
	btAlignedObjectArray<btElement> m_elements;

public:
	btUnionFind();
	~btUnionFind();

	//this is a special operation, destroying the content of btUnionFind.
	//it sorts the elements, based on island id, in order to make it easy to iterate over islands
	void sortIslands();

	void reset(int N);

	SIMD_FORCE_INLINE int getNumElements() const
	{
		return int(m_elements.size());
	}
	SIMD_FORCE_INLINE bool isRoot(int x) const
	{
		return (x == m_elements[x].m_id);
	}

	btElement& getElement(int index)
	{
		return m_elements[index];
	}
	const btElement& getElement(int index) const
	{
		return m_elements[index];
	}

	void allocate(int N);
	void Free();

	int find(int p, int q)
	{
		return (find(p) == find(q));
	}

	void unite(int p, int q)
	{
		int i = find(p), j = find(q);
		if (i == j)
			return;

#ifndef USE_PATH_COMPRESSION
		//weighted quick union, this keeps the 'trees' balanced, and keeps performance of unite O( log(n) )
		if (m_elements[i].m_sz < m_elements[j].m_sz)
		{
			m_elements[i].m_id = j;
			m_elements[j].m_sz += m_elements[i].m_sz;
		}
		else
		{
			m_elements[j].m_id = i;
			m_elements[i].m_sz += m_elements[j].m_sz;
		}
#else
		m_elements[i].m_id = j;
		m_elements[j].m_sz += m_elements[i].m_sz;
#endif  //USE_PATH_COMPRESSION
	}

	int find(int x)
	{
		//btAssert(x < m_N);
		//btAssert(x >= 0);

		while (x != m_elements[x].m_id)
		{
			//not really a reason not to use path compression, and it flattens the trees/improves find performance dramatically

#ifdef USE_PATH_COMPRESSION
			const btElement* elementPtr = &m_elements[m_elements[x].m_id];
			m_elements[x].m_id = elementPtr->m_id;
			x = elementPtr->m_id;
#else  //
			x = m_elements[x].m_id;
#endif
			//btAssert(x < m_N);
			//btAssert(x >= 0);
		}
		return x;
	}
};

#endif  //BT_UNION_FIND_H
