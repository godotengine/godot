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

#include "btUnionFind.h"

btUnionFind::~btUnionFind()
{
	Free();
}

btUnionFind::btUnionFind()
{
}

void btUnionFind::allocate(int N)
{
	m_elements.resize(N);
}
void btUnionFind::Free()
{
	m_elements.clear();
}

void btUnionFind::reset(int N)
{
	allocate(N);

	for (int i = 0; i < N; i++)
	{
		m_elements[i].m_id = i;
		m_elements[i].m_sz = 1;
	}
}

class btUnionFindElementSortPredicate
{
public:
	bool operator()(const btElement& lhs, const btElement& rhs) const
	{
		return lhs.m_id < rhs.m_id;
	}
};

///this is a special operation, destroying the content of btUnionFind.
///it sorts the elements, based on island id, in order to make it easy to iterate over islands
void btUnionFind::sortIslands()
{
	//first store the original body index, and islandId
	int numElements = m_elements.size();

	for (int i = 0; i < numElements; i++)
	{
		m_elements[i].m_id = find(i);
#ifndef STATIC_SIMULATION_ISLAND_OPTIMIZATION
		m_elements[i].m_sz = i;
#endif  //STATIC_SIMULATION_ISLAND_OPTIMIZATION
	}

	// Sort the vector using predicate and std::sort
	//std::sort(m_elements.begin(), m_elements.end(), btUnionFindElementSortPredicate);
	m_elements.quickSort(btUnionFindElementSortPredicate());
}
