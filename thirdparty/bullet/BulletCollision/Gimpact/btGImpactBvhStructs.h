#ifndef GIM_BOX_SET_STRUCT_H_INCLUDED
#define GIM_BOX_SET_STRUCT_H_INCLUDED

/*! \file gim_box_set.h
\author Francisco Leon Najera
*/
/*
This source file is part of GIMPACT Library.

For the latest info, see http://gimpact.sourceforge.net/

Copyright (c) 2007 Francisco Leon Najera. C.C. 80087371.
email: projectileman@yahoo.com


This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/


#include "LinearMath/btAlignedObjectArray.h"

#include "btBoxCollision.h"
#include "btTriangleShapeEx.h"

//! Overlapping pair
struct GIM_PAIR
{
    int m_index1;
    int m_index2;
    GIM_PAIR()
    {}

    GIM_PAIR(const GIM_PAIR & p)
    {
    	m_index1 = p.m_index1;
    	m_index2 = p.m_index2;
	}

	GIM_PAIR(int index1, int index2)
    {
    	m_index1 = index1;
    	m_index2 = index2;
	}
};

///GIM_BVH_DATA is an internal GIMPACT collision structure to contain axis aligned bounding box
struct GIM_BVH_DATA
{
	btAABB m_bound;
	int m_data;
};

//! Node Structure for trees
class GIM_BVH_TREE_NODE
{
public:
	btAABB m_bound;
protected:
	int	m_escapeIndexOrDataIndex;
public:
	GIM_BVH_TREE_NODE()
	{
		m_escapeIndexOrDataIndex = 0;
	}

	SIMD_FORCE_INLINE bool isLeafNode() const
	{
		//skipindex is negative (internal node), triangleindex >=0 (leafnode)
		return (m_escapeIndexOrDataIndex>=0);
	}

	SIMD_FORCE_INLINE int getEscapeIndex() const
	{
		//btAssert(m_escapeIndexOrDataIndex < 0);
		return -m_escapeIndexOrDataIndex;
	}

	SIMD_FORCE_INLINE void setEscapeIndex(int index)
	{
		m_escapeIndexOrDataIndex = -index;
	}

	SIMD_FORCE_INLINE int getDataIndex() const
	{
		//btAssert(m_escapeIndexOrDataIndex >= 0);

		return m_escapeIndexOrDataIndex;
	}

	SIMD_FORCE_INLINE void setDataIndex(int index)
	{
		m_escapeIndexOrDataIndex = index;
	}

};

#endif // GIM_BOXPRUNING_H_INCLUDED
