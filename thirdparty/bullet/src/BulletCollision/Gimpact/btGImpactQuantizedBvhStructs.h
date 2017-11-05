#ifndef GIM_QUANTIZED_SET_STRUCTS_H_INCLUDED
#define GIM_QUANTIZED_SET_STRUCTS_H_INCLUDED

/*! \file btGImpactQuantizedBvh.h
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

#include "btGImpactBvh.h"
#include "btQuantization.h"

///btQuantizedBvhNode is a compressed aabb node, 16 bytes.
///Node can be used for leafnode or internal node. Leafnodes can point to 32-bit triangle index (non-negative range).
ATTRIBUTE_ALIGNED16	(struct) BT_QUANTIZED_BVH_NODE
{
	//12 bytes
	unsigned short int	m_quantizedAabbMin[3];
	unsigned short int	m_quantizedAabbMax[3];
	//4 bytes
	int	m_escapeIndexOrDataIndex;

	BT_QUANTIZED_BVH_NODE()
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

	SIMD_FORCE_INLINE bool testQuantizedBoxOverlapp(
		unsigned short * quantizedMin,unsigned short * quantizedMax) const
	{
		if(m_quantizedAabbMin[0] > quantizedMax[0] ||
		   m_quantizedAabbMax[0] < quantizedMin[0] ||
		   m_quantizedAabbMin[1] > quantizedMax[1] ||
		   m_quantizedAabbMax[1] < quantizedMin[1] ||
		   m_quantizedAabbMin[2] > quantizedMax[2] ||
		   m_quantizedAabbMax[2] < quantizedMin[2])
		{
			return false;
		}
		return true;
	}

};

#endif // GIM_QUANTIZED_SET_STRUCTS_H_INCLUDED
