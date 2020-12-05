/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2009 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

///Contains contributions from Disney Studio's

#ifndef B3_OPTIMIZED_BVH_H
#define B3_OPTIMIZED_BVH_H

#include "b3QuantizedBvh.h"

class b3StridingMeshInterface;

///The b3OptimizedBvh extends the b3QuantizedBvh to create AABB tree for triangle meshes, through the b3StridingMeshInterface.
B3_ATTRIBUTE_ALIGNED16(class)
b3OptimizedBvh : public b3QuantizedBvh
{
public:
	B3_DECLARE_ALIGNED_ALLOCATOR();

protected:
public:
	b3OptimizedBvh();

	virtual ~b3OptimizedBvh();

	void build(b3StridingMeshInterface * triangles, bool useQuantizedAabbCompression, const b3Vector3& bvhAabbMin, const b3Vector3& bvhAabbMax);

	void refit(b3StridingMeshInterface * triangles, const b3Vector3& aabbMin, const b3Vector3& aabbMax);

	void refitPartial(b3StridingMeshInterface * triangles, const b3Vector3& aabbMin, const b3Vector3& aabbMax);

	void updateBvhNodes(b3StridingMeshInterface * meshInterface, int firstNode, int endNode, int index);

	/// Data buffer MUST be 16 byte aligned
	virtual bool serializeInPlace(void* o_alignedDataBuffer, unsigned i_dataBufferSize, bool i_swapEndian) const
	{
		return b3QuantizedBvh::serialize(o_alignedDataBuffer, i_dataBufferSize, i_swapEndian);
	}

	///deSerializeInPlace loads and initializes a BVH from a buffer in memory 'in place'
	static b3OptimizedBvh* deSerializeInPlace(void* i_alignedDataBuffer, unsigned int i_dataBufferSize, bool i_swapEndian);
};

#endif  //B3_OPTIMIZED_BVH_H
