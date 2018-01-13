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

#ifndef BT_OPTIMIZED_BVH_H
#define BT_OPTIMIZED_BVH_H

#include "BulletCollision/BroadphaseCollision/btQuantizedBvh.h"

class btStridingMeshInterface;


///The btOptimizedBvh extends the btQuantizedBvh to create AABB tree for triangle meshes, through the btStridingMeshInterface.
ATTRIBUTE_ALIGNED16(class) btOptimizedBvh : public btQuantizedBvh
{
	
public:
	BT_DECLARE_ALIGNED_ALLOCATOR();

protected:

public:

	btOptimizedBvh();

	virtual ~btOptimizedBvh();

	void	build(btStridingMeshInterface* triangles,bool useQuantizedAabbCompression, const btVector3& bvhAabbMin, const btVector3& bvhAabbMax);

	void	refit(btStridingMeshInterface* triangles,const btVector3& aabbMin,const btVector3& aabbMax);

	void	refitPartial(btStridingMeshInterface* triangles,const btVector3& aabbMin, const btVector3& aabbMax);

	void	updateBvhNodes(btStridingMeshInterface* meshInterface,int firstNode,int endNode,int index);

	/// Data buffer MUST be 16 byte aligned
	virtual bool serializeInPlace(void *o_alignedDataBuffer, unsigned i_dataBufferSize, bool i_swapEndian) const
	{
		return btQuantizedBvh::serialize(o_alignedDataBuffer,i_dataBufferSize,i_swapEndian);

	}

	///deSerializeInPlace loads and initializes a BVH from a buffer in memory 'in place'
	static btOptimizedBvh *deSerializeInPlace(void *i_alignedDataBuffer, unsigned int i_dataBufferSize, bool i_swapEndian);


};


#endif //BT_OPTIMIZED_BVH_H


