//Bullet Continuous Collision Detection and Physics Library
//Copyright (c) 2003-2006 Erwin Coumans  https://bulletphysics.org

//
// btAxisSweep3.h
//
// Copyright (c) 2006 Simon Hobbs
//
// This software is provided 'as-is', without any express or implied warranty. In no event will the authors be held liable for any damages arising from the use of this software.
//
// Permission is granted to anyone to use this software for any purpose, including commercial applications, and to alter it and redistribute it freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
//
// 2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
//
// 3. This notice may not be removed or altered from any source distribution.

#ifndef BT_AXIS_SWEEP_3_H
#define BT_AXIS_SWEEP_3_H

#include "LinearMath/btVector3.h"
#include "btOverlappingPairCache.h"
#include "btBroadphaseInterface.h"
#include "btBroadphaseProxy.h"
#include "btOverlappingPairCallback.h"
#include "btDbvtBroadphase.h"
#include "btAxisSweep3Internal.h"

/// The btAxisSweep3 is an efficient implementation of the 3d axis sweep and prune broadphase.
/// It uses arrays rather then lists for storage of the 3 axis. Also it operates using 16 bit integer coordinates instead of floats.
/// For large worlds and many objects, use bt32BitAxisSweep3 or btDbvtBroadphase instead. bt32BitAxisSweep3 has higher precision and allows more then 16384 objects at the cost of more memory and bit of performance.
class btAxisSweep3 : public btAxisSweep3Internal<unsigned short int>
{
public:
	btAxisSweep3(const btVector3& worldAabbMin, const btVector3& worldAabbMax, unsigned short int maxHandles = 16384, btOverlappingPairCache* pairCache = 0, bool disableRaycastAccelerator = false);
};

/// The bt32BitAxisSweep3 allows higher precision quantization and more objects compared to the btAxisSweep3 sweep and prune.
/// This comes at the cost of more memory per handle, and a bit slower performance.
/// It uses arrays rather then lists for storage of the 3 axis.
class bt32BitAxisSweep3 : public btAxisSweep3Internal<unsigned int>
{
public:
	bt32BitAxisSweep3(const btVector3& worldAabbMin, const btVector3& worldAabbMax, unsigned int maxHandles = 1500000, btOverlappingPairCache* pairCache = 0, bool disableRaycastAccelerator = false);
};

#endif
