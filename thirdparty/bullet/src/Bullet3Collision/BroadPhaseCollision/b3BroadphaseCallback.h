/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2013 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/

#ifndef B3_BROADPHASE_CALLBACK_H
#define B3_BROADPHASE_CALLBACK_H

#include "Bullet3Common/b3Vector3.h"
struct b3BroadphaseProxy;


struct	b3BroadphaseAabbCallback
{
	virtual ~b3BroadphaseAabbCallback() {}
	virtual bool	process(const b3BroadphaseProxy* proxy) = 0;
};


struct	b3BroadphaseRayCallback : public b3BroadphaseAabbCallback
{
	///added some cached data to accelerate ray-AABB tests
	b3Vector3		m_rayDirectionInverse;
	unsigned int	m_signs[3];
	b3Scalar		m_lambda_max;

	virtual ~b3BroadphaseRayCallback() {}
};

#endif //B3_BROADPHASE_CALLBACK_H
