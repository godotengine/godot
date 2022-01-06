/*
 * Box-Box collision detection re-distributed under the ZLib license with permission from Russell L. Smith
 * Original version is from Open Dynamics Engine, Copyright (C) 2001,2002 Russell L. Smith.
 * All rights reserved.  Email: russ@q12.org   Web: www.q12.org

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
#ifndef BT_BOX_BOX_DETECTOR_H
#define BT_BOX_BOX_DETECTOR_H

class btBoxShape;
#include "BulletCollision/NarrowPhaseCollision/btDiscreteCollisionDetectorInterface.h"

/// btBoxBoxDetector wraps the ODE box-box collision detector
/// re-distributed under the Zlib license with permission from Russell L. Smith
struct btBoxBoxDetector : public btDiscreteCollisionDetectorInterface
{
	const btBoxShape* m_box1;
	const btBoxShape* m_box2;

public:
	btBoxBoxDetector(const btBoxShape* box1, const btBoxShape* box2);

	virtual ~btBoxBoxDetector(){};

	virtual void getClosestPoints(const ClosestPointInput& input, Result& output, class btIDebugDraw* debugDraw, bool swapResults = false);
};

#endif  //BT_BOX_BOX_DETECTOR_H
