/*
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

#ifndef BT_SPHERE_TRIANGLE_DETECTOR_H
#define BT_SPHERE_TRIANGLE_DETECTOR_H

#include "BulletCollision/NarrowPhaseCollision/btDiscreteCollisionDetectorInterface.h"

class btSphereShape;
class btTriangleShape;

/// sphere-triangle to match the btDiscreteCollisionDetectorInterface
struct SphereTriangleDetector : public btDiscreteCollisionDetectorInterface
{
	virtual void getClosestPoints(const ClosestPointInput& input, Result& output, class btIDebugDraw* debugDraw, bool swapResults = false);

	SphereTriangleDetector(btSphereShape* sphere, btTriangleShape* triangle, btScalar contactBreakingThreshold);

	virtual ~SphereTriangleDetector(){};

	bool collide(const btVector3& sphereCenter, btVector3& point, btVector3& resultNormal, btScalar& depth, btScalar& timeOfImpact, btScalar contactBreakingThreshold);

private:
	bool pointInTriangle(const btVector3 vertices[], const btVector3& normal, btVector3* p);
	bool facecontains(const btVector3& p, const btVector3* vertices, btVector3& normal);

	btSphereShape* m_sphere;
	btTriangleShape* m_triangle;
	btScalar m_contactBreakingThreshold;
};
#endif  //BT_SPHERE_TRIANGLE_DETECTOR_H
