/*
Copyright (c) 2013 Advanced Micro Devices, Inc.  

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose, 
including commercial applications, and to alter it and redistribute it freely, 
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/
//Originally written by Erwin Coumans

#ifndef B3_CPU_RIGIDBODY_PIPELINE_H
#define B3_CPU_RIGIDBODY_PIPELINE_H



#include "Bullet3Common/b3AlignedObjectArray.h"
#include "Bullet3Collision/NarrowPhaseCollision/b3RaycastInfo.h"

class b3CpuRigidBodyPipeline
{
protected:
	struct b3CpuRigidBodyPipelineInternalData*	m_data;

	int allocateCollidable();

public:


	b3CpuRigidBodyPipeline(class b3CpuNarrowPhase* narrowphase, struct b3DynamicBvhBroadphase* broadphaseDbvt, const struct b3Config& config);
	virtual ~b3CpuRigidBodyPipeline();

	virtual void	stepSimulation(float deltaTime);
	virtual void	integrate(float timeStep);
	virtual void	updateAabbWorldSpace();
	virtual void	computeOverlappingPairs();
	virtual void	computeContactPoints();
	virtual void	solveContactConstraints();

	int		registerConvexPolyhedron(class b3ConvexUtility* convex);

	int		registerPhysicsInstance(float mass, const float* position, const float* orientation, int collisionShapeIndex, int userData);
	void	writeAllInstancesToGpu();
	void	copyConstraintsToHost();
	void	setGravity(const float* grav);
	void	reset();
	
	int createPoint2PointConstraint(int bodyA, int bodyB, const float* pivotInA, const float* pivotInB,float breakingThreshold);
	int createFixedConstraint(int bodyA, int bodyB, const float* pivotInA, const float* pivotInB, const float* relTargetAB, float breakingThreshold);
	void removeConstraintByUid(int uid);

	void	addConstraint(class b3TypedConstraint* constraint);
	void	removeConstraint(b3TypedConstraint* constraint);

	void	castRays(const b3AlignedObjectArray<b3RayInfo>& rays,	b3AlignedObjectArray<b3RayHit>& hitResults);

	const struct b3RigidBodyData* getBodyBuffer() const;

	int	getNumBodies() const;

};

#endif //B3_CPU_RIGIDBODY_PIPELINE_H