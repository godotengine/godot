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

#ifndef B3_GPU_RIGIDBODY_PIPELINE_H
#define B3_GPU_RIGIDBODY_PIPELINE_H

#include "Bullet3OpenCL/Initialize/b3OpenCLInclude.h"
#include "Bullet3Collision/NarrowPhaseCollision/b3Config.h"

#include "Bullet3Common/b3AlignedObjectArray.h"
#include "Bullet3Collision/NarrowPhaseCollision/b3RaycastInfo.h"

class b3GpuRigidBodyPipeline
{
protected:
	struct b3GpuRigidBodyPipelineInternalData*	m_data;

	int allocateCollidable();

public:


	b3GpuRigidBodyPipeline(cl_context ctx,cl_device_id device, cl_command_queue  q , class b3GpuNarrowPhase* narrowphase, class b3GpuBroadphaseInterface* broadphaseSap, struct b3DynamicBvhBroadphase* broadphaseDbvt, const b3Config& config);
	virtual ~b3GpuRigidBodyPipeline();

	void	stepSimulation(float deltaTime);
	void	integrate(float timeStep);
	void	setupGpuAabbsFull();

	int		registerConvexPolyhedron(class b3ConvexUtility* convex);

	//int		registerConvexPolyhedron(const float* vertices, int strideInBytes, int numVertices, const float* scaling);
	//int		registerSphereShape(float radius);
	//int		registerPlaneShape(const b3Vector3& planeNormal, float planeConstant);
	
	//int		registerConcaveMesh(b3AlignedObjectArray<b3Vector3>* vertices, b3AlignedObjectArray<int>* indices, const float* scaling);
	//int		registerCompoundShape(b3AlignedObjectArray<b3GpuChildShape>* childShapes);

	
	int		registerPhysicsInstance(float mass, const float* position, const float* orientation, int collisionShapeIndex, int userData, bool writeInstanceToGpu);
	//if you passed "writeInstanceToGpu" false in the registerPhysicsInstance method (for performance) you need to call writeAllInstancesToGpu after all instances are registered
	void	writeAllInstancesToGpu();
	void	copyConstraintsToHost();
	void	setGravity(const float* grav);
	void reset();
	
	int createPoint2PointConstraint(int bodyA, int bodyB, const float* pivotInA, const float* pivotInB,float breakingThreshold);
	int createFixedConstraint(int bodyA, int bodyB, const float* pivotInA, const float* pivotInB, const float* relTargetAB, float breakingThreshold);
	void removeConstraintByUid(int uid);

	void	addConstraint(class b3TypedConstraint* constraint);
	void	removeConstraint(b3TypedConstraint* constraint);

	void	castRays(const b3AlignedObjectArray<b3RayInfo>& rays,	b3AlignedObjectArray<b3RayHit>& hitResults);

	cl_mem	getBodyBuffer();

	int	getNumBodies() const;

};

#endif //B3_GPU_RIGIDBODY_PIPELINE_H