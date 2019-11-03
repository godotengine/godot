#ifndef B3_GPU_RAYCAST_H
#define B3_GPU_RAYCAST_H

#include "Bullet3Common/b3Vector3.h"
#include "Bullet3OpenCL/Initialize/b3OpenCLInclude.h"

#include "Bullet3Common/b3AlignedObjectArray.h"
#include "Bullet3Collision/NarrowPhaseCollision/b3RaycastInfo.h"

class b3GpuRaycast
{
protected:
	struct b3GpuRaycastInternalData* m_data;

public:
	b3GpuRaycast(cl_context ctx, cl_device_id device, cl_command_queue q);
	virtual ~b3GpuRaycast();

	void castRaysHost(const b3AlignedObjectArray<b3RayInfo>& raysIn, b3AlignedObjectArray<b3RayHit>& hitResults,
					  int numBodies, const struct b3RigidBodyData* bodies, int numCollidables, const struct b3Collidable* collidables,
					  const struct b3GpuNarrowPhaseInternalData* narrowphaseData);

	void castRays(const b3AlignedObjectArray<b3RayInfo>& rays, b3AlignedObjectArray<b3RayHit>& hitResults,
				  int numBodies, const struct b3RigidBodyData* bodies, int numCollidables, const struct b3Collidable* collidables,
				  const struct b3GpuNarrowPhaseInternalData* narrowphaseData, class b3GpuBroadphaseInterface* broadphase);
};

#endif  //B3_GPU_RAYCAST_H
