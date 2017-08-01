#ifndef B3_CPU_NARROWPHASE_H
#define B3_CPU_NARROWPHASE_H

#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Collidable.h"
#include "Bullet3Common/b3AlignedObjectArray.h"
#include "Bullet3Common/b3Vector3.h"
#include "Bullet3Collision/BroadPhaseCollision/shared/b3Aabb.h"
#include "Bullet3Common/shared/b3Int4.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3RigidBodyData.h"
#include "Bullet3Collision/NarrowPhaseCollision/shared/b3Contact4Data.h"

class b3CpuNarrowPhase
{
protected:

	struct b3CpuNarrowPhaseInternalData*	m_data;
	int m_acceleratedCompanionShapeIndex;
	int m_planeBodyIndex;
	int	m_static0Index;

	int registerConvexHullShapeInternal(class b3ConvexUtility* convexPtr,b3Collidable& col);
	int registerConcaveMeshShape(b3AlignedObjectArray<b3Vector3>* vertices, b3AlignedObjectArray<int>* indices, b3Collidable& col, const float* scaling);

public:

	


	b3CpuNarrowPhase(const struct b3Config& config);

	virtual ~b3CpuNarrowPhase(void);

	int		registerSphereShape(float radius);
	int		registerPlaneShape(const b3Vector3& planeNormal, float planeConstant);

	int registerCompoundShape(b3AlignedObjectArray<b3GpuChildShape>* childShapes);
	int registerFace(const b3Vector3& faceNormal, float faceConstant);
	
	int	registerConcaveMesh(b3AlignedObjectArray<b3Vector3>* vertices, b3AlignedObjectArray<int>* indices,const float* scaling);
	
	//do they need to be merged?
	
	int	registerConvexHullShape(b3ConvexUtility* utilPtr);
	int	registerConvexHullShape(const float* vertices, int strideInBytes, int numVertices, const float* scaling);

	//int registerRigidBody(int collidableIndex, float mass, const float* position, const float* orientation, const float* aabbMin, const float* aabbMax,bool writeToGpu);
	void setObjectTransform(const float* position, const float* orientation , int bodyIndex);

	void	writeAllBodiesToGpu();
	void  reset();
	void	readbackAllBodiesToCpu();
	bool	getObjectTransformFromCpu(float* position, float* orientation , int bodyIndex) const;

	void setObjectTransformCpu(float* position, float* orientation , int bodyIndex);
	void setObjectVelocityCpu(float* linVel, float* angVel, int bodyIndex);

	
	//virtual void computeContacts(cl_mem broadphasePairs, int numBroadphasePairs, cl_mem aabbsWorldSpace, int numObjects);
	virtual void computeContacts(b3AlignedObjectArray<b3Int4>& pairs, b3AlignedObjectArray<b3Aabb>& aabbsWorldSpace, b3AlignedObjectArray<b3RigidBodyData>& bodies);


	
	const struct b3RigidBodyData* getBodiesCpu() const;
	//struct b3RigidBodyData* getBodiesCpu();

	int	getNumBodiesGpu() const;

	
	int	getNumBodyInertiasGpu() const;

	
	const struct b3Collidable* getCollidablesCpu() const;
	int		getNumCollidablesGpu() const;


	/*const struct b3Contact4* getContactsCPU() const;

	
	int	getNumContactsGpu() const;
	*/

	const b3AlignedObjectArray<b3Contact4Data>& getContacts() const;
	
	
	int getNumRigidBodies() const;

	int allocateCollidable();

	int getStatic0Index() const
	{
		return m_static0Index;
	}
	b3Collidable& getCollidableCpu(int collidableIndex);
	const b3Collidable& getCollidableCpu(int collidableIndex) const;

	const b3CpuNarrowPhaseInternalData*	getInternalData() const
	{
			return m_data;
	}

	const struct b3Aabb& getLocalSpaceAabb(int collidableIndex) const;
};

#endif //B3_CPU_NARROWPHASE_H

