/*
Bullet Continuous Collision Detection and Physics Library
Copyright (c) 2003-2014 Erwin Coumans  http://bulletphysics.org

This software is provided 'as-is', without any express or implied warranty.
In no event will the authors be held liable for any damages arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it freely,
subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not claim that you wrote the original software. If you use this software in a product, an acknowledgment in the product documentation would be appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
*/


#ifndef BT_COLLISION_WORLD_IMPORTER_H
#define BT_COLLISION_WORLD_IMPORTER_H

#include "LinearMath/btTransform.h"
#include "LinearMath/btVector3.h"
#include "LinearMath/btAlignedObjectArray.h"
#include "LinearMath/btHashMap.h"

class btCollisionShape;
class btCollisionObject;
struct btBulletSerializedArrays;


struct ConstraintInput;
class btCollisionWorld;
struct btCollisionShapeData;
class btTriangleIndexVertexArray;
class btStridingMeshInterface;
struct btStridingMeshInterfaceData;
class btGImpactMeshShape;
class btOptimizedBvh;
struct btTriangleInfoMap;
class btBvhTriangleMeshShape;
class btPoint2PointConstraint;
class btHingeConstraint;
class btConeTwistConstraint;
class btGeneric6DofConstraint;
class btGeneric6DofSpringConstraint;
class btSliderConstraint;
class btGearConstraint;
struct btContactSolverInfo;




class btCollisionWorldImporter
{
protected:
	btCollisionWorld* m_collisionWorld;

	int m_verboseMode;

	btAlignedObjectArray<btCollisionShape*>  m_allocatedCollisionShapes;
	btAlignedObjectArray<btCollisionObject*> m_allocatedRigidBodies;

	btAlignedObjectArray<btOptimizedBvh*>	 m_allocatedBvhs;
	btAlignedObjectArray<btTriangleInfoMap*> m_allocatedTriangleInfoMaps;
	btAlignedObjectArray<btTriangleIndexVertexArray*> m_allocatedTriangleIndexArrays;
	btAlignedObjectArray<btStridingMeshInterfaceData*> m_allocatedbtStridingMeshInterfaceDatas;
	btAlignedObjectArray<btCollisionObject*> m_allocatedCollisionObjects;


	btAlignedObjectArray<char*>				m_allocatedNames;

	btAlignedObjectArray<int*>				m_indexArrays;
	btAlignedObjectArray<short int*>		m_shortIndexArrays;
	btAlignedObjectArray<unsigned char*>	m_charIndexArrays;

	btAlignedObjectArray<btVector3FloatData*>	m_floatVertexArrays;
	btAlignedObjectArray<btVector3DoubleData*>	m_doubleVertexArrays;


	btHashMap<btHashPtr,btOptimizedBvh*>	m_bvhMap;
	btHashMap<btHashPtr,btTriangleInfoMap*>	m_timMap;

	btHashMap<btHashString,btCollisionShape*>	m_nameShapeMap;
	btHashMap<btHashString,btCollisionObject*>	m_nameColObjMap;

	btHashMap<btHashPtr,const char*>	m_objectNameMap;

	btHashMap<btHashPtr,btCollisionShape*>	m_shapeMap;
	btHashMap<btHashPtr,btCollisionObject*>	m_bodyMap;


	//methods



	char*	duplicateName(const char* name);

	btCollisionShape* convertCollisionShape(  btCollisionShapeData* shapeData  );


public:

	btCollisionWorldImporter(btCollisionWorld* world);

	virtual ~btCollisionWorldImporter();

    bool	convertAllObjects( btBulletSerializedArrays* arrays);

		///delete all memory collision shapes, rigid bodies, constraints etc. allocated during the load.
	///make sure you don't use the dynamics world containing objects after you call this method
	virtual void deleteAllData();

	void	setVerboseMode(int verboseMode)
	{
		m_verboseMode = verboseMode;
	}

	int getVerboseMode() const
	{
		return m_verboseMode;
	}

		// query for data
	int	getNumCollisionShapes() const;
	btCollisionShape* getCollisionShapeByIndex(int index);
	int getNumRigidBodies() const;
	btCollisionObject* getRigidBodyByIndex(int index) const;

	int getNumBvhs() const;
	btOptimizedBvh*  getBvhByIndex(int index) const;
	int getNumTriangleInfoMaps() const;
	btTriangleInfoMap* getTriangleInfoMapByIndex(int index) const;

	// queris involving named objects
	btCollisionShape* getCollisionShapeByName(const char* name);
	btCollisionObject* getCollisionObjectByName(const char* name);


	const char*	getNameForPointer(const void* ptr) const;

	///those virtuals are called by load and can be overridden by the user



	//bodies

	virtual btCollisionObject*  createCollisionObject(	const btTransform& startTransform,	btCollisionShape* shape,const char* bodyName);

	///shapes

	virtual btCollisionShape* createPlaneShape(const btVector3& planeNormal,btScalar planeConstant);
	virtual btCollisionShape* createBoxShape(const btVector3& halfExtents);
	virtual btCollisionShape* createSphereShape(btScalar radius);
	virtual btCollisionShape* createCapsuleShapeX(btScalar radius, btScalar height);
	virtual btCollisionShape* createCapsuleShapeY(btScalar radius, btScalar height);
	virtual btCollisionShape* createCapsuleShapeZ(btScalar radius, btScalar height);

	virtual btCollisionShape* createCylinderShapeX(btScalar radius,btScalar height);
	virtual btCollisionShape* createCylinderShapeY(btScalar radius,btScalar height);
	virtual btCollisionShape* createCylinderShapeZ(btScalar radius,btScalar height);
	virtual btCollisionShape* createConeShapeX(btScalar radius,btScalar height);
	virtual btCollisionShape* createConeShapeY(btScalar radius,btScalar height);
	virtual btCollisionShape* createConeShapeZ(btScalar radius,btScalar height);
	virtual class btTriangleIndexVertexArray*	createTriangleMeshContainer();
	virtual	btBvhTriangleMeshShape* createBvhTriangleMeshShape(btStridingMeshInterface* trimesh, btOptimizedBvh* bvh);
	virtual btCollisionShape* createConvexTriangleMeshShape(btStridingMeshInterface* trimesh);
#ifdef SUPPORT_GIMPACT_SHAPE_IMPORT
	virtual btGImpactMeshShape* createGimpactShape(btStridingMeshInterface* trimesh);
#endif //SUPPORT_GIMPACT_SHAPE_IMPORT
	virtual btStridingMeshInterfaceData* createStridingMeshInterfaceData(btStridingMeshInterfaceData* interfaceData);

	virtual class btConvexHullShape* createConvexHullShape();
	virtual class btCompoundShape* createCompoundShape();
	virtual class btScaledBvhTriangleMeshShape* createScaledTrangleMeshShape(btBvhTriangleMeshShape* meshShape,const btVector3& localScalingbtBvhTriangleMeshShape);

	virtual class btMultiSphereShape* createMultiSphereShape(const btVector3* positions,const btScalar* radi,int numSpheres);

	virtual btTriangleIndexVertexArray* createMeshInterface(btStridingMeshInterfaceData& meshData);

	///acceleration and connectivity structures
	virtual btOptimizedBvh*	createOptimizedBvh();
	virtual btTriangleInfoMap* createTriangleInfoMap();




};


#endif //BT_WORLD_IMPORTER_H
