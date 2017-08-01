INCLUDE_DIRECTORIES( ${BULLET_PHYSICS_SOURCE_DIR}/src  )

SET(BulletCollision_SRCS
	BroadphaseCollision/btAxisSweep3.cpp
	BroadphaseCollision/btBroadphaseProxy.cpp
	BroadphaseCollision/btCollisionAlgorithm.cpp
	BroadphaseCollision/btDbvt.cpp
	BroadphaseCollision/btDbvtBroadphase.cpp
	BroadphaseCollision/btDispatcher.cpp
	BroadphaseCollision/btOverlappingPairCache.cpp
	BroadphaseCollision/btQuantizedBvh.cpp
	BroadphaseCollision/btSimpleBroadphase.cpp
	CollisionDispatch/btActivatingCollisionAlgorithm.cpp
	CollisionDispatch/btBoxBoxCollisionAlgorithm.cpp
	CollisionDispatch/btBox2dBox2dCollisionAlgorithm.cpp
	CollisionDispatch/btBoxBoxDetector.cpp
	CollisionDispatch/btCollisionDispatcher.cpp
	CollisionDispatch/btCollisionDispatcherMt.cpp
	CollisionDispatch/btCollisionObject.cpp
	CollisionDispatch/btCollisionWorld.cpp
	CollisionDispatch/btCollisionWorldImporter.cpp
	CollisionDispatch/btCompoundCollisionAlgorithm.cpp
	CollisionDispatch/btCompoundCompoundCollisionAlgorithm.cpp
	CollisionDispatch/btConvexConcaveCollisionAlgorithm.cpp
	CollisionDispatch/btConvexConvexAlgorithm.cpp
	CollisionDispatch/btConvexPlaneCollisionAlgorithm.cpp
	CollisionDispatch/btConvex2dConvex2dAlgorithm.cpp
	CollisionDispatch/btDefaultCollisionConfiguration.cpp
	CollisionDispatch/btEmptyCollisionAlgorithm.cpp
	CollisionDispatch/btGhostObject.cpp
	CollisionDispatch/btHashedSimplePairCache.cpp
	CollisionDispatch/btInternalEdgeUtility.cpp
	CollisionDispatch/btInternalEdgeUtility.h
	CollisionDispatch/btManifoldResult.cpp
	CollisionDispatch/btSimulationIslandManager.cpp
	CollisionDispatch/btSphereBoxCollisionAlgorithm.cpp
	CollisionDispatch/btSphereSphereCollisionAlgorithm.cpp
	CollisionDispatch/btSphereTriangleCollisionAlgorithm.cpp
	CollisionDispatch/btUnionFind.cpp
	CollisionDispatch/SphereTriangleDetector.cpp
	CollisionShapes/btBoxShape.cpp
	CollisionShapes/btBox2dShape.cpp
	CollisionShapes/btBvhTriangleMeshShape.cpp
	CollisionShapes/btCapsuleShape.cpp
	CollisionShapes/btCollisionShape.cpp
	CollisionShapes/btCompoundShape.cpp
	CollisionShapes/btConcaveShape.cpp
	CollisionShapes/btConeShape.cpp
	CollisionShapes/btConvexHullShape.cpp
	CollisionShapes/btConvexInternalShape.cpp
	CollisionShapes/btConvexPointCloudShape.cpp
	CollisionShapes/btConvexPolyhedron.cpp
	CollisionShapes/btConvexShape.cpp
	CollisionShapes/btConvex2dShape.cpp
	CollisionShapes/btConvexTriangleMeshShape.cpp
	CollisionShapes/btCylinderShape.cpp
	CollisionShapes/btEmptyShape.cpp
	CollisionShapes/btHeightfieldTerrainShape.cpp
	CollisionShapes/btMinkowskiSumShape.cpp
	CollisionShapes/btMultimaterialTriangleMeshShape.cpp
	CollisionShapes/btMultiSphereShape.cpp
	CollisionShapes/btOptimizedBvh.cpp
	CollisionShapes/btPolyhedralConvexShape.cpp
	CollisionShapes/btScaledBvhTriangleMeshShape.cpp
	CollisionShapes/btShapeHull.cpp
	CollisionShapes/btSphereShape.cpp
	CollisionShapes/btStaticPlaneShape.cpp
	CollisionShapes/btStridingMeshInterface.cpp
	CollisionShapes/btTetrahedronShape.cpp
	CollisionShapes/btTriangleBuffer.cpp
	CollisionShapes/btTriangleCallback.cpp
	CollisionShapes/btTriangleIndexVertexArray.cpp
	CollisionShapes/btTriangleIndexVertexMaterialArray.cpp
	CollisionShapes/btTriangleMesh.cpp
	CollisionShapes/btTriangleMeshShape.cpp
	CollisionShapes/btUniformScalingShape.cpp
	Gimpact/btContactProcessing.cpp
	Gimpact/btGenericPoolAllocator.cpp
	Gimpact/btGImpactBvh.cpp
	Gimpact/btGImpactCollisionAlgorithm.cpp
	Gimpact/btGImpactQuantizedBvh.cpp
	Gimpact/btGImpactShape.cpp
	Gimpact/btTriangleShapeEx.cpp
	Gimpact/gim_box_set.cpp
	Gimpact/gim_contact.cpp
	Gimpact/gim_memory.cpp
	Gimpact/gim_tri_collision.cpp
	NarrowPhaseCollision/btContinuousConvexCollision.cpp
	NarrowPhaseCollision/btConvexCast.cpp
	NarrowPhaseCollision/btGjkConvexCast.cpp
	NarrowPhaseCollision/btGjkEpa2.cpp
	NarrowPhaseCollision/btGjkEpaPenetrationDepthSolver.cpp
	NarrowPhaseCollision/btGjkPairDetector.cpp
	NarrowPhaseCollision/btMinkowskiPenetrationDepthSolver.cpp
	NarrowPhaseCollision/btPersistentManifold.cpp
	NarrowPhaseCollision/btRaycastCallback.cpp
	NarrowPhaseCollision/btSubSimplexConvexCast.cpp
	NarrowPhaseCollision/btVoronoiSimplexSolver.cpp
	NarrowPhaseCollision/btPolyhedralContactClipping.cpp
)

SET(Root_HDRS
	../btBulletCollisionCommon.h
)
SET(BroadphaseCollision_HDRS
    BroadphaseCollision/btAxisSweep3Internal.h
	BroadphaseCollision/btAxisSweep3.h
	BroadphaseCollision/btBroadphaseInterface.h
	BroadphaseCollision/btBroadphaseProxy.h
	BroadphaseCollision/btCollisionAlgorithm.h
	BroadphaseCollision/btDbvt.h
	BroadphaseCollision/btDbvtBroadphase.h
	BroadphaseCollision/btDispatcher.h
	BroadphaseCollision/btOverlappingPairCache.h
	BroadphaseCollision/btOverlappingPairCallback.h
	BroadphaseCollision/btQuantizedBvh.h
	BroadphaseCollision/btSimpleBroadphase.h
)
SET(CollisionDispatch_HDRS
	CollisionDispatch/btActivatingCollisionAlgorithm.h
	CollisionDispatch/btBoxBoxCollisionAlgorithm.h
	CollisionDispatch/btBox2dBox2dCollisionAlgorithm.h
	CollisionDispatch/btBoxBoxDetector.h
	CollisionDispatch/btCollisionConfiguration.h
	CollisionDispatch/btCollisionCreateFunc.h
	CollisionDispatch/btCollisionDispatcher.h
	CollisionDispatch/btCollisionDispatcherMt.h
	CollisionDispatch/btCollisionObject.h
	CollisionDispatch/btCollisionObjectWrapper.h
	CollisionDispatch/btCollisionWorld.h
	CollisionDispatch/btCollisionWorldImporter.h
	CollisionDispatch/btCompoundCollisionAlgorithm.h
	CollisionDispatch/btCompoundCompoundCollisionAlgorithm.h
	CollisionDispatch/btConvexConcaveCollisionAlgorithm.h
	CollisionDispatch/btConvexConvexAlgorithm.h
	CollisionDispatch/btConvex2dConvex2dAlgorithm.h
	CollisionDispatch/btConvexPlaneCollisionAlgorithm.h
	CollisionDispatch/btDefaultCollisionConfiguration.h
	CollisionDispatch/btEmptyCollisionAlgorithm.h
	CollisionDispatch/btGhostObject.h
	CollisionDispatch/btHashedSimplePairCache.h
	CollisionDispatch/btManifoldResult.h
	CollisionDispatch/btSimulationIslandManager.h
	CollisionDispatch/btSphereBoxCollisionAlgorithm.h
	CollisionDispatch/btSphereSphereCollisionAlgorithm.h
	CollisionDispatch/btSphereTriangleCollisionAlgorithm.h
	CollisionDispatch/btUnionFind.h
	CollisionDispatch/SphereTriangleDetector.h
)
SET(CollisionShapes_HDRS
	CollisionShapes/btBoxShape.h
	CollisionShapes/btBox2dShape.h
	CollisionShapes/btBvhTriangleMeshShape.h
	CollisionShapes/btCapsuleShape.h
	CollisionShapes/btCollisionMargin.h
	CollisionShapes/btCollisionShape.h
	CollisionShapes/btCompoundShape.h
	CollisionShapes/btConcaveShape.h
	CollisionShapes/btConeShape.h
	CollisionShapes/btConvexHullShape.h
	CollisionShapes/btConvexInternalShape.h
	CollisionShapes/btConvexPointCloudShape.h
	CollisionShapes/btConvexPolyhedron.h
	CollisionShapes/btConvexShape.h
	CollisionShapes/btConvex2dShape.h
	CollisionShapes/btConvexTriangleMeshShape.h
	CollisionShapes/btCylinderShape.h
	CollisionShapes/btEmptyShape.h
	CollisionShapes/btHeightfieldTerrainShape.h
	CollisionShapes/btMaterial.h
	CollisionShapes/btMinkowskiSumShape.h
	CollisionShapes/btMultimaterialTriangleMeshShape.h
	CollisionShapes/btMultiSphereShape.h
	CollisionShapes/btOptimizedBvh.h
	CollisionShapes/btPolyhedralConvexShape.h
	CollisionShapes/btScaledBvhTriangleMeshShape.h
	CollisionShapes/btShapeHull.h
	CollisionShapes/btSphereShape.h
	CollisionShapes/btStaticPlaneShape.h
	CollisionShapes/btStridingMeshInterface.h
	CollisionShapes/btTetrahedronShape.h
	CollisionShapes/btTriangleBuffer.h
	CollisionShapes/btTriangleCallback.h
	CollisionShapes/btTriangleIndexVertexArray.h
	CollisionShapes/btTriangleIndexVertexMaterialArray.h
	CollisionShapes/btTriangleInfoMap.h
	CollisionShapes/btTriangleMesh.h
	CollisionShapes/btTriangleMeshShape.h
	CollisionShapes/btTriangleShape.h
	CollisionShapes/btUniformScalingShape.h
)
SET(Gimpact_HDRS
	Gimpact/btBoxCollision.h
	Gimpact/btClipPolygon.h
	Gimpact/btContactProcessingStructs.h
	Gimpact/btContactProcessing.h
	Gimpact/btGenericPoolAllocator.h
	Gimpact/btGeometryOperations.h
	Gimpact/btGImpactBvhStructs.h
	Gimpact/btGImpactBvh.h
	Gimpact/btGImpactCollisionAlgorithm.h
	Gimpact/btGImpactMassUtil.h
	Gimpact/btGImpactQuantizedBvhStructs.h
	Gimpact/btGImpactQuantizedBvh.h
	Gimpact/btGImpactShape.h
	Gimpact/btQuantization.h
	Gimpact/btTriangleShapeEx.h
	Gimpact/gim_array.h
	Gimpact/gim_basic_geometry_operations.h
	Gimpact/gim_bitset.h
	Gimpact/gim_box_collision.h
	Gimpact/gim_box_set.h
	Gimpact/gim_clip_polygon.h
	Gimpact/gim_contact.h
	Gimpact/gim_geom_types.h
	Gimpact/gim_geometry.h
	Gimpact/gim_hash_table.h
	Gimpact/gim_linear_math.h
	Gimpact/gim_math.h
	Gimpact/gim_memory.h
	Gimpact/gim_radixsort.h
	Gimpact/gim_tri_collision.h
)
SET(NarrowPhaseCollision_HDRS
	NarrowPhaseCollision/btContinuousConvexCollision.h
	NarrowPhaseCollision/btConvexCast.h
	NarrowPhaseCollision/btConvexPenetrationDepthSolver.h
	NarrowPhaseCollision/btDiscreteCollisionDetectorInterface.h
	NarrowPhaseCollision/btGjkConvexCast.h
	NarrowPhaseCollision/btGjkEpa2.h
	NarrowPhaseCollision/btGjkEpaPenetrationDepthSolver.h
	NarrowPhaseCollision/btGjkPairDetector.h
	NarrowPhaseCollision/btManifoldPoint.h
	NarrowPhaseCollision/btMinkowskiPenetrationDepthSolver.h
	NarrowPhaseCollision/btPersistentManifold.h
	NarrowPhaseCollision/btPointCollector.h
	NarrowPhaseCollision/btRaycastCallback.h
	NarrowPhaseCollision/btSimplexSolverInterface.h
	NarrowPhaseCollision/btSubSimplexConvexCast.h
	NarrowPhaseCollision/btVoronoiSimplexSolver.h
	NarrowPhaseCollision/btPolyhedralContactClipping.h
)

SET(BulletCollision_HDRS
	${Root_HDRS}
	${BroadphaseCollision_HDRS}
	${CollisionDispatch_HDRS}
	${CollisionShapes_HDRS}
	${Gimpact_HDRS}
	${NarrowPhaseCollision_HDRS}
)


ADD_LIBRARY(BulletCollision ${BulletCollision_SRCS} ${BulletCollision_HDRS})
SET_TARGET_PROPERTIES(BulletCollision PROPERTIES VERSION ${BULLET_VERSION})
SET_TARGET_PROPERTIES(BulletCollision PROPERTIES SOVERSION ${BULLET_VERSION})
IF (BUILD_SHARED_LIBS)
  TARGET_LINK_LIBRARIES(BulletCollision LinearMath)
ENDIF (BUILD_SHARED_LIBS)


IF (INSTALL_LIBS)
	IF (NOT INTERNAL_CREATE_DISTRIBUTABLE_MSVC_PROJECTFILES)
		#INSTALL of other files requires CMake 2.6
		IF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 2.5)
			IF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
				INSTALL(TARGETS BulletCollision DESTINATION .)
			ELSE (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
				INSTALL(TARGETS BulletCollision RUNTIME DESTINATION bin
								LIBRARY DESTINATION lib${LIB_SUFFIX}
								ARCHIVE DESTINATION lib${LIB_SUFFIX})
				INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
DESTINATION ${INCLUDE_INSTALL_DIR} FILES_MATCHING PATTERN "*.h" PATTERN ".svn" EXCLUDE PATTERN "CMakeFiles" EXCLUDE)
				INSTALL(FILES ../btBulletCollisionCommon.h
DESTINATION ${INCLUDE_INSTALL_DIR}/BulletCollision)
			ENDIF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
		ENDIF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 2.5)

		IF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
			SET_TARGET_PROPERTIES(BulletCollision PROPERTIES FRAMEWORK true)

			SET_TARGET_PROPERTIES(BulletCollision PROPERTIES PUBLIC_HEADER "${Root_HDRS}")
			# Have to list out sub-directories manually:
			SET_PROPERTY(SOURCE ${BroadphaseCollision_HDRS} PROPERTY MACOSX_PACKAGE_LOCATION Headers/BroadphaseCollision)
			SET_PROPERTY(SOURCE ${CollisionDispatch_HDRS} PROPERTY MACOSX_PACKAGE_LOCATION Headers/CollisionDispatch)
			SET_PROPERTY(SOURCE ${CollisionShapes_HDRS} PROPERTY MACOSX_PACKAGE_LOCATION Headers/CollisionShapes)
			SET_PROPERTY(SOURCE ${Gimpact_HDRS} PROPERTY MACOSX_PACKAGE_LOCATION Headers/Gimpact)
			SET_PROPERTY(SOURCE ${NarrowPhaseCollision_HDRS} PROPERTY MACOSX_PACKAGE_LOCATION Headers/NarrowPhaseCollision)

		ENDIF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
	ENDIF (NOT INTERNAL_CREATE_DISTRIBUTABLE_MSVC_PROJECTFILES)
ENDIF (INSTALL_LIBS)
