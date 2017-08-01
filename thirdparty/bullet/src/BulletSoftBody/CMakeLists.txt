
INCLUDE_DIRECTORIES(
${BULLET_PHYSICS_SOURCE_DIR}/src

)

#SUBDIRS( Solvers )

SET(BulletSoftBody_SRCS
	btSoftBody.cpp
	btSoftBodyConcaveCollisionAlgorithm.cpp
	btSoftBodyHelpers.cpp
	btSoftBodyRigidBodyCollisionConfiguration.cpp
	btSoftRigidCollisionAlgorithm.cpp
	btSoftRigidDynamicsWorld.cpp
	btSoftMultiBodyDynamicsWorld.cpp
	btSoftSoftCollisionAlgorithm.cpp
	btDefaultSoftBodySolver.cpp

)

SET(BulletSoftBody_HDRS
	btSoftBody.h
	btSoftBodyData.h
	btSoftBodyConcaveCollisionAlgorithm.h
	btSoftBodyHelpers.h
	btSoftBodyRigidBodyCollisionConfiguration.h
	btSoftRigidCollisionAlgorithm.h
	btSoftRigidDynamicsWorld.h
	btSoftMultiBodyDynamicsWorld.h
	btSoftSoftCollisionAlgorithm.h
	btSparseSDF.h

	btSoftBodySolvers.h
	btDefaultSoftBodySolver.h

	btSoftBodySolverVertexBuffer.h
)



ADD_LIBRARY(BulletSoftBody  ${BulletSoftBody_SRCS} ${BulletSoftBody_HDRS})
SET_TARGET_PROPERTIES(BulletSoftBody PROPERTIES VERSION ${BULLET_VERSION})
SET_TARGET_PROPERTIES(BulletSoftBody PROPERTIES SOVERSION ${BULLET_VERSION})
IF (BUILD_SHARED_LIBS)
	TARGET_LINK_LIBRARIES(BulletSoftBody BulletDynamics)
ENDIF (BUILD_SHARED_LIBS)

IF (INSTALL_LIBS)
	IF (NOT INTERNAL_CREATE_DISTRIBUTABLE_MSVC_PROJECTFILES)
		IF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 2.5)
			IF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
				INSTALL(TARGETS BulletSoftBody DESTINATION .)
			ELSE (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
			INSTALL(TARGETS BulletSoftBody RUNTIME DESTINATION bin
							LIBRARY DESTINATION lib${LIB_SUFFIX}
							ARCHIVE DESTINATION lib${LIB_SUFFIX})
				INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
DESTINATION ${INCLUDE_INSTALL_DIR} FILES_MATCHING PATTERN "*.h"  PATTERN
".svn" EXCLUDE PATTERN "CMakeFiles" EXCLUDE)
			ENDIF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
		ENDIF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 2.5)

		IF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
			SET_TARGET_PROPERTIES(BulletSoftBody PROPERTIES FRAMEWORK true)
			SET_TARGET_PROPERTIES(BulletSoftBody PROPERTIES PUBLIC_HEADER "${BulletSoftBody_HDRS}")
		ENDIF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
	ENDIF (NOT INTERNAL_CREATE_DISTRIBUTABLE_MSVC_PROJECTFILES)
ENDIF (INSTALL_LIBS)
