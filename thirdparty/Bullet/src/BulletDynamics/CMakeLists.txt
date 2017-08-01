INCLUDE_DIRECTORIES( ${BULLET_PHYSICS_SOURCE_DIR}/src  )



SET(BulletDynamics_SRCS
	Character/btKinematicCharacterController.cpp
	ConstraintSolver/btConeTwistConstraint.cpp
	ConstraintSolver/btContactConstraint.cpp
	ConstraintSolver/btFixedConstraint.cpp
	ConstraintSolver/btGearConstraint.cpp
	ConstraintSolver/btGeneric6DofConstraint.cpp
	ConstraintSolver/btGeneric6DofSpringConstraint.cpp
	ConstraintSolver/btGeneric6DofSpring2Constraint.cpp
	ConstraintSolver/btHinge2Constraint.cpp
	ConstraintSolver/btHingeConstraint.cpp
	ConstraintSolver/btPoint2PointConstraint.cpp
	ConstraintSolver/btSequentialImpulseConstraintSolver.cpp
	ConstraintSolver/btNNCGConstraintSolver.cpp
	ConstraintSolver/btSliderConstraint.cpp
	ConstraintSolver/btSolve2LinearConstraint.cpp
	ConstraintSolver/btTypedConstraint.cpp
	ConstraintSolver/btUniversalConstraint.cpp
	Dynamics/btDiscreteDynamicsWorld.cpp
	Dynamics/btDiscreteDynamicsWorldMt.cpp
	Dynamics/btSimulationIslandManagerMt.cpp
	Dynamics/btRigidBody.cpp
	Dynamics/btSimpleDynamicsWorld.cpp
#	Dynamics/Bullet-C-API.cpp
	Vehicle/btRaycastVehicle.cpp
	Vehicle/btWheelInfo.cpp
	Featherstone/btMultiBody.cpp
	Featherstone/btMultiBodyConstraintSolver.cpp
	Featherstone/btMultiBodyDynamicsWorld.cpp
	Featherstone/btMultiBodyJointLimitConstraint.cpp
	Featherstone/btMultiBodyConstraint.cpp
	Featherstone/btMultiBodyPoint2Point.cpp
	Featherstone/btMultiBodyFixedConstraint.cpp
	Featherstone/btMultiBodySliderConstraint.cpp
	Featherstone/btMultiBodyJointMotor.cpp
	Featherstone/btMultiBodyGearConstraint.cpp
	MLCPSolvers/btDantzigLCP.cpp
	MLCPSolvers/btMLCPSolver.cpp
	MLCPSolvers/btLemkeAlgorithm.cpp
)

SET(Root_HDRS
	../btBulletDynamicsCommon.h
	../btBulletCollisionCommon.h
)
SET(ConstraintSolver_HDRS
	ConstraintSolver/btConeTwistConstraint.h
	ConstraintSolver/btConstraintSolver.h
	ConstraintSolver/btContactConstraint.h
	ConstraintSolver/btContactSolverInfo.h
	ConstraintSolver/btFixedConstraint.h
	ConstraintSolver/btGearConstraint.h
	ConstraintSolver/btGeneric6DofConstraint.h
	ConstraintSolver/btGeneric6DofSpringConstraint.h
	ConstraintSolver/btGeneric6DofSpring2Constraint.h
	ConstraintSolver/btHinge2Constraint.h
	ConstraintSolver/btHingeConstraint.h
	ConstraintSolver/btJacobianEntry.h
	ConstraintSolver/btPoint2PointConstraint.h
	ConstraintSolver/btSequentialImpulseConstraintSolver.h
	ConstraintSolver/btNNCGConstraintSolver.h
	ConstraintSolver/btSliderConstraint.h
	ConstraintSolver/btSolve2LinearConstraint.h
	ConstraintSolver/btSolverBody.h
	ConstraintSolver/btSolverConstraint.h
	ConstraintSolver/btTypedConstraint.h
	ConstraintSolver/btUniversalConstraint.h
)
SET(Dynamics_HDRS
	Dynamics/btActionInterface.h
	Dynamics/btDiscreteDynamicsWorld.h
	Dynamics/btDiscreteDynamicsWorldMt.h
	Dynamics/btSimulationIslandManagerMt.h
	Dynamics/btDynamicsWorld.h
	Dynamics/btSimpleDynamicsWorld.h
	Dynamics/btRigidBody.h
)
SET(Vehicle_HDRS
	Vehicle/btRaycastVehicle.h
	Vehicle/btVehicleRaycaster.h
	Vehicle/btWheelInfo.h
)

SET(Featherstone_HDRS
	Featherstone/btMultiBody.h
	Featherstone/btMultiBodyConstraintSolver.h
	Featherstone/btMultiBodyDynamicsWorld.h
	Featherstone/btMultiBodyLink.h
	Featherstone/btMultiBodyLinkCollider.h
	Featherstone/btMultiBodySolverConstraint.h
	Featherstone/btMultiBodyConstraint.h
	Featherstone/btMultiBodyJointLimitConstraint.h
	Featherstone/btMultiBodyConstraint.h
	Featherstone/btMultiBodyPoint2Point.h
	Featherstone/btMultiBodyFixedConstraint.h
	Featherstone/btMultiBodySliderConstraint.h
	Featherstone/btMultiBodyJointMotor.h
	Featherstone/btMultiBodyGearConstraint.h	
)

SET(MLCPSolvers_HDRS
	MLCPSolvers/btDantzigLCP.h
	MLCPSolvers/btDantzigSolver.h
	MLCPSolvers/btMLCPSolver.h
	MLCPSolvers/btMLCPSolverInterface.h
	MLCPSolvers/btPATHSolver.h
	MLCPSolvers/btSolveProjectedGaussSeidel.h
	MLCPSolvers/btLemkeSolver.h
	MLCPSolvers/btLemkeAlgorithm.h
)

SET(Character_HDRS
	Character/btCharacterControllerInterface.h
	Character/btKinematicCharacterController.h
)



SET(BulletDynamics_HDRS
	${Root_HDRS}
	${ConstraintSolver_HDRS}
	${Dynamics_HDRS}
	${Vehicle_HDRS}
	${Character_HDRS}
	${Featherstone_HDRS}
	${MLCPSolvers_HDRS}
)


ADD_LIBRARY(BulletDynamics ${BulletDynamics_SRCS} ${BulletDynamics_HDRS})
SET_TARGET_PROPERTIES(BulletDynamics PROPERTIES VERSION ${BULLET_VERSION})
SET_TARGET_PROPERTIES(BulletDynamics PROPERTIES SOVERSION ${BULLET_VERSION})
IF (BUILD_SHARED_LIBS)
	TARGET_LINK_LIBRARIES(BulletDynamics BulletCollision LinearMath)
ENDIF (BUILD_SHARED_LIBS)

IF (INSTALL_LIBS)
	IF (NOT INTERNAL_CREATE_DISTRIBUTABLE_MSVC_PROJECTFILES)
		IF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 2.5)
			IF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
				INSTALL(TARGETS BulletDynamics DESTINATION .)
			ELSE (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
				INSTALL(TARGETS BulletDynamics RUNTIME DESTINATION bin
								LIBRARY DESTINATION lib${LIB_SUFFIX}
								ARCHIVE DESTINATION lib${LIB_SUFFIX})
				INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
DESTINATION ${INCLUDE_INSTALL_DIR} FILES_MATCHING PATTERN "*.h"  PATTERN
".svn" EXCLUDE PATTERN "CMakeFiles" EXCLUDE)
				INSTALL(FILES ../btBulletDynamicsCommon.h
DESTINATION ${INCLUDE_INSTALL_DIR}/BulletDynamics)
			ENDIF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
		ENDIF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 2.5)

		IF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
			SET_TARGET_PROPERTIES(BulletDynamics PROPERTIES FRAMEWORK true)
			SET_TARGET_PROPERTIES(BulletDynamics PROPERTIES PUBLIC_HEADER "${Root_HDRS}")
			# Have to list out sub-directories manually:
			SET_PROPERTY(SOURCE ${ConstraintSolver_HDRS} PROPERTY MACOSX_PACKAGE_LOCATION Headers/ConstraintSolver)
			SET_PROPERTY(SOURCE ${Dynamics_HDRS} PROPERTY MACOSX_PACKAGE_LOCATION Headers/Dynamics)
			SET_PROPERTY(SOURCE ${Vehicle_HDRS} PROPERTY MACOSX_PACKAGE_LOCATION Headers/Vehicle)
			SET_PROPERTY(SOURCE ${Character_HDRS} PROPERTY MACOSX_PACKAGE_LOCATION Headers/Character)
			SET_PROPERTY(SOURCE ${Featherstone_HDRS} PROPERTY MACOSX_PACKAGE_LOCATION Headers/Featherstone)
			SET_PROPERTY(SOURCE ${MLCPSolvers_HDRS} PROPERTY MACOSX_PACKAGE_LOCATION Headers/MLCPSolvers)
		ENDIF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
	ENDIF (NOT INTERNAL_CREATE_DISTRIBUTABLE_MSVC_PROJECTFILES)
ENDIF (INSTALL_LIBS)
