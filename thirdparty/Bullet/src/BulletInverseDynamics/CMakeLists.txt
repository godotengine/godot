INCLUDE_DIRECTORIES( ${BULLET_PHYSICS_SOURCE_DIR}/src  )

SET(BulletInverseDynamics_SRCS
	IDMath.cpp
	MultiBodyTree.cpp
	details/MultiBodyTreeInitCache.cpp
	details/MultiBodyTreeImpl.cpp
)

SET(BulletInverseDynamicsRoot_HDRS
	IDConfig.hpp
	IDConfigEigen.hpp
	IDMath.hpp
	IDConfigBuiltin.hpp
	IDErrorMessages.hpp
	MultiBodyTree.hpp
)
SET(BulletInverseDynamicsDetails_HDRS
	details/IDEigenInterface.hpp
	details/IDMatVec.hpp
	details/IDLinearMathInterface.hpp
	details/MultiBodyTreeImpl.hpp
	details/MultiBodyTreeInitCache.hpp
)

SET(BulletInverseDynamics_HDRS
	${BulletInverseDynamicsRoot_HDRS}
	${BulletInverseDynamicsDetails_HDRS}
)


ADD_LIBRARY(BulletInverseDynamics ${BulletInverseDynamics_SRCS} ${BulletInverseDynamics_HDRS})
SET_TARGET_PROPERTIES(BulletInverseDynamics PROPERTIES VERSION ${BULLET_VERSION})
SET_TARGET_PROPERTIES(BulletInverseDynamics PROPERTIES SOVERSION ${BULLET_VERSION})
IF (BUILD_SHARED_LIBS)
  TARGET_LINK_LIBRARIES(BulletInverseDynamics Bullet3Common LinearMath)
ENDIF (BUILD_SHARED_LIBS)


IF (INSTALL_LIBS)
	IF (NOT INTERNAL_CREATE_DISTRIBUTABLE_MSVC_PROJECTFILES)
		#INSTALL of other files requires CMake 2.6
		IF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 2.5)
			IF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
				INSTALL(TARGETS BulletInverseDynamics DESTINATION .)
			ELSE (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
				INSTALL(TARGETS BulletInverseDynamics RUNTIME DESTINATION bin
								LIBRARY DESTINATION lib${LIB_SUFFIX}
								ARCHIVE DESTINATION lib${LIB_SUFFIX})
				INSTALL(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
DESTINATION ${INCLUDE_INSTALL_DIR} FILES_MATCHING PATTERN "*.h" PATTERN ".svn" EXCLUDE PATTERN "CMakeFiles" EXCLUDE)
				INSTALL(FILES ../btBulletCollisionCommon.h
DESTINATION ${INCLUDE_INSTALL_DIR}/BulletInverseDynamics)
			ENDIF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
		ENDIF (${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION} GREATER 2.5)

		IF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
			SET_TARGET_PROPERTIES(BulletInverseDynamics PROPERTIES FRAMEWORK true)

			SET_TARGET_PROPERTIES(BulletInverseDynamics PROPERTIES PUBLIC_HEADER "${BulletInverseDynamicsRoot_HDRS}")
			# Have to list out sub-directories manually:
			SET_PROPERTY(SOURCE ${BulletInverseDynamicsDetails_HDRS} PROPERTY MACOSX_PACKAGE_LOCATION Headers/details)

		ENDIF (APPLE AND BUILD_SHARED_LIBS AND FRAMEWORK)
	ENDIF (NOT INTERNAL_CREATE_DISTRIBUTABLE_MSVC_PROJECTFILES)
ENDIF (INSTALL_LIBS)
