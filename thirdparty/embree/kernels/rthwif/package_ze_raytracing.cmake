## Copyright 2009-2021 Intel Corporation
## SPDX-License-Identifier: Apache-2.0

INCLUDE(GNUInstallDirs)

##############################################################
# Install Documentation
##############################################################

INSTALL(FILES "${PROJECT_SOURCE_DIR}/../../LICENSE.txt" DESTINATION doc COMPONENT lib)
INSTALL(FILES "${PROJECT_SOURCE_DIR}/../../CHANGELOG.md" DESTINATION doc COMPONENT lib)
INSTALL(FILES "${PROJECT_SOURCE_DIR}/../../third-party-programs.txt" DESTINATION doc COMPONENT lib)
INSTALL(FILES "${PROJECT_SOURCE_DIR}/../../third-party-programs-TBB.txt" DESTINATION doc COMPONENT lib)
INSTALL(FILES "${PROJECT_SOURCE_DIR}/../../third-party-programs-OIDN.txt" DESTINATION doc COMPONENT lib)
INSTALL(FILES "${PROJECT_SOURCE_DIR}/../../third-party-programs-DPCPP.txt" DESTINATION doc COMPONENT lib)
INSTALL(FILES "${PROJECT_SOURCE_DIR}/../../third-party-programs-oneAPI-DPCPP.txt" DESTINATION doc COMPONENT lib)

##############################################################
# CPack specific stuff
##############################################################

SET(CPACK_PACKAGE_NAME "L0 Ray Tracing Build API")
SET(CPACK_PACKAGE_FILE_NAME "ze_raytracing-${RTHWIF_VERSION}")
SET(CPACK_STRIP_FILES TRUE)

SET(CPACK_PACKAGE_VERSION_MAJOR ${EMBREE_VERSION_MAJOR})
SET(CPACK_PACKAGE_VERSION_MINOR ${EMBREE_VERSION_MINOR})
SET(CPACK_PACKAGE_VERSION_PATCH ${EMBREE_VERSION_PATCH})
SET(CPACK_PACKAGE_VERSION ${EMBREE_VERSION})
SET(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Implements acceleration structure build for L0 ray tracing extension.")
SET(CPACK_PACKAGE_VENDOR "Intel Corporation")
SET(CPACK_PACKAGE_CONTACT embree_support@intel.com)
SET(CPACK_MONOLITHIC_INSTALL 1)

SET(CPACK_COMPONENT_LIB_DISPLAY_NAME "Library")
SET(CPACK_COMPONENT_LIB_DESCRIPTION "Library")

SET(CPACK_COMPONENT_DEVEL_DISPLAY_NAME "Development")
SET(CPACK_COMPONENT_DEVEL_DESCRIPTION "Development")

SET(CPACK_COMPONENT_EXAMPLES_DISPLAY_NAME "Examples")
SET(CPACK_COMPONENT_EXAMPLES_DESCRIPTION "Examples")

# Windows specific settings
IF(WIN32)
  SET(CPACK_GENERATOR ZIP)
  SET(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}.x64.windows")

# MacOSX specific settings
ELSEIF(APPLE)
  SET(CPACK_GENERATOR ZIP)
  SET(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}.x86_64.macosx")

# Linux specific settings
ELSE()

  SET(CPACK_GENERATOR TGZ)
  SET(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}.x86_64.linux")
 
ENDIF()
