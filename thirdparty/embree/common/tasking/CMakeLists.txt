## ======================================================================== ##
## Copyright 2009-2018 Intel Corporation                                    ##
##                                                                          ##
## Licensed under the Apache License, Version 2.0 (the "License");          ##
## you may not use this file except in compliance with the License.         ##
## You may obtain a copy of the License at                                  ##
##                                                                          ##
##     http://www.apache.org/licenses/LICENSE-2.0                           ##
##                                                                          ##
## Unless required by applicable law or agreed to in writing, software      ##
## distributed under the License is distributed on an "AS IS" BASIS,        ##
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. ##
## See the License for the specific language governing permissions and      ##
## limitations under the License.                                           ##
## ======================================================================== ##

IF (TASKING_INTERNAL)
  ADD_LIBRARY(tasking STATIC taskschedulerinternal.cpp)
ELSEIF (TASKING_TBB)
  ADD_LIBRARY(tasking STATIC taskschedulertbb.cpp)
  TARGET_LINK_LIBRARIES(tasking sys math ${TBB_LIBRARIES})
ELSEIF (TASKING_PPL)
  ADD_LIBRARY(tasking STATIC taskschedulerppl.cpp)
  TARGET_LINK_LIBRARIES(tasking sys math ${PPL_LIBRARIES})
ENDIF()

SET_PROPERTY(TARGET tasking PROPERTY FOLDER common)
SET_PROPERTY(TARGET tasking APPEND PROPERTY COMPILE_FLAGS " ${FLAGS_LOWEST}")

IF (EMBREE_STATIC_LIB)
  INSTALL(TARGETS tasking EXPORT tasking-targets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel)
  INSTALL(EXPORT tasking-targets DESTINATION ${EMBREE_CMAKECONFIG_DIR} COMPONENT devel)
  SIGN_TARGET(tasking)
ENDIF()
