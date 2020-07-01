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

SET(CMAKE_THREAD_PREFER_PTHREAD TRUE)
FIND_PACKAGE(Threads REQUIRED)

ADD_LIBRARY(sys STATIC
  sysinfo.cpp
  alloc.cpp
  filename.cpp
  library.cpp
  thread.cpp
  string.cpp
  regression.cpp
  mutex.cpp
  condition.cpp
  barrier.cpp
)

TARGET_LINK_LIBRARIES(sys ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS})
SET_PROPERTY(TARGET sys PROPERTY FOLDER common)
SET_PROPERTY(TARGET sys APPEND PROPERTY COMPILE_FLAGS " ${FLAGS_LOWEST}")

IF (EMBREE_STATIC_LIB)
  INSTALL(TARGETS sys EXPORT sys-targets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel)
  INSTALL(EXPORT sys-targets DESTINATION ${EMBREE_CMAKECONFIG_DIR} COMPONENT devel)
  SIGN_TARGET(sys)
ENDIF()
