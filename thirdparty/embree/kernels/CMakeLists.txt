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

MACRO(DISABLE_STACK_PROTECTOR_FOR_INTERSECTORS)
  FOREACH(src ${ARGN})
    IF (src MATCHES ".*intersector.*")
      DISABLE_STACK_PROTECTOR_FOR_FILE(${src})
    ENDIF()
  ENDFOREACH()
ENDMACRO()

SOURCE_GROUP("Source Files\\bvh4" "/bvh4/" )
SOURCE_GROUP("Source Files\\bvh8" "/bvh8/" )
SOURCE_GROUP("Source Files\\geometry" "/geometry/" )
SOURCE_GROUP("Source Files\\builders" "/builders.*/" )
SOURCE_GROUP("Source Files\\common" "/kernels/common/" )
SOURCE_GROUP("Source Files\\tasking" "/tasking/" )
SOURCE_GROUP("Source Files\\algorithms" "/algorithms/")
SOURCE_GROUP("" FILES embree.rc)

SET(EMBREE_LIBRARY_FILES
 
  embree.rc

  common/device.cpp
  common/stat.cpp
  common/acceln.cpp
  common/accelset.cpp
  common/state.cpp
  common/rtcore.cpp
  common/rtcore_builder.cpp
  common/scene.cpp
  common/alloc.cpp
  common/geometry.cpp
  common/scene_user_geometry.cpp
  common/scene_instance.cpp
  common/scene_triangle_mesh.cpp
  common/scene_quad_mesh.cpp
  common/scene_curves.cpp
  common/scene_line_segments.cpp
  common/scene_grid_mesh.cpp
  common/scene_points.cpp

  subdiv/bezier_curve.cpp
  subdiv/bspline_curve.cpp

  geometry/primitive4.cpp
  geometry/instance_intersector.cpp
  geometry/curve_intersector_virtual.cpp
  builders/primrefgen.cpp

  bvh/bvh.cpp
  bvh/bvh_statistics.cpp
  bvh/bvh4_factory.cpp
  bvh/bvh8_factory.cpp

  bvh/bvh_rotate.cpp
  bvh/bvh_refit.cpp
  bvh/bvh_builder.cpp
  bvh/bvh_builder_hair.cpp
  bvh/bvh_builder_hair_mb.cpp
  bvh/bvh_builder_morton.cpp
  bvh/bvh_builder_sah.cpp
  bvh/bvh_builder_sah_spatial.cpp
  bvh/bvh_builder_sah_mb.cpp
  bvh/bvh_builder_twolevel.cpp

  bvh/bvh_intersector1_bvh4.cpp
  )

IF (EMBREE_GEOMETRY_SUBDIVISION)
  SET(EMBREE_LIBRARY_FILES ${EMBREE_LIBRARY_FILES}
  common/scene_subdiv_mesh.cpp
  subdiv/tessellation_cache.cpp
  subdiv/subdivpatch1base.cpp
  subdiv/catmullclark_coefficients.cpp
  geometry/grid_soa.cpp	
  subdiv/subdivpatch1base_eval.cpp
  bvh/bvh_builder_subdiv.cpp)
ENDIF()

IF (EMBREE_RAY_PACKETS)
  SET(EMBREE_LIBRARY_FILES ${EMBREE_LIBRARY_FILES}

  bvh/bvh_intersector_hybrid4_bvh4.cpp
  bvh/bvh_intersector_stream_bvh4.cpp
  bvh/bvh_intersector_stream_filters.cpp)
ENDIF()

MACRO(embree_files TARGET ISA)
      
  SET(${TARGET}
    geometry/instance_intersector.cpp
    geometry/curve_intersector_virtual.cpp
    bvh/bvh_intersector1_bvh4.cpp)

  IF (${ISA} EQUAL ${ISA_LOWEST_AVX})
    LIST(APPEND ${TARGET} geometry/primitive8.cpp)
  ENDIF() 

  IF (${ISA} EQUAL ${SSE2} OR ${ISA} EQUAL ${AVX} OR ${ISA} EQUAL ${AVX2} OR ${ISA} EQUAL ${AVX512KNL} OR ${ISA} EQUAL ${AVX512SKX} OR ${ISA_LOWEST} EQUAL ${ISA})
    
    LIST(APPEND ${TARGET}
      common/scene_user_geometry.cpp
      common/scene_instance.cpp
      common/scene_triangle_mesh.cpp
      common/scene_quad_mesh.cpp 
      common/scene_curves.cpp
      common/scene_line_segments.cpp
      common/scene_grid_mesh.cpp
      common/scene_points.cpp
      
      bvh/bvh_refit.cpp
      bvh/bvh_builder.cpp
      bvh/bvh_builder_hair.cpp
      bvh/bvh_builder_hair_mb.cpp
      bvh/bvh_builder_sah.cpp
      bvh/bvh_builder_sah_spatial.cpp
      bvh/bvh_builder_sah_mb.cpp
      bvh/bvh_builder_twolevel.cpp)

    IF (EMBREE_GEOMETRY_SUBDIVISION)
      LIST(APPEND ${TARGET} bvh/bvh_builder_subdiv.cpp)
    ENDIF()
  ENDIF()

  IF (${ISA} EQUAL ${SSE2} OR ${ISA} EQUAL ${AVX} OR ${ISA} EQUAL ${AVX2} OR ${ISA} EQUAL ${AVX512KNL} OR ${ISA_LOWEST} EQUAL ${ISA})
    LIST(APPEND ${TARGET}
      bvh/bvh_builder_morton.cpp
      bvh/bvh_rotate.cpp
      builders/primrefgen.cpp)
  ENDIF()
    
  IF (${ISA} GREATER ${SSE42})
    LIST(APPEND ${TARGET} bvh/bvh_intersector1_bvh8.cpp)
  ENDIF()

  IF (${ISA} EQUAL ${AVX})
    LIST(APPEND ${TARGET}
      bvh/bvh.cpp
      bvh/bvh_statistics.cpp)
  ENDIF()

  IF (EMBREE_GEOMETRY_SUBDIVISION)
    LIST(APPEND ${TARGET}
        common/scene_subdiv_mesh.cpp
        geometry/grid_soa.cpp
        subdiv/subdivpatch1base_eval.cpp)
  ENDIF()
    
  IF (EMBREE_RAY_PACKETS)
    LIST(APPEND ${TARGET}
      bvh/bvh_intersector_hybrid4_bvh4.cpp
      bvh/bvh_intersector_stream_bvh4.cpp
      bvh/bvh_intersector_stream_filters.cpp)

    IF (${ISA} GREATER ${SSE42})
      LIST(APPEND ${TARGET}
        bvh/bvh_intersector_hybrid8_bvh4.cpp
        bvh/bvh_intersector_hybrid4_bvh8.cpp
        bvh/bvh_intersector_hybrid8_bvh8.cpp
        bvh/bvh_intersector_stream_bvh8.cpp)
    ENDIF()

    IF (${ISA} GREATER ${AVX2})
      LIST(APPEND ${TARGET}
        bvh/bvh_intersector_hybrid16_bvh8.cpp
        bvh/bvh_intersector_hybrid16_bvh4.cpp)
    ENDIF()
  ENDIF()
  
ENDMACRO()

#embree_files(EMBREE_LIBRARY_FILES_SSE2 ${SSE2})
embree_files(EMBREE_LIBRARY_FILES_SSE42 ${SSE42})
embree_files(EMBREE_LIBRARY_FILES_AVX ${AVX})
embree_files(EMBREE_LIBRARY_FILES_AVX2 ${AVX2})
embree_files(EMBREE_LIBRARY_FILES_AVX512KNL ${AVX512KNL})
embree_files(EMBREE_LIBRARY_FILES_AVX512SKX ${AVX512SKX})

#message("SSE2: ${EMBREE_LIBRARY_FILES_SSE2}")
#message("SSE42: ${EMBREE_LIBRARY_FILES_SSE42}")
#message("AVX: ${EMBREE_LIBRARY_FILES_AVX}")
#message("AVX2: ${EMBREE_LIBRARY_FILES_AVX2}")
#message("AVX512KNL: ${EMBREE_LIBRARY_FILES_AVX512KNL}")
#message("AVX512SKX: ${EMBREE_LIBRARY_FILES_AVX512SKX}")

# replaces all .cpp files with a dummy file that includes that .cpp file
# this is to work around an ICC name mangling issue related to lambda functions under windows
MACRO (CreateISADummyFiles list isa)
  SET(${list})
  FOREACH(src ${ARGN})
    SET(src_file ${CMAKE_CURRENT_SOURCE_DIR}/${src})
    SET(dst_file ${CMAKE_BINARY_DIR}/${src}.${isa}.cpp)
    SET(${list} ${${list}} ${dst_file})
    ADD_CUSTOM_COMMAND(OUTPUT ${dst_file}
      COMMAND ${CMAKE_COMMAND} 
       -D src=${src_file}
       -D dst=${dst_file}
       -P ${PROJECT_SOURCE_DIR}/common/cmake/create_isa_dummy_file.cmake
      DEPENDS ${src_file})
  ENDFOREACH()
ENDMACRO()
#CreateISADummyFiles(EMBREE_LIBRARY_FILES_SSE2      sse2      ${EMBREE_LIBRARY_FILES_SSE2})
CreateISADummyFiles(EMBREE_LIBRARY_FILES_SSE42     sse42     ${EMBREE_LIBRARY_FILES_SSE42})
CreateISADummyFiles(EMBREE_LIBRARY_FILES_AVX       avx       ${EMBREE_LIBRARY_FILES_AVX})
CreateISADummyFiles(EMBREE_LIBRARY_FILES_AVX2      avx2      ${EMBREE_LIBRARY_FILES_AVX2})
CreateISADummyFiles(EMBREE_LIBRARY_FILES_AVX512KNL avx512knl ${EMBREE_LIBRARY_FILES_AVX512KNL})
CreateISADummyFiles(EMBREE_LIBRARY_FILES_AVX512SKX avx512skx ${EMBREE_LIBRARY_FILES_AVX512SKX})

MACRO (CheckGlobals library)
  IF (NOT WIN32 AND NOT APPLE)
    ADD_CUSTOM_TARGET(${library}_check_globals ALL COMMAND ${CMAKE_COMMAND} -D file=$<TARGET_FILE:${library}> -P ${PROJECT_SOURCE_DIR}/common/cmake/check_globals.cmake DEPENDS ${library})
  ENDIF()
ENDMACRO()

DISABLE_STACK_PROTECTOR_FOR_INTERSECTORS(${EMBREE_LIBRARY_FILES})
ADD_LIBRARY(embree ${EMBREE_LIB_TYPE} ${EMBREE_LIBRARY_FILES} $<TARGET_OBJECTS:algorithms>)
SET_TARGET_PROPERTIES(embree PROPERTIES COMPILE_FLAGS "${FLAGS_LOWEST}")
SET_TARGET_PROPERTIES(embree PROPERTIES COMPILE_DEFINITIONS "EMBREE_LOWEST_ISA")
SET_PROPERTY(TARGET embree PROPERTY FOLDER kernels)
IF (NOT WIN32 AND NOT  APPLE)
  ADD_CUSTOM_TARGET(embree_check_stack_frame_size COMMAND ${CMAKE_COMMAND} -D file=$<TARGET_FILE:embree> -P ${PROJECT_SOURCE_DIR}/common/cmake/check_stack_frame_size.cmake DEPENDS embree)
ENDIF()

#IF (EMBREE_ISA_SSE2 AND EMBREE_LIBRARY_FILES_SSE2)
#  ADD_LIBRARY(embree_sse2 STATIC ${EMBREE_LIBRARY_FILES_SSE2})
#  SET_TARGET_PROPERTIES(embree_sse2 PROPERTIES COMPILE_FLAGS "${FLAGS_SSE2}")
#  SET_PROPERTY(TARGET embree_sse2 PROPERTY FOLDER kernels)
#  SET(EMBREE_LIBRARIES ${EMBREE_LIBRARIES} embree_sse2)
#  CheckGlobals(embree_sse2)
#IF (EMBREE_STATIC_LIB)
#    INSTALL(TARGETS embree_sse2 ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel)
#  ENDIF()
#ENDIF ()

IF (EMBREE_ISA_SSE42 AND EMBREE_LIBRARY_FILES_SSE42)
  DISABLE_STACK_PROTECTOR_FOR_INTERSECTORS(${EMBREE_LIBRARY_FILES_SSE42})
  ADD_LIBRARY(embree_sse42 STATIC ${EMBREE_LIBRARY_FILES_SSE42})
  SET_TARGET_PROPERTIES(embree_sse42 PROPERTIES COMPILE_FLAGS "${FLAGS_SSE42}")
  SET_PROPERTY(TARGET embree_sse42 PROPERTY FOLDER kernels)
  SET(EMBREE_LIBRARIES ${EMBREE_LIBRARIES} embree_sse42)
  CheckGlobals(embree_sse42)
  IF (EMBREE_STATIC_LIB)
    INSTALL(TARGETS embree_sse42 EXPORT embree_sse42-targets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel)
    INSTALL(EXPORT embree_sse42-targets DESTINATION ${EMBREE_CMAKECONFIG_DIR} COMPONENT devel)
  ENDIF()
ENDIF ()

IF (EMBREE_ISA_AVX  AND EMBREE_LIBRARY_FILES_AVX)
  DISABLE_STACK_PROTECTOR_FOR_INTERSECTORS(${EMBREE_LIBRARY_FILES_AVX})
  ADD_LIBRARY(embree_avx STATIC ${EMBREE_LIBRARY_FILES_AVX})
  SET_TARGET_PROPERTIES(embree_avx PROPERTIES COMPILE_FLAGS "${FLAGS_AVX}")
  SET_PROPERTY(TARGET embree_avx PROPERTY FOLDER kernels)
  SET(EMBREE_LIBRARIES ${EMBREE_LIBRARIES} embree_avx)
  CheckGlobals(embree_avx)
  IF (EMBREE_STATIC_LIB)
    INSTALL(TARGETS embree_avx EXPORT embree_avx-targets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel)
    INSTALL(EXPORT embree_avx-targets DESTINATION ${EMBREE_CMAKECONFIG_DIR} COMPONENT devel)
  ENDIF()
ENDIF()

IF (EMBREE_ISA_AVX2 AND EMBREE_LIBRARY_FILES_AVX2)
  DISABLE_STACK_PROTECTOR_FOR_INTERSECTORS(${EMBREE_LIBRARY_FILES_AVX2})
  ADD_LIBRARY(embree_avx2 STATIC ${EMBREE_LIBRARY_FILES_AVX2})
  SET_TARGET_PROPERTIES(embree_avx2 PROPERTIES COMPILE_FLAGS "${FLAGS_AVX2}")
  SET_PROPERTY(TARGET embree_avx2 PROPERTY FOLDER kernels)
  SET(EMBREE_LIBRARIES ${EMBREE_LIBRARIES} embree_avx2)
  CheckGlobals(embree_avx2)
  IF (EMBREE_STATIC_LIB)
    INSTALL(TARGETS embree_avx2 EXPORT embree_avx2-targets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel)
    INSTALL(EXPORT embree_avx2-targets DESTINATION ${EMBREE_CMAKECONFIG_DIR} COMPONENT devel)
  ENDIF()
ENDIF()

IF (EMBREE_ISA_AVX512KNL AND EMBREE_LIBRARY_FILES_AVX512KNL)
  DISABLE_STACK_PROTECTOR_FOR_INTERSECTORS(${EMBREE_LIBRARY_FILES_AVX512KNL})
  ADD_LIBRARY(embree_avx512knl STATIC ${EMBREE_LIBRARY_FILES_AVX512KNL})
  SET_TARGET_PROPERTIES(embree_avx512knl PROPERTIES COMPILE_FLAGS "${FLAGS_AVX512KNL}")
  SET_PROPERTY(TARGET embree_avx512knl PROPERTY FOLDER kernels)
  SET(EMBREE_LIBRARIES ${EMBREE_LIBRARIES} embree_avx512knl)
  CheckGlobals(embree_avx512knl)
  IF (EMBREE_STATIC_LIB)
    INSTALL(TARGETS embree_avx512knl EXPORT embree_avx512knl-targets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel)
    INSTALL(EXPORT embree_avx512knl-targets DESTINATION ${EMBREE_CMAKECONFIG_DIR} COMPONENT devel)
  ENDIF()
ENDIF()

IF (EMBREE_ISA_AVX512SKX AND EMBREE_LIBRARY_FILES_AVX512SKX)
  DISABLE_STACK_PROTECTOR_FOR_INTERSECTORS(${EMBREE_LIBRARY_FILES_AVX512SKX})
  ADD_LIBRARY(embree_avx512skx STATIC ${EMBREE_LIBRARY_FILES_AVX512SKX})
  SET_TARGET_PROPERTIES(embree_avx512skx PROPERTIES COMPILE_FLAGS "${FLAGS_AVX512SKX}")
  SET_PROPERTY(TARGET embree_avx512skx PROPERTY FOLDER kernels)
  SET(EMBREE_LIBRARIES ${EMBREE_LIBRARIES} embree_avx512skx)
  CheckGlobals(embree_avx512skx)
  IF (EMBREE_STATIC_LIB)
    INSTALL(TARGETS embree_avx512skx EXPORT embree_avx512skx-targets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel)
    INSTALL(EXPORT embree_avx512skx-targets DESTINATION ${EMBREE_CMAKECONFIG_DIR} COMPONENT devel)
  ENDIF()
ENDIF()

TARGET_LINK_LIBRARIES(embree ${EMBREE_LIBRARIES} sys math simd lexers tasking)
SET_TARGET_PROPERTIES(embree PROPERTIES INSTALL_RPATH "") # no RPATH for installed libembree
SET_TARGET_PROPERTIES(embree PROPERTIES OUTPUT_NAME ${EMBREE_LIBRARY_NAME})

IF (WIN32)
ELSEIF (APPLE)
  SET_TARGET_PROPERTIES(embree PROPERTIES LINK_FLAGS -Wl,-exported_symbols_list,${PROJECT_SOURCE_DIR}/kernels/export.macosx.map)
  SET_SOURCE_FILES_PROPERTIES(common/rtcore.cpp PROPERTIES OBJECT_DEPENDS ${PROJECT_SOURCE_DIR}/kernels/export.macosx.map) 
ELSE()
  SET_TARGET_PROPERTIES(embree PROPERTIES LINK_FLAGS -Wl,--version-script=${PROJECT_SOURCE_DIR}/kernels/export.linux.map)
  SET_SOURCE_FILES_PROPERTIES(common/rtcore.cpp PROPERTIES OBJECT_DEPENDS ${PROJECT_SOURCE_DIR}/kernels/export.linux.map)
ENDIF()

IF (EMBREE_ZIP_MODE)
  SET_TARGET_PROPERTIES(embree PROPERTIES VERSION ${EMBREE_VERSION_MAJOR} SOVERSION ${EMBREE_VERSION_MAJOR})
ELSE()
  SET_TARGET_PROPERTIES(embree PROPERTIES VERSION ${EMBREE_VERSION} SOVERSION ${EMBREE_VERSION_MAJOR})
ENDIF()

INSTALL(TARGETS embree EXPORT embree-targets
  LIBRARY NAMELINK_SKIP DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT lib
  # on Windows put the dlls into bin
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT examples
  # ... and the import lib into the devel package
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel
)

IF (EMBREE_STATIC_LIB)
  INSTALL(EXPORT embree-targets DESTINATION ${EMBREE_CMAKECONFIG_DIR} COMPONENT devel)
ENDIF()

IF (NOT EMBREE_STATIC_LIB)
  INSTALL(TARGETS embree
    LIBRARY NAMELINK_ONLY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT devel
  )
ENDIF()

SIGN_TARGET(embree)
