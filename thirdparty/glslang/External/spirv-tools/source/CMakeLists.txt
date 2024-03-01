# Copyright (c) 2015-2016 The Khronos Group Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set(GRAMMAR_PROCESSING_SCRIPT "${spirv-tools_SOURCE_DIR}/utils/generate_grammar_tables.py")
set(VIMSYNTAX_PROCESSING_SCRIPT "${spirv-tools_SOURCE_DIR}/utils/generate_vim_syntax.py")
set(XML_REGISTRY_PROCESSING_SCRIPT "${spirv-tools_SOURCE_DIR}/utils/generate_registry_tables.py")
set(LANG_HEADER_PROCESSING_SCRIPT "${spirv-tools_SOURCE_DIR}/utils/generate_language_headers.py")

# Pull in grammar files that have migrated to SPIRV-Headers
set(DEBUGINFO_GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/unified1/extinst.debuginfo.grammar.json")
set(CLDEBUGINFO100_GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/unified1/extinst.opencl.debuginfo.100.grammar.json")
set(VKDEBUGINFO100_GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/unified1/extinst.nonsemantic.shader.debuginfo.100.grammar.json")

# macro() definitions are used in the following because we need to append .inc
# file paths into some global lists (*_CPP_DEPENDS). And those global lists are
# later used by set_source_files_properties() calls.
# function() definitions are not suitable because they create new scopes.
macro(spvtools_core_tables CONFIG_VERSION)
  set(GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/${CONFIG_VERSION}/spirv.core.grammar.json")
  set(GRAMMAR_INSTS_INC_FILE "${spirv-tools_BINARY_DIR}/core.insts-${CONFIG_VERSION}.inc")
  set(GRAMMAR_KINDS_INC_FILE "${spirv-tools_BINARY_DIR}/operand.kinds-${CONFIG_VERSION}.inc")
  add_custom_command(OUTPUT ${GRAMMAR_INSTS_INC_FILE} ${GRAMMAR_KINDS_INC_FILE}
    COMMAND Python3::Interpreter ${GRAMMAR_PROCESSING_SCRIPT}
      --spirv-core-grammar=${GRAMMAR_JSON_FILE}
      --extinst-debuginfo-grammar=${DEBUGINFO_GRAMMAR_JSON_FILE}
      --extinst-cldebuginfo100-grammar=${CLDEBUGINFO100_GRAMMAR_JSON_FILE}
      --core-insts-output=${GRAMMAR_INSTS_INC_FILE}
      --operand-kinds-output=${GRAMMAR_KINDS_INC_FILE}
      --output-language=c++
    DEPENDS ${GRAMMAR_PROCESSING_SCRIPT}
            ${GRAMMAR_JSON_FILE}
            ${DEBUGINFO_GRAMMAR_JSON_FILE}
            ${CLDEBUGINFO100_GRAMMAR_JSON_FILE}
    COMMENT "Generate info tables for SPIR-V v${CONFIG_VERSION} core instructions and operands.")
  list(APPEND OPCODE_CPP_DEPENDS ${GRAMMAR_INSTS_INC_FILE})
  list(APPEND OPERAND_CPP_DEPENDS ${GRAMMAR_KINDS_INC_FILE})
endmacro(spvtools_core_tables)

macro(spvtools_enum_string_mapping CONFIG_VERSION)
  set(GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/${CONFIG_VERSION}/spirv.core.grammar.json")
  set(GRAMMAR_EXTENSION_ENUM_INC_FILE "${spirv-tools_BINARY_DIR}/extension_enum.inc")
  set(GRAMMAR_ENUM_STRING_MAPPING_INC_FILE "${spirv-tools_BINARY_DIR}/enum_string_mapping.inc")
  add_custom_command(OUTPUT ${GRAMMAR_EXTENSION_ENUM_INC_FILE}
     ${GRAMMAR_ENUM_STRING_MAPPING_INC_FILE}
    COMMAND Python3::Interpreter ${GRAMMAR_PROCESSING_SCRIPT}
      --spirv-core-grammar=${GRAMMAR_JSON_FILE}
      --extinst-debuginfo-grammar=${DEBUGINFO_GRAMMAR_JSON_FILE}
      --extinst-cldebuginfo100-grammar=${CLDEBUGINFO100_GRAMMAR_JSON_FILE}
      --extension-enum-output=${GRAMMAR_EXTENSION_ENUM_INC_FILE}
      --enum-string-mapping-output=${GRAMMAR_ENUM_STRING_MAPPING_INC_FILE}
      --output-language=c++
    DEPENDS ${GRAMMAR_PROCESSING_SCRIPT}
            ${GRAMMAR_JSON_FILE}
            ${DEBUGINFO_GRAMMAR_JSON_FILE}
            ${CLDEBUGINFO100_GRAMMAR_JSON_FILE}
    COMMENT "Generate enum-string mapping for SPIR-V v${CONFIG_VERSION}.")
  list(APPEND EXTENSION_H_DEPENDS ${GRAMMAR_EXTENSION_ENUM_INC_FILE})
  list(APPEND ENUM_STRING_MAPPING_CPP_DEPENDS ${GRAMMAR_ENUM_STRING_MAPPING_INC_FILE})
endmacro(spvtools_enum_string_mapping)

macro(spvtools_vimsyntax CONFIG_VERSION CLVERSION)
  set(GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/${CONFIG_VERSION}/spirv.core.grammar.json")
  set(GLSL_GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/${CONFIG_VERSION}/extinst.glsl.std.450.grammar.json")
  set(OPENCL_GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/${CONFIG_VERSION}/extinst.opencl.std.100.grammar.json")
  set(VIMSYNTAX_FILE "${spirv-tools_BINARY_DIR}/spvasm.vim")
  add_custom_command(OUTPUT ${VIMSYNTAX_FILE}
      COMMAND Python3::Interpreter ${VIMSYNTAX_PROCESSING_SCRIPT}
      --spirv-core-grammar=${GRAMMAR_JSON_FILE}
      --extinst-debuginfo-grammar=${DEBUGINFO_GRAMMAR_JSON_FILE}
      --extinst-glsl-grammar=${GLSL_GRAMMAR_JSON_FILE}
      --extinst-opencl-grammar=${OPENCL_GRAMMAR_JSON_FILE}
      >${VIMSYNTAX_FILE}
    DEPENDS ${VIMSYNTAX_PROCESSING_SCRIPT} ${GRAMMAR_JSON_FILE}
            ${GLSL_GRAMMAR_JSON_FILE} ${OPENCL_GRAMMAR_JSON_FILE} ${DEBUGINFO_GRAMMAR_JSON_FILE}
    COMMENT "Generate spvasm.vim: Vim syntax file for SPIR-V assembly.")
endmacro(spvtools_vimsyntax)

macro(spvtools_glsl_tables CONFIG_VERSION)
  set(CORE_GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/${CONFIG_VERSION}/spirv.core.grammar.json")
  set(GLSL_GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/${CONFIG_VERSION}/extinst.glsl.std.450.grammar.json")
  set(GRAMMAR_INC_FILE "${spirv-tools_BINARY_DIR}/glsl.std.450.insts.inc")
  add_custom_command(OUTPUT ${GRAMMAR_INC_FILE}
    COMMAND Python3::Interpreter ${GRAMMAR_PROCESSING_SCRIPT}
      --extinst-glsl-grammar=${GLSL_GRAMMAR_JSON_FILE}
      --glsl-insts-output=${GRAMMAR_INC_FILE}
      --output-language=c++
    DEPENDS ${GRAMMAR_PROCESSING_SCRIPT} ${CORE_GRAMMAR_JSON_FILE} ${GLSL_GRAMMAR_JSON_FILE}
    COMMENT "Generate info tables for GLSL extended instructions and operands v${CONFIG_VERSION}.")
  list(APPEND EXTINST_CPP_DEPENDS ${GRAMMAR_INC_FILE})
endmacro(spvtools_glsl_tables)

macro(spvtools_opencl_tables CONFIG_VERSION)
  set(CORE_GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/${CONFIG_VERSION}/spirv.core.grammar.json")
  set(OPENCL_GRAMMAR_JSON_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/${CONFIG_VERSION}/extinst.opencl.std.100.grammar.json")
  set(GRAMMAR_INC_FILE "${spirv-tools_BINARY_DIR}/opencl.std.insts.inc")
  add_custom_command(OUTPUT ${GRAMMAR_INC_FILE}
    COMMAND Python3::Interpreter ${GRAMMAR_PROCESSING_SCRIPT}
      --extinst-opencl-grammar=${OPENCL_GRAMMAR_JSON_FILE}
      --opencl-insts-output=${GRAMMAR_INC_FILE}
    DEPENDS ${GRAMMAR_PROCESSING_SCRIPT} ${CORE_GRAMMAR_JSON_FILE} ${OPENCL_GRAMMAR_JSON_FILE}
    COMMENT "Generate info tables for OpenCL extended instructions and operands v${CONFIG_VERSION}.")
  list(APPEND EXTINST_CPP_DEPENDS ${GRAMMAR_INC_FILE})
endmacro(spvtools_opencl_tables)

macro(spvtools_vendor_tables VENDOR_TABLE SHORT_NAME OPERAND_KIND_PREFIX)
  set(INSTS_FILE "${spirv-tools_BINARY_DIR}/${VENDOR_TABLE}.insts.inc")
  set(GRAMMAR_FILE "${SPIRV_HEADER_INCLUDE_DIR}/spirv/unified1/extinst.${VENDOR_TABLE}.grammar.json")
  if(NOT EXISTS ${GRAMMAR_FILE})
    set(GRAMMAR_FILE "${spirv-tools_SOURCE_DIR}/source/extinst.${VENDOR_TABLE}.grammar.json")
  endif()
  add_custom_command(OUTPUT ${INSTS_FILE}
    COMMAND Python3::Interpreter ${GRAMMAR_PROCESSING_SCRIPT}
      --extinst-vendor-grammar=${GRAMMAR_FILE}
      --vendor-insts-output=${INSTS_FILE}
      --vendor-operand-kind-prefix=${OPERAND_KIND_PREFIX}
    DEPENDS ${GRAMMAR_PROCESSING_SCRIPT} ${GRAMMAR_FILE}
    COMMENT "Generate extended instruction tables for ${VENDOR_TABLE}.")
  add_custom_target(spv-tools-${SHORT_NAME} DEPENDS ${INSTS_FILE})
  set_property(TARGET spv-tools-${SHORT_NAME} PROPERTY FOLDER "SPIRV-Tools build")
  list(APPEND EXTINST_CPP_DEPENDS spv-tools-${SHORT_NAME})
endmacro(spvtools_vendor_tables)

macro(spvtools_extinst_lang_headers NAME GRAMMAR_FILE)
  set(OUT_H ${spirv-tools_BINARY_DIR}/${NAME}.h)
  add_custom_command(OUTPUT ${OUT_H}
    COMMAND Python3::Interpreter ${LANG_HEADER_PROCESSING_SCRIPT}
      --extinst-grammar=${GRAMMAR_FILE}
      --extinst-output-path=${OUT_H}
    DEPENDS ${LANG_HEADER_PROCESSING_SCRIPT} ${GRAMMAR_FILE}
    COMMENT "Generate language specific header for ${NAME}.")
  add_custom_target(spirv-tools-header-${NAME} DEPENDS ${OUT_H})
  set_property(TARGET spirv-tools-header-${NAME} PROPERTY FOLDER "SPIRV-Tools build")
  list(APPEND EXTINST_CPP_DEPENDS spirv-tools-header-${NAME})
endmacro(spvtools_extinst_lang_headers)

spvtools_core_tables("unified1")
spvtools_enum_string_mapping("unified1")
spvtools_opencl_tables("unified1")
spvtools_glsl_tables("unified1")
spvtools_vendor_tables("spv-amd-shader-explicit-vertex-parameter" "spv-amd-sevp" "")
spvtools_vendor_tables("spv-amd-shader-trinary-minmax" "spv-amd-stm" "")
spvtools_vendor_tables("spv-amd-gcn-shader" "spv-amd-gs" "")
spvtools_vendor_tables("spv-amd-shader-ballot" "spv-amd-sb" "")
spvtools_vendor_tables("debuginfo" "debuginfo" "")
spvtools_vendor_tables("opencl.debuginfo.100" "cldi100" "CLDEBUG100_")
spvtools_vendor_tables("nonsemantic.shader.debuginfo.100" "shdi100" "SHDEBUG100_")
spvtools_vendor_tables("nonsemantic.clspvreflection" "clspvreflection" "")
spvtools_extinst_lang_headers("DebugInfo" ${DEBUGINFO_GRAMMAR_JSON_FILE})
spvtools_extinst_lang_headers("OpenCLDebugInfo100" ${CLDEBUGINFO100_GRAMMAR_JSON_FILE})
spvtools_extinst_lang_headers("NonSemanticShaderDebugInfo100" ${VKDEBUGINFO100_GRAMMAR_JSON_FILE})

spvtools_vimsyntax("unified1" "1.0")
add_custom_target(spirv-tools-vimsyntax DEPENDS ${VIMSYNTAX_FILE})
set_property(TARGET spirv-tools-vimsyntax PROPERTY FOLDER "SPIRV-Tools utilities")

# Extract the list of known generators from the SPIR-V XML registry file.
set(GENERATOR_INC_FILE ${spirv-tools_BINARY_DIR}/generators.inc)
set(SPIRV_XML_REGISTRY_FILE ${SPIRV_HEADER_INCLUDE_DIR}/spirv/spir-v.xml)
add_custom_command(OUTPUT ${GENERATOR_INC_FILE}
  COMMAND Python3::Interpreter ${XML_REGISTRY_PROCESSING_SCRIPT}
    --xml=${SPIRV_XML_REGISTRY_FILE}
    --generator-output=${GENERATOR_INC_FILE}
  DEPENDS ${XML_REGISTRY_PROCESSING_SCRIPT} ${SPIRV_XML_REGISTRY_FILE}
  COMMENT "Generate tables based on the SPIR-V XML registry.")
list(APPEND OPCODE_CPP_DEPENDS ${GENERATOR_INC_FILE})

# The following .cpp files include the above generated .inc files.
# Add those .inc files as their dependencies.
#
# We need to wrap the .inc files with a custom target to avoid problems when
# multiple targets depend on the same custom command.
add_custom_target(core_tables
  DEPENDS ${OPCODE_CPP_DEPENDS} ${OPERAND_CPP_DEPENDS})
add_custom_target(enum_string_mapping
  DEPENDS ${EXTENSION_H_DEPENDS} ${ENUM_STRING_MAPPING_CPP_DEPENDS})
add_custom_target(extinst_tables
  DEPENDS ${EXTINST_CPP_DEPENDS})

set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/extensions.h
  PROPERTIES HEADER_FILE_ONLY TRUE)

set(SPIRV_TOOLS_BUILD_VERSION_INC
  ${spirv-tools_BINARY_DIR}/build-version.inc)
set(SPIRV_TOOLS_BUILD_VERSION_INC_GENERATOR
  ${spirv-tools_SOURCE_DIR}/utils/update_build_version.py)
set(SPIRV_TOOLS_CHANGES_FILE
  ${spirv-tools_SOURCE_DIR}/CHANGES)
add_custom_command(OUTPUT ${SPIRV_TOOLS_BUILD_VERSION_INC}
   COMMAND Python3::Interpreter
           ${SPIRV_TOOLS_BUILD_VERSION_INC_GENERATOR}
           ${SPIRV_TOOLS_CHANGES_FILE} ${SPIRV_TOOLS_BUILD_VERSION_INC}
   DEPENDS ${SPIRV_TOOLS_BUILD_VERSION_INC_GENERATOR}
           ${SPIRV_TOOLS_CHANGES_FILE}
   COMMENT "Update build-version.inc in the SPIRV-Tools build directory (if necessary).")
# Convenience target for standalone generation of the build-version.inc file.
# This is not required for any dependence chain.
add_custom_target(spirv-tools-build-version
   DEPENDS ${SPIRV_TOOLS_BUILD_VERSION_INC})
set_property(TARGET spirv-tools-build-version PROPERTY FOLDER "SPIRV-Tools build")

list(APPEND PCH_DEPENDS ${ENUM_STRING_MAPPING_CPP_DEPENDS} ${OPCODE_CPP_DEPENDS} ${OPERAND_CPP_DEPENDS} ${EXTENSION_H_DEPENDS} ${EXTINST_CPP_DEPENDS} ${SPIRV_TOOLS_BUILD_VERSION_INC})
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/pch_source.cpp
  PROPERTIES OBJECT_DEPENDS "${PCH_DEPENDS}")

add_subdirectory(opt)
add_subdirectory(reduce)
add_subdirectory(fuzz)
add_subdirectory(link)
add_subdirectory(lint)
add_subdirectory(diff)

set(SPIRV_SOURCES
  ${spirv-tools_SOURCE_DIR}/include/spirv-tools/libspirv.h

  ${CMAKE_CURRENT_SOURCE_DIR}/util/bitutils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/util/bit_vector.h
  ${CMAKE_CURRENT_SOURCE_DIR}/util/hash_combine.h
  ${CMAKE_CURRENT_SOURCE_DIR}/util/hex_float.h
  ${CMAKE_CURRENT_SOURCE_DIR}/util/make_unique.h
  ${CMAKE_CURRENT_SOURCE_DIR}/util/parse_number.h
  ${CMAKE_CURRENT_SOURCE_DIR}/util/small_vector.h
  ${CMAKE_CURRENT_SOURCE_DIR}/util/string_utils.h
  ${CMAKE_CURRENT_SOURCE_DIR}/util/timer.h
  ${CMAKE_CURRENT_SOURCE_DIR}/assembly_grammar.h
  ${CMAKE_CURRENT_SOURCE_DIR}/binary.h
  ${CMAKE_CURRENT_SOURCE_DIR}/cfa.h
  ${CMAKE_CURRENT_SOURCE_DIR}/common_debug_info.h
  ${CMAKE_CURRENT_SOURCE_DIR}/diagnostic.h
  ${CMAKE_CURRENT_SOURCE_DIR}/disassemble.h
  ${CMAKE_CURRENT_SOURCE_DIR}/enum_set.h
  ${CMAKE_CURRENT_SOURCE_DIR}/enum_string_mapping.h
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_inst.h
  ${CMAKE_CURRENT_SOURCE_DIR}/extensions.h
  ${CMAKE_CURRENT_SOURCE_DIR}/instruction.h
  ${CMAKE_CURRENT_SOURCE_DIR}/latest_version_glsl_std_450_header.h
  ${CMAKE_CURRENT_SOURCE_DIR}/latest_version_opencl_std_header.h
  ${CMAKE_CURRENT_SOURCE_DIR}/latest_version_spirv_header.h
  ${CMAKE_CURRENT_SOURCE_DIR}/macro.h
  ${CMAKE_CURRENT_SOURCE_DIR}/name_mapper.h
  ${CMAKE_CURRENT_SOURCE_DIR}/opcode.h
  ${CMAKE_CURRENT_SOURCE_DIR}/operand.h
  ${CMAKE_CURRENT_SOURCE_DIR}/parsed_operand.h
  ${CMAKE_CURRENT_SOURCE_DIR}/print.h
  ${CMAKE_CURRENT_SOURCE_DIR}/spirv_constant.h
  ${CMAKE_CURRENT_SOURCE_DIR}/spirv_definition.h
  ${CMAKE_CURRENT_SOURCE_DIR}/spirv_endian.h
  ${CMAKE_CURRENT_SOURCE_DIR}/spirv_fuzzer_options.h
  ${CMAKE_CURRENT_SOURCE_DIR}/spirv_optimizer_options.h
  ${CMAKE_CURRENT_SOURCE_DIR}/spirv_reducer_options.h
  ${CMAKE_CURRENT_SOURCE_DIR}/spirv_target_env.h
  ${CMAKE_CURRENT_SOURCE_DIR}/spirv_validator_options.h
  ${CMAKE_CURRENT_SOURCE_DIR}/table.h
  ${CMAKE_CURRENT_SOURCE_DIR}/text.h
  ${CMAKE_CURRENT_SOURCE_DIR}/text_handler.h
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate.h

  ${CMAKE_CURRENT_SOURCE_DIR}/util/bit_vector.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/util/parse_number.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/util/string_utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/assembly_grammar.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/binary.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/diagnostic.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/disassemble.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/enum_string_mapping.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ext_inst.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/extensions.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/libspirv.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/name_mapper.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/opcode.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/operand.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/parsed_operand.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/print.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/software_version.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/spirv_endian.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/spirv_fuzzer_options.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/spirv_optimizer_options.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/spirv_reducer_options.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/spirv_target_env.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/spirv_validator_options.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/table.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/text_handler.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_adjacency.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_annotation.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_arithmetics.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_atomics.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_barriers.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_bitwise.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_builtins.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_capability.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_cfg.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_composites.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_constants.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_conversion.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_debug.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_decorations.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_derivatives.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_extensions.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_execution_limitations.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_function.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_id.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_image.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_interfaces.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_instruction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_layout.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_literals.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_logicals.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_memory.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_memory_semantics.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_mesh_shading.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_misc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_mode_setting.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_non_uniform.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_primitives.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_ray_query.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_ray_tracing.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_ray_tracing_reorder.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_scopes.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_small_type_uses.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validate_type.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/decoration.h
  ${CMAKE_CURRENT_SOURCE_DIR}/val/basic_block.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/construct.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/function.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/instruction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/val/validation_state.cpp)

if (${SPIRV_TIMER_ENABLED})
  set(SPIRV_SOURCES
    ${SPIRV_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/util/timer.cpp)
endif()

# The software_version.cpp file includes build-version.inc.
# Rebuild the software_version.cpp object file if it is older than
# build-version.inc or whenever build-version.inc itself is out of
# date.  In the latter case, rebuild build-version.inc first.
# CMake is not smart enough to detect this dependency automatically.
# Without this, the dependency detection system for #included files
# does not kick in on a clean build for the following reason:  The
# build will fail early because it doesn't know how to build the
# missing source file build-version.inc. That occurs before the
# preprocessor is run on software_version.cpp to detect the
# #include dependency.
set_source_files_properties(
  ${CMAKE_CURRENT_SOURCE_DIR}/software_version.cpp
  PROPERTIES OBJECT_DEPENDS "${SPIRV_TOOLS_BUILD_VERSION_INC}")

spvtools_pch(SPIRV_SOURCES pch_source)

# spirv_tools_default_target_options() sets the target options that are common
# for all ${SPIRV_TOOLS} targets.
function(spirv_tools_default_target_options target)
  spvtools_default_compile_options(${target})
  target_include_directories(${target}
    PUBLIC
      $<BUILD_INTERFACE:${spirv-tools_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE ${spirv-tools_BINARY_DIR}
    PRIVATE ${SPIRV_HEADER_INCLUDE_DIR}
  )
  set_property(TARGET ${target} PROPERTY FOLDER "SPIRV-Tools libraries")
  spvtools_check_symbol_exports(${target})
  add_dependencies(${target} spirv-tools-build-version core_tables enum_string_mapping extinst_tables)
endfunction()

# Always build ${SPIRV_TOOLS}-shared. This is expected distro packages, and
# unlike the other SPIRV_TOOLS target, defaults to hidden symbol visibility.
add_library(${SPIRV_TOOLS}-shared SHARED ${SPIRV_SOURCES})
spirv_tools_default_target_options(${SPIRV_TOOLS}-shared)
set_target_properties(${SPIRV_TOOLS}-shared PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_compile_definitions(${SPIRV_TOOLS}-shared
  PRIVATE SPIRV_TOOLS_IMPLEMENTATION
  PUBLIC SPIRV_TOOLS_SHAREDLIB
)

if(SPIRV_TOOLS_BUILD_STATIC)
  add_library(${SPIRV_TOOLS}-static STATIC ${SPIRV_SOURCES})
  spirv_tools_default_target_options(${SPIRV_TOOLS}-static)
  # The static target does not have the '-static' suffix.
  set_target_properties(${SPIRV_TOOLS}-static PROPERTIES OUTPUT_NAME "${SPIRV_TOOLS}")

  # Create the "${SPIRV_TOOLS}" target as an alias to either "${SPIRV_TOOLS}-static"
  # or "${SPIRV_TOOLS}-shared" depending on the value of BUILD_SHARED_LIBS.
  if(BUILD_SHARED_LIBS)
    add_library(${SPIRV_TOOLS} ALIAS ${SPIRV_TOOLS}-shared)
  else()
    add_library(${SPIRV_TOOLS} ALIAS ${SPIRV_TOOLS}-static)
  endif()

  set(SPIRV_TOOLS_TARGETS ${SPIRV_TOOLS}-static ${SPIRV_TOOLS}-shared)
else()
  add_library(${SPIRV_TOOLS} ${SPIRV_TOOLS_LIBRARY_TYPE} ${SPIRV_SOURCES})
  spirv_tools_default_target_options(${SPIRV_TOOLS})
  set(SPIRV_TOOLS_TARGETS ${SPIRV_TOOLS} ${SPIRV_TOOLS}-shared)
endif()

if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
  find_library(LIBRT rt)
  if(LIBRT)
    foreach(target ${SPIRV_TOOLS_TARGETS})
      target_link_libraries(${target} rt)
    endforeach()
  endif()
endif()

if(ENABLE_SPIRV_TOOLS_INSTALL)
  install(TARGETS ${SPIRV_TOOLS_TARGETS} EXPORT ${SPIRV_TOOLS}Targets)
  export(EXPORT ${SPIRV_TOOLS}Targets FILE ${SPIRV_TOOLS}Target.cmake)

  spvtools_config_package_dir(${SPIRV_TOOLS} PACKAGE_DIR)
  install(EXPORT ${SPIRV_TOOLS}Targets FILE ${SPIRV_TOOLS}Target.cmake DESTINATION ${PACKAGE_DIR})

  # Special config file for root library compared to other libs.
  file(WRITE ${CMAKE_BINARY_DIR}/${SPIRV_TOOLS}Config.cmake
    "include(\${CMAKE_CURRENT_LIST_DIR}/${SPIRV_TOOLS}Target.cmake)\n"
    "if(TARGET ${SPIRV_TOOLS})\n"
    "    set(${SPIRV_TOOLS}_LIBRARIES ${SPIRV_TOOLS})\n"
    "    get_target_property(${SPIRV_TOOLS}_INCLUDE_DIRS ${SPIRV_TOOLS} INTERFACE_INCLUDE_DIRECTORIES)\n"
    "endif()\n")
  install(FILES ${CMAKE_BINARY_DIR}/${SPIRV_TOOLS}Config.cmake DESTINATION ${PACKAGE_DIR})
endif(ENABLE_SPIRV_TOOLS_INSTALL)

if(MSVC AND (NOT ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")))
  # Enable parallel builds across four cores for this lib
  add_definitions(/MP4)
endif()
