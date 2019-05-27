# Copyright (c) 2018 Google LLC

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
set(SPIRV_TOOLS_REDUCE_SOURCES
        change_operand_reduction_opportunity.h
        change_operand_to_undef_reduction_opportunity.h
        merge_blocks_reduction_opportunity.h
        merge_blocks_reduction_opportunity_finder.h
        operand_to_const_reduction_opportunity_finder.h
        operand_to_undef_reduction_opportunity_finder.h
        operand_to_dominating_id_reduction_opportunity_finder.h
        reducer.h
        reduction_opportunity.h
        reduction_opportunity_finder.h
        reduction_pass.h
        reduction_util.h
        remove_block_reduction_opportunity.h
        remove_block_reduction_opportunity_finder.h
        remove_instruction_reduction_opportunity.h
        remove_function_reduction_opportunity.h
        remove_function_reduction_opportunity_finder.h
        remove_opname_instruction_reduction_opportunity_finder.h
        remove_selection_reduction_opportunity.h
        remove_selection_reduction_opportunity_finder.h
        remove_unreferenced_instruction_reduction_opportunity_finder.h
        structured_loop_to_selection_reduction_opportunity.h
        structured_loop_to_selection_reduction_opportunity_finder.h

        change_operand_reduction_opportunity.cpp
        change_operand_to_undef_reduction_opportunity.cpp
        merge_blocks_reduction_opportunity.cpp
        merge_blocks_reduction_opportunity_finder.cpp
        operand_to_const_reduction_opportunity_finder.cpp
        operand_to_undef_reduction_opportunity_finder.cpp
        operand_to_dominating_id_reduction_opportunity_finder.cpp
        reducer.cpp
        reduction_opportunity.cpp
        reduction_pass.cpp
        reduction_util.cpp
        remove_block_reduction_opportunity.cpp
        remove_block_reduction_opportunity_finder.cpp
        remove_function_reduction_opportunity.cpp
        remove_function_reduction_opportunity_finder.cpp
        remove_instruction_reduction_opportunity.cpp
        remove_selection_reduction_opportunity.cpp
        remove_selection_reduction_opportunity_finder.cpp
        remove_unreferenced_instruction_reduction_opportunity_finder.cpp
        remove_opname_instruction_reduction_opportunity_finder.cpp
        structured_loop_to_selection_reduction_opportunity.cpp
        structured_loop_to_selection_reduction_opportunity_finder.cpp
        )

if(MSVC)
  # Enable parallel builds across four cores for this lib
  add_definitions(/MP4)
endif()

spvtools_pch(SPIRV_TOOLS_REDUCE_SOURCES pch_source_reduce)

add_library(SPIRV-Tools-reduce ${SPIRV_TOOLS_REDUCE_SOURCES})

spvtools_default_compile_options(SPIRV-Tools-reduce)
target_include_directories(SPIRV-Tools-reduce
  PUBLIC ${spirv-tools_SOURCE_DIR}/include
  PUBLIC ${SPIRV_HEADER_INCLUDE_DIR}
  PRIVATE ${spirv-tools_BINARY_DIR}
)
# The reducer reuses a lot of functionality from the SPIRV-Tools library.
target_link_libraries(SPIRV-Tools-reduce
  PUBLIC ${SPIRV_TOOLS}
  PUBLIC SPIRV-Tools-opt)

set_property(TARGET SPIRV-Tools-reduce PROPERTY FOLDER "SPIRV-Tools libraries")
spvtools_check_symbol_exports(SPIRV-Tools-reduce)

if(ENABLE_SPIRV_TOOLS_INSTALL)
  install(TARGETS SPIRV-Tools-reduce
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif(ENABLE_SPIRV_TOOLS_INSTALL)
