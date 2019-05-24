# Copyright (c) 2016 Google Inc.

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
set(SPIRV_TOOLS_OPT_SOURCES
  aggressive_dead_code_elim_pass.h
  basic_block.h
  block_merge_pass.h
  block_merge_util.h
  build_module.h
  ccp_pass.h
  cfg_cleanup_pass.h
  cfg.h
  code_sink.h
  combine_access_chains.h
  common_uniform_elim_pass.h
  compact_ids_pass.h
  composite.h
  const_folding_rules.h
  constants.h
  copy_prop_arrays.h
  dead_branch_elim_pass.h
  dead_insert_elim_pass.h
  dead_variable_elimination.h
  decoration_manager.h
  def_use_manager.h
  dominator_analysis.h
  dominator_tree.h
  eliminate_dead_constant_pass.h
  eliminate_dead_functions_pass.h
  eliminate_dead_functions_util.h
  eliminate_dead_members_pass.h
  feature_manager.h
  fix_storage_class.h
  flatten_decoration_pass.h
  fold.h
  folding_rules.h
  fold_spec_constant_op_and_composite_pass.h
  freeze_spec_constant_value_pass.h
  function.h
  generate_webgpu_initializers_pass.h
  if_conversion.h
  inline_exhaustive_pass.h
  inline_opaque_pass.h
  inline_pass.h
  inst_bindless_check_pass.h
  instruction.h
  instruction_list.h
  instrument_pass.h
  ir_builder.h
  ir_context.h
  ir_loader.h
  licm_pass.h
  local_access_chain_convert_pass.h
  local_redundancy_elimination.h
  local_single_block_elim_pass.h
  local_single_store_elim_pass.h
  local_ssa_elim_pass.h
  log.h
  loop_dependence.h
  loop_descriptor.h
  loop_fission.h
  loop_fusion.h
  loop_fusion_pass.h
  loop_peeling.h
  loop_unroller.h
  loop_utils.h
  loop_unswitch_pass.h
  mem_pass.h
  merge_return_pass.h
  module.h
  null_pass.h
  passes.h
  pass.h
  pass_manager.h
  private_to_local_pass.h
  process_lines_pass.h
  propagator.h
  reduce_load_size.h
  redundancy_elimination.h
  reflect.h
  register_pressure.h
  remove_duplicates_pass.h
  replace_invalid_opc.h
  scalar_analysis.h
  scalar_analysis_nodes.h
  scalar_replacement_pass.h
  set_spec_constant_default_value_pass.h
  simplification_pass.h
  ssa_rewrite_pass.h
  strength_reduction_pass.h
  strip_atomic_counter_memory_pass.h
  strip_debug_info_pass.h
  strip_reflect_info_pass.h
  struct_cfg_analysis.h
  tree_iterator.h
  type_manager.h
  types.h
  unify_const_pass.h
  upgrade_memory_model.h
  value_number_table.h
  vector_dce.h
  workaround1209.h

  aggressive_dead_code_elim_pass.cpp
  basic_block.cpp
  block_merge_pass.cpp
  block_merge_util.cpp
  build_module.cpp
  ccp_pass.cpp
  cfg_cleanup_pass.cpp
  cfg.cpp
  code_sink.cpp
  combine_access_chains.cpp
  common_uniform_elim_pass.cpp
  compact_ids_pass.cpp
  composite.cpp
  const_folding_rules.cpp
  constants.cpp
  copy_prop_arrays.cpp
  dead_branch_elim_pass.cpp
  dead_insert_elim_pass.cpp
  dead_variable_elimination.cpp
  decoration_manager.cpp
  def_use_manager.cpp
  dominator_analysis.cpp
  dominator_tree.cpp
  eliminate_dead_constant_pass.cpp
  eliminate_dead_functions_pass.cpp
  eliminate_dead_functions_util.cpp
  eliminate_dead_members_pass.cpp
  feature_manager.cpp
  fix_storage_class.cpp
  flatten_decoration_pass.cpp
  fold.cpp
  folding_rules.cpp
  fold_spec_constant_op_and_composite_pass.cpp
  freeze_spec_constant_value_pass.cpp
  function.cpp
  generate_webgpu_initializers_pass.cpp
  if_conversion.cpp
  inline_exhaustive_pass.cpp
  inline_opaque_pass.cpp
  inline_pass.cpp
  inst_bindless_check_pass.cpp
  instruction.cpp
  instruction_list.cpp
  instrument_pass.cpp
  ir_context.cpp
  ir_loader.cpp
  licm_pass.cpp
  local_access_chain_convert_pass.cpp
  local_redundancy_elimination.cpp
  local_single_block_elim_pass.cpp
  local_single_store_elim_pass.cpp
  local_ssa_elim_pass.cpp
  loop_dependence.cpp
  loop_dependence_helpers.cpp
  loop_descriptor.cpp
  loop_fission.cpp
  loop_fusion.cpp
  loop_fusion_pass.cpp
  loop_peeling.cpp
  loop_utils.cpp
  loop_unroller.cpp
  loop_unswitch_pass.cpp
  mem_pass.cpp
  merge_return_pass.cpp
  module.cpp
  optimizer.cpp
  pass.cpp
  pass_manager.cpp
  private_to_local_pass.cpp
  process_lines_pass.cpp
  propagator.cpp
  reduce_load_size.cpp
  redundancy_elimination.cpp
  register_pressure.cpp
  remove_duplicates_pass.cpp
  replace_invalid_opc.cpp
  scalar_analysis.cpp
  scalar_analysis_simplification.cpp
  scalar_replacement_pass.cpp
  set_spec_constant_default_value_pass.cpp
  simplification_pass.cpp
  ssa_rewrite_pass.cpp
  strength_reduction_pass.cpp
  strip_atomic_counter_memory_pass.cpp
  strip_debug_info_pass.cpp
  strip_reflect_info_pass.cpp
  struct_cfg_analysis.cpp
  type_manager.cpp
  types.cpp
  unify_const_pass.cpp
  upgrade_memory_model.cpp
  value_number_table.cpp
  vector_dce.cpp
  workaround1209.cpp
)

if(MSVC)
  # Enable parallel builds across four cores for this lib
  add_definitions(/MP4)
endif()

spvtools_pch(SPIRV_TOOLS_OPT_SOURCES pch_source_opt)

add_library(SPIRV-Tools-opt ${SPIRV_TOOLS_OPT_SOURCES})

spvtools_default_compile_options(SPIRV-Tools-opt)
target_include_directories(SPIRV-Tools-opt
  PUBLIC ${spirv-tools_SOURCE_DIR}/include
  PUBLIC ${SPIRV_HEADER_INCLUDE_DIR}
  PRIVATE ${spirv-tools_BINARY_DIR}
)
# We need the assembling and disassembling functionalities in the main library.
target_link_libraries(SPIRV-Tools-opt
  PUBLIC ${SPIRV_TOOLS})

set_property(TARGET SPIRV-Tools-opt PROPERTY FOLDER "SPIRV-Tools libraries")
spvtools_check_symbol_exports(SPIRV-Tools-opt)

if(ENABLE_SPIRV_TOOLS_INSTALL)
  install(TARGETS SPIRV-Tools-opt
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif(ENABLE_SPIRV_TOOLS_INSTALL)
