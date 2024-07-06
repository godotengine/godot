# Copyright (c) 2019 Google LLC

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

if(SPIRV_BUILD_FUZZER)

  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/protobufs)

  set(PROTOBUF_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/protobufs/spvtoolsfuzz.proto)

  set(
        SPIRV_FUZZ_PROTOC_COMMAND
        "protobuf::protoc"
        CACHE
        STRING
        "The command to invoke the protobuf compiler (protoc). By default it is the protobufs::protoc CMake target. It should be overridden when cross-compiling, such as for Android.")

  add_custom_command(
        OUTPUT protobufs/spvtoolsfuzz.pb.cc protobufs/spvtoolsfuzz.pb.h
        COMMAND "${SPIRV_FUZZ_PROTOC_COMMAND}"
        -I=${CMAKE_CURRENT_SOURCE_DIR}/protobufs
        --cpp_out=protobufs
        ${PROTOBUF_SOURCE}
        DEPENDS ${PROTOBUF_SOURCE}
        COMMENT "Generate protobuf sources from proto definition file."
  )

  set(SPIRV_TOOLS_FUZZ_SOURCES
        added_function_reducer.h
        available_instructions.h
        call_graph.h
        comparator_deep_blocks_first.h
        counter_overflow_id_source.h
        data_descriptor.h
        equivalence_relation.h
        fact_manager/constant_uniform_facts.h
        fact_manager/data_synonym_and_id_equation_facts.h
        fact_manager/dead_block_facts.h
        fact_manager/fact_manager.h
        fact_manager/irrelevant_value_facts.h
        fact_manager/livesafe_function_facts.h
        force_render_red.h
        fuzzer.h
        fuzzer_context.h
        fuzzer_pass.h
        fuzzer_pass_add_access_chains.h
        fuzzer_pass_add_bit_instruction_synonyms.h
        fuzzer_pass_add_composite_extract.h
        fuzzer_pass_add_composite_inserts.h
        fuzzer_pass_add_composite_types.h
        fuzzer_pass_add_copy_memory.h
        fuzzer_pass_add_dead_blocks.h
        fuzzer_pass_add_dead_breaks.h
        fuzzer_pass_add_dead_continues.h
        fuzzer_pass_add_equation_instructions.h
        fuzzer_pass_add_function_calls.h
        fuzzer_pass_add_global_variables.h
        fuzzer_pass_add_image_sample_unused_components.h
        fuzzer_pass_add_loads.h
        fuzzer_pass_add_local_variables.h
        fuzzer_pass_add_loop_preheaders.h
        fuzzer_pass_add_loops_to_create_int_constant_synonyms.h
        fuzzer_pass_add_no_contraction_decorations.h
        fuzzer_pass_add_opphi_synonyms.h
        fuzzer_pass_add_parameters.h
        fuzzer_pass_add_relaxed_decorations.h
        fuzzer_pass_add_stores.h
        fuzzer_pass_add_synonyms.h
        fuzzer_pass_add_vector_shuffle_instructions.h
        fuzzer_pass_adjust_branch_weights.h
        fuzzer_pass_adjust_function_controls.h
        fuzzer_pass_adjust_loop_controls.h
        fuzzer_pass_adjust_memory_operands_masks.h
        fuzzer_pass_adjust_selection_controls.h
        fuzzer_pass_apply_id_synonyms.h
        fuzzer_pass_construct_composites.h
        fuzzer_pass_copy_objects.h
        fuzzer_pass_donate_modules.h
        fuzzer_pass_duplicate_regions_with_selections.h
        fuzzer_pass_expand_vector_reductions.h
        fuzzer_pass_flatten_conditional_branches.h
        fuzzer_pass_inline_functions.h
        fuzzer_pass_invert_comparison_operators.h
        fuzzer_pass_interchange_signedness_of_integer_operands.h
        fuzzer_pass_interchange_zero_like_constants.h
        fuzzer_pass_make_vector_operations_dynamic.h
        fuzzer_pass_merge_blocks.h
        fuzzer_pass_merge_function_returns.h
        fuzzer_pass_mutate_pointers.h
        fuzzer_pass_obfuscate_constants.h
        fuzzer_pass_outline_functions.h
        fuzzer_pass_permute_blocks.h
        fuzzer_pass_permute_function_parameters.h
        fuzzer_pass_permute_function_variables.h
        fuzzer_pass_permute_instructions.h
        fuzzer_pass_permute_phi_operands.h
        fuzzer_pass_propagate_instructions_down.h
        fuzzer_pass_propagate_instructions_up.h
        fuzzer_pass_push_ids_through_variables.h
        fuzzer_pass_replace_adds_subs_muls_with_carrying_extended.h
        fuzzer_pass_replace_branches_from_dead_blocks_with_exits.h
        fuzzer_pass_replace_copy_memories_with_loads_stores.h
        fuzzer_pass_replace_copy_objects_with_stores_loads.h
        fuzzer_pass_replace_irrelevant_ids.h
        fuzzer_pass_replace_linear_algebra_instructions.h
        fuzzer_pass_replace_loads_stores_with_copy_memories.h
        fuzzer_pass_replace_opphi_ids_from_dead_predecessors.h
        fuzzer_pass_replace_opselects_with_conditional_branches.h
        fuzzer_pass_replace_parameter_with_global.h
        fuzzer_pass_replace_params_with_struct.h
        fuzzer_pass_split_blocks.h
        fuzzer_pass_swap_commutable_operands.h
        fuzzer_pass_swap_conditional_branch_operands.h
        fuzzer_pass_swap_functions.h
        fuzzer_pass_toggle_access_chain_instruction.h
        fuzzer_pass_wrap_regions_in_selections.h
        fuzzer_pass_wrap_vector_synonym.h
        fuzzer_util.h
        id_use_descriptor.h
        instruction_descriptor.h
        instruction_message.h
        overflow_id_source.h
        pass_management/repeated_pass_instances.h
        pass_management/repeated_pass_manager.h
        pass_management/repeated_pass_manager_looped_with_recommendations.h
        pass_management/repeated_pass_manager_random_with_recommendations.h
        pass_management/repeated_pass_manager_simple.h
        pass_management/repeated_pass_recommender.h
        pass_management/repeated_pass_recommender_standard.h
        protobufs/spirvfuzz_protobufs.h
        pseudo_random_generator.h
        random_generator.h
        replayer.h
        shrinker.h
        transformation.h
        transformation_access_chain.h
        transformation_add_bit_instruction_synonym.h
        transformation_add_constant_boolean.h
        transformation_add_constant_composite.h
        transformation_add_constant_null.h
        transformation_add_constant_scalar.h
        transformation_add_copy_memory.h
        transformation_add_dead_block.h
        transformation_add_dead_break.h
        transformation_add_dead_continue.h
        transformation_add_early_terminator_wrapper.h
        transformation_add_function.h
        transformation_add_global_undef.h
        transformation_add_global_variable.h
        transformation_add_image_sample_unused_components.h
        transformation_add_local_variable.h
        transformation_add_loop_preheader.h
        transformation_add_loop_to_create_int_constant_synonym.h
        transformation_add_no_contraction_decoration.h
        transformation_add_opphi_synonym.h
        transformation_add_parameter.h
        transformation_add_relaxed_decoration.h
        transformation_add_spec_constant_op.h
        transformation_add_synonym.h
        transformation_add_type_array.h
        transformation_add_type_boolean.h
        transformation_add_type_float.h
        transformation_add_type_function.h
        transformation_add_type_int.h
        transformation_add_type_matrix.h
        transformation_add_type_pointer.h
        transformation_add_type_struct.h
        transformation_add_type_vector.h
        transformation_adjust_branch_weights.h
        transformation_composite_construct.h
        transformation_composite_extract.h
        transformation_composite_insert.h
        transformation_compute_data_synonym_fact_closure.h
        transformation_context.h
        transformation_duplicate_region_with_selection.h
        transformation_equation_instruction.h
        transformation_expand_vector_reduction.h
        transformation_flatten_conditional_branch.h
        transformation_function_call.h
        transformation_inline_function.h
        transformation_invert_comparison_operator.h
        transformation_load.h
        transformation_make_vector_operation_dynamic.h
        transformation_merge_blocks.h
        transformation_merge_function_returns.h
        transformation_move_block_down.h
        transformation_move_instruction_down.h
        transformation_mutate_pointer.h
        transformation_outline_function.h
        transformation_permute_function_parameters.h
        transformation_permute_phi_operands.h
        transformation_propagate_instruction_down.h
        transformation_propagate_instruction_up.h
        transformation_push_id_through_variable.h
        transformation_record_synonymous_constants.h
        transformation_replace_add_sub_mul_with_carrying_extended.h
        transformation_replace_boolean_constant_with_constant_binary.h
        transformation_replace_branch_from_dead_block_with_exit.h
        transformation_replace_constant_with_uniform.h
        transformation_replace_copy_memory_with_load_store.h
        transformation_replace_copy_object_with_store_load.h
        transformation_replace_id_with_synonym.h
        transformation_replace_irrelevant_id.h
        transformation_replace_linear_algebra_instruction.h
        transformation_replace_load_store_with_copy_memory.h
        transformation_replace_opphi_id_from_dead_predecessor.h
        transformation_replace_opselect_with_conditional_branch.h
        transformation_replace_parameter_with_global.h
        transformation_replace_params_with_struct.h
        transformation_set_function_control.h
        transformation_set_loop_control.h
        transformation_set_memory_operands_mask.h
        transformation_set_selection_control.h
        transformation_split_block.h
        transformation_store.h
        transformation_swap_commutable_operands.h
        transformation_swap_conditional_branch_operands.h
        transformation_swap_function_variables.h
        transformation_swap_two_functions.h
        transformation_toggle_access_chain_instruction.h
        transformation_vector_shuffle.h
        transformation_wrap_early_terminator_in_function.h
        transformation_wrap_region_in_selection.h
        transformation_wrap_vector_synonym.h
        uniform_buffer_element_descriptor.h
        ${CMAKE_CURRENT_BINARY_DIR}/protobufs/spvtoolsfuzz.pb.h

        added_function_reducer.cpp
        available_instructions.cpp
        call_graph.cpp
        counter_overflow_id_source.cpp
        data_descriptor.cpp
        fact_manager/constant_uniform_facts.cpp
        fact_manager/data_synonym_and_id_equation_facts.cpp
        fact_manager/dead_block_facts.cpp
        fact_manager/fact_manager.cpp
        fact_manager/irrelevant_value_facts.cpp
        fact_manager/livesafe_function_facts.cpp
        force_render_red.cpp
        fuzzer.cpp
        fuzzer_context.cpp
        fuzzer_pass.cpp
        fuzzer_pass_add_access_chains.cpp
        fuzzer_pass_add_bit_instruction_synonyms.cpp
        fuzzer_pass_add_composite_extract.cpp
        fuzzer_pass_add_composite_inserts.cpp
        fuzzer_pass_add_composite_types.cpp
        fuzzer_pass_add_copy_memory.cpp
        fuzzer_pass_add_dead_blocks.cpp
        fuzzer_pass_add_dead_breaks.cpp
        fuzzer_pass_add_dead_continues.cpp
        fuzzer_pass_add_equation_instructions.cpp
        fuzzer_pass_add_function_calls.cpp
        fuzzer_pass_add_global_variables.cpp
        fuzzer_pass_add_image_sample_unused_components.cpp
        fuzzer_pass_add_loads.cpp
        fuzzer_pass_add_local_variables.cpp
        fuzzer_pass_add_loop_preheaders.cpp
        fuzzer_pass_add_loops_to_create_int_constant_synonyms.cpp
        fuzzer_pass_add_no_contraction_decorations.cpp
        fuzzer_pass_add_opphi_synonyms.cpp
        fuzzer_pass_add_parameters.cpp
        fuzzer_pass_add_relaxed_decorations.cpp
        fuzzer_pass_add_stores.cpp
        fuzzer_pass_add_synonyms.cpp
        fuzzer_pass_add_vector_shuffle_instructions.cpp
        fuzzer_pass_adjust_branch_weights.cpp
        fuzzer_pass_adjust_function_controls.cpp
        fuzzer_pass_adjust_loop_controls.cpp
        fuzzer_pass_adjust_memory_operands_masks.cpp
        fuzzer_pass_adjust_selection_controls.cpp
        fuzzer_pass_apply_id_synonyms.cpp
        fuzzer_pass_construct_composites.cpp
        fuzzer_pass_copy_objects.cpp
        fuzzer_pass_donate_modules.cpp
        fuzzer_pass_duplicate_regions_with_selections.cpp
        fuzzer_pass_expand_vector_reductions.cpp
        fuzzer_pass_flatten_conditional_branches.cpp
        fuzzer_pass_inline_functions.cpp
        fuzzer_pass_invert_comparison_operators.cpp
        fuzzer_pass_interchange_signedness_of_integer_operands.cpp
        fuzzer_pass_interchange_zero_like_constants.cpp
        fuzzer_pass_make_vector_operations_dynamic.cpp
        fuzzer_pass_merge_blocks.cpp
        fuzzer_pass_merge_function_returns.cpp
        fuzzer_pass_mutate_pointers.cpp
        fuzzer_pass_obfuscate_constants.cpp
        fuzzer_pass_outline_functions.cpp
        fuzzer_pass_permute_blocks.cpp
        fuzzer_pass_permute_function_parameters.cpp
        fuzzer_pass_permute_function_variables.cpp
        fuzzer_pass_permute_instructions.cpp
        fuzzer_pass_permute_phi_operands.cpp
        fuzzer_pass_propagate_instructions_down.cpp
        fuzzer_pass_propagate_instructions_up.cpp
        fuzzer_pass_push_ids_through_variables.cpp
        fuzzer_pass_replace_adds_subs_muls_with_carrying_extended.cpp
        fuzzer_pass_replace_branches_from_dead_blocks_with_exits.cpp
        fuzzer_pass_replace_copy_memories_with_loads_stores.cpp
        fuzzer_pass_replace_copy_objects_with_stores_loads.cpp
        fuzzer_pass_replace_irrelevant_ids.cpp
        fuzzer_pass_replace_linear_algebra_instructions.cpp
        fuzzer_pass_replace_loads_stores_with_copy_memories.cpp
        fuzzer_pass_replace_opphi_ids_from_dead_predecessors.cpp
        fuzzer_pass_replace_opselects_with_conditional_branches.cpp
        fuzzer_pass_replace_parameter_with_global.cpp
        fuzzer_pass_replace_params_with_struct.cpp
        fuzzer_pass_split_blocks.cpp
        fuzzer_pass_swap_commutable_operands.cpp
        fuzzer_pass_swap_conditional_branch_operands.cpp
        fuzzer_pass_swap_functions.cpp
        fuzzer_pass_toggle_access_chain_instruction.cpp
        fuzzer_pass_wrap_regions_in_selections.cpp
        fuzzer_pass_wrap_vector_synonym.cpp
        fuzzer_util.cpp
        id_use_descriptor.cpp
        instruction_descriptor.cpp
        instruction_message.cpp
        overflow_id_source.cpp
        pass_management/repeated_pass_manager.cpp
        pass_management/repeated_pass_manager_looped_with_recommendations.cpp
        pass_management/repeated_pass_manager_random_with_recommendations.cpp
        pass_management/repeated_pass_manager_simple.cpp
        pass_management/repeated_pass_recommender.cpp
        pass_management/repeated_pass_recommender_standard.cpp
        pseudo_random_generator.cpp
        random_generator.cpp
        replayer.cpp
        shrinker.cpp
        transformation.cpp
        transformation_access_chain.cpp
        transformation_add_bit_instruction_synonym.cpp
        transformation_add_constant_boolean.cpp
        transformation_add_constant_composite.cpp
        transformation_add_constant_null.cpp
        transformation_add_constant_scalar.cpp
        transformation_add_copy_memory.cpp
        transformation_add_dead_block.cpp
        transformation_add_dead_break.cpp
        transformation_add_dead_continue.cpp
        transformation_add_early_terminator_wrapper.cpp
        transformation_add_function.cpp
        transformation_add_global_undef.cpp
        transformation_add_global_variable.cpp
        transformation_add_image_sample_unused_components.cpp
        transformation_add_local_variable.cpp
        transformation_add_loop_preheader.cpp
        transformation_add_loop_to_create_int_constant_synonym.cpp
        transformation_add_no_contraction_decoration.cpp
        transformation_add_opphi_synonym.cpp
        transformation_add_parameter.cpp
        transformation_add_relaxed_decoration.cpp
        transformation_add_spec_constant_op.cpp
        transformation_add_synonym.cpp
        transformation_add_type_array.cpp
        transformation_add_type_boolean.cpp
        transformation_add_type_float.cpp
        transformation_add_type_function.cpp
        transformation_add_type_int.cpp
        transformation_add_type_matrix.cpp
        transformation_add_type_pointer.cpp
        transformation_add_type_struct.cpp
        transformation_add_type_vector.cpp
        transformation_adjust_branch_weights.cpp
        transformation_composite_construct.cpp
        transformation_composite_extract.cpp
        transformation_composite_insert.cpp
        transformation_compute_data_synonym_fact_closure.cpp
        transformation_context.cpp
        transformation_duplicate_region_with_selection.cpp
        transformation_equation_instruction.cpp
        transformation_expand_vector_reduction.cpp
        transformation_flatten_conditional_branch.cpp
        transformation_function_call.cpp
        transformation_inline_function.cpp
        transformation_invert_comparison_operator.cpp
        transformation_load.cpp
        transformation_make_vector_operation_dynamic.cpp
        transformation_merge_blocks.cpp
        transformation_merge_function_returns.cpp
        transformation_move_block_down.cpp
        transformation_move_instruction_down.cpp
        transformation_mutate_pointer.cpp
        transformation_outline_function.cpp
        transformation_permute_function_parameters.cpp
        transformation_permute_phi_operands.cpp
        transformation_propagate_instruction_down.cpp
        transformation_propagate_instruction_up.cpp
        transformation_push_id_through_variable.cpp
        transformation_record_synonymous_constants.cpp
        transformation_replace_add_sub_mul_with_carrying_extended.cpp
        transformation_replace_boolean_constant_with_constant_binary.cpp
        transformation_replace_branch_from_dead_block_with_exit.cpp
        transformation_replace_constant_with_uniform.cpp
        transformation_replace_copy_memory_with_load_store.cpp
        transformation_replace_copy_object_with_store_load.cpp
        transformation_replace_id_with_synonym.cpp
        transformation_replace_irrelevant_id.cpp
        transformation_replace_linear_algebra_instruction.cpp
        transformation_replace_load_store_with_copy_memory.cpp
        transformation_replace_opphi_id_from_dead_predecessor.cpp
        transformation_replace_opselect_with_conditional_branch.cpp
        transformation_replace_parameter_with_global.cpp
        transformation_replace_params_with_struct.cpp
        transformation_set_function_control.cpp
        transformation_set_loop_control.cpp
        transformation_set_memory_operands_mask.cpp
        transformation_set_selection_control.cpp
        transformation_split_block.cpp
        transformation_store.cpp
        transformation_swap_commutable_operands.cpp
        transformation_swap_conditional_branch_operands.cpp
        transformation_swap_function_variables.cpp
        transformation_swap_two_functions.cpp
        transformation_toggle_access_chain_instruction.cpp
        transformation_vector_shuffle.cpp
        transformation_wrap_early_terminator_in_function.cpp
        transformation_wrap_region_in_selection.cpp
        transformation_wrap_vector_synonym.cpp
        uniform_buffer_element_descriptor.cpp
        ${CMAKE_CURRENT_BINARY_DIR}/protobufs/spvtoolsfuzz.pb.cc
        )

  if(MSVC AND (NOT ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")))
    # Enable parallel builds across four cores for this lib
    add_definitions(/MP4)
  endif()

  spvtools_pch(SPIRV_TOOLS_FUZZ_SOURCES pch_source_fuzz)

  add_library(SPIRV-Tools-fuzz ${SPIRV_TOOLS_FUZZ_SOURCES})

  spvtools_default_compile_options(SPIRV-Tools-fuzz)

  # Compilation of the auto-generated protobuf source file will yield warnings,
  # which we have no control over and thus wish to ignore.
  if(${COMPILER_IS_LIKE_GNU})
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/protobufs/spvtoolsfuzz.pb.cc PROPERTIES COMPILE_FLAGS -w)
  endif()
  if(MSVC)
    set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/protobufs/spvtoolsfuzz.pb.cc PROPERTIES COMPILE_FLAGS /w)
  endif()

  target_include_directories(SPIRV-Tools-fuzz
		PUBLIC
			$<BUILD_INTERFACE:${spirv-tools_SOURCE_DIR}/include>
			$<BUILD_INTERFACE:${SPIRV_HEADER_INCLUDE_DIR}>
			$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE ${spirv-tools_BINARY_DIR}
        PRIVATE ${CMAKE_BINARY_DIR})

  # The fuzzer reuses a lot of functionality from the SPIRV-Tools library.
  target_link_libraries(SPIRV-Tools-fuzz
        PUBLIC ${SPIRV_TOOLS_FULL_VISIBILITY}
        PUBLIC SPIRV-Tools-opt
        PUBLIC SPIRV-Tools-reduce
        PUBLIC protobuf::libprotobuf)

  set_property(TARGET SPIRV-Tools-fuzz PROPERTY FOLDER "SPIRV-Tools libraries")
  spvtools_check_symbol_exports(SPIRV-Tools-fuzz)

  if(ENABLE_SPIRV_TOOLS_INSTALL)
      install(TARGETS SPIRV-Tools-fuzz EXPORT SPIRV-Tools-fuzzTargets)
      export(EXPORT SPIRV-Tools-fuzzTargets FILE SPIRV-Tools-fuzzTarget.cmake)

      spvtools_config_package_dir(SPIRV-Tools-fuzz PACKAGE_DIR)
      install(EXPORT SPIRV-Tools-fuzzTargets FILE SPIRV-Tools-fuzzTarget.cmake
            DESTINATION ${PACKAGE_DIR})

      spvtools_generate_config_file(SPIRV-Tools-fuzz)
      install(FILES ${CMAKE_BINARY_DIR}/SPIRV-Tools-fuzzConfig.cmake DESTINATION ${PACKAGE_DIR})
  endif(ENABLE_SPIRV_TOOLS_INSTALL)

endif(SPIRV_BUILD_FUZZER)
