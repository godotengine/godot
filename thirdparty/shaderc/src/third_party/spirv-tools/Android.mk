LOCAL_PATH := $(call my-dir)
SPVTOOLS_OUT_PATH=$(if $(call host-path-is-absolute,$(TARGET_OUT)),$(TARGET_OUT),$(abspath $(TARGET_OUT)))

ifeq ($(SPVHEADERS_LOCAL_PATH),)
	SPVHEADERS_LOCAL_PATH := $(LOCAL_PATH)/external/spirv-headers
endif

SPVTOOLS_SRC_FILES := \
		source/assembly_grammar.cpp \
		source/binary.cpp \
		source/diagnostic.cpp \
		source/disassemble.cpp \
		source/ext_inst.cpp \
		source/enum_string_mapping.cpp \
		source/extensions.cpp \
		source/id_descriptor.cpp \
		source/libspirv.cpp \
		source/name_mapper.cpp \
		source/opcode.cpp \
		source/operand.cpp \
		source/parsed_operand.cpp \
		source/print.cpp \
		source/software_version.cpp \
		source/spirv_endian.cpp \
		source/spirv_optimizer_options.cpp \
		source/spirv_target_env.cpp \
		source/spirv_validator_options.cpp \
		source/table.cpp \
		source/text.cpp \
		source/text_handler.cpp \
		source/util/bit_vector.cpp \
		source/util/parse_number.cpp \
		source/util/string_utils.cpp \
		source/util/timer.cpp \
		source/val/basic_block.cpp \
		source/val/construct.cpp \
		source/val/function.cpp \
		source/val/instruction.cpp \
		source/val/validation_state.cpp \
		source/val/validate.cpp \
		source/val/validate_adjacency.cpp \
		source/val/validate_annotation.cpp \
		source/val/validate_arithmetics.cpp \
		source/val/validate_atomics.cpp \
		source/val/validate_barriers.cpp \
		source/val/validate_bitwise.cpp \
		source/val/validate_builtins.cpp \
		source/val/validate_capability.cpp \
		source/val/validate_cfg.cpp \
		source/val/validate_composites.cpp \
		source/val/validate_constants.cpp \
		source/val/validate_conversion.cpp \
		source/val/validate_datarules.cpp \
		source/val/validate_debug.cpp \
		source/val/validate_decorations.cpp \
		source/val/validate_derivatives.cpp \
		source/val/validate_extensions.cpp \
		source/val/validate_execution_limitations.cpp \
		source/val/validate_function.cpp \
		source/val/validate_id.cpp \
		source/val/validate_image.cpp \
		source/val/validate_interfaces.cpp \
		source/val/validate_instruction.cpp \
		source/val/validate_memory.cpp \
		source/val/validate_memory_semantics.cpp \
		source/val/validate_mode_setting.cpp \
		source/val/validate_layout.cpp \
		source/val/validate_literals.cpp \
		source/val/validate_logicals.cpp \
		source/val/validate_non_uniform.cpp \
		source/val/validate_primitives.cpp \
		source/val/validate_scopes.cpp \
		source/val/validate_type.cpp

SPVTOOLS_OPT_SRC_FILES := \
		source/opt/aggressive_dead_code_elim_pass.cpp \
		source/opt/basic_block.cpp \
		source/opt/block_merge_pass.cpp \
		source/opt/block_merge_util.cpp \
		source/opt/build_module.cpp \
		source/opt/cfg.cpp \
		source/opt/cfg_cleanup_pass.cpp \
		source/opt/ccp_pass.cpp \
		source/opt/code_sink.cpp \
		source/opt/combine_access_chains.cpp \
		source/opt/common_uniform_elim_pass.cpp \
		source/opt/compact_ids_pass.cpp \
		source/opt/composite.cpp \
		source/opt/const_folding_rules.cpp \
		source/opt/constants.cpp \
		source/opt/copy_prop_arrays.cpp \
		source/opt/dead_branch_elim_pass.cpp \
		source/opt/dead_insert_elim_pass.cpp \
		source/opt/dead_variable_elimination.cpp \
		source/opt/decoration_manager.cpp \
		source/opt/def_use_manager.cpp \
		source/opt/dominator_analysis.cpp \
		source/opt/dominator_tree.cpp \
		source/opt/eliminate_dead_constant_pass.cpp \
		source/opt/eliminate_dead_functions_pass.cpp \
		source/opt/eliminate_dead_functions_util.cpp \
		source/opt/eliminate_dead_members_pass.cpp \
		source/opt/feature_manager.cpp \
		source/opt/fix_storage_class.cpp \
		source/opt/flatten_decoration_pass.cpp \
		source/opt/fold.cpp \
		source/opt/folding_rules.cpp \
		source/opt/fold_spec_constant_op_and_composite_pass.cpp \
		source/opt/freeze_spec_constant_value_pass.cpp \
		source/opt/function.cpp \
		source/opt/generate_webgpu_initializers_pass.cpp \
		source/opt/if_conversion.cpp \
		source/opt/inline_pass.cpp \
		source/opt/inline_exhaustive_pass.cpp \
		source/opt/inline_opaque_pass.cpp \
		source/opt/inst_bindless_check_pass.cpp \
		source/opt/instruction.cpp \
		source/opt/instruction_list.cpp \
		source/opt/instrument_pass.cpp \
		source/opt/ir_context.cpp \
		source/opt/ir_loader.cpp \
		source/opt/licm_pass.cpp \
		source/opt/local_access_chain_convert_pass.cpp \
		source/opt/local_redundancy_elimination.cpp \
		source/opt/local_single_block_elim_pass.cpp \
		source/opt/local_single_store_elim_pass.cpp \
		source/opt/local_ssa_elim_pass.cpp \
		source/opt/loop_dependence.cpp \
		source/opt/loop_dependence_helpers.cpp \
		source/opt/loop_descriptor.cpp \
		source/opt/loop_fission.cpp \
		source/opt/loop_fusion.cpp \
		source/opt/loop_fusion_pass.cpp \
		source/opt/loop_peeling.cpp \
		source/opt/loop_unroller.cpp \
		source/opt/loop_unswitch_pass.cpp \
		source/opt/loop_utils.cpp \
		source/opt/mem_pass.cpp \
		source/opt/merge_return_pass.cpp \
		source/opt/module.cpp \
		source/opt/optimizer.cpp \
		source/opt/pass.cpp \
		source/opt/pass_manager.cpp \
		source/opt/private_to_local_pass.cpp \
		source/opt/process_lines_pass.cpp \
		source/opt/propagator.cpp \
		source/opt/reduce_load_size.cpp \
		source/opt/redundancy_elimination.cpp \
		source/opt/register_pressure.cpp \
		source/opt/remove_duplicates_pass.cpp \
		source/opt/replace_invalid_opc.cpp \
		source/opt/scalar_analysis.cpp \
		source/opt/scalar_analysis_simplification.cpp \
		source/opt/scalar_replacement_pass.cpp \
		source/opt/set_spec_constant_default_value_pass.cpp \
		source/opt/simplification_pass.cpp \
		source/opt/ssa_rewrite_pass.cpp \
		source/opt/strength_reduction_pass.cpp \
		source/opt/strip_atomic_counter_memory_pass.cpp \
		source/opt/strip_debug_info_pass.cpp \
		source/opt/strip_reflect_info_pass.cpp \
		source/opt/struct_cfg_analysis.cpp \
		source/opt/type_manager.cpp \
		source/opt/types.cpp \
		source/opt/unify_const_pass.cpp \
		source/opt/upgrade_memory_model.cpp \
		source/opt/value_number_table.cpp \
		source/opt/vector_dce.cpp \
		source/opt/workaround1209.cpp

# Locations of grammar files.
#
# TODO(dneto): Build a single set of tables that embeds versioning differences on
# a per-item basis.  That must happen before SPIR-V 1.4, etc.
# https://github.com/KhronosGroup/SPIRV-Tools/issues/1195
SPV_CORE10_GRAMMAR=$(SPVHEADERS_LOCAL_PATH)/include/spirv/1.0/spirv.core.grammar.json
SPV_CORE11_GRAMMAR=$(SPVHEADERS_LOCAL_PATH)/include/spirv/1.1/spirv.core.grammar.json
SPV_CORE12_GRAMMAR=$(SPVHEADERS_LOCAL_PATH)/include/spirv/1.2/spirv.core.grammar.json
SPV_COREUNIFIED1_GRAMMAR=$(SPVHEADERS_LOCAL_PATH)/include/spirv/unified1/spirv.core.grammar.json
SPV_CORELATEST_GRAMMAR=$(SPV_COREUNIFIED1_GRAMMAR)
SPV_GLSL_GRAMMAR=$(SPVHEADERS_LOCAL_PATH)/include/spirv/1.2/extinst.glsl.std.450.grammar.json
SPV_OPENCL_GRAMMAR=$(SPVHEADERS_LOCAL_PATH)/include/spirv/1.2/extinst.opencl.std.100.grammar.json
# TODO(dneto): I expect the DebugInfo grammar file to eventually migrate to SPIRV-Headers
SPV_DEBUGINFO_GRAMMAR=$(LOCAL_PATH)/source/extinst.debuginfo.grammar.json

define gen_spvtools_grammar_tables
$(call generate-file-dir,$(1)/core.insts-1.0.inc)
$(1)/core.insts-1.0.inc $(1)/operand.kinds-1.0.inc $(1)/glsl.std.450.insts.inc $(1)/opencl.std.insts.inc: \
        $(LOCAL_PATH)/utils/generate_grammar_tables.py \
        $(SPV_CORE10_GRAMMAR) \
        $(SPV_GLSL_GRAMMAR) \
        $(SPV_OPENCL_GRAMMAR) \
        $(SPV_DEBUGINFO_GRAMMAR)
		@$(HOST_PYTHON) $(LOCAL_PATH)/utils/generate_grammar_tables.py \
		                --spirv-core-grammar=$(SPV_CORE10_GRAMMAR) \
		                --extinst-glsl-grammar=$(SPV_GLSL_GRAMMAR) \
		                --extinst-opencl-grammar=$(SPV_OPENCL_GRAMMAR) \
		                --extinst-debuginfo-grammar=$(SPV_DEBUGINFO_GRAMMAR) \
		                --core-insts-output=$(1)/core.insts-1.0.inc \
		                --glsl-insts-output=$(1)/glsl.std.450.insts.inc \
		                --opencl-insts-output=$(1)/opencl.std.insts.inc \
		                --operand-kinds-output=$(1)/operand.kinds-1.0.inc
		@echo "[$(TARGET_ARCH_ABI)] Grammar v1.0   : instructions & operands <= grammar JSON files"
$(1)/core.insts-1.1.inc $(1)/operand.kinds-1.1.inc: \
        $(LOCAL_PATH)/utils/generate_grammar_tables.py \
        $(SPV_CORE11_GRAMMAR) \
        $(SPV_DEBUGINFO_GRAMMAR)
		@$(HOST_PYTHON) $(LOCAL_PATH)/utils/generate_grammar_tables.py \
		                --spirv-core-grammar=$(SPV_CORE11_GRAMMAR) \
		                --extinst-debuginfo-grammar=$(SPV_DEBUGINFO_GRAMMAR) \
		                --core-insts-output=$(1)/core.insts-1.1.inc \
		                --operand-kinds-output=$(1)/operand.kinds-1.1.inc
		@echo "[$(TARGET_ARCH_ABI)] Grammar v1.1   : instructions & operands <= grammar JSON files"
$(1)/core.insts-1.2.inc $(1)/operand.kinds-1.2.inc: \
        $(LOCAL_PATH)/utils/generate_grammar_tables.py \
        $(SPV_CORE12_GRAMMAR) \
        $(SPV_DEBUGINFO_GRAMMAR)
		@$(HOST_PYTHON) $(LOCAL_PATH)/utils/generate_grammar_tables.py \
		                --spirv-core-grammar=$(SPV_CORE12_GRAMMAR) \
		                --extinst-debuginfo-grammar=$(SPV_DEBUGINFO_GRAMMAR) \
		                --core-insts-output=$(1)/core.insts-1.2.inc \
		                --operand-kinds-output=$(1)/operand.kinds-1.2.inc
		@echo "[$(TARGET_ARCH_ABI)] Grammar v1.2   : instructions & operands <= grammar JSON files"
$(1)/core.insts-unified1.inc $(1)/operand.kinds-unified1.inc: \
        $(LOCAL_PATH)/utils/generate_grammar_tables.py \
        $(SPV_COREUNIFIED1_GRAMMAR) \
        $(SPV_DEBUGINFO_GRAMMAR)
		@$(HOST_PYTHON) $(LOCAL_PATH)/utils/generate_grammar_tables.py \
		                --spirv-core-grammar=$(SPV_COREUNIFIED1_GRAMMAR) \
		                --extinst-debuginfo-grammar=$(SPV_DEBUGINFO_GRAMMAR) \
		                --core-insts-output=$(1)/core.insts-unified1.inc \
		                --operand-kinds-output=$(1)/operand.kinds-unified1.inc
		@echo "[$(TARGET_ARCH_ABI)] Grammar v1.3 (from unified1)  : instructions & operands <= grammar JSON files"
$(LOCAL_PATH)/source/opcode.cpp: $(1)/core.insts-1.0.inc $(1)/core.insts-1.1.inc $(1)/core.insts-1.2.inc $(1)/core.insts-unified1.inc
$(LOCAL_PATH)/source/operand.cpp: $(1)/operand.kinds-1.0.inc $(1)/operand.kinds-1.1.inc $(1)/operand.kinds-1.2.inc $(1)/operand.kinds-unified1.inc
$(LOCAL_PATH)/source/ext_inst.cpp: \
	$(1)/glsl.std.450.insts.inc \
	$(1)/opencl.std.insts.inc \
	$(1)/debuginfo.insts.inc \
	$(1)/spv-amd-gcn-shader.insts.inc \
	$(1)/spv-amd-shader-ballot.insts.inc \
	$(1)/spv-amd-shader-explicit-vertex-parameter.insts.inc \
	$(1)/spv-amd-shader-trinary-minmax.insts.inc
endef
$(eval $(call gen_spvtools_grammar_tables,$(SPVTOOLS_OUT_PATH)))


define gen_spvtools_lang_headers
# Generate language-specific headers.  So far we only generate C headers
# $1 is the output directory.
# $2 is the base name of the header file, e.g. "DebugInfo".
# $3 is the grammar file containing token definitions.
$(call generate-file-dir,$(1)/$(2).h)
$(1)/$(2).h : \
        $(LOCAL_PATH)/utils/generate_language_headers.py \
        $(3)
		@$(HOST_PYTHON) $(LOCAL_PATH)/utils/generate_language_headers.py \
		    --extinst-name=$(2) \
		    --extinst-grammar=$(3) \
		    --extinst-output-base=$(1)/$(2)
		@echo "[$(TARGET_ARCH_ABI)] Generate language specific header for $(2): headers <= grammar"
$(LOCAL_PATH)/source/ext_inst.cpp: $(1)/$(2).h
endef
# We generate language-specific headers for DebugInfo
$(eval $(call gen_spvtools_lang_headers,$(SPVTOOLS_OUT_PATH),DebugInfo,$(SPV_DEBUGINFO_GRAMMAR)))


define gen_spvtools_vendor_tables
$(call generate-file-dir,$(1)/$(2).insts.inc)
$(1)/$(2).insts.inc : \
        $(LOCAL_PATH)/utils/generate_grammar_tables.py \
        $(LOCAL_PATH)/source/extinst.$(2).grammar.json
		@$(HOST_PYTHON) $(LOCAL_PATH)/utils/generate_grammar_tables.py \
		    --extinst-vendor-grammar=$(LOCAL_PATH)/source/extinst.$(2).grammar.json \
		    --vendor-insts-output=$(1)/$(2).insts.inc
		@echo "[$(TARGET_ARCH_ABI)] Vendor extended instruction set: $(2) tables <= grammar"
$(LOCAL_PATH)/source/ext_inst.cpp: $(1)/$(2).insts.inc
endef
# Vendor extended instruction sets, with grammars from SPIRV-Tools source tree.
SPV_NONSTANDARD_EXTINST_GRAMMARS=$(foreach F,$(wildcard $(LOCAL_PATH)/source/extinst.*.grammar.json),$(patsubst extinst.%.grammar.json,%,$(notdir $F)))
$(foreach E,$(SPV_NONSTANDARD_EXTINST_GRAMMARS),$(eval $(call gen_spvtools_vendor_tables,$(SPVTOOLS_OUT_PATH),$E)))

define gen_spvtools_enum_string_mapping
$(call generate-file-dir,$(1)/extension_enum.inc.inc)
$(1)/extension_enum.inc $(1)/enum_string_mapping.inc: \
        $(LOCAL_PATH)/utils/generate_grammar_tables.py \
        $(SPV_CORELATEST_GRAMMAR)
		@$(HOST_PYTHON) $(LOCAL_PATH)/utils/generate_grammar_tables.py \
		                --spirv-core-grammar=$(SPV_CORELATEST_GRAMMAR) \
		                --extinst-debuginfo-grammar=$(SPV_DEBUGINFO_GRAMMAR) \
		                --extension-enum-output=$(1)/extension_enum.inc \
		                --enum-string-mapping-output=$(1)/enum_string_mapping.inc
		@echo "[$(TARGET_ARCH_ABI)] Generate enum<->string mapping <= grammar JSON files"
# Generated header extension_enum.inc is transitively included by table.h, which is
# used pervasively.  Capture the pervasive dependency.
$(foreach F,$(SPVTOOLS_SRC_FILES) $(SPVTOOLS_OPT_SRC_FILES),$(LOCAL_PATH)/$F ) \
  : $(1)/extension_enum.inc
$(LOCAL_PATH)/source/enum_string_mapping.cpp: $(1)/enum_string_mapping.inc
endef
$(eval $(call gen_spvtools_enum_string_mapping,$(SPVTOOLS_OUT_PATH)))

define gen_spvtools_build_version_inc
$(call generate-file-dir,$(1)/dummy_filename)
$(1)/build-version.inc: \
        $(LOCAL_PATH)/utils/update_build_version.py \
        $(LOCAL_PATH)/CHANGES
		@$(HOST_PYTHON) $(LOCAL_PATH)/utils/update_build_version.py \
		                $(LOCAL_PATH) $(1)/build-version.inc
		@echo "[$(TARGET_ARCH_ABI)] Generate       : build-version.inc <= CHANGES"
$(LOCAL_PATH)/source/software_version.cpp: $(1)/build-version.inc
endef
$(eval $(call gen_spvtools_build_version_inc,$(SPVTOOLS_OUT_PATH)))

define gen_spvtools_generators_inc
$(call generate-file-dir,$(1)/dummy_filename)
$(1)/generators.inc: \
        $(LOCAL_PATH)/utils/generate_registry_tables.py \
        $(SPVHEADERS_LOCAL_PATH)/include/spirv/spir-v.xml
		@$(HOST_PYTHON) $(LOCAL_PATH)/utils/generate_registry_tables.py \
		                --xml=$(SPVHEADERS_LOCAL_PATH)/include/spirv/spir-v.xml \
				--generator-output=$(1)/generators.inc
		@echo "[$(TARGET_ARCH_ABI)] Generate       : generators.inc <= spir-v.xml"
$(LOCAL_PATH)/source/opcode.cpp: $(1)/generators.inc
endef
$(eval $(call gen_spvtools_generators_inc,$(SPVTOOLS_OUT_PATH)))

include $(CLEAR_VARS)
LOCAL_MODULE := SPIRV-Tools
LOCAL_C_INCLUDES := \
		$(LOCAL_PATH)/include \
		$(SPVHEADERS_LOCAL_PATH)/include \
		$(SPVTOOLS_OUT_PATH)
LOCAL_EXPORT_C_INCLUDES := \
		$(LOCAL_PATH)/include
LOCAL_CXXFLAGS:=-std=c++11 -fno-exceptions -fno-rtti -Werror
LOCAL_SRC_FILES:= $(SPVTOOLS_SRC_FILES)
include $(BUILD_STATIC_LIBRARY)

include $(CLEAR_VARS)
LOCAL_MODULE := SPIRV-Tools-opt
LOCAL_C_INCLUDES := \
		$(LOCAL_PATH)/include \
		$(LOCAL_PATH)/source \
		$(SPVHEADERS_LOCAL_PATH)/include \
		$(SPVTOOLS_OUT_PATH)
LOCAL_CXXFLAGS:=-std=c++11 -fno-exceptions -fno-rtti -Werror
LOCAL_STATIC_LIBRARIES:=SPIRV-Tools
LOCAL_SRC_FILES:= $(SPVTOOLS_OPT_SRC_FILES)
include $(BUILD_STATIC_LIBRARY)
