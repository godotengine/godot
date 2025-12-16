/*
 * Copyright 2015-2021 Arm Limited
 * SPDX-License-Identifier: Apache-2.0 OR MIT
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * At your option, you may choose to accept this material under either:
 *  1. The Apache License, Version 2.0, found at <http://www.apache.org/licenses/LICENSE-2.0>, or
 *  2. The MIT License, found at <http://opensource.org/licenses/MIT>.
 */

#ifndef SPIRV_CROSS_HPP
#define SPIRV_CROSS_HPP

#ifndef SPV_ENABLE_UTILITY_CODE
#define SPV_ENABLE_UTILITY_CODE
#endif

// Pragmatic hack to avoid symbol conflicts when including both hpp11 and hpp headers in same translation unit.
// This is an unfortunate SPIRV-Headers issue that we cannot easily deal with ourselves.
#ifdef SPIRV_CROSS_SPV_HEADER_NAMESPACE_OVERRIDE
#define spv SPIRV_CROSS_SPV_HEADER_NAMESPACE_OVERRIDE
#endif

#include "spirv.hpp"
#include "spirv_cfg.hpp"
#include "spirv_cross_parsed_ir.hpp"

namespace SPIRV_CROSS_NAMESPACE
{
using namespace SPIRV_CROSS_SPV_HEADER_NAMESPACE;
struct Resource
{
	// Resources are identified with their SPIR-V ID.
	// This is the ID of the OpVariable.
	ID id;

	// The type ID of the variable which includes arrays and all type modifications.
	// This type ID is not suitable for parsing OpMemberDecoration of a struct and other decorations in general
	// since these modifications typically happen on the base_type_id.
	TypeID type_id;

	// The base type of the declared resource.
	// This type is the base type which ignores pointers and arrays of the type_id.
	// This is mostly useful to parse decorations of the underlying type.
	// base_type_id can also be obtained with get_type(get_type(type_id).self).
	TypeID base_type_id;

	// The declared name (OpName) of the resource.
	// For Buffer blocks, the name actually reflects the externally
	// visible Block name.
	//
	// This name can be retrieved again by using either
	// get_name(id) or get_name(base_type_id) depending if it's a buffer block or not.
	//
	// This name can be an empty string in which case get_fallback_name(id) can be
	// used which obtains a suitable fallback identifier for an ID.
	std::string name;
};

struct BuiltInResource
{
	// This is mostly here to support reflection of builtins such as Position/PointSize/CullDistance/ClipDistance.
	// This needs to be different from Resource since we can collect builtins from blocks.
	// A builtin present here does not necessarily mean it's considered an active builtin,
	// since variable ID "activeness" is only tracked on OpVariable level, not Block members.
	// For that, update_active_builtins() -> has_active_builtin() can be used to further refine the reflection.
	BuiltIn builtin;

	// This is the actual value type of the builtin.
	// Typically float4, float, array<float, N> for the gl_PerVertex builtins.
	// If the builtin is a control point, the control point array type will be stripped away here as appropriate.
	TypeID value_type_id;

	// This refers to the base resource which contains the builtin.
	// If resource is a Block, it can hold multiple builtins, or it might not be a block.
	// For advanced reflection scenarios, all information in builtin/value_type_id can be deduced,
	// it's just more convenient this way.
	Resource resource;
};

struct ShaderResources
{
	SmallVector<Resource> uniform_buffers;
	SmallVector<Resource> storage_buffers;
	SmallVector<Resource> stage_inputs;
	SmallVector<Resource> stage_outputs;
	SmallVector<Resource> subpass_inputs;
	SmallVector<Resource> storage_images;
	SmallVector<Resource> sampled_images;
	SmallVector<Resource> atomic_counters;
	SmallVector<Resource> acceleration_structures;
	SmallVector<Resource> gl_plain_uniforms;
	SmallVector<Resource> tensors;

	// There can only be one push constant block,
	// but keep the vector in case this restriction is lifted in the future.
	SmallVector<Resource> push_constant_buffers;

	SmallVector<Resource> shader_record_buffers;

	// For Vulkan GLSL and HLSL source,
	// these correspond to separate texture2D and samplers respectively.
	SmallVector<Resource> separate_images;
	SmallVector<Resource> separate_samplers;

	SmallVector<BuiltInResource> builtin_inputs;
	SmallVector<BuiltInResource> builtin_outputs;
};

struct CombinedImageSampler
{
	// The ID of the sampler2D variable.
	VariableID combined_id;
	// The ID of the texture2D variable.
	VariableID image_id;
	// The ID of the sampler variable.
	VariableID sampler_id;
};

struct SpecializationConstant
{
	// The ID of the specialization constant.
	ConstantID id;
	// The constant ID of the constant, used in Vulkan during pipeline creation.
	uint32_t constant_id;
};

struct BufferRange
{
	unsigned index;
	size_t offset;
	size_t range;
};

enum BufferPackingStandard
{
	BufferPackingStd140,
	BufferPackingStd430,
	BufferPackingStd140EnhancedLayout,
	BufferPackingStd430EnhancedLayout,
	BufferPackingHLSLCbuffer,
	BufferPackingHLSLCbufferPackOffset,
	BufferPackingScalar,
	BufferPackingScalarEnhancedLayout
};

struct EntryPoint
{
	std::string name;
	ExecutionModel execution_model;
};

class Compiler
{
public:
	friend class CFG;
	friend class DominatorBuilder;

	// The constructor takes a buffer of SPIR-V words and parses it.
	// It will create its own parser, parse the SPIR-V and move the parsed IR
	// as if you had called the constructors taking ParsedIR directly.
	explicit Compiler(std::vector<uint32_t> ir);
	Compiler(const uint32_t *ir, size_t word_count);

	// This is more modular. We can also consume a ParsedIR structure directly, either as a move, or copy.
	// With copy, we can reuse the same parsed IR for multiple Compiler instances.
	explicit Compiler(const ParsedIR &ir);
	explicit Compiler(ParsedIR &&ir);

	virtual ~Compiler() = default;

	// After parsing, API users can modify the SPIR-V via reflection and call this
	// to disassemble the SPIR-V into the desired langauage.
	// Sub-classes actually implement this.
	virtual std::string compile();

	// Gets the identifier (OpName) of an ID. If not defined, an empty string will be returned.
	const std::string &get_name(ID id) const;

	// Applies a decoration to an ID. Effectively injects OpDecorate.
	void set_decoration(ID id, Decoration decoration, uint32_t argument = 0);
	void set_decoration_string(ID id, Decoration decoration, const std::string &argument);

	// Overrides the identifier OpName of an ID.
	// Identifiers beginning with underscores or identifiers which contain double underscores
	// are reserved by the implementation.
	void set_name(ID id, const std::string &name);

	// Gets a bitmask for the decorations which are applied to ID.
	// I.e. (1ull << DecorationFoo) | (1ull << DecorationBar)
	const Bitset &get_decoration_bitset(ID id) const;

	// Returns whether the decoration has been applied to the ID.
	bool has_decoration(ID id, Decoration decoration) const;

	// Gets the value for decorations which take arguments.
	// If the decoration is a boolean (i.e. DecorationNonWritable),
	// 1 will be returned.
	// If decoration doesn't exist or decoration is not recognized,
	// 0 will be returned.
	uint32_t get_decoration(ID id, Decoration decoration) const;
	const std::string &get_decoration_string(ID id, Decoration decoration) const;

	// Removes the decoration for an ID.
	void unset_decoration(ID id, Decoration decoration);

	// Gets the SPIR-V type associated with ID.
	// Mostly used with Resource::type_id and Resource::base_type_id to parse the underlying type of a resource.
	const SPIRType &get_type(TypeID id) const;

	// Gets the SPIR-V type of a variable.
	const SPIRType &get_type_from_variable(VariableID id) const;

	// Gets the underlying storage class for an OpVariable.
	StorageClass get_storage_class(VariableID id) const;

	// If get_name() is an empty string, get the fallback name which will be used
	// instead in the disassembled source.
	virtual const std::string get_fallback_name(ID id) const;

	// If get_name() of a Block struct is an empty string, get the fallback name.
	// This needs to be per-variable as multiple variables can use the same block type.
	virtual const std::string get_block_fallback_name(VariableID id) const;

	// Given an OpTypeStruct in ID, obtain the identifier for member number "index".
	// This may be an empty string.
	const std::string &get_member_name(TypeID id, uint32_t index) const;

	// Given an OpTypeStruct in ID, obtain the OpMemberDecoration for member number "index".
	uint32_t get_member_decoration(TypeID id, uint32_t index, Decoration decoration) const;
	const std::string &get_member_decoration_string(TypeID id, uint32_t index, Decoration decoration) const;

	// Sets the member identifier for OpTypeStruct ID, member number "index".
	void set_member_name(TypeID id, uint32_t index, const std::string &name);

	// Returns the qualified member identifier for OpTypeStruct ID, member number "index",
	// or an empty string if no qualified alias exists
	const std::string &get_member_qualified_name(TypeID type_id, uint32_t index) const;

	// Gets the decoration mask for a member of a struct, similar to get_decoration_mask.
	const Bitset &get_member_decoration_bitset(TypeID id, uint32_t index) const;

	// Returns whether the decoration has been applied to a member of a struct.
	bool has_member_decoration(TypeID id, uint32_t index, Decoration decoration) const;

	// Similar to set_decoration, but for struct members.
	void set_member_decoration(TypeID id, uint32_t index, Decoration decoration, uint32_t argument = 0);
	void set_member_decoration_string(TypeID id, uint32_t index, Decoration decoration,
	                                  const std::string &argument);

	// Unsets a member decoration, similar to unset_decoration.
	void unset_member_decoration(TypeID id, uint32_t index, Decoration decoration);

	// Gets the fallback name for a member, similar to get_fallback_name.
	virtual const std::string get_fallback_member_name(uint32_t index) const
	{
		return join("_", index);
	}

	// Returns a vector of which members of a struct are potentially in use by a
	// SPIR-V shader. The granularity of this analysis is per-member of a struct.
	// This can be used for Buffer (UBO), BufferBlock/StorageBuffer (SSBO) and PushConstant blocks.
	// ID is the Resource::id obtained from get_shader_resources().
	SmallVector<BufferRange> get_active_buffer_ranges(VariableID id) const;

	// Returns the effective size of a buffer block.
	size_t get_declared_struct_size(const SPIRType &struct_type) const;

	// Returns the effective size of a buffer block, with a given array size
	// for a runtime array.
	// SSBOs are typically declared as runtime arrays. get_declared_struct_size() will return 0 for the size.
	// This is not very helpful for applications which might need to know the array stride of its last member.
	// This can be done through the API, but it is not very intuitive how to accomplish this, so here we provide a helper function
	// to query the size of the buffer, assuming that the last member has a certain size.
	// If the buffer does not contain a runtime array, array_size is ignored, and the function will behave as
	// get_declared_struct_size().
	// To get the array stride of the last member, something like:
	// get_declared_struct_size_runtime_array(type, 1) - get_declared_struct_size_runtime_array(type, 0) will work.
	size_t get_declared_struct_size_runtime_array(const SPIRType &struct_type, size_t array_size) const;

	// Returns the effective size of a buffer block struct member.
	size_t get_declared_struct_member_size(const SPIRType &struct_type, uint32_t index) const;

	// Returns a set of all global variables which are statically accessed
	// by the control flow graph from the current entry point.
	// Only variables which change the interface for a shader are returned, that is,
	// variables with storage class of Input, Output, Uniform, UniformConstant, PushConstant and AtomicCounter
	// storage classes are returned.
	//
	// To use the returned set as the filter for which variables are used during compilation,
	// this set can be moved to set_enabled_interface_variables().
	std::unordered_set<VariableID> get_active_interface_variables() const;

	// Sets the interface variables which are used during compilation.
	// By default, all variables are used.
	// Once set, compile() will only consider the set in active_variables.
	void set_enabled_interface_variables(std::unordered_set<VariableID> active_variables);

	// Query shader resources, use ids with reflection interface to modify or query binding points, etc.
	ShaderResources get_shader_resources() const;

	// Query shader resources, but only return the variables which are part of active_variables.
	// E.g.: get_shader_resources(get_active_variables()) to only return the variables which are statically
	// accessed.
	ShaderResources get_shader_resources(const std::unordered_set<VariableID> &active_variables) const;

	// Remapped variables are considered built-in variables and a backend will
	// not emit a declaration for this variable.
	// This is mostly useful for making use of builtins which are dependent on extensions.
	void set_remapped_variable_state(VariableID id, bool remap_enable);
	bool get_remapped_variable_state(VariableID id) const;

	// For subpassInput variables which are remapped to plain variables,
	// the number of components in the remapped
	// variable must be specified as the backing type of subpass inputs are opaque.
	void set_subpass_input_remapped_components(VariableID id, uint32_t components);
	uint32_t get_subpass_input_remapped_components(VariableID id) const;

	// All operations work on the current entry point.
	// Entry points can be swapped out with set_entry_point().
	// Entry points should be set right after the constructor completes as some reflection functions traverse the graph from the entry point.
	// Resource reflection also depends on the entry point.
	// By default, the current entry point is set to the first OpEntryPoint which appears in the SPIR-V module.

	// Some shader languages restrict the names that can be given to entry points, and the
	// corresponding backend will automatically rename an entry point name, during the call
	// to compile() if it is illegal. For example, the common entry point name main() is
	// illegal in MSL, and is renamed to an alternate name by the MSL backend.
	// Given the original entry point name contained in the SPIR-V, this function returns
	// the name, as updated by the backend during the call to compile(). If the name is not
	// illegal, and has not been renamed, or if this function is called before compile(),
	// this function will simply return the same name.

	// New variants of entry point query and reflection.
	// Names for entry points in the SPIR-V module may alias if they belong to different execution models.
	// To disambiguate, we must pass along with the entry point names the execution model.
	SmallVector<EntryPoint> get_entry_points_and_stages() const;
	void set_entry_point(const std::string &entry, ExecutionModel execution_model);

	// Renames an entry point from old_name to new_name.
	// If old_name is currently selected as the current entry point, it will continue to be the current entry point,
	// albeit with a new name.
	// get_entry_points() is essentially invalidated at this point.
	void rename_entry_point(const std::string &old_name, const std::string &new_name,
	                        ExecutionModel execution_model);
	const SPIREntryPoint &get_entry_point(const std::string &name, ExecutionModel execution_model) const;
	SPIREntryPoint &get_entry_point(const std::string &name, ExecutionModel execution_model);
	const std::string &get_cleansed_entry_point_name(const std::string &name,
	                                                 ExecutionModel execution_model) const;

	// Traverses all reachable opcodes and sets active_builtins to a bitmask of all builtin variables which are accessed in the shader.
	void update_active_builtins();
	bool has_active_builtin(BuiltIn builtin, StorageClass storage) const;

	// Query and modify OpExecutionMode.
	const Bitset &get_execution_mode_bitset() const;

	void unset_execution_mode(ExecutionMode mode);
	void set_execution_mode(ExecutionMode mode, uint32_t arg0 = 0, uint32_t arg1 = 0, uint32_t arg2 = 0);

	// Gets argument for an execution mode (LocalSize, Invocations, OutputVertices).
	// For LocalSize or LocalSizeId, the index argument is used to select the dimension (X = 0, Y = 1, Z = 2).
	// For execution modes which do not have arguments, 0 is returned.
	// LocalSizeId query returns an ID. If LocalSizeId execution mode is not used, it returns 0.
	// LocalSize always returns a literal. If execution mode is LocalSizeId,
	// the literal (spec constant or not) is still returned.
	uint32_t get_execution_mode_argument(ExecutionMode mode, uint32_t index = 0) const;
	ExecutionModel get_execution_model() const;

	bool is_tessellation_shader() const;
	bool is_tessellating_triangles() const;

	// In SPIR-V, the compute work group size can be represented by a constant vector, in which case
	// the LocalSize execution mode is ignored.
	//
	// This constant vector can be a constant vector, specialization constant vector, or partly specialized constant vector.
	// To modify and query work group dimensions which are specialization constants, SPIRConstant values must be modified
	// directly via get_constant() rather than using LocalSize directly. This function will return which constants should be modified.
	//
	// To modify dimensions which are *not* specialization constants, set_execution_mode should be used directly.
	// Arguments to set_execution_mode which are specialization constants are effectively ignored during compilation.
	// NOTE: This is somewhat different from how SPIR-V works. In SPIR-V, the constant vector will completely replace LocalSize,
	// while in this interface, LocalSize is only ignored for specialization constants.
	//
	// The specialization constant will be written to x, y and z arguments.
	// If the component is not a specialization constant, a zeroed out struct will be written.
	// The return value is the constant ID of the builtin WorkGroupSize, but this is not expected to be useful
	// for most use cases.
	// If LocalSizeId is used, there is no uvec3 value representing the workgroup size, so the return value is 0,
	// but x, y and z are written as normal if the components are specialization constants.
	uint32_t get_work_group_size_specialization_constants(SpecializationConstant &x, SpecializationConstant &y,
	                                                      SpecializationConstant &z) const;

	// Analyzes all OpImageFetch (texelFetch) opcodes and checks if there are instances where
	// said instruction is used without a combined image sampler.
	// GLSL targets do not support the use of texelFetch without a sampler.
	// To workaround this, we must inject a dummy sampler which can be used to form a sampler2D at the call-site of
	// texelFetch as necessary.
	//
	// This must be called before build_combined_image_samplers().
	// build_combined_image_samplers() may refer to the ID returned by this method if the returned ID is non-zero.
	// The return value will be the ID of a sampler object if a dummy sampler is necessary, or 0 if no sampler object
	// is required.
	//
	// If the returned ID is non-zero, it can be decorated with set/bindings as desired before calling compile().
	// Calling this function also invalidates get_active_interface_variables(), so this should be called
	// before that function.
	VariableID build_dummy_sampler_for_combined_images();

	// Analyzes all separate image and samplers used from the currently selected entry point,
	// and re-routes them all to a combined image sampler instead.
	// This is required to "support" separate image samplers in targets which do not natively support
	// this feature, like GLSL/ESSL.
	//
	// This must be called before compile() if such remapping is desired.
	// This call will add new sampled images to the SPIR-V,
	// so it will appear in reflection if get_shader_resources() is called after build_combined_image_samplers.
	//
	// If any image/sampler remapping was found, no separate image/samplers will appear in the decompiled output,
	// but will still appear in reflection.
	//
	// The resulting samplers will be void of any decorations like name, descriptor sets and binding points,
	// so this can be added before compile() if desired.
	//
	// Combined image samplers originating from this set are always considered active variables.
	// Arrays of separate samplers are not supported, but arrays of separate images are supported.
	// Array of images + sampler -> Array of combined image samplers.
	void build_combined_image_samplers();

	// Gets a remapping for the combined image samplers.
	const SmallVector<CombinedImageSampler> &get_combined_image_samplers() const
	{
		return combined_image_samplers;
	}

	// Set a new variable type remap callback.
	// The type remapping is designed to allow global interface variable to assume more special types.
	// A typical example here is to remap sampler2D into samplerExternalOES, which currently isn't supported
	// directly by SPIR-V.
	//
	// In compile() while emitting code,
	// for every variable that is declared, including function parameters, the callback will be called
	// and the API user has a chance to change the textual representation of the type used to declare the variable.
	// The API user can detect special patterns in names to guide the remapping.
	void set_variable_type_remap_callback(VariableTypeRemapCallback cb)
	{
		variable_remap_callback = std::move(cb);
	}

	// API for querying which specialization constants exist.
	// To modify a specialization constant before compile(), use get_constant(constant.id),
	// then update constants directly in the SPIRConstant data structure.
	// For composite types, the subconstants can be iterated over and modified.
	// constant_type is the SPIRType for the specialization constant,
	// which can be queried to determine which fields in the unions should be poked at.
	SmallVector<SpecializationConstant> get_specialization_constants() const;
	SPIRConstant &get_constant(ConstantID id);
	const SPIRConstant &get_constant(ConstantID id) const;

	uint32_t get_current_id_bound() const
	{
		return uint32_t(ir.ids.size());
	}

	// API for querying buffer objects.
	// The type passed in here should be the base type of a resource, i.e.
	// get_type(resource.base_type_id)
	// as decorations are set in the basic Block type.
	// The type passed in here must have these decorations set, or an exception is raised.
	// Only UBOs and SSBOs or sub-structs which are part of these buffer types will have these decorations set.
	uint32_t type_struct_member_offset(const SPIRType &type, uint32_t index) const;
	uint32_t type_struct_member_array_stride(const SPIRType &type, uint32_t index) const;
	uint32_t type_struct_member_matrix_stride(const SPIRType &type, uint32_t index) const;

	// Gets the offset in SPIR-V words (uint32_t) for a decoration which was originally declared in the SPIR-V binary.
	// The offset will point to one or more uint32_t literals which can be modified in-place before using the SPIR-V binary.
	// Note that adding or removing decorations using the reflection API will not change the behavior of this function.
	// If the decoration was declared, sets the word_offset to an offset into the provided SPIR-V binary buffer and returns true,
	// otherwise, returns false.
	// If the decoration does not have any value attached to it (e.g. DecorationRelaxedPrecision), this function will also return false.
	bool get_binary_offset_for_decoration(VariableID id, Decoration decoration, uint32_t &word_offset) const;

	// HLSL counter buffer reflection interface.
	// Append/Consume/Increment/Decrement in HLSL is implemented as two "neighbor" buffer objects where
	// one buffer implements the storage, and a single buffer containing just a lone "int" implements the counter.
	// To SPIR-V these will be exposed as two separate buffers, but glslang HLSL frontend emits a special indentifier
	// which lets us link the two buffers together.

	// Queries if a variable ID is a counter buffer which "belongs" to a regular buffer object.

	// If SPV_GOOGLE_hlsl_functionality1 is used, this can be used even with a stripped SPIR-V module.
	// Otherwise, this query is purely based on OpName identifiers as found in the SPIR-V module, and will
	// only return true if OpSource was reported HLSL.
	// To rely on this functionality, ensure that the SPIR-V module is not stripped.

	bool buffer_is_hlsl_counter_buffer(VariableID id) const;

	// Queries if a buffer object has a neighbor "counter" buffer.
	// If so, the ID of that counter buffer will be returned in counter_id.
	// If SPV_GOOGLE_hlsl_functionality1 is used, this can be used even with a stripped SPIR-V module.
	// Otherwise, this query is purely based on OpName identifiers as found in the SPIR-V module, and will
	// only return true if OpSource was reported HLSL.
	// To rely on this functionality, ensure that the SPIR-V module is not stripped.
	bool buffer_get_hlsl_counter_buffer(VariableID id, uint32_t &counter_id) const;

	// Gets the list of all SPIR-V Capabilities which were declared in the SPIR-V module.
	const SmallVector<Capability> &get_declared_capabilities() const;

	// Gets the list of all SPIR-V extensions which were declared in the SPIR-V module.
	const SmallVector<std::string> &get_declared_extensions() const;

	// When declaring buffer blocks in GLSL, the name declared in the GLSL source
	// might not be the same as the name declared in the SPIR-V module due to naming conflicts.
	// In this case, SPIRV-Cross needs to find a fallback-name, and it might only
	// be possible to know this name after compiling to GLSL.
	// This is particularly important for HLSL input and UAVs which tends to reuse the same block type
	// for multiple distinct blocks. For these cases it is not possible to modify the name of the type itself
	// because it might be unique. Instead, you can use this interface to check after compilation which
	// name was actually used if your input SPIR-V tends to have this problem.
	// For other names like remapped names for variables, etc, it's generally enough to query the name of the variables
	// after compiling, block names are an exception to this rule.
	// ID is the name of a variable as returned by Resource::id, and must be a variable with a Block-like type.
	//
	// This also applies to HLSL cbuffers.
	std::string get_remapped_declared_block_name(VariableID id) const;

	// For buffer block variables, get the decorations for that variable.
	// Sometimes, decorations for buffer blocks are found in member decorations instead
	// of direct decorations on the variable itself.
	// The most common use here is to check if a buffer is readonly or writeonly.
	Bitset get_buffer_block_flags(VariableID id) const;

	// Returns whether the position output is invariant
	bool is_position_invariant() const
	{
		return position_invariant;
	}

protected:
	const uint32_t *stream(const Instruction &instr) const
	{
		// If we're not going to use any arguments, just return nullptr.
		// We want to avoid case where we return an out of range pointer
		// that trips debug assertions on some platforms.
		if (!instr.length)
			return nullptr;

		if (instr.is_embedded())
		{
			auto &embedded = static_cast<const EmbeddedInstruction &>(instr);
			assert(embedded.ops.size() == instr.length);
			return embedded.ops.data();
		}
		else
		{
			if (instr.offset + instr.length > ir.spirv.size())
				SPIRV_CROSS_THROW("Compiler::stream() out of range.");
			return &ir.spirv[instr.offset];
		}
	}

	uint32_t *stream_mutable(const Instruction &instr) const
	{
		return const_cast<uint32_t *>(stream(instr));
	}

	ParsedIR ir;
	// Marks variables which have global scope and variables which can alias with other variables
	// (SSBO, image load store, etc)
	SmallVector<uint32_t> global_variables;
	SmallVector<uint32_t> aliased_variables;

	SPIRFunction *current_function = nullptr;
	SPIRBlock *current_block = nullptr;
	uint32_t current_loop_level = 0;
	std::unordered_set<VariableID> active_interface_variables;
	bool check_active_interface_variables = false;

	void add_loop_level();

	void set_initializers(SPIRExpression &e)
	{
		e.emitted_loop_level = current_loop_level;
	}

	template <typename T>
	void set_initializers(const T &)
	{
	}

	// If our IDs are out of range here as part of opcodes, throw instead of
	// undefined behavior.
	template <typename T, typename... P>
	T &set(uint32_t id, P &&... args)
	{
		ir.add_typed_id(static_cast<Types>(T::type), id);
		auto &var = variant_set<T>(ir.ids[id], std::forward<P>(args)...);
		var.self = id;
		set_initializers(var);
		return var;
	}

	template <typename T>
	T &get(uint32_t id)
	{
		return variant_get<T>(ir.ids[id]);
	}

	template <typename T>
	T *maybe_get(uint32_t id)
	{
		if (id >= ir.ids.size())
			return nullptr;
		else if (ir.ids[id].get_type() == static_cast<Types>(T::type))
			return &get<T>(id);
		else
			return nullptr;
	}

	template <typename T>
	const T &get(uint32_t id) const
	{
		return variant_get<T>(ir.ids[id]);
	}

	template <typename T>
	const T *maybe_get(uint32_t id) const
	{
		if (id >= ir.ids.size())
			return nullptr;
		else if (ir.ids[id].get_type() == static_cast<Types>(T::type))
			return &get<T>(id);
		else
			return nullptr;
	}

	// Gets the id of SPIR-V type underlying the given type_id, which might be a pointer.
	uint32_t get_pointee_type_id(uint32_t type_id) const;

	// Gets the SPIR-V type underlying the given type, which might be a pointer.
	const SPIRType &get_pointee_type(const SPIRType &type) const;

	// Gets the SPIR-V type underlying the given type_id, which might be a pointer.
	const SPIRType &get_pointee_type(uint32_t type_id) const;

	// Gets the ID of the SPIR-V type underlying a variable.
	uint32_t get_variable_data_type_id(const SPIRVariable &var) const;

	// Gets the SPIR-V type underlying a variable.
	SPIRType &get_variable_data_type(const SPIRVariable &var);

	// Gets the SPIR-V type underlying a variable.
	const SPIRType &get_variable_data_type(const SPIRVariable &var) const;

	// Gets the SPIR-V element type underlying an array variable.
	SPIRType &get_variable_element_type(const SPIRVariable &var);

	// Gets the SPIR-V element type underlying an array variable.
	const SPIRType &get_variable_element_type(const SPIRVariable &var) const;

	// Sets the qualified member identifier for OpTypeStruct ID, member number "index".
	void set_member_qualified_name(uint32_t type_id, uint32_t index, const std::string &name);
	void set_qualified_name(uint32_t id, const std::string &name);

	// Returns if the given type refers to a sampled image.
	bool is_sampled_image_type(const SPIRType &type);

	const SPIREntryPoint &get_entry_point() const;
	SPIREntryPoint &get_entry_point();
	static bool is_tessellation_shader(ExecutionModel model);

	virtual std::string to_name(uint32_t id, bool allow_alias = true) const;
	bool is_builtin_variable(const SPIRVariable &var) const;
	bool is_builtin_type(const SPIRType &type) const;
	bool is_hidden_variable(const SPIRVariable &var, bool include_builtins = false) const;
	bool is_immutable(uint32_t id) const;
	bool is_member_builtin(const SPIRType &type, uint32_t index, BuiltIn *builtin) const;
	bool is_scalar(const SPIRType &type) const;
	bool is_vector(const SPIRType &type) const;
	bool is_matrix(const SPIRType &type) const;
	bool is_array(const SPIRType &type) const;
	bool is_pointer(const SPIRType &type) const;
	bool is_physical_pointer(const SPIRType &type) const;
	bool is_physical_or_buffer_pointer(const SPIRType &type) const;
	bool is_physical_pointer_to_buffer_block(const SPIRType &type) const;
	static bool is_runtime_size_array(const SPIRType &type);
	uint32_t expression_type_id(uint32_t id) const;
	const SPIRType &expression_type(uint32_t id) const;
	bool expression_is_lvalue(uint32_t id) const;
	bool variable_storage_is_aliased(const SPIRVariable &var);
	SPIRVariable *maybe_get_backing_variable(uint32_t chain);

	void register_read(uint32_t expr, uint32_t chain, bool forwarded);
	void register_write(uint32_t chain);

	inline bool is_continue(uint32_t next) const
	{
		return (ir.block_meta[next] & ParsedIR::BLOCK_META_CONTINUE_BIT) != 0;
	}

	inline bool is_single_block_loop(uint32_t next) const
	{
		auto &block = get<SPIRBlock>(next);
		return block.merge == SPIRBlock::MergeLoop && block.continue_block == ID(next);
	}

	inline bool is_break(uint32_t next) const
	{
		return (ir.block_meta[next] &
		        (ParsedIR::BLOCK_META_LOOP_MERGE_BIT | ParsedIR::BLOCK_META_MULTISELECT_MERGE_BIT)) != 0;
	}

	inline bool is_loop_break(uint32_t next) const
	{
		return (ir.block_meta[next] & ParsedIR::BLOCK_META_LOOP_MERGE_BIT) != 0;
	}

	inline bool is_conditional(uint32_t next) const
	{
		return (ir.block_meta[next] &
		        (ParsedIR::BLOCK_META_SELECTION_MERGE_BIT | ParsedIR::BLOCK_META_MULTISELECT_MERGE_BIT)) != 0;
	}

	// Dependency tracking for temporaries read from variables.
	void flush_dependees(SPIRVariable &var);
	void flush_all_active_variables();
	void flush_control_dependent_expressions(uint32_t block);
	void flush_all_atomic_capable_variables();
	void flush_all_aliased_variables();
	void register_global_read_dependencies(const SPIRBlock &func, uint32_t id);
	void register_global_read_dependencies(const SPIRFunction &func, uint32_t id);
	std::unordered_set<uint32_t> invalid_expressions;

	void update_name_cache(std::unordered_set<std::string> &cache, std::string &name);

	// A variant which takes two sets of names. The secondary is only used to verify there are no collisions,
	// but the set is not updated when we have found a new name.
	// Used primarily when adding block interface names.
	void update_name_cache(std::unordered_set<std::string> &cache_primary,
	                       const std::unordered_set<std::string> &cache_secondary, std::string &name);

	bool function_is_pure(const SPIRFunction &func);
	bool block_is_pure(const SPIRBlock &block);
	bool function_is_control_dependent(const SPIRFunction &func);
	bool block_is_control_dependent(const SPIRBlock &block);

	bool execution_is_branchless(const SPIRBlock &from, const SPIRBlock &to) const;
	bool execution_is_direct_branch(const SPIRBlock &from, const SPIRBlock &to) const;
	bool execution_is_noop(const SPIRBlock &from, const SPIRBlock &to) const;
	SPIRBlock::ContinueBlockType continue_block_type(const SPIRBlock &continue_block) const;

	void force_recompile();
	void force_recompile_guarantee_forward_progress();
	void clear_force_recompile();
	bool is_forcing_recompilation() const;
	bool is_force_recompile = false;
	bool is_force_recompile_forward_progress = false;

	bool block_is_noop(const SPIRBlock &block) const;
	bool block_is_loop_candidate(const SPIRBlock &block, SPIRBlock::Method method) const;

	bool types_are_logically_equivalent(const SPIRType &a, const SPIRType &b) const;
	void inherit_expression_dependencies(uint32_t dst, uint32_t source);
	void add_implied_read_expression(SPIRExpression &e, uint32_t source);
	void add_implied_read_expression(SPIRAccessChain &e, uint32_t source);
	void add_active_interface_variable(uint32_t var_id);

	// For proper multiple entry point support, allow querying if an Input or Output
	// variable is part of that entry points interface.
	bool interface_variable_exists_in_entry_point(uint32_t id) const;

	SmallVector<CombinedImageSampler> combined_image_samplers;

	void remap_variable_type_name(const SPIRType &type, const std::string &var_name, std::string &type_name) const
	{
		if (variable_remap_callback)
			variable_remap_callback(type, var_name, type_name);
	}

	void set_ir(const ParsedIR &parsed);
	void set_ir(ParsedIR &&parsed);
	void parse_fixup();

	// Used internally to implement various traversals for queries.
	struct OpcodeHandler
	{
		explicit OpcodeHandler(Compiler &compiler_) : compiler(compiler_) {}
		virtual ~OpcodeHandler() = default;

		// Return true if traversal should continue.
		// If false, traversal will end immediately.
		virtual bool handle(Op opcode, const uint32_t *args, uint32_t length) = 0;
		virtual bool handle_terminator(const SPIRBlock &)
		{
			return true;
		}

		virtual bool follow_function_call(const SPIRFunction &)
		{
			return true;
		}

		virtual void set_current_block(const SPIRBlock &)
		{
		}

		// Called after returning from a function or when entering a block,
		// can be called multiple times per block,
		// while set_current_block is only called on block entry.
		virtual void rearm_current_block(const SPIRBlock &)
		{
		}

		virtual bool begin_function_scope(const uint32_t *, uint32_t)
		{
			return true;
		}

		virtual bool end_function_scope(const uint32_t *, uint32_t)
		{
			return true;
		}

		Compiler &compiler;
		std::unordered_map<uint32_t, uint32_t> result_types;
		const SPIRType *get_expression_result_type(uint32_t id) const;
		bool enable_result_types = false;

		template <typename T> T &get(uint32_t id)
		{
			return compiler.get<T>(id);
		}

		template <typename T> const T &get(uint32_t id) const
		{
			return compiler.get<T>(id);
		}

		template <typename T, typename... P>
		T &set(uint32_t id, P &&... args)
		{
			return compiler.set<T>(id, std::forward<P>(args)...);
		}
	};

	struct BufferAccessHandler : OpcodeHandler
	{
		BufferAccessHandler(const Compiler &compiler_, SmallVector<BufferRange> &ranges_, uint32_t id_)
		    : OpcodeHandler(const_cast<Compiler &>(compiler_))
		    , ranges(ranges_)
		    , id(id_)
		{
		}

		bool handle(Op opcode, const uint32_t *args, uint32_t length) override;

		SmallVector<BufferRange> &ranges;
		uint32_t id;

		std::unordered_set<uint32_t> seen;
	};

	struct InterfaceVariableAccessHandler : OpcodeHandler
	{
		InterfaceVariableAccessHandler(const Compiler &compiler_, std::unordered_set<VariableID> &variables_)
		    : OpcodeHandler(const_cast<Compiler &>(compiler_))
		    , variables(variables_)
		{
		}

		bool handle(Op opcode, const uint32_t *args, uint32_t length) override;

		std::unordered_set<VariableID> &variables;
	};

	struct CombinedImageSamplerHandler : OpcodeHandler
	{
		explicit CombinedImageSamplerHandler(Compiler &compiler_)
		    : OpcodeHandler(compiler_)
		{
		}
		bool handle(Op opcode, const uint32_t *args, uint32_t length) override;
		bool begin_function_scope(const uint32_t *args, uint32_t length) override;
		bool end_function_scope(const uint32_t *args, uint32_t length) override;

		// Each function in the call stack needs its own remapping for parameters so we can deduce which global variable each texture/sampler the parameter is statically bound to.
		std::stack<std::unordered_map<uint32_t, uint32_t>> parameter_remapping;
		std::stack<SPIRFunction *> functions;

		uint32_t remap_parameter(uint32_t id);
		void push_remap_parameters(const SPIRFunction &func, const uint32_t *args, uint32_t length);
		void pop_remap_parameters();
		void register_combined_image_sampler(SPIRFunction &caller, VariableID combined_id, VariableID texture_id,
		                                     VariableID sampler_id, bool depth);
	};

	struct DummySamplerForCombinedImageHandler : OpcodeHandler
	{
		explicit DummySamplerForCombinedImageHandler(Compiler &compiler_)
		    : OpcodeHandler(compiler_)
		{
		}
		bool handle(Op opcode, const uint32_t *args, uint32_t length) override;
		bool need_dummy_sampler = false;
	};

	struct ActiveBuiltinHandler : OpcodeHandler
	{
		explicit ActiveBuiltinHandler(Compiler &compiler_)
		    : OpcodeHandler(compiler_)
		{
		}

		bool handle(Op opcode, const uint32_t *args, uint32_t length) override;

		void handle_builtin(const SPIRType &type, BuiltIn builtin, const Bitset &decoration_flags);
		void add_if_builtin(uint32_t id);
		void add_if_builtin_or_block(uint32_t id);
		void add_if_builtin(uint32_t id, bool allow_blocks);
	};

	bool traverse_all_reachable_opcodes(const SPIRBlock &block, OpcodeHandler &handler) const;
	bool traverse_all_reachable_opcodes(const SPIRFunction &block, OpcodeHandler &handler) const;
	// This must be an ordered data structure so we always pick the same type aliases.
	SmallVector<uint32_t> global_struct_cache;

	ShaderResources get_shader_resources(const std::unordered_set<VariableID> *active_variables) const;

	VariableTypeRemapCallback variable_remap_callback;

	bool get_common_basic_type(const SPIRType &type, SPIRType::BaseType &base_type);

	std::unordered_set<uint32_t> forced_temporaries;
	std::unordered_set<uint32_t> forwarded_temporaries;
	std::unordered_set<uint32_t> suppressed_usage_tracking;
	std::unordered_set<uint32_t> hoisted_temporaries;
	std::unordered_set<uint32_t> forced_invariant_temporaries;

	Bitset active_input_builtins;
	Bitset active_output_builtins;
	uint32_t clip_distance_count = 0;
	uint32_t cull_distance_count = 0;
	bool position_invariant = false;

	void analyze_parameter_preservation(
	    SPIRFunction &entry, const CFG &cfg,
	    const std::unordered_map<uint32_t, std::unordered_set<uint32_t>> &variable_to_blocks,
	    const std::unordered_map<uint32_t, std::unordered_set<uint32_t>> &complete_write_blocks);

	// If a variable ID or parameter ID is found in this set, a sampler is actually a shadow/comparison sampler.
	// SPIR-V does not support this distinction, so we must keep track of this information outside the type system.
	// There might be unrelated IDs found in this set which do not correspond to actual variables.
	// This set should only be queried for the existence of samplers which are already known to be variables or parameter IDs.
	// Similar is implemented for images, as well as if subpass inputs are needed.
	std::unordered_set<uint32_t> comparison_ids;
	bool need_subpass_input = false;
	bool need_subpass_input_ms = false;

	// In certain backends, we will need to use a dummy sampler to be able to emit code.
	// GLSL does not support texelFetch on texture2D objects, but SPIR-V does,
	// so we need to workaround by having the application inject a dummy sampler.
	uint32_t dummy_sampler_id = 0;

	void analyze_image_and_sampler_usage();

	struct CombinedImageSamplerDrefHandler : OpcodeHandler
	{
		explicit CombinedImageSamplerDrefHandler(Compiler &compiler_)
		    : OpcodeHandler(compiler_)
		{
		}
		bool handle(Op opcode, const uint32_t *args, uint32_t length) override;

		std::unordered_set<uint32_t> dref_combined_samplers;
	};

	struct CombinedImageSamplerUsageHandler : OpcodeHandler
	{
		CombinedImageSamplerUsageHandler(Compiler &compiler_,
		                                 const std::unordered_set<uint32_t> &dref_combined_samplers_)
		    : OpcodeHandler(compiler_)
		    , dref_combined_samplers(dref_combined_samplers_)
		{
		}

		bool begin_function_scope(const uint32_t *args, uint32_t length) override;
		bool handle(Op opcode, const uint32_t *args, uint32_t length) override;
		const std::unordered_set<uint32_t> &dref_combined_samplers;

		std::unordered_map<uint32_t, std::unordered_set<uint32_t>> dependency_hierarchy;
		std::unordered_set<uint32_t> comparison_ids;

		void add_hierarchy_to_comparison_ids(uint32_t ids);
		bool need_subpass_input = false;
		bool need_subpass_input_ms = false;
		void add_dependency(uint32_t dst, uint32_t src);
	};

	void build_function_control_flow_graphs_and_analyze();
	std::unordered_map<uint32_t, std::unique_ptr<CFG>> function_cfgs;
	const CFG &get_cfg_for_current_function() const;
	const CFG &get_cfg_for_function(uint32_t id) const;

	struct CFGBuilder : OpcodeHandler
	{
		explicit CFGBuilder(Compiler &compiler_);

		bool follow_function_call(const SPIRFunction &func) override;
		bool handle(Op op, const uint32_t *args, uint32_t length) override;
		std::unordered_map<uint32_t, std::unique_ptr<CFG>> function_cfgs;
	};

	struct AnalyzeVariableScopeAccessHandler : OpcodeHandler
	{
		AnalyzeVariableScopeAccessHandler(Compiler &compiler_, SPIRFunction &entry_);

		bool follow_function_call(const SPIRFunction &) override;
		void set_current_block(const SPIRBlock &block) override;

		void notify_variable_access(uint32_t id, uint32_t block);
		bool id_is_phi_variable(uint32_t id) const;
		bool id_is_potential_temporary(uint32_t id) const;
		bool handle(Op op, const uint32_t *args, uint32_t length) override;
		bool handle_terminator(const SPIRBlock &block) override;

		SPIRFunction &entry;
		std::unordered_map<uint32_t, std::unordered_set<uint32_t>> accessed_variables_to_block;
		std::unordered_map<uint32_t, std::unordered_set<uint32_t>> accessed_temporaries_to_block;
		std::unordered_map<uint32_t, uint32_t> result_id_to_type;
		std::unordered_map<uint32_t, std::unordered_set<uint32_t>> complete_write_variables_to_block;
		std::unordered_map<uint32_t, std::unordered_set<uint32_t>> partial_write_variables_to_block;
		std::unordered_set<uint32_t> access_chain_expressions;
		// Access chains used in multiple blocks mean hoisting all the variables used to construct the access chain as not all backends can use pointers.
		// This is also relevant when forwarding opaque objects since we cannot lower these to temporaries.
		std::unordered_map<uint32_t, std::unordered_set<uint32_t>> rvalue_forward_children;
		const SPIRBlock *current_block = nullptr;
	};

	struct StaticExpressionAccessHandler : OpcodeHandler
	{
		StaticExpressionAccessHandler(Compiler &compiler_, uint32_t variable_id_);
		bool follow_function_call(const SPIRFunction &) override;
		bool handle(Op op, const uint32_t *args, uint32_t length) override;

		uint32_t variable_id;
		uint32_t static_expression = 0;
		uint32_t write_count = 0;
	};

	struct PhysicalBlockMeta
	{
		uint32_t alignment = 0;
	};

	struct PhysicalStorageBufferPointerHandler : OpcodeHandler
	{
		explicit PhysicalStorageBufferPointerHandler(Compiler &compiler_);
		bool handle(Op op, const uint32_t *args, uint32_t length) override;

		std::unordered_set<uint32_t> non_block_types;
		std::unordered_map<uint32_t, PhysicalBlockMeta> physical_block_type_meta;
		std::unordered_map<uint32_t, PhysicalBlockMeta *> access_chain_to_physical_block;
		std::unordered_set<uint32_t> analyzed_type_ids;

		void mark_aligned_access(uint32_t id, const uint32_t *args, uint32_t length);
		PhysicalBlockMeta *find_block_meta(uint32_t id) const;
		bool type_is_bda_block_entry(uint32_t type_id) const;
		void setup_meta_chain(uint32_t type_id, uint32_t var_id);
		uint32_t get_minimum_scalar_alignment(const SPIRType &type) const;
		void analyze_non_block_types_from_block(const SPIRType &type);
		uint32_t get_base_non_block_type_id(uint32_t type_id) const;
	};
	void analyze_non_block_pointer_types();
	SmallVector<uint32_t> physical_storage_non_block_pointer_types;
	std::unordered_map<uint32_t, PhysicalBlockMeta> physical_storage_type_to_alignment;

	void analyze_variable_scope(SPIRFunction &function, AnalyzeVariableScopeAccessHandler &handler);
	void find_function_local_luts(SPIRFunction &function, const AnalyzeVariableScopeAccessHandler &handler,
	                              bool single_function);
	bool may_read_undefined_variable_in_block(const SPIRBlock &block, uint32_t var);

	struct GeometryEmitDisocveryHandler : OpcodeHandler
	{
		explicit GeometryEmitDisocveryHandler(Compiler &compiler_)
		    : OpcodeHandler(compiler_)
		{
		}

		bool handle(Op opcode, const uint32_t *args, uint32_t length) override;
		bool begin_function_scope(const uint32_t *, uint32_t) override;
		bool end_function_scope(const uint32_t *, uint32_t) override;
		SmallVector<SPIRFunction *> function_stack;
	};

	void discover_geometry_emitters();

	// Finds all resources that are written to from inside the critical section, if present.
	// The critical section is delimited by OpBeginInvocationInterlockEXT and
	// OpEndInvocationInterlockEXT instructions. In MSL and HLSL, any resources written
	// while inside the critical section must be placed in a raster order group.
	struct InterlockedResourceAccessHandler : OpcodeHandler
	{
		InterlockedResourceAccessHandler(Compiler &compiler_, uint32_t entry_point_id)
		    : OpcodeHandler(compiler_)
		{
			call_stack.push_back(entry_point_id);
		}

		bool handle(Op op, const uint32_t *args, uint32_t length) override;
		bool begin_function_scope(const uint32_t *args, uint32_t length) override;
		bool end_function_scope(const uint32_t *args, uint32_t length) override;

		bool in_crit_sec = false;

		uint32_t interlock_function_id = 0;
		bool split_function_case = false;
		bool control_flow_interlock = false;
		bool use_critical_section = false;
		bool call_stack_is_interlocked = false;
		SmallVector<uint32_t> call_stack;

		void access_potential_resource(uint32_t id);
	};

	struct InterlockedResourceAccessPrepassHandler : OpcodeHandler
	{
		InterlockedResourceAccessPrepassHandler(Compiler &compiler_, uint32_t entry_point_id)
		    : OpcodeHandler(compiler_)
		{
			call_stack.push_back(entry_point_id);
		}

		void rearm_current_block(const SPIRBlock &block) override;
		bool handle(Op op, const uint32_t *args, uint32_t length) override;
		bool begin_function_scope(const uint32_t *args, uint32_t length) override;
		bool end_function_scope(const uint32_t *args, uint32_t length) override;

		uint32_t interlock_function_id = 0;
		uint32_t current_block_id = 0;
		bool split_function_case = false;
		bool control_flow_interlock = false;
		SmallVector<uint32_t> call_stack;
	};

	void analyze_interlocked_resource_usage();
	// The set of all resources written while inside the critical section, if present.
	std::unordered_set<uint32_t> interlocked_resources;
	bool interlocked_is_complex = false;

	void make_constant_null(uint32_t id, uint32_t type);

	std::unordered_map<uint32_t, std::string> declared_block_names;

	static bool instruction_to_result_type(
			uint32_t &result_type, uint32_t &result_id, Op op, const uint32_t *args, uint32_t length);

	Bitset combined_decoration_for_member(const SPIRType &type, uint32_t index) const;
	static bool is_desktop_only_format(ImageFormat format);

	bool is_depth_image(const SPIRType &type, uint32_t id) const;

	void set_extended_decoration(uint32_t id, ExtendedDecorations decoration, uint32_t value = 0);
	uint32_t get_extended_decoration(uint32_t id, ExtendedDecorations decoration) const;
	bool has_extended_decoration(uint32_t id, ExtendedDecorations decoration) const;
	void unset_extended_decoration(uint32_t id, ExtendedDecorations decoration);

	void set_extended_member_decoration(uint32_t type, uint32_t index, ExtendedDecorations decoration,
	                                    uint32_t value = 0);
	uint32_t get_extended_member_decoration(uint32_t type, uint32_t index, ExtendedDecorations decoration) const;
	bool has_extended_member_decoration(uint32_t type, uint32_t index, ExtendedDecorations decoration) const;
	void unset_extended_member_decoration(uint32_t type, uint32_t index, ExtendedDecorations decoration);

	bool check_internal_recursion(const SPIRType &type, std::unordered_set<uint32_t> &checked_ids);
	bool type_contains_recursion(const SPIRType &type);
	bool type_is_array_of_pointers(const SPIRType &type) const;
	bool type_is_block_like(const SPIRType &type) const;
	bool type_is_explicit_layout(const SPIRType &type) const;
	bool type_is_top_level_block(const SPIRType &type) const;
	bool type_is_opaque_value(const SPIRType &type) const;

	bool reflection_ssbo_instance_name_is_significant() const;
	std::string get_remapped_declared_block_name(uint32_t id, bool fallback_prefer_instance_name) const;

	bool flush_phi_required(BlockID from, BlockID to) const;

	uint32_t evaluate_spec_constant_u32(const SPIRConstantOp &spec) const;
	uint32_t evaluate_constant_u32(uint32_t id) const;

	bool is_vertex_like_shader() const;

	// Get the correct case list for the OpSwitch, since it can be either a
	// 32 bit wide condition or a 64 bit, but the type is not embedded in the
	// instruction itself.
	const SmallVector<SPIRBlock::Case> &get_case_list(const SPIRBlock &block) const;

private:
	// Used only to implement the old deprecated get_entry_point() interface.
	const SPIREntryPoint &get_first_entry_point(const std::string &name) const;
	SPIREntryPoint &get_first_entry_point(const std::string &name);
};
} // namespace SPIRV_CROSS_NAMESPACE

#ifdef SPIRV_CROSS_SPV_HEADER_NAMESPACE_OVERRIDE
#undef spv
#endif

#endif
