// Copyright (c) 2015-2016 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SOURCE_VAL_VALIDATION_STATE_H_
#define SOURCE_VAL_VALIDATION_STATE_H_

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "source/assembly_grammar.h"
#include "source/diagnostic.h"
#include "source/disassemble.h"
#include "source/enum_set.h"
#include "source/latest_version_spirv_header.h"
#include "source/name_mapper.h"
#include "source/spirv_definition.h"
#include "source/spirv_validator_options.h"
#include "source/val/decoration.h"
#include "source/val/function.h"
#include "source/val/instruction.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {
namespace val {

/// This enum represents the sections of a SPIRV module. See section 2.4
/// of the SPIRV spec for additional details of the order. The enumerant values
/// are in the same order as the vector returned by GetModuleOrder
enum ModuleLayoutSection {
  kLayoutCapabilities,             /// < Section 2.4 #1
  kLayoutExtensions,               /// < Section 2.4 #2
  kLayoutExtInstImport,            /// < Section 2.4 #3
  kLayoutMemoryModel,              /// < Section 2.4 #4
  kLayoutSamplerImageAddressMode,  /// < Section 2.4 #5
  kLayoutEntryPoint,               /// < Section 2.4 #6
  kLayoutExecutionMode,            /// < Section 2.4 #7
  kLayoutDebug1,                   /// < Section 2.4 #8 > 1
  kLayoutDebug2,                   /// < Section 2.4 #8 > 2
  kLayoutDebug3,                   /// < Section 2.4 #8 > 3
  kLayoutAnnotations,              /// < Section 2.4 #9
  kLayoutTypes,                    /// < Section 2.4 #10
  kLayoutFunctionDeclarations,     /// < Section 2.4 #11
  kLayoutFunctionDefinitions       /// < Section 2.4 #12
};

/// This class manages the state of the SPIR-V validation as it is being parsed.
class ValidationState_t {
 public:
  // Features that can optionally be turned on by a capability or environment.
  struct Feature {
    bool declare_int16_type = false;     // Allow OpTypeInt with 16 bit width?
    bool declare_float16_type = false;   // Allow OpTypeFloat with 16 bit width?
    bool free_fp_rounding_mode = false;  // Allow the FPRoundingMode decoration
                                         // and its values to be used without
                                         // requiring any capability

    // Allow functionalities enabled by VariablePointers or
    // VariablePointersStorageBuffer capability.
    bool variable_pointers = false;

    // Permit group oerations Reduce, InclusiveScan, ExclusiveScan
    bool group_ops_reduce_and_scans = false;

    // Allow OpTypeInt with 8 bit width?
    bool declare_int8_type = false;

    // Target environment uses relaxed block layout.
    // This is true for Vulkan 1.1 or later.
    bool env_relaxed_block_layout = false;

    // Allow an OpTypeInt with 8 bit width to be used in more than just int
    // conversion opcodes
    bool use_int8_type = false;

    // SPIR-V 1.4 allows us to select between any two composite values
    // of the same type.
    bool select_between_composites = false;

    // SPIR-V 1.4 allows two memory access operands for OpCopyMemory and
    // OpCopyMemorySized.
    bool copy_memory_permits_two_memory_accesses = false;

    // SPIR-V 1.4 allows UConvert as a spec constant op in any environment.
    // The Kernel capability already enables it, separately from this flag.
    bool uconvert_spec_constant_op = false;

    // SPIR-V 1.4 allows Function and Private variables to be NonWritable
    bool nonwritable_var_in_function_or_private = false;

    // Whether LocalSizeId execution mode is allowed by the environment.
    bool env_allow_localsizeid = false;
  };

  ValidationState_t(const spv_const_context context,
                    const spv_const_validator_options opt,
                    const uint32_t* words, const size_t num_words,
                    const uint32_t max_warnings);

  /// Returns the context
  spv_const_context context() const { return context_; }

  /// Returns the command line options
  spv_const_validator_options options() const { return options_; }

  /// Sets the ID of the generator for this module.
  void setGenerator(uint32_t gen) { generator_ = gen; }

  /// Returns the ID of the generator for this module.
  uint32_t generator() const { return generator_; }

  /// Sets the SPIR-V version of this module.
  void setVersion(uint32_t ver) { version_ = ver; }

  /// Gets the SPIR-V version of this module.
  uint32_t version() const { return version_; }

  /// Forward declares the id in the module
  spv_result_t ForwardDeclareId(uint32_t id);

  /// Removes a forward declared ID if it has been defined
  spv_result_t RemoveIfForwardDeclared(uint32_t id);

  /// Registers an ID as a forward pointer
  spv_result_t RegisterForwardPointer(uint32_t id);

  /// Returns whether or not an ID is a forward pointer
  bool IsForwardPointer(uint32_t id) const;

  /// Assigns a name to an ID
  void AssignNameToId(uint32_t id, std::string name);

  /// Returns a string representation of the ID in the format <id>[Name] where
  /// the <id> is the numeric valid of the id and the Name is a name assigned by
  /// the OpName instruction
  std::string getIdName(uint32_t id) const;

  /// Accessor function for ID bound.
  uint32_t getIdBound() const;

  /// Mutator function for ID bound.
  void setIdBound(uint32_t bound);

  /// Returns the number of ID which have been forward referenced but not
  /// defined
  size_t unresolved_forward_id_count() const;

  /// Returns a vector of unresolved forward ids.
  std::vector<uint32_t> UnresolvedForwardIds() const;

  /// Returns true if the id has been defined
  bool IsDefinedId(uint32_t id) const;

  /// Increments the total number of instructions in the file.
  void increment_total_instructions() { total_instructions_++; }

  /// Increments the total number of functions in the file.
  void increment_total_functions() { total_functions_++; }

  /// Allocates internal storage. Note, calling this will invalidate any
  /// pointers to |ordered_instructions_| or |module_functions_| and, hence,
  /// should only be called at the beginning of validation.
  void preallocateStorage();

  /// Returns the current layout section which is being processed
  ModuleLayoutSection current_layout_section() const;

  /// Increments the module_layout_order_section_
  void ProgressToNextLayoutSectionOrder();

  /// Determines if the op instruction is in a previous layout section
  bool IsOpcodeInPreviousLayoutSection(spv::Op op);

  /// Determines if the op instruction is part of the current section
  bool IsOpcodeInCurrentLayoutSection(spv::Op op);

  DiagnosticStream diag(spv_result_t error_code, const Instruction* inst);

  /// Returns the function states
  std::vector<Function>& functions();

  /// Returns the function states
  Function& current_function();
  const Function& current_function() const;

  /// Returns function state with the given id, or nullptr if no such function.
  const Function* function(uint32_t id) const;
  Function* function(uint32_t id);

  /// Returns true if the called after a function instruction but before the
  /// function end instruction
  bool in_function_body() const;

  /// Returns true if called after a label instruction but before a branch
  /// instruction
  bool in_block() const;

  struct EntryPointDescription {
    std::string name;
    std::vector<uint32_t> interfaces;
  };

  /// Registers |id| as an entry point with |execution_model| and |interfaces|.
  void RegisterEntryPoint(const uint32_t id,
                          spv::ExecutionModel execution_model,
                          EntryPointDescription&& desc) {
    entry_points_.push_back(id);
    entry_point_to_execution_models_[id].insert(execution_model);
    entry_point_descriptions_[id].emplace_back(desc);
  }

  /// Returns a list of entry point function ids
  const std::vector<uint32_t>& entry_points() const { return entry_points_; }

  /// Returns the set of entry points that root call graphs that contain
  /// recursion.
  const std::set<uint32_t>& recursive_entry_points() const {
    return recursive_entry_points_;
  }

  /// Registers execution mode for the given entry point.
  void RegisterExecutionModeForEntryPoint(uint32_t entry_point,
                                          spv::ExecutionMode execution_mode) {
    entry_point_to_execution_modes_[entry_point].insert(execution_mode);
  }

  /// Returns the interface descriptions of a given entry point.
  const std::vector<EntryPointDescription>& entry_point_descriptions(
      uint32_t entry_point) {
    return entry_point_descriptions_.at(entry_point);
  }

  /// Returns Execution Models for the given Entry Point.
  /// Returns nullptr if none found (would trigger assertion).
  const std::set<spv::ExecutionModel>* GetExecutionModels(
      uint32_t entry_point) const {
    const auto it = entry_point_to_execution_models_.find(entry_point);
    if (it == entry_point_to_execution_models_.end()) {
      assert(0);
      return nullptr;
    }
    return &it->second;
  }

  /// Returns Execution Modes for the given Entry Point.
  /// Returns nullptr if none found.
  const std::set<spv::ExecutionMode>* GetExecutionModes(
      uint32_t entry_point) const {
    const auto it = entry_point_to_execution_modes_.find(entry_point);
    if (it == entry_point_to_execution_modes_.end()) {
      return nullptr;
    }
    return &it->second;
  }

  /// Traverses call tree and computes function_to_entry_points_.
  /// Note: called after fully parsing the binary.
  void ComputeFunctionToEntryPointMapping();

  /// Traverse call tree and computes recursive_entry_points_.
  /// Note: called after fully parsing the binary and calling
  /// ComputeFunctionToEntryPointMapping.
  void ComputeRecursiveEntryPoints();

  /// Returns all the entry points that can call |func|.
  const std::vector<uint32_t>& FunctionEntryPoints(uint32_t func) const;

  /// Returns all the entry points that statically use |id|.
  ///
  /// Note: requires ComputeFunctionToEntryPointMapping to have been called.
  std::set<uint32_t> EntryPointReferences(uint32_t id) const;

  /// Inserts an <id> to the set of functions that are target of OpFunctionCall.
  void AddFunctionCallTarget(const uint32_t id) {
    function_call_targets_.insert(id);
    current_function().AddFunctionCallTarget(id);
  }

  /// Returns whether or not a function<id> is the target of OpFunctionCall.
  bool IsFunctionCallTarget(const uint32_t id) {
    return (function_call_targets_.find(id) != function_call_targets_.end());
  }

  bool IsFunctionCallDefined(const uint32_t id) {
    return (id_to_function_.find(id) != id_to_function_.end());
  }
  /// Registers the capability and its dependent capabilities
  void RegisterCapability(spv::Capability cap);

  /// Registers the extension.
  void RegisterExtension(Extension ext);

  /// Registers the function in the module. Subsequent instructions will be
  /// called against this function
  spv_result_t RegisterFunction(uint32_t id, uint32_t ret_type_id,
                                spv::FunctionControlMask function_control,
                                uint32_t function_type_id);

  /// Register a function end instruction
  spv_result_t RegisterFunctionEnd();

  /// Returns true if the capability is enabled in the module.
  bool HasCapability(spv::Capability cap) const {
    return module_capabilities_.Contains(cap);
  }

  /// Returns a reference to the set of capabilities in the module.
  /// This is provided for debuggability.
  const CapabilitySet& module_capabilities() const {
    return module_capabilities_;
  }

  /// Returns true if the extension is enabled in the module.
  bool HasExtension(Extension ext) const {
    return module_extensions_.Contains(ext);
  }

  /// Returns true if any of the capabilities is enabled, or if |capabilities|
  /// is an empty set.
  bool HasAnyOfCapabilities(const CapabilitySet& capabilities) const;

  /// Returns true if any of the extensions is enabled, or if |extensions|
  /// is an empty set.
  bool HasAnyOfExtensions(const ExtensionSet& extensions) const;

  /// Sets the addressing model of this module (logical/physical).
  void set_addressing_model(spv::AddressingModel am);

  /// Returns true if the OpMemoryModel was found.
  bool has_memory_model_specified() const {
    return addressing_model_ != spv::AddressingModel::Max &&
           memory_model_ != spv::MemoryModel::Max;
  }

  /// Returns the addressing model of this module, or Logical if uninitialized.
  spv::AddressingModel addressing_model() const;

  /// Returns the addressing model of this module, or Logical if uninitialized.
  uint32_t pointer_size_and_alignment() const {
    return pointer_size_and_alignment_;
  }

  /// Sets the memory model of this module.
  void set_memory_model(spv::MemoryModel mm);

  /// Returns the memory model of this module, or Simple if uninitialized.
  spv::MemoryModel memory_model() const;

  /// Sets the bit width for sampler/image type variables. If not set, they are
  /// considered opaque
  void set_samplerimage_variable_address_mode(uint32_t bit_width);

  /// Get the addressing mode currently set. If 0, it means addressing mode is
  /// invalid Sampler/Image type variables must be considered opaque This mode
  /// is only valid after the instruction has been read
  uint32_t samplerimage_variable_address_mode() const;

  /// Returns true if the OpSamplerImageAddressingModeNV was found.
  bool has_samplerimage_variable_address_mode_specified() const {
    return sampler_image_addressing_mode_ != 0;
  }

  const AssemblyGrammar& grammar() const { return grammar_; }

  /// Inserts the instruction into the list of ordered instructions in the file.
  Instruction* AddOrderedInstruction(const spv_parsed_instruction_t* inst);

  /// Registers the instruction. This will add the instruction to the list of
  /// definitions and register sampled image consumers.
  void RegisterInstruction(Instruction* inst);

  /// Registers the debug instruction information.
  void RegisterDebugInstruction(const Instruction* inst);

  /// Registers the decoration for the given <id>
  void RegisterDecorationForId(uint32_t id, const Decoration& dec) {
    auto& dec_list = id_decorations_[id];
    dec_list.insert(dec);
  }

  /// Registers the list of decorations for the given <id>
  template <class InputIt>
  void RegisterDecorationsForId(uint32_t id, InputIt begin, InputIt end) {
    std::set<Decoration>& cur_decs = id_decorations_[id];
    cur_decs.insert(begin, end);
  }

  /// Registers the list of decorations for the given member of the given
  /// structure.
  template <class InputIt>
  void RegisterDecorationsForStructMember(uint32_t struct_id,
                                          uint32_t member_index, InputIt begin,
                                          InputIt end) {
    std::set<Decoration>& cur_decs = id_decorations_[struct_id];
    for (InputIt iter = begin; iter != end; ++iter) {
      Decoration dec = *iter;
      dec.set_struct_member_index(member_index);
      cur_decs.insert(dec);
    }
  }

  /// Returns all the decorations for the given <id>. If no decorations exist
  /// for the <id>, it registers an empty set for it in the map and
  /// returns the empty set.
  std::set<Decoration>& id_decorations(uint32_t id) {
    return id_decorations_[id];
  }

  /// Returns the range of decorations for the given field of the given <id>.
  struct FieldDecorationsIter {
    std::set<Decoration>::const_iterator begin;
    std::set<Decoration>::const_iterator end;
  };
  FieldDecorationsIter id_member_decorations(uint32_t id,
                                             uint32_t member_index) {
    const auto& decorations = id_decorations_[id];

    // The decorations are sorted by member_index, so this look up will give the
    // exact range of decorations for this member index.
    Decoration min_decoration((spv::Decoration)0, {}, member_index);
    Decoration max_decoration(spv::Decoration::Max, {}, member_index);

    FieldDecorationsIter result;
    result.begin = decorations.lower_bound(min_decoration);
    result.end = decorations.upper_bound(max_decoration);

    return result;
  }

  // Returns const pointer to the internal decoration container.
  const std::map<uint32_t, std::set<Decoration>>& id_decorations() const {
    return id_decorations_;
  }

  /// Returns true if the given id <id> has the given decoration <dec>,
  /// otherwise returns false.
  bool HasDecoration(uint32_t id, spv::Decoration dec) {
    const auto& decorations = id_decorations_.find(id);
    if (decorations == id_decorations_.end()) return false;

    return std::any_of(
        decorations->second.begin(), decorations->second.end(),
        [dec](const Decoration& d) { return dec == d.dec_type(); });
  }

  /// Finds id's def, if it exists.  If found, returns the definition otherwise
  /// nullptr
  const Instruction* FindDef(uint32_t id) const;

  /// Finds id's def, if it exists.  If found, returns the definition otherwise
  /// nullptr
  Instruction* FindDef(uint32_t id);

  /// Returns the instructions in the order they appear in the binary
  const std::vector<Instruction>& ordered_instructions() const {
    return ordered_instructions_;
  }

  /// Returns a map of instructions mapped by their result id
  const std::unordered_map<uint32_t, Instruction*>& all_definitions() const {
    return all_definitions_;
  }

  /// Returns a vector containing the instructions that consume the given
  /// SampledImage id.
  std::vector<Instruction*> getSampledImageConsumers(uint32_t id) const;

  /// Records cons_id as a consumer of sampled_image_id.
  void RegisterSampledImageConsumer(uint32_t sampled_image_id,
                                    Instruction* consumer);

  // Record a function's storage class consumer instruction
  void RegisterStorageClassConsumer(spv::StorageClass storage_class,
                                    Instruction* consumer);

  /// Returns the set of Global Variables.
  std::unordered_set<uint32_t>& global_vars() { return global_vars_; }

  /// Returns the set of Local Variables.
  std::unordered_set<uint32_t>& local_vars() { return local_vars_; }

  /// Returns the number of Global Variables.
  size_t num_global_vars() { return global_vars_.size(); }

  /// Returns the number of Local Variables.
  size_t num_local_vars() { return local_vars_.size(); }

  /// Inserts a new <id> to the set of Global Variables.
  void registerGlobalVariable(const uint32_t id) { global_vars_.insert(id); }

  /// Inserts a new <id> to the set of Local Variables.
  void registerLocalVariable(const uint32_t id) { local_vars_.insert(id); }

  // Returns true if using relaxed block layout, equivalent to
  // VK_KHR_relaxed_block_layout.
  bool IsRelaxedBlockLayout() const {
    return features_.env_relaxed_block_layout || options()->relax_block_layout;
  }

  // Returns true if allowing localsizeid, either because the environment always
  // allows it, or because it is enabled from the command-line.
  bool IsLocalSizeIdAllowed() const {
    return features_.env_allow_localsizeid || options()->allow_localsizeid;
  }

  /// Sets the struct nesting depth for a given struct ID
  void set_struct_nesting_depth(uint32_t id, uint32_t depth) {
    struct_nesting_depth_[id] = depth;
  }

  /// Returns the nesting depth of a given structure ID
  uint32_t struct_nesting_depth(uint32_t id) {
    return struct_nesting_depth_[id];
  }

  /// Records the has a nested block/bufferblock decorated struct for a given
  /// struct ID
  void SetHasNestedBlockOrBufferBlockStruct(uint32_t id, bool has) {
    struct_has_nested_blockorbufferblock_struct_[id] = has;
  }

  /// For a given struct ID returns true if it has a nested block/bufferblock
  /// decorated struct
  bool GetHasNestedBlockOrBufferBlockStruct(uint32_t id) {
    return struct_has_nested_blockorbufferblock_struct_[id];
  }

  /// Records that the structure type has a member decorated with a built-in.
  void RegisterStructTypeWithBuiltInMember(uint32_t id) {
    builtin_structs_.insert(id);
  }

  /// Returns true if the struct type with the given Id has a BuiltIn member.
  bool IsStructTypeWithBuiltInMember(uint32_t id) const {
    return (builtin_structs_.find(id) != builtin_structs_.end());
  }

  // Returns the state of optional features.
  const Feature& features() const { return features_; }

  /// Adds the instruction data to unique_type_declarations_.
  /// Returns false if an identical type declaration already exists.
  bool RegisterUniqueTypeDeclaration(const Instruction* inst);

  // Returns type_id of the scalar component of |id|.
  // |id| can be either
  // - scalar, vector or matrix type
  // - object of either scalar, vector or matrix type
  uint32_t GetComponentType(uint32_t id) const;

  // Returns
  // - 1 for scalar types or objects
  // - vector size for vector types or objects
  // - num columns for matrix types or objects
  // Should not be called with any other arguments (will return zero and invoke
  // assertion).
  uint32_t GetDimension(uint32_t id) const;

  // Returns bit width of scalar or component.
  // |id| can be
  // - scalar, vector or matrix type
  // - object of either scalar, vector or matrix type
  // Will invoke assertion and return 0 if |id| is none of the above.
  uint32_t GetBitWidth(uint32_t id) const;

  // Provides detailed information on matrix type.
  // Returns false iff |id| is not matrix type.
  bool GetMatrixTypeInfo(uint32_t id, uint32_t* num_rows, uint32_t* num_cols,
                         uint32_t* column_type, uint32_t* component_type) const;

  // Collects struct member types into |member_types|.
  // Returns false iff not struct type or has no members.
  // Deletes prior contents of |member_types|.
  bool GetStructMemberTypes(uint32_t struct_type_id,
                            std::vector<uint32_t>* member_types) const;

  // Returns true iff |id| is a type corresponding to the name of the function.
  // Only works for types not for objects.
  bool IsVoidType(uint32_t id) const;
  bool IsFloatScalarType(uint32_t id) const;
  bool IsFloatVectorType(uint32_t id) const;
  bool IsFloatScalarOrVectorType(uint32_t id) const;
  bool IsFloatMatrixType(uint32_t id) const;
  bool IsIntScalarType(uint32_t id) const;
  bool IsIntVectorType(uint32_t id) const;
  bool IsIntScalarOrVectorType(uint32_t id) const;
  bool IsUnsignedIntScalarType(uint32_t id) const;
  bool IsUnsignedIntVectorType(uint32_t id) const;
  bool IsSignedIntScalarType(uint32_t id) const;
  bool IsSignedIntVectorType(uint32_t id) const;
  bool IsBoolScalarType(uint32_t id) const;
  bool IsBoolVectorType(uint32_t id) const;
  bool IsBoolScalarOrVectorType(uint32_t id) const;
  bool IsPointerType(uint32_t id) const;
  bool IsAccelerationStructureType(uint32_t id) const;
  bool IsCooperativeMatrixType(uint32_t id) const;
  bool IsFloatCooperativeMatrixType(uint32_t id) const;
  bool IsIntCooperativeMatrixType(uint32_t id) const;
  bool IsUnsignedIntCooperativeMatrixType(uint32_t id) const;
  bool IsUnsigned64BitHandle(uint32_t id) const;

  // Returns true if |id| is a type id that contains |type| (or integer or
  // floating point type) of |width| bits.
  bool ContainsSizedIntOrFloatType(uint32_t id, spv::Op type,
                                   uint32_t width) const;
  // Returns true if |id| is a type id that contains a 8- or 16-bit int or
  // 16-bit float that is not generally enabled for use.
  bool ContainsLimitedUseIntOrFloatType(uint32_t id) const;

  // Returns true if |id| is a type that contains a runtime-sized array.
  // Does not consider a pointers as contains the array.
  bool ContainsRuntimeArray(uint32_t id) const;

  // Generic type traversal.
  // Only traverse pointers and functions if |traverse_all_types| is true.
  // Recursively tests |f| against the type hierarchy headed by |id|.
  bool ContainsType(uint32_t id,
                    const std::function<bool(const Instruction*)>& f,
                    bool traverse_all_types = true) const;

  // Gets value from OpConstant and OpSpecConstant as uint64.
  // Returns false on failure (no instruction, wrong instruction, not int).
  bool GetConstantValUint64(uint32_t id, uint64_t* val) const;

  // Returns type_id if id has type or zero otherwise.
  uint32_t GetTypeId(uint32_t id) const;

  // Returns opcode of the instruction which issued the id or OpNop if the
  // instruction is not registered.
  spv::Op GetIdOpcode(uint32_t id) const;

  // Returns type_id for given id operand if it has a type or zero otherwise.
  // |operand_index| is expected to be pointing towards an operand which is an
  // id.
  uint32_t GetOperandTypeId(const Instruction* inst,
                            size_t operand_index) const;

  // Provides information on pointer type. Returns false iff not pointer type.
  bool GetPointerTypeInfo(uint32_t id, uint32_t* data_type,
                          spv::StorageClass* storage_class) const;

  // Is the ID the type of a pointer to a uniform block: Block-decorated struct
  // in uniform storage class? The result is only valid after internal method
  // CheckDecorationsOfBuffers has been called.
  bool IsPointerToUniformBlock(uint32_t type_id) const {
    return pointer_to_uniform_block_.find(type_id) !=
           pointer_to_uniform_block_.cend();
  }
  // Save the ID of a pointer to uniform block.
  void RegisterPointerToUniformBlock(uint32_t type_id) {
    pointer_to_uniform_block_.insert(type_id);
  }
  // Is the ID the type of a struct used as a uniform block?
  // The result is only valid after internal method CheckDecorationsOfBuffers
  // has been called.
  bool IsStructForUniformBlock(uint32_t type_id) const {
    return struct_for_uniform_block_.find(type_id) !=
           struct_for_uniform_block_.cend();
  }
  // Save the ID of a struct of a uniform block.
  void RegisterStructForUniformBlock(uint32_t type_id) {
    struct_for_uniform_block_.insert(type_id);
  }
  // Is the ID the type of a pointer to a storage buffer: BufferBlock-decorated
  // struct in uniform storage class, or Block-decorated struct in StorageBuffer
  // storage class? The result is only valid after internal method
  // CheckDecorationsOfBuffers has been called.
  bool IsPointerToStorageBuffer(uint32_t type_id) const {
    return pointer_to_storage_buffer_.find(type_id) !=
           pointer_to_storage_buffer_.cend();
  }
  // Save the ID of a pointer to a storage buffer.
  void RegisterPointerToStorageBuffer(uint32_t type_id) {
    pointer_to_storage_buffer_.insert(type_id);
  }
  // Is the ID the type of a struct for storage buffer?
  // The result is only valid after internal method CheckDecorationsOfBuffers
  // has been called.
  bool IsStructForStorageBuffer(uint32_t type_id) const {
    return struct_for_storage_buffer_.find(type_id) !=
           struct_for_storage_buffer_.cend();
  }
  // Save the ID of a struct of a storage buffer.
  void RegisterStructForStorageBuffer(uint32_t type_id) {
    struct_for_storage_buffer_.insert(type_id);
  }

  // Is the ID the type of a pointer to a storage image?  That is, the pointee
  // type is an image type which is known to not use a sampler.
  bool IsPointerToStorageImage(uint32_t type_id) const {
    return pointer_to_storage_image_.find(type_id) !=
           pointer_to_storage_image_.cend();
  }
  // Save the ID of a pointer to a storage image.
  void RegisterPointerToStorageImage(uint32_t type_id) {
    pointer_to_storage_image_.insert(type_id);
  }

  // Tries to evaluate a 32-bit signed or unsigned scalar integer constant.
  // Returns tuple <is_int32, is_const_int32, value>.
  // OpSpecConstant* return |is_const_int32| as false since their values cannot
  // be relied upon during validation.
  std::tuple<bool, bool, uint32_t> EvalInt32IfConst(uint32_t id) const;

  // Returns the disassembly string for the given instruction.
  std::string Disassemble(const Instruction& inst) const;

  // Returns the disassembly string for the given instruction.
  std::string Disassemble(const uint32_t* words, uint16_t num_words) const;

  // Returns the string name for |decoration|.
  std::string SpvDecorationString(uint32_t decoration) {
    spv_operand_desc desc = nullptr;
    if (grammar_.lookupOperand(SPV_OPERAND_TYPE_DECORATION, decoration,
                               &desc) != SPV_SUCCESS) {
      return std::string("Unknown");
    }
    return std::string(desc->name);
  }
  std::string SpvDecorationString(spv::Decoration decoration) {
    return SpvDecorationString(uint32_t(decoration));
  }

  // Returns whether type m1 and type m2 are cooperative matrices with
  // the same "shape" (matching scope, rows, cols). If any are specialization
  // constants, we assume they can match because we can't prove they don't.
  spv_result_t CooperativeMatrixShapesMatch(const Instruction* inst,
                                            uint32_t m1, uint32_t m2);

  // Returns true if |lhs| and |rhs| logically match and, if the decorations of
  // |rhs| are a subset of |lhs|.
  //
  // 1. Must both be either OpTypeArray or OpTypeStruct
  // 2. If OpTypeArray, then
  //  * Length must be the same
  //  * Element type must match or logically match
  // 3. If OpTypeStruct, then
  //  * Both have same number of elements
  //  * Element N for both structs must match or logically match
  //
  // If |check_decorations| is false, then the decorations are not checked.
  bool LogicallyMatch(const Instruction* lhs, const Instruction* rhs,
                      bool check_decorations);

  // Traces |inst| to find a single base pointer. Returns the base pointer.
  // Will trace through the following instructions:
  // * OpAccessChain
  // * OpInBoundsAccessChain
  // * OpPtrAccessChain
  // * OpInBoundsPtrAccessChain
  // * OpCopyObject
  const Instruction* TracePointer(const Instruction* inst) const;

  // Validates the storage class for the target environment.
  bool IsValidStorageClass(spv::StorageClass storage_class) const;

  // Takes a Vulkan Valid Usage ID (VUID) as |id| and optional |reference| and
  // will return a non-empty string only if ID is known and targeting Vulkan.
  // VUIDs are found in the Vulkan-Docs repo in the form "[[VUID-ref-ref-id]]"
  // where "id" is always an 5 char long number (with zeros padding) and matches
  // to |id|. |reference| is used if there is a "common validity" and the VUID
  // shares the same |id| value.
  //
  // More details about Vulkan validation can be found in Vulkan Guide:
  // https://github.com/KhronosGroup/Vulkan-Guide/blob/master/chapters/validation_overview.md
  std::string VkErrorID(uint32_t id, const char* reference = nullptr) const;

  // Testing method to allow setting the current layout section.
  void SetCurrentLayoutSectionForTesting(ModuleLayoutSection section) {
    current_layout_section_ = section;
  }

 private:
  ValidationState_t(const ValidationState_t&);

  const spv_const_context context_;

  /// Stores the Validator command line options. Must be a valid options object.
  const spv_const_validator_options options_;

  /// The SPIR-V binary module we're validating.
  const uint32_t* words_;
  const size_t num_words_;

  /// The generator of the SPIR-V.
  uint32_t generator_ = 0;

  /// The version of the SPIR-V.
  uint32_t version_ = 0;

  /// The total number of instructions in the binary.
  size_t total_instructions_ = 0;
  /// The total number of functions in the binary.
  size_t total_functions_ = 0;

  /// IDs which have been forward declared but have not been defined
  std::unordered_set<uint32_t> unresolved_forward_ids_;

  /// IDs that have been declared as forward pointers.
  std::unordered_set<uint32_t> forward_pointer_ids_;

  /// Stores a vector of instructions that use the result of a given
  /// OpSampledImage instruction.
  std::unordered_map<uint32_t, std::vector<Instruction*>>
      sampled_image_consumers_;

  /// A map of operand IDs and their names defined by the OpName instruction
  std::unordered_map<uint32_t, std::string> operand_names_;

  /// The section of the code being processed
  ModuleLayoutSection current_layout_section_;

  /// A list of functions in the module.
  /// Pointers to objects in this container are guaranteed to be stable and
  /// valid until the end of lifetime of the validation state.
  std::vector<Function> module_functions_;

  /// Capabilities declared in the module
  CapabilitySet module_capabilities_;

  /// Extensions declared in the module
  ExtensionSet module_extensions_;

  /// List of all instructions in the order they appear in the binary
  std::vector<Instruction> ordered_instructions_;

  /// Instructions that can be referenced by Ids
  std::unordered_map<uint32_t, Instruction*> all_definitions_;

  /// IDs that are entry points, ie, arguments to OpEntryPoint.
  std::vector<uint32_t> entry_points_;

  /// Maps an entry point id to its descriptions.
  std::unordered_map<uint32_t, std::vector<EntryPointDescription>>
      entry_point_descriptions_;

  /// IDs that are entry points, ie, arguments to OpEntryPoint, and root a call
  /// graph that recurses.
  std::set<uint32_t> recursive_entry_points_;

  /// Functions IDs that are target of OpFunctionCall.
  std::unordered_set<uint32_t> function_call_targets_;

  /// ID Bound from the Header
  uint32_t id_bound_;

  /// Set of Global Variable IDs (Storage Class other than 'Function')
  std::unordered_set<uint32_t> global_vars_;

  /// Set of Local Variable IDs ('Function' Storage Class)
  std::unordered_set<uint32_t> local_vars_;

  /// Set of struct types that have members with a BuiltIn decoration.
  std::unordered_set<uint32_t> builtin_structs_;

  /// Structure Nesting Depth
  std::unordered_map<uint32_t, uint32_t> struct_nesting_depth_;

  /// Structure has nested blockorbufferblock struct
  std::unordered_map<uint32_t, bool>
      struct_has_nested_blockorbufferblock_struct_;

  /// Stores the list of decorations for a given <id>
  std::map<uint32_t, std::set<Decoration>> id_decorations_;

  /// Stores type declarations which need to be unique (i.e. non-aggregates),
  /// in the form [opcode, operand words], result_id is not stored.
  /// Using ordered set to avoid the need for a vector hash function.
  /// The size of this container is expected not to exceed double-digits.
  std::set<std::vector<uint32_t>> unique_type_declarations_;

  AssemblyGrammar grammar_;

  spv::AddressingModel addressing_model_;
  spv::MemoryModel memory_model_;
  // pointer size derived from addressing model. Assumes all storage classes
  // have the same pointer size (for physical pointer types).
  uint32_t pointer_size_and_alignment_;

  /// bit width of sampler/image type variables. Valid values are 32 and 64
  uint32_t sampler_image_addressing_mode_;

  /// NOTE: See correspoding getter functions
  bool in_function_;

  /// The state of optional features.  These are determined by capabilities
  /// declared by the module and the environment.
  Feature features_;

  /// Maps function ids to function stat objects.
  std::unordered_map<uint32_t, Function*> id_to_function_;

  /// Mapping entry point -> execution models. It is presumed that the same
  /// function could theoretically be used as 'main' by multiple OpEntryPoint
  /// instructions.
  std::unordered_map<uint32_t, std::set<spv::ExecutionModel>>
      entry_point_to_execution_models_;

  /// Mapping entry point -> execution modes.
  std::unordered_map<uint32_t, std::set<spv::ExecutionMode>>
      entry_point_to_execution_modes_;

  /// Mapping function -> array of entry points inside this
  /// module which can (indirectly) call the function.
  std::unordered_map<uint32_t, std::vector<uint32_t>> function_to_entry_points_;
  const std::vector<uint32_t> empty_ids_;

  // The IDs of types of pointers to Block-decorated structs in Uniform storage
  // class. This is populated at the start of ValidateDecorations.
  std::unordered_set<uint32_t> pointer_to_uniform_block_;
  // The IDs of struct types for uniform blocks.
  // This is populated at the start of ValidateDecorations.
  std::unordered_set<uint32_t> struct_for_uniform_block_;
  // The IDs of types of pointers to BufferBlock-decorated structs in Uniform
  // storage class, or Block-decorated structs in StorageBuffer storage class.
  // This is populated at the start of ValidateDecorations.
  std::unordered_set<uint32_t> pointer_to_storage_buffer_;
  // The IDs of struct types for storage buffers.
  // This is populated at the start of ValidateDecorations.
  std::unordered_set<uint32_t> struct_for_storage_buffer_;
  // The IDs of types of pointers to storage images.  This is populated in the
  // TypePass.
  std::unordered_set<uint32_t> pointer_to_storage_image_;

  /// Maps ids to friendly names.
  std::unique_ptr<spvtools::FriendlyNameMapper> friendly_mapper_;
  spvtools::NameMapper name_mapper_;

  /// Variables used to reduce the number of diagnostic messages.
  uint32_t num_of_warnings_;
  uint32_t max_num_of_warnings_;
};

}  // namespace val
}  // namespace spvtools

#endif  // SOURCE_VAL_VALIDATION_STATE_H_
