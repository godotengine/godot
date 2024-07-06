// Copyright (c) 2022 Google LLC.
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

#include "source/diff/diff.h"

#include "source/diff/lcs.h"
#include "source/disassemble.h"
#include "source/ext_inst.h"
#include "source/latest_version_spirv_header.h"
#include "source/print.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace diff {

namespace {

// A map from an id to the instruction that defines it.
using IdToInstructionMap = std::vector<const opt::Instruction*>;
// A map from an id to the instructions that decorate it, or name it, etc.
using IdToInfoMap = std::vector<std::vector<const opt::Instruction*>>;
// A map from an instruction to another, used for instructions without id.
using InstructionToInstructionMap =
    std::unordered_map<const opt::Instruction*, const opt::Instruction*>;
// A flat list of instructions in a function for easier iteration.
using InstructionList = std::vector<const opt::Instruction*>;
// A map from a function to its list of instructions.
using FunctionInstMap = std::map<uint32_t, InstructionList>;
// A list of ids with some similar property, for example functions with the same
// name.
using IdGroup = std::vector<uint32_t>;
// A map of names to ids with the same name.  This is an ordered map so
// different implementations produce identical results.
using IdGroupMapByName = std::map<std::string, IdGroup>;
using IdGroupMapByTypeId = std::map<uint32_t, IdGroup>;
using IdGroupMapByOp = std::map<spv::Op, IdGroup>;
using IdGroupMapByStorageClass = std::map<spv::StorageClass, IdGroup>;

// A set of potential id mappings that haven't been resolved yet.  Any id in src
// may map in any id in dst.  Note that ids are added in the same order as they
// appear in src and dst to facilitate matching dependent instructions.  For
// example, this guarantees that when matching OpTypeVector, the basic type of
// the vector is already (potentially) matched.
struct PotentialIdMap {
  std::vector<uint32_t> src_ids;
  std::vector<uint32_t> dst_ids;
};

void CompactIds(std::vector<uint32_t>& ids) {
  size_t write_index = 0;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] != 0) {
      ids[write_index++] = ids[i];
    }
  }
  ids.resize(write_index);
}

// A mapping between src and dst ids.
class IdMap {
 public:
  IdMap(size_t id_bound) { id_map_.resize(id_bound, 0); }

  void MapIds(uint32_t from, uint32_t to) {
    assert(from != 0);
    assert(to != 0);
    assert(from < id_map_.size());
    assert(id_map_[from] == 0);

    id_map_[from] = to;
  }

  uint32_t MappedId(uint32_t from) const {
    assert(from != 0);
    return from < id_map_.size() ? id_map_[from] : 0;
  }
  const opt::Instruction* MappedInst(const opt::Instruction* from_inst) const {
    assert(from_inst != nullptr);
    assert(!from_inst->HasResultId());

    auto mapped = inst_map_.find(from_inst);
    if (mapped == inst_map_.end()) {
      return nullptr;
    }
    return mapped->second;
  }

  bool IsMapped(uint32_t from) const {
    assert(from != 0);
    return from < id_map_.size() && id_map_[from] != 0;
  }

  bool IsMapped(const opt::Instruction* from_inst) const {
    assert(from_inst != nullptr);
    assert(!from_inst->HasResultId());

    return inst_map_.find(from_inst) != inst_map_.end();
  }

  // Some instructions don't have result ids.  Those are mapped by pointer.
  void MapInsts(const opt::Instruction* from_inst,
                const opt::Instruction* to_inst) {
    assert(from_inst != nullptr);
    assert(to_inst != nullptr);
    assert(inst_map_.find(from_inst) == inst_map_.end());

    inst_map_[from_inst] = to_inst;
  }

  uint32_t IdBound() const { return static_cast<uint32_t>(id_map_.size()); }

  // Generate a fresh id in this mapping's domain.
  uint32_t MakeFreshId() {
    id_map_.push_back(0);
    return static_cast<uint32_t>(id_map_.size()) - 1;
  }

 private:
  // Given an id, returns the corresponding id in the other module, or 0 if not
  // matched yet.
  std::vector<uint32_t> id_map_;

  // Same for instructions that don't have an id.
  InstructionToInstructionMap inst_map_;
};

// Two way mapping of ids.
class SrcDstIdMap {
 public:
  SrcDstIdMap(size_t src_id_bound, size_t dst_id_bound)
      : src_to_dst_(src_id_bound), dst_to_src_(dst_id_bound) {}

  void MapIds(uint32_t src, uint32_t dst) {
    src_to_dst_.MapIds(src, dst);
    dst_to_src_.MapIds(dst, src);
  }

  uint32_t MappedDstId(uint32_t src) {
    uint32_t dst = src_to_dst_.MappedId(src);
    assert(dst == 0 || dst_to_src_.MappedId(dst) == src);
    return dst;
  }
  uint32_t MappedSrcId(uint32_t dst) {
    uint32_t src = dst_to_src_.MappedId(dst);
    assert(src == 0 || src_to_dst_.MappedId(src) == dst);
    return src;
  }

  bool IsSrcMapped(uint32_t src) { return src_to_dst_.IsMapped(src); }
  bool IsDstMapped(uint32_t dst) { return dst_to_src_.IsMapped(dst); }
  bool IsDstMapped(const opt::Instruction* dst_inst) {
    return dst_to_src_.IsMapped(dst_inst);
  }

  // Map any ids in src and dst that have not been mapped to new ids in dst and
  // src respectively. Use src_insn_defined and dst_insn_defined to ignore ids
  // that are simply never defined. (Since we assume the inputs are valid
  // SPIR-V, this implies they are also never used.)
  void MapUnmatchedIds(std::function<bool(uint32_t)> src_insn_defined,
                       std::function<bool(uint32_t)> dst_insn_defined);

  // Some instructions don't have result ids.  Those are mapped by pointer.
  void MapInsts(const opt::Instruction* src_inst,
                const opt::Instruction* dst_inst) {
    assert(src_inst->HasResultId() == dst_inst->HasResultId());
    if (src_inst->HasResultId()) {
      MapIds(src_inst->result_id(), dst_inst->result_id());
    } else {
      src_to_dst_.MapInsts(src_inst, dst_inst);
      dst_to_src_.MapInsts(dst_inst, src_inst);
    }
  }

  const IdMap& SrcToDstMap() const { return src_to_dst_; }
  const IdMap& DstToSrcMap() const { return dst_to_src_; }

 private:
  IdMap src_to_dst_;
  IdMap dst_to_src_;
};

struct IdInstructions {
  IdInstructions(const opt::Module* module)
      : inst_map_(module->IdBound(), nullptr),
        name_map_(module->IdBound()),
        decoration_map_(module->IdBound()),
        forward_pointer_map_(module->IdBound()) {
    // Map ids from all sections to instructions that define them.
    MapIdsToInstruction(module->ext_inst_imports());
    MapIdsToInstruction(module->types_values());
    for (const opt::Function& function : *module) {
      function.ForEachInst(
          [this](const opt::Instruction* inst) {
            if (inst->HasResultId()) {
              MapIdToInstruction(inst->result_id(), inst);
            }
          },
          true, true);
    }

    // Gather decorations applied to ids that could be useful in matching them
    // between src and dst modules.
    MapIdsToInfos(module->debugs2());
    MapIdsToInfos(module->annotations());
    MapIdsToInfos(module->types_values());
  }

  void MapIdToInstruction(uint32_t id, const opt::Instruction* inst);

  // Return true if id is mapped to any instruction, false otherwise.
  bool IsDefined(uint32_t id) {
    return id < inst_map_.size() && inst_map_[id] != nullptr;
  }

  void MapIdsToInstruction(
      opt::IteratorRange<opt::Module::const_inst_iterator> section);
  void MapIdsToInfos(
      opt::IteratorRange<opt::Module::const_inst_iterator> section);

  IdToInstructionMap inst_map_;
  IdToInfoMap name_map_;
  IdToInfoMap decoration_map_;
  IdToInstructionMap forward_pointer_map_;
};

class Differ {
 public:
  Differ(opt::IRContext* src, opt::IRContext* dst, std::ostream& out,
         Options options)
      : src_context_(src),
        dst_context_(dst),
        src_(src->module()),
        dst_(dst->module()),
        options_(options),
        out_(out),
        src_id_to_(src_),
        dst_id_to_(dst_),
        id_map_(src_->IdBound(), dst_->IdBound()) {
    // Cache function bodies in canonicalization order.
    GetFunctionBodies(src_context_, &src_funcs_, &src_func_insts_);
    GetFunctionBodies(dst_context_, &dst_funcs_, &dst_func_insts_);
  }

  // Match ids or instructions of different sections.
  void MatchCapabilities();
  void MatchExtensions();
  void MatchExtInstImportIds();
  void MatchMemoryModel();
  void MatchEntryPointIds();
  void MatchExecutionModes();
  void MatchTypeForwardPointers();
  void MatchTypeIds();
  void MatchConstants();
  void MatchVariableIds();
  void MatchFunctions();

  // Debug info and annotations are matched only after ids are matched.
  void MatchDebugs1();
  void MatchDebugs2();
  void MatchDebugs3();
  void MatchExtInstDebugInfo();
  void MatchAnnotations();

  // Output the diff.
  spv_result_t Output();

  void DumpIdMap() {
    if (!options_.dump_id_map) {
      return;
    }

    out_ << " Src ->  Dst\n";
    for (uint32_t src_id = 1; src_id < src_->IdBound(); ++src_id) {
      uint32_t dst_id = id_map_.MappedDstId(src_id);
      if (src_id_to_.inst_map_[src_id] != nullptr && dst_id != 0)
        out_ << std::setw(4) << src_id << " -> " << std::setw(4) << dst_id
             << " [" << spvOpcodeString(src_id_to_.inst_map_[src_id]->opcode())
             << "]\n";
    }
  }

 private:
  // Helper functions that match ids between src and dst
  void PoolPotentialIds(
      opt::IteratorRange<opt::Module::const_inst_iterator> section,
      std::vector<uint32_t>& ids, bool is_src,
      std::function<bool(const opt::Instruction&)> filter,
      std::function<uint32_t(const opt::Instruction&)> get_id);
  void MatchIds(
      PotentialIdMap& potential,
      std::function<bool(const opt::Instruction*, const opt::Instruction*)>
          match);
  // Helper functions that match id-less instructions between src and dst.
  void MatchPreambleInstructions(
      opt::IteratorRange<opt::Module::const_inst_iterator> src_insts,
      opt::IteratorRange<opt::Module::const_inst_iterator> dst_insts);
  InstructionList SortPreambleInstructions(
      const opt::Module* module,
      opt::IteratorRange<opt::Module::const_inst_iterator> insts);
  int ComparePreambleInstructions(const opt::Instruction* a,
                                  const opt::Instruction* b,
                                  const opt::Module* src_inst_module,
                                  const opt::Module* dst_inst_module);
  // Helper functions that match debug and annotation instructions of already
  // matched ids.
  void MatchDebugAndAnnotationInstructions(
      opt::IteratorRange<opt::Module::const_inst_iterator> src_insts,
      opt::IteratorRange<opt::Module::const_inst_iterator> dst_insts);

  // Get various properties from an id.  These Helper functions are passed to
  // `GroupIds` and `GroupIdsAndMatch` below (as the `get_group` argument).
  uint32_t GroupIdsHelperGetTypeId(const IdInstructions& id_to, uint32_t id);
  spv::StorageClass GroupIdsHelperGetTypePointerStorageClass(
      const IdInstructions& id_to, uint32_t id);
  spv::Op GroupIdsHelperGetTypePointerTypeOp(const IdInstructions& id_to,
                                             uint32_t id);

  // Given a list of ids, groups them based on some value.  The `get_group`
  // function extracts a piece of information corresponding to each id, and the
  // ids are bucketed based on that (and output in `groups`).  This is useful to
  // attempt to match ids between src and dst only when said property is
  // identical.
  template <typename T>
  void GroupIds(const IdGroup& ids, bool is_src, std::map<T, IdGroup>* groups,
                T (Differ::*get_group)(const IdInstructions&, uint32_t));

  // Calls GroupIds to bucket ids in src and dst based on a property returned by
  // `get_group`.  This function then calls `match_group` for each bucket (i.e.
  // "group") with identical values for said property.
  //
  // For example, say src and dst ids have the following properties
  // correspondingly:
  //
  // - src ids' properties: {id0: A, id1: A, id2: B, id3: C, id4: B}
  // - dst ids' properties: {id0': B, id1': C, id2': B, id3': D, id4': B}
  //
  // Then `match_group` is called 2 times:
  //
  // - Once with: ([id2, id4], [id0', id2', id4']) corresponding to B
  // - Once with: ([id3], [id2']) corresponding to C
  //
  // Ids corresponding to A and D cannot match based on this property.
  template <typename T>
  void GroupIdsAndMatch(
      const IdGroup& src_ids, const IdGroup& dst_ids, T invalid_group_key,
      T (Differ::*get_group)(const IdInstructions&, uint32_t),
      std::function<void(const IdGroup& src_group, const IdGroup& dst_group)>
          match_group);

  // Bucket `src_ids` and `dst_ids` by the key ids returned by `get_group`, and
  // then call `match_group` on pairs of buckets whose key ids are matched with
  // each other.
  //
  // For example, suppose we want to pair up groups of instructions with the
  // same type. Naturally, the source instructions refer to their types by their
  // ids in the source, and the destination instructions use destination type
  // ids, so simply comparing source and destination type ids as integers, as
  // `GroupIdsAndMatch` would do, is meaningless. But if a prior call to
  // `MatchTypeIds` has established type matches between the two modules, then
  // we can consult those to pair source and destination buckets whose types are
  // equivalent.
  //
  // Suppose our input groups are as follows:
  //
  // - src_ids: { 1 -> 100, 2 -> 300, 3 -> 100, 4 -> 200 }
  // - dst_ids: { 5 -> 10, 6 -> 20, 7 -> 10, 8 -> 300 }
  //
  // Here, `X -> Y` means that the instruction with SPIR-V id `X` is a member of
  // the group, and `Y` is the id of its type. If we use
  // `Differ::GroupIdsHelperGetTypeId` for `get_group`, then
  // `get_group(X) == Y`.
  //
  // These instructions are bucketed by type as follows:
  //
  // - source:      [1, 3] -> 100
  //                [4] -> 200
  //                [2] -> 300
  //
  // - destination: [5, 7] -> 10
  //                [6] -> 20
  //                [8] -> 300
  //
  // Now suppose that we have previously matched up src type 100 with dst type
  // 10, and src type 200 with dst type 20, but no other types are matched.
  //
  // Then `match_group` is called twice:
  // - Once with ([1,3], [5, 7]), corresponding to 100/10
  // - Once with ([4],[6]), corresponding to 200/20
  //
  // The source type 300 isn't matched with anything, so the fact that there's a
  // destination type 300 is irrelevant, and thus 2 and 8 are never passed to
  // `match_group`.
  //
  // This function isn't specific to types; it simply buckets by the ids
  // returned from `get_group`, and consults existing matches to pair up the
  // resulting buckets.
  void GroupIdsAndMatchByMappedId(
      const IdGroup& src_ids, const IdGroup& dst_ids,
      uint32_t (Differ::*get_group)(const IdInstructions&, uint32_t),
      std::function<void(const IdGroup& src_group, const IdGroup& dst_group)>
          match_group);

  // Helper functions that determine if two instructions match
  bool DoIdsMatch(uint32_t src_id, uint32_t dst_id);
  bool DoesOperandMatch(const opt::Operand& src_operand,
                        const opt::Operand& dst_operand);
  bool DoOperandsMatch(const opt::Instruction* src_inst,
                       const opt::Instruction* dst_inst,
                       uint32_t in_operand_index_start,
                       uint32_t in_operand_count);
  bool DoInstructionsMatch(const opt::Instruction* src_inst,
                           const opt::Instruction* dst_inst);
  bool DoIdsMatchFuzzy(uint32_t src_id, uint32_t dst_id);
  bool DoesOperandMatchFuzzy(const opt::Operand& src_operand,
                             const opt::Operand& dst_operand);
  bool DoInstructionsMatchFuzzy(const opt::Instruction* src_inst,
                                const opt::Instruction* dst_inst);
  bool AreIdenticalUintConstants(uint32_t src_id, uint32_t dst_id);
  bool DoDebugAndAnnotationInstructionsMatch(const opt::Instruction* src_inst,
                                             const opt::Instruction* dst_inst);
  bool AreVariablesMatchable(uint32_t src_id, uint32_t dst_id,
                             uint32_t flexibility);
  bool MatchOpTypeStruct(const opt::Instruction* src_inst,
                         const opt::Instruction* dst_inst,
                         uint32_t flexibility);
  bool MatchOpConstant(const opt::Instruction* src_inst,
                       const opt::Instruction* dst_inst, uint32_t flexibility);
  bool MatchOpSpecConstant(const opt::Instruction* src_inst,
                           const opt::Instruction* dst_inst);
  bool MatchOpVariable(const opt::Instruction* src_inst,
                       const opt::Instruction* dst_inst, uint32_t flexibility);
  bool MatchPerVertexType(uint32_t src_type_id, uint32_t dst_type_id);
  bool MatchPerVertexVariable(const opt::Instruction* src_inst,
                              const opt::Instruction* dst_inst);

  // Helper functions for matching OpTypeForwardPointer
  void MatchTypeForwardPointersByName(const IdGroup& src, const IdGroup& dst);
  void MatchTypeForwardPointersByTypeOp(const IdGroup& src, const IdGroup& dst);

  // Helper functions for function matching.
  using FunctionMap = std::map<uint32_t, const opt::Function*>;

  InstructionList GetFunctionBody(opt::IRContext* context,
                                  opt::Function& function);
  InstructionList GetFunctionHeader(const opt::Function& function);
  void GetFunctionBodies(opt::IRContext* context, FunctionMap* functions,
                         FunctionInstMap* function_insts);
  void GetFunctionHeaderInstructions(const opt::Module* module,
                                     FunctionInstMap* function_insts);
  void BestEffortMatchFunctions(const IdGroup& src_func_ids,
                                const IdGroup& dst_func_ids,
                                const FunctionInstMap& src_func_insts,
                                const FunctionInstMap& dst_func_insts);

  // Calculates the diff of two function bodies.  Note that the matched
  // instructions themselves may not be identical; output of exact matches
  // should produce the exact instruction while inexact matches should produce a
  // diff as well.
  //
  // Returns the similarity of the two bodies = 2*N_match / (N_src + N_dst)
  void MatchFunctionParamIds(const opt::Function* src_func,
                             const opt::Function* dst_func);
  float MatchFunctionBodies(const InstructionList& src_body,
                            const InstructionList& dst_body,
                            DiffMatch* src_match_result,
                            DiffMatch* dst_match_result);
  void MatchIdsInFunctionBodies(const InstructionList& src_body,
                                const InstructionList& dst_body,
                                const DiffMatch& src_match_result,
                                const DiffMatch& dst_match_result,
                                uint32_t flexibility);
  void MatchVariablesUsedByMatchedInstructions(const opt::Instruction* src_inst,
                                               const opt::Instruction* dst_inst,
                                               uint32_t flexibility);

  // Helper functions to retrieve information pertaining to an id
  const opt::Instruction* GetInst(const IdInstructions& id_to, uint32_t id);
  uint32_t GetConstantUint(const IdInstructions& id_to, uint32_t constant_id);
  spv::ExecutionModel GetExecutionModel(const opt::Module* module,
                                        uint32_t entry_point_id);
  bool HasName(const IdInstructions& id_to, uint32_t id);
  // Get the OpName associated with an id
  std::string GetName(const IdInstructions& id_to, uint32_t id, bool* has_name);
  // Get the OpName associated with an id, with argument types stripped for
  // functions.  Some tools don't encode function argument types in the OpName
  // string, and this improves diff between SPIR-V from those tools and others.
  std::string GetSanitizedName(const IdInstructions& id_to, uint32_t id);
  uint32_t GetVarTypeId(const IdInstructions& id_to, uint32_t var_id,
                        spv::StorageClass* storage_class);
  bool GetDecorationValue(const IdInstructions& id_to, uint32_t id,
                          spv::Decoration decoration,
                          uint32_t* decoration_value);
  const opt::Instruction* GetForwardPointerInst(const IdInstructions& id_to,
                                                uint32_t id);
  bool IsIntType(const IdInstructions& id_to, uint32_t type_id);
  bool IsFloatType(const IdInstructions& id_to, uint32_t type_id);
  bool IsConstantUint(const IdInstructions& id_to, uint32_t id);
  bool IsVariable(const IdInstructions& id_to, uint32_t pointer_id);
  bool IsOp(const IdInstructions& id_to, uint32_t id, spv::Op opcode);
  bool IsPerVertexType(const IdInstructions& id_to, uint32_t type_id);
  bool IsPerVertexVariable(const IdInstructions& id_to, uint32_t type_id);
  spv::StorageClass GetPerVertexStorageClass(const opt::Module* module,
                                             uint32_t type_id);
  spv_ext_inst_type_t GetExtInstType(const IdInstructions& id_to,
                                     uint32_t set_id);
  spv_number_kind_t GetNumberKind(const IdInstructions& id_to,
                                  const opt::Instruction& inst,
                                  uint32_t operand_index,
                                  uint32_t* number_bit_width);
  spv_number_kind_t GetTypeNumberKind(const IdInstructions& id_to, uint32_t id,
                                      uint32_t* number_bit_width);

  // Helper functions to output a diff line
  const opt::Instruction* MappedDstInst(const opt::Instruction* src_inst);
  const opt::Instruction* MappedSrcInst(const opt::Instruction* dst_inst);
  const opt::Instruction* MappedInstImpl(const opt::Instruction* inst,
                                         const IdMap& to_other,
                                         const IdInstructions& other_id_to);
  void OutputLine(std::function<bool()> are_lines_identical,
                  std::function<void()> output_src_line,
                  std::function<void()> output_dst_line);
  template <typename InstList>
  void OutputSection(
      const InstList& src_insts, const InstList& dst_insts,
      std::function<void(const opt::Instruction&, const IdInstructions&,
                         const opt::Instruction&)>
          write_inst);
  void ToParsedInstruction(const opt::Instruction& inst,
                           const IdInstructions& id_to,
                           const opt::Instruction& original_inst,
                           spv_parsed_instruction_t* parsed_inst,
                           std::vector<spv_parsed_operand_t>& parsed_operands,
                           std::vector<uint32_t>& inst_binary);
  opt::Instruction ToMappedSrcIds(const opt::Instruction& dst_inst);

  void OutputRed() {
    if (options_.color_output) out_ << spvtools::clr::red{true};
  }
  void OutputGreen() {
    if (options_.color_output) out_ << spvtools::clr::green{true};
  }
  void OutputResetColor() {
    if (options_.color_output) out_ << spvtools::clr::reset{true};
  }

  opt::IRContext* src_context_;
  opt::IRContext* dst_context_;
  const opt::Module* src_;
  const opt::Module* dst_;
  Options options_;
  std::ostream& out_;

  // Helpers to look up instructions based on id.
  IdInstructions src_id_to_;
  IdInstructions dst_id_to_;

  // The ids that have been matched between src and dst so far.
  SrcDstIdMap id_map_;

  // List of instructions in function bodies after canonicalization.  Cached
  // here to avoid duplicate work.  More importantly, some maps use
  // opt::Instruction pointers so they need to be unique.
  FunctionInstMap src_func_insts_;
  FunctionInstMap dst_func_insts_;
  FunctionMap src_funcs_;
  FunctionMap dst_funcs_;
};

void SrcDstIdMap::MapUnmatchedIds(
    std::function<bool(uint32_t)> src_insn_defined,
    std::function<bool(uint32_t)> dst_insn_defined) {
  const uint32_t src_id_bound = static_cast<uint32_t>(src_to_dst_.IdBound());
  const uint32_t dst_id_bound = static_cast<uint32_t>(dst_to_src_.IdBound());

  for (uint32_t src_id = 1; src_id < src_id_bound; ++src_id) {
    if (!src_to_dst_.IsMapped(src_id) && src_insn_defined(src_id)) {
      uint32_t fresh_dst_id = dst_to_src_.MakeFreshId();
      MapIds(src_id, fresh_dst_id);
    }
  }

  for (uint32_t dst_id = 1; dst_id < dst_id_bound; ++dst_id) {
    if (!dst_to_src_.IsMapped(dst_id) && dst_insn_defined(dst_id)) {
      uint32_t fresh_src_id = src_to_dst_.MakeFreshId();
      MapIds(fresh_src_id, dst_id);
    }
  }
}

void IdInstructions::MapIdToInstruction(uint32_t id,
                                        const opt::Instruction* inst) {
  assert(id != 0);
  assert(id < inst_map_.size());
  assert(inst_map_[id] == nullptr);

  inst_map_[id] = inst;
}

void IdInstructions::MapIdsToInstruction(
    opt::IteratorRange<opt::Module::const_inst_iterator> section) {
  for (const opt::Instruction& inst : section) {
    uint32_t result_id = inst.result_id();
    if (result_id == 0) {
      continue;
    }

    MapIdToInstruction(result_id, &inst);
  }
}

void IdInstructions::MapIdsToInfos(
    opt::IteratorRange<opt::Module::const_inst_iterator> section) {
  for (const opt::Instruction& inst : section) {
    IdToInfoMap* info_map = nullptr;
    uint32_t id_operand = 0;

    switch (inst.opcode()) {
      case spv::Op::OpName:
        info_map = &name_map_;
        break;
      case spv::Op::OpMemberName:
        info_map = &name_map_;
        break;
      case spv::Op::OpDecorate:
        info_map = &decoration_map_;
        break;
      case spv::Op::OpMemberDecorate:
        info_map = &decoration_map_;
        break;
      case spv::Op::OpTypeForwardPointer: {
        uint32_t id = inst.GetSingleWordOperand(0);
        assert(id != 0);

        assert(id < forward_pointer_map_.size());
        forward_pointer_map_[id] = &inst;
        continue;
      }
      default:
        // Currently unsupported instruction, don't attempt to use it for
        // matching.
        break;
    }

    if (info_map == nullptr) {
      continue;
    }

    uint32_t id = inst.GetOperand(id_operand).AsId();
    assert(id != 0);

    assert(id < info_map->size());
    assert(std::find((*info_map)[id].begin(), (*info_map)[id].end(), &inst) ==
           (*info_map)[id].end());

    (*info_map)[id].push_back(&inst);
  }
}

void Differ::PoolPotentialIds(
    opt::IteratorRange<opt::Module::const_inst_iterator> section,
    std::vector<uint32_t>& ids, bool is_src,
    std::function<bool(const opt::Instruction&)> filter,
    std::function<uint32_t(const opt::Instruction&)> get_id) {
  for (const opt::Instruction& inst : section) {
    if (!filter(inst)) {
      continue;
    }

    uint32_t result_id = get_id(inst);
    assert(result_id != 0);

    assert(std::find(ids.begin(), ids.end(), result_id) == ids.end());

    // Don't include ids that are already matched, for example through
    // OpTypeForwardPointer.
    const bool is_matched = is_src ? id_map_.IsSrcMapped(result_id)
                                   : id_map_.IsDstMapped(result_id);
    if (is_matched) {
      continue;
    }

    ids.push_back(result_id);
  }
}

void Differ::MatchIds(
    PotentialIdMap& potential,
    std::function<bool(const opt::Instruction*, const opt::Instruction*)>
        match) {
  for (size_t src_index = 0; src_index < potential.src_ids.size();
       ++src_index) {
    for (size_t dst_index = 0; dst_index < potential.dst_ids.size();
         ++dst_index) {
      const uint32_t src_id = potential.src_ids[src_index];
      const uint32_t dst_id = potential.dst_ids[dst_index];

      if (dst_id == 0) {
        // Already matched.
        continue;
      }

      const opt::Instruction* src_inst = src_id_to_.inst_map_[src_id];
      const opt::Instruction* dst_inst = dst_id_to_.inst_map_[dst_id];

      if (match(src_inst, dst_inst)) {
        id_map_.MapIds(src_id, dst_id);

        // Remove the ids from the potential list.
        potential.src_ids[src_index] = 0;
        potential.dst_ids[dst_index] = 0;

        // Find a match for the next src id.
        break;
      }
    }
  }

  // Remove matched ids to make the next iteration faster.
  CompactIds(potential.src_ids);
  CompactIds(potential.dst_ids);
}

void Differ::MatchPreambleInstructions(
    opt::IteratorRange<opt::Module::const_inst_iterator> src_insts,
    opt::IteratorRange<opt::Module::const_inst_iterator> dst_insts) {
  // First, pool all instructions from each section and sort them.
  InstructionList sorted_src_insts = SortPreambleInstructions(src_, src_insts);
  InstructionList sorted_dst_insts = SortPreambleInstructions(dst_, dst_insts);

  // Then walk and match them.
  size_t src_cur = 0;
  size_t dst_cur = 0;

  while (src_cur < sorted_src_insts.size() &&
         dst_cur < sorted_dst_insts.size()) {
    const opt::Instruction* src_inst = sorted_src_insts[src_cur];
    const opt::Instruction* dst_inst = sorted_dst_insts[dst_cur];

    int compare = ComparePreambleInstructions(src_inst, dst_inst, src_, dst_);
    if (compare == 0) {
      id_map_.MapInsts(src_inst, dst_inst);
    }
    if (compare <= 0) {
      ++src_cur;
    }
    if (compare >= 0) {
      ++dst_cur;
    }
  }
}

InstructionList Differ::SortPreambleInstructions(
    const opt::Module* module,
    opt::IteratorRange<opt::Module::const_inst_iterator> insts) {
  InstructionList sorted;
  for (const opt::Instruction& inst : insts) {
    sorted.push_back(&inst);
  }
  std::sort(
      sorted.begin(), sorted.end(),
      [this, module](const opt::Instruction* a, const opt::Instruction* b) {
        return ComparePreambleInstructions(a, b, module, module) < 0;
      });
  return sorted;
}

int Differ::ComparePreambleInstructions(const opt::Instruction* a,
                                        const opt::Instruction* b,
                                        const opt::Module* src_inst_module,
                                        const opt::Module* dst_inst_module) {
  assert(a->opcode() == b->opcode());
  assert(!a->HasResultId());
  assert(!a->HasResultType());

  const uint32_t a_operand_count = a->NumOperands();
  const uint32_t b_operand_count = b->NumOperands();

  if (a_operand_count < b_operand_count) {
    return -1;
  }
  if (a_operand_count > b_operand_count) {
    return 1;
  }

  // Instead of comparing OpExecutionMode entry point ids as ids, compare them
  // through their corresponding execution model.  This simplifies traversing
  // the sorted list of instructions between src and dst modules.
  if (a->opcode() == spv::Op::OpExecutionMode) {
    const spv::ExecutionModel src_model =
        GetExecutionModel(src_inst_module, a->GetSingleWordOperand(0));
    const spv::ExecutionModel dst_model =
        GetExecutionModel(dst_inst_module, b->GetSingleWordOperand(0));

    if (src_model < dst_model) {
      return -1;
    }
    if (src_model > dst_model) {
      return 1;
    }
  }

  // Match every operand of the instruction.
  for (uint32_t operand_index = 0; operand_index < a_operand_count;
       ++operand_index) {
    const opt::Operand& a_operand = a->GetOperand(operand_index);
    const opt::Operand& b_operand = b->GetOperand(operand_index);

    if (a_operand.type < b_operand.type) {
      return -1;
    }
    if (a_operand.type > b_operand.type) {
      return 1;
    }

    switch (a_operand.type) {
      case SPV_OPERAND_TYPE_ID:
        // Don't compare ids, there can't be multiple instances of the
        // OpExecutionMode with different ids of the same execution model.
        break;
      case SPV_OPERAND_TYPE_TYPE_ID:
      case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
      case SPV_OPERAND_TYPE_SCOPE_ID:
        assert(false && "Unreachable");
        break;
      case SPV_OPERAND_TYPE_LITERAL_STRING: {
        int str_compare =
            strcmp(a_operand.AsString().c_str(), b_operand.AsString().c_str());
        if (str_compare != 0) {
          return str_compare;
        }
        break;
      }
      default:
        // Expect literal values to match.
        assert(a_operand.words.size() == 1);
        assert(b_operand.words.size() == 1);

        if (a_operand.words[0] < b_operand.words[0]) {
          return -1;
        }
        if (a_operand.words[0] > b_operand.words[0]) {
          return 1;
        }
        break;
    }
  }

  return 0;
}

void Differ::MatchDebugAndAnnotationInstructions(
    opt::IteratorRange<opt::Module::const_inst_iterator> src_insts,
    opt::IteratorRange<opt::Module::const_inst_iterator> dst_insts) {
  for (const opt::Instruction& src_inst : src_insts) {
    for (const opt::Instruction& dst_inst : dst_insts) {
      if (MappedSrcInst(&dst_inst) != nullptr) {
        continue;
      }

      // Map instructions as soon as they match.  Debug and annotation
      // instructions are matched such that there can't be multiple matches.
      if (DoDebugAndAnnotationInstructionsMatch(&src_inst, &dst_inst)) {
        id_map_.MapInsts(&src_inst, &dst_inst);
        break;
      }
    }
  }
}

uint32_t Differ::GroupIdsHelperGetTypeId(const IdInstructions& id_to,
                                         uint32_t id) {
  return GetInst(id_to, id)->type_id();
}

spv::StorageClass Differ::GroupIdsHelperGetTypePointerStorageClass(
    const IdInstructions& id_to, uint32_t id) {
  const opt::Instruction* inst = GetInst(id_to, id);
  assert(inst && inst->opcode() == spv::Op::OpTypePointer);
  return spv::StorageClass(inst->GetSingleWordInOperand(0));
}

spv::Op Differ::GroupIdsHelperGetTypePointerTypeOp(const IdInstructions& id_to,
                                                   uint32_t id) {
  const opt::Instruction* inst = GetInst(id_to, id);
  assert(inst && inst->opcode() == spv::Op::OpTypePointer);

  const uint32_t type_id = inst->GetSingleWordInOperand(1);
  const opt::Instruction* type_inst = GetInst(id_to, type_id);
  assert(type_inst);

  return type_inst->opcode();
}

template <typename T>
void Differ::GroupIds(const IdGroup& ids, bool is_src,
                      std::map<T, IdGroup>* groups,
                      T (Differ::*get_group)(const IdInstructions&, uint32_t)) {
  assert(groups->empty());

  const IdInstructions& id_to = is_src ? src_id_to_ : dst_id_to_;

  for (const uint32_t id : ids) {
    // Don't include ids that are already matched, for example through
    // OpEntryPoint.
    const bool is_matched =
        is_src ? id_map_.IsSrcMapped(id) : id_map_.IsDstMapped(id);
    if (is_matched) {
      continue;
    }

    T group = (this->*get_group)(id_to, id);
    (*groups)[group].push_back(id);
  }
}

template <typename T>
void Differ::GroupIdsAndMatch(
    const IdGroup& src_ids, const IdGroup& dst_ids, T invalid_group_key,
    T (Differ::*get_group)(const IdInstructions&, uint32_t),
    std::function<void(const IdGroup& src_group, const IdGroup& dst_group)>
        match_group) {
  // Group the ids based on a key (get_group)
  std::map<T, IdGroup> src_groups;
  std::map<T, IdGroup> dst_groups;

  GroupIds<T>(src_ids, true, &src_groups, get_group);
  GroupIds<T>(dst_ids, false, &dst_groups, get_group);

  // Iterate over the groups, and match those with identical keys
  for (const auto& iter : src_groups) {
    const T& key = iter.first;
    const IdGroup& src_group = iter.second;

    if (key == invalid_group_key) {
      continue;
    }

    const IdGroup& dst_group = dst_groups[key];

    // Let the caller match the groups as appropriate.
    match_group(src_group, dst_group);
  }
}

void Differ::GroupIdsAndMatchByMappedId(
    const IdGroup& src_ids, const IdGroup& dst_ids,
    uint32_t (Differ::*get_group)(const IdInstructions&, uint32_t),
    std::function<void(const IdGroup& src_group, const IdGroup& dst_group)>
        match_group) {
  // Group the ids based on a key (get_group)
  std::map<uint32_t, IdGroup> src_groups;
  std::map<uint32_t, IdGroup> dst_groups;

  GroupIds<uint32_t>(src_ids, true, &src_groups, get_group);
  GroupIds<uint32_t>(dst_ids, false, &dst_groups, get_group);

  // Iterate over pairs of groups whose keys map to each other.
  for (const auto& iter : src_groups) {
    const uint32_t& src_key = iter.first;
    const IdGroup& src_group = iter.second;

    if (src_key == 0) {
      continue;
    }

    if (id_map_.IsSrcMapped(src_key)) {
      const uint32_t& dst_key = id_map_.MappedDstId(src_key);
      const IdGroup& dst_group = dst_groups[dst_key];

      // Let the caller match the groups as appropriate.
      match_group(src_group, dst_group);
    }
  }
}

bool Differ::DoIdsMatch(uint32_t src_id, uint32_t dst_id) {
  assert(dst_id != 0);
  return id_map_.MappedDstId(src_id) == dst_id;
}

bool Differ::DoesOperandMatch(const opt::Operand& src_operand,
                              const opt::Operand& dst_operand) {
  assert(src_operand.type == dst_operand.type);

  switch (src_operand.type) {
    case SPV_OPERAND_TYPE_ID:
    case SPV_OPERAND_TYPE_TYPE_ID:
    case SPV_OPERAND_TYPE_RESULT_ID:
    case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
    case SPV_OPERAND_TYPE_SCOPE_ID:
      // Match ids only if they are already matched in the id map.
      return DoIdsMatch(src_operand.AsId(), dst_operand.AsId());
    case SPV_OPERAND_TYPE_LITERAL_STRING:
      return src_operand.AsString() == dst_operand.AsString();
    default:
      // Otherwise expect them to match exactly.
      assert(src_operand.type != SPV_OPERAND_TYPE_LITERAL_STRING);
      if (src_operand.words.size() != dst_operand.words.size()) {
        return false;
      }
      for (size_t i = 0; i < src_operand.words.size(); ++i) {
        if (src_operand.words[i] != dst_operand.words[i]) {
          return false;
        }
      }
      return true;
  }
}

bool Differ::DoOperandsMatch(const opt::Instruction* src_inst,
                             const opt::Instruction* dst_inst,
                             uint32_t in_operand_index_start,
                             uint32_t in_operand_count) {
  // Caller should have returned early for instructions with different opcode.
  assert(src_inst->opcode() == dst_inst->opcode());

  bool match = true;
  for (uint32_t i = 0; i < in_operand_count; ++i) {
    const uint32_t in_operand_index = in_operand_index_start + i;

    const opt::Operand& src_operand = src_inst->GetInOperand(in_operand_index);
    const opt::Operand& dst_operand = dst_inst->GetInOperand(in_operand_index);

    match = match && DoesOperandMatch(src_operand, dst_operand);
  }

  return match;
}

bool Differ::DoInstructionsMatch(const opt::Instruction* src_inst,
                                 const opt::Instruction* dst_inst) {
  // Check whether the two instructions are identical, that is the instructions
  // themselves are matched, every id is matched, and every other value is
  // identical.
  if (MappedDstInst(src_inst) != dst_inst) {
    return false;
  }

  assert(src_inst->opcode() == dst_inst->opcode());
  if (src_inst->NumOperands() != dst_inst->NumOperands()) {
    return false;
  }

  for (uint32_t operand_index = 0; operand_index < src_inst->NumOperands();
       ++operand_index) {
    const opt::Operand& src_operand = src_inst->GetOperand(operand_index);
    const opt::Operand& dst_operand = dst_inst->GetOperand(operand_index);

    if (!DoesOperandMatch(src_operand, dst_operand)) {
      return false;
    }
  }

  return true;
}

bool Differ::DoIdsMatchFuzzy(uint32_t src_id, uint32_t dst_id) {
  assert(dst_id != 0);
  const uint32_t mapped_dst_id = id_map_.MappedDstId(src_id);

  // Consider unmatched ids as a match.  In function bodies, no result id is
  // matched yet and thus they are excluded from instruction matching when used
  // as parameters in subsequent instructions.
  if (mapped_dst_id == 0 || mapped_dst_id == dst_id) {
    return true;
  }

  // Int and Uint constants are interchangeable, match them in that case.
  if (AreIdenticalUintConstants(src_id, dst_id)) {
    return true;
  }

  return false;
}

bool Differ::DoesOperandMatchFuzzy(const opt::Operand& src_operand,
                                   const opt::Operand& dst_operand) {
  if (src_operand.type != dst_operand.type) {
    return false;
  }

  assert(src_operand.type != SPV_OPERAND_TYPE_RESULT_ID);
  assert(dst_operand.type != SPV_OPERAND_TYPE_RESULT_ID);

  switch (src_operand.type) {
    case SPV_OPERAND_TYPE_ID:
    case SPV_OPERAND_TYPE_TYPE_ID:
    case SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID:
    case SPV_OPERAND_TYPE_SCOPE_ID:
      // Match id operands only if they are already matched in the id map.
      return DoIdsMatchFuzzy(src_operand.AsId(), dst_operand.AsId());
    default:
      // Otherwise allow everything to match.
      return true;
  }
}

bool Differ::DoInstructionsMatchFuzzy(const opt::Instruction* src_inst,
                                      const opt::Instruction* dst_inst) {
  // Similar to DoOperandsMatch, but only checks that ids that have already been
  // matched are identical.  Ids that are unknown are allowed to match, as well
  // as any non-id operand.
  if (src_inst->opcode() != dst_inst->opcode()) {
    return false;
  }
  // For external instructions, make sure the set and opcode of the external
  // instruction matches too.
  if (src_inst->opcode() == spv::Op::OpExtInst) {
    if (!DoOperandsMatch(src_inst, dst_inst, 0, 2)) {
      return false;
    }
  }

  assert(src_inst->HasResultType() == dst_inst->HasResultType());
  if (src_inst->HasResultType() &&
      !DoIdsMatchFuzzy(src_inst->type_id(), dst_inst->type_id())) {
    return false;
  }

  // TODO: allow some instructions to match with different instruction lengths,
  // for example OpImage* with additional operands.
  if (src_inst->NumInOperandWords() != dst_inst->NumInOperandWords()) {
    return false;
  }

  bool match = true;
  for (uint32_t in_operand_index = 0;
       in_operand_index < src_inst->NumInOperandWords(); ++in_operand_index) {
    const opt::Operand& src_operand = src_inst->GetInOperand(in_operand_index);
    const opt::Operand& dst_operand = dst_inst->GetInOperand(in_operand_index);

    match = match && DoesOperandMatchFuzzy(src_operand, dst_operand);
  }

  return match;
}

bool Differ::AreIdenticalUintConstants(uint32_t src_id, uint32_t dst_id) {
  return IsConstantUint(src_id_to_, src_id) &&
         IsConstantUint(dst_id_to_, dst_id) &&
         GetConstantUint(src_id_to_, src_id) ==
             GetConstantUint(dst_id_to_, dst_id);
}

bool Differ::DoDebugAndAnnotationInstructionsMatch(
    const opt::Instruction* src_inst, const opt::Instruction* dst_inst) {
  if (src_inst->opcode() != dst_inst->opcode()) {
    return false;
  }

  switch (src_inst->opcode()) {
    case spv::Op::OpString:
    case spv::Op::OpSourceExtension:
    case spv::Op::OpModuleProcessed:
      return DoesOperandMatch(src_inst->GetOperand(0), dst_inst->GetOperand(0));
    case spv::Op::OpSource:
      return DoOperandsMatch(src_inst, dst_inst, 0, 2);
    case spv::Op::OpSourceContinued:
      return true;
    case spv::Op::OpName:
      return DoOperandsMatch(src_inst, dst_inst, 0, 1);
    case spv::Op::OpMemberName:
      return DoOperandsMatch(src_inst, dst_inst, 0, 2);
    case spv::Op::OpDecorate:
      return DoOperandsMatch(src_inst, dst_inst, 0, 2);
    case spv::Op::OpMemberDecorate:
      return DoOperandsMatch(src_inst, dst_inst, 0, 3);
    case spv::Op::OpExtInst:
    case spv::Op::OpDecorationGroup:
    case spv::Op::OpGroupDecorate:
    case spv::Op::OpGroupMemberDecorate:
      return false;
    default:
      return false;
  }
}

bool Differ::AreVariablesMatchable(uint32_t src_id, uint32_t dst_id,
                                   uint32_t flexibility) {
  // Variables must match by their built-in decorations.
  uint32_t src_built_in_decoration = 0, dst_built_in_decoration = 0;
  const bool src_is_built_in = GetDecorationValue(
      src_id_to_, src_id, spv::Decoration::BuiltIn, &src_built_in_decoration);
  const bool dst_is_built_in = GetDecorationValue(
      dst_id_to_, dst_id, spv::Decoration::BuiltIn, &dst_built_in_decoration);

  if (src_is_built_in != dst_is_built_in) {
    return false;
  }
  if (src_is_built_in && src_built_in_decoration != dst_built_in_decoration) {
    return false;
  }

  // Check their types and storage classes.
  spv::StorageClass src_storage_class, dst_storage_class;
  const uint32_t src_type_id =
      GetVarTypeId(src_id_to_, src_id, &src_storage_class);
  const uint32_t dst_type_id =
      GetVarTypeId(dst_id_to_, dst_id, &dst_storage_class);

  if (!DoIdsMatch(src_type_id, dst_type_id)) {
    return false;
  }
  switch (flexibility) {
    case 0:
      if (src_storage_class != dst_storage_class) {
        return false;
      }
      break;
    case 1:
      if (src_storage_class != dst_storage_class) {
        // Allow one of the two to be Private while the other is Input or
        // Output, this allows matching in/out variables that have been turned
        // global as part of linking two stages (as done in ANGLE).
        const bool src_is_io = src_storage_class == spv::StorageClass::Input ||
                               src_storage_class == spv::StorageClass::Output;
        const bool dst_is_io = dst_storage_class == spv::StorageClass::Input ||
                               dst_storage_class == spv::StorageClass::Output;
        const bool src_is_private =
            src_storage_class == spv::StorageClass::Private;
        const bool dst_is_private =
            dst_storage_class == spv::StorageClass::Private;

        if (!((src_is_io && dst_is_private) || (src_is_private && dst_is_io))) {
          return false;
        }
      }
      break;
    default:
      assert(false && "Unreachable");
      return false;
  }

  // TODO: Is there any other way to check compatiblity of the variables?  It's
  // easy to tell when the variables definitely don't match, but there's little
  // information that can be used for a definite match.
  return true;
}

bool Differ::MatchOpTypeStruct(const opt::Instruction* src_inst,
                               const opt::Instruction* dst_inst,
                               uint32_t flexibility) {
  const uint32_t src_type_id = src_inst->result_id();
  const uint32_t dst_type_id = dst_inst->result_id();

  bool src_has_name = false, dst_has_name = false;
  std::string src_name = GetName(src_id_to_, src_type_id, &src_has_name);
  std::string dst_name = GetName(dst_id_to_, dst_type_id, &dst_has_name);

  // If debug info is present, always match the structs by name.
  if (src_has_name && dst_has_name) {
    if (src_name != dst_name) {
      return false;
    }

    // For gl_PerVertex, find the type pointer of this type (array) and make
    // sure the storage classes of src and dst match; geometry and tessellation
    // shaders have two instances of gl_PerVertex.
    if (src_name == "gl_PerVertex") {
      return MatchPerVertexType(src_type_id, dst_type_id);
    }

    return true;
  }

  // If debug info is not present, match the structs by their type.

  // For gl_PerVertex, find the type pointer of this type (array) and match by
  // storage class. The gl_PerVertex struct is itself found by the BuiltIn
  // decorations applied to its members.
  const bool src_is_per_vertex = IsPerVertexType(src_id_to_, src_type_id);
  const bool dst_is_per_vertex = IsPerVertexType(dst_id_to_, dst_type_id);
  if (src_is_per_vertex != dst_is_per_vertex) {
    return false;
  }

  if (src_is_per_vertex) {
    return MatchPerVertexType(src_type_id, dst_type_id);
  }

  switch (flexibility) {
    case 0:
      if (src_inst->NumInOperandWords() != dst_inst->NumInOperandWords()) {
        return false;
      }
      return DoOperandsMatch(src_inst, dst_inst, 0,
                             src_inst->NumInOperandWords());
    case 1:
      // TODO: match by taking a diff of the fields, and see if there's a >75%
      // match.  Need to then make sure OpMemberName, OpMemberDecorate,
      // OpAccessChain etc are aware of the struct field matching.
      return false;
    default:
      assert(false && "Unreachable");
      return false;
  }
}

bool Differ::MatchOpConstant(const opt::Instruction* src_inst,
                             const opt::Instruction* dst_inst,
                             uint32_t flexibility) {
  // The constants' type must match.  In flexibility == 1, match constants of
  // int and uint, as they are generally interchangeable.
  switch (flexibility) {
    case 0:
      if (!DoesOperandMatch(src_inst->GetOperand(0), dst_inst->GetOperand(0))) {
        return false;
      }
      break;
    case 1:
      if (!IsIntType(src_id_to_, src_inst->type_id()) ||
          !IsIntType(dst_id_to_, dst_inst->type_id())) {
        return false;
      }
      break;
    default:
      assert(false && "Unreachable");
      return false;
  }

  const opt::Operand& src_value_operand = src_inst->GetOperand(2);
  const opt::Operand& dst_value_operand = dst_inst->GetOperand(2);

  const uint64_t src_value = src_value_operand.AsLiteralUint64();
  const uint64_t dst_value = dst_value_operand.AsLiteralUint64();

  // If values are identical, it's a match.
  if (src_value == dst_value) {
    return true;
  }

  // Otherwise, only allow flexibility for float types.
  if (IsFloatType(src_id_to_, src_inst->type_id()) && flexibility == 1) {
    // Tolerance is:
    //
    // - For float: allow 4 bits of mantissa as error
    // - For double: allow 6 bits of mantissa as error
    //
    // TODO: the above values are arbitrary and a placeholder; investigate the
    // amount of error resulting from using `printf("%f", f)` and `printf("%lf",
    // d)` and having glslang parse them.
    const uint64_t tolerance = src_value_operand.words.size() == 1 ? 16 : 64;
    return src_value - dst_value < tolerance ||
           dst_value - src_value < tolerance;
  }

  return false;
}

bool Differ::MatchOpSpecConstant(const opt::Instruction* src_inst,
                                 const opt::Instruction* dst_inst) {
  const uint32_t src_id = src_inst->result_id();
  const uint32_t dst_id = dst_inst->result_id();

  bool src_has_name = false, dst_has_name = false;
  std::string src_name = GetName(src_id_to_, src_id, &src_has_name);
  std::string dst_name = GetName(dst_id_to_, dst_id, &dst_has_name);

  // If debug info is present, always match the spec consts by name.
  if (src_has_name && dst_has_name) {
    return src_name == dst_name;
  }

  // Otherwise, match them by SpecId.
  uint32_t src_spec_id, dst_spec_id;

  if (GetDecorationValue(src_id_to_, src_id, spv::Decoration::SpecId,
                         &src_spec_id) &&
      GetDecorationValue(dst_id_to_, dst_id, spv::Decoration::SpecId,
                         &dst_spec_id)) {
    return src_spec_id == dst_spec_id;
  }

  // There is no SpecId decoration, while not practical, still valid.
  // SpecConstantOp don't have SpecId and can be matched by operands
  if (src_inst->opcode() == spv::Op::OpSpecConstantOp) {
    if (src_inst->NumInOperandWords() == dst_inst->NumInOperandWords()) {
      return DoOperandsMatch(src_inst, dst_inst, 0,
                             src_inst->NumInOperandWords());
    }
  }

  return false;
}

bool Differ::MatchOpVariable(const opt::Instruction* src_inst,
                             const opt::Instruction* dst_inst,
                             uint32_t flexibility) {
  const uint32_t src_id = src_inst->result_id();
  const uint32_t dst_id = dst_inst->result_id();

  const bool src_is_pervertex = IsPerVertexVariable(src_id_to_, src_id);
  const bool dst_is_pervertex = IsPerVertexVariable(dst_id_to_, dst_id);

  // For gl_PerVertex, make sure the input and output instances are matched
  // correctly.
  if (src_is_pervertex != dst_is_pervertex) {
    return false;
  }
  if (src_is_pervertex) {
    return MatchPerVertexVariable(src_inst, dst_inst);
  }

  bool src_has_name = false, dst_has_name = false;
  std::string src_name = GetName(src_id_to_, src_id, &src_has_name);
  std::string dst_name = GetName(dst_id_to_, dst_id, &dst_has_name);

  // If debug info is present, always match the variables by name.
  if (src_has_name && dst_has_name) {
    return src_name == dst_name;
  }

  // If debug info is not present, see if the variables can be matched by their
  // built-in decorations.
  uint32_t src_built_in_decoration;
  const bool src_is_built_in = GetDecorationValue(
      src_id_to_, src_id, spv::Decoration::BuiltIn, &src_built_in_decoration);

  if (src_is_built_in && AreVariablesMatchable(src_id, dst_id, flexibility)) {
    return true;
  }

  spv::StorageClass src_storage_class, dst_storage_class;
  GetVarTypeId(src_id_to_, src_id, &src_storage_class);
  GetVarTypeId(dst_id_to_, dst_id, &dst_storage_class);

  if (src_storage_class != dst_storage_class) {
    return false;
  }

  // If variables are decorated with set/binding, match by the value of those
  // decorations.
  if (!options_.ignore_set_binding) {
    uint32_t src_set = 0, dst_set = 0;
    uint32_t src_binding = 0, dst_binding = 0;

    const bool src_has_set = GetDecorationValue(
        src_id_to_, src_id, spv::Decoration::DescriptorSet, &src_set);
    const bool dst_has_set = GetDecorationValue(
        dst_id_to_, dst_id, spv::Decoration::DescriptorSet, &dst_set);
    const bool src_has_binding = GetDecorationValue(
        src_id_to_, src_id, spv::Decoration::Binding, &src_set);
    const bool dst_has_binding = GetDecorationValue(
        dst_id_to_, dst_id, spv::Decoration::Binding, &dst_set);

    if (src_has_set && dst_has_set && src_has_binding && dst_has_binding) {
      return src_set == dst_set && src_binding == dst_binding;
    }
  }

  // If variables are decorated with location, match by the value of that
  // decoration.
  if (!options_.ignore_location) {
    uint32_t src_location, dst_location;

    const bool src_has_location = GetDecorationValue(
        src_id_to_, src_id, spv::Decoration::Location, &src_location);
    const bool dst_has_location = GetDecorationValue(
        dst_id_to_, dst_id, spv::Decoration::Location, &dst_location);

    if (src_has_location && dst_has_location) {
      return src_location == dst_location;
    }
  }

  // Currently, there's no other way to match variables.
  return false;
}

bool Differ::MatchPerVertexType(uint32_t src_type_id, uint32_t dst_type_id) {
  // For gl_PerVertex, find the type pointer of this type (array) and make sure
  // the storage classes of src and dst match; geometry and tessellation shaders
  // have two instances of gl_PerVertex.
  spv::StorageClass src_storage_class =
      GetPerVertexStorageClass(src_, src_type_id);
  spv::StorageClass dst_storage_class =
      GetPerVertexStorageClass(dst_, dst_type_id);

  assert(src_storage_class == spv::StorageClass::Input ||
         src_storage_class == spv::StorageClass::Output);
  assert(dst_storage_class == spv::StorageClass::Input ||
         dst_storage_class == spv::StorageClass::Output);

  return src_storage_class == dst_storage_class;
}

bool Differ::MatchPerVertexVariable(const opt::Instruction* src_inst,
                                    const opt::Instruction* dst_inst) {
  spv::StorageClass src_storage_class =
      spv::StorageClass(src_inst->GetSingleWordInOperand(0));
  spv::StorageClass dst_storage_class =
      spv::StorageClass(dst_inst->GetSingleWordInOperand(0));

  return src_storage_class == dst_storage_class;
}

void Differ::MatchTypeForwardPointersByName(const IdGroup& src,
                                            const IdGroup& dst) {
  // Given two sets of compatible groups of OpTypeForwardPointer instructions,
  // attempts to match them by name.

  // Group them by debug info and loop over them.
  GroupIdsAndMatch<std::string>(
      src, dst, "", &Differ::GetSanitizedName,
      [this](const IdGroup& src_group, const IdGroup& dst_group) {
        // Match only if there's a unique forward declaration with this debug
        // name.
        if (src_group.size() == 1 && dst_group.size() == 1) {
          id_map_.MapIds(src_group[0], dst_group[0]);
        }
      });
}

void Differ::MatchTypeForwardPointersByTypeOp(const IdGroup& src,
                                              const IdGroup& dst) {
  // Given two sets of compatible groups of OpTypeForwardPointer instructions,
  // attempts to match them by type op.  Must be called after
  // MatchTypeForwardPointersByName to match as many as possible by debug info.

  // Remove ids that are matched with debug info in
  // MatchTypeForwardPointersByName.
  IdGroup src_unmatched_ids;
  IdGroup dst_unmatched_ids;

  std::copy_if(src.begin(), src.end(), std::back_inserter(src_unmatched_ids),
               [this](uint32_t id) { return !id_map_.IsSrcMapped(id); });
  std::copy_if(dst.begin(), dst.end(), std::back_inserter(dst_unmatched_ids),
               [this](uint32_t id) { return !id_map_.IsDstMapped(id); });

  // Match only if there's a unique forward declaration with this
  // storage class and type opcode.  If both have debug info, they
  // must not have been matchable.
  if (src_unmatched_ids.size() == 1 && dst_unmatched_ids.size() == 1) {
    uint32_t src_id = src_unmatched_ids[0];
    uint32_t dst_id = dst_unmatched_ids[0];
    if (!HasName(src_id_to_, src_id) || !HasName(dst_id_to_, dst_id)) {
      id_map_.MapIds(src_id, dst_id);
    }
  }
}

InstructionList Differ::GetFunctionBody(opt::IRContext* context,
                                        opt::Function& function) {
  // Canonicalize the blocks of the function to produce better diff, for example
  // to not produce any diff if the src and dst have the same switch/case blocks
  // but with the cases simply reordered.
  std::list<opt::BasicBlock*> order;
  context->cfg()->ComputeStructuredOrder(&function, &*function.begin(), &order);

  // Go over the instructions of the function and add the instructions to a flat
  // list to simplify future iterations.
  InstructionList body;
  for (opt::BasicBlock* block : order) {
    block->ForEachInst(
        [&body](const opt::Instruction* inst) { body.push_back(inst); }, true);
  }
  body.push_back(function.EndInst());

  return body;
}

InstructionList Differ::GetFunctionHeader(const opt::Function& function) {
  // Go over the instructions of the function and add the header instructions to
  // a flat list to simplify diff generation.
  InstructionList body;
  function.WhileEachInst(
      [&body](const opt::Instruction* inst) {
        if (inst->opcode() == spv::Op::OpLabel) {
          return false;
        }
        body.push_back(inst);
        return true;
      },
      true, true);

  return body;
}

void Differ::GetFunctionBodies(opt::IRContext* context, FunctionMap* functions,
                               FunctionInstMap* function_insts) {
  for (opt::Function& function : *context->module()) {
    uint32_t id = function.result_id();
    assert(functions->find(id) == functions->end());
    assert(function_insts->find(id) == function_insts->end());

    (*functions)[id] = &function;

    InstructionList body = GetFunctionBody(context, function);
    (*function_insts)[id] = std::move(body);
  }
}

void Differ::GetFunctionHeaderInstructions(const opt::Module* module,
                                           FunctionInstMap* function_insts) {
  for (opt::Function& function : *module) {
    InstructionList body = GetFunctionHeader(function);
    (*function_insts)[function.result_id()] = std::move(body);
  }
}

void Differ::BestEffortMatchFunctions(const IdGroup& src_func_ids,
                                      const IdGroup& dst_func_ids,
                                      const FunctionInstMap& src_func_insts,
                                      const FunctionInstMap& dst_func_insts) {
  struct MatchResult {
    uint32_t src_id;
    uint32_t dst_id;
    DiffMatch src_match;
    DiffMatch dst_match;
    float match_rate;
    bool operator<(const MatchResult& other) const {
      return match_rate > other.match_rate;
    }
  };
  std::vector<MatchResult> all_match_results;

  for (const uint32_t src_func_id : src_func_ids) {
    if (id_map_.IsSrcMapped(src_func_id)) {
      continue;
    }
    const std::string src_name = GetSanitizedName(src_id_to_, src_func_id);

    for (const uint32_t dst_func_id : dst_func_ids) {
      if (id_map_.IsDstMapped(dst_func_id)) {
        continue;
      }

      // Don't match functions that are named, but the names are different.
      const std::string dst_name = GetSanitizedName(dst_id_to_, dst_func_id);
      if (src_name != "" && dst_name != "" && src_name != dst_name) {
        continue;
      }

      DiffMatch src_match_result, dst_match_result;
      float match_rate = MatchFunctionBodies(
          src_func_insts.at(src_func_id), dst_func_insts.at(dst_func_id),
          &src_match_result, &dst_match_result);

      // Only consider the functions a match if there's at least 60% match.
      // This is an arbitrary limit that should be tuned.
      constexpr float pass_match_rate = 0.6f;
      if (match_rate >= pass_match_rate) {
        all_match_results.emplace_back(
            MatchResult{src_func_id, dst_func_id, std::move(src_match_result),
                        std::move(dst_match_result), match_rate});
      }
    }
  }

  std::sort(all_match_results.begin(), all_match_results.end());

  for (const MatchResult& match_result : all_match_results) {
    if (id_map_.IsSrcMapped(match_result.src_id) ||
        id_map_.IsDstMapped(match_result.dst_id)) {
      continue;
    }

    id_map_.MapIds(match_result.src_id, match_result.dst_id);

    MatchFunctionParamIds(src_funcs_[match_result.src_id],
                          dst_funcs_[match_result.dst_id]);
    MatchIdsInFunctionBodies(src_func_insts.at(match_result.src_id),
                             dst_func_insts.at(match_result.dst_id),
                             match_result.src_match, match_result.dst_match, 0);
  }
}

void Differ::MatchFunctionParamIds(const opt::Function* src_func,
                                   const opt::Function* dst_func) {
  IdGroup src_params;
  IdGroup dst_params;
  src_func->ForEachParam(
      [&src_params](const opt::Instruction* param) {
        src_params.push_back(param->result_id());
      },
      false);
  dst_func->ForEachParam(
      [&dst_params](const opt::Instruction* param) {
        dst_params.push_back(param->result_id());
      },
      false);

  GroupIdsAndMatch<std::string>(
      src_params, dst_params, "", &Differ::GetSanitizedName,
      [this](const IdGroup& src_group, const IdGroup& dst_group) {
        // There shouldn't be two parameters with the same name, so the ids
        // should match. There is nothing restricting the SPIR-V however to have
        // two parameters with the same name, so be resilient against that.
        if (src_group.size() == 1 && dst_group.size() == 1) {
          id_map_.MapIds(src_group[0], dst_group[0]);
        }
      });

  // Then match the parameters by their type.  If there are multiple of them,
  // match them by their order.
  GroupIdsAndMatchByMappedId(
      src_params, dst_params, &Differ::GroupIdsHelperGetTypeId,
      [this](const IdGroup& src_group_by_type_id,
             const IdGroup& dst_group_by_type_id) {
        const size_t shared_param_count =
            std::min(src_group_by_type_id.size(), dst_group_by_type_id.size());

        for (size_t param_index = 0; param_index < shared_param_count;
             ++param_index) {
          id_map_.MapIds(src_group_by_type_id[param_index],
                         dst_group_by_type_id[param_index]);
        }
      });
}

float Differ::MatchFunctionBodies(const InstructionList& src_body,
                                  const InstructionList& dst_body,
                                  DiffMatch* src_match_result,
                                  DiffMatch* dst_match_result) {
  LongestCommonSubsequence<std::vector<const opt::Instruction*>> lcs(src_body,
                                                                     dst_body);

  uint32_t best_match_length = lcs.Get<const opt::Instruction*>(
      [this](const opt::Instruction* src_inst,
             const opt::Instruction* dst_inst) {
        return DoInstructionsMatchFuzzy(src_inst, dst_inst);
      },
      src_match_result, dst_match_result);

  // TODO: take the gaps in between matches and match those again with a relaxed
  // instruction-and-type-only comparison.  This can produce a better diff for
  // example if an array index is changed, causing the OpAccessChain id to not
  // match and subsequently every operation that's derived from that id.
  // Usually this mismatch cascades until the next OpStore which doesn't produce
  // an id.

  return static_cast<float>(best_match_length) * 2.0f /
         static_cast<float>(src_body.size() + dst_body.size());
}

void Differ::MatchIdsInFunctionBodies(const InstructionList& src_body,
                                      const InstructionList& dst_body,
                                      const DiffMatch& src_match_result,
                                      const DiffMatch& dst_match_result,
                                      uint32_t flexibility) {
  size_t src_cur = 0;
  size_t dst_cur = 0;

  while (src_cur < src_body.size() && dst_cur < dst_body.size()) {
    if (src_match_result[src_cur] && dst_match_result[dst_cur]) {
      // Match instructions the src and dst instructions.
      //
      // TODO: count the matchings between variables discovered this way and
      // choose the "best match" after all functions have been diffed and all
      // instructions analyzed.
      const opt::Instruction* src_inst = src_body[src_cur++];
      const opt::Instruction* dst_inst = dst_body[dst_cur++];

      // Record the matching between the instructions.  This is done only once
      // (hence flexibility == 0).  Calls with non-zero flexibility values will
      // only deal with matching other ids based on the operands.
      if (flexibility == 0) {
        id_map_.MapInsts(src_inst, dst_inst);
      }

      // Match any unmatched variables referenced by the instructions.
      MatchVariablesUsedByMatchedInstructions(src_inst, dst_inst, flexibility);
      continue;
    }
    if (!src_match_result[src_cur]) {
      ++src_cur;
    }
    if (!dst_match_result[dst_cur]) {
      ++dst_cur;
    }
  }
}

void Differ::MatchVariablesUsedByMatchedInstructions(
    const opt::Instruction* src_inst, const opt::Instruction* dst_inst,
    uint32_t flexibility) {
  // For OpAccessChain, OpLoad and OpStore instructions that reference unmatched
  // variables, match them as a best effort.
  assert(src_inst->opcode() == dst_inst->opcode());
  switch (src_inst->opcode()) {
    default:
      // TODO: match functions based on OpFunctionCall?
      break;
    case spv::Op::OpAccessChain:
    case spv::Op::OpInBoundsAccessChain:
    case spv::Op::OpPtrAccessChain:
    case spv::Op::OpInBoundsPtrAccessChain:
    case spv::Op::OpLoad:
    case spv::Op::OpStore:
      const uint32_t src_pointer_id = src_inst->GetSingleWordInOperand(0);
      const uint32_t dst_pointer_id = dst_inst->GetSingleWordInOperand(0);
      if (IsVariable(src_id_to_, src_pointer_id) &&
          IsVariable(dst_id_to_, dst_pointer_id) &&
          !id_map_.IsSrcMapped(src_pointer_id) &&
          !id_map_.IsDstMapped(dst_pointer_id) &&
          AreVariablesMatchable(src_pointer_id, dst_pointer_id, flexibility)) {
        id_map_.MapIds(src_pointer_id, dst_pointer_id);
      }
      break;
  }
}

const opt::Instruction* Differ::GetInst(const IdInstructions& id_to,
                                        uint32_t id) {
  assert(id != 0);
  assert(id < id_to.inst_map_.size());

  const opt::Instruction* inst = id_to.inst_map_[id];
  assert(inst != nullptr);

  return inst;
}

uint32_t Differ::GetConstantUint(const IdInstructions& id_to,
                                 uint32_t constant_id) {
  const opt::Instruction* constant_inst = GetInst(id_to, constant_id);
  assert(constant_inst->opcode() == spv::Op::OpConstant);
  assert(GetInst(id_to, constant_inst->type_id())->opcode() ==
         spv::Op::OpTypeInt);

  return constant_inst->GetSingleWordInOperand(0);
}

spv::ExecutionModel Differ::GetExecutionModel(const opt::Module* module,
                                              uint32_t entry_point_id) {
  for (const opt::Instruction& inst : module->entry_points()) {
    assert(inst.opcode() == spv::Op::OpEntryPoint);
    if (inst.GetSingleWordOperand(1) == entry_point_id) {
      return spv::ExecutionModel(inst.GetSingleWordOperand(0));
    }
  }

  assert(false && "Unreachable");
  return spv::ExecutionModel(0xFFF);
}

bool Differ::HasName(const IdInstructions& id_to, uint32_t id) {
  assert(id != 0);
  assert(id < id_to.name_map_.size());

  for (const opt::Instruction* inst : id_to.name_map_[id]) {
    if (inst->opcode() == spv::Op::OpName) {
      return true;
    }
  }

  return false;
}

std::string Differ::GetName(const IdInstructions& id_to, uint32_t id,
                            bool* has_name) {
  assert(id != 0);
  assert(id < id_to.name_map_.size());

  for (const opt::Instruction* inst : id_to.name_map_[id]) {
    if (inst->opcode() == spv::Op::OpName) {
      *has_name = true;
      return inst->GetOperand(1).AsString();
    }
  }

  *has_name = false;
  return "";
}

std::string Differ::GetSanitizedName(const IdInstructions& id_to, uint32_t id) {
  bool has_name = false;
  std::string name = GetName(id_to, id, &has_name);

  if (!has_name) {
    return "";
  }

  // Remove args from the name, in case this is a function name
  return name.substr(0, name.find('('));
}

uint32_t Differ::GetVarTypeId(const IdInstructions& id_to, uint32_t var_id,
                              spv::StorageClass* storage_class) {
  const opt::Instruction* var_inst = GetInst(id_to, var_id);
  assert(var_inst->opcode() == spv::Op::OpVariable);

  *storage_class = spv::StorageClass(var_inst->GetSingleWordInOperand(0));

  // Get the type pointer from the variable.
  const uint32_t type_pointer_id = var_inst->type_id();
  const opt::Instruction* type_pointer_inst = GetInst(id_to, type_pointer_id);

  // Get the type from the type pointer.
  return type_pointer_inst->GetSingleWordInOperand(1);
}

bool Differ::GetDecorationValue(const IdInstructions& id_to, uint32_t id,
                                spv::Decoration decoration,
                                uint32_t* decoration_value) {
  assert(id != 0);
  assert(id < id_to.decoration_map_.size());

  for (const opt::Instruction* inst : id_to.decoration_map_[id]) {
    if (inst->opcode() == spv::Op::OpDecorate &&
        inst->GetSingleWordOperand(0) == id &&
        spv::Decoration(inst->GetSingleWordOperand(1)) == decoration) {
      *decoration_value = inst->GetSingleWordOperand(2);
      return true;
    }
  }

  return false;
}

const opt::Instruction* Differ::GetForwardPointerInst(
    const IdInstructions& id_to, uint32_t id) {
  assert(id != 0);
  assert(id < id_to.forward_pointer_map_.size());
  return id_to.forward_pointer_map_[id];
}

bool Differ::IsIntType(const IdInstructions& id_to, uint32_t type_id) {
  return IsOp(id_to, type_id, spv::Op::OpTypeInt);
}

bool Differ::IsFloatType(const IdInstructions& id_to, uint32_t type_id) {
  return IsOp(id_to, type_id, spv::Op::OpTypeFloat);
}

bool Differ::IsConstantUint(const IdInstructions& id_to, uint32_t id) {
  const opt::Instruction* constant_inst = GetInst(id_to, id);
  if (constant_inst->opcode() != spv::Op::OpConstant) {
    return false;
  }

  const opt::Instruction* type_inst = GetInst(id_to, constant_inst->type_id());
  return type_inst->opcode() == spv::Op::OpTypeInt;
}

bool Differ::IsVariable(const IdInstructions& id_to, uint32_t pointer_id) {
  return IsOp(id_to, pointer_id, spv::Op::OpVariable);
}

bool Differ::IsOp(const IdInstructions& id_to, uint32_t id, spv::Op op) {
  return GetInst(id_to, id)->opcode() == op;
}

bool Differ::IsPerVertexType(const IdInstructions& id_to, uint32_t type_id) {
  assert(type_id != 0);
  assert(type_id < id_to.decoration_map_.size());

  for (const opt::Instruction* inst : id_to.decoration_map_[type_id]) {
    if (inst->opcode() == spv::Op::OpMemberDecorate &&
        inst->GetSingleWordOperand(0) == type_id &&
        spv::Decoration(inst->GetSingleWordOperand(2)) ==
            spv::Decoration::BuiltIn) {
      spv::BuiltIn built_in = spv::BuiltIn(inst->GetSingleWordOperand(3));

      // Only gl_PerVertex can have, and it can only have, the following
      // built-in decorations.
      return built_in == spv::BuiltIn::Position ||
             built_in == spv::BuiltIn::PointSize ||
             built_in == spv::BuiltIn::ClipDistance ||
             built_in == spv::BuiltIn::CullDistance;
    }
  }

  return false;
}

bool Differ::IsPerVertexVariable(const IdInstructions& id_to, uint32_t var_id) {
  // Get the type from the type pointer.
  spv::StorageClass storage_class;
  uint32_t type_id = GetVarTypeId(id_to, var_id, &storage_class);
  const opt::Instruction* type_inst = GetInst(id_to, type_id);

  // If array, get the element type.
  if (type_inst->opcode() == spv::Op::OpTypeArray) {
    type_id = type_inst->GetSingleWordInOperand(0);
  }

  // Now check if the type is gl_PerVertex.
  return IsPerVertexType(id_to, type_id);
}

spv::StorageClass Differ::GetPerVertexStorageClass(const opt::Module* module,
                                                   uint32_t type_id) {
  for (const opt::Instruction& inst : module->types_values()) {
    switch (inst.opcode()) {
      case spv::Op::OpTypeArray:
        // The gl_PerVertex instance could be an array, look for a variable of
        // the array type instead.
        if (inst.GetSingleWordInOperand(0) == type_id) {
          type_id = inst.result_id();
        }
        break;
      case spv::Op::OpTypePointer:
        // Find the storage class of the pointer to this type.
        if (inst.GetSingleWordInOperand(1) == type_id) {
          return spv::StorageClass(inst.GetSingleWordInOperand(0));
        }
        break;
      default:
        break;
    }
  }

  // gl_PerVertex is declared, but is unused.  Return either of Input or Output
  // classes just so it matches one in the other module.  This should be highly
  // unlikely, perhaps except for ancient GS-used-to-emulate-CS scenarios.
  return spv::StorageClass::Output;
}

spv_ext_inst_type_t Differ::GetExtInstType(const IdInstructions& id_to,
                                           uint32_t set_id) {
  const opt::Instruction* set_inst = GetInst(id_to, set_id);
  return spvExtInstImportTypeGet(set_inst->GetInOperand(0).AsString().c_str());
}

spv_number_kind_t Differ::GetNumberKind(const IdInstructions& id_to,
                                        const opt::Instruction& inst,
                                        uint32_t operand_index,
                                        uint32_t* number_bit_width) {
  const opt::Operand& operand = inst.GetOperand(operand_index);
  *number_bit_width = 0;

  // A very limited version of Parser::parseOperand.
  switch (operand.type) {
    case SPV_OPERAND_TYPE_LITERAL_INTEGER:
    case SPV_OPERAND_TYPE_OPTIONAL_LITERAL_INTEGER:
      // Always unsigned integers.
      *number_bit_width = 32;
      return SPV_NUMBER_UNSIGNED_INT;
    case SPV_OPERAND_TYPE_LITERAL_FLOAT:
      // Always float.
      *number_bit_width = 32;
      return SPV_NUMBER_FLOATING;
    case SPV_OPERAND_TYPE_TYPED_LITERAL_NUMBER:
    case SPV_OPERAND_TYPE_OPTIONAL_TYPED_LITERAL_INTEGER:
      switch (inst.opcode()) {
        case spv::Op::OpSwitch:
        case spv::Op::OpConstant:
        case spv::Op::OpSpecConstant:
          // Same kind of number as the selector (OpSwitch) or the type
          // (Op*Constant).
          return GetTypeNumberKind(id_to, inst.GetSingleWordOperand(0),
                                   number_bit_width);
        default:
          assert(false && "Unreachable");
          break;
      }
      break;
    default:
      break;
  }

  return SPV_NUMBER_NONE;
}

spv_number_kind_t Differ::GetTypeNumberKind(const IdInstructions& id_to,
                                            uint32_t id,
                                            uint32_t* number_bit_width) {
  const opt::Instruction* type_inst = GetInst(id_to, id);
  if (!spvOpcodeIsScalarType(type_inst->opcode())) {
    type_inst = GetInst(id_to, type_inst->type_id());
  }

  switch (type_inst->opcode()) {
    case spv::Op::OpTypeInt:
      *number_bit_width = type_inst->GetSingleWordOperand(1);
      return type_inst->GetSingleWordOperand(2) == 0 ? SPV_NUMBER_UNSIGNED_INT
                                                     : SPV_NUMBER_SIGNED_INT;
      break;
    case spv::Op::OpTypeFloat:
      *number_bit_width = type_inst->GetSingleWordOperand(1);
      return SPV_NUMBER_FLOATING;
    default:
      assert(false && "Unreachable");
      return SPV_NUMBER_NONE;
  }
}

void Differ::MatchCapabilities() {
  MatchPreambleInstructions(src_->capabilities(), dst_->capabilities());
}

void Differ::MatchExtensions() {
  MatchPreambleInstructions(src_->extensions(), dst_->extensions());
}

void Differ::MatchExtInstImportIds() {
  // Bunch all of this section's ids as potential matches.
  PotentialIdMap potential_id_map;
  auto get_result_id = [](const opt::Instruction& inst) {
    return inst.result_id();
  };
  auto accept_all = [](const opt::Instruction&) { return true; };

  PoolPotentialIds(src_->ext_inst_imports(), potential_id_map.src_ids, true,
                   accept_all, get_result_id);
  PoolPotentialIds(dst_->ext_inst_imports(), potential_id_map.dst_ids, false,
                   accept_all, get_result_id);

  // Then match the ids.
  MatchIds(potential_id_map, [](const opt::Instruction* src_inst,
                                const opt::Instruction* dst_inst) {
    // Match OpExtInstImport by exact name, which is operand 1
    const opt::Operand& src_name = src_inst->GetOperand(1);
    const opt::Operand& dst_name = dst_inst->GetOperand(1);

    return src_name.AsString() == dst_name.AsString();
  });
}
void Differ::MatchMemoryModel() {
  // Always match the memory model instructions, there is always a single one of
  // it.
  id_map_.MapInsts(src_->GetMemoryModel(), dst_->GetMemoryModel());
}

void Differ::MatchEntryPointIds() {
  // Match OpEntryPoint ids (at index 1) by ExecutionModel (at index 0) and
  // possibly name (at index 2).  OpEntryPoint doesn't produce a result id, so
  // this function doesn't use the helpers the other functions use.

  // Map from execution model to OpEntryPoint instructions of that model.
  using ExecutionModelMap =
      std::unordered_map<uint32_t, std::vector<const opt::Instruction*>>;
  ExecutionModelMap src_entry_points_map;
  ExecutionModelMap dst_entry_points_map;
  std::set<uint32_t> all_execution_models;

  for (const opt::Instruction& src_inst : src_->entry_points()) {
    uint32_t execution_model = src_inst.GetSingleWordOperand(0);
    src_entry_points_map[execution_model].push_back(&src_inst);
    all_execution_models.insert(execution_model);
  }
  for (const opt::Instruction& dst_inst : dst_->entry_points()) {
    uint32_t execution_model = dst_inst.GetSingleWordOperand(0);
    dst_entry_points_map[execution_model].push_back(&dst_inst);
    all_execution_models.insert(execution_model);
  }

  // Go through each model and match the ids.
  for (const uint32_t execution_model : all_execution_models) {
    auto& src_insts = src_entry_points_map[execution_model];
    auto& dst_insts = dst_entry_points_map[execution_model];

    // If there is only one entry point in src and dst with that model, match
    // them unconditionally.
    if (src_insts.size() == 1 && dst_insts.size() == 1) {
      uint32_t src_id = src_insts[0]->GetSingleWordOperand(1);
      uint32_t dst_id = dst_insts[0]->GetSingleWordOperand(1);
      id_map_.MapIds(src_id, dst_id);
      id_map_.MapInsts(src_insts[0], dst_insts[0]);
      continue;
    }

    // Otherwise match them by name.
    for (const opt::Instruction* src_inst : src_insts) {
      for (const opt::Instruction* dst_inst : dst_insts) {
        if (id_map_.IsDstMapped(dst_inst)) continue;

        const opt::Operand& src_name = src_inst->GetOperand(2);
        const opt::Operand& dst_name = dst_inst->GetOperand(2);

        if (src_name.AsString() == dst_name.AsString()) {
          uint32_t src_id = src_inst->GetSingleWordOperand(1);
          uint32_t dst_id = dst_inst->GetSingleWordOperand(1);
          id_map_.MapIds(src_id, dst_id);
          id_map_.MapInsts(src_inst, dst_inst);
          break;
        }
      }
    }
  }
}

void Differ::MatchExecutionModes() {
  MatchPreambleInstructions(src_->execution_modes(), dst_->execution_modes());
}

void Differ::MatchTypeForwardPointers() {
  // Bunch all of type forward pointers as potential matches.
  PotentialIdMap potential_id_map;
  auto get_pointer_type_id = [](const opt::Instruction& inst) {
    return inst.GetSingleWordOperand(0);
  };
  auto accept_type_forward_pointer_ops = [](const opt::Instruction& inst) {
    return inst.opcode() == spv::Op::OpTypeForwardPointer;
  };

  PoolPotentialIds(src_->types_values(), potential_id_map.src_ids, true,
                   accept_type_forward_pointer_ops, get_pointer_type_id);
  PoolPotentialIds(dst_->types_values(), potential_id_map.dst_ids, false,
                   accept_type_forward_pointer_ops, get_pointer_type_id);

  // Matching types with cyclical references (i.e. in the style of linked lists)
  // can get very complex.  Currently, the diff tool matches types bottom up, so
  // on every instruction it expects to know if its operands are already matched
  // or not.  With cyclical references, it cannot know that.  Type matching may
  // need significant modifications to be able to support this use case.
  //
  // Currently, forwarded types are only matched by storage class and debug
  // info, with minimal matching of the type being forwarded:
  //
  // - Group by class
  //   - Group by OpType being pointed to
  //     - Group by debug info
  //       - If same name and unique, match
  //     - If leftover is unique, match

  // Group forwarded pointers by storage class first and loop over them.
  GroupIdsAndMatch<spv::StorageClass>(
      potential_id_map.src_ids, potential_id_map.dst_ids,
      spv::StorageClass::Max, &Differ::GroupIdsHelperGetTypePointerStorageClass,
      [this](const IdGroup& src_group_by_storage_class,
             const IdGroup& dst_group_by_storage_class) {
        // Group them further by the type they are pointing to and loop over
        // them.
        GroupIdsAndMatch<spv::Op>(
            src_group_by_storage_class, dst_group_by_storage_class,
            spv::Op::Max, &Differ::GroupIdsHelperGetTypePointerTypeOp,
            [this](const IdGroup& src_group_by_type_op,
                   const IdGroup& dst_group_by_type_op) {
              // Group them even further by debug info, if possible and match by
              // debug name.
              MatchTypeForwardPointersByName(src_group_by_type_op,
                                             dst_group_by_type_op);

              // Match the leftovers only if they lack debug info and there is
              // only one instance of them.
              MatchTypeForwardPointersByTypeOp(src_group_by_type_op,
                                               dst_group_by_type_op);
            });
      });

  // Match the instructions that forward declare the same type themselves
  for (uint32_t src_id : potential_id_map.src_ids) {
    uint32_t dst_id = id_map_.MappedDstId(src_id);
    if (dst_id == 0) continue;

    const opt::Instruction* src_forward_inst =
        GetForwardPointerInst(src_id_to_, src_id);
    const opt::Instruction* dst_forward_inst =
        GetForwardPointerInst(dst_id_to_, dst_id);

    assert(src_forward_inst);
    assert(dst_forward_inst);

    id_map_.MapInsts(src_forward_inst, dst_forward_inst);
  }
}

void Differ::MatchTypeIds() {
  // Bunch all of type ids as potential matches.
  PotentialIdMap potential_id_map;
  auto get_result_id = [](const opt::Instruction& inst) {
    return inst.result_id();
  };
  auto accept_type_ops = [](const opt::Instruction& inst) {
    return spvOpcodeGeneratesType(inst.opcode());
  };

  PoolPotentialIds(src_->types_values(), potential_id_map.src_ids, true,
                   accept_type_ops, get_result_id);
  PoolPotentialIds(dst_->types_values(), potential_id_map.dst_ids, false,
                   accept_type_ops, get_result_id);

  // Then match the ids.  Start with exact matches, then match the leftover with
  // gradually loosening degrees of strictness.  For example, in the absence of
  // debug info, two block types will be matched if they differ only in a few of
  // the fields.
  for (uint32_t flexibility = 0; flexibility < 2; ++flexibility) {
    MatchIds(potential_id_map, [this, flexibility](
                                   const opt::Instruction* src_inst,
                                   const opt::Instruction* dst_inst) {
      const spv::Op src_op = src_inst->opcode();
      const spv::Op dst_op = dst_inst->opcode();

      // Don't match if the opcode is not the same.
      if (src_op != dst_op) {
        return false;
      }

      switch (src_op) {
        case spv::Op::OpTypeVoid:
        case spv::Op::OpTypeBool:
        case spv::Op::OpTypeSampler:
        case spv::Op::OpTypeAccelerationStructureNV:
        case spv::Op::OpTypeRayQueryKHR:
          // the above types have no operands and are unique, match them.
          return true;
        case spv::Op::OpTypeInt:
        case spv::Op::OpTypeFloat:
        case spv::Op::OpTypeVector:
        case spv::Op::OpTypeMatrix:
        case spv::Op::OpTypeSampledImage:
        case spv::Op::OpTypeRuntimeArray:
        case spv::Op::OpTypePointer:
          // Match these instructions when all operands match.
          assert(src_inst->NumInOperandWords() ==
                 dst_inst->NumInOperandWords());
          return DoOperandsMatch(src_inst, dst_inst, 0,
                                 src_inst->NumInOperandWords());

        case spv::Op::OpTypeFunction:
        case spv::Op::OpTypeImage:
          // Match function types only if they have the same number of operands,
          // and they all match.
          // Match image types similarly, expecting the optional final parameter
          // to match (if provided in both)
          if (src_inst->NumInOperandWords() != dst_inst->NumInOperandWords()) {
            return false;
          }
          return DoOperandsMatch(src_inst, dst_inst, 0,
                                 src_inst->NumInOperandWords());

        case spv::Op::OpTypeArray:
          // Match arrays only if the element type and length match.  The length
          // is an id of a constant, so the actual constant it's defining is
          // compared instead.
          if (!DoOperandsMatch(src_inst, dst_inst, 0, 1)) {
            return false;
          }

          if (AreIdenticalUintConstants(src_inst->GetSingleWordInOperand(1),
                                        dst_inst->GetSingleWordInOperand(1))) {
            return true;
          }

          // If size is not OpConstant, expect the ids to match exactly (for
          // example if a spec contant is used).
          return DoOperandsMatch(src_inst, dst_inst, 1, 1);

        case spv::Op::OpTypeStruct:
          return MatchOpTypeStruct(src_inst, dst_inst, flexibility);

        default:
          return false;
      }
    });
  }
}

void Differ::MatchConstants() {
  // Bunch all of constant ids as potential matches.
  PotentialIdMap potential_id_map;
  auto get_result_id = [](const opt::Instruction& inst) {
    return inst.result_id();
  };
  auto accept_type_ops = [](const opt::Instruction& inst) {
    return spvOpcodeIsConstant(inst.opcode());
  };

  PoolPotentialIds(src_->types_values(), potential_id_map.src_ids, true,
                   accept_type_ops, get_result_id);
  PoolPotentialIds(dst_->types_values(), potential_id_map.dst_ids, false,
                   accept_type_ops, get_result_id);

  // Then match the ids.  Constants are matched exactly, except for float types
  // that are first matched exactly, then leftovers are matched with a small
  // error.
  for (uint32_t flexibility = 0; flexibility < 2; ++flexibility) {
    MatchIds(potential_id_map, [this, flexibility](
                                   const opt::Instruction* src_inst,
                                   const opt::Instruction* dst_inst) {
      const spv::Op src_op = src_inst->opcode();
      const spv::Op dst_op = dst_inst->opcode();

      // Don't match if the opcode is not the same.
      if (src_op != dst_op) {
        return false;
      }

      switch (src_op) {
        case spv::Op::OpConstantTrue:
        case spv::Op::OpConstantFalse:
          // true and false are unique, match them.
          return true;
        case spv::Op::OpConstant:
          return MatchOpConstant(src_inst, dst_inst, flexibility);
        case spv::Op::OpConstantComposite:
        case spv::Op::OpSpecConstantComposite:
          // Composite constants must match in type and value.
          //
          // TODO: match OpConstantNull with OpConstantComposite with all zeros
          // at flexibility == 1
          // TODO: match constants from structs that have been flexibly-matched.
          if (src_inst->NumInOperandWords() != dst_inst->NumInOperandWords()) {
            return false;
          }
          return DoesOperandMatch(src_inst->GetOperand(0),
                                  dst_inst->GetOperand(0)) &&
                 DoOperandsMatch(src_inst, dst_inst, 0,
                                 src_inst->NumInOperandWords());
        case spv::Op::OpConstantSampler:
          // Match sampler constants exactly.
          // TODO: Allow flexibility in parameters to better diff shaders where
          // the sampler param has changed.
          assert(src_inst->NumInOperandWords() ==
                 dst_inst->NumInOperandWords());
          return DoOperandsMatch(src_inst, dst_inst, 0,
                                 src_inst->NumInOperandWords());
        case spv::Op::OpConstantNull:
          // Match null constants as long as the type matches.
          return DoesOperandMatch(src_inst->GetOperand(0),
                                  dst_inst->GetOperand(0));

        case spv::Op::OpSpecConstantTrue:
        case spv::Op::OpSpecConstantFalse:
        case spv::Op::OpSpecConstant:
        case spv::Op::OpSpecConstantOp:
          // Match spec constants by name if available, then by the SpecId
          // decoration.
          return MatchOpSpecConstant(src_inst, dst_inst);

        default:
          return false;
      }
    });
  }
}

void Differ::MatchVariableIds() {
  // Bunch all of variable ids as potential matches.
  PotentialIdMap potential_id_map;
  auto get_result_id = [](const opt::Instruction& inst) {
    return inst.result_id();
  };
  auto accept_type_ops = [](const opt::Instruction& inst) {
    return inst.opcode() == spv::Op::OpVariable;
  };

  PoolPotentialIds(src_->types_values(), potential_id_map.src_ids, true,
                   accept_type_ops, get_result_id);
  PoolPotentialIds(dst_->types_values(), potential_id_map.dst_ids, false,
                   accept_type_ops, get_result_id);

  // Then match the ids.  Start with exact matches, then match the leftover with
  // gradually loosening degrees of strictness.  For example, in the absence of
  // debug info, two otherwise identical variables will be matched if one of
  // them has a Private storage class and the other doesn't.
  for (uint32_t flexibility = 0; flexibility < 2; ++flexibility) {
    MatchIds(potential_id_map,
             [this, flexibility](const opt::Instruction* src_inst,
                                 const opt::Instruction* dst_inst) {
               assert(src_inst->opcode() == spv::Op::OpVariable);
               assert(dst_inst->opcode() == spv::Op::OpVariable);

               return MatchOpVariable(src_inst, dst_inst, flexibility);
             });
  }
}

void Differ::MatchFunctions() {
  IdGroup src_func_ids;
  IdGroup dst_func_ids;

  for (const auto& func : src_funcs_) {
    src_func_ids.push_back(func.first);
  }
  for (const auto& func : dst_funcs_) {
    dst_func_ids.push_back(func.first);
  }

  // Base the matching of functions on debug info when available.
  GroupIdsAndMatch<std::string>(
      src_func_ids, dst_func_ids, "", &Differ::GetSanitizedName,
      [this](const IdGroup& src_group, const IdGroup& dst_group) {
        // If there is a single function with this name in src and dst, it's a
        // definite match.
        if (src_group.size() == 1 && dst_group.size() == 1) {
          id_map_.MapIds(src_group[0], dst_group[0]);
          return;
        }

        // If there are multiple functions with the same name, group them by
        // type, and match only if the types match (and are unique).
        GroupIdsAndMatch<uint32_t>(src_group, dst_group, 0,
                                   &Differ::GroupIdsHelperGetTypeId,
                                   [this](const IdGroup& src_group_by_type_id,
                                          const IdGroup& dst_group_by_type_id) {
                                     if (src_group_by_type_id.size() == 1 &&
                                         dst_group_by_type_id.size() == 1) {
                                       id_map_.MapIds(src_group_by_type_id[0],
                                                      dst_group_by_type_id[0]);
                                     }
                                   });
      });

  // Any functions that are left are pooled together and matched as if unnamed,
  // with the only exception that two functions with mismatching names are not
  // matched.
  //
  // Before that however, the diff of the functions that are matched are taken
  // and processed, so that more of the global variables can be matched before
  // attempting to match the rest of the functions.  They can contribute to the
  // precision of the diff of those functions.
  for (const uint32_t src_func_id : src_func_ids) {
    const uint32_t dst_func_id = id_map_.MappedDstId(src_func_id);
    if (dst_func_id == 0) {
      continue;
    }

    // Since these functions are definite matches, match their parameters for a
    // better diff.
    MatchFunctionParamIds(src_funcs_[src_func_id], dst_funcs_[dst_func_id]);

    // Take the diff of the two functions.
    DiffMatch src_match_result, dst_match_result;
    MatchFunctionBodies(src_func_insts_[src_func_id],
                        dst_func_insts_[dst_func_id], &src_match_result,
                        &dst_match_result);

    // Match ids between the two function bodies; which can also result in
    // global variables getting matched.
    MatchIdsInFunctionBodies(src_func_insts_[src_func_id],
                             dst_func_insts_[dst_func_id], src_match_result,
                             dst_match_result, 0);
  }

  // Best effort match functions with matching type.
  GroupIdsAndMatch<uint32_t>(
      src_func_ids, dst_func_ids, 0, &Differ::GroupIdsHelperGetTypeId,
      [this](const IdGroup& src_group_by_type_id,
             const IdGroup& dst_group_by_type_id) {
        BestEffortMatchFunctions(src_group_by_type_id, dst_group_by_type_id,
                                 src_func_insts_, dst_func_insts_);
      });

  // Any function that's left, best effort match them.
  BestEffortMatchFunctions(src_func_ids, dst_func_ids, src_func_insts_,
                           dst_func_insts_);
}

void Differ::MatchDebugs1() {
  // This section in cludes: OpString, OpSourceExtension, OpSource,
  // OpSourceContinued
  MatchDebugAndAnnotationInstructions(src_->debugs1(), dst_->debugs1());
}

void Differ::MatchDebugs2() {
  // This section includes: OpName, OpMemberName
  MatchDebugAndAnnotationInstructions(src_->debugs2(), dst_->debugs2());
}

void Differ::MatchDebugs3() {
  // This section includes: OpModuleProcessed
  MatchDebugAndAnnotationInstructions(src_->debugs3(), dst_->debugs3());
}

void Differ::MatchExtInstDebugInfo() {
  // This section includes OpExtInst for DebugInfo extension
  MatchDebugAndAnnotationInstructions(src_->ext_inst_debuginfo(),
                                      dst_->ext_inst_debuginfo());
}

void Differ::MatchAnnotations() {
  // This section includes OpDecorate and family.
  MatchDebugAndAnnotationInstructions(src_->annotations(), dst_->annotations());
}

const opt::Instruction* Differ::MappedDstInst(
    const opt::Instruction* src_inst) {
  return MappedInstImpl(src_inst, id_map_.SrcToDstMap(), dst_id_to_);
}

const opt::Instruction* Differ::MappedSrcInst(
    const opt::Instruction* dst_inst) {
  return MappedInstImpl(dst_inst, id_map_.DstToSrcMap(), src_id_to_);
}

const opt::Instruction* Differ::MappedInstImpl(
    const opt::Instruction* inst, const IdMap& to_other,
    const IdInstructions& other_id_to) {
  if (inst->HasResultId()) {
    if (to_other.IsMapped(inst->result_id())) {
      const uint32_t other_result_id = to_other.MappedId(inst->result_id());

      assert(other_result_id < other_id_to.inst_map_.size());
      return other_id_to.inst_map_[other_result_id];
    }

    return nullptr;
  }

  return to_other.MappedInst(inst);
}

void Differ::OutputLine(std::function<bool()> are_lines_identical,
                        std::function<void()> output_src_line,
                        std::function<void()> output_dst_line) {
  if (are_lines_identical()) {
    out_ << " ";
    output_src_line();
  } else {
    OutputRed();
    out_ << "-";
    output_src_line();

    OutputGreen();
    out_ << "+";
    output_dst_line();

    OutputResetColor();
  }
}

const opt::Instruction* IterInst(opt::Module::const_inst_iterator& iter) {
  return &*iter;
}

const opt::Instruction* IterInst(InstructionList::const_iterator& iter) {
  return *iter;
}

template <typename InstList>
void Differ::OutputSection(
    const InstList& src_insts, const InstList& dst_insts,
    std::function<void(const opt::Instruction&, const IdInstructions&,
                       const opt::Instruction&)>
        write_inst) {
  auto src_iter = src_insts.begin();
  auto dst_iter = dst_insts.begin();

  // - While src_inst doesn't have a match, output it with -
  // - While dst_inst doesn't have a match, output it with +
  // - Now src_inst and dst_inst both have matches; might not match each other!
  //   * If section is unordered, just process src_inst and its match (dst_inst
  //   or not),
  //     dst_inst will eventually be processed when its match is seen.
  //   * If section is ordered, also just process src_inst and its match.  Its
  //   match must
  //     necessarily be dst_inst.
  while (src_iter != src_insts.end() || dst_iter != dst_insts.end()) {
    OutputRed();
    while (src_iter != src_insts.end() &&
           MappedDstInst(IterInst(src_iter)) == nullptr) {
      out_ << "-";
      write_inst(*IterInst(src_iter), src_id_to_, *IterInst(src_iter));
      ++src_iter;
    }
    OutputGreen();
    while (dst_iter != dst_insts.end() &&
           MappedSrcInst(IterInst(dst_iter)) == nullptr) {
      out_ << "+";
      write_inst(ToMappedSrcIds(*IterInst(dst_iter)), dst_id_to_,
                 *IterInst(dst_iter));
      ++dst_iter;
    }
    OutputResetColor();

    if (src_iter != src_insts.end() && dst_iter != dst_insts.end()) {
      const opt::Instruction* src_inst = IterInst(src_iter);
      const opt::Instruction* matched_dst_inst = MappedDstInst(src_inst);

      assert(matched_dst_inst != nullptr);
      assert(MappedSrcInst(IterInst(dst_iter)) != nullptr);

      OutputLine(
          [this, src_inst, matched_dst_inst]() {
            return DoInstructionsMatch(src_inst, matched_dst_inst);
          },
          [this, src_inst, &write_inst]() {
            write_inst(*src_inst, src_id_to_, *src_inst);
          },
          [this, matched_dst_inst, &write_inst]() {
            write_inst(ToMappedSrcIds(*matched_dst_inst), dst_id_to_,
                       *matched_dst_inst);
          });

      ++src_iter;
      ++dst_iter;
    }
  }
}

void Differ::ToParsedInstruction(
    const opt::Instruction& inst, const IdInstructions& id_to,
    const opt::Instruction& original_inst,
    spv_parsed_instruction_t* parsed_inst,
    std::vector<spv_parsed_operand_t>& parsed_operands,
    std::vector<uint32_t>& inst_binary) {
  inst.ToBinaryWithoutAttachedDebugInsts(&inst_binary);
  parsed_operands.resize(inst.NumOperands());

  parsed_inst->words = inst_binary.data();
  parsed_inst->num_words = static_cast<uint16_t>(inst_binary.size());
  parsed_inst->opcode = static_cast<uint16_t>(inst.opcode());
  parsed_inst->ext_inst_type =
      inst.opcode() == spv::Op::OpExtInst
          ? GetExtInstType(id_to, original_inst.GetSingleWordInOperand(0))
          : SPV_EXT_INST_TYPE_NONE;
  parsed_inst->type_id =
      inst.HasResultType() ? inst.GetSingleWordOperand(0) : 0;
  parsed_inst->result_id = inst.HasResultId() ? inst.result_id() : 0;
  parsed_inst->operands = parsed_operands.data();
  parsed_inst->num_operands = static_cast<uint16_t>(parsed_operands.size());

  // Word 0 is always op and num_words, so operands start at offset 1.
  uint32_t offset = 1;
  for (uint16_t operand_index = 0; operand_index < parsed_inst->num_operands;
       ++operand_index) {
    const opt::Operand& operand = inst.GetOperand(operand_index);
    spv_parsed_operand_t& parsed_operand = parsed_operands[operand_index];

    parsed_operand.offset = static_cast<uint16_t>(offset);
    parsed_operand.num_words = static_cast<uint16_t>(operand.words.size());
    parsed_operand.type = operand.type;
    parsed_operand.number_kind = GetNumberKind(
        id_to, original_inst, operand_index, &parsed_operand.number_bit_width);

    offset += parsed_operand.num_words;
  }
}

opt::Instruction Differ::ToMappedSrcIds(const opt::Instruction& dst_inst) {
  // Create an identical instruction to dst_inst, except ids are changed to the
  // mapped one.
  opt::Instruction mapped_inst = dst_inst;

  for (uint32_t operand_index = 0; operand_index < mapped_inst.NumOperands();
       ++operand_index) {
    opt::Operand& operand = mapped_inst.GetOperand(operand_index);

    if (spvIsIdType(operand.type)) {
      assert(id_map_.IsDstMapped(operand.AsId()));
      operand.words[0] = id_map_.MappedSrcId(operand.AsId());
    }
  }

  return mapped_inst;
}

spv_result_t Differ::Output() {
  id_map_.MapUnmatchedIds(
      [this](uint32_t src_id) { return src_id_to_.IsDefined(src_id); },
      [this](uint32_t dst_id) { return dst_id_to_.IsDefined(dst_id); });
  src_id_to_.inst_map_.resize(id_map_.SrcToDstMap().IdBound(), nullptr);
  dst_id_to_.inst_map_.resize(id_map_.DstToSrcMap().IdBound(), nullptr);

  const spv_target_env target_env = SPV_ENV_UNIVERSAL_1_6;
  spv_opcode_table opcode_table;
  spv_operand_table operand_table;
  spv_ext_inst_table ext_inst_table;
  spv_result_t result;

  result = spvOpcodeTableGet(&opcode_table, target_env);
  if (result != SPV_SUCCESS) return result;

  result = spvOperandTableGet(&operand_table, target_env);
  if (result != SPV_SUCCESS) return result;

  result = spvExtInstTableGet(&ext_inst_table, target_env);
  if (result != SPV_SUCCESS) return result;

  spv_context_t context{
      target_env,
      opcode_table,
      operand_table,
      ext_inst_table,
  };

  const AssemblyGrammar grammar(&context);
  if (!grammar.isValid()) return SPV_ERROR_INVALID_TABLE;

  uint32_t disassembly_options = SPV_BINARY_TO_TEXT_OPTION_PRINT;
  if (options_.indent) {
    disassembly_options |= SPV_BINARY_TO_TEXT_OPTION_INDENT;
  }

  NameMapper name_mapper = GetTrivialNameMapper();
  disassemble::InstructionDisassembler dis(grammar, out_, disassembly_options,
                                           name_mapper);

  if (!options_.no_header) {
    // Output the header
    // TODO: when using diff with text, the assembler overrides the version and
    // generator, so these aren't reflected correctly in the output.  Could
    // potentially extract this info from the header comment.
    OutputLine([]() { return true; }, [&dis]() { dis.EmitHeaderSpirv(); },
               []() { assert(false && "Unreachable"); });
    OutputLine([this]() { return src_->version() == dst_->version(); },
               [this, &dis]() { dis.EmitHeaderVersion(src_->version()); },
               [this, &dis]() { dis.EmitHeaderVersion(dst_->version()); });
    OutputLine([this]() { return src_->generator() == dst_->generator(); },
               [this, &dis]() { dis.EmitHeaderGenerator(src_->generator()); },
               [this, &dis]() { dis.EmitHeaderGenerator(dst_->generator()); });
    OutputLine(
        [this]() { return src_->IdBound() == id_map_.SrcToDstMap().IdBound(); },
        [this, &dis]() { dis.EmitHeaderIdBound(src_->IdBound()); },
        [this, &dis]() {
          dis.EmitHeaderIdBound(id_map_.SrcToDstMap().IdBound());
        });
    OutputLine([this]() { return src_->schema() == dst_->schema(); },
               [this, &dis]() { dis.EmitHeaderSchema(src_->schema()); },
               [this, &dis]() { dis.EmitHeaderSchema(dst_->schema()); });
  }

  // For each section, iterate both modules and output the disassembly.
  auto write_inst = [this, &dis](const opt::Instruction& inst,
                                 const IdInstructions& id_to,
                                 const opt::Instruction& original_inst) {
    spv_parsed_instruction_t parsed_inst;
    std::vector<spv_parsed_operand_t> parsed_operands;
    std::vector<uint32_t> inst_binary;

    ToParsedInstruction(inst, id_to, original_inst, &parsed_inst,
                        parsed_operands, inst_binary);

    dis.EmitInstruction(parsed_inst, 0);
  };

  OutputSection(src_->capabilities(), dst_->capabilities(), write_inst);
  OutputSection(src_->extensions(), dst_->extensions(), write_inst);
  OutputSection(src_->ext_inst_imports(), dst_->ext_inst_imports(), write_inst);

  // There is only one memory model.
  OutputLine(
      [this]() {
        return DoInstructionsMatch(src_->GetMemoryModel(),
                                   dst_->GetMemoryModel());
      },
      [this, &write_inst]() {
        write_inst(*src_->GetMemoryModel(), src_id_to_,
                   *src_->GetMemoryModel());
      },
      [this, &write_inst]() {
        write_inst(*dst_->GetMemoryModel(), dst_id_to_,
                   *dst_->GetMemoryModel());
      });

  OutputSection(src_->entry_points(), dst_->entry_points(), write_inst);
  OutputSection(src_->execution_modes(), dst_->execution_modes(), write_inst);
  OutputSection(src_->debugs1(), dst_->debugs1(), write_inst);
  OutputSection(src_->debugs2(), dst_->debugs2(), write_inst);
  OutputSection(src_->debugs3(), dst_->debugs3(), write_inst);
  OutputSection(src_->ext_inst_debuginfo(), dst_->ext_inst_debuginfo(),
                write_inst);
  OutputSection(src_->annotations(), dst_->annotations(), write_inst);
  OutputSection(src_->types_values(), dst_->types_values(), write_inst);

  // Get the body of all the functions.
  FunctionInstMap src_func_header_insts;
  FunctionInstMap dst_func_header_insts;

  GetFunctionHeaderInstructions(src_, &src_func_header_insts);
  GetFunctionHeaderInstructions(dst_, &dst_func_header_insts);

  for (const auto& src_func : src_func_insts_) {
    const uint32_t src_func_id = src_func.first;
    const InstructionList& src_insts = src_func.second;
    const InstructionList& src_header_insts =
        src_func_header_insts[src_func_id];

    const uint32_t dst_func_id = id_map_.MappedDstId(src_func_id);
    if (dst_func_insts_.find(dst_func_id) == dst_func_insts_.end()) {
      OutputSection(src_header_insts, InstructionList(), write_inst);
      OutputSection(src_insts, InstructionList(), write_inst);
      continue;
    }

    const InstructionList& dst_insts = dst_func_insts_[dst_func_id];
    const InstructionList& dst_header_insts =
        dst_func_header_insts[dst_func_id];
    OutputSection(src_header_insts, dst_header_insts, write_inst);
    OutputSection(src_insts, dst_insts, write_inst);
  }

  for (const auto& dst_func : dst_func_insts_) {
    const uint32_t dst_func_id = dst_func.first;
    const InstructionList& dst_insts = dst_func.second;
    const InstructionList& dst_header_insts =
        dst_func_header_insts[dst_func_id];

    const uint32_t src_func_id = id_map_.MappedSrcId(dst_func_id);
    if (src_func_insts_.find(src_func_id) == src_func_insts_.end()) {
      OutputSection(InstructionList(), dst_header_insts, write_inst);
      OutputSection(InstructionList(), dst_insts, write_inst);
    }
  }

  out_ << std::flush;

  return SPV_SUCCESS;
}

}  // anonymous namespace

spv_result_t Diff(opt::IRContext* src, opt::IRContext* dst, std::ostream& out,
                  Options options) {
  // High level algorithm:
  //
  // - Some sections of SPIR-V don't deal with ids; instructions in those
  //   sections are matched identically.  For example OpCapability instructions.
  // - Some sections produce ids, and they can be trivially matched by their
  //   parameters.  For example OpExtInstImport instructions.
  // - Some sections annotate ids.  These are matched at the end, after the ids
  //   themselves are matched.  For example OpName or OpDecorate instructions.
  // - Some sections produce ids that depend on other ids and they can be
  //   recursively matched.  For example OpType* instructions.
  // - Some sections produce ids that are not trivially matched.  For these ids,
  //   the debug info is used when possible, or a best guess (such as through
  //   decorations) is used.  For example OpVariable instructions.
  // - Matching functions is done with multiple attempts:
  //   * Functions with identical debug names are matched if there are no
  //     overloads.
  //   * Otherwise, functions with identical debug names and types are matched.
  //   * The rest of the functions are best-effort matched, first in groups of
  //     identical type, then any with any.
  //     * The best-effort matching takes the diff of every pair of functions in
  //       a group and selects the top matches that also meet a similarity
  //       index.
  //   * Once a pair of functions are matched, the fuzzy diff of the
  //     instructions is used to match the instructions in the function body.
  //     The fuzzy diff makes sure that sufficiently similar instructions are
  //     matched and that yet-to-be-matched result ids don't result in a larger
  //     diff.
  //
  // Once the instructions are matched between the src and dst SPIR-V, the src
  // is traversed and its disassembly is output.  In the process, any unmatched
  // instruction is prefixed with -, and any unmatched instruction in dst in the
  // same section is output prefixed with +.  To avoid confusion, the
  // instructions in dst are output with matching ids in src so the output
  // assembly is consistent.

  Differ differ(src, dst, out, options);

  // First, match instructions between the different non-annotation sections of
  // the SPIR-V.
  differ.MatchCapabilities();
  differ.MatchExtensions();
  differ.MatchExtInstImportIds();
  differ.MatchMemoryModel();
  differ.MatchEntryPointIds();
  differ.MatchExecutionModes();
  differ.MatchTypeForwardPointers();
  differ.MatchTypeIds();
  differ.MatchConstants();
  differ.MatchVariableIds();
  differ.MatchFunctions();

  // Match instructions that annotate previously-matched ids.
  differ.MatchDebugs1();
  differ.MatchDebugs2();
  differ.MatchDebugs3();
  differ.MatchExtInstDebugInfo();
  differ.MatchAnnotations();

  // Show the disassembly with the diff.
  //
  // TODO: Based on an option, output either based on src or dst, i.e. the diff
  // can show the ids and instruction/function order either from src or dst.
  spv_result_t result = differ.Output();

  differ.DumpIdMap();

  return result;
}

}  // namespace diff
}  // namespace spvtools
