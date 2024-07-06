// Copyright (c) 2018 Google LLC
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

#ifndef LIBSPIRV_OPT_UPGRADE_MEMORY_MODEL_H_
#define LIBSPIRV_OPT_UPGRADE_MEMORY_MODEL_H_

#include <functional>
#include <tuple>

#include "pass.h"

namespace spvtools {
namespace opt {

// Hashing functor for the memoized result store.
struct CacheHash {
  size_t operator()(
      const std::pair<uint32_t, std::vector<uint32_t>>& item) const {
    std::u32string to_hash;
    to_hash.push_back(item.first);
    for (auto i : item.second) to_hash.push_back(i);
    return std::hash<std::u32string>()(to_hash);
  }
};

// Upgrades the memory model from Logical GLSL450 to Logical VulkanKHR.
//
// This pass remove deprecated decorations (Volatile and Coherent) and replaces
// them with new flags on individual instructions. It adds the Output storage
// class semantic to control barriers in tessellation control shaders that have
// an access to Output memory.
class UpgradeMemoryModel : public Pass {
 public:
  const char* name() const override { return "upgrade-memory-model"; }
  Status Process() override;

 private:
  // Used to indicate whether the operation performs an availability or
  // visibility operation.
  enum OperationType { kVisibility, kAvailability };

  // Used to indicate whether the instruction is a memory or image instruction.
  enum InstructionType { kMemory, kImage };

  // Modifies the OpMemoryModel to use VulkanKHR. Adds the Vulkan memory model
  // capability and extension.
  void UpgradeMemoryModelInstruction();

  // Upgrades memory, image and atomic instructions.
  // Memory and image instructions convert coherent and volatile decorations
  // into flags on the instruction.
  // Atomic memory semantics convert volatile decoration into flags on the
  // instruction.
  void UpgradeInstructions();

  // Upgrades memory and image operands for instructions that have them.
  void UpgradeMemoryAndImages();

  // Adds the volatile memory semantic if necessary.
  void UpgradeAtomics();

  // Returns whether |id| is coherent and/or volatile.
  std::tuple<bool, bool, spv::Scope> GetInstructionAttributes(uint32_t id);

  // Traces |inst| to determine if it is coherent and/or volatile.
  // |indices| tracks the access chain indices seen so far.
  std::pair<bool, bool> TraceInstruction(Instruction* inst,
                                         std::vector<uint32_t> indices,
                                         std::unordered_set<uint32_t>* visited);

  // Return true if |inst| is decorated with |decoration|.
  // If |inst| is decorated by member decorations then either |value| must
  // match the index or |value| must be a maximum allowable value. The max
  // value allows any element to match.
  bool HasDecoration(const Instruction* inst, uint32_t value,
                     spv::Decoration decoration);

  // Returns whether |type_id| indexed via |indices| is coherent and/or
  // volatile.
  std::pair<bool, bool> CheckType(uint32_t type_id,
                                  const std::vector<uint32_t>& indices);

  // Returns whether any type/element under |inst| is coherent and/or volatile.
  std::pair<bool, bool> CheckAllTypes(const Instruction* inst);

  // Modifies the flags of |inst| to include the new flags for the Vulkan
  // memory model. |operation_type| indicates whether flags should use
  // MakeVisible or MakeAvailable variants. |inst_type| indicates whether the
  // Pointer or Texel variants of flags should be used.
  void UpgradeFlags(Instruction* inst, uint32_t in_operand, bool is_coherent,
                    bool is_volatile, OperationType operation_type,
                    InstructionType inst_type);

  // Modifies the semantics at |in_operand| of |inst| to include the volatile
  // bit if |is_volatile| is true.
  void UpgradeSemantics(Instruction* inst, uint32_t in_operand,
                        bool is_volatile);

  // Returns the result id for a constant for |scope|.
  uint32_t GetScopeConstant(spv::Scope scope);

  // Returns the value of |index_inst|. |index_inst| must be an OpConstant of
  // integer type.g
  uint64_t GetIndexValue(Instruction* index_inst);

  // Removes coherent and volatile decorations.
  void CleanupDecorations();

  // For all tessellation control entry points, if there is an operation on
  // Output storage class, then all barriers are modified to include the
  // OutputMemoryKHR semantic.
  void UpgradeBarriers();

  // If the Vulkan memory model is specified, device scope actually means
  // device scope. The memory scope must be modified to be QueueFamilyKHR
  // scope.
  void UpgradeMemoryScope();

  // Returns true if |scope_id| is spv::Scope::Device.
  bool IsDeviceScope(uint32_t scope_id);

  // Upgrades GLSL.std.450 modf and frexp. Both instructions are replaced with
  // their struct versions. New extracts and a store are added in order to
  // facilitate adding memory model flags.
  void UpgradeExtInst(Instruction* modf);

  // Returns the number of words taken up by a memory access argument and its
  // implied operands.
  uint32_t MemoryAccessNumWords(uint32_t mask);

  // Caches the result of TraceInstruction. For a given result id and set of
  // indices, stores whether that combination is coherent and/or volatile.
  std::unordered_map<std::pair<uint32_t, std::vector<uint32_t>>,
                     std::pair<bool, bool>, CacheHash>
      cache_;
};
}  // namespace opt
}  // namespace spvtools
#endif  // LIBSPIRV_OPT_UPGRADE_MEMORY_MODEL_H_
