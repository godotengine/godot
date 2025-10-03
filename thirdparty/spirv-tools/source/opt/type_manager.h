// Copyright (c) 2016 Google Inc.
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

#ifndef SOURCE_OPT_TYPE_MANAGER_H_
#define SOURCE_OPT_TYPE_MANAGER_H_

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/opt/module.h"
#include "source/opt/types.h"
#include "spirv-tools/libspirv.hpp"

namespace spvtools {
namespace opt {

class IRContext;

namespace analysis {

// Hashing functor.
//
// All type pointers must be non-null.
struct HashTypePointer {
  size_t operator()(const Type* type) const {
    assert(type);
    return type->HashValue();
  }
};
struct HashTypeUniquePointer {
  size_t operator()(const std::unique_ptr<Type>& type) const {
    assert(type);
    return type->HashValue();
  }
};

// Equality functor.
//
// Checks if two types pointers are the same type.
//
// All type pointers must be non-null.
struct CompareTypePointers {
  bool operator()(const Type* lhs, const Type* rhs) const {
    assert(lhs && rhs);
    return lhs->IsSame(rhs);
  }
};
struct CompareTypeUniquePointers {
  bool operator()(const std::unique_ptr<Type>& lhs,
                  const std::unique_ptr<Type>& rhs) const {
    assert(lhs && rhs);
    return lhs->IsSame(rhs.get());
  }
};

// A class for managing the SPIR-V type hierarchy.
class TypeManager {
 public:
  using IdToTypeMap = std::unordered_map<uint32_t, Type*>;

  // Constructs a type manager from the given |module|. All internal messages
  // will be communicated to the outside via the given message |consumer|.
  // This instance only keeps a reference to the |consumer|, so the |consumer|
  // should outlive this instance.
  TypeManager(const MessageConsumer& consumer, IRContext* c);

  TypeManager(const TypeManager&) = delete;
  TypeManager(TypeManager&&) = delete;
  TypeManager& operator=(const TypeManager&) = delete;
  TypeManager& operator=(TypeManager&&) = delete;

  // Returns the type for the given type |id|. Returns nullptr if the given |id|
  // does not define a type.
  Type* GetType(uint32_t id) const;
  // Returns the id for the given |type|. Returns 0 if can not find the given
  // |type|.
  uint32_t GetId(const Type* type) const;
  // Returns the number of types hold in this manager.
  size_t NumTypes() const { return id_to_type_.size(); }
  // Iterators for all types contained in this manager.
  IdToTypeMap::const_iterator begin() const { return id_to_type_.cbegin(); }
  IdToTypeMap::const_iterator end() const { return id_to_type_.cend(); }

  // Returns a pair of the type and pointer to the type in |sc|.
  //
  // |id| must be a registered type.
  std::pair<Type*, std::unique_ptr<Pointer>> GetTypeAndPointerType(
      uint32_t id, spv::StorageClass sc) const;

  // Returns an id for a declaration representing |type|.  Returns 0 if the type
  // does not exists, and could not be generated.
  //
  // If |type| is registered, then the registered id is returned. Otherwise,
  // this function recursively adds type and annotation instructions as
  // necessary to fully define |type|.
  uint32_t GetTypeInstruction(const Type* type);

  // Find pointer to type and storage in module, return its resultId.  If it is
  // not found, a new type is created, and its id is returned.  Returns 0 if the
  // type could not be created.
  uint32_t FindPointerToType(uint32_t type_id, spv::StorageClass storage_class);

  // Registers |id| to |type|.
  //
  // If GetId(|type|) already returns a non-zero id, that mapping will be
  // unchanged.
  void RegisterType(uint32_t id, const Type& type);

  // Return the registered type object that is the same as |type|.
  Type* GetRegisteredType(const Type* type);

  // Removes knowledge of |id| from the manager.
  //
  // If |id| is an ambiguous type the multiple ids may be registered to |id|'s
  // type (e.g. %struct1 and %struct1 might hash to the same type). In that
  // case, calling GetId() with |id|'s type will return another suitable id
  // defining that type.
  void RemoveId(uint32_t id);

  // Returns the type of the member of |parent_type| that is identified by
  // |access_chain|.  The vector |access_chain| is a series of integers that are
  // used to pick members as in the |OpCompositeExtract| instructions.  If you
  // want a member of an array, vector, or matrix that does not have a constant
  // index, you can use 0 in that position.  All elements have the same type.
  const Type* GetMemberType(const Type* parent_type,
                            const std::vector<uint32_t>& access_chain);

  // Attaches the decoration encoded in |inst| to |type|. Does nothing if the
  // given instruction is not a decoration instruction. Assumes the target is
  // |type| (e.g. should be called in loop of |type|'s decorations).
  void AttachDecoration(const Instruction& inst, Type* type);

  Type* GetUIntType() {
    Integer int_type(32, false);
    return GetRegisteredType(&int_type);
  }

  uint32_t GetUIntTypeId() { return GetTypeInstruction(GetUIntType()); }

  Type* GetSIntType() {
    Integer int_type(32, true);
    return GetRegisteredType(&int_type);
  }

  uint32_t GetSIntTypeId() { return GetTypeInstruction(GetSIntType()); }

  Type* GetFloatType() {
    Float float_type(32);
    return GetRegisteredType(&float_type);
  }

  uint32_t GetFloatTypeId() { return GetTypeInstruction(GetFloatType()); }

  Type* GetDoubleType() {
    Float float_type(64);
    return GetRegisteredType(&float_type);
  }

  uint32_t GetDoubleTypeId() { return GetTypeInstruction(GetDoubleType()); }

  Type* GetUIntVectorType(uint32_t size) {
    Vector vec_type(GetUIntType(), size);
    return GetRegisteredType(&vec_type);
  }

  uint32_t GetUIntVectorTypeId(uint32_t size) {
    return GetTypeInstruction(GetUIntVectorType(size));
  }

  Type* GetSIntVectorType(uint32_t size) {
    Vector vec_type(GetSIntType(), size);
    return GetRegisteredType(&vec_type);
  }

  uint32_t GetSIntVectorTypeId(uint32_t size) {
    return GetTypeInstruction(GetSIntVectorType(size));
  }

  Type* GetFloatVectorType(uint32_t size) {
    Vector vec_type(GetFloatType(), size);
    return GetRegisteredType(&vec_type);
  }

  uint32_t GetFloatVectorTypeId(uint32_t size) {
    return GetTypeInstruction(GetFloatVectorType(size));
  }

  Type* GetBoolType() {
    Bool bool_type;
    return GetRegisteredType(&bool_type);
  }

  uint32_t GetBoolTypeId() { return GetTypeInstruction(GetBoolType()); }

  Type* GetVoidType() {
    Void void_type;
    return GetRegisteredType(&void_type);
  }

  uint32_t GetVoidTypeId() { return GetTypeInstruction(GetVoidType()); }

 private:
  using TypeToIdMap = std::unordered_map<const Type*, uint32_t, HashTypePointer,
                                         CompareTypePointers>;
  using TypePool =
      std::unordered_set<std::unique_ptr<Type>, HashTypeUniquePointer,
                         CompareTypeUniquePointers>;

  class UnresolvedType {
   public:
    UnresolvedType(uint32_t i, Type* t) : id_(i), type_(t) {}
    UnresolvedType(const UnresolvedType&) = delete;
    UnresolvedType(UnresolvedType&& that)
        : id_(that.id_), type_(std::move(that.type_)) {}

    uint32_t id() { return id_; }
    Type* type() { return type_.get(); }
    std::unique_ptr<Type>&& ReleaseType() { return std::move(type_); }
    void ResetType(Type* t) { type_.reset(t); }

   private:
    uint32_t id_;
    std::unique_ptr<Type> type_;
  };
  using IdToUnresolvedType = std::vector<UnresolvedType>;

  // Analyzes the types and decorations on types in the given |module|.
  void AnalyzeTypes(const Module& module);

  IRContext* context() { return context_; }

  // Attaches the decorations on |type| to |id|.
  void AttachDecorations(uint32_t id, const Type* type);

  // Create the annotation instruction.
  //
  // If |is_member| is false, an OpDecorate of |decoration| on |id| is created,
  // otherwise an OpMemberDecorate is created at |element|. The annotation is
  // registered with the DefUseManager and the DecorationManager.
  void CreateDecoration(uint32_t id, const std::vector<uint32_t>& decoration,
                        bool is_member = false, uint32_t element = 0);

  // Creates and returns a type from the given SPIR-V |inst|. Returns nullptr if
  // the given instruction is not for defining a type.
  Type* RecordIfTypeDefinition(const Instruction& inst);

  // Returns an equivalent pointer to |type| built in terms of pointers owned by
  // |type_pool_|. For example, if |type| is a vec3 of bool, it will be rebuilt
  // replacing the bool subtype with one owned by |type_pool_|.
  Type* RebuildType(const Type& type);

  // Completes the incomplete type |type|, by replaces all references to
  // ForwardPointer by the defining Pointer.
  void ReplaceForwardPointers(Type* type);

  // Replaces all references to |original_type| in |incomplete_types_| by
  // |new_type|.
  void ReplaceType(Type* new_type, Type* original_type);

  const MessageConsumer& consumer_;  // Message consumer.
  IRContext* context_;
  IdToTypeMap id_to_type_;  // Mapping from ids to their type representations.
  TypeToIdMap type_to_id_;  // Mapping from types to their defining ids.
  TypePool type_pool_;      // Memory owner of type pointers.
  IdToUnresolvedType incomplete_types_;  // All incomplete types.  Stored in an
                                         // std::vector to make traversals
                                         // deterministic.

  IdToTypeMap id_to_incomplete_type_;  // Maps ids to their type representations
                                       // for incomplete types.

  std::unordered_map<uint32_t, const Instruction*> id_to_constant_inst_;
};

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_TYPE_MANAGER_H_
