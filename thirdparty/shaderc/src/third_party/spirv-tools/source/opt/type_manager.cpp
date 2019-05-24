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

#include "source/opt/type_manager.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <utility>

#include "source/opt/ir_context.h"
#include "source/opt/log.h"
#include "source/opt/reflect.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace opt {
namespace analysis {
namespace {

const int kSpvTypePointerStorageClass = 1;
const int kSpvTypePointerTypeIdInIdx = 2;

}  // namespace

TypeManager::TypeManager(const MessageConsumer& consumer, IRContext* c)
    : consumer_(consumer), context_(c) {
  AnalyzeTypes(*c->module());
}

Type* TypeManager::GetType(uint32_t id) const {
  auto iter = id_to_type_.find(id);
  if (iter != id_to_type_.end()) return (*iter).second;
  iter = id_to_incomplete_type_.find(id);
  if (iter != id_to_incomplete_type_.end()) return (*iter).second;
  return nullptr;
}

std::pair<Type*, std::unique_ptr<Pointer>> TypeManager::GetTypeAndPointerType(
    uint32_t id, SpvStorageClass sc) const {
  Type* type = GetType(id);
  if (type) {
    return std::make_pair(type, MakeUnique<Pointer>(type, sc));
  } else {
    return std::make_pair(type, std::unique_ptr<Pointer>());
  }
}

uint32_t TypeManager::GetId(const Type* type) const {
  auto iter = type_to_id_.find(type);
  if (iter != type_to_id_.end()) {
    return (*iter).second;
  }
  return 0;
}

void TypeManager::AnalyzeTypes(const Module& module) {
  // First pass through the types.  Any types that reference a forward pointer
  // (directly or indirectly) are incomplete, and are added to incomplete types.
  for (const auto* inst : module.GetTypes()) {
    RecordIfTypeDefinition(*inst);
  }

  if (incomplete_types_.empty()) {
    return;
  }

  // Get the real pointer definition for all of the forward pointers.
  for (auto& type : incomplete_types_) {
    if (type.type()->kind() == Type::kForwardPointer) {
      auto* t = GetType(type.id());
      assert(t);
      auto* p = t->AsPointer();
      assert(p);
      type.type()->AsForwardPointer()->SetTargetPointer(p);
    }
  }

  // Replaces the references to the forward pointers in the incomplete types.
  for (auto& type : incomplete_types_) {
    ReplaceForwardPointers(type.type());
  }

  // Delete the forward pointers now that they are not referenced anymore.
  for (auto& type : incomplete_types_) {
    if (type.type()->kind() == Type::kForwardPointer) {
      type.ResetType(nullptr);
    }
  }

  // Compare the complete types looking for types that are the same.  If there
  // are two types that are the same, then replace one with the other.
  // Continue until we reach a fixed point.
  bool restart = true;
  while (restart) {
    restart = false;
    for (auto it1 = incomplete_types_.begin(); it1 != incomplete_types_.end();
         ++it1) {
      uint32_t id1 = it1->id();
      Type* type1 = it1->type();
      if (!type1) {
        continue;
      }

      for (auto it2 = it1 + 1; it2 != incomplete_types_.end(); ++it2) {
        uint32_t id2 = it2->id();
        (void)(id2 + id1);
        Type* type2 = it2->type();
        if (!type2) {
          continue;
        }

        if (type1->IsSame(type2)) {
          ReplaceType(type1, type2);
          it2->ResetType(nullptr);
          id_to_incomplete_type_[it2->id()] = type1;
          restart = true;
        }
      }
    }
  }

  // Add the remaining incomplete types to the type pool.
  for (auto& type : incomplete_types_) {
    if (type.type() && !type.type()->AsForwardPointer()) {
      std::vector<Instruction*> decorations =
          context()->get_decoration_mgr()->GetDecorationsFor(type.id(), true);
      for (auto dec : decorations) {
        AttachDecoration(*dec, type.type());
      }
      auto pair = type_pool_.insert(type.ReleaseType());
      id_to_type_[type.id()] = pair.first->get();
      type_to_id_[pair.first->get()] = type.id();
      id_to_incomplete_type_.erase(type.id());
    }
  }

  // Add a mapping for any ids that whose original type was replaced by an
  // equivalent type.
  for (auto& type : id_to_incomplete_type_) {
    id_to_type_[type.first] = type.second;
  }

#ifndef NDEBUG
  // Check if the type pool contains two types that are the same.  This
  // is an indication that the hashing and comparision are wrong.  It
  // will cause a problem if the type pool gets resized and everything
  // is rehashed.
  for (auto& i : type_pool_) {
    for (auto& j : type_pool_) {
      Type* ti = i.get();
      Type* tj = j.get();
      assert((ti == tj || !ti->IsSame(tj)) &&
             "Type pool contains two types that are the same.");
    }
  }
#endif
}

void TypeManager::RemoveId(uint32_t id) {
  auto iter = id_to_type_.find(id);
  if (iter == id_to_type_.end()) return;

  auto& type = iter->second;
  if (!type->IsUniqueType(true)) {
    auto tIter = type_to_id_.find(type);
    if (tIter != type_to_id_.end() && tIter->second == id) {
      // |type| currently maps to |id|.
      // Search for an equivalent type to re-map.
      bool found = false;
      for (auto& pair : id_to_type_) {
        if (pair.first != id && *pair.second == *type) {
          // Equivalent ambiguous type, re-map type.
          type_to_id_.erase(type);
          type_to_id_[pair.second] = pair.first;
          found = true;
          break;
        }
      }
      // No equivalent ambiguous type, remove mapping.
      if (!found) type_to_id_.erase(tIter);
    }
  } else {
    // Unique type, so just erase the entry.
    type_to_id_.erase(type);
  }

  // Erase the entry for |id|.
  id_to_type_.erase(iter);
}

uint32_t TypeManager::GetTypeInstruction(const Type* type) {
  uint32_t id = GetId(type);
  if (id != 0) return id;

  std::unique_ptr<Instruction> typeInst;
  // TODO(1841): Handle id overflow.
  id = context()->TakeNextId();
  RegisterType(id, *type);
  switch (type->kind()) {
#define DefineParameterlessCase(kind)                                     \
  case Type::k##kind:                                                     \
    typeInst = MakeUnique<Instruction>(context(), SpvOpType##kind, 0, id, \
                                       std::initializer_list<Operand>{}); \
    break;
    DefineParameterlessCase(Void);
    DefineParameterlessCase(Bool);
    DefineParameterlessCase(Sampler);
    DefineParameterlessCase(Event);
    DefineParameterlessCase(DeviceEvent);
    DefineParameterlessCase(ReserveId);
    DefineParameterlessCase(Queue);
    DefineParameterlessCase(PipeStorage);
    DefineParameterlessCase(NamedBarrier);
    DefineParameterlessCase(AccelerationStructureNV);
#undef DefineParameterlessCase
    case Type::kInteger:
      typeInst = MakeUnique<Instruction>(
          context(), SpvOpTypeInt, 0, id,
          std::initializer_list<Operand>{
              {SPV_OPERAND_TYPE_LITERAL_INTEGER, {type->AsInteger()->width()}},
              {SPV_OPERAND_TYPE_LITERAL_INTEGER,
               {(type->AsInteger()->IsSigned() ? 1u : 0u)}}});
      break;
    case Type::kFloat:
      typeInst = MakeUnique<Instruction>(
          context(), SpvOpTypeFloat, 0, id,
          std::initializer_list<Operand>{
              {SPV_OPERAND_TYPE_LITERAL_INTEGER, {type->AsFloat()->width()}}});
      break;
    case Type::kVector: {
      uint32_t subtype = GetTypeInstruction(type->AsVector()->element_type());
      typeInst =
          MakeUnique<Instruction>(context(), SpvOpTypeVector, 0, id,
                                  std::initializer_list<Operand>{
                                      {SPV_OPERAND_TYPE_ID, {subtype}},
                                      {SPV_OPERAND_TYPE_LITERAL_INTEGER,
                                       {type->AsVector()->element_count()}}});
      break;
    }
    case Type::kMatrix: {
      uint32_t subtype = GetTypeInstruction(type->AsMatrix()->element_type());
      typeInst =
          MakeUnique<Instruction>(context(), SpvOpTypeMatrix, 0, id,
                                  std::initializer_list<Operand>{
                                      {SPV_OPERAND_TYPE_ID, {subtype}},
                                      {SPV_OPERAND_TYPE_LITERAL_INTEGER,
                                       {type->AsMatrix()->element_count()}}});
      break;
    }
    case Type::kImage: {
      const Image* image = type->AsImage();
      uint32_t subtype = GetTypeInstruction(image->sampled_type());
      typeInst = MakeUnique<Instruction>(
          context(), SpvOpTypeImage, 0, id,
          std::initializer_list<Operand>{
              {SPV_OPERAND_TYPE_ID, {subtype}},
              {SPV_OPERAND_TYPE_DIMENSIONALITY,
               {static_cast<uint32_t>(image->dim())}},
              {SPV_OPERAND_TYPE_LITERAL_INTEGER, {image->depth()}},
              {SPV_OPERAND_TYPE_LITERAL_INTEGER,
               {(image->is_arrayed() ? 1u : 0u)}},
              {SPV_OPERAND_TYPE_LITERAL_INTEGER,
               {(image->is_multisampled() ? 1u : 0u)}},
              {SPV_OPERAND_TYPE_LITERAL_INTEGER, {image->sampled()}},
              {SPV_OPERAND_TYPE_SAMPLER_IMAGE_FORMAT,
               {static_cast<uint32_t>(image->format())}},
              {SPV_OPERAND_TYPE_ACCESS_QUALIFIER,
               {static_cast<uint32_t>(image->access_qualifier())}}});
      break;
    }
    case Type::kSampledImage: {
      uint32_t subtype =
          GetTypeInstruction(type->AsSampledImage()->image_type());
      typeInst = MakeUnique<Instruction>(
          context(), SpvOpTypeSampledImage, 0, id,
          std::initializer_list<Operand>{{SPV_OPERAND_TYPE_ID, {subtype}}});
      break;
    }
    case Type::kArray: {
      uint32_t subtype = GetTypeInstruction(type->AsArray()->element_type());
      typeInst = MakeUnique<Instruction>(
          context(), SpvOpTypeArray, 0, id,
          std::initializer_list<Operand>{
              {SPV_OPERAND_TYPE_ID, {subtype}},
              {SPV_OPERAND_TYPE_ID, {type->AsArray()->LengthId()}}});
      break;
    }
    case Type::kRuntimeArray: {
      uint32_t subtype =
          GetTypeInstruction(type->AsRuntimeArray()->element_type());
      typeInst = MakeUnique<Instruction>(
          context(), SpvOpTypeRuntimeArray, 0, id,
          std::initializer_list<Operand>{{SPV_OPERAND_TYPE_ID, {subtype}}});
      break;
    }
    case Type::kStruct: {
      std::vector<Operand> ops;
      const Struct* structTy = type->AsStruct();
      for (auto ty : structTy->element_types()) {
        ops.push_back(Operand(SPV_OPERAND_TYPE_ID, {GetTypeInstruction(ty)}));
      }
      typeInst =
          MakeUnique<Instruction>(context(), SpvOpTypeStruct, 0, id, ops);
      break;
    }
    case Type::kOpaque: {
      const Opaque* opaque = type->AsOpaque();
      size_t size = opaque->name().size();
      // Convert to null-terminated packed UTF-8 string.
      std::vector<uint32_t> words(size / 4 + 1, 0);
      char* dst = reinterpret_cast<char*>(words.data());
      strncpy(dst, opaque->name().c_str(), size);
      typeInst = MakeUnique<Instruction>(
          context(), SpvOpTypeOpaque, 0, id,
          std::initializer_list<Operand>{
              {SPV_OPERAND_TYPE_LITERAL_STRING, words}});
      break;
    }
    case Type::kPointer: {
      const Pointer* pointer = type->AsPointer();
      uint32_t subtype = GetTypeInstruction(pointer->pointee_type());
      typeInst = MakeUnique<Instruction>(
          context(), SpvOpTypePointer, 0, id,
          std::initializer_list<Operand>{
              {SPV_OPERAND_TYPE_STORAGE_CLASS,
               {static_cast<uint32_t>(pointer->storage_class())}},
              {SPV_OPERAND_TYPE_ID, {subtype}}});
      break;
    }
    case Type::kFunction: {
      std::vector<Operand> ops;
      const Function* function = type->AsFunction();
      ops.push_back(Operand(SPV_OPERAND_TYPE_ID,
                            {GetTypeInstruction(function->return_type())}));
      for (auto ty : function->param_types()) {
        ops.push_back(Operand(SPV_OPERAND_TYPE_ID, {GetTypeInstruction(ty)}));
      }
      typeInst =
          MakeUnique<Instruction>(context(), SpvOpTypeFunction, 0, id, ops);
      break;
    }
    case Type::kPipe:
      typeInst = MakeUnique<Instruction>(
          context(), SpvOpTypePipe, 0, id,
          std::initializer_list<Operand>{
              {SPV_OPERAND_TYPE_ACCESS_QUALIFIER,
               {static_cast<uint32_t>(type->AsPipe()->access_qualifier())}}});
      break;
    case Type::kForwardPointer:
      typeInst = MakeUnique<Instruction>(
          context(), SpvOpTypeForwardPointer, 0, 0,
          std::initializer_list<Operand>{
              {SPV_OPERAND_TYPE_ID, {type->AsForwardPointer()->target_id()}},
              {SPV_OPERAND_TYPE_STORAGE_CLASS,
               {static_cast<uint32_t>(
                   type->AsForwardPointer()->storage_class())}}});
      break;
    default:
      assert(false && "Unexpected type");
      break;
  }
  context()->AddType(std::move(typeInst));
  context()->AnalyzeDefUse(&*--context()->types_values_end());
  AttachDecorations(id, type);
  return id;
}

uint32_t TypeManager::FindPointerToType(uint32_t type_id,
                                        SpvStorageClass storage_class) {
  Type* pointeeTy = GetType(type_id);
  Pointer pointerTy(pointeeTy, storage_class);
  if (pointeeTy->IsUniqueType(true)) {
    // Non-ambiguous type. Get the pointer type through the type manager.
    return GetTypeInstruction(&pointerTy);
  }

  // Ambiguous type, do a linear search.
  Module::inst_iterator type_itr = context()->module()->types_values_begin();
  for (; type_itr != context()->module()->types_values_end(); ++type_itr) {
    const Instruction* type_inst = &*type_itr;
    if (type_inst->opcode() == SpvOpTypePointer &&
        type_inst->GetSingleWordOperand(kSpvTypePointerTypeIdInIdx) ==
            type_id &&
        type_inst->GetSingleWordOperand(kSpvTypePointerStorageClass) ==
            storage_class)
      return type_inst->result_id();
  }

  // Must create the pointer type.
  // TODO(1841): Handle id overflow.
  uint32_t resultId = context()->TakeNextId();
  std::unique_ptr<Instruction> type_inst(
      new Instruction(context(), SpvOpTypePointer, 0, resultId,
                      {{spv_operand_type_t::SPV_OPERAND_TYPE_STORAGE_CLASS,
                        {uint32_t(storage_class)}},
                       {spv_operand_type_t::SPV_OPERAND_TYPE_ID, {type_id}}}));
  context()->AddType(std::move(type_inst));
  context()->get_type_mgr()->RegisterType(resultId, pointerTy);
  return resultId;
}

void TypeManager::AttachDecorations(uint32_t id, const Type* type) {
  for (auto vec : type->decorations()) {
    CreateDecoration(id, vec);
  }
  if (const Struct* structTy = type->AsStruct()) {
    for (auto pair : structTy->element_decorations()) {
      uint32_t element = pair.first;
      for (auto vec : pair.second) {
        CreateDecoration(id, vec, element);
      }
    }
  }
}

void TypeManager::CreateDecoration(uint32_t target,
                                   const std::vector<uint32_t>& decoration,
                                   uint32_t element) {
  std::vector<Operand> ops;
  ops.push_back(Operand(SPV_OPERAND_TYPE_ID, {target}));
  if (element != 0) {
    ops.push_back(Operand(SPV_OPERAND_TYPE_LITERAL_INTEGER, {element}));
  }
  ops.push_back(Operand(SPV_OPERAND_TYPE_DECORATION, {decoration[0]}));
  for (size_t i = 1; i < decoration.size(); ++i) {
    ops.push_back(Operand(SPV_OPERAND_TYPE_LITERAL_INTEGER, {decoration[i]}));
  }
  context()->AddAnnotationInst(MakeUnique<Instruction>(
      context(), (element == 0 ? SpvOpDecorate : SpvOpMemberDecorate), 0, 0,
      ops));
  Instruction* inst = &*--context()->annotation_end();
  context()->get_def_use_mgr()->AnalyzeInstUse(inst);
}

Type* TypeManager::RebuildType(const Type& type) {
  // The comparison and hash on the type pool will avoid inserting the rebuilt
  // type if an equivalent type already exists. The rebuilt type will be deleted
  // when it goes out of scope at the end of the function in that case. Repeated
  // insertions of the same Type will, at most, keep one corresponding object in
  // the type pool.
  std::unique_ptr<Type> rebuilt_ty;
  switch (type.kind()) {
#define DefineNoSubtypeCase(kind)             \
  case Type::k##kind:                         \
    rebuilt_ty.reset(type.Clone().release()); \
    return type_pool_.insert(std::move(rebuilt_ty)).first->get();

    DefineNoSubtypeCase(Void);
    DefineNoSubtypeCase(Bool);
    DefineNoSubtypeCase(Integer);
    DefineNoSubtypeCase(Float);
    DefineNoSubtypeCase(Sampler);
    DefineNoSubtypeCase(Opaque);
    DefineNoSubtypeCase(Event);
    DefineNoSubtypeCase(DeviceEvent);
    DefineNoSubtypeCase(ReserveId);
    DefineNoSubtypeCase(Queue);
    DefineNoSubtypeCase(Pipe);
    DefineNoSubtypeCase(PipeStorage);
    DefineNoSubtypeCase(NamedBarrier);
    DefineNoSubtypeCase(AccelerationStructureNV);
#undef DefineNoSubtypeCase
    case Type::kVector: {
      const Vector* vec_ty = type.AsVector();
      const Type* ele_ty = vec_ty->element_type();
      rebuilt_ty =
          MakeUnique<Vector>(RebuildType(*ele_ty), vec_ty->element_count());
      break;
    }
    case Type::kMatrix: {
      const Matrix* mat_ty = type.AsMatrix();
      const Type* ele_ty = mat_ty->element_type();
      rebuilt_ty =
          MakeUnique<Matrix>(RebuildType(*ele_ty), mat_ty->element_count());
      break;
    }
    case Type::kImage: {
      const Image* image_ty = type.AsImage();
      const Type* ele_ty = image_ty->sampled_type();
      rebuilt_ty =
          MakeUnique<Image>(RebuildType(*ele_ty), image_ty->dim(),
                            image_ty->depth(), image_ty->is_arrayed(),
                            image_ty->is_multisampled(), image_ty->sampled(),
                            image_ty->format(), image_ty->access_qualifier());
      break;
    }
    case Type::kSampledImage: {
      const SampledImage* image_ty = type.AsSampledImage();
      const Type* ele_ty = image_ty->image_type();
      rebuilt_ty = MakeUnique<SampledImage>(RebuildType(*ele_ty));
      break;
    }
    case Type::kArray: {
      const Array* array_ty = type.AsArray();
      const Type* ele_ty = array_ty->element_type();
      rebuilt_ty =
          MakeUnique<Array>(RebuildType(*ele_ty), array_ty->LengthId());
      break;
    }
    case Type::kRuntimeArray: {
      const RuntimeArray* array_ty = type.AsRuntimeArray();
      const Type* ele_ty = array_ty->element_type();
      rebuilt_ty = MakeUnique<RuntimeArray>(RebuildType(*ele_ty));
      break;
    }
    case Type::kStruct: {
      const Struct* struct_ty = type.AsStruct();
      std::vector<const Type*> subtypes;
      subtypes.reserve(struct_ty->element_types().size());
      for (const auto* ele_ty : struct_ty->element_types()) {
        subtypes.push_back(RebuildType(*ele_ty));
      }
      rebuilt_ty = MakeUnique<Struct>(subtypes);
      Struct* rebuilt_struct = rebuilt_ty->AsStruct();
      for (auto pair : struct_ty->element_decorations()) {
        uint32_t index = pair.first;
        for (const auto& dec : pair.second) {
          // Explicit copy intended.
          std::vector<uint32_t> copy(dec);
          rebuilt_struct->AddMemberDecoration(index, std::move(copy));
        }
      }
      break;
    }
    case Type::kPointer: {
      const Pointer* pointer_ty = type.AsPointer();
      const Type* ele_ty = pointer_ty->pointee_type();
      rebuilt_ty = MakeUnique<Pointer>(RebuildType(*ele_ty),
                                       pointer_ty->storage_class());
      break;
    }
    case Type::kFunction: {
      const Function* function_ty = type.AsFunction();
      const Type* ret_ty = function_ty->return_type();
      std::vector<const Type*> param_types;
      param_types.reserve(function_ty->param_types().size());
      for (const auto* param_ty : function_ty->param_types()) {
        param_types.push_back(RebuildType(*param_ty));
      }
      rebuilt_ty = MakeUnique<Function>(RebuildType(*ret_ty), param_types);
      break;
    }
    case Type::kForwardPointer: {
      const ForwardPointer* forward_ptr_ty = type.AsForwardPointer();
      rebuilt_ty = MakeUnique<ForwardPointer>(forward_ptr_ty->target_id(),
                                              forward_ptr_ty->storage_class());
      const Pointer* target_ptr = forward_ptr_ty->target_pointer();
      if (target_ptr) {
        rebuilt_ty->AsForwardPointer()->SetTargetPointer(
            RebuildType(*target_ptr)->AsPointer());
      }
      break;
    }
    default:
      assert(false && "Unhandled type");
      return nullptr;
  }
  for (const auto& dec : type.decorations()) {
    // Explicit copy intended.
    std::vector<uint32_t> copy(dec);
    rebuilt_ty->AddDecoration(std::move(copy));
  }

  return type_pool_.insert(std::move(rebuilt_ty)).first->get();
}

void TypeManager::RegisterType(uint32_t id, const Type& type) {
  // Rebuild |type| so it and all its constituent types are owned by the type
  // pool.
  Type* rebuilt = RebuildType(type);
  assert(rebuilt->IsSame(&type));
  id_to_type_[id] = rebuilt;
  if (GetId(rebuilt) == 0) {
    type_to_id_[rebuilt] = id;
  }
}

Type* TypeManager::GetRegisteredType(const Type* type) {
  uint32_t id = GetTypeInstruction(type);
  return GetType(id);
}

Type* TypeManager::RecordIfTypeDefinition(const Instruction& inst) {
  if (!IsTypeInst(inst.opcode())) return nullptr;

  Type* type = nullptr;
  switch (inst.opcode()) {
    case SpvOpTypeVoid:
      type = new Void();
      break;
    case SpvOpTypeBool:
      type = new Bool();
      break;
    case SpvOpTypeInt:
      type = new Integer(inst.GetSingleWordInOperand(0),
                         inst.GetSingleWordInOperand(1));
      break;
    case SpvOpTypeFloat:
      type = new Float(inst.GetSingleWordInOperand(0));
      break;
    case SpvOpTypeVector:
      type = new Vector(GetType(inst.GetSingleWordInOperand(0)),
                        inst.GetSingleWordInOperand(1));
      break;
    case SpvOpTypeMatrix:
      type = new Matrix(GetType(inst.GetSingleWordInOperand(0)),
                        inst.GetSingleWordInOperand(1));
      break;
    case SpvOpTypeImage: {
      const SpvAccessQualifier access =
          inst.NumInOperands() < 8
              ? SpvAccessQualifierReadOnly
              : static_cast<SpvAccessQualifier>(inst.GetSingleWordInOperand(7));
      type = new Image(
          GetType(inst.GetSingleWordInOperand(0)),
          static_cast<SpvDim>(inst.GetSingleWordInOperand(1)),
          inst.GetSingleWordInOperand(2), inst.GetSingleWordInOperand(3) == 1,
          inst.GetSingleWordInOperand(4) == 1, inst.GetSingleWordInOperand(5),
          static_cast<SpvImageFormat>(inst.GetSingleWordInOperand(6)), access);
    } break;
    case SpvOpTypeSampler:
      type = new Sampler();
      break;
    case SpvOpTypeSampledImage:
      type = new SampledImage(GetType(inst.GetSingleWordInOperand(0)));
      break;
    case SpvOpTypeArray:
      type = new Array(GetType(inst.GetSingleWordInOperand(0)),
                       inst.GetSingleWordInOperand(1));
      if (id_to_incomplete_type_.count(inst.GetSingleWordInOperand(0))) {
        incomplete_types_.emplace_back(inst.result_id(), type);
        id_to_incomplete_type_[inst.result_id()] = type;
        return type;
      }
      break;
    case SpvOpTypeRuntimeArray:
      type = new RuntimeArray(GetType(inst.GetSingleWordInOperand(0)));
      if (id_to_incomplete_type_.count(inst.GetSingleWordInOperand(0))) {
        incomplete_types_.emplace_back(inst.result_id(), type);
        id_to_incomplete_type_[inst.result_id()] = type;
        return type;
      }
      break;
    case SpvOpTypeStruct: {
      std::vector<const Type*> element_types;
      bool incomplete_type = false;
      for (uint32_t i = 0; i < inst.NumInOperands(); ++i) {
        uint32_t type_id = inst.GetSingleWordInOperand(i);
        element_types.push_back(GetType(type_id));
        if (id_to_incomplete_type_.count(type_id)) {
          incomplete_type = true;
        }
      }
      type = new Struct(element_types);

      if (incomplete_type) {
        incomplete_types_.emplace_back(inst.result_id(), type);
        id_to_incomplete_type_[inst.result_id()] = type;
        return type;
      }
    } break;
    case SpvOpTypeOpaque: {
      const uint32_t* data = inst.GetInOperand(0).words.data();
      type = new Opaque(reinterpret_cast<const char*>(data));
    } break;
    case SpvOpTypePointer: {
      uint32_t pointee_type_id = inst.GetSingleWordInOperand(1);
      type = new Pointer(
          GetType(pointee_type_id),
          static_cast<SpvStorageClass>(inst.GetSingleWordInOperand(0)));

      if (id_to_incomplete_type_.count(pointee_type_id)) {
        incomplete_types_.emplace_back(inst.result_id(), type);
        id_to_incomplete_type_[inst.result_id()] = type;
        return type;
      }
      id_to_incomplete_type_.erase(inst.result_id());

    } break;
    case SpvOpTypeFunction: {
      bool incomplete_type = false;
      uint32_t return_type_id = inst.GetSingleWordInOperand(0);
      if (id_to_incomplete_type_.count(return_type_id)) {
        incomplete_type = true;
      }
      Type* return_type = GetType(return_type_id);
      std::vector<const Type*> param_types;
      for (uint32_t i = 1; i < inst.NumInOperands(); ++i) {
        uint32_t param_type_id = inst.GetSingleWordInOperand(i);
        param_types.push_back(GetType(param_type_id));
        if (id_to_incomplete_type_.count(param_type_id)) {
          incomplete_type = true;
        }
      }

      type = new Function(return_type, param_types);

      if (incomplete_type) {
        incomplete_types_.emplace_back(inst.result_id(), type);
        id_to_incomplete_type_[inst.result_id()] = type;
        return type;
      }
    } break;
    case SpvOpTypeEvent:
      type = new Event();
      break;
    case SpvOpTypeDeviceEvent:
      type = new DeviceEvent();
      break;
    case SpvOpTypeReserveId:
      type = new ReserveId();
      break;
    case SpvOpTypeQueue:
      type = new Queue();
      break;
    case SpvOpTypePipe:
      type = new Pipe(
          static_cast<SpvAccessQualifier>(inst.GetSingleWordInOperand(0)));
      break;
    case SpvOpTypeForwardPointer: {
      // Handling of forward pointers is different from the other types.
      uint32_t target_id = inst.GetSingleWordInOperand(0);
      type = new ForwardPointer(target_id, static_cast<SpvStorageClass>(
                                               inst.GetSingleWordInOperand(1)));
      incomplete_types_.emplace_back(target_id, type);
      id_to_incomplete_type_[target_id] = type;
      return type;
    }
    case SpvOpTypePipeStorage:
      type = new PipeStorage();
      break;
    case SpvOpTypeNamedBarrier:
      type = new NamedBarrier();
      break;
    case SpvOpTypeAccelerationStructureNV:
      type = new AccelerationStructureNV();
      break;
    default:
      SPIRV_UNIMPLEMENTED(consumer_, "unhandled type");
      break;
  }

  uint32_t id = inst.result_id();
  SPIRV_ASSERT(consumer_, id != 0, "instruction without result id found");
  SPIRV_ASSERT(consumer_, type != nullptr,
               "type should not be nullptr at this point");
  std::vector<Instruction*> decorations =
      context()->get_decoration_mgr()->GetDecorationsFor(id, true);
  for (auto dec : decorations) {
    AttachDecoration(*dec, type);
  }
  std::unique_ptr<Type> unique(type);
  auto pair = type_pool_.insert(std::move(unique));
  id_to_type_[id] = pair.first->get();
  type_to_id_[pair.first->get()] = id;
  return type;
}

void TypeManager::AttachDecoration(const Instruction& inst, Type* type) {
  const SpvOp opcode = inst.opcode();
  if (!IsAnnotationInst(opcode)) return;

  switch (opcode) {
    case SpvOpDecorate: {
      const auto count = inst.NumOperands();
      std::vector<uint32_t> data;
      for (uint32_t i = 1; i < count; ++i) {
        data.push_back(inst.GetSingleWordOperand(i));
      }
      type->AddDecoration(std::move(data));
    } break;
    case SpvOpMemberDecorate: {
      const auto count = inst.NumOperands();
      const uint32_t index = inst.GetSingleWordOperand(1);
      std::vector<uint32_t> data;
      for (uint32_t i = 2; i < count; ++i) {
        data.push_back(inst.GetSingleWordOperand(i));
      }
      if (Struct* st = type->AsStruct()) {
        st->AddMemberDecoration(index, std::move(data));
      } else {
        SPIRV_UNIMPLEMENTED(consumer_, "OpMemberDecorate non-struct type");
      }
    } break;
    default:
      SPIRV_UNREACHABLE(consumer_);
      break;
  }
}

const Type* TypeManager::GetMemberType(
    const Type* parent_type, const std::vector<uint32_t>& access_chain) {
  for (uint32_t element_index : access_chain) {
    if (const Struct* struct_type = parent_type->AsStruct()) {
      parent_type = struct_type->element_types()[element_index];
    } else if (const Array* array_type = parent_type->AsArray()) {
      parent_type = array_type->element_type();
    } else if (const RuntimeArray* runtime_array_type =
                   parent_type->AsRuntimeArray()) {
      parent_type = runtime_array_type->element_type();
    } else if (const Vector* vector_type = parent_type->AsVector()) {
      parent_type = vector_type->element_type();
    } else if (const Matrix* matrix_type = parent_type->AsMatrix()) {
      parent_type = matrix_type->element_type();
    } else {
      assert(false && "Trying to get a member of a type without members.");
    }
  }
  return parent_type;
}

void TypeManager::ReplaceForwardPointers(Type* type) {
  switch (type->kind()) {
    case Type::kArray: {
      const ForwardPointer* element_type =
          type->AsArray()->element_type()->AsForwardPointer();
      if (element_type) {
        type->AsArray()->ReplaceElementType(element_type->target_pointer());
      }
    } break;
    case Type::kRuntimeArray: {
      const ForwardPointer* element_type =
          type->AsRuntimeArray()->element_type()->AsForwardPointer();
      if (element_type) {
        type->AsRuntimeArray()->ReplaceElementType(
            element_type->target_pointer());
      }
    } break;
    case Type::kStruct: {
      auto& member_types = type->AsStruct()->element_types();
      for (auto& member_type : member_types) {
        if (member_type->AsForwardPointer()) {
          member_type = member_type->AsForwardPointer()->target_pointer();
          assert(member_type);
        }
      }
    } break;
    case Type::kPointer: {
      const ForwardPointer* pointee_type =
          type->AsPointer()->pointee_type()->AsForwardPointer();
      if (pointee_type) {
        type->AsPointer()->SetPointeeType(pointee_type->target_pointer());
      }
    } break;
    case Type::kFunction: {
      Function* func_type = type->AsFunction();
      const ForwardPointer* return_type =
          func_type->return_type()->AsForwardPointer();
      if (return_type) {
        func_type->SetReturnType(return_type->target_pointer());
      }

      auto& param_types = func_type->param_types();
      for (auto& param_type : param_types) {
        if (param_type->AsForwardPointer()) {
          param_type = param_type->AsForwardPointer()->target_pointer();
        }
      }
    } break;
    default:
      break;
  }
}

void TypeManager::ReplaceType(Type* new_type, Type* original_type) {
  assert(original_type->kind() == new_type->kind() &&
         "Types must be the same for replacement.\n");
  for (auto& p : incomplete_types_) {
    Type* type = p.type();
    if (!type) {
      continue;
    }

    switch (type->kind()) {
      case Type::kArray: {
        const Type* element_type = type->AsArray()->element_type();
        if (element_type == original_type) {
          type->AsArray()->ReplaceElementType(new_type);
        }
      } break;
      case Type::kRuntimeArray: {
        const Type* element_type = type->AsRuntimeArray()->element_type();
        if (element_type == original_type) {
          type->AsRuntimeArray()->ReplaceElementType(new_type);
        }
      } break;
      case Type::kStruct: {
        auto& member_types = type->AsStruct()->element_types();
        for (auto& member_type : member_types) {
          if (member_type == original_type) {
            member_type = new_type;
          }
        }
      } break;
      case Type::kPointer: {
        const Type* pointee_type = type->AsPointer()->pointee_type();
        if (pointee_type == original_type) {
          type->AsPointer()->SetPointeeType(new_type);
        }
      } break;
      case Type::kFunction: {
        Function* func_type = type->AsFunction();
        const Type* return_type = func_type->return_type();
        if (return_type == original_type) {
          func_type->SetReturnType(new_type);
        }

        auto& param_types = func_type->param_types();
        for (auto& param_type : param_types) {
          if (param_type == original_type) {
            param_type = new_type;
          }
        }
      } break;
      default:
        break;
    }
  }
}

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
