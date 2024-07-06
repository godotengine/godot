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

#include "source/opt/types.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <sstream>
#include <string>
#include <unordered_set>

#include "source/util/hash_combine.h"
#include "source/util/make_unique.h"

namespace spvtools {
namespace opt {
namespace analysis {

using spvtools::utils::hash_combine;
using U32VecVec = std::vector<std::vector<uint32_t>>;

namespace {

// Returns true if the two vector of vectors are identical.
bool CompareTwoVectors(const U32VecVec a, const U32VecVec b) {
  const auto size = a.size();
  if (size != b.size()) return false;

  if (size == 0) return true;
  if (size == 1) return a.front() == b.front();

  std::vector<const std::vector<uint32_t>*> a_ptrs, b_ptrs;
  a_ptrs.reserve(size);
  a_ptrs.reserve(size);
  for (uint32_t i = 0; i < size; ++i) {
    a_ptrs.push_back(&a[i]);
    b_ptrs.push_back(&b[i]);
  }

  const auto cmp = [](const std::vector<uint32_t>* m,
                      const std::vector<uint32_t>* n) {
    return m->front() < n->front();
  };

  std::sort(a_ptrs.begin(), a_ptrs.end(), cmp);
  std::sort(b_ptrs.begin(), b_ptrs.end(), cmp);

  for (uint32_t i = 0; i < size; ++i) {
    if (*a_ptrs[i] != *b_ptrs[i]) return false;
  }
  return true;
}

}  // namespace

std::string Type::GetDecorationStr() const {
  std::ostringstream oss;
  oss << "[[";
  for (const auto& decoration : decorations_) {
    oss << "(";
    for (size_t i = 0; i < decoration.size(); ++i) {
      oss << (i > 0 ? ", " : "");
      oss << decoration.at(i);
    }
    oss << ")";
  }
  oss << "]]";
  return oss.str();
}

bool Type::HasSameDecorations(const Type* that) const {
  return CompareTwoVectors(decorations_, that->decorations_);
}

bool Type::IsUniqueType() const {
  switch (kind_) {
    case kPointer:
    case kStruct:
    case kArray:
    case kRuntimeArray:
      return false;
    default:
      return true;
  }
}

std::unique_ptr<Type> Type::Clone() const {
  std::unique_ptr<Type> type;
  switch (kind_) {
#define DeclareKindCase(kind)                   \
  case k##kind:                                 \
    type = MakeUnique<kind>(*this->As##kind()); \
    break
    DeclareKindCase(Void);
    DeclareKindCase(Bool);
    DeclareKindCase(Integer);
    DeclareKindCase(Float);
    DeclareKindCase(Vector);
    DeclareKindCase(Matrix);
    DeclareKindCase(Image);
    DeclareKindCase(Sampler);
    DeclareKindCase(SampledImage);
    DeclareKindCase(Array);
    DeclareKindCase(RuntimeArray);
    DeclareKindCase(Struct);
    DeclareKindCase(Opaque);
    DeclareKindCase(Pointer);
    DeclareKindCase(Function);
    DeclareKindCase(Event);
    DeclareKindCase(DeviceEvent);
    DeclareKindCase(ReserveId);
    DeclareKindCase(Queue);
    DeclareKindCase(Pipe);
    DeclareKindCase(ForwardPointer);
    DeclareKindCase(PipeStorage);
    DeclareKindCase(NamedBarrier);
    DeclareKindCase(AccelerationStructureNV);
    DeclareKindCase(CooperativeMatrixNV);
    DeclareKindCase(CooperativeMatrixKHR);
    DeclareKindCase(RayQueryKHR);
    DeclareKindCase(HitObjectNV);
#undef DeclareKindCase
    default:
      assert(false && "Unhandled type");
  }
  return type;
}

std::unique_ptr<Type> Type::RemoveDecorations() const {
  std::unique_ptr<Type> type(Clone());
  type->ClearDecorations();
  return type;
}

bool Type::operator==(const Type& other) const {
  if (kind_ != other.kind_) return false;

  switch (kind_) {
#define DeclareKindCase(kind) \
  case k##kind:               \
    return As##kind()->IsSame(&other)
    DeclareKindCase(Void);
    DeclareKindCase(Bool);
    DeclareKindCase(Integer);
    DeclareKindCase(Float);
    DeclareKindCase(Vector);
    DeclareKindCase(Matrix);
    DeclareKindCase(Image);
    DeclareKindCase(Sampler);
    DeclareKindCase(SampledImage);
    DeclareKindCase(Array);
    DeclareKindCase(RuntimeArray);
    DeclareKindCase(Struct);
    DeclareKindCase(Opaque);
    DeclareKindCase(Pointer);
    DeclareKindCase(Function);
    DeclareKindCase(Event);
    DeclareKindCase(DeviceEvent);
    DeclareKindCase(ReserveId);
    DeclareKindCase(Queue);
    DeclareKindCase(Pipe);
    DeclareKindCase(ForwardPointer);
    DeclareKindCase(PipeStorage);
    DeclareKindCase(NamedBarrier);
    DeclareKindCase(AccelerationStructureNV);
    DeclareKindCase(CooperativeMatrixNV);
    DeclareKindCase(CooperativeMatrixKHR);
    DeclareKindCase(RayQueryKHR);
    DeclareKindCase(HitObjectNV);
#undef DeclareKindCase
    default:
      assert(false && "Unhandled type");
      return false;
  }
}

size_t Type::ComputeHashValue(size_t hash, SeenTypes* seen) const {
  // Linear search through a dense, cache coherent vector is faster than O(log
  // n) search in a complex data structure (eg std::set) for the generally small
  // number of nodes.  It also skips the overhead of an new/delete per Type
  // (when inserting/removing from a set).
  if (std::find(seen->begin(), seen->end(), this) != seen->end()) {
    return hash;
  }

  seen->push_back(this);

  hash = hash_combine(hash, uint32_t(kind_));
  for (const auto& d : decorations_) {
    hash = hash_combine(hash, d);
  }

  switch (kind_) {
#define DeclareKindCase(type)                             \
  case k##type:                                           \
    hash = As##type()->ComputeExtraStateHash(hash, seen); \
    break
    DeclareKindCase(Void);
    DeclareKindCase(Bool);
    DeclareKindCase(Integer);
    DeclareKindCase(Float);
    DeclareKindCase(Vector);
    DeclareKindCase(Matrix);
    DeclareKindCase(Image);
    DeclareKindCase(Sampler);
    DeclareKindCase(SampledImage);
    DeclareKindCase(Array);
    DeclareKindCase(RuntimeArray);
    DeclareKindCase(Struct);
    DeclareKindCase(Opaque);
    DeclareKindCase(Pointer);
    DeclareKindCase(Function);
    DeclareKindCase(Event);
    DeclareKindCase(DeviceEvent);
    DeclareKindCase(ReserveId);
    DeclareKindCase(Queue);
    DeclareKindCase(Pipe);
    DeclareKindCase(ForwardPointer);
    DeclareKindCase(PipeStorage);
    DeclareKindCase(NamedBarrier);
    DeclareKindCase(AccelerationStructureNV);
    DeclareKindCase(CooperativeMatrixNV);
    DeclareKindCase(CooperativeMatrixKHR);
    DeclareKindCase(RayQueryKHR);
    DeclareKindCase(HitObjectNV);
#undef DeclareKindCase
    default:
      assert(false && "Unhandled type");
      break;
  }

  seen->pop_back();
  return hash;
}

size_t Type::HashValue() const {
  SeenTypes seen;
  return ComputeHashValue(0, &seen);
}

uint64_t Type::NumberOfComponents() const {
  switch (kind()) {
    case kVector:
      return AsVector()->element_count();
    case kMatrix:
      return AsMatrix()->element_count();
    case kArray: {
      Array::LengthInfo length_info = AsArray()->length_info();
      if (length_info.words[0] != Array::LengthInfo::kConstant) {
        return UINT64_MAX;
      }
      assert(length_info.words.size() <= 3 &&
             "The size of the array could not fit size_t.");
      uint64_t length = 0;
      length |= length_info.words[1];
      if (length_info.words.size() > 2) {
        length |= static_cast<uint64_t>(length_info.words[2]) << 32;
      }
      return length;
    }
    case kRuntimeArray:
      return UINT64_MAX;
    case kStruct:
      return AsStruct()->element_types().size();
    default:
      return 0;
  }
}

bool Integer::IsSameImpl(const Type* that, IsSameCache*) const {
  const Integer* it = that->AsInteger();
  return it && width_ == it->width_ && signed_ == it->signed_ &&
         HasSameDecorations(that);
}

std::string Integer::str() const {
  std::ostringstream oss;
  oss << (signed_ ? "s" : "u") << "int" << width_;
  return oss.str();
}

size_t Integer::ComputeExtraStateHash(size_t hash, SeenTypes*) const {
  return hash_combine(hash, width_, signed_);
}

bool Float::IsSameImpl(const Type* that, IsSameCache*) const {
  const Float* ft = that->AsFloat();
  return ft && width_ == ft->width_ && HasSameDecorations(that);
}

std::string Float::str() const {
  std::ostringstream oss;
  oss << "float" << width_;
  return oss.str();
}

size_t Float::ComputeExtraStateHash(size_t hash, SeenTypes*) const {
  return hash_combine(hash, width_);
}

Vector::Vector(const Type* type, uint32_t count)
    : Type(kVector), element_type_(type), count_(count) {
  assert(type->AsBool() || type->AsInteger() || type->AsFloat());
}

bool Vector::IsSameImpl(const Type* that, IsSameCache* seen) const {
  const Vector* vt = that->AsVector();
  if (!vt) return false;
  return count_ == vt->count_ &&
         element_type_->IsSameImpl(vt->element_type_, seen) &&
         HasSameDecorations(that);
}

std::string Vector::str() const {
  std::ostringstream oss;
  oss << "<" << element_type_->str() << ", " << count_ << ">";
  return oss.str();
}

size_t Vector::ComputeExtraStateHash(size_t hash, SeenTypes* seen) const {
  // prefer form that doesn't require push/pop from stack: add state and
  // make tail call.
  hash = hash_combine(hash, count_);
  return element_type_->ComputeHashValue(hash, seen);
}

Matrix::Matrix(const Type* type, uint32_t count)
    : Type(kMatrix), element_type_(type), count_(count) {
  assert(type->AsVector());
}

bool Matrix::IsSameImpl(const Type* that, IsSameCache* seen) const {
  const Matrix* mt = that->AsMatrix();
  if (!mt) return false;
  return count_ == mt->count_ &&
         element_type_->IsSameImpl(mt->element_type_, seen) &&
         HasSameDecorations(that);
}

std::string Matrix::str() const {
  std::ostringstream oss;
  oss << "<" << element_type_->str() << ", " << count_ << ">";
  return oss.str();
}

size_t Matrix::ComputeExtraStateHash(size_t hash, SeenTypes* seen) const {
  hash = hash_combine(hash, count_);
  return element_type_->ComputeHashValue(hash, seen);
}

Image::Image(Type* type, spv::Dim dimen, uint32_t d, bool array,
             bool multisample, uint32_t sampling, spv::ImageFormat f,
             spv::AccessQualifier qualifier)
    : Type(kImage),
      sampled_type_(type),
      dim_(dimen),
      depth_(d),
      arrayed_(array),
      ms_(multisample),
      sampled_(sampling),
      format_(f),
      access_qualifier_(qualifier) {
  // TODO(antiagainst): check sampled_type
}

bool Image::IsSameImpl(const Type* that, IsSameCache* seen) const {
  const Image* it = that->AsImage();
  if (!it) return false;
  return dim_ == it->dim_ && depth_ == it->depth_ && arrayed_ == it->arrayed_ &&
         ms_ == it->ms_ && sampled_ == it->sampled_ && format_ == it->format_ &&
         access_qualifier_ == it->access_qualifier_ &&
         sampled_type_->IsSameImpl(it->sampled_type_, seen) &&
         HasSameDecorations(that);
}

std::string Image::str() const {
  std::ostringstream oss;
  oss << "image(" << sampled_type_->str() << ", " << uint32_t(dim_) << ", "
      << depth_ << ", " << arrayed_ << ", " << ms_ << ", " << sampled_ << ", "
      << uint32_t(format_) << ", " << uint32_t(access_qualifier_) << ")";
  return oss.str();
}

size_t Image::ComputeExtraStateHash(size_t hash, SeenTypes* seen) const {
  hash = hash_combine(hash, uint32_t(dim_), depth_, arrayed_, ms_, sampled_,
                      uint32_t(format_), uint32_t(access_qualifier_));
  return sampled_type_->ComputeHashValue(hash, seen);
}

bool SampledImage::IsSameImpl(const Type* that, IsSameCache* seen) const {
  const SampledImage* sit = that->AsSampledImage();
  if (!sit) return false;
  return image_type_->IsSameImpl(sit->image_type_, seen) &&
         HasSameDecorations(that);
}

std::string SampledImage::str() const {
  std::ostringstream oss;
  oss << "sampled_image(" << image_type_->str() << ")";
  return oss.str();
}

size_t SampledImage::ComputeExtraStateHash(size_t hash, SeenTypes* seen) const {
  return image_type_->ComputeHashValue(hash, seen);
}

Array::Array(const Type* type, const Array::LengthInfo& length_info_arg)
    : Type(kArray), element_type_(type), length_info_(length_info_arg) {
  assert(type != nullptr);
  assert(!type->AsVoid());
  // We always have a word to say which case we're in, followed
  // by at least one more word.
  assert(length_info_arg.words.size() >= 2);
}

bool Array::IsSameImpl(const Type* that, IsSameCache* seen) const {
  const Array* at = that->AsArray();
  if (!at) return false;
  bool is_same = element_type_->IsSameImpl(at->element_type_, seen);
  is_same = is_same && HasSameDecorations(that);
  is_same = is_same && (length_info_.words == at->length_info_.words);
  return is_same;
}

std::string Array::str() const {
  std::ostringstream oss;
  oss << "[" << element_type_->str() << ", id(" << LengthId() << "), words(";
  const char* spacer = "";
  for (auto w : length_info_.words) {
    oss << spacer << w;
    spacer = ",";
  }
  oss << ")]";
  return oss.str();
}

size_t Array::ComputeExtraStateHash(size_t hash, SeenTypes* seen) const {
  hash = hash_combine(hash, length_info_.words);
  return element_type_->ComputeHashValue(hash, seen);
}

void Array::ReplaceElementType(const Type* type) { element_type_ = type; }

Array::LengthInfo Array::GetConstantLengthInfo(uint32_t const_id,
                                               uint32_t length) const {
  std::vector<uint32_t> extra_words{LengthInfo::Case::kConstant, length};
  return {const_id, extra_words};
}

RuntimeArray::RuntimeArray(const Type* type)
    : Type(kRuntimeArray), element_type_(type) {
  assert(!type->AsVoid());
}

bool RuntimeArray::IsSameImpl(const Type* that, IsSameCache* seen) const {
  const RuntimeArray* rat = that->AsRuntimeArray();
  if (!rat) return false;
  return element_type_->IsSameImpl(rat->element_type_, seen) &&
         HasSameDecorations(that);
}

std::string RuntimeArray::str() const {
  std::ostringstream oss;
  oss << "[" << element_type_->str() << "]";
  return oss.str();
}

size_t RuntimeArray::ComputeExtraStateHash(size_t hash, SeenTypes* seen) const {
  return element_type_->ComputeHashValue(hash, seen);
}

void RuntimeArray::ReplaceElementType(const Type* type) {
  element_type_ = type;
}

Struct::Struct(const std::vector<const Type*>& types)
    : Type(kStruct), element_types_(types) {
  for (const auto* t : types) {
    (void)t;
    assert(!t->AsVoid());
  }
}

void Struct::AddMemberDecoration(uint32_t index,
                                 std::vector<uint32_t>&& decoration) {
  if (index >= element_types_.size()) {
    assert(0 && "index out of bound");
    return;
  }

  element_decorations_[index].push_back(std::move(decoration));
}

bool Struct::IsSameImpl(const Type* that, IsSameCache* seen) const {
  const Struct* st = that->AsStruct();
  if (!st) return false;
  if (element_types_.size() != st->element_types_.size()) return false;
  const auto size = element_decorations_.size();
  if (size != st->element_decorations_.size()) return false;
  if (!HasSameDecorations(that)) return false;

  for (size_t i = 0; i < element_types_.size(); ++i) {
    if (!element_types_[i]->IsSameImpl(st->element_types_[i], seen))
      return false;
  }
  for (const auto& p : element_decorations_) {
    if (st->element_decorations_.count(p.first) == 0) return false;
    if (!CompareTwoVectors(p.second, st->element_decorations_.at(p.first)))
      return false;
  }
  return true;
}

std::string Struct::str() const {
  std::ostringstream oss;
  oss << "{";
  const size_t count = element_types_.size();
  for (size_t i = 0; i < count; ++i) {
    oss << element_types_[i]->str();
    if (i + 1 != count) oss << ", ";
  }
  oss << "}";
  return oss.str();
}

size_t Struct::ComputeExtraStateHash(size_t hash, SeenTypes* seen) const {
  for (auto* t : element_types_) {
    hash = t->ComputeHashValue(hash, seen);
  }
  for (const auto& pair : element_decorations_) {
    hash = hash_combine(hash, pair.first, pair.second);
  }
  return hash;
}

bool Opaque::IsSameImpl(const Type* that, IsSameCache*) const {
  const Opaque* ot = that->AsOpaque();
  if (!ot) return false;
  return name_ == ot->name_ && HasSameDecorations(that);
}

std::string Opaque::str() const {
  std::ostringstream oss;
  oss << "opaque('" << name_ << "')";
  return oss.str();
}

size_t Opaque::ComputeExtraStateHash(size_t hash, SeenTypes*) const {
  return hash_combine(hash, name_);
}

Pointer::Pointer(const Type* type, spv::StorageClass sc)
    : Type(kPointer), pointee_type_(type), storage_class_(sc) {}

bool Pointer::IsSameImpl(const Type* that, IsSameCache* seen) const {
  const Pointer* pt = that->AsPointer();
  if (!pt) return false;
  if (storage_class_ != pt->storage_class_) return false;
  auto p = seen->insert(std::make_pair(this, that->AsPointer()));
  if (!p.second) {
    return true;
  }
  bool same_pointee = pointee_type_->IsSameImpl(pt->pointee_type_, seen);
  seen->erase(p.first);
  if (!same_pointee) {
    return false;
  }
  return HasSameDecorations(that);
}

std::string Pointer::str() const {
  std::ostringstream os;
  os << pointee_type_->str() << " " << static_cast<uint32_t>(storage_class_)
     << "*";
  return os.str();
}

size_t Pointer::ComputeExtraStateHash(size_t hash, SeenTypes* seen) const {
  hash = hash_combine(hash, uint32_t(storage_class_));
  return pointee_type_->ComputeHashValue(hash, seen);
}

void Pointer::SetPointeeType(const Type* type) { pointee_type_ = type; }

Function::Function(const Type* ret_type, const std::vector<const Type*>& params)
    : Type(kFunction), return_type_(ret_type), param_types_(params) {}

Function::Function(const Type* ret_type, std::vector<const Type*>& params)
    : Type(kFunction), return_type_(ret_type), param_types_(params) {}

bool Function::IsSameImpl(const Type* that, IsSameCache* seen) const {
  const Function* ft = that->AsFunction();
  if (!ft) return false;
  if (!return_type_->IsSameImpl(ft->return_type_, seen)) return false;
  if (param_types_.size() != ft->param_types_.size()) return false;
  for (size_t i = 0; i < param_types_.size(); ++i) {
    if (!param_types_[i]->IsSameImpl(ft->param_types_[i], seen)) return false;
  }
  return HasSameDecorations(that);
}

std::string Function::str() const {
  std::ostringstream oss;
  const size_t count = param_types_.size();
  oss << "(";
  for (size_t i = 0; i < count; ++i) {
    oss << param_types_[i]->str();
    if (i + 1 != count) oss << ", ";
  }
  oss << ") -> " << return_type_->str();
  return oss.str();
}

size_t Function::ComputeExtraStateHash(size_t hash, SeenTypes* seen) const {
  for (const auto* t : param_types_) {
    hash = t->ComputeHashValue(hash, seen);
  }
  return return_type_->ComputeHashValue(hash, seen);
}

void Function::SetReturnType(const Type* type) { return_type_ = type; }

bool Pipe::IsSameImpl(const Type* that, IsSameCache*) const {
  const Pipe* pt = that->AsPipe();
  if (!pt) return false;
  return access_qualifier_ == pt->access_qualifier_ && HasSameDecorations(that);
}

std::string Pipe::str() const {
  std::ostringstream oss;
  oss << "pipe(" << uint32_t(access_qualifier_) << ")";
  return oss.str();
}

size_t Pipe::ComputeExtraStateHash(size_t hash, SeenTypes*) const {
  return hash_combine(hash, uint32_t(access_qualifier_));
}

bool ForwardPointer::IsSameImpl(const Type* that, IsSameCache*) const {
  const ForwardPointer* fpt = that->AsForwardPointer();
  if (!fpt) return false;
  return (pointer_ && fpt->pointer_ ? *pointer_ == *fpt->pointer_
                                    : target_id_ == fpt->target_id_) &&
         storage_class_ == fpt->storage_class_ && HasSameDecorations(that);
}

std::string ForwardPointer::str() const {
  std::ostringstream oss;
  oss << "forward_pointer(";
  if (pointer_ != nullptr) {
    oss << pointer_->str();
  } else {
    oss << target_id_;
  }
  oss << ")";
  return oss.str();
}

size_t ForwardPointer::ComputeExtraStateHash(size_t hash,
                                             SeenTypes* seen) const {
  hash = hash_combine(hash, target_id_, uint32_t(storage_class_));
  if (pointer_) hash = pointer_->ComputeHashValue(hash, seen);
  return hash;
}

CooperativeMatrixNV::CooperativeMatrixNV(const Type* type, const uint32_t scope,
                                         const uint32_t rows,
                                         const uint32_t columns)
    : Type(kCooperativeMatrixNV),
      component_type_(type),
      scope_id_(scope),
      rows_id_(rows),
      columns_id_(columns) {
  assert(type != nullptr);
  assert(scope != 0);
  assert(rows != 0);
  assert(columns != 0);
}

std::string CooperativeMatrixNV::str() const {
  std::ostringstream oss;
  oss << "<" << component_type_->str() << ", " << scope_id_ << ", " << rows_id_
      << ", " << columns_id_ << ">";
  return oss.str();
}

size_t CooperativeMatrixNV::ComputeExtraStateHash(size_t hash,
                                                  SeenTypes* seen) const {
  hash = hash_combine(hash, scope_id_, rows_id_, columns_id_);
  return component_type_->ComputeHashValue(hash, seen);
}

bool CooperativeMatrixNV::IsSameImpl(const Type* that,
                                     IsSameCache* seen) const {
  const CooperativeMatrixNV* mt = that->AsCooperativeMatrixNV();
  if (!mt) return false;
  return component_type_->IsSameImpl(mt->component_type_, seen) &&
         scope_id_ == mt->scope_id_ && rows_id_ == mt->rows_id_ &&
         columns_id_ == mt->columns_id_ && HasSameDecorations(that);
}

CooperativeMatrixKHR::CooperativeMatrixKHR(const Type* type,
                                           const uint32_t scope,
                                           const uint32_t rows,
                                           const uint32_t columns,
                                           const uint32_t use)
    : Type(kCooperativeMatrixKHR),
      component_type_(type),
      scope_id_(scope),
      rows_id_(rows),
      columns_id_(columns),
      use_id_(use) {
  assert(type != nullptr);
  assert(scope != 0);
  assert(rows != 0);
  assert(columns != 0);
}

std::string CooperativeMatrixKHR::str() const {
  std::ostringstream oss;
  oss << "<" << component_type_->str() << ", " << scope_id_ << ", " << rows_id_
      << ", " << columns_id_ << ", " << use_id_ << ">";
  return oss.str();
}

size_t CooperativeMatrixKHR::ComputeExtraStateHash(size_t hash,
                                                   SeenTypes* seen) const {
  hash = hash_combine(hash, scope_id_, rows_id_, columns_id_, use_id_);
  return component_type_->ComputeHashValue(hash, seen);
}

bool CooperativeMatrixKHR::IsSameImpl(const Type* that,
                                      IsSameCache* seen) const {
  const CooperativeMatrixKHR* mt = that->AsCooperativeMatrixKHR();
  if (!mt) return false;
  return component_type_->IsSameImpl(mt->component_type_, seen) &&
         scope_id_ == mt->scope_id_ && rows_id_ == mt->rows_id_ &&
         columns_id_ == mt->columns_id_ && HasSameDecorations(that);
}

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools
