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

// This file provides a class hierarchy for representing SPIR-V types.

#ifndef SOURCE_OPT_TYPES_H_
#define SOURCE_OPT_TYPES_H_

#include <map>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "source/latest_version_spirv_header.h"
#include "source/opt/instruction.h"
#include "source/util/small_vector.h"
#include "spirv-tools/libspirv.h"

namespace spvtools {
namespace opt {
namespace analysis {

class Void;
class Bool;
class Integer;
class Float;
class Vector;
class Matrix;
class Image;
class Sampler;
class SampledImage;
class Array;
class RuntimeArray;
class Struct;
class Opaque;
class Pointer;
class Function;
class Event;
class DeviceEvent;
class ReserveId;
class Queue;
class Pipe;
class ForwardPointer;
class PipeStorage;
class NamedBarrier;
class AccelerationStructureNV;
class CooperativeMatrixNV;
class CooperativeMatrixKHR;
class RayQueryKHR;
class HitObjectNV;

// Abstract class for a SPIR-V type. It has a bunch of As<sublcass>() methods,
// which is used as a way to probe the actual <subclass>.
class Type {
 public:
  typedef std::set<std::pair<const Pointer*, const Pointer*>> IsSameCache;

  using SeenTypes = spvtools::utils::SmallVector<const Type*, 8>;

  // Available subtypes.
  //
  // When adding a new derived class of Type, please add an entry to the enum.
  enum Kind {
    kVoid,
    kBool,
    kInteger,
    kFloat,
    kVector,
    kMatrix,
    kImage,
    kSampler,
    kSampledImage,
    kArray,
    kRuntimeArray,
    kStruct,
    kOpaque,
    kPointer,
    kFunction,
    kEvent,
    kDeviceEvent,
    kReserveId,
    kQueue,
    kPipe,
    kForwardPointer,
    kPipeStorage,
    kNamedBarrier,
    kAccelerationStructureNV,
    kCooperativeMatrixNV,
    kCooperativeMatrixKHR,
    kRayQueryKHR,
    kHitObjectNV,
    kLast
  };

  Type(Kind k) : kind_(k) {}

  virtual ~Type() = default;

  // Attaches a decoration directly on this type.
  void AddDecoration(std::vector<uint32_t>&& d) {
    decorations_.push_back(std::move(d));
  }
  // Returns the decorations on this type as a string.
  std::string GetDecorationStr() const;
  // Returns true if this type has exactly the same decorations as |that| type.
  bool HasSameDecorations(const Type* that) const;
  // Returns true if this type is exactly the same as |that| type, including
  // decorations.
  bool IsSame(const Type* that) const {
    IsSameCache seen;
    return IsSameImpl(that, &seen);
  }

  // Returns true if this type is exactly the same as |that| type, including
  // decorations.  |seen| is the set of |Pointer*| pair that are currently being
  // compared in a parent call to |IsSameImpl|.
  virtual bool IsSameImpl(const Type* that, IsSameCache* seen) const = 0;

  // Returns a human-readable string to represent this type.
  virtual std::string str() const = 0;

  Kind kind() const { return kind_; }
  const std::vector<std::vector<uint32_t>>& decorations() const {
    return decorations_;
  }

  // Returns true if there is no decoration on this type. For struct types,
  // returns true only when there is no decoration for both the struct type
  // and the struct members.
  virtual bool decoration_empty() const { return decorations_.empty(); }

  // Creates a clone of |this|.
  std::unique_ptr<Type> Clone() const;

  // Returns a clone of |this| minus any decorations.
  std::unique_ptr<Type> RemoveDecorations() const;

  // Returns true if this cannot hash to the same value as another type in the
  // module. For example, structs are not unique types because the module could
  // have two types
  //
  //  %1 = OpTypeStruct %int
  //  %2 = OpTypeStruct %int
  //
  // The only way to distinguish these types is the result id. The type manager
  // will hash them to the same value.
  bool IsUniqueType() const;

  bool operator==(const Type& other) const;

  // Returns the hash value of this type.
  size_t HashValue() const;

  size_t ComputeHashValue(size_t hash, SeenTypes* seen) const;

  // Returns the number of components in a composite type.  Returns 0 for a
  // non-composite type.
  uint64_t NumberOfComponents() const;

// A bunch of methods for casting this type to a given type. Returns this if the
// cast can be done, nullptr otherwise.
// clang-format off
#define DeclareCastMethod(target)                  \
  virtual target* As##target() { return nullptr; } \
  virtual const target* As##target() const { return nullptr; }
  DeclareCastMethod(Void)
  DeclareCastMethod(Bool)
  DeclareCastMethod(Integer)
  DeclareCastMethod(Float)
  DeclareCastMethod(Vector)
  DeclareCastMethod(Matrix)
  DeclareCastMethod(Image)
  DeclareCastMethod(Sampler)
  DeclareCastMethod(SampledImage)
  DeclareCastMethod(Array)
  DeclareCastMethod(RuntimeArray)
  DeclareCastMethod(Struct)
  DeclareCastMethod(Opaque)
  DeclareCastMethod(Pointer)
  DeclareCastMethod(Function)
  DeclareCastMethod(Event)
  DeclareCastMethod(DeviceEvent)
  DeclareCastMethod(ReserveId)
  DeclareCastMethod(Queue)
  DeclareCastMethod(Pipe)
  DeclareCastMethod(ForwardPointer)
  DeclareCastMethod(PipeStorage)
  DeclareCastMethod(NamedBarrier)
  DeclareCastMethod(AccelerationStructureNV)
  DeclareCastMethod(CooperativeMatrixNV)
  DeclareCastMethod(CooperativeMatrixKHR)
  DeclareCastMethod(RayQueryKHR)
  DeclareCastMethod(HitObjectNV)
#undef DeclareCastMethod

protected:
  // Add any type-specific state to |hash| and returns new hash.
  virtual size_t ComputeExtraStateHash(size_t hash, SeenTypes* seen) const = 0;

 protected:
  // Decorations attached to this type. Each decoration is encoded as a vector
  // of uint32_t numbers. The first uint32_t number is the decoration value,
  // and the rest are the parameters to the decoration (if exists).
  std::vector<std::vector<uint32_t>> decorations_;

 private:
  // Removes decorations on this type. For struct types, also removes element
  // decorations.
  virtual void ClearDecorations() { decorations_.clear(); }

  Kind kind_;
};
// clang-format on

class Integer : public Type {
 public:
  Integer(uint32_t w, bool is_signed)
      : Type(kInteger), width_(w), signed_(is_signed) {}
  Integer(const Integer&) = default;

  std::string str() const override;

  Integer* AsInteger() override { return this; }
  const Integer* AsInteger() const override { return this; }
  uint32_t width() const { return width_; }
  bool IsSigned() const { return signed_; }

  size_t ComputeExtraStateHash(size_t hash, SeenTypes* seen) const override;

 private:
  bool IsSameImpl(const Type* that, IsSameCache*) const override;

  uint32_t width_;  // bit width
  bool signed_;     // true if this integer is signed
};

class Float : public Type {
 public:
  Float(uint32_t w) : Type(kFloat), width_(w) {}
  Float(const Float&) = default;

  std::string str() const override;

  Float* AsFloat() override { return this; }
  const Float* AsFloat() const override { return this; }
  uint32_t width() const { return width_; }

  size_t ComputeExtraStateHash(size_t hash, SeenTypes* seen) const override;

 private:
  bool IsSameImpl(const Type* that, IsSameCache*) const override;

  uint32_t width_;  // bit width
};

class Vector : public Type {
 public:
  Vector(const Type* element_type, uint32_t count);
  Vector(const Vector&) = default;

  std::string str() const override;
  const Type* element_type() const { return element_type_; }
  uint32_t element_count() const { return count_; }

  Vector* AsVector() override { return this; }
  const Vector* AsVector() const override { return this; }

  size_t ComputeExtraStateHash(size_t hash, SeenTypes* seen) const override;

 private:
  bool IsSameImpl(const Type* that, IsSameCache*) const override;

  const Type* element_type_;
  uint32_t count_;
};

class Matrix : public Type {
 public:
  Matrix(const Type* element_type, uint32_t count);
  Matrix(const Matrix&) = default;

  std::string str() const override;
  const Type* element_type() const { return element_type_; }
  uint32_t element_count() const { return count_; }

  Matrix* AsMatrix() override { return this; }
  const Matrix* AsMatrix() const override { return this; }

  size_t ComputeExtraStateHash(size_t hash, SeenTypes* seen) const override;

 private:
  bool IsSameImpl(const Type* that, IsSameCache*) const override;

  const Type* element_type_;
  uint32_t count_;
};

class Image : public Type {
 public:
  Image(Type* type, spv::Dim dimen, uint32_t d, bool array, bool multisample,
        uint32_t sampling, spv::ImageFormat f,
        spv::AccessQualifier qualifier = spv::AccessQualifier::ReadOnly);
  Image(const Image&) = default;

  std::string str() const override;

  Image* AsImage() override { return this; }
  const Image* AsImage() const override { return this; }

  const Type* sampled_type() const { return sampled_type_; }
  spv::Dim dim() const { return dim_; }
  uint32_t depth() const { return depth_; }
  bool is_arrayed() const { return arrayed_; }
  bool is_multisampled() const { return ms_; }
  uint32_t sampled() const { return sampled_; }
  spv::ImageFormat format() const { return format_; }
  spv::AccessQualifier access_qualifier() const { return access_qualifier_; }

  size_t ComputeExtraStateHash(size_t hash, SeenTypes* seen) const override;

 private:
  bool IsSameImpl(const Type* that, IsSameCache*) const override;

  Type* sampled_type_;
  spv::Dim dim_;
  uint32_t depth_;
  bool arrayed_;
  bool ms_;
  uint32_t sampled_;
  spv::ImageFormat format_;
  spv::AccessQualifier access_qualifier_;
};

class SampledImage : public Type {
 public:
  SampledImage(Type* image) : Type(kSampledImage), image_type_(image) {}
  SampledImage(const SampledImage&) = default;

  std::string str() const override;

  SampledImage* AsSampledImage() override { return this; }
  const SampledImage* AsSampledImage() const override { return this; }

  const Type* image_type() const { return image_type_; }

  size_t ComputeExtraStateHash(size_t hash, SeenTypes* seen) const override;

 private:
  bool IsSameImpl(const Type* that, IsSameCache*) const override;
  Type* image_type_;
};

class Array : public Type {
 public:
  // Data about the length operand, that helps us distinguish between one
  // array length and another.
  struct LengthInfo {
    // The result id of the instruction defining the length.
    const uint32_t id;
    enum Case : uint32_t {
      kConstant = 0,
      kConstantWithSpecId = 1,
      kDefiningId = 2
    };
    // Extra words used to distinshish one array length and another.
    //  - if OpConstant, then it's 0, then the words in the literal constant
    //    value.
    //  - if OpSpecConstant, then it's 1, then the SpecID decoration if there
    //    is one, followed by the words in the literal constant value.
    //    The spec might not be overridden, in which case we'll end up using
    //    the literal value.
    //  - Otherwise, it's an OpSpecConsant, and this 2, then the ID (again).
    const std::vector<uint32_t> words;
  };

  // Constructs an array type with given element and length.  If the length
  // is an OpSpecConstant, then |spec_id| should be its SpecId decoration.
  Array(const Type* element_type, const LengthInfo& length_info_arg);
  Array(const Array&) = default;

  std::string str() const override;
  const Type* element_type() const { return element_type_; }
  uint32_t LengthId() const { return length_info_.id; }
  const LengthInfo& length_info() const { return length_info_; }

  Array* AsArray() override { return this; }
  const Array* AsArray() const override { return this; }

  size_t ComputeExtraStateHash(size_t hash, SeenTypes* seen) const override;

  void ReplaceElementType(const Type* element_type);
  LengthInfo GetConstantLengthInfo(uint32_t const_id, uint32_t length) const;

 private:
  bool IsSameImpl(const Type* that, IsSameCache*) const override;

  const Type* element_type_;
  const LengthInfo length_info_;
};

class RuntimeArray : public Type {
 public:
  RuntimeArray(const Type* element_type);
  RuntimeArray(const RuntimeArray&) = default;

  std::string str() const override;
  const Type* element_type() const { return element_type_; }

  RuntimeArray* AsRuntimeArray() override { return this; }
  const RuntimeArray* AsRuntimeArray() const override { return this; }

  size_t ComputeExtraStateHash(size_t hash, SeenTypes* seen) const override;

  void ReplaceElementType(const Type* element_type);

 private:
  bool IsSameImpl(const Type* that, IsSameCache*) const override;

  const Type* element_type_;
};

class Struct : public Type {
 public:
  Struct(const std::vector<const Type*>& element_types);
  Struct(const Struct&) = default;

  // Adds a decoration to the member at the given index.  The first word is the
  // decoration enum, and the remaining words, if any, are its operands.
  void AddMemberDecoration(uint32_t index, std::vector<uint32_t>&& decoration);

  std::string str() const override;
  const std::vector<const Type*>& element_types() const {
    return element_types_;
  }
  std::vector<const Type*>& element_types() { return element_types_; }
  bool decoration_empty() const override {
    return decorations_.empty() && element_decorations_.empty();
  }

  const std::map<uint32_t, std::vector<std::vector<uint32_t>>>&
  element_decorations() const {
    return element_decorations_;
  }

  Struct* AsStruct() override { return this; }
  const Struct* AsStruct() const override { return this; }

  size_t ComputeExtraStateHash(size_t hash, SeenTypes* seen) const override;

 private:
  bool IsSameImpl(const Type* that, IsSameCache*) const override;

  void ClearDecorations() override {
    decorations_.clear();
    element_decorations_.clear();
  }

  std::vector<const Type*> element_types_;
  // We can attach decorations to struct members and that should not affect the
  // underlying element type. So we need an extra data structure here to keep
  // track of element type decorations.  They must be stored in an ordered map
  // because |GetExtraHashWords| will traverse the structure.  It must have a
  // fixed order in order to hash to the same value every time.
  std::map<uint32_t, std::vector<std::vector<uint32_t>>> element_decorations_;
};

class Opaque : public Type {
 public:
  Opaque(std::string n) : Type(kOpaque), name_(std::move(n)) {}
  Opaque(const Opaque&) = default;

  std::string str() const override;

  Opaque* AsOpaque() override { return this; }
  const Opaque* AsOpaque() const override { return this; }

  const std::string& name() const { return name_; }

  size_t ComputeExtraStateHash(size_t hash, SeenTypes* seen) const override;

 private:
  bool IsSameImpl(const Type* that, IsSameCache*) const override;

  std::string name_;
};

class Pointer : public Type {
 public:
  Pointer(const Type* pointee, spv::StorageClass sc);
  Pointer(const Pointer&) = default;

  std::string str() const override;
  const Type* pointee_type() const { return pointee_type_; }
  spv::StorageClass storage_class() const { return storage_class_; }

  Pointer* AsPointer() override { return this; }
  const Pointer* AsPointer() const override { return this; }

  size_t ComputeExtraStateHash(size_t hash, SeenTypes* seen) const override;

  void SetPointeeType(const Type* type);

 private:
  bool IsSameImpl(const Type* that, IsSameCache*) const override;

  const Type* pointee_type_;
  spv::StorageClass storage_class_;
};

class Function : public Type {
 public:
  Function(const Type* ret_type, const std::vector<const Type*>& params);
  Function(const Type* ret_type, std::vector<const Type*>& params);
  Function(const Function&) = default;

  std::string str() const override;

  Function* AsFunction() override { return this; }
  const Function* AsFunction() const override { return this; }

  const Type* return_type() const { return return_type_; }
  const std::vector<const Type*>& param_types() const { return param_types_; }
  std::vector<const Type*>& param_types() { return param_types_; }

  size_t ComputeExtraStateHash(size_t hash, SeenTypes* seen) const override;

  void SetReturnType(const Type* type);

 private:
  bool IsSameImpl(const Type* that, IsSameCache*) const override;

  const Type* return_type_;
  std::vector<const Type*> param_types_;
};

class Pipe : public Type {
 public:
  Pipe(spv::AccessQualifier qualifier)
      : Type(kPipe), access_qualifier_(qualifier) {}
  Pipe(const Pipe&) = default;

  std::string str() const override;

  Pipe* AsPipe() override { return this; }
  const Pipe* AsPipe() const override { return this; }

  spv::AccessQualifier access_qualifier() const { return access_qualifier_; }

  size_t ComputeExtraStateHash(size_t hash, SeenTypes* seen) const override;

 private:
  bool IsSameImpl(const Type* that, IsSameCache*) const override;

  spv::AccessQualifier access_qualifier_;
};

class ForwardPointer : public Type {
 public:
  ForwardPointer(uint32_t id, spv::StorageClass sc)
      : Type(kForwardPointer),
        target_id_(id),
        storage_class_(sc),
        pointer_(nullptr) {}
  ForwardPointer(const ForwardPointer&) = default;

  uint32_t target_id() const { return target_id_; }
  void SetTargetPointer(const Pointer* pointer) { pointer_ = pointer; }
  spv::StorageClass storage_class() const { return storage_class_; }
  const Pointer* target_pointer() const { return pointer_; }

  std::string str() const override;

  ForwardPointer* AsForwardPointer() override { return this; }
  const ForwardPointer* AsForwardPointer() const override { return this; }

  size_t ComputeExtraStateHash(size_t hash, SeenTypes* seen) const override;

 private:
  bool IsSameImpl(const Type* that, IsSameCache*) const override;

  uint32_t target_id_;
  spv::StorageClass storage_class_;
  const Pointer* pointer_;
};

class CooperativeMatrixNV : public Type {
 public:
  CooperativeMatrixNV(const Type* type, const uint32_t scope,
                      const uint32_t rows, const uint32_t columns);
  CooperativeMatrixNV(const CooperativeMatrixNV&) = default;

  std::string str() const override;

  CooperativeMatrixNV* AsCooperativeMatrixNV() override { return this; }
  const CooperativeMatrixNV* AsCooperativeMatrixNV() const override {
    return this;
  }

  size_t ComputeExtraStateHash(size_t hash, SeenTypes* seen) const override;

  const Type* component_type() const { return component_type_; }
  uint32_t scope_id() const { return scope_id_; }
  uint32_t rows_id() const { return rows_id_; }
  uint32_t columns_id() const { return columns_id_; }

 private:
  bool IsSameImpl(const Type* that, IsSameCache*) const override;

  const Type* component_type_;
  const uint32_t scope_id_;
  const uint32_t rows_id_;
  const uint32_t columns_id_;
};

class CooperativeMatrixKHR : public Type {
 public:
  CooperativeMatrixKHR(const Type* type, const uint32_t scope,
                       const uint32_t rows, const uint32_t columns,
                       const uint32_t use);
  CooperativeMatrixKHR(const CooperativeMatrixKHR&) = default;

  std::string str() const override;

  CooperativeMatrixKHR* AsCooperativeMatrixKHR() override { return this; }
  const CooperativeMatrixKHR* AsCooperativeMatrixKHR() const override {
    return this;
  }

  size_t ComputeExtraStateHash(size_t hash, SeenTypes* seen) const override;

  const Type* component_type() const { return component_type_; }
  uint32_t scope_id() const { return scope_id_; }
  uint32_t rows_id() const { return rows_id_; }
  uint32_t columns_id() const { return columns_id_; }
  uint32_t use_id() const { return use_id_; }

 private:
  bool IsSameImpl(const Type* that, IsSameCache*) const override;

  const Type* component_type_;
  const uint32_t scope_id_;
  const uint32_t rows_id_;
  const uint32_t columns_id_;
  const uint32_t use_id_;
};

#define DefineParameterlessType(type, name)                                \
  class type : public Type {                                               \
   public:                                                                 \
    type() : Type(k##type) {}                                              \
    type(const type&) = default;                                           \
                                                                           \
    std::string str() const override { return #name; }                     \
                                                                           \
    type* As##type() override { return this; }                             \
    const type* As##type() const override { return this; }                 \
                                                                           \
    size_t ComputeExtraStateHash(size_t hash, SeenTypes*) const override { \
      return hash;                                                         \
    }                                                                      \
                                                                           \
   private:                                                                \
    bool IsSameImpl(const Type* that, IsSameCache*) const override {       \
      return that->As##type() && HasSameDecorations(that);                 \
    }                                                                      \
  }
DefineParameterlessType(Void, void);
DefineParameterlessType(Bool, bool);
DefineParameterlessType(Sampler, sampler);
DefineParameterlessType(Event, event);
DefineParameterlessType(DeviceEvent, device_event);
DefineParameterlessType(ReserveId, reserve_id);
DefineParameterlessType(Queue, queue);
DefineParameterlessType(PipeStorage, pipe_storage);
DefineParameterlessType(NamedBarrier, named_barrier);
DefineParameterlessType(AccelerationStructureNV, accelerationStructureNV);
DefineParameterlessType(RayQueryKHR, rayQueryKHR);
DefineParameterlessType(HitObjectNV, hitObjectNV);
#undef DefineParameterlessType

}  // namespace analysis
}  // namespace opt
}  // namespace spvtools

#endif  // SOURCE_OPT_TYPES_H_
