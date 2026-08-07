// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_FIELDS_H_
#define LIB_JXL_FIELDS_H_

// Forward/backward-compatible 'bundles' with auto-serialized 'fields'.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "lib/jxl/base/compiler_specific.h"
#include "lib/jxl/base/status.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/field_encodings.h"

namespace jxl {

struct AuxOut;
enum class LayerType : uint8_t;
struct BitWriter;

// Integer coders: BitsCoder (raw), U32Coder (table), U64Coder (varint).

// Reads/writes a given (fixed) number of bits <= 32.
namespace BitsCoder {
size_t MaxEncodedBits(size_t bits);

Status CanEncode(size_t bits, uint32_t value,
                 size_t* JXL_RESTRICT encoded_bits);

uint32_t Read(size_t bits, BitReader* JXL_RESTRICT reader);

// Returns false if the value is too large to encode.
Status Write(size_t bits, uint32_t value, BitWriter* JXL_RESTRICT writer);
}  // namespace BitsCoder

// Encodes u32 using a lookup table and/or extra bits, governed by a per-field
// encoding `enc` which consists of four distributions `d` chosen via a 2-bit
// selector (least significant = 0). Each d may have two modes:
// - direct: if d.IsDirect(), the value is d.Direct();
// - offset: the value is derived from d.ExtraBits() extra bits plus d.Offset();
// This encoding is denser than Exp-Golomb or Gamma codes when both small and
// large values occur.
//
// Examples:
// Direct: U32Enc(Val(8), Val(16), Val(32), Bits(6)), value 32 => 10b.
// Offset: U32Enc(Val(0), BitsOffset(1, 1), BitsOffset(2, 3), BitsOffset(8, 8))
//   defines the following prefix code:
//   00 -> 0
//   01x -> 1..2
//   10xx -> 3..7
//   11xxxxxxxx -> 8..263
namespace U32Coder {
size_t MaxEncodedBits(U32Enc enc);
Status CanEncode(U32Enc enc, uint32_t value, size_t* JXL_RESTRICT encoded_bits);
uint32_t Read(U32Enc enc, BitReader* JXL_RESTRICT reader);

// Returns false if the value is too large to encode.
Status Write(U32Enc enc, uint32_t value, BitWriter* JXL_RESTRICT writer);

// "private"
Status ChooseSelector(U32Enc enc, uint32_t value,
                      uint32_t* JXL_RESTRICT selector,
                      size_t* JXL_RESTRICT total_bits);
}  // namespace U32Coder

// Encodes 64-bit unsigned integers with a fixed distribution, taking 2 bits
// to encode 0, 6 bits to encode 1 to 16, 10 bits to encode 17 to 272, 15 bits
// to encode up to 4095, and on the order of log2(value) * 1.125 bits for
// larger values.
namespace U64Coder {
constexpr size_t MaxEncodedBits() { return 2 + 12 + 6 * (8 + 1) + (4 + 1); }

uint64_t Read(BitReader* JXL_RESTRICT reader);

// Returns false if the value is too large to encode.
Status Write(uint64_t value, BitWriter* JXL_RESTRICT writer);

// Can always encode, but useful because it also returns bit size.
Status CanEncode(uint64_t value, size_t* JXL_RESTRICT encoded_bits);
}  // namespace U64Coder

// IEEE 754 half-precision (binary16). Refuses to read/write NaN/Inf.
namespace F16Coder {
constexpr size_t MaxEncodedBits() { return 16; }

// Returns false if the bit representation is NaN or infinity
Status Read(BitReader* JXL_RESTRICT reader, float* JXL_RESTRICT value);

// Returns false if the value is too large to encode.
Status Write(float value, BitWriter* JXL_RESTRICT writer);
Status CanEncode(float value, size_t* JXL_RESTRICT encoded_bits);
}  // namespace F16Coder

// A "bundle" is a forward- and backward compatible collection of fields.
// They are used for SizeHeader/FrameHeader/GroupHeader. Bundles can be
// extended by appending(!) fields. Optional fields may be omitted from the
// bitstream by conditionally visiting them. When reading new bitstreams with
// old code, we skip unknown fields at the end of the bundle. This requires
// storing the amount of extra appended bits, and that fields are visited in
// chronological order of being added to the format, because old decoders
// cannot skip some future fields and resume reading old fields. Similarly,
// new readers query bits in an "extensions" field to skip (groups of) fields
// not present in old bitstreams. Note that each bundle must include an
// "extensions" field prior to freezing the format, otherwise it cannot be
// extended.
//
// To ensure interoperability, there will be no opaque fields.
//
// HOWTO:
// - basic usage: define a struct with member variables ("fields") and a
//   VisitFields(v) member function that calls v->U32/Bool etc. for each
//   field, specifying their default values. The ctor must call
//   Bundle::Init(this).
//
// - print a trace of visitors: ensure each bundle has a static Name() member
//   function, and change Bundle::Print* to return true.
//
// - optional fields: in VisitFields, add if (v->Conditional(your_condition))
//   { v->Bool(default, &field); }. This prevents reading/writing field
//   if !your_condition, which is typically computed from a prior field.
//   WARNING: to ensure all fields are initialized, do not add an else branch;
//   instead add another if (v->Conditional(!your_condition)).
//
// - repeated fields: for dynamic sizes, use e.g. std::vector and in
//   VisitFields, if (v->IsReading()) field.resize(size) before accessing field.
//   For static or bounded sizes, use an array or std::array. In all cases,
//   simply visit each array element as if it were a normal field.
//
// - nested bundles: add a bundle as a normal field and in VisitFields call
//   JXL_RETURN_IF_ERROR(v->VisitNested(&nested));
//
// - allow future extensions: define a "uint64_t extensions" field and call
//   v->BeginExtensions(&extensions) after visiting all non-extension fields,
//   and `return v->EndExtensions();` after the last extension field.
//
// - encode an entire bundle in one bit if ALL its fields equal their default
//   values: add a "mutable bool all_default" field and as the first visitor:
//   if (v->AllDefault(*this, &all_default)) {
//     // Overwrite all serialized fields, but not any nonserialized_*.
//     v->SetDefault(this);
//     return true;
//   }
//   Note: if extensions are present, AllDefault() == false.

namespace Bundle {
constexpr size_t kMaxExtensions = 64;  // bits in u64

// Initializes fields to the default values. It is not recursive to nested
// fields, this function is intended to be called in the constructors so
// each nested field will already Init itself.
void Init(Fields* JXL_RESTRICT fields);

// Similar to Init, but recursive to nested fields.
void SetDefault(Fields* JXL_RESTRICT fields);

// Returns whether ALL fields (including `extensions`, if present) are equal
// to their default value.
bool AllDefault(const Fields& fields);

// Returns max number of bits required to encode a T.
size_t MaxBits(const Fields& fields);

// Returns whether a header's fields can all be encoded, i.e. they have a
// valid representation. If so, "*total_bits" is the exact number of bits
// required. Called by Write.
Status CanEncode(const Fields& fields, size_t* JXL_RESTRICT extension_bits,
                 size_t* JXL_RESTRICT total_bits);

Status Read(BitReader* reader, Fields* JXL_RESTRICT fields);

// Returns whether enough bits are available to fully read this bundle using
// Read. Also returns true in case of a codestream error (other than not being
// large enough): that means enough bits are available to determine there's an
// error, use Read to get such error status.
// NOTE: this advances the BitReader, a different one pointing back at the
// original bit position in the codestream must be created to use Read after
// this.
bool CanRead(BitReader* reader, Fields* JXL_RESTRICT fields);

Status Write(const Fields& fields, BitWriter* JXL_RESTRICT writer,
             LayerType layer, AuxOut* aux_out);
}  // namespace Bundle

// Different subclasses of Visitor are passed to implementations of Fields
// throughout their lifetime. Templates used to be used for this but dynamic
// polymorphism produces more compact executables than template reification did.
class Visitor {
 public:
  virtual ~Visitor() = default;
  virtual Status Visit(Fields* fields) = 0;

  virtual Status Bool(bool default_value, bool* JXL_RESTRICT value) = 0;
  virtual Status U32(U32Enc, uint32_t, uint32_t*) = 0;

  // Helper to construct U32Enc from U32Distr.
  Status U32(const U32Distr d0, const U32Distr d1, const U32Distr d2,
             const U32Distr d3, const uint32_t default_value,
             uint32_t* JXL_RESTRICT value) {
    return U32(U32Enc(d0, d1, d2, d3), default_value, value);
  }

  template <typename EnumT>
  Status Enum(const EnumT default_value, EnumT* JXL_RESTRICT value) {
    uint32_t u32 = static_cast<uint32_t>(*value);
    // 00 -> 0
    // 01 -> 1
    // 10xxxx -> 2..17
    // 11yyyyyy -> 18..81
    JXL_RETURN_IF_ERROR(U32(Val(0), Val(1), BitsOffset(4, 2), BitsOffset(6, 18),
                            static_cast<uint32_t>(default_value), &u32));
    *value = static_cast<EnumT>(u32);
    return EnumValid(*value);
  }

  virtual Status Bits(size_t bits, uint32_t default_value,
                      uint32_t* JXL_RESTRICT value) = 0;
  virtual Status U64(uint64_t default_value, uint64_t* JXL_RESTRICT value) = 0;
  virtual Status F16(float default_value, float* JXL_RESTRICT value) = 0;

  // Returns whether VisitFields should visit some subsequent fields.
  // "condition" is typically from prior fields, e.g. flags.
  // Overridden by InitVisitor and MaxBitsVisitor.
  virtual Status Conditional(bool condition) { return condition; }

  // Overridden by InitVisitor, AllDefaultVisitor and CanEncodeVisitor.
  virtual Status AllDefault(const Fields& /*fields*/,
                            bool* JXL_RESTRICT all_default) {
    JXL_RETURN_IF_ERROR(Bool(true, all_default));
    return *all_default;
  }

  virtual void SetDefault(Fields* /*fields*/) {
    // Do nothing by default, this is overridden by ReadVisitor.
  }

  // Returns the result of visiting a nested Bundle.
  // Overridden by InitVisitor.
  virtual Status VisitNested(Fields* fields) { return Visit(fields); }

  // Overridden by ReadVisitor. Enables dynamically-sized fields.
  virtual bool IsReading() const { return false; }

  virtual Status BeginExtensions(uint64_t* JXL_RESTRICT extensions) = 0;
  virtual Status EndExtensions() = 0;
};

namespace fields_internal {
// A bundle can be in one of three states concerning extensions: not-begun,
// active, ended. Bundles may be nested, so we need a stack of states.
class ExtensionStates {
 public:
  void Push() {
    // Initial state = not-begun.
    begun_ <<= 1;
    ended_ <<= 1;
  }

  // Clears current state; caller must check IsEnded beforehand.
  void Pop() {
    begun_ >>= 1;
    ended_ >>= 1;
  }

  // Returns true if state == active || state == ended.
  Status IsBegun() const { return (begun_ & 1) != 0; }
  // Returns true if state != not-begun && state != active.
  Status IsEnded() const { return (ended_ & 1) != 0; }

  void Begin() {
    JXL_DASSERT(!IsBegun());
    JXL_DASSERT(!IsEnded());
    begun_ += 1;
  }

  void End() {
    JXL_DASSERT(IsBegun());
    JXL_DASSERT(!IsEnded());
    ended_ += 1;
  }

 private:
  // Current state := least-significant bit of begun_ and ended_.
  uint64_t begun_ = 0;
  uint64_t ended_ = 0;
};

// Visitors generate Init/AllDefault/Read/Write logic for all fields. Each
// bundle's VisitFields member function calls visitor->U32 etc. We do not
// overload operator() because a function name is easier to search for.

class VisitorBase : public Visitor {
 public:
  explicit VisitorBase() = default;
  ~VisitorBase() override { JXL_DASSERT(depth_ == 0); }

  // This is the only call site of Fields::VisitFields.
  // Ensures EndExtensions was called.
  Status Visit(Fields* fields) override {
    JXL_ENSURE(depth_ < Bundle::kMaxExtensions);
    depth_ += 1;
    extension_states_.Push();

    const Status ok = fields->VisitFields(this);

    if (ok) {
      // If VisitFields called BeginExtensions, must also call
      // EndExtensions.
      JXL_DASSERT(!extension_states_.IsBegun() || extension_states_.IsEnded());
    } else {
      // Failed, undefined state: don't care whether EndExtensions was
      // called.
    }

    extension_states_.Pop();
    JXL_DASSERT(depth_ != 0);
    depth_ -= 1;

    return ok;
  }

  // For visitors accepting a const Visitor, need to const-cast so we can call
  // the non-const Visitor::VisitFields. NOTE: C is not modified except the
  // `all_default` field by CanEncodeVisitor.
  Status VisitConst(const Fields& t) { return Visit(const_cast<Fields*>(&t)); }

  // Derived types (overridden by InitVisitor because it is unsafe to read
  // from *value there)

  Status Bool(bool default_value, bool* JXL_RESTRICT value) override {
    uint32_t bits = *value ? 1 : 0;
    JXL_RETURN_IF_ERROR(Bits(1, static_cast<uint32_t>(default_value), &bits));
    JXL_DASSERT(bits <= 1);
    *value = bits == 1;
    return true;
  }

  // Overridden by ReadVisitor and WriteVisitor.
  // Called before any conditional visit based on "extensions".
  // Overridden by ReadVisitor, CanEncodeVisitor and WriteVisitor.
  Status BeginExtensions(uint64_t* JXL_RESTRICT extensions) override {
    JXL_RETURN_IF_ERROR(U64(0, extensions));

    extension_states_.Begin();
    return true;
  }

  // Called after all extension fields (if any). Although non-extension
  // fields could be visited afterward, we prefer the convention that
  // extension fields are always the last to be visited. Overridden by
  // ReadVisitor.
  Status EndExtensions() override {
    extension_states_.End();
    return true;
  }

 private:
  size_t depth_ = 0;  // to check nesting
  ExtensionStates extension_states_;
};
}  // namespace fields_internal

Status CheckHasEnoughBits(Visitor* visitor, size_t bits);

}  // namespace jxl

#endif  // LIB_JXL_FIELDS_H_
