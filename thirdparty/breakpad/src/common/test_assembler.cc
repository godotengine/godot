// Copyright 2010 Google LLC
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google LLC nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// Original author: Jim Blandy <jimb@mozilla.com> <jimb@red-bean.com>

// test_assembler.cc: Implementation of google_breakpad::TestAssembler.
// See test_assembler.h for details.

#ifdef HAVE_CONFIG_H
#include <config.h>  // Must come first
#endif

#include "common/test_assembler.h"

#include <assert.h>
#include <stdio.h>

#include <iterator>

namespace google_breakpad {
namespace test_assembler {

using std::back_insert_iterator;

Label::Label() : value_(new Binding()) { }
Label::Label(uint64_t value) : value_(new Binding(value)) { }
Label::Label(const Label& label) {
  value_ = label.value_;
  value_->Acquire();
}
Label::~Label() {
  if (value_->Release()) delete value_;
}

Label& Label::operator=(uint64_t value) {
  value_->Set(NULL, value);
  return *this;
}

Label& Label::operator=(const Label& label) {
  value_->Set(label.value_, 0);
  return *this;
}

Label Label::operator+(uint64_t addend) const {
  Label l;
  l.value_->Set(this->value_, addend);
  return l;
}

Label Label::operator-(uint64_t subtrahend) const {
  Label l;
  l.value_->Set(this->value_, -subtrahend);
  return l;
}

// When NDEBUG is #defined, assert doesn't evaluate its argument. This
// means you can't simply use assert to check the return value of a
// function with necessary side effects.
//
// ALWAYS_EVALUATE_AND_ASSERT(x) evaluates x regardless of whether
// NDEBUG is #defined; when NDEBUG is not #defined, it further asserts
// that x is true.
#ifdef NDEBUG
#define ALWAYS_EVALUATE_AND_ASSERT(x) x
#else
#define ALWAYS_EVALUATE_AND_ASSERT(x) assert(x)
#endif

uint64_t Label::operator-(const Label& label) const {
  uint64_t offset;
  ALWAYS_EVALUATE_AND_ASSERT(IsKnownOffsetFrom(label, &offset));
  return offset;
}

uint64_t Label::Value() const {
  uint64_t v = 0;
  ALWAYS_EVALUATE_AND_ASSERT(IsKnownConstant(&v));
  return v;
};

bool Label::IsKnownConstant(uint64_t* value_p) const {
  Binding* base;
  uint64_t addend;
  value_->Get(&base, &addend);
  if (base != NULL) return false;
  if (value_p) *value_p = addend;
  return true;
}

bool Label::IsKnownOffsetFrom(const Label& label, uint64_t* offset_p) const
{
  Binding* label_base, *this_base;
  uint64_t label_addend, this_addend;
  label.value_->Get(&label_base, &label_addend);
  value_->Get(&this_base, &this_addend);
  // If this and label are related, Get will find their final
  // common ancestor, regardless of how indirect the relation is. This
  // comparison also handles the constant vs. constant case.
  if (this_base != label_base) return false;
  if (offset_p) *offset_p = this_addend - label_addend;
  return true;
}

Label::Binding::Binding() : base_(this), addend_(), reference_count_(1) { }

Label::Binding::Binding(uint64_t addend)
    : base_(NULL), addend_(addend), reference_count_(1) { }

Label::Binding::~Binding() {
  assert(reference_count_ == 0);
  if (base_ && base_ != this && base_->Release())
    delete base_;
}

void Label::Binding::Set(Binding* binding, uint64_t addend) {
  if (!base_ && !binding) {
    // We're equating two constants. This could be okay.
    assert(addend_ == addend);
  } else if (!base_) {
    // We are a known constant, but BINDING may not be, so turn the
    // tables and try to set BINDING's value instead.
    binding->Set(NULL, addend_ - addend);
  } else {
    if (binding) {
      // Find binding's final value. Since the final value is always either
      // completely unconstrained or a constant, never a reference to
      // another variable (otherwise, it wouldn't be final), this
      // guarantees we won't create cycles here, even for code like this:
      //   l = m, m = n, n = l;
      uint64_t binding_addend;
      binding->Get(&binding, &binding_addend);
      addend += binding_addend;
    }

    // It seems likely that setting a binding to itself is a bug
    // (although I can imagine this might turn out to be helpful to
    // permit).
    assert(binding != this);

    if (base_ != this) {
      // Set the other bindings on our chain as well. Note that this
      // is sufficient even though binding relationships form trees:
      // All binding operations traverse their chains to the end, and
      // all bindings related to us share some tail of our chain, so
      // they will see the changes we make here.
      base_->Set(binding, addend - addend_);
      // We're not going to use base_ any more.
      if (base_->Release()) delete base_;
    }
    
    // Adopt BINDING as our base. Note that it should be correct to
    // acquire here, after the release above, even though the usual
    // reference-counting rules call for acquiring first, and then
    // releasing: the self-reference assertion above should have
    // complained if BINDING were 'this' or anywhere along our chain,
    // so we didn't release BINDING.
    if (binding) binding->Acquire();
    base_ = binding;
    addend_ = addend;
  }
}

void Label::Binding::Get(Binding** base, uint64_t* addend) {
  if (base_ && base_ != this) {
    // Recurse to find the end of our reference chain (the root of our
    // tree), and then rewrite every binding along the chain to refer
    // to it directly, adjusting addends appropriately. (This is why
    // this member function isn't this-const.)
    Binding* final_base;
    uint64_t final_addend;
    base_->Get(&final_base, &final_addend);
    if (final_base) final_base->Acquire();
    if (base_->Release()) delete base_;
    base_ = final_base;
    addend_ += final_addend;
  }
  *base = base_;
  *addend = addend_;
}

template<typename Inserter>
static inline void InsertEndian(test_assembler::Endianness endianness,
                                size_t size, uint64_t number, Inserter dest) {
  assert(size > 0);
  if (endianness == kLittleEndian) {
    for (size_t i = 0; i < size; i++) {
      *dest++ = (char) (number & 0xff);
      number >>= 8;
    }
  } else {
    assert(endianness == kBigEndian);
    // The loop condition is odd, but it's correct for size_t.
    for (size_t i = size - 1; i < size; i--)
      *dest++ = (char) ((number >> (i * 8)) & 0xff);
  }
}

Section& Section::Append(Endianness endianness, size_t size, uint64_t number) {
  InsertEndian(endianness, size, number,
               back_insert_iterator<string>(contents_));
  return *this;
}

Section& Section::Append(Endianness endianness, size_t size,
                         const Label& label) {
  // If this label's value is known, there's no reason to waste an
  // entry in references_ on it.
  uint64_t value;
  if (label.IsKnownConstant(&value))
    return Append(endianness, size, value);

  // This will get caught when the references are resolved, but it's
  // nicer to find out earlier.
  assert(endianness != kUnsetEndian);

  references_.push_back(Reference(contents_.size(), endianness, size, label));
  contents_.append(size, 0);
  return *this;
}

#define ENDIANNESS_L kLittleEndian
#define ENDIANNESS_B kBigEndian
#define ENDIANNESS(e) ENDIANNESS_ ## e

#define DEFINE_SHORT_APPEND_NUMBER_ENDIAN(e, bits)                      \
  Section& Section::e ## bits(uint ## bits ## _t v) {                   \
    InsertEndian(ENDIANNESS(e), bits / 8, v,                            \
                 back_insert_iterator<string>(contents_));              \
    return *this;                                                       \
  }

#define DEFINE_SHORT_APPEND_LABEL_ENDIAN(e, bits)                       \
  Section& Section::e ## bits(const Label& v) {                         \
    return Append(ENDIANNESS(e), bits / 8, v);                          \
  }

// Define L16, B32, and friends.
#define DEFINE_SHORT_APPEND_ENDIAN(e, bits)                             \
  DEFINE_SHORT_APPEND_NUMBER_ENDIAN(e, bits)                            \
  DEFINE_SHORT_APPEND_LABEL_ENDIAN(e, bits)

DEFINE_SHORT_APPEND_LABEL_ENDIAN(L, 8);
DEFINE_SHORT_APPEND_LABEL_ENDIAN(B, 8);
DEFINE_SHORT_APPEND_ENDIAN(L, 16);
DEFINE_SHORT_APPEND_ENDIAN(L, 32);
DEFINE_SHORT_APPEND_ENDIAN(L, 64);
DEFINE_SHORT_APPEND_ENDIAN(B, 16);
DEFINE_SHORT_APPEND_ENDIAN(B, 32);
DEFINE_SHORT_APPEND_ENDIAN(B, 64);

#define DEFINE_SHORT_APPEND_NUMBER_DEFAULT(bits)                        \
  Section& Section::D ## bits(uint ## bits ## _t v) {                   \
    InsertEndian(endianness_, bits / 8, v,                              \
                 back_insert_iterator<string>(contents_));              \
    return *this;                                                       \
  }
#define DEFINE_SHORT_APPEND_LABEL_DEFAULT(bits)                         \
  Section& Section::D ## bits(const Label& v) {                         \
    return Append(endianness_, bits / 8, v);                            \
  }
#define DEFINE_SHORT_APPEND_DEFAULT(bits)                               \
  DEFINE_SHORT_APPEND_NUMBER_DEFAULT(bits)                              \
  DEFINE_SHORT_APPEND_LABEL_DEFAULT(bits)

DEFINE_SHORT_APPEND_LABEL_DEFAULT(8)
DEFINE_SHORT_APPEND_DEFAULT(16);
DEFINE_SHORT_APPEND_DEFAULT(32);
DEFINE_SHORT_APPEND_DEFAULT(64);

Section& Section::Append(const Section& section) {
  size_t base = contents_.size();
  contents_.append(section.contents_);
  for (vector<Reference>::const_iterator it = section.references_.begin();
       it != section.references_.end(); it++)
    references_.push_back(Reference(base + it->offset, it->endianness,
                                    it->size, it->label));
  return *this;
}

Section& Section::LEB128(long long value) {
  while (value < -0x40 || 0x3f < value) {
    contents_ += (value & 0x7f) | 0x80;
    if (value < 0)
      value = (value >> 7) | ~(((unsigned long long) -1) >> 7);
    else
      value = (value >> 7);
  }
  contents_ += value & 0x7f;
  return *this;
}

Section& Section::ULEB128(uint64_t value) {
  while (value > 0x7f) {
    contents_ += (value & 0x7f) | 0x80;
    value = (value >> 7);
  }
  contents_ += value;
  return *this;
}

Section& Section::Align(size_t alignment, uint8_t pad_byte) {
  // ALIGNMENT must be a power of two.
  assert(((alignment - 1) & alignment) == 0);
  size_t new_size = (contents_.size() + alignment - 1) & ~(alignment - 1);
  contents_.append(new_size - contents_.size(), pad_byte);
  assert((contents_.size() & (alignment - 1)) == 0);
  return *this;
}

void Section::Clear() {
  contents_.clear();
  references_.clear();
}

bool Section::GetContents(string* contents) {
  // For each label reference, find the label's value, and patch it into
  // the section's contents.
  for (size_t i = 0; i < references_.size(); i++) {
    Reference& r = references_[i];
    uint64_t value;
    if (!r.label.IsKnownConstant(&value)) {
      fprintf(stderr, "Undefined label #%zu at offset 0x%zx\n", i, r.offset);
      return false;
    }
    assert(r.offset < contents_.size());
    assert(contents_.size() - r.offset >= r.size);
    InsertEndian(r.endianness, r.size, value, contents_.begin() + r.offset);
  }
  contents->clear();
  std::swap(contents_, *contents);
  references_.clear();
  return true;
}

}  // namespace test_assembler
}  // namespace google_breakpad
