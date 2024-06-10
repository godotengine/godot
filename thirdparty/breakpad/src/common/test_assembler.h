// -*- mode: C++ -*-

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

// test-assembler.h: interface to class for building complex binary streams.

// To test the Breakpad symbol dumper and processor thoroughly, for
// all combinations of host system and minidump processor
// architecture, we need to be able to easily generate complex test
// data like debugging information and minidump files.
// 
// For example, if we want our unit tests to provide full code
// coverage for stack walking, it may be difficult to persuade the
// compiler to generate every possible sort of stack walking
// information that we want to support; there are probably DWARF CFI
// opcodes that GCC never emits. Similarly, if we want to test our
// error handling, we will need to generate damaged minidumps or
// debugging information that (we hope) the client or compiler will
// never produce on its own.
//
// google_breakpad::TestAssembler provides a predictable and
// (relatively) simple way to generate complex formatted data streams
// like minidumps and CFI. Furthermore, because TestAssembler is
// portable, developers without access to (say) Visual Studio or a
// SPARC assembler can still work on test data for those targets.

#ifndef PROCESSOR_TEST_ASSEMBLER_H_
#define PROCESSOR_TEST_ASSEMBLER_H_

#include <list>
#include <vector>
#include <string>

#include "common/using_std_string.h"
#include "google_breakpad/common/breakpad_types.h"

namespace google_breakpad {

using std::list;
using std::vector;

namespace test_assembler {

// A Label represents a value not yet known that we need to store in a
// section. As long as all the labels a section refers to are defined
// by the time we retrieve its contents as bytes, we can use undefined
// labels freely in that section's construction.
//
// A label can be in one of three states:
// - undefined,
// - defined as the sum of some other label and a constant, or
// - a constant.
// 
// A label's value never changes, but it can accumulate constraints.
// Adding labels and integers is permitted, and yields a label.
// Subtracting a constant from a label is permitted, and also yields a
// label. Subtracting two labels that have some relationship to each
// other is permitted, and yields a constant.
//
// For example:
//
//   Label a;               // a's value is undefined
//   Label b;               // b's value is undefined
//   {
//     Label c = a + 4;     // okay, even though a's value is unknown
//     b = c + 4;           // also okay; b is now a+8
//   }
//   Label d = b - 2;       // okay; d == a+6, even though c is gone
//   d.Value();             // error: d's value is not yet known
//   d - a;                 // is 6, even though their values are not known
//   a = 12;                // now b == 20, and d == 18
//   d.Value();             // 18: no longer an error
//   b.Value();             // 20
//   d = 10;                // error: d is already defined.
//
// Label objects' lifetimes are unconstrained: notice that, in the
// above example, even though a and b are only related through c, and
// c goes out of scope, the assignment to a sets b's value as well. In
// particular, it's not necessary to ensure that a Label lives beyond
// Sections that refer to it.
class Label {
 public:
  Label();                      // An undefined label.
  Label(uint64_t value);       // A label with a fixed value
  Label(const Label& value);    // A label equal to another.
  ~Label();

  // Return this label's value; it must be known.
  //
  // Providing this as a cast operator is nifty, but the conversions
  // happen in unexpected places. In particular, ISO C++ says that
  // Label + size_t becomes ambigious, because it can't decide whether
  // to convert the Label to a uint64_t and then to a size_t, or use
  // the overloaded operator that returns a new label, even though the
  // former could fail if the label is not yet defined and the latter won't.
  uint64_t Value() const;

  Label& operator=(uint64_t value);
  Label& operator=(const Label& value);
  Label operator+(uint64_t addend) const;
  Label operator-(uint64_t subtrahend) const;
  uint64_t operator-(const Label& subtrahend) const;

  // We could also provide == and != that work on undefined, but
  // related, labels.

  // Return true if this label's value is known. If VALUE_P is given,
  // set *VALUE_P to the known value if returning true.
  bool IsKnownConstant(uint64_t* value_p = NULL) const;

  // Return true if the offset from LABEL to this label is known. If
  // OFFSET_P is given, set *OFFSET_P to the offset when returning true.
  //
  // You can think of l.KnownOffsetFrom(m, &d) as being like 'd = l-m',
  // except that it also returns a value indicating whether the
  // subtraction is possible given what we currently know of l and m.
  // It can be possible even if we don't know l and m's values. For
  // example:
  // 
  //   Label l, m;
  //   m = l + 10;
  //   l.IsKnownConstant();             // false
  //   m.IsKnownConstant();             // false
  //   uint64_t d;                     
  //   l.IsKnownOffsetFrom(m, &d);      // true, and sets d to -10.
  //   l-m                              // -10
  //   m-l                              // 10
  //   m.Value()                        // error: m's value is not known
  bool IsKnownOffsetFrom(const Label& label, uint64_t* offset_p = NULL) const;

 private:
  // A label's value, or if that is not yet known, how the value is
  // related to other labels' values. A binding may be:
  // - a known constant,
  // - constrained to be equal to some other binding plus a constant, or
  // - unconstrained, and free to take on any value.
  //
  // Many labels may point to a single binding, and each binding may
  // refer to another, so bindings and labels form trees whose leaves
  // are labels, whose interior nodes (and roots) are bindings, and
  // where links point from children to parents. Bindings are
  // reference counted, allowing labels to be lightweight, copyable,
  // assignable, placed in containers, and so on.
  class Binding {
   public:
    Binding();
    Binding(uint64_t addend);
    ~Binding();

    // Increment our reference count.
    void Acquire() { reference_count_++; };
    // Decrement our reference count, and return true if it is zero.
    bool Release() { return --reference_count_ == 0; }

    // Set this binding to be equal to BINDING + ADDEND. If BINDING is
    // NULL, then set this binding to the known constant ADDEND.
    // Update every binding on this binding's chain to point directly
    // to BINDING, or to be a constant, with addends adjusted
    // appropriately.
    void Set(Binding* binding, uint64_t value);

    // Return what we know about the value of this binding.
    // - If this binding's value is a known constant, set BASE to
    //   NULL, and set ADDEND to its value.
    // - If this binding is not a known constant but related to other
    //   bindings, set BASE to the binding at the end of the relation
    //   chain (which will always be unconstrained), and set ADDEND to the
    //   value to add to that binding's value to get this binding's
    //   value.
    // - If this binding is unconstrained, set BASE to this, and leave
    //   ADDEND unchanged.
    void Get(Binding** base, uint64_t* addend);

   private:
    // There are three cases:
    //
    // - A binding representing a known constant value has base_ NULL,
    //   and addend_ equal to the value.
    //
    // - A binding representing a completely unconstrained value has
    //   base_ pointing to this; addend_ is unused.
    //
    // - A binding whose value is related to some other binding's
    //   value has base_ pointing to that other binding, and addend_
    //   set to the amount to add to that binding's value to get this
    //   binding's value. We only represent relationships of the form
    //   x = y+c.
    //
    // Thus, the bind_ links form a chain terminating in either a
    // known constant value or a completely unconstrained value. Most
    // operations on bindings do path compression: they change every
    // binding on the chain to point directly to the final value,
    // adjusting addends as appropriate.
    Binding* base_;
    uint64_t addend_;

    // The number of Labels and Bindings pointing to this binding.
    // (When a binding points to itself, indicating a completely
    // unconstrained binding, that doesn't count as a reference.)
    int reference_count_;
  };

  // This label's value.
  Binding* value_;
};

inline Label operator+(uint64_t a, const Label& l) { return l + a; }
// Note that int-Label isn't defined, as negating a Label is not an
// operation we support.

// Conventions for representing larger numbers as sequences of bytes.
enum Endianness {
  kBigEndian,        // Big-endian: the most significant byte comes first.
  kLittleEndian,     // Little-endian: the least significant byte comes first.
  kUnsetEndian,      // used internally
};
 
// A section is a sequence of bytes, constructed by appending bytes
// to the end. Sections have a convenient and flexible set of member
// functions for appending data in various formats: big-endian and
// little-endian signed and unsigned values of different sizes;
// LEB128 and ULEB128 values (see below), and raw blocks of bytes.
//
// If you need to append a value to a section that is not convenient
// to compute immediately, you can create a label, append the
// label's value to the section, and then set the label's value
// later, when it's convenient to do so. Once a label's value is
// known, the section class takes care of updating all previously
// appended references to it.
//
// Once all the labels to which a section refers have had their
// values determined, you can get a copy of the section's contents
// as a string.
//
// Note that there is no specified "start of section" label. This is
// because there are typically several different meanings for "the
// start of a section": the offset of the section within an object
// file, the address in memory at which the section's content appear,
// and so on. It's up to the code that uses the Section class to 
// keep track of these explicitly, as they depend on the application.
class Section {
 public:
  Section(Endianness endianness = kUnsetEndian)
      : endianness_(endianness) { };

  // A base class destructor should be either public and virtual,
  // or protected and nonvirtual.
  virtual ~Section() { };

  // Set the default endianness of this section to ENDIANNESS. This
  // sets the behavior of the D<N> appending functions. If the
  // assembler's default endianness was set, this is the 
  void set_endianness(Endianness endianness) {
    endianness_ = endianness;
  }

  // Return the default endianness of this section.
  Endianness endianness() const { return endianness_; }

  // Append the SIZE bytes at DATA or the contents of STRING to the
  // end of this section. Return a reference to this section.
  Section& Append(const uint8_t* data, size_t size) {
    contents_.append(reinterpret_cast<const char*>(data), size);
    return *this;
  };
  Section& Append(const string& data) {
    contents_.append(data);
    return *this;
  };

  // Append SIZE copies of BYTE to the end of this section. Return a
  // reference to this section.
  Section& Append(size_t size, uint8_t byte) {
    contents_.append(size, (char) byte);
    return *this;
  }
      
  // Append NUMBER to this section. ENDIANNESS is the endianness to
  // use to write the number. SIZE is the length of the number in
  // bytes. Return a reference to this section.
  Section& Append(Endianness endianness, size_t size, uint64_t number);
  Section& Append(Endianness endianness, size_t size, const Label& label);

  // Append SECTION to the end of this section. The labels SECTION
  // refers to need not be defined yet.
  //
  // Note that this has no effect on any Labels' values, or on
  // SECTION. If placing SECTION within 'this' provides new
  // constraints on existing labels' values, then it's up to the
  // caller to fiddle with those labels as needed.
  Section& Append(const Section& section);

  // Append the contents of DATA as a series of bytes terminated by
  // a NULL character.
  Section& AppendCString(const string& data) {
    Append(data);
    contents_ += '\0';
    return *this;
  }

  // Append at most SIZE bytes from DATA; if DATA is less than SIZE bytes
  // long, pad with '\0' characters.
  Section& AppendCString(const string& data, size_t size) {
    contents_.append(data, 0, size);
    if (data.size() < size)
      Append(size - data.size(), 0);
    return *this;
  }

  // Append VALUE or LABEL to this section, with the given bit width and
  // endianness. Return a reference to this section.
  //
  // The names of these functions have the form <ENDIANNESS><BITWIDTH>:
  // <ENDIANNESS> is either 'L' (little-endian, least significant byte first),
  //                        'B' (big-endian, most significant byte first), or
  //                        'D' (default, the section's default endianness)
  // <BITWIDTH> is 8, 16, 32, or 64.
  //
  // Since endianness doesn't matter for a single byte, all the
  // <BITWIDTH>=8 functions are equivalent.
  //
  // These can be used to write both signed and unsigned values, as
  // the compiler will properly sign-extend a signed value before
  // passing it to the function, at which point the function's
  // behavior is the same either way.
  Section& L8(uint8_t value) { contents_ += value; return *this; }
  Section& B8(uint8_t value) { contents_ += value; return *this; }
  Section& D8(uint8_t value) { contents_ += value; return *this; }
  Section &L16(uint16_t), &L32(uint32_t), &L64(uint64_t),
          &B16(uint16_t), &B32(uint32_t), &B64(uint64_t),
          &D16(uint16_t), &D32(uint32_t), &D64(uint64_t);
  Section &L8(const Label& label),  &L16(const Label& label),
          &L32(const Label& label), &L64(const Label& label),
          &B8(const Label& label),  &B16(const Label& label),
          &B32(const Label& label), &B64(const Label& label),
          &D8(const Label& label),  &D16(const Label& label),
          &D32(const Label& label), &D64(const Label& label);

  // Append VALUE in a signed LEB128 (Little-Endian Base 128) form.
  // 
  // The signed LEB128 representation of an integer N is a variable
  // number of bytes:
  //
  // - If N is between -0x40 and 0x3f, then its signed LEB128
  //   representation is a single byte whose value is N.
  // 
  // - Otherwise, its signed LEB128 representation is (N & 0x7f) |
  //   0x80, followed by the signed LEB128 representation of N / 128,
  //   rounded towards negative infinity.
  //
  // In other words, we break VALUE into groups of seven bits, put
  // them in little-endian order, and then write them as eight-bit
  // bytes with the high bit on all but the last.
  //
  // Note that VALUE cannot be a Label (we would have to implement
  // relaxation).
  Section& LEB128(long long value);

  // Append VALUE in unsigned LEB128 (Little-Endian Base 128) form.
  // 
  // The unsigned LEB128 representation of an integer N is a variable
  // number of bytes:
  //
  // - If N is between 0 and 0x7f, then its unsigned LEB128
  //   representation is a single byte whose value is N.
  // 
  // - Otherwise, its unsigned LEB128 representation is (N & 0x7f) |
  //   0x80, followed by the unsigned LEB128 representation of N /
  //   128, rounded towards negative infinity.
  //
  // Note that VALUE cannot be a Label (we would have to implement
  // relaxation).
  Section& ULEB128(uint64_t value);

  // Jump to the next location aligned on an ALIGNMENT-byte boundary,
  // relative to the start of the section. Fill the gap with PAD_BYTE.
  // ALIGNMENT must be a power of two. Return a reference to this
  // section.
  Section& Align(size_t alignment, uint8_t pad_byte = 0);

  // Clear the contents of this section.
  void Clear();

  // Return the current size of the section.
  size_t Size() const { return contents_.size(); }

  // Return a label representing the start of the section.
  // 
  // It is up to the user whether this label represents the section's
  // position in an object file, the section's address in memory, or
  // what have you; some applications may need both, in which case
  // this simple-minded interface won't be enough. This class only
  // provides a single start label, for use with the Here and Mark
  // member functions.
  //
  // Ideally, we'd provide this in a subclass that actually knows more
  // about the application at hand and can provide an appropriate
  // collection of start labels. But then the appending member
  // functions like Append and D32 would return a reference to the
  // base class, not the derived class, and the chaining won't work.
  // Since the only value here is in pretty notation, that's a fatal
  // flaw.
  Label start() const { return start_; }

  // Return a label representing the point at which the next Appended
  // item will appear in the section, relative to start().
  Label Here() const { return start_ + Size(); }

  // Set *LABEL to Here, and return a reference to this section.
  Section& Mark(Label* label) { *label = Here(); return *this; }

  // If there are no undefined label references left in this
  // section, set CONTENTS to the contents of this section, as a
  // string, and clear this section. Return true on success, or false
  // if there were still undefined labels.
  bool GetContents(string* contents);

 private:
  // Used internally. A reference to a label's value.
  struct Reference {
    Reference(size_t set_offset, Endianness set_endianness,  size_t set_size,
              const Label& set_label)
        : offset(set_offset), endianness(set_endianness), size(set_size),
          label(set_label) { }
      
    // The offset of the reference within the section.
    size_t offset;

    // The endianness of the reference.
    Endianness endianness;

    // The size of the reference.
    size_t size;

    // The label to which this is a reference.
    Label label;
  };

  // The default endianness of this section.
  Endianness endianness_;

  // The contents of the section.
  string contents_;
  
  // References to labels within those contents.
  vector<Reference> references_;

  // A label referring to the beginning of the section.
  Label start_;
};

}  // namespace test_assembler
}  // namespace google_breakpad

#endif  // PROCESSOR_TEST_ASSEMBLER_H_
