//===-- llvm/Attributes.h - Container for Attributes ------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file contains the simple types necessary to represent the
/// attributes associated with functions and their calls.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_IR_ATTRIBUTES_H
#define LLVM_IR_ATTRIBUTES_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/PointerLikeTypeTraits.h"
#include <bitset>
#include <cassert>
#include <map>
#include <string>

namespace llvm {

class AttrBuilder;
class AttributeImpl;
class AttributeSetImpl;
class AttributeSetNode;
class Constant;
template<typename T> struct DenseMapInfo;
class LLVMContext;
class Type;

//===----------------------------------------------------------------------===//
/// \class
/// \brief Functions, function parameters, and return types can have attributes
/// to indicate how they should be treated by optimizations and code
/// generation. This class represents one of those attributes. It's light-weight
/// and should be passed around by-value.
class Attribute {
public:
  /// This enumeration lists the attributes that can be associated with
  /// parameters, function results, or the function itself.
  ///
  /// Note: The `uwtable' attribute is about the ABI or the user mandating an
  /// entry in the unwind table. The `nounwind' attribute is about an exception
  /// passing by the function.
  ///
  /// In a theoretical system that uses tables for profiling and SjLj for
  /// exceptions, they would be fully independent. In a normal system that uses
  /// tables for both, the semantics are:
  ///
  /// nil                = Needs an entry because an exception might pass by.
  /// nounwind           = No need for an entry
  /// uwtable            = Needs an entry because the ABI says so and because
  ///                      an exception might pass by.
  /// uwtable + nounwind = Needs an entry because the ABI says so.

  enum AttrKind {
    // IR-Level Attributes
    None,                  ///< No attributes have been set
    Alignment,             ///< Alignment of parameter (5 bits)
                           ///< stored as log2 of alignment with +1 bias
                           ///< 0 means unaligned (different from align(1))
    AlwaysInline,          ///< inline=always
    Builtin,               ///< Callee is recognized as a builtin, despite
                           ///< nobuiltin attribute on its declaration.
    ByVal,                 ///< Pass structure by value
    InAlloca,              ///< Pass structure in an alloca
    Cold,                  ///< Marks function as being in a cold path.
    Convergent,            ///< Can only be moved to control-equivalent blocks
    InlineHint,            ///< Source said inlining was desirable
    InReg,                 ///< Force argument to be passed in register
    JumpTable,             ///< Build jump-instruction tables and replace refs.
    MinSize,               ///< Function must be optimized for size first
    Naked,                 ///< Naked function
    Nest,                  ///< Nested function static chain
    NoAlias,               ///< Considered to not alias after call
    NoBuiltin,             ///< Callee isn't recognized as a builtin
    NoCapture,             ///< Function creates no aliases of pointer
    NoDuplicate,           ///< Call cannot be duplicated
    NoImplicitFloat,       ///< Disable implicit floating point insts
    NoInline,              ///< inline=never
    NonLazyBind,           ///< Function is called early and/or
                           ///< often, so lazy binding isn't worthwhile
    NonNull,               ///< Pointer is known to be not null
    Dereferenceable,       ///< Pointer is known to be dereferenceable
    DereferenceableOrNull, ///< Pointer is either null or dereferenceable
    NoRedZone,             ///< Disable redzone
    NoReturn,              ///< Mark the function as not returning
    NoUnwind,              ///< Function doesn't unwind stack
    OptimizeForSize,       ///< opt_size
    OptimizeNone,          ///< Function must not be optimized.
    ReadNone,              ///< Function does not access memory
    ReadOnly,              ///< Function only reads from memory
    ArgMemOnly,            ///< Funciton can access memory only using pointers
                           ///< based on its arguments.
    Returned,              ///< Return value is always equal to this argument
    ReturnsTwice,          ///< Function can return twice
    SExt,                  ///< Sign extended before/after call
    StackAlignment,        ///< Alignment of stack for function (3 bits)
                           ///< stored as log2 of alignment with +1 bias 0
                           ///< means unaligned (different from
                           ///< alignstack=(1))
    StackProtect,          ///< Stack protection.
    StackProtectReq,       ///< Stack protection required.
    StackProtectStrong,    ///< Strong Stack protection.
    SafeStack,             ///< Safe Stack protection.
    StructRet,             ///< Hidden pointer to structure to return
    SanitizeAddress,       ///< AddressSanitizer is on.
    SanitizeThread,        ///< ThreadSanitizer is on.
    SanitizeMemory,        ///< MemorySanitizer is on.
    UWTable,               ///< Function must be in a unwind table
    ZExt,                  ///< Zero extended before/after call

    EndAttrKinds           ///< Sentinal value useful for loops
  };
private:
  AttributeImpl *pImpl;
  Attribute(AttributeImpl *A) : pImpl(A) {}
public:
  Attribute() : pImpl(nullptr) {}

  //===--------------------------------------------------------------------===//
  // Attribute Construction
  //===--------------------------------------------------------------------===//

  /// \brief Return a uniquified Attribute object.
  static Attribute get(LLVMContext &Context, AttrKind Kind, uint64_t Val = 0);
  static Attribute get(LLVMContext &Context, StringRef Kind,
                       StringRef Val = StringRef());

  /// \brief Return a uniquified Attribute object that has the specific
  /// alignment set.
  static Attribute getWithAlignment(LLVMContext &Context, uint64_t Align);
  static Attribute getWithStackAlignment(LLVMContext &Context, uint64_t Align);
  static Attribute getWithDereferenceableBytes(LLVMContext &Context,
                                              uint64_t Bytes);
  static Attribute getWithDereferenceableOrNullBytes(LLVMContext &Context,
                                                     uint64_t Bytes);

  //===--------------------------------------------------------------------===//
  // Attribute Accessors
  //===--------------------------------------------------------------------===//

  /// \brief Return true if the attribute is an Attribute::AttrKind type.
  bool isEnumAttribute() const;

  /// \brief Return true if the attribute is an integer attribute.
  bool isIntAttribute() const;

  /// \brief Return true if the attribute is a string (target-dependent)
  /// attribute.
  bool isStringAttribute() const;

  /// \brief Return true if the attribute is present.
  bool hasAttribute(AttrKind Val) const;

  /// \brief Return true if the target-dependent attribute is present.
  bool hasAttribute(StringRef Val) const;

  /// \brief Return the attribute's kind as an enum (Attribute::AttrKind). This
  /// requires the attribute to be an enum or alignment attribute.
  Attribute::AttrKind getKindAsEnum() const;

  /// \brief Return the attribute's value as an integer. This requires that the
  /// attribute be an alignment attribute.
  uint64_t getValueAsInt() const;

  /// \brief Return the attribute's kind as a string. This requires the
  /// attribute to be a string attribute.
  StringRef getKindAsString() const;

  /// \brief Return the attribute's value as a string. This requires the
  /// attribute to be a string attribute.
  StringRef getValueAsString() const;

  /// \brief Returns the alignment field of an attribute as a byte alignment
  /// value.
  unsigned getAlignment() const;

  /// \brief Returns the stack alignment field of an attribute as a byte
  /// alignment value.
  unsigned getStackAlignment() const;

  /// \brief Returns the number of dereferenceable bytes from the
  /// dereferenceable attribute (or zero if unknown).
  uint64_t getDereferenceableBytes() const;

  /// \brief Returns the number of dereferenceable_or_null bytes from the
  /// dereferenceable_or_null attribute (or zero if unknown).
  uint64_t getDereferenceableOrNullBytes() const;

  /// \brief The Attribute is converted to a string of equivalent mnemonic. This
  /// is, presumably, for writing out the mnemonics for the assembly writer.
  std::string getAsString(bool InAttrGrp = false) const;

  /// \brief Equality and non-equality operators.
  bool operator==(Attribute A) const { return pImpl == A.pImpl; }
  bool operator!=(Attribute A) const { return pImpl != A.pImpl; }

  /// \brief Less-than operator. Useful for sorting the attributes list.
  bool operator<(Attribute A) const;

  void Profile(FoldingSetNodeID &ID) const {
    ID.AddPointer(pImpl);
  }
};

//===----------------------------------------------------------------------===//
/// \class
/// \brief This class holds the attributes for a function, its return value, and
/// its parameters. You access the attributes for each of them via an index into
/// the AttributeSet object. The function attributes are at index
/// `AttributeSet::FunctionIndex', the return value is at index
/// `AttributeSet::ReturnIndex', and the attributes for the parameters start at
/// index `1'.
class AttributeSet {
public:
  enum AttrIndex : unsigned {
    ReturnIndex = 0U,
    FunctionIndex = ~0U
  };
private:
  friend class AttrBuilder;
  friend class AttributeSetImpl;
  template <typename Ty> friend struct DenseMapInfo;

  /// \brief The attributes that we are managing. This can be null to represent
  /// the empty attributes list.
  AttributeSetImpl *pImpl;

  /// \brief The attributes for the specified index are returned.
  AttributeSetNode *getAttributes(unsigned Index) const;

  /// \brief Create an AttributeSet with the specified parameters in it.
  static AttributeSet get(LLVMContext &C,
                          ArrayRef<std::pair<unsigned, Attribute> > Attrs);
  static AttributeSet get(LLVMContext &C,
                          ArrayRef<std::pair<unsigned,
                                             AttributeSetNode*> > Attrs);

  static AttributeSet getImpl(LLVMContext &C,
                              ArrayRef<std::pair<unsigned,
                                                 AttributeSetNode*> > Attrs);


  explicit AttributeSet(AttributeSetImpl *LI) : pImpl(LI) {}
public:
  AttributeSet() : pImpl(nullptr) {}

  //===--------------------------------------------------------------------===//
  // AttributeSet Construction and Mutation
  //===--------------------------------------------------------------------===//

  /// \brief Return an AttributeSet with the specified parameters in it.
  static AttributeSet get(LLVMContext &C, ArrayRef<AttributeSet> Attrs);
  static AttributeSet get(LLVMContext &C, unsigned Index,
                          ArrayRef<Attribute::AttrKind> Kind);
  static AttributeSet get(LLVMContext &C, unsigned Index, const AttrBuilder &B);

  /// \brief Add an attribute to the attribute set at the given index. Because
  /// attribute sets are immutable, this returns a new set.
  AttributeSet addAttribute(LLVMContext &C, unsigned Index,
                            Attribute::AttrKind Attr) const;

  /// \brief Add an attribute to the attribute set at the given index. Because
  /// attribute sets are immutable, this returns a new set.
  AttributeSet addAttribute(LLVMContext &C, unsigned Index,
                            StringRef Kind) const;
  AttributeSet addAttribute(LLVMContext &C, unsigned Index,
                            StringRef Kind, StringRef Value) const;

  /// \brief Add attributes to the attribute set at the given index. Because
  /// attribute sets are immutable, this returns a new set.
  AttributeSet addAttributes(LLVMContext &C, unsigned Index,
                             AttributeSet Attrs) const;

  /// \brief Remove the specified attribute at the specified index from this
  /// attribute list. Because attribute lists are immutable, this returns the
  /// new list.
  AttributeSet removeAttribute(LLVMContext &C, unsigned Index, 
                               Attribute::AttrKind Attr) const;

  /// \brief Remove the specified attributes at the specified index from this
  /// attribute list. Because attribute lists are immutable, this returns the
  /// new list.
  AttributeSet removeAttributes(LLVMContext &C, unsigned Index, 
                                AttributeSet Attrs) const;

  /// \brief Remove the specified attributes at the specified index from this
  /// attribute list. Because attribute lists are immutable, this returns the
  /// new list.
  AttributeSet removeAttributes(LLVMContext &C, unsigned Index,
                                const AttrBuilder &Attrs) const;

  /// \brief Add the dereferenceable attribute to the attribute set at the given
  /// index. Because attribute sets are immutable, this returns a new set.
  AttributeSet addDereferenceableAttr(LLVMContext &C, unsigned Index,
                                      uint64_t Bytes) const;

  /// \brief Add the dereferenceable_or_null attribute to the attribute set at
  /// the given index. Because attribute sets are immutable, this returns a new
  /// set.
  AttributeSet addDereferenceableOrNullAttr(LLVMContext &C, unsigned Index,
                                            uint64_t Bytes) const;

  //===--------------------------------------------------------------------===//
  // AttributeSet Accessors
  //===--------------------------------------------------------------------===//

  /// \brief Retrieve the LLVM context.
  LLVMContext &getContext() const;

  /// \brief The attributes for the specified index are returned.
  AttributeSet getParamAttributes(unsigned Index) const;

  /// \brief The attributes for the ret value are returned.
  AttributeSet getRetAttributes() const;

  /// \brief The function attributes are returned.
  AttributeSet getFnAttributes() const;

  /// \brief Return true if the attribute exists at the given index.
  bool hasAttribute(unsigned Index, Attribute::AttrKind Kind) const;

  /// \brief Return true if the attribute exists at the given index.
  bool hasAttribute(unsigned Index, StringRef Kind) const;

  /// \brief Return true if attribute exists at the given index.
  bool hasAttributes(unsigned Index) const;

  /// \brief Return true if the specified attribute is set for at least one
  /// parameter or for the return value.
  bool hasAttrSomewhere(Attribute::AttrKind Attr) const;

  /// \brief Return the attribute object that exists at the given index.
  Attribute getAttribute(unsigned Index, Attribute::AttrKind Kind) const;

  /// \brief Return the attribute object that exists at the given index.
  Attribute getAttribute(unsigned Index, StringRef Kind) const;

  /// \brief Return the alignment for the specified function parameter.
  unsigned getParamAlignment(unsigned Index) const;

  /// \brief Get the stack alignment.
  unsigned getStackAlignment(unsigned Index) const;

  /// \brief Get the number of dereferenceable bytes (or zero if unknown).
  uint64_t getDereferenceableBytes(unsigned Index) const;

  /// \brief Get the number of dereferenceable_or_null bytes (or zero if
  /// unknown).
  uint64_t getDereferenceableOrNullBytes(unsigned Index) const;

  /// \brief Return the attributes at the index as a string.
  std::string getAsString(unsigned Index, bool InAttrGrp = false) const;

  typedef ArrayRef<Attribute>::iterator iterator;

  iterator begin(unsigned Slot) const;
  iterator end(unsigned Slot) const;

  /// operator==/!= - Provide equality predicates.
  bool operator==(const AttributeSet &RHS) const {
    return pImpl == RHS.pImpl;
  }
  bool operator!=(const AttributeSet &RHS) const {
    return pImpl != RHS.pImpl;
  }

  //===--------------------------------------------------------------------===//
  // AttributeSet Introspection
  //===--------------------------------------------------------------------===//

  // FIXME: Remove this.
  uint64_t Raw(unsigned Index) const;

  /// \brief Return a raw pointer that uniquely identifies this attribute list.
  void *getRawPointer() const {
    return pImpl;
  }

  /// \brief Return true if there are no attributes.
  bool isEmpty() const {
    return getNumSlots() == 0;
  }

  /// \brief Return the number of slots used in this attribute list.  This is
  /// the number of arguments that have an attribute set on them (including the
  /// function itself).
  unsigned getNumSlots() const;

  /// \brief Return the index for the given slot.
  unsigned getSlotIndex(unsigned Slot) const;

  /// \brief Return the attributes at the given slot.
  AttributeSet getSlotAttributes(unsigned Slot) const;

  void dump() const;
};

//===----------------------------------------------------------------------===//
/// \class
/// \brief Provide DenseMapInfo for AttributeSet.
template<> struct DenseMapInfo<AttributeSet> {
  static inline AttributeSet getEmptyKey() {
    uintptr_t Val = static_cast<uintptr_t>(-1);
    Val <<= PointerLikeTypeTraits<void*>::NumLowBitsAvailable;
    return AttributeSet(reinterpret_cast<AttributeSetImpl*>(Val));
  }
  static inline AttributeSet getTombstoneKey() {
    uintptr_t Val = static_cast<uintptr_t>(-2);
    Val <<= PointerLikeTypeTraits<void*>::NumLowBitsAvailable;
    return AttributeSet(reinterpret_cast<AttributeSetImpl*>(Val));
  }
  static unsigned getHashValue(AttributeSet AS) {
    return (unsigned((uintptr_t)AS.pImpl) >> 4) ^
           (unsigned((uintptr_t)AS.pImpl) >> 9);
  }
  static bool isEqual(AttributeSet LHS, AttributeSet RHS) { return LHS == RHS; }
};
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
/// \class
/// \brief This class is used in conjunction with the Attribute::get method to
/// create an Attribute object. The object itself is uniquified. The Builder's
/// value, however, is not. So this can be used as a quick way to test for
/// equality, presence of attributes, etc.
class AttrBuilder {
  std::bitset<Attribute::EndAttrKinds> Attrs;
  std::map<std::string, std::string> TargetDepAttrs;
  uint64_t Alignment;
  uint64_t StackAlignment;
  uint64_t DerefBytes;
  uint64_t DerefOrNullBytes;
public:
  AttrBuilder()
      : Attrs(0), Alignment(0), StackAlignment(0), DerefBytes(0),
        DerefOrNullBytes(0) {}
  explicit AttrBuilder(uint64_t Val)
      : Attrs(0), Alignment(0), StackAlignment(0), DerefBytes(0),
        DerefOrNullBytes(0) {
    addRawValue(Val);
  }
  AttrBuilder(const Attribute &A)
      : Attrs(0), Alignment(0), StackAlignment(0), DerefBytes(0),
        DerefOrNullBytes(0) {
    addAttribute(A);
  }
  AttrBuilder(AttributeSet AS, unsigned Idx);

  void clear();

  /// \brief Add an attribute to the builder.
  AttrBuilder &addAttribute(Attribute::AttrKind Val);

  /// \brief Add the Attribute object to the builder.
  AttrBuilder &addAttribute(Attribute A);

  /// \brief Add the target-dependent attribute to the builder.
  AttrBuilder &addAttribute(StringRef A, StringRef V = StringRef());

  /// \brief Remove an attribute from the builder.
  AttrBuilder &removeAttribute(Attribute::AttrKind Val);

  /// \brief Remove the attributes from the builder.
  AttrBuilder &removeAttributes(AttributeSet A, uint64_t Index);

  /// \brief Remove the target-dependent attribute to the builder.
  AttrBuilder &removeAttribute(StringRef A);

  /// \brief Add the attributes from the builder.
  AttrBuilder &merge(const AttrBuilder &B);

  /// \brief Remove the attributes from the builder.
  AttrBuilder &remove(const AttrBuilder &B);

  /// \brief Return true if the builder has any attribute that's in the
  /// specified builder.
  bool overlaps(const AttrBuilder &B) const;

  /// \brief Return true if the builder has the specified attribute.
  bool contains(Attribute::AttrKind A) const {
    assert((unsigned)A < Attribute::EndAttrKinds && "Attribute out of range!");
    return Attrs[A];
  }

  /// \brief Return true if the builder has the specified target-dependent
  /// attribute.
  bool contains(StringRef A) const;

  /// \brief Return true if the builder has IR-level attributes.
  bool hasAttributes() const;

  /// \brief Return true if the builder has any attribute that's in the
  /// specified attribute.
  bool hasAttributes(AttributeSet A, uint64_t Index) const;

  /// \brief Return true if the builder has an alignment attribute.
  bool hasAlignmentAttr() const;

  /// \brief Retrieve the alignment attribute, if it exists.
  uint64_t getAlignment() const { return Alignment; }

  /// \brief Retrieve the stack alignment attribute, if it exists.
  uint64_t getStackAlignment() const { return StackAlignment; }

  /// \brief Retrieve the number of dereferenceable bytes, if the dereferenceable
  /// attribute exists (zero is returned otherwise).
  uint64_t getDereferenceableBytes() const { return DerefBytes; }

  /// \brief Retrieve the number of dereferenceable_or_null bytes, if the
  /// dereferenceable_or_null attribute exists (zero is returned otherwise).
  uint64_t getDereferenceableOrNullBytes() const { return DerefOrNullBytes; }

  /// \brief This turns an int alignment (which must be a power of 2) into the
  /// form used internally in Attribute.
  AttrBuilder &addAlignmentAttr(unsigned Align);

  /// \brief This turns an int stack alignment (which must be a power of 2) into
  /// the form used internally in Attribute.
  AttrBuilder &addStackAlignmentAttr(unsigned Align);

  /// \brief This turns the number of dereferenceable bytes into the form used
  /// internally in Attribute.
  AttrBuilder &addDereferenceableAttr(uint64_t Bytes);

  /// \brief This turns the number of dereferenceable_or_null bytes into the
  /// form used internally in Attribute.
  AttrBuilder &addDereferenceableOrNullAttr(uint64_t Bytes);

  /// \brief Return true if the builder contains no target-independent
  /// attributes.
  bool empty() const { return Attrs.none(); }

  // Iterators for target-dependent attributes.
  typedef std::pair<std::string, std::string>                td_type;
  typedef std::map<std::string, std::string>::iterator       td_iterator;
  typedef std::map<std::string, std::string>::const_iterator td_const_iterator;
  typedef llvm::iterator_range<td_iterator>                  td_range;
  typedef llvm::iterator_range<td_const_iterator>            td_const_range;

  td_iterator td_begin()             { return TargetDepAttrs.begin(); }
  td_iterator td_end()               { return TargetDepAttrs.end(); }

  td_const_iterator td_begin() const { return TargetDepAttrs.begin(); }
  td_const_iterator td_end() const   { return TargetDepAttrs.end(); }

  td_range td_attrs() { return td_range(td_begin(), td_end()); }
  td_const_range td_attrs() const {
    return td_const_range(td_begin(), td_end());
  }

  bool td_empty() const              { return TargetDepAttrs.empty(); }

  bool operator==(const AttrBuilder &B);
  bool operator!=(const AttrBuilder &B) {
    return !(*this == B);
  }

  // FIXME: Remove this in 4.0.

  /// \brief Add the raw value to the internal representation.
  AttrBuilder &addRawValue(uint64_t Val);
};

namespace AttributeFuncs {

/// \brief Which attributes cannot be applied to a type.
AttrBuilder typeIncompatible(const Type *Ty);

} // end AttributeFuncs namespace

} // end llvm namespace

#endif
