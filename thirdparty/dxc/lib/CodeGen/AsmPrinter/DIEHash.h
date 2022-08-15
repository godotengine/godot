//===-- llvm/CodeGen/DIEHash.h - Dwarf Hashing Framework -------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for DWARF4 hashing of DIEs.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_CODEGEN_ASMPRINTER_DIEHASH_H
#define LLVM_LIB_CODEGEN_ASMPRINTER_DIEHASH_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/DIE.h"
#include "llvm/Support/MD5.h"

namespace llvm {

class AsmPrinter;
class CompileUnit;

/// \brief An object containing the capability of hashing and adding hash
/// attributes onto a DIE.
class DIEHash {
  // Collection of all attributes used in hashing a particular DIE.
  struct DIEAttrs {
    DIEValue DW_AT_name;
    DIEValue DW_AT_accessibility;
    DIEValue DW_AT_address_class;
    DIEValue DW_AT_allocated;
    DIEValue DW_AT_artificial;
    DIEValue DW_AT_associated;
    DIEValue DW_AT_binary_scale;
    DIEValue DW_AT_bit_offset;
    DIEValue DW_AT_bit_size;
    DIEValue DW_AT_bit_stride;
    DIEValue DW_AT_byte_size;
    DIEValue DW_AT_byte_stride;
    DIEValue DW_AT_const_expr;
    DIEValue DW_AT_const_value;
    DIEValue DW_AT_containing_type;
    DIEValue DW_AT_count;
    DIEValue DW_AT_data_bit_offset;
    DIEValue DW_AT_data_location;
    DIEValue DW_AT_data_member_location;
    DIEValue DW_AT_decimal_scale;
    DIEValue DW_AT_decimal_sign;
    DIEValue DW_AT_default_value;
    DIEValue DW_AT_digit_count;
    DIEValue DW_AT_discr;
    DIEValue DW_AT_discr_list;
    DIEValue DW_AT_discr_value;
    DIEValue DW_AT_encoding;
    DIEValue DW_AT_enum_class;
    DIEValue DW_AT_endianity;
    DIEValue DW_AT_explicit;
    DIEValue DW_AT_is_optional;
    DIEValue DW_AT_location;
    DIEValue DW_AT_lower_bound;
    DIEValue DW_AT_mutable;
    DIEValue DW_AT_ordering;
    DIEValue DW_AT_picture_string;
    DIEValue DW_AT_prototyped;
    DIEValue DW_AT_small;
    DIEValue DW_AT_segment;
    DIEValue DW_AT_string_length;
    DIEValue DW_AT_threads_scaled;
    DIEValue DW_AT_upper_bound;
    DIEValue DW_AT_use_location;
    DIEValue DW_AT_use_UTF8;
    DIEValue DW_AT_variable_parameter;
    DIEValue DW_AT_virtuality;
    DIEValue DW_AT_visibility;
    DIEValue DW_AT_vtable_elem_location;
    DIEValue DW_AT_type;

    // Insert any additional ones here...
  };

public:
  DIEHash(AsmPrinter *A = nullptr) : AP(A) {}

  /// \brief Computes the ODR signature.
  uint64_t computeDIEODRSignature(const DIE &Die);

  /// \brief Computes the CU signature.
  uint64_t computeCUSignature(const DIE &Die);

  /// \brief Computes the type signature.
  uint64_t computeTypeSignature(const DIE &Die);

  // Helper routines to process parts of a DIE.
private:
  /// \brief Adds the parent context of \param Die to the hash.
  void addParentContext(const DIE &Die);

  /// \brief Adds the attributes of \param Die to the hash.
  void addAttributes(const DIE &Die);

  /// \brief Computes the full DWARF4 7.27 hash of the DIE.
  void computeHash(const DIE &Die);

  // Routines that add DIEValues to the hash.
public:
  /// \brief Adds \param Value to the hash.
  void update(uint8_t Value) { Hash.update(Value); }

  /// \brief Encodes and adds \param Value to the hash as a ULEB128.
  void addULEB128(uint64_t Value);

  /// \brief Encodes and adds \param Value to the hash as a SLEB128.
  void addSLEB128(int64_t Value);

private:
  /// \brief Adds \param Str to the hash and includes a NULL byte.
  void addString(StringRef Str);

  /// \brief Collects the attributes of DIE \param Die into the \param Attrs
  /// structure.
  void collectAttributes(const DIE &Die, DIEAttrs &Attrs);

  /// \brief Hashes the attributes in \param Attrs in order.
  void hashAttributes(const DIEAttrs &Attrs, dwarf::Tag Tag);

  /// \brief Hashes the data in a block like DIEValue, e.g. DW_FORM_block or
  /// DW_FORM_exprloc.
  void hashBlockData(const DIE::const_value_range &Values);

  /// \brief Hashes the contents pointed to in the .debug_loc section.
  void hashLocList(const DIELocList &LocList);

  /// \brief Hashes an individual attribute.
  void hashAttribute(DIEValue Value, dwarf::Tag Tag);

  /// \brief Hashes an attribute that refers to another DIE.
  void hashDIEEntry(dwarf::Attribute Attribute, dwarf::Tag Tag,
                    const DIE &Entry);

  /// \brief Hashes a reference to a named type in such a way that is
  /// independent of whether that type is described by a declaration or a
  /// definition.
  void hashShallowTypeReference(dwarf::Attribute Attribute, const DIE &Entry,
                                StringRef Name);

  /// \brief Hashes a reference to a previously referenced type DIE.
  void hashRepeatedTypeReference(dwarf::Attribute Attribute,
                                 unsigned DieNumber);

  void hashNestedType(const DIE &Die, StringRef Name);

private:
  MD5 Hash;
  AsmPrinter *AP;
  DenseMap<const DIE *, unsigned> Numbering;
};
}

#endif
