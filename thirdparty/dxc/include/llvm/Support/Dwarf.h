//===-- llvm/Support/Dwarf.h ---Dwarf Constants------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// \file
// \brief This file contains constants used for implementing Dwarf
// debug support.
//
// For details on the Dwarf specfication see the latest DWARF Debugging
// Information Format standard document on http://www.dwarfstd.org. This
// file often includes support for non-released standard features.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_SUPPORT_DWARF_H
#define LLVM_SUPPORT_DWARF_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {

namespace dwarf {
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
// Dwarf constants as gleaned from the DWARF Debugging Information Format V.4
// reference manual http://www.dwarfstd.org/.
//

// Do not mix the following two enumerations sets.  DW_TAG_invalid changes the
// enumeration base type.

enum LLVMConstants : uint32_t {
  // LLVM mock tags (see also llvm/Support/Dwarf.def).
  DW_TAG_invalid = ~0U,        // Tag for invalid results.
  DW_VIRTUALITY_invalid = ~0U, // Virtuality for invalid results.

  // Other constants.
  DWARF_VERSION = 4,       // Default dwarf version we output.
  DW_PUBTYPES_VERSION = 2, // Section version number for .debug_pubtypes.
  DW_PUBNAMES_VERSION = 2, // Section version number for .debug_pubnames.
  DW_ARANGES_VERSION = 2   // Section version number for .debug_aranges.
};

// Special ID values that distinguish a CIE from a FDE in DWARF CFI.
// Not inside an enum because a 64-bit value is needed.
const uint32_t DW_CIE_ID = UINT32_MAX;
const uint64_t DW64_CIE_ID = UINT64_MAX;

enum Tag : uint16_t {
#define HANDLE_DW_TAG(ID, NAME) DW_TAG_##NAME = ID,
#include "llvm/Support/Dwarf.def"
  DW_TAG_lo_user = 0x4080,
  DW_TAG_hi_user = 0xffff,
  DW_TAG_user_base = 0x1000 // Recommended base for user tags.
};

inline bool isType(Tag T) {
  switch (T) {
  case DW_TAG_array_type:
  case DW_TAG_class_type:
  case DW_TAG_interface_type:
  case DW_TAG_enumeration_type:
  case DW_TAG_pointer_type:
  case DW_TAG_reference_type:
  case DW_TAG_rvalue_reference_type:
  case DW_TAG_string_type:
  case DW_TAG_structure_type:
  case DW_TAG_subroutine_type:
  case DW_TAG_union_type:
  case DW_TAG_ptr_to_member_type:
  case DW_TAG_set_type:
  case DW_TAG_subrange_type:
  case DW_TAG_base_type:
  case DW_TAG_const_type:
  case DW_TAG_file_type:
  case DW_TAG_packed_type:
  case DW_TAG_volatile_type:
  case DW_TAG_typedef:
    return true;
  default:
    return false;
  }
}

enum Attribute : uint16_t {
  // Attributes
  DW_AT_sibling = 0x01,
  DW_AT_location = 0x02,
  DW_AT_name = 0x03,
  DW_AT_ordering = 0x09,
  DW_AT_byte_size = 0x0b,
  DW_AT_bit_offset = 0x0c,
  DW_AT_bit_size = 0x0d,
  DW_AT_stmt_list = 0x10,
  DW_AT_low_pc = 0x11,
  DW_AT_high_pc = 0x12,
  DW_AT_language = 0x13,
  DW_AT_discr = 0x15,
  DW_AT_discr_value = 0x16,
  DW_AT_visibility = 0x17,
  DW_AT_import = 0x18,
  DW_AT_string_length = 0x19,
  DW_AT_common_reference = 0x1a,
  DW_AT_comp_dir = 0x1b,
  DW_AT_const_value = 0x1c,
  DW_AT_containing_type = 0x1d,
  DW_AT_default_value = 0x1e,
  DW_AT_inline = 0x20,
  DW_AT_is_optional = 0x21,
  DW_AT_lower_bound = 0x22,
  DW_AT_producer = 0x25,
  DW_AT_prototyped = 0x27,
  DW_AT_return_addr = 0x2a,
  DW_AT_start_scope = 0x2c,
  DW_AT_bit_stride = 0x2e,
  DW_AT_upper_bound = 0x2f,
  DW_AT_abstract_origin = 0x31,
  DW_AT_accessibility = 0x32,
  DW_AT_address_class = 0x33,
  DW_AT_artificial = 0x34,
  DW_AT_base_types = 0x35,
  DW_AT_calling_convention = 0x36,
  DW_AT_count = 0x37,
  DW_AT_data_member_location = 0x38,
  DW_AT_decl_column = 0x39,
  DW_AT_decl_file = 0x3a,
  DW_AT_decl_line = 0x3b,
  DW_AT_declaration = 0x3c,
  DW_AT_discr_list = 0x3d,
  DW_AT_encoding = 0x3e,
  DW_AT_external = 0x3f,
  DW_AT_frame_base = 0x40,
  DW_AT_friend = 0x41,
  DW_AT_identifier_case = 0x42,
  DW_AT_macro_info = 0x43,
  DW_AT_namelist_item = 0x44,
  DW_AT_priority = 0x45,
  DW_AT_segment = 0x46,
  DW_AT_specification = 0x47,
  DW_AT_static_link = 0x48,
  DW_AT_type = 0x49,
  DW_AT_use_location = 0x4a,
  DW_AT_variable_parameter = 0x4b,
  DW_AT_virtuality = 0x4c,
  DW_AT_vtable_elem_location = 0x4d,
  DW_AT_allocated = 0x4e,
  DW_AT_associated = 0x4f,
  DW_AT_data_location = 0x50,
  DW_AT_byte_stride = 0x51,
  DW_AT_entry_pc = 0x52,
  DW_AT_use_UTF8 = 0x53,
  DW_AT_extension = 0x54,
  DW_AT_ranges = 0x55,
  DW_AT_trampoline = 0x56,
  DW_AT_call_column = 0x57,
  DW_AT_call_file = 0x58,
  DW_AT_call_line = 0x59,
  DW_AT_description = 0x5a,
  DW_AT_binary_scale = 0x5b,
  DW_AT_decimal_scale = 0x5c,
  DW_AT_small = 0x5d,
  DW_AT_decimal_sign = 0x5e,
  DW_AT_digit_count = 0x5f,
  DW_AT_picture_string = 0x60,
  DW_AT_mutable = 0x61,
  DW_AT_threads_scaled = 0x62,
  DW_AT_explicit = 0x63,
  DW_AT_object_pointer = 0x64,
  DW_AT_endianity = 0x65,
  DW_AT_elemental = 0x66,
  DW_AT_pure = 0x67,
  DW_AT_recursive = 0x68,
  DW_AT_signature = 0x69,
  DW_AT_main_subprogram = 0x6a,
  DW_AT_data_bit_offset = 0x6b,
  DW_AT_const_expr = 0x6c,
  DW_AT_enum_class = 0x6d,
  DW_AT_linkage_name = 0x6e,

  // New in DWARF 5:
  DW_AT_string_length_bit_size = 0x6f,
  DW_AT_string_length_byte_size = 0x70,
  DW_AT_rank = 0x71,
  DW_AT_str_offsets_base = 0x72,
  DW_AT_addr_base = 0x73,
  DW_AT_ranges_base = 0x74,
  DW_AT_dwo_id = 0x75,
  DW_AT_dwo_name = 0x76,
  DW_AT_reference = 0x77,
  DW_AT_rvalue_reference = 0x78,

  DW_AT_lo_user = 0x2000,
  DW_AT_hi_user = 0x3fff,

  DW_AT_MIPS_loop_begin = 0x2002,
  DW_AT_MIPS_tail_loop_begin = 0x2003,
  DW_AT_MIPS_epilog_begin = 0x2004,
  DW_AT_MIPS_loop_unroll_factor = 0x2005,
  DW_AT_MIPS_software_pipeline_depth = 0x2006,
  DW_AT_MIPS_linkage_name = 0x2007,
  DW_AT_MIPS_stride = 0x2008,
  DW_AT_MIPS_abstract_name = 0x2009,
  DW_AT_MIPS_clone_origin = 0x200a,
  DW_AT_MIPS_has_inlines = 0x200b,
  DW_AT_MIPS_stride_byte = 0x200c,
  DW_AT_MIPS_stride_elem = 0x200d,
  DW_AT_MIPS_ptr_dopetype = 0x200e,
  DW_AT_MIPS_allocatable_dopetype = 0x200f,
  DW_AT_MIPS_assumed_shape_dopetype = 0x2010,

  // This one appears to have only been implemented by Open64 for
  // fortran and may conflict with other extensions.
  DW_AT_MIPS_assumed_size = 0x2011,

  // GNU extensions
  DW_AT_sf_names = 0x2101,
  DW_AT_src_info = 0x2102,
  DW_AT_mac_info = 0x2103,
  DW_AT_src_coords = 0x2104,
  DW_AT_body_begin = 0x2105,
  DW_AT_body_end = 0x2106,
  DW_AT_GNU_vector = 0x2107,
  DW_AT_GNU_template_name = 0x2110,

  DW_AT_GNU_odr_signature = 0x210f,

  // Extensions for Fission proposal.
  DW_AT_GNU_dwo_name = 0x2130,
  DW_AT_GNU_dwo_id = 0x2131,
  DW_AT_GNU_ranges_base = 0x2132,
  DW_AT_GNU_addr_base = 0x2133,
  DW_AT_GNU_pubnames = 0x2134,
  DW_AT_GNU_pubtypes = 0x2135,

  // LLVM project extensions.
  DW_AT_LLVM_include_path = 0x3e00,
  DW_AT_LLVM_config_macros = 0x3e01,
  DW_AT_LLVM_isysroot = 0x3e02,

  // Apple extensions.
  DW_AT_APPLE_optimized = 0x3fe1,
  DW_AT_APPLE_flags = 0x3fe2,
  DW_AT_APPLE_isa = 0x3fe3,
  DW_AT_APPLE_block = 0x3fe4,
  DW_AT_APPLE_major_runtime_vers = 0x3fe5,
  DW_AT_APPLE_runtime_class = 0x3fe6,
  DW_AT_APPLE_omit_frame_ptr = 0x3fe7,
  DW_AT_APPLE_property_name = 0x3fe8,
  DW_AT_APPLE_property_getter = 0x3fe9,
  DW_AT_APPLE_property_setter = 0x3fea,
  DW_AT_APPLE_property_attribute = 0x3feb,
  DW_AT_APPLE_objc_complete_type = 0x3fec,
  DW_AT_APPLE_property = 0x3fed
};

enum Form : uint16_t {
  // Attribute form encodings
  DW_FORM_addr = 0x01,
  DW_FORM_block2 = 0x03,
  DW_FORM_block4 = 0x04,
  DW_FORM_data2 = 0x05,
  DW_FORM_data4 = 0x06,
  DW_FORM_data8 = 0x07,
  DW_FORM_string = 0x08,
  DW_FORM_block = 0x09,
  DW_FORM_block1 = 0x0a,
  DW_FORM_data1 = 0x0b,
  DW_FORM_flag = 0x0c,
  DW_FORM_sdata = 0x0d,
  DW_FORM_strp = 0x0e,
  DW_FORM_udata = 0x0f,
  DW_FORM_ref_addr = 0x10,
  DW_FORM_ref1 = 0x11,
  DW_FORM_ref2 = 0x12,
  DW_FORM_ref4 = 0x13,
  DW_FORM_ref8 = 0x14,
  DW_FORM_ref_udata = 0x15,
  DW_FORM_indirect = 0x16,
  DW_FORM_sec_offset = 0x17,
  DW_FORM_exprloc = 0x18,
  DW_FORM_flag_present = 0x19,
  DW_FORM_ref_sig8 = 0x20,

  // Extensions for Fission proposal
  DW_FORM_GNU_addr_index = 0x1f01,
  DW_FORM_GNU_str_index = 0x1f02,

  // Alternate debug sections proposal (output of "dwz" tool).
  DW_FORM_GNU_ref_alt = 0x1f20,
  DW_FORM_GNU_strp_alt = 0x1f21
};

enum LocationAtom {
#define HANDLE_DW_OP(ID, NAME) DW_OP_##NAME = ID,
#include "llvm/Support/Dwarf.def"
  DW_OP_lo_user = 0xe0,
  DW_OP_hi_user = 0xff
};

enum TypeKind {
#define HANDLE_DW_ATE(ID, NAME) DW_ATE_##NAME = ID,
#include "llvm/Support/Dwarf.def"
  DW_ATE_lo_user = 0x80,
  DW_ATE_hi_user = 0xff
};

enum DecimalSignEncoding {
  // Decimal sign attribute values
  DW_DS_unsigned = 0x01,
  DW_DS_leading_overpunch = 0x02,
  DW_DS_trailing_overpunch = 0x03,
  DW_DS_leading_separate = 0x04,
  DW_DS_trailing_separate = 0x05
};

enum EndianityEncoding {
  // Endianity attribute values
  DW_END_default = 0x00,
  DW_END_big = 0x01,
  DW_END_little = 0x02,
  DW_END_lo_user = 0x40,
  DW_END_hi_user = 0xff
};

enum AccessAttribute {
  // Accessibility codes
  DW_ACCESS_public = 0x01,
  DW_ACCESS_protected = 0x02,
  DW_ACCESS_private = 0x03
};

enum VisibilityAttribute {
  // Visibility codes
  DW_VIS_local = 0x01,
  DW_VIS_exported = 0x02,
  DW_VIS_qualified = 0x03
};

enum VirtualityAttribute {
#define HANDLE_DW_VIRTUALITY(ID, NAME) DW_VIRTUALITY_##NAME = ID,
#include "llvm/Support/Dwarf.def"
  DW_VIRTUALITY_max = 0x02
};

enum SourceLanguage {
#define HANDLE_DW_LANG(ID, NAME) DW_LANG_##NAME = ID,
#include "llvm/Support/Dwarf.def"
  DW_LANG_lo_user = 0x8000,
  DW_LANG_hi_user = 0xffff
};

enum CaseSensitivity {
  // Identifier case codes
  DW_ID_case_sensitive = 0x00,
  DW_ID_up_case = 0x01,
  DW_ID_down_case = 0x02,
  DW_ID_case_insensitive = 0x03
};

enum CallingConvention {
  // Calling convention codes
  DW_CC_normal = 0x01,
  DW_CC_program = 0x02,
  DW_CC_nocall = 0x03,
  DW_CC_lo_user = 0x40,
  DW_CC_hi_user = 0xff
};

enum InlineAttribute {
  // Inline codes
  DW_INL_not_inlined = 0x00,
  DW_INL_inlined = 0x01,
  DW_INL_declared_not_inlined = 0x02,
  DW_INL_declared_inlined = 0x03
};

enum ArrayDimensionOrdering {
  // Array ordering
  DW_ORD_row_major = 0x00,
  DW_ORD_col_major = 0x01
};

enum DiscriminantList {
  // Discriminant descriptor values
  DW_DSC_label = 0x00,
  DW_DSC_range = 0x01
};

enum LineNumberOps {
  // Line Number Standard Opcode Encodings
  DW_LNS_extended_op = 0x00,
  DW_LNS_copy = 0x01,
  DW_LNS_advance_pc = 0x02,
  DW_LNS_advance_line = 0x03,
  DW_LNS_set_file = 0x04,
  DW_LNS_set_column = 0x05,
  DW_LNS_negate_stmt = 0x06,
  DW_LNS_set_basic_block = 0x07,
  DW_LNS_const_add_pc = 0x08,
  DW_LNS_fixed_advance_pc = 0x09,
  DW_LNS_set_prologue_end = 0x0a,
  DW_LNS_set_epilogue_begin = 0x0b,
  DW_LNS_set_isa = 0x0c
};

enum LineNumberExtendedOps {
  // Line Number Extended Opcode Encodings
  DW_LNE_end_sequence = 0x01,
  DW_LNE_set_address = 0x02,
  DW_LNE_define_file = 0x03,
  DW_LNE_set_discriminator = 0x04,
  DW_LNE_lo_user = 0x80,
  DW_LNE_hi_user = 0xff
};

enum MacinfoRecordType {
  // Macinfo Type Encodings
  DW_MACINFO_define = 0x01,
  DW_MACINFO_undef = 0x02,
  DW_MACINFO_start_file = 0x03,
  DW_MACINFO_end_file = 0x04,
  DW_MACINFO_vendor_ext = 0xff
};

enum CallFrameInfo {
  // Call frame instruction encodings
  DW_CFA_extended = 0x00,
  DW_CFA_nop = 0x00,
  DW_CFA_advance_loc = 0x40,
  DW_CFA_offset = 0x80,
  DW_CFA_restore = 0xc0,
  DW_CFA_set_loc = 0x01,
  DW_CFA_advance_loc1 = 0x02,
  DW_CFA_advance_loc2 = 0x03,
  DW_CFA_advance_loc4 = 0x04,
  DW_CFA_offset_extended = 0x05,
  DW_CFA_restore_extended = 0x06,
  DW_CFA_undefined = 0x07,
  DW_CFA_same_value = 0x08,
  DW_CFA_register = 0x09,
  DW_CFA_remember_state = 0x0a,
  DW_CFA_restore_state = 0x0b,
  DW_CFA_def_cfa = 0x0c,
  DW_CFA_def_cfa_register = 0x0d,
  DW_CFA_def_cfa_offset = 0x0e,
  DW_CFA_def_cfa_expression = 0x0f,
  DW_CFA_expression = 0x10,
  DW_CFA_offset_extended_sf = 0x11,
  DW_CFA_def_cfa_sf = 0x12,
  DW_CFA_def_cfa_offset_sf = 0x13,
  DW_CFA_val_offset = 0x14,
  DW_CFA_val_offset_sf = 0x15,
  DW_CFA_val_expression = 0x16,
  DW_CFA_MIPS_advance_loc8 = 0x1d,
  DW_CFA_GNU_window_save = 0x2d,
  DW_CFA_GNU_args_size = 0x2e,
  DW_CFA_lo_user = 0x1c,
  DW_CFA_hi_user = 0x3f
};

enum Constants {
  // Children flag
  DW_CHILDREN_no = 0x00,
  DW_CHILDREN_yes = 0x01,

  DW_EH_PE_absptr = 0x00,
  DW_EH_PE_omit = 0xff,
  DW_EH_PE_uleb128 = 0x01,
  DW_EH_PE_udata2 = 0x02,
  DW_EH_PE_udata4 = 0x03,
  DW_EH_PE_udata8 = 0x04,
  DW_EH_PE_sleb128 = 0x09,
  DW_EH_PE_sdata2 = 0x0A,
  DW_EH_PE_sdata4 = 0x0B,
  DW_EH_PE_sdata8 = 0x0C,
  DW_EH_PE_signed = 0x08,
  DW_EH_PE_pcrel = 0x10,
  DW_EH_PE_textrel = 0x20,
  DW_EH_PE_datarel = 0x30,
  DW_EH_PE_funcrel = 0x40,
  DW_EH_PE_aligned = 0x50,
  DW_EH_PE_indirect = 0x80
};

// Constants for debug_loc.dwo in the DWARF5 Split Debug Info Proposal
enum LocationListEntry : unsigned char {
  DW_LLE_end_of_list_entry,
  DW_LLE_base_address_selection_entry,
  DW_LLE_start_end_entry,
  DW_LLE_start_length_entry,
  DW_LLE_offset_pair_entry
};

/// Contstants for the DW_APPLE_PROPERTY_attributes attribute.
/// Keep this list in sync with clang's DeclSpec.h ObjCPropertyAttributeKind.
enum ApplePropertyAttributes {
  // Apple Objective-C Property Attributes
  DW_APPLE_PROPERTY_readonly = 0x01,
  DW_APPLE_PROPERTY_getter = 0x02,
  DW_APPLE_PROPERTY_assign = 0x04,
  DW_APPLE_PROPERTY_readwrite = 0x08,
  DW_APPLE_PROPERTY_retain = 0x10,
  DW_APPLE_PROPERTY_copy = 0x20,
  DW_APPLE_PROPERTY_nonatomic = 0x40,
  DW_APPLE_PROPERTY_setter = 0x80,
  DW_APPLE_PROPERTY_atomic = 0x100,
  DW_APPLE_PROPERTY_weak =   0x200,
  DW_APPLE_PROPERTY_strong = 0x400,
  DW_APPLE_PROPERTY_unsafe_unretained = 0x800
};

// Constants for the DWARF5 Accelerator Table Proposal
enum AcceleratorTable {
  // Data layout descriptors.
  DW_ATOM_null = 0u,       // Marker as the end of a list of atoms.
  DW_ATOM_die_offset = 1u, // DIE offset in the debug_info section.
  DW_ATOM_cu_offset = 2u, // Offset of the compile unit header that contains the
                          // item in question.
  DW_ATOM_die_tag = 3u,   // A tag entry.
  DW_ATOM_type_flags = 4u, // Set of flags for a type.

  // DW_ATOM_type_flags values.

  // Always set for C++, only set for ObjC if this is the @implementation for a
  // class.
  DW_FLAG_type_implementation = 2u,

  // Hash functions.

  // Daniel J. Bernstein hash.
  DW_hash_function_djb = 0u
};

// Constants for the GNU pubnames/pubtypes extensions supporting gdb index.
enum GDBIndexEntryKind {
  GIEK_NONE,
  GIEK_TYPE,
  GIEK_VARIABLE,
  GIEK_FUNCTION,
  GIEK_OTHER,
  GIEK_UNUSED5,
  GIEK_UNUSED6,
  GIEK_UNUSED7
};

enum GDBIndexEntryLinkage {
  GIEL_EXTERNAL,
  GIEL_STATIC
};

/// \defgroup DwarfConstantsDumping Dwarf constants dumping functions
///
/// All these functions map their argument's value back to the
/// corresponding enumerator name or return nullptr if the value isn't
/// known.
///
/// @{
const char *TagString(unsigned Tag);
const char *ChildrenString(unsigned Children);
const char *AttributeString(unsigned Attribute);
const char *FormEncodingString(unsigned Encoding);
const char *OperationEncodingString(unsigned Encoding);
const char *AttributeEncodingString(unsigned Encoding);
const char *DecimalSignString(unsigned Sign);
const char *EndianityString(unsigned Endian);
const char *AccessibilityString(unsigned Access);
const char *VisibilityString(unsigned Visibility);
const char *VirtualityString(unsigned Virtuality);
const char *LanguageString(unsigned Language);
const char *CaseString(unsigned Case);
const char *ConventionString(unsigned Convention);
const char *InlineCodeString(unsigned Code);
const char *ArrayOrderString(unsigned Order);
const char *DiscriminantString(unsigned Discriminant);
const char *LNStandardString(unsigned Standard);
const char *LNExtendedString(unsigned Encoding);
const char *MacinfoString(unsigned Encoding);
const char *CallFrameString(unsigned Encoding);
const char *ApplePropertyString(unsigned);
const char *AtomTypeString(unsigned Atom);
const char *GDBIndexEntryKindString(GDBIndexEntryKind Kind);
const char *GDBIndexEntryLinkageString(GDBIndexEntryLinkage Linkage);
/// @}

/// \defgroup DwarfConstantsParsing Dwarf constants parsing functions
///
/// These functions map their strings back to the corresponding enumeration
/// value or return 0 if there is none, except for these exceptions:
///
/// \li \a getTag() returns \a DW_TAG_invalid on invalid input.
/// \li \a getVirtuality() returns \a DW_VIRTUALITY_invalid on invalid input.
///
/// @{
unsigned getTag(StringRef TagString);
unsigned getOperationEncoding(StringRef OperationEncodingString);
unsigned getVirtuality(StringRef VirtualityString);
unsigned getLanguage(StringRef LanguageString);
unsigned getAttributeEncoding(StringRef EncodingString);
/// @}

/// \brief Returns the symbolic string representing Val when used as a value
/// for attribute Attr.
const char *AttributeValueString(uint16_t Attr, unsigned Val);

/// \brief Decsribes an entry of the various gnu_pub* debug sections.
/// 
/// The gnu_pub* kind looks like:
///
/// 0-3  reserved
/// 4-6  symbol kind
/// 7    0 == global, 1 == static
///
/// A gdb_index descriptor includes the above kind, shifted 24 bits up with the
/// offset of the cu within the debug_info section stored in those 24 bits.
struct PubIndexEntryDescriptor {
  GDBIndexEntryKind Kind;
  GDBIndexEntryLinkage Linkage;
  PubIndexEntryDescriptor(GDBIndexEntryKind Kind, GDBIndexEntryLinkage Linkage)
      : Kind(Kind), Linkage(Linkage) {}
  /* implicit */ PubIndexEntryDescriptor(GDBIndexEntryKind Kind)
      : Kind(Kind), Linkage(GIEL_EXTERNAL) {}
  explicit PubIndexEntryDescriptor(uint8_t Value)
      : Kind(static_cast<GDBIndexEntryKind>((Value & KIND_MASK) >>
                                            KIND_OFFSET)),
        Linkage(static_cast<GDBIndexEntryLinkage>((Value & LINKAGE_MASK) >>
                                                  LINKAGE_OFFSET)) {}
  uint8_t toBits() { return Kind << KIND_OFFSET | Linkage << LINKAGE_OFFSET; }

private:
  enum {
    KIND_OFFSET = 4,
    KIND_MASK = 7 << KIND_OFFSET,
    LINKAGE_OFFSET = 7,
    LINKAGE_MASK = 1 << LINKAGE_OFFSET
  };
};


} // End of namespace dwarf

} // End of namespace llvm

#endif
