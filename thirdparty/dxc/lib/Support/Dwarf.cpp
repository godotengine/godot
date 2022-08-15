//===-- llvm/Support/Dwarf.cpp - Dwarf Framework ----------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for generic dwarf information.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Dwarf.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"

using namespace llvm;
using namespace dwarf;

const char *llvm::dwarf::TagString(unsigned Tag) {
  switch (Tag) {
  default: return nullptr;
#define HANDLE_DW_TAG(ID, NAME)                                                \
  case DW_TAG_##NAME:                                                          \
    return "DW_TAG_" #NAME;
#include "llvm/Support/Dwarf.def"
  }
}

unsigned llvm::dwarf::getTag(StringRef TagString) {
  return StringSwitch<unsigned>(TagString)
#define HANDLE_DW_TAG(ID, NAME) .Case("DW_TAG_" #NAME, DW_TAG_##NAME)
#include "llvm/Support/Dwarf.def"
      .Default(DW_TAG_invalid);
}

const char *llvm::dwarf::ChildrenString(unsigned Children) {
  switch (Children) {
  case DW_CHILDREN_no:                   return "DW_CHILDREN_no";
  case DW_CHILDREN_yes:                  return "DW_CHILDREN_yes";
  }
  return nullptr;
}

const char *llvm::dwarf::AttributeString(unsigned Attribute) {
  switch (Attribute) {
  case DW_AT_sibling:                    return "DW_AT_sibling";
  case DW_AT_location:                   return "DW_AT_location";
  case DW_AT_name:                       return "DW_AT_name";
  case DW_AT_ordering:                   return "DW_AT_ordering";
  case DW_AT_byte_size:                  return "DW_AT_byte_size";
  case DW_AT_bit_offset:                 return "DW_AT_bit_offset";
  case DW_AT_bit_size:                   return "DW_AT_bit_size";
  case DW_AT_stmt_list:                  return "DW_AT_stmt_list";
  case DW_AT_low_pc:                     return "DW_AT_low_pc";
  case DW_AT_high_pc:                    return "DW_AT_high_pc";
  case DW_AT_language:                   return "DW_AT_language";
  case DW_AT_discr:                      return "DW_AT_discr";
  case DW_AT_discr_value:                return "DW_AT_discr_value";
  case DW_AT_visibility:                 return "DW_AT_visibility";
  case DW_AT_import:                     return "DW_AT_import";
  case DW_AT_string_length:              return "DW_AT_string_length";
  case DW_AT_common_reference:           return "DW_AT_common_reference";
  case DW_AT_comp_dir:                   return "DW_AT_comp_dir";
  case DW_AT_const_value:                return "DW_AT_const_value";
  case DW_AT_containing_type:            return "DW_AT_containing_type";
  case DW_AT_default_value:              return "DW_AT_default_value";
  case DW_AT_inline:                     return "DW_AT_inline";
  case DW_AT_is_optional:                return "DW_AT_is_optional";
  case DW_AT_lower_bound:                return "DW_AT_lower_bound";
  case DW_AT_producer:                   return "DW_AT_producer";
  case DW_AT_prototyped:                 return "DW_AT_prototyped";
  case DW_AT_return_addr:                return "DW_AT_return_addr";
  case DW_AT_start_scope:                return "DW_AT_start_scope";
  case DW_AT_bit_stride:                 return "DW_AT_bit_stride";
  case DW_AT_upper_bound:                return "DW_AT_upper_bound";
  case DW_AT_abstract_origin:            return "DW_AT_abstract_origin";
  case DW_AT_accessibility:              return "DW_AT_accessibility";
  case DW_AT_address_class:              return "DW_AT_address_class";
  case DW_AT_artificial:                 return "DW_AT_artificial";
  case DW_AT_base_types:                 return "DW_AT_base_types";
  case DW_AT_calling_convention:         return "DW_AT_calling_convention";
  case DW_AT_count:                      return "DW_AT_count";
  case DW_AT_data_member_location:       return "DW_AT_data_member_location";
  case DW_AT_decl_column:                return "DW_AT_decl_column";
  case DW_AT_decl_file:                  return "DW_AT_decl_file";
  case DW_AT_decl_line:                  return "DW_AT_decl_line";
  case DW_AT_declaration:                return "DW_AT_declaration";
  case DW_AT_discr_list:                 return "DW_AT_discr_list";
  case DW_AT_encoding:                   return "DW_AT_encoding";
  case DW_AT_external:                   return "DW_AT_external";
  case DW_AT_frame_base:                 return "DW_AT_frame_base";
  case DW_AT_friend:                     return "DW_AT_friend";
  case DW_AT_identifier_case:            return "DW_AT_identifier_case";
  case DW_AT_macro_info:                 return "DW_AT_macro_info";
  case DW_AT_namelist_item:              return "DW_AT_namelist_item";
  case DW_AT_priority:                   return "DW_AT_priority";
  case DW_AT_segment:                    return "DW_AT_segment";
  case DW_AT_specification:              return "DW_AT_specification";
  case DW_AT_static_link:                return "DW_AT_static_link";
  case DW_AT_type:                       return "DW_AT_type";
  case DW_AT_use_location:               return "DW_AT_use_location";
  case DW_AT_variable_parameter:         return "DW_AT_variable_parameter";
  case DW_AT_virtuality:                 return "DW_AT_virtuality";
  case DW_AT_vtable_elem_location:       return "DW_AT_vtable_elem_location";
  case DW_AT_allocated:                  return "DW_AT_allocated";
  case DW_AT_associated:                 return "DW_AT_associated";
  case DW_AT_data_location:              return "DW_AT_data_location";
  case DW_AT_byte_stride:                return "DW_AT_byte_stride";
  case DW_AT_entry_pc:                   return "DW_AT_entry_pc";
  case DW_AT_use_UTF8:                   return "DW_AT_use_UTF8";
  case DW_AT_extension:                  return "DW_AT_extension";
  case DW_AT_ranges:                     return "DW_AT_ranges";
  case DW_AT_trampoline:                 return "DW_AT_trampoline";
  case DW_AT_call_column:                return "DW_AT_call_column";
  case DW_AT_call_file:                  return "DW_AT_call_file";
  case DW_AT_call_line:                  return "DW_AT_call_line";
  case DW_AT_description:                return "DW_AT_description";
  case DW_AT_binary_scale:               return "DW_AT_binary_scale";
  case DW_AT_decimal_scale:              return "DW_AT_decimal_scale";
  case DW_AT_small:                      return "DW_AT_small";
  case DW_AT_decimal_sign:               return "DW_AT_decimal_sign";
  case DW_AT_digit_count:                return "DW_AT_digit_count";
  case DW_AT_picture_string:             return "DW_AT_picture_string";
  case DW_AT_mutable:                    return "DW_AT_mutable";
  case DW_AT_threads_scaled:             return "DW_AT_threads_scaled";
  case DW_AT_explicit:                   return "DW_AT_explicit";
  case DW_AT_object_pointer:             return "DW_AT_object_pointer";
  case DW_AT_endianity:                  return "DW_AT_endianity";
  case DW_AT_elemental:                  return "DW_AT_elemental";
  case DW_AT_pure:                       return "DW_AT_pure";
  case DW_AT_recursive:                  return "DW_AT_recursive";
  case DW_AT_signature:                  return "DW_AT_signature";
  case DW_AT_main_subprogram:            return "DW_AT_main_subprogram";
  case DW_AT_data_bit_offset:            return "DW_AT_data_bit_offset";
  case DW_AT_const_expr:                 return "DW_AT_const_expr";
  case DW_AT_enum_class:                 return "DW_AT_enum_class";
  case DW_AT_linkage_name:               return "DW_AT_linkage_name";
  case DW_AT_string_length_bit_size:     return "DW_AT_string_length_bit_size";
  case DW_AT_string_length_byte_size:    return "DW_AT_string_length_byte_size";
  case DW_AT_rank:                       return "DW_AT_rank";
  case DW_AT_str_offsets_base:           return "DW_AT_str_offsets_base";
  case DW_AT_addr_base:                  return "DW_AT_addr_base";
  case DW_AT_ranges_base:                return "DW_AT_ranges_base";
  case DW_AT_dwo_id:                     return "DW_AT_dwo_id";
  case DW_AT_dwo_name:                   return "DW_AT_dwo_name";
  case DW_AT_reference:                  return "DW_AT_reference";
  case DW_AT_rvalue_reference:           return "DW_AT_rvalue_reference";
  case DW_AT_MIPS_loop_begin:            return "DW_AT_MIPS_loop_begin";
  case DW_AT_MIPS_tail_loop_begin:       return "DW_AT_MIPS_tail_loop_begin";
  case DW_AT_MIPS_epilog_begin:          return "DW_AT_MIPS_epilog_begin";
  case DW_AT_MIPS_loop_unroll_factor:    return "DW_AT_MIPS_loop_unroll_factor";
  case DW_AT_MIPS_software_pipeline_depth:
    return "DW_AT_MIPS_software_pipeline_depth";
  case DW_AT_MIPS_linkage_name:          return "DW_AT_MIPS_linkage_name";
  case DW_AT_MIPS_stride:                return "DW_AT_MIPS_stride";
  case DW_AT_MIPS_abstract_name:         return "DW_AT_MIPS_abstract_name";
  case DW_AT_MIPS_clone_origin:          return "DW_AT_MIPS_clone_origin";
  case DW_AT_MIPS_has_inlines:           return "DW_AT_MIPS_has_inlines";
  case DW_AT_MIPS_stride_byte:           return "DW_AT_MIPS_stride_byte";
  case DW_AT_MIPS_stride_elem:           return "DW_AT_MIPS_stride_elem";
  case DW_AT_MIPS_ptr_dopetype:          return "DW_AT_MIPS_ptr_dopetype";
  case DW_AT_MIPS_allocatable_dopetype:
    return "DW_AT_MIPS_allocatable_dopetype";
  case DW_AT_MIPS_assumed_shape_dopetype:
    return "DW_AT_MIPS_assumed_shape_dopetype";
  case DW_AT_sf_names:                   return "DW_AT_sf_names";
  case DW_AT_src_info:                   return "DW_AT_src_info";
  case DW_AT_mac_info:                   return "DW_AT_mac_info";
  case DW_AT_src_coords:                 return "DW_AT_src_coords";
  case DW_AT_body_begin:                 return "DW_AT_body_begin";
  case DW_AT_body_end:                   return "DW_AT_body_end";
  case DW_AT_GNU_vector:                 return "DW_AT_GNU_vector";
  case DW_AT_GNU_template_name:          return "DW_AT_GNU_template_name";
  case DW_AT_GNU_odr_signature:          return "DW_AT_GNU_odr_signature";
  case DW_AT_MIPS_assumed_size:          return "DW_AT_MIPS_assumed_size";
  case DW_AT_lo_user:                    return "DW_AT_lo_user";
  case DW_AT_hi_user:                    return "DW_AT_hi_user";
  case DW_AT_APPLE_optimized:            return "DW_AT_APPLE_optimized";
  case DW_AT_APPLE_flags:                return "DW_AT_APPLE_flags";
  case DW_AT_APPLE_isa:                  return "DW_AT_APPLE_isa";
  case DW_AT_APPLE_block:                return "DW_AT_APPLE_block";
  case DW_AT_APPLE_major_runtime_vers:   return "DW_AT_APPLE_major_runtime_vers";
  case DW_AT_APPLE_runtime_class:        return "DW_AT_APPLE_runtime_class";
  case DW_AT_APPLE_omit_frame_ptr:       return "DW_AT_APPLE_omit_frame_ptr";
  case DW_AT_APPLE_property_name:        return "DW_AT_APPLE_property_name";
  case DW_AT_APPLE_property_getter:      return "DW_AT_APPLE_property_getter";
  case DW_AT_APPLE_property_setter:      return "DW_AT_APPLE_property_setter";
  case DW_AT_APPLE_property_attribute:   return "DW_AT_APPLE_property_attribute";
  case DW_AT_APPLE_property:             return "DW_AT_APPLE_property";
  case DW_AT_APPLE_objc_complete_type:   return "DW_AT_APPLE_objc_complete_type";
  case DW_AT_LLVM_include_path:          return "DW_AT_LLVM_include_path";
  case DW_AT_LLVM_config_macros:         return "DW_AT_LLVM_config_macros";
  case DW_AT_LLVM_isysroot:              return "DW_AT_LLVM_isysroot";

    // DWARF5 Fission Extension Attribute
  case DW_AT_GNU_dwo_name:               return "DW_AT_GNU_dwo_name";
  case DW_AT_GNU_dwo_id:                 return "DW_AT_GNU_dwo_id";
  case DW_AT_GNU_ranges_base:            return "DW_AT_GNU_ranges_base";
  case DW_AT_GNU_addr_base:              return "DW_AT_GNU_addr_base";
  case DW_AT_GNU_pubnames:               return "DW_AT_GNU_pubnames";
  case DW_AT_GNU_pubtypes:               return "DW_AT_GNU_pubtypes";
  }
  return nullptr;
}

const char *llvm::dwarf::FormEncodingString(unsigned Encoding) {
  switch (Encoding) {
  case DW_FORM_addr:                     return "DW_FORM_addr";
  case DW_FORM_block2:                   return "DW_FORM_block2";
  case DW_FORM_block4:                   return "DW_FORM_block4";
  case DW_FORM_data2:                    return "DW_FORM_data2";
  case DW_FORM_data4:                    return "DW_FORM_data4";
  case DW_FORM_data8:                    return "DW_FORM_data8";
  case DW_FORM_string:                   return "DW_FORM_string";
  case DW_FORM_block:                    return "DW_FORM_block";
  case DW_FORM_block1:                   return "DW_FORM_block1";
  case DW_FORM_data1:                    return "DW_FORM_data1";
  case DW_FORM_flag:                     return "DW_FORM_flag";
  case DW_FORM_sdata:                    return "DW_FORM_sdata";
  case DW_FORM_strp:                     return "DW_FORM_strp";
  case DW_FORM_udata:                    return "DW_FORM_udata";
  case DW_FORM_ref_addr:                 return "DW_FORM_ref_addr";
  case DW_FORM_ref1:                     return "DW_FORM_ref1";
  case DW_FORM_ref2:                     return "DW_FORM_ref2";
  case DW_FORM_ref4:                     return "DW_FORM_ref4";
  case DW_FORM_ref8:                     return "DW_FORM_ref8";
  case DW_FORM_ref_udata:                return "DW_FORM_ref_udata";
  case DW_FORM_indirect:                 return "DW_FORM_indirect";
  case DW_FORM_sec_offset:               return "DW_FORM_sec_offset";
  case DW_FORM_exprloc:                  return "DW_FORM_exprloc";
  case DW_FORM_flag_present:             return "DW_FORM_flag_present";
  case DW_FORM_ref_sig8:                 return "DW_FORM_ref_sig8";

    // DWARF5 Fission Extension Forms
  case DW_FORM_GNU_addr_index:           return "DW_FORM_GNU_addr_index";
  case DW_FORM_GNU_str_index:            return "DW_FORM_GNU_str_index";

  // Alternate debug sections proposal (output of "dwz" tool).
  case DW_FORM_GNU_ref_alt:              return "DW_FORM_GNU_ref_alt";
  case DW_FORM_GNU_strp_alt:             return "DW_FORM_GNU_strp_alt";
  }
  return nullptr;
}

const char *llvm::dwarf::OperationEncodingString(unsigned Encoding) {
  switch (Encoding) {
  default: return nullptr;
#define HANDLE_DW_OP(ID, NAME)                                                 \
  case DW_OP_##NAME:                                                           \
    return "DW_OP_" #NAME;
#include "llvm/Support/Dwarf.def"
  }
}

unsigned llvm::dwarf::getOperationEncoding(StringRef OperationEncodingString) {
  return StringSwitch<unsigned>(OperationEncodingString)
#define HANDLE_DW_OP(ID, NAME) .Case("DW_OP_" #NAME, DW_OP_##NAME)
#include "llvm/Support/Dwarf.def"
      .Default(0);
}

const char *llvm::dwarf::AttributeEncodingString(unsigned Encoding) {
  switch (Encoding) {
  default: return nullptr;
#define HANDLE_DW_ATE(ID, NAME)                                                \
  case DW_ATE_##NAME:                                                          \
    return "DW_ATE_" #NAME;
#include "llvm/Support/Dwarf.def"
  }
}

unsigned llvm::dwarf::getAttributeEncoding(StringRef EncodingString) {
  return StringSwitch<unsigned>(EncodingString)
#define HANDLE_DW_ATE(ID, NAME) .Case("DW_ATE_" #NAME, DW_ATE_##NAME)
#include "llvm/Support/Dwarf.def"
      .Default(0);
}

const char *llvm::dwarf::DecimalSignString(unsigned Sign) {
  switch (Sign) {
  case DW_DS_unsigned:                   return "DW_DS_unsigned";
  case DW_DS_leading_overpunch:          return "DW_DS_leading_overpunch";
  case DW_DS_trailing_overpunch:         return "DW_DS_trailing_overpunch";
  case DW_DS_leading_separate:           return "DW_DS_leading_separate";
  case DW_DS_trailing_separate:          return "DW_DS_trailing_separate";
  }
  return nullptr;
}

const char *llvm::dwarf::EndianityString(unsigned Endian) {
  switch (Endian) {
  case DW_END_default:                   return "DW_END_default";
  case DW_END_big:                       return "DW_END_big";
  case DW_END_little:                    return "DW_END_little";
  case DW_END_lo_user:                   return "DW_END_lo_user";
  case DW_END_hi_user:                   return "DW_END_hi_user";
  }
  return nullptr;
}

const char *llvm::dwarf::AccessibilityString(unsigned Access) {
  switch (Access) {
  // Accessibility codes
  case DW_ACCESS_public:                 return "DW_ACCESS_public";
  case DW_ACCESS_protected:              return "DW_ACCESS_protected";
  case DW_ACCESS_private:                return "DW_ACCESS_private";
  }
  return nullptr;
}

const char *llvm::dwarf::VisibilityString(unsigned Visibility) {
  switch (Visibility) {
  case DW_VIS_local:                     return "DW_VIS_local";
  case DW_VIS_exported:                  return "DW_VIS_exported";
  case DW_VIS_qualified:                 return "DW_VIS_qualified";
  }
  return nullptr;
}

const char *llvm::dwarf::VirtualityString(unsigned Virtuality) {
  switch (Virtuality) {
  default:
    return nullptr;
#define HANDLE_DW_VIRTUALITY(ID, NAME)                                         \
  case DW_VIRTUALITY_##NAME:                                                   \
    return "DW_VIRTUALITY_" #NAME;
#include "llvm/Support/Dwarf.def"
  }
}

unsigned llvm::dwarf::getVirtuality(StringRef VirtualityString) {
  return StringSwitch<unsigned>(VirtualityString)
#define HANDLE_DW_VIRTUALITY(ID, NAME)                                         \
  .Case("DW_VIRTUALITY_" #NAME, DW_VIRTUALITY_##NAME)
#include "llvm/Support/Dwarf.def"
      .Default(DW_VIRTUALITY_invalid);
}

const char *llvm::dwarf::LanguageString(unsigned Language) {
  switch (Language) {
  default:
    return nullptr;
#define HANDLE_DW_LANG(ID, NAME)                                               \
  case DW_LANG_##NAME:                                                         \
    return "DW_LANG_" #NAME;
#include "llvm/Support/Dwarf.def"
  }
}

unsigned llvm::dwarf::getLanguage(StringRef LanguageString) {
  return StringSwitch<unsigned>(LanguageString)
#define HANDLE_DW_LANG(ID, NAME) .Case("DW_LANG_" #NAME, DW_LANG_##NAME)
#include "llvm/Support/Dwarf.def"
      .Default(0);
}

const char *llvm::dwarf::CaseString(unsigned Case) {
  switch (Case) {
  case DW_ID_case_sensitive:             return "DW_ID_case_sensitive";
  case DW_ID_up_case:                    return "DW_ID_up_case";
  case DW_ID_down_case:                  return "DW_ID_down_case";
  case DW_ID_case_insensitive:           return "DW_ID_case_insensitive";
  }
  return nullptr;
}

const char *llvm::dwarf::ConventionString(unsigned Convention) {
   switch (Convention) {
   case DW_CC_normal:                     return "DW_CC_normal";
   case DW_CC_program:                    return "DW_CC_program";
   case DW_CC_nocall:                     return "DW_CC_nocall";
   case DW_CC_lo_user:                    return "DW_CC_lo_user";
   case DW_CC_hi_user:                    return "DW_CC_hi_user";
  }
  return nullptr;
}

const char *llvm::dwarf::InlineCodeString(unsigned Code) {
  switch (Code) {
  case DW_INL_not_inlined:               return "DW_INL_not_inlined";
  case DW_INL_inlined:                   return "DW_INL_inlined";
  case DW_INL_declared_not_inlined:      return "DW_INL_declared_not_inlined";
  case DW_INL_declared_inlined:          return "DW_INL_declared_inlined";
  }
  return nullptr;
}

const char *llvm::dwarf::ArrayOrderString(unsigned Order) {
  switch (Order) {
  case DW_ORD_row_major:                 return "DW_ORD_row_major";
  case DW_ORD_col_major:                 return "DW_ORD_col_major";
  }
  return nullptr;
}

const char *llvm::dwarf::DiscriminantString(unsigned Discriminant) {
  switch (Discriminant) {
  case DW_DSC_label:                     return "DW_DSC_label";
  case DW_DSC_range:                     return "DW_DSC_range";
  }
  return nullptr;
}

const char *llvm::dwarf::LNStandardString(unsigned Standard) {
  switch (Standard) {
  case DW_LNS_copy:                      return "DW_LNS_copy";
  case DW_LNS_advance_pc:                return "DW_LNS_advance_pc";
  case DW_LNS_advance_line:              return "DW_LNS_advance_line";
  case DW_LNS_set_file:                  return "DW_LNS_set_file";
  case DW_LNS_set_column:                return "DW_LNS_set_column";
  case DW_LNS_negate_stmt:               return "DW_LNS_negate_stmt";
  case DW_LNS_set_basic_block:           return "DW_LNS_set_basic_block";
  case DW_LNS_const_add_pc:              return "DW_LNS_const_add_pc";
  case DW_LNS_fixed_advance_pc:          return "DW_LNS_fixed_advance_pc";
  case DW_LNS_set_prologue_end:          return "DW_LNS_set_prologue_end";
  case DW_LNS_set_epilogue_begin:        return "DW_LNS_set_epilogue_begin";
  case DW_LNS_set_isa:                   return "DW_LNS_set_isa";
  }
  return nullptr;
}

const char *llvm::dwarf::LNExtendedString(unsigned Encoding) {
  switch (Encoding) {
  // Line Number Extended Opcode Encodings
  case DW_LNE_end_sequence:              return "DW_LNE_end_sequence";
  case DW_LNE_set_address:               return "DW_LNE_set_address";
  case DW_LNE_define_file:               return "DW_LNE_define_file";
  case DW_LNE_set_discriminator:         return "DW_LNE_set_discriminator";
  case DW_LNE_lo_user:                   return "DW_LNE_lo_user";
  case DW_LNE_hi_user:                   return "DW_LNE_hi_user";
  }
  return nullptr;
}

const char *llvm::dwarf::MacinfoString(unsigned Encoding) {
  switch (Encoding) {
  // Macinfo Type Encodings
  case DW_MACINFO_define:                return "DW_MACINFO_define";
  case DW_MACINFO_undef:                 return "DW_MACINFO_undef";
  case DW_MACINFO_start_file:            return "DW_MACINFO_start_file";
  case DW_MACINFO_end_file:              return "DW_MACINFO_end_file";
  case DW_MACINFO_vendor_ext:            return "DW_MACINFO_vendor_ext";
  }
  return nullptr;
}

const char *llvm::dwarf::CallFrameString(unsigned Encoding) {
  switch (Encoding) {
  case DW_CFA_nop:                       return "DW_CFA_nop";
  case DW_CFA_advance_loc:               return "DW_CFA_advance_loc";
  case DW_CFA_offset:                    return "DW_CFA_offset";
  case DW_CFA_restore:                   return "DW_CFA_restore";
  case DW_CFA_set_loc:                   return "DW_CFA_set_loc";
  case DW_CFA_advance_loc1:              return "DW_CFA_advance_loc1";
  case DW_CFA_advance_loc2:              return "DW_CFA_advance_loc2";
  case DW_CFA_advance_loc4:              return "DW_CFA_advance_loc4";
  case DW_CFA_offset_extended:           return "DW_CFA_offset_extended";
  case DW_CFA_restore_extended:          return "DW_CFA_restore_extended";
  case DW_CFA_undefined:                 return "DW_CFA_undefined";
  case DW_CFA_same_value:                return "DW_CFA_same_value";
  case DW_CFA_register:                  return "DW_CFA_register";
  case DW_CFA_remember_state:            return "DW_CFA_remember_state";
  case DW_CFA_restore_state:             return "DW_CFA_restore_state";
  case DW_CFA_def_cfa:                   return "DW_CFA_def_cfa";
  case DW_CFA_def_cfa_register:          return "DW_CFA_def_cfa_register";
  case DW_CFA_def_cfa_offset:            return "DW_CFA_def_cfa_offset";
  case DW_CFA_def_cfa_expression:        return "DW_CFA_def_cfa_expression";
  case DW_CFA_expression:                return "DW_CFA_expression";
  case DW_CFA_offset_extended_sf:        return "DW_CFA_offset_extended_sf";
  case DW_CFA_def_cfa_sf:                return "DW_CFA_def_cfa_sf";
  case DW_CFA_def_cfa_offset_sf:         return "DW_CFA_def_cfa_offset_sf";
  case DW_CFA_val_offset:                return "DW_CFA_val_offset";
  case DW_CFA_val_offset_sf:             return "DW_CFA_val_offset_sf";
  case DW_CFA_val_expression:            return "DW_CFA_val_expression";
  case DW_CFA_MIPS_advance_loc8:         return "DW_CFA_MIPS_advance_loc8";
  case DW_CFA_GNU_window_save:           return "DW_CFA_GNU_window_save";
  case DW_CFA_GNU_args_size:             return "DW_CFA_GNU_args_size";
  case DW_CFA_lo_user:                   return "DW_CFA_lo_user";
  case DW_CFA_hi_user:                   return "DW_CFA_hi_user";
  }
  return nullptr;
}

const char *llvm::dwarf::ApplePropertyString(unsigned Prop) {
  switch (Prop) {
  case DW_APPLE_PROPERTY_readonly:
    return "DW_APPLE_PROPERTY_readonly";
  case DW_APPLE_PROPERTY_getter:
    return "DW_APPLE_PROPERTY_getter";
  case DW_APPLE_PROPERTY_assign:
    return "DW_APPLE_PROPERTY_assign";
  case DW_APPLE_PROPERTY_readwrite:
    return "DW_APPLE_PROPERTY_readwrite";
  case DW_APPLE_PROPERTY_retain:
    return "DW_APPLE_PROPERTY_retain";
  case DW_APPLE_PROPERTY_copy:
    return "DW_APPLE_PROPERTY_copy";
  case DW_APPLE_PROPERTY_nonatomic:
    return "DW_APPLE_PROPERTY_nonatomic";
  case DW_APPLE_PROPERTY_setter:
    return "DW_APPLE_PROPERTY_setter";
  case DW_APPLE_PROPERTY_atomic:
    return "DW_APPLE_PROPERTY_atomic";
  case DW_APPLE_PROPERTY_weak:
    return "DW_APPLE_PROPERTY_weak";
  case DW_APPLE_PROPERTY_strong:
    return "DW_APPLE_PROPERTY_strong";
  case DW_APPLE_PROPERTY_unsafe_unretained:
    return "DW_APPLE_PROPERTY_unsafe_unretained";
  }
  return nullptr;
}

const char *llvm::dwarf::AtomTypeString(unsigned AT) {
  switch (AT) {
  case dwarf::DW_ATOM_null:
    return "DW_ATOM_null";
  case dwarf::DW_ATOM_die_offset:
    return "DW_ATOM_die_offset";
  case DW_ATOM_cu_offset:
    return "DW_ATOM_cu_offset";
  case DW_ATOM_die_tag:
    return "DW_ATOM_die_tag";
  case DW_ATOM_type_flags:
    return "DW_ATOM_type_flags";
  }
  return nullptr;
}

const char *llvm::dwarf::GDBIndexEntryKindString(GDBIndexEntryKind Kind) {
  switch (Kind) {
  case GIEK_NONE:
    return "NONE";
  case GIEK_TYPE:
    return "TYPE";
  case GIEK_VARIABLE:
    return "VARIABLE";
  case GIEK_FUNCTION:
    return "FUNCTION";
  case GIEK_OTHER:
    return "OTHER";
  case GIEK_UNUSED5:
    return "UNUSED5";
  case GIEK_UNUSED6:
    return "UNUSED6";
  case GIEK_UNUSED7:
    return "UNUSED7";
  }
  llvm_unreachable("Unknown GDBIndexEntryKind value");
}

const char *llvm::dwarf::GDBIndexEntryLinkageString(GDBIndexEntryLinkage Linkage) {
  switch (Linkage) {
  case GIEL_EXTERNAL:
    return "EXTERNAL";
  case GIEL_STATIC:
    return "STATIC";
  }
  llvm_unreachable("Unknown GDBIndexEntryLinkage value");
}

const char *llvm::dwarf::AttributeValueString(uint16_t Attr, unsigned Val) {
  switch (Attr) {
  case DW_AT_accessibility:
    return AccessibilityString(Val);
  case DW_AT_virtuality:
    return VirtualityString(Val);
  case DW_AT_language:
    return LanguageString(Val);
  case DW_AT_encoding:
    return AttributeEncodingString(Val);
  case DW_AT_decimal_sign:
    return DecimalSignString(Val);
  case DW_AT_endianity:
    return EndianityString(Val);
  case DW_AT_visibility:
    return VisibilityString(Val);
  case DW_AT_identifier_case:
    return CaseString(Val);
  case DW_AT_calling_convention:
    return ConventionString(Val);
  case DW_AT_inline:
    return InlineCodeString(Val);
  case DW_AT_ordering:
    return ArrayOrderString(Val);
  case DW_AT_discr_value:
    return DiscriminantString(Val);
  }

  return nullptr;
}
