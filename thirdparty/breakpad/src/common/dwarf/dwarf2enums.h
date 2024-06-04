// -*- mode: c++ -*-

// Copyright (c) 2010 Google Inc. All Rights Reserved.
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
//     * Neither the name of Google Inc. nor the names of its
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

#ifndef COMMON_DWARF_DWARF2ENUMS_H__
#define COMMON_DWARF_DWARF2ENUMS_H__

namespace google_breakpad {

// These enums do not follow the google3 style only because they are
// known universally (specs, other implementations) by the names in
// exactly this capitalization.
// Tag names and codes.
enum DwarfTag {
  DW_TAG_padding = 0x00,
  DW_TAG_array_type = 0x01,
  DW_TAG_class_type = 0x02,
  DW_TAG_entry_point = 0x03,
  DW_TAG_enumeration_type = 0x04,
  DW_TAG_formal_parameter = 0x05,
  DW_TAG_imported_declaration = 0x08,
  DW_TAG_label = 0x0a,
  DW_TAG_lexical_block = 0x0b,
  DW_TAG_member = 0x0d,
  DW_TAG_pointer_type = 0x0f,
  DW_TAG_reference_type = 0x10,
  DW_TAG_compile_unit = 0x11,
  DW_TAG_string_type = 0x12,
  DW_TAG_structure_type = 0x13,
  DW_TAG_subroutine_type = 0x15,
  DW_TAG_typedef = 0x16,
  DW_TAG_union_type = 0x17,
  DW_TAG_unspecified_parameters = 0x18,
  DW_TAG_variant = 0x19,
  DW_TAG_common_block = 0x1a,
  DW_TAG_common_inclusion = 0x1b,
  DW_TAG_inheritance = 0x1c,
  DW_TAG_inlined_subroutine = 0x1d,
  DW_TAG_module = 0x1e,
  DW_TAG_ptr_to_member_type = 0x1f,
  DW_TAG_set_type = 0x20,
  DW_TAG_subrange_type = 0x21,
  DW_TAG_with_stmt = 0x22,
  DW_TAG_access_declaration = 0x23,
  DW_TAG_base_type = 0x24,
  DW_TAG_catch_block = 0x25,
  DW_TAG_const_type = 0x26,
  DW_TAG_constant = 0x27,
  DW_TAG_enumerator = 0x28,
  DW_TAG_file_type = 0x29,
  DW_TAG_friend = 0x2a,
  DW_TAG_namelist = 0x2b,
  DW_TAG_namelist_item = 0x2c,
  DW_TAG_packed_type = 0x2d,
  DW_TAG_subprogram = 0x2e,
  DW_TAG_template_type_param = 0x2f,
  DW_TAG_template_value_param = 0x30,
  DW_TAG_thrown_type = 0x31,
  DW_TAG_try_block = 0x32,
  DW_TAG_variant_part = 0x33,
  DW_TAG_variable = 0x34,
  DW_TAG_volatile_type = 0x35,
  // DWARF 3.
  DW_TAG_dwarf_procedure = 0x36,
  DW_TAG_restrict_type = 0x37,
  DW_TAG_interface_type = 0x38,
  DW_TAG_namespace = 0x39,
  DW_TAG_imported_module = 0x3a,
  DW_TAG_unspecified_type = 0x3b,
  DW_TAG_partial_unit = 0x3c,
  DW_TAG_imported_unit = 0x3d,
  // DWARF 4.
  DW_TAG_type_unit = 0x41,
  // DWARF 5.
  DW_TAG_skeleton_unit = 0x4a,
  // SGI/MIPS Extensions.
  DW_TAG_MIPS_loop = 0x4081,
  // HP extensions.  See:
  // ftp://ftp.hp.com/pub/lang/tools/WDB/wdb-4.0.tar.gz
  DW_TAG_HP_array_descriptor = 0x4090,
  // GNU extensions.
  DW_TAG_format_label = 0x4101,  // For FORTRAN 77 and Fortran 90.
  DW_TAG_function_template = 0x4102,  // For C++.
  DW_TAG_class_template = 0x4103,  // For C++.
  DW_TAG_GNU_BINCL = 0x4104,
  DW_TAG_GNU_EINCL = 0x4105,
  // Extensions for UPC.  See: http://upc.gwu.edu/~upc.
  DW_TAG_upc_shared_type = 0x8765,
  DW_TAG_upc_strict_type = 0x8766,
  DW_TAG_upc_relaxed_type = 0x8767,
  // PGI (STMicroelectronics) extensions.  No documentation available.
  DW_TAG_PGI_kanji_type      = 0xA000,
  DW_TAG_PGI_interface_block = 0xA020
};

enum DwarfUnitHeader {
  DW_UT_compile = 0x01,
  DW_UT_type = 0x02,
  DW_UT_partial = 0x03,
  DW_UT_skeleton = 0x04,
  DW_UT_split_compile = 0x05,
  DW_UT_split_type = 0x06,
  DW_UT_lo_user = 0x80,
  DW_UT_hi_user = 0xFF
};

enum DwarfHasChild {
  DW_children_no = 0,
  DW_children_yes = 1
};

// Form names and codes.
enum DwarfForm {
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

  // Added in DWARF 4:
  DW_FORM_sec_offset = 0x17,
  DW_FORM_exprloc = 0x18,
  DW_FORM_flag_present = 0x19,

  // Added in DWARF 5:
  DW_FORM_strx = 0x1a,
  DW_FORM_addrx = 0x1b,
  DW_FORM_ref_sup4 = 0x1c,
  DW_FORM_strp_sup = 0x1d,
  DW_FORM_data16 = 0x1e,
  DW_FORM_line_strp = 0x1f,

  // DWARF 4, but value out of order.
  DW_FORM_ref_sig8 = 0x20,

  // Added in DWARF 5:
  DW_FORM_implicit_const = 0x21,
  DW_FORM_loclistx = 0x22,
  DW_FORM_rnglistx = 0x23,
  DW_FORM_ref_sup8 = 0x24,
  DW_FORM_strx1 = 0x25,
  DW_FORM_strx2 = 0x26,
  DW_FORM_strx3 = 0x27,
  DW_FORM_strx4 = 0x28,

  DW_FORM_addrx1 = 0x29,
  DW_FORM_addrx2 = 0x2a,
  DW_FORM_addrx3 = 0x2b,
  DW_FORM_addrx4 = 0x2c,

  // Extensions for Fission.  See http://gcc.gnu.org/wiki/DebugFission.
  DW_FORM_GNU_addr_index = 0x1f01,
  DW_FORM_GNU_str_index = 0x1f02
};

// Attribute names and codes
enum DwarfAttribute {
  DW_AT_sibling = 0x01,
  DW_AT_location = 0x02,
  DW_AT_name = 0x03,
  DW_AT_ordering = 0x09,
  DW_AT_subscr_data = 0x0a,
  DW_AT_byte_size = 0x0b,
  DW_AT_bit_offset = 0x0c,
  DW_AT_bit_size = 0x0d,
  DW_AT_element_list = 0x0f,
  DW_AT_stmt_list = 0x10,
  DW_AT_low_pc = 0x11,
  DW_AT_high_pc = 0x12,
  DW_AT_language = 0x13,
  DW_AT_member = 0x14,
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
  DW_AT_stride_size = 0x2e,
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
  DW_AT_namelist_items = 0x44,
  DW_AT_priority = 0x45,
  DW_AT_segment = 0x46,
  DW_AT_specification = 0x47,
  DW_AT_static_link = 0x48,
  DW_AT_type = 0x49,
  DW_AT_use_location = 0x4a,
  DW_AT_variable_parameter = 0x4b,
  DW_AT_virtuality = 0x4c,
  DW_AT_vtable_elem_location = 0x4d,
  // DWARF 3 values.
  DW_AT_allocated     = 0x4e,
  DW_AT_associated    = 0x4f,
  DW_AT_data_location = 0x50,
  DW_AT_stride        = 0x51,
  DW_AT_entry_pc      = 0x52,
  DW_AT_use_UTF8      = 0x53,
  DW_AT_extension     = 0x54,
  DW_AT_ranges        = 0x55,
  DW_AT_trampoline    = 0x56,
  DW_AT_call_column   = 0x57,
  DW_AT_call_file     = 0x58,
  DW_AT_call_line     = 0x59,
  // DWARF 4
  DW_AT_linkage_name  = 0x6e,
  // DWARF 5
  DW_AT_str_offsets_base = 0x72,
  DW_AT_addr_base = 0x73,
  DW_AT_rnglists_base = 0x74,
  DW_AT_dwo_name = 0x76,
  // SGI/MIPS extensions.
  DW_AT_MIPS_fde = 0x2001,
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
  // HP extensions.
  DW_AT_HP_block_index         = 0x2000,
  DW_AT_HP_unmodifiable        = 0x2001,  // Same as DW_AT_MIPS_fde.
  DW_AT_HP_actuals_stmt_list   = 0x2010,
  DW_AT_HP_proc_per_section    = 0x2011,
  DW_AT_HP_raw_data_ptr        = 0x2012,
  DW_AT_HP_pass_by_reference   = 0x2013,
  DW_AT_HP_opt_level           = 0x2014,
  DW_AT_HP_prof_version_id     = 0x2015,
  DW_AT_HP_opt_flags           = 0x2016,
  DW_AT_HP_cold_region_low_pc  = 0x2017,
  DW_AT_HP_cold_region_high_pc = 0x2018,
  DW_AT_HP_all_variables_modifiable = 0x2019,
  DW_AT_HP_linkage_name        = 0x201a,
  DW_AT_HP_prof_flags          = 0x201b,  // In comp unit of procs_info for -g.
  // GNU extensions.
  DW_AT_sf_names   = 0x2101,
  DW_AT_src_info   = 0x2102,
  DW_AT_mac_info   = 0x2103,
  DW_AT_src_coords = 0x2104,
  DW_AT_body_begin = 0x2105,
  DW_AT_body_end   = 0x2106,
  DW_AT_GNU_vector = 0x2107,
  // Extensions for Fission.  See http://gcc.gnu.org/wiki/DebugFission.
  DW_AT_GNU_dwo_name = 0x2130,
  DW_AT_GNU_dwo_id = 0x2131,
  DW_AT_GNU_ranges_base = 0x2132,
  DW_AT_GNU_addr_base = 0x2133,
  DW_AT_GNU_pubnames = 0x2134,
  DW_AT_GNU_pubtypes = 0x2135,
  // VMS extensions.
  DW_AT_VMS_rtnbeg_pd_address = 0x2201,
  // UPC extension.
  DW_AT_upc_threads_scaled = 0x3210,
  // PGI (STMicroelectronics) extensions.
  DW_AT_PGI_lbase    = 0x3a00,
  DW_AT_PGI_soffset  = 0x3a01,
  DW_AT_PGI_lstride  = 0x3a02
};

// .debug_rngslist entry types
enum DwarfRngListEntry {
  DW_RLE_end_of_list = 0,
  DW_RLE_base_addressx = 1,
  DW_RLE_startx_endx = 2,
  DW_RLE_startx_length = 3,
  DW_RLE_offset_pair = 4,
  DW_RLE_base_address = 5,
  DW_RLE_start_end = 6,
  DW_RLE_start_length = 7,
};

// Line number content type codes (DWARF 5).
enum DwarfLineNumberContentType {
  DW_LNCT_path = 1,
  DW_LNCT_directory_index = 2,
  DW_LNCT_timestamp = 3,
  DW_LNCT_size = 4,
  DW_LNCT_MD5 = 5,
};

// Line number opcodes.
enum DwarfLineNumberOps {
  DW_LNS_extended_op = 0,
  DW_LNS_copy = 1,
  DW_LNS_advance_pc = 2,
  DW_LNS_advance_line = 3,
  DW_LNS_set_file = 4,
  DW_LNS_set_column = 5,
  DW_LNS_negate_stmt = 6,
  DW_LNS_set_basic_block = 7,
  DW_LNS_const_add_pc = 8,
  DW_LNS_fixed_advance_pc = 9,
  // DWARF 3.
  DW_LNS_set_prologue_end = 10,
  DW_LNS_set_epilogue_begin = 11,
  DW_LNS_set_isa = 12
};

// Line number extended opcodes.
enum DwarfLineNumberExtendedOps {
  DW_LNE_end_sequence = 1,
  DW_LNE_set_address = 2,
  DW_LNE_define_file = 3,
  // HP extensions.
  DW_LNE_HP_negate_is_UV_update      = 0x11,
  DW_LNE_HP_push_context             = 0x12,
  DW_LNE_HP_pop_context              = 0x13,
  DW_LNE_HP_set_file_line_column     = 0x14,
  DW_LNE_HP_set_routine_name         = 0x15,
  DW_LNE_HP_set_sequence             = 0x16,
  DW_LNE_HP_negate_post_semantics    = 0x17,
  DW_LNE_HP_negate_function_exit     = 0x18,
  DW_LNE_HP_negate_front_end_logical = 0x19,
  DW_LNE_HP_define_proc              = 0x20
};

// Type encoding names and codes
enum DwarfEncoding {
  DW_ATE_address                     =0x1,
  DW_ATE_boolean                     =0x2,
  DW_ATE_complex_float               =0x3,
  DW_ATE_float                       =0x4,
  DW_ATE_signed                      =0x5,
  DW_ATE_signed_char                 =0x6,
  DW_ATE_unsigned                    =0x7,
  DW_ATE_unsigned_char               =0x8,
  // DWARF3/DWARF3f
  DW_ATE_imaginary_float             =0x9,
  DW_ATE_packed_decimal              =0xa,
  DW_ATE_numeric_string              =0xb,
  DW_ATE_edited                      =0xc,
  DW_ATE_signed_fixed                =0xd,
  DW_ATE_unsigned_fixed              =0xe,
  DW_ATE_decimal_float               =0xf,
  DW_ATE_lo_user                     =0x80,
  DW_ATE_hi_user                     =0xff
};

// Location virtual machine opcodes
enum DwarfOpcode {
  DW_OP_addr                         =0x03,
  DW_OP_deref                        =0x06,
  DW_OP_const1u                      =0x08,
  DW_OP_const1s                      =0x09,
  DW_OP_const2u                      =0x0a,
  DW_OP_const2s                      =0x0b,
  DW_OP_const4u                      =0x0c,
  DW_OP_const4s                      =0x0d,
  DW_OP_const8u                      =0x0e,
  DW_OP_const8s                      =0x0f,
  DW_OP_constu                       =0x10,
  DW_OP_consts                       =0x11,
  DW_OP_dup                          =0x12,
  DW_OP_drop                         =0x13,
  DW_OP_over                         =0x14,
  DW_OP_pick                         =0x15,
  DW_OP_swap                         =0x16,
  DW_OP_rot                          =0x17,
  DW_OP_xderef                       =0x18,
  DW_OP_abs                          =0x19,
  DW_OP_and                          =0x1a,
  DW_OP_div                          =0x1b,
  DW_OP_minus                        =0x1c,
  DW_OP_mod                          =0x1d,
  DW_OP_mul                          =0x1e,
  DW_OP_neg                          =0x1f,
  DW_OP_not                          =0x20,
  DW_OP_or                           =0x21,
  DW_OP_plus                         =0x22,
  DW_OP_plus_uconst                  =0x23,
  DW_OP_shl                          =0x24,
  DW_OP_shr                          =0x25,
  DW_OP_shra                         =0x26,
  DW_OP_xor                          =0x27,
  DW_OP_bra                          =0x28,
  DW_OP_eq                           =0x29,
  DW_OP_ge                           =0x2a,
  DW_OP_gt                           =0x2b,
  DW_OP_le                           =0x2c,
  DW_OP_lt                           =0x2d,
  DW_OP_ne                           =0x2e,
  DW_OP_skip                         =0x2f,
  DW_OP_lit0                         =0x30,
  DW_OP_lit1                         =0x31,
  DW_OP_lit2                         =0x32,
  DW_OP_lit3                         =0x33,
  DW_OP_lit4                         =0x34,
  DW_OP_lit5                         =0x35,
  DW_OP_lit6                         =0x36,
  DW_OP_lit7                         =0x37,
  DW_OP_lit8                         =0x38,
  DW_OP_lit9                         =0x39,
  DW_OP_lit10                        =0x3a,
  DW_OP_lit11                        =0x3b,
  DW_OP_lit12                        =0x3c,
  DW_OP_lit13                        =0x3d,
  DW_OP_lit14                        =0x3e,
  DW_OP_lit15                        =0x3f,
  DW_OP_lit16                        =0x40,
  DW_OP_lit17                        =0x41,
  DW_OP_lit18                        =0x42,
  DW_OP_lit19                        =0x43,
  DW_OP_lit20                        =0x44,
  DW_OP_lit21                        =0x45,
  DW_OP_lit22                        =0x46,
  DW_OP_lit23                        =0x47,
  DW_OP_lit24                        =0x48,
  DW_OP_lit25                        =0x49,
  DW_OP_lit26                        =0x4a,
  DW_OP_lit27                        =0x4b,
  DW_OP_lit28                        =0x4c,
  DW_OP_lit29                        =0x4d,
  DW_OP_lit30                        =0x4e,
  DW_OP_lit31                        =0x4f,
  DW_OP_reg0                         =0x50,
  DW_OP_reg1                         =0x51,
  DW_OP_reg2                         =0x52,
  DW_OP_reg3                         =0x53,
  DW_OP_reg4                         =0x54,
  DW_OP_reg5                         =0x55,
  DW_OP_reg6                         =0x56,
  DW_OP_reg7                         =0x57,
  DW_OP_reg8                         =0x58,
  DW_OP_reg9                         =0x59,
  DW_OP_reg10                        =0x5a,
  DW_OP_reg11                        =0x5b,
  DW_OP_reg12                        =0x5c,
  DW_OP_reg13                        =0x5d,
  DW_OP_reg14                        =0x5e,
  DW_OP_reg15                        =0x5f,
  DW_OP_reg16                        =0x60,
  DW_OP_reg17                        =0x61,
  DW_OP_reg18                        =0x62,
  DW_OP_reg19                        =0x63,
  DW_OP_reg20                        =0x64,
  DW_OP_reg21                        =0x65,
  DW_OP_reg22                        =0x66,
  DW_OP_reg23                        =0x67,
  DW_OP_reg24                        =0x68,
  DW_OP_reg25                        =0x69,
  DW_OP_reg26                        =0x6a,
  DW_OP_reg27                        =0x6b,
  DW_OP_reg28                        =0x6c,
  DW_OP_reg29                        =0x6d,
  DW_OP_reg30                        =0x6e,
  DW_OP_reg31                        =0x6f,
  DW_OP_breg0                        =0x70,
  DW_OP_breg1                        =0x71,
  DW_OP_breg2                        =0x72,
  DW_OP_breg3                        =0x73,
  DW_OP_breg4                        =0x74,
  DW_OP_breg5                        =0x75,
  DW_OP_breg6                        =0x76,
  DW_OP_breg7                        =0x77,
  DW_OP_breg8                        =0x78,
  DW_OP_breg9                        =0x79,
  DW_OP_breg10                       =0x7a,
  DW_OP_breg11                       =0x7b,
  DW_OP_breg12                       =0x7c,
  DW_OP_breg13                       =0x7d,
  DW_OP_breg14                       =0x7e,
  DW_OP_breg15                       =0x7f,
  DW_OP_breg16                       =0x80,
  DW_OP_breg17                       =0x81,
  DW_OP_breg18                       =0x82,
  DW_OP_breg19                       =0x83,
  DW_OP_breg20                       =0x84,
  DW_OP_breg21                       =0x85,
  DW_OP_breg22                       =0x86,
  DW_OP_breg23                       =0x87,
  DW_OP_breg24                       =0x88,
  DW_OP_breg25                       =0x89,
  DW_OP_breg26                       =0x8a,
  DW_OP_breg27                       =0x8b,
  DW_OP_breg28                       =0x8c,
  DW_OP_breg29                       =0x8d,
  DW_OP_breg30                       =0x8e,
  DW_OP_breg31                       =0x8f,
  DW_OP_regX                         =0x90,
  DW_OP_fbreg                        =0x91,
  DW_OP_bregX                        =0x92,
  DW_OP_piece                        =0x93,
  DW_OP_deref_size                   =0x94,
  DW_OP_xderef_size                  =0x95,
  DW_OP_nop                          =0x96,
  // DWARF3/DWARF3f
  DW_OP_push_object_address          =0x97,
  DW_OP_call2                        =0x98,
  DW_OP_call4                        =0x99,
  DW_OP_call_ref                     =0x9a,
  DW_OP_form_tls_address             =0x9b,
  DW_OP_call_frame_cfa               =0x9c,
  DW_OP_bit_piece                    =0x9d,
  DW_OP_lo_user                      =0xe0,
  DW_OP_hi_user                      =0xff,
  // GNU extensions
  DW_OP_GNU_push_tls_address         =0xe0,
  // Extensions for Fission.  See http://gcc.gnu.org/wiki/DebugFission.
  DW_OP_GNU_addr_index               =0xfb,
  DW_OP_GNU_const_index              =0xfc
};

// Section identifiers for DWP files
enum DwarfSectionId {
  DW_SECT_INFO = 1,
  DW_SECT_TYPES = 2,
  DW_SECT_ABBREV = 3,
  DW_SECT_LINE = 4,
  DW_SECT_LOC = 5,
  DW_SECT_STR_OFFSETS = 6,
  DW_SECT_MACINFO = 7,
  DW_SECT_MACRO = 8
};

// Source languages.  These are values for DW_AT_language.
enum DwarfLanguage
  {
    DW_LANG_none                     =0x0000,
    DW_LANG_C89                      =0x0001,
    DW_LANG_C                        =0x0002,
    DW_LANG_Ada83                    =0x0003,
    DW_LANG_C_plus_plus              =0x0004,
    DW_LANG_Cobol74                  =0x0005,
    DW_LANG_Cobol85                  =0x0006,
    DW_LANG_Fortran77                =0x0007,
    DW_LANG_Fortran90                =0x0008,
    DW_LANG_Pascal83                 =0x0009,
    DW_LANG_Modula2                  =0x000a,
    DW_LANG_Java                     =0x000b,
    DW_LANG_C99                      =0x000c,
    DW_LANG_Ada95                    =0x000d,
    DW_LANG_Fortran95                =0x000e,
    DW_LANG_PLI                      =0x000f,
    DW_LANG_ObjC                     =0x0010,
    DW_LANG_ObjC_plus_plus           =0x0011,
    DW_LANG_UPC                      =0x0012,
    DW_LANG_D                        =0x0013,
    DW_LANG_Rust                     =0x001c,
    DW_LANG_Swift                    =0x001e,
    // Implementation-defined language code range.
    DW_LANG_lo_user = 0x8000,
    DW_LANG_hi_user = 0xffff,

    // Extensions.

    // MIPS assembly language.  The GNU toolchain uses this for all
    // assembly languages, since there's no generic DW_LANG_ value for that.
    // See include/dwarf2.h in the binutils, gdb, or gcc source trees.
    DW_LANG_Mips_Assembler           =0x8001,
    DW_LANG_Upc                      =0x8765 // Unified Parallel C
  };

// Inline codes.  These are values for DW_AT_inline.
enum DwarfInline {
  DW_INL_not_inlined                 =0x0,
  DW_INL_inlined                     =0x1,
  DW_INL_declared_not_inlined        =0x2,
  DW_INL_declared_inlined            =0x3
};

// Call Frame Info instructions.
enum DwarfCFI
  {
    DW_CFA_advance_loc        = 0x40,
    DW_CFA_offset             = 0x80,
    DW_CFA_restore            = 0xc0,
    DW_CFA_nop                = 0x00,
    DW_CFA_set_loc            = 0x01,
    DW_CFA_advance_loc1       = 0x02,
    DW_CFA_advance_loc2       = 0x03,
    DW_CFA_advance_loc4       = 0x04,
    DW_CFA_offset_extended    = 0x05,
    DW_CFA_restore_extended   = 0x06,
    DW_CFA_undefined          = 0x07,
    DW_CFA_same_value         = 0x08,
    DW_CFA_register           = 0x09,
    DW_CFA_remember_state     = 0x0a,
    DW_CFA_restore_state      = 0x0b,
    DW_CFA_def_cfa            = 0x0c,
    DW_CFA_def_cfa_register   = 0x0d,
    DW_CFA_def_cfa_offset     = 0x0e,
    DW_CFA_def_cfa_expression = 0x0f,
    DW_CFA_expression         = 0x10,
    DW_CFA_offset_extended_sf = 0x11,
    DW_CFA_def_cfa_sf         = 0x12,
    DW_CFA_def_cfa_offset_sf  = 0x13,
    DW_CFA_val_offset         = 0x14,
    DW_CFA_val_offset_sf      = 0x15,
    DW_CFA_val_expression     = 0x16,

    // Opcodes in this range are reserved for user extensions.
    DW_CFA_lo_user = 0x1c,
    DW_CFA_hi_user = 0x3f,

    // SGI/MIPS specific.
    DW_CFA_MIPS_advance_loc8 = 0x1d,

    // GNU extensions.
    DW_CFA_GNU_window_save = 0x2d,
    DW_CFA_GNU_args_size = 0x2e,
    DW_CFA_GNU_negative_offset_extended = 0x2f
  };

// Exception handling 'z' augmentation letters.
enum DwarfZAugmentationCodes {
  // If the CFI augmentation string begins with 'z', then the CIE and FDE
  // have an augmentation data area just before the instructions, whose
  // contents are determined by the subsequent augmentation letters.
  DW_Z_augmentation_start = 'z',

  // If this letter is present in a 'z' augmentation string, the CIE
  // augmentation data includes a pointer encoding, and the FDE
  // augmentation data includes a language-specific data area pointer,
  // represented using that encoding.
  DW_Z_has_LSDA = 'L',

  // If this letter is present in a 'z' augmentation string, the CIE
  // augmentation data includes a pointer encoding, followed by a pointer
  // to a personality routine, represented using that encoding.
  DW_Z_has_personality_routine = 'P',

  // If this letter is present in a 'z' augmentation string, the CIE
  // augmentation data includes a pointer encoding describing how the FDE's
  // initial location, address range, and DW_CFA_set_loc operands are
  // encoded.
  DW_Z_has_FDE_address_encoding = 'R',

  // If this letter is present in a 'z' augmentation string, then code
  // addresses covered by FDEs that cite this CIE are signal delivery
  // trampolines. Return addresses of frames in trampolines should not be
  // adjusted as described in section 6.4.4 of the DWARF 3 spec.
  DW_Z_is_signal_trampoline = 'S'
};

// Exception handling frame description pointer formats, as described
// by the Linux Standard Base Core Specification 4.0, section 11.5,
// DWARF Extensions.
enum DwarfPointerEncoding
  {
    DW_EH_PE_absptr	= 0x00,
    DW_EH_PE_omit	= 0xff,
    DW_EH_PE_uleb128    = 0x01,
    DW_EH_PE_udata2	= 0x02,
    DW_EH_PE_udata4	= 0x03,
    DW_EH_PE_udata8	= 0x04,
    DW_EH_PE_sleb128    = 0x09,
    DW_EH_PE_sdata2	= 0x0A,
    DW_EH_PE_sdata4	= 0x0B,
    DW_EH_PE_sdata8	= 0x0C,
    DW_EH_PE_pcrel	= 0x10,
    DW_EH_PE_textrel	= 0x20,
    DW_EH_PE_datarel	= 0x30,
    DW_EH_PE_funcrel	= 0x40,
    DW_EH_PE_aligned	= 0x50,

    // The GNU toolchain sources define this enum value as well,
    // simply to help classify the lower nybble values into signed and
    // unsigned groups.
    DW_EH_PE_signed	= 0x08,

    // This is not documented in LSB 4.0, but it is used in both the
    // Linux and OS X toolchains. It can be added to any other
    // encoding (except DW_EH_PE_aligned), and indicates that the
    // encoded value represents the address at which the true address
    // is stored, not the true address itself.
    DW_EH_PE_indirect	= 0x80
  };

}  // namespace google_breakpad
#endif  // COMMON_DWARF_DWARF2ENUMS_H__
