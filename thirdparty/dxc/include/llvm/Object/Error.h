//===- Error.h - system_error extensions for Object -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This declares a new error_category for the Object library.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_ERROR_H
#define LLVM_OBJECT_ERROR_H

#include <system_error>

namespace llvm {
namespace object {

const std::error_category &object_category();

enum class object_error {
  // Error code 0 is absent. Use std::error_code() instead.
  arch_not_found = 1,
  invalid_file_type,
  parse_failed,
  unexpected_eof,
  string_table_non_null_end,
  invalid_section_index,
  bitcode_section_not_found,
  macho_small_load_command,
  macho_load_segment_too_many_sections,
  macho_load_segment_too_small,
};

inline std::error_code make_error_code(object_error e) {
  return std::error_code(static_cast<int>(e), object_category());
}

} // end namespace object.

} // end namespace llvm.

namespace std {
template <>
struct is_error_code_enum<llvm::object::object_error> : std::true_type {};
}

#endif
