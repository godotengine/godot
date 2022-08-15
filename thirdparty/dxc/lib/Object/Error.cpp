//===- Error.cpp - system_error extensions for Object -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This defines a new error_category for the Object library.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/Error.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"

using namespace llvm;
using namespace object;

namespace {
class _object_error_category : public std::error_category {
public:
  const char* name() const LLVM_NOEXCEPT override;
  std::string message(int ev) const override;
};
}

const char *_object_error_category::name() const LLVM_NOEXCEPT {
  return "llvm.object";
}

std::string _object_error_category::message(int EV) const {
  object_error E = static_cast<object_error>(EV);
  switch (E) {
  case object_error::arch_not_found:
    return "No object file for requested architecture";
  case object_error::invalid_file_type:
    return "The file was not recognized as a valid object file";
  case object_error::parse_failed:
    return "Invalid data was encountered while parsing the file";
  case object_error::unexpected_eof:
    return "The end of the file was unexpectedly encountered";
  case object_error::string_table_non_null_end:
    return "String table must end with a null terminator";
  case object_error::invalid_section_index:
    return "Invalid section index";
  case object_error::bitcode_section_not_found:
    return "Bitcode section not found in object file";
  case object_error::macho_small_load_command:
    return "Mach-O load command with size < 8 bytes";
  case object_error::macho_load_segment_too_many_sections:
    return "Mach-O segment load command contains too many sections";
  case object_error::macho_load_segment_too_small:
    return "Mach-O segment load command size is too small";
  }
  llvm_unreachable("An enumerator of object_error does not have a message "
                   "defined.");
}

static ManagedStatic<_object_error_category> error_category;

const std::error_category &object::object_category() {
  return *error_category;
}
