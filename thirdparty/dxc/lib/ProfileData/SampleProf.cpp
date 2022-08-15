//=-- SampleProf.cpp - Sample profiling format support --------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains common definitions used in the reading and writing of
// sample profile data.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/SampleProf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"

using namespace llvm;

namespace {
class SampleProfErrorCategoryType : public std::error_category {
  const char *name() const LLVM_NOEXCEPT override { return "llvm.sampleprof"; }
  std::string message(int IE) const override {
    sampleprof_error E = static_cast<sampleprof_error>(IE);
    switch (E) {
    case sampleprof_error::success:
      return "Success";
    case sampleprof_error::bad_magic:
      return "Invalid file format (bad magic)";
    case sampleprof_error::unsupported_version:
      return "Unsupported format version";
    case sampleprof_error::too_large:
      return "Too much profile data";
    case sampleprof_error::truncated:
      return "Truncated profile data";
    case sampleprof_error::malformed:
      return "Malformed profile data";
    case sampleprof_error::unrecognized_format:
      return "Unrecognized profile encoding format";
    }
    llvm_unreachable("A value of sampleprof_error has no message.");
  }
};
}

static ManagedStatic<SampleProfErrorCategoryType> ErrorCategory;

const std::error_category &llvm::sampleprof_category() {
  return *ErrorCategory;
}
