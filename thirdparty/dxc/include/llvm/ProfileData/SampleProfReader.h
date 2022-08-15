//===- SampleProfReader.h - Read LLVM sample profile data -----------------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions needed for reading sample profiles.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_PROFILEDATA_SAMPLEPROFREADER_H
#define LLVM_PROFILEDATA_SAMPLEPROFREADER_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/ProfileData/SampleProf.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {

namespace sampleprof {

/// \brief Sample-based profile reader.
///
/// Each profile contains sample counts for all the functions
/// executed. Inside each function, statements are annotated with the
/// collected samples on all the instructions associated with that
/// statement.
///
/// For this to produce meaningful data, the program needs to be
/// compiled with some debug information (at minimum, line numbers:
/// -gline-tables-only). Otherwise, it will be impossible to match IR
/// instructions to the line numbers collected by the profiler.
///
/// From the profile file, we are interested in collecting the
/// following information:
///
/// * A list of functions included in the profile (mangled names).
///
/// * For each function F:
///   1. The total number of samples collected in F.
///
///   2. The samples collected at each line in F. To provide some
///      protection against source code shuffling, line numbers should
///      be relative to the start of the function.
///
/// The reader supports two file formats: text and binary. The text format
/// is useful for debugging and testing, while the binary format is more
/// compact. They can both be used interchangeably.
class SampleProfileReader {
public:
  SampleProfileReader(std::unique_ptr<MemoryBuffer> B, LLVMContext &C)
      : Profiles(0), Ctx(C), Buffer(std::move(B)) {}

  virtual ~SampleProfileReader() {}

  /// \brief Read and validate the file header.
  virtual std::error_code readHeader() = 0;

  /// \brief Read sample profiles from the associated file.
  virtual std::error_code read() = 0;

  /// \brief Print the profile for \p FName on stream \p OS.
  void dumpFunctionProfile(StringRef FName, raw_ostream &OS = dbgs());

  /// \brief Print all the profiles on stream \p OS.
  void dump(raw_ostream &OS = dbgs());

  /// \brief Return the samples collected for function \p F.
  FunctionSamples *getSamplesFor(const Function &F) {
    return &Profiles[F.getName()];
  }

  /// \brief Return all the profiles.
  StringMap<FunctionSamples> &getProfiles() { return Profiles; }

  /// \brief Report a parse error message.
  void reportParseError(int64_t LineNumber, Twine Msg) const {
    Ctx.diagnose(DiagnosticInfoSampleProfile(Buffer->getBufferIdentifier(),
                                             LineNumber, Msg));
  }

  /// \brief Create a sample profile reader appropriate to the file format.
  static ErrorOr<std::unique_ptr<SampleProfileReader>>
  create(StringRef Filename, LLVMContext &C);

protected:
  /// \brief Map every function to its associated profile.
  ///
  /// The profile of every function executed at runtime is collected
  /// in the structure FunctionSamples. This maps function objects
  /// to their corresponding profiles.
  StringMap<FunctionSamples> Profiles;

  /// \brief LLVM context used to emit diagnostics.
  LLVMContext &Ctx;

  /// \brief Memory buffer holding the profile file.
  std::unique_ptr<MemoryBuffer> Buffer;
};

class SampleProfileReaderText : public SampleProfileReader {
public:
  SampleProfileReaderText(std::unique_ptr<MemoryBuffer> B, LLVMContext &C)
      : SampleProfileReader(std::move(B), C) {}

  /// \brief Read and validate the file header.
  std::error_code readHeader() override { return sampleprof_error::success; }

  /// \brief Read sample profiles from the associated file.
  std::error_code read() override;
};

class SampleProfileReaderBinary : public SampleProfileReader {
public:
  SampleProfileReaderBinary(std::unique_ptr<MemoryBuffer> B, LLVMContext &C)
      : SampleProfileReader(std::move(B), C), Data(nullptr), End(nullptr) {}

  /// \brief Read and validate the file header.
  std::error_code readHeader() override;

  /// \brief Read sample profiles from the associated file.
  std::error_code read() override;

  /// \brief Return true if \p Buffer is in the format supported by this class.
  static bool hasFormat(const MemoryBuffer &Buffer);

protected:
  /// \brief Read a numeric value of type T from the profile.
  ///
  /// If an error occurs during decoding, a diagnostic message is emitted and
  /// EC is set.
  ///
  /// \returns the read value.
  template <typename T> ErrorOr<T> readNumber();

  /// \brief Read a string from the profile.
  ///
  /// If an error occurs during decoding, a diagnostic message is emitted and
  /// EC is set.
  ///
  /// \returns the read value.
  ErrorOr<StringRef> readString();

  /// \brief Return true if we've reached the end of file.
  bool at_eof() const { return Data >= End; }

  /// \brief Points to the current location in the buffer.
  const uint8_t *Data;

  /// \brief Points to the end of the buffer.
  const uint8_t *End;
};

} // End namespace sampleprof

} // End namespace llvm

#endif // LLVM_PROFILEDATA_SAMPLEPROFREADER_H
