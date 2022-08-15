//===- SampleProfWriter.cpp - Write LLVM sample profile data --------------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the class that writes LLVM sample profiles. It
// supports two file formats: text and binary. The textual representation
// is useful for debugging and testing purposes. The binary representation
// is more compact, resulting in smaller file sizes. However, they can
// both be used interchangeably.
//
// See lib/ProfileData/SampleProfReader.cpp for documentation on each of the
// supported formats.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/SampleProfWriter.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Regex.h"

using namespace llvm::sampleprof;
using namespace llvm;

/// \brief Write samples to a text file.
bool SampleProfileWriterText::write(StringRef FName, const FunctionSamples &S) {
  if (S.empty())
    return true;

  OS << FName << ":" << S.getTotalSamples() << ":" << S.getHeadSamples()
     << "\n";

  for (const auto &I : S.getBodySamples()) {
    LineLocation Loc = I.first;
    const SampleRecord &Sample = I.second;
    if (Loc.Discriminator == 0)
      OS << Loc.LineOffset << ": ";
    else
      OS << Loc.LineOffset << "." << Loc.Discriminator << ": ";

    OS << Sample.getSamples();

    for (const auto &J : Sample.getCallTargets())
      OS << " " << J.first() << ":" << J.second;
    OS << "\n";
  }

  return true;
}

SampleProfileWriterBinary::SampleProfileWriterBinary(StringRef F,
                                                     std::error_code &EC)
    : SampleProfileWriter(F, EC, sys::fs::F_None) {
  if (EC)
    return;

  // Write the file header.
  encodeULEB128(SPMagic(), OS);
  encodeULEB128(SPVersion(), OS);
}

/// \brief Write samples to a binary file.
///
/// \returns true if the samples were written successfully, false otherwise.
bool SampleProfileWriterBinary::write(StringRef FName,
                                      const FunctionSamples &S) {
  if (S.empty())
    return true;

  OS << FName;
  encodeULEB128(0, OS);
  encodeULEB128(S.getTotalSamples(), OS);
  encodeULEB128(S.getHeadSamples(), OS);
  encodeULEB128(S.getBodySamples().size(), OS);
  for (const auto &I : S.getBodySamples()) {
    LineLocation Loc = I.first;
    const SampleRecord &Sample = I.second;
    encodeULEB128(Loc.LineOffset, OS);
    encodeULEB128(Loc.Discriminator, OS);
    encodeULEB128(Sample.getSamples(), OS);
    encodeULEB128(Sample.getCallTargets().size(), OS);
    for (const auto &J : Sample.getCallTargets()) {
      std::string Callee = J.first();
      unsigned CalleeSamples = J.second;
      OS << Callee;
      encodeULEB128(0, OS);
      encodeULEB128(CalleeSamples, OS);
    }
  }

  return true;
}

/// \brief Create a sample profile writer based on the specified format.
///
/// \param Filename The file to create.
///
/// \param Writer The writer to instantiate according to the specified format.
///
/// \param Format Encoding format for the profile file.
///
/// \returns an error code indicating the status of the created writer.
ErrorOr<std::unique_ptr<SampleProfileWriter>>
SampleProfileWriter::create(StringRef Filename, SampleProfileFormat Format) {
  std::error_code EC;
  std::unique_ptr<SampleProfileWriter> Writer;

  if (Format == SPF_Binary)
    Writer.reset(new SampleProfileWriterBinary(Filename, EC));
  else if (Format == SPF_Text)
    Writer.reset(new SampleProfileWriterText(Filename, EC));
  else
    EC = sampleprof_error::unrecognized_format;

  if (EC)
    return EC;

  return std::move(Writer);
}
