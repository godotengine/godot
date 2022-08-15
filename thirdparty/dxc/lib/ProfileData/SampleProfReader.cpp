//===- SampleProfReader.cpp - Read LLVM sample profile data ---------------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the class that reads LLVM sample profiles. It
// supports two file formats: text and binary. The textual representation
// is useful for debugging and testing purposes. The binary representation
// is more compact, resulting in smaller file sizes. However, they can
// both be used interchangeably.
//
// NOTE: If you are making changes to the file format, please remember
//       to document them in the Clang documentation at
//       tools/clang/docs/UsersManual.rst.
//
// Text format
// -----------
//
// Sample profiles are written as ASCII text. The file is divided into
// sections, which correspond to each of the functions executed at runtime.
// Each section has the following format
//
//     function1:total_samples:total_head_samples
//     offset1[.discriminator]: number_of_samples [fn1:num fn2:num ... ]
//     offset2[.discriminator]: number_of_samples [fn3:num fn4:num ... ]
//     ...
//     offsetN[.discriminator]: number_of_samples [fn5:num fn6:num ... ]
//
// The file may contain blank lines between sections and within a
// section. However, the spacing within a single line is fixed. Additional
// spaces will result in an error while reading the file.
//
// Function names must be mangled in order for the profile loader to
// match them in the current translation unit. The two numbers in the
// function header specify how many total samples were accumulated in the
// function (first number), and the total number of samples accumulated
// in the prologue of the function (second number). This head sample
// count provides an indicator of how frequently the function is invoked.
//
// Each sampled line may contain several items. Some are optional (marked
// below):
//
// a. Source line offset. This number represents the line number
//    in the function where the sample was collected. The line number is
//    always relative to the line where symbol of the function is
//    defined. So, if the function has its header at line 280, the offset
//    13 is at line 293 in the file.
//
//    Note that this offset should never be a negative number. This could
//    happen in cases like macros. The debug machinery will register the
//    line number at the point of macro expansion. So, if the macro was
//    expanded in a line before the start of the function, the profile
//    converter should emit a 0 as the offset (this means that the optimizers
//    will not be able to associate a meaningful weight to the instructions
//    in the macro).
//
// b. [OPTIONAL] Discriminator. This is used if the sampled program
//    was compiled with DWARF discriminator support
//    (http://wiki.dwarfstd.org/index.php?title=Path_Discriminators).
//    DWARF discriminators are unsigned integer values that allow the
//    compiler to distinguish between multiple execution paths on the
//    same source line location.
//
//    For example, consider the line of code ``if (cond) foo(); else bar();``.
//    If the predicate ``cond`` is true 80% of the time, then the edge
//    into function ``foo`` should be considered to be taken most of the
//    time. But both calls to ``foo`` and ``bar`` are at the same source
//    line, so a sample count at that line is not sufficient. The
//    compiler needs to know which part of that line is taken more
//    frequently.
//
//    This is what discriminators provide. In this case, the calls to
//    ``foo`` and ``bar`` will be at the same line, but will have
//    different discriminator values. This allows the compiler to correctly
//    set edge weights into ``foo`` and ``bar``.
//
// c. Number of samples. This is an integer quantity representing the
//    number of samples collected by the profiler at this source
//    location.
//
// d. [OPTIONAL] Potential call targets and samples. If present, this
//    line contains a call instruction. This models both direct and
//    number of samples. For example,
//
//      130: 7  foo:3  bar:2  baz:7
//
//    The above means that at relative line offset 130 there is a call
//    instruction that calls one of ``foo()``, ``bar()`` and ``baz()``,
//    with ``baz()`` being the relatively more frequently called target.
//
//===----------------------------------------------------------------------===//

#include "llvm/ProfileData/SampleProfReader.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Regex.h"

using namespace llvm::sampleprof;
using namespace llvm;

/// \brief Print the samples collected for a function on stream \p OS.
///
/// \param OS Stream to emit the output to.
void FunctionSamples::print(raw_ostream &OS) {
  OS << TotalSamples << ", " << TotalHeadSamples << ", " << BodySamples.size()
     << " sampled lines\n";
  for (const auto &SI : BodySamples) {
    LineLocation Loc = SI.first;
    const SampleRecord &Sample = SI.second;
    OS << "\tline offset: " << Loc.LineOffset
       << ", discriminator: " << Loc.Discriminator
       << ", number of samples: " << Sample.getSamples();
    if (Sample.hasCalls()) {
      OS << ", calls:";
      for (const auto &I : Sample.getCallTargets())
        OS << " " << I.first() << ":" << I.second;
    }
    OS << "\n";
  }
  OS << "\n";
}

/// \brief Dump the function profile for \p FName.
///
/// \param FName Name of the function to print.
/// \param OS Stream to emit the output to.
void SampleProfileReader::dumpFunctionProfile(StringRef FName,
                                              raw_ostream &OS) {
  OS << "Function: " << FName << ": ";
  Profiles[FName].print(OS);
}

/// \brief Dump all the function profiles found on stream \p OS.
void SampleProfileReader::dump(raw_ostream &OS) {
  for (const auto &I : Profiles)
    dumpFunctionProfile(I.getKey(), OS);
}

/// \brief Load samples from a text file.
///
/// See the documentation at the top of the file for an explanation of
/// the expected format.
///
/// \returns true if the file was loaded successfully, false otherwise.
std::error_code SampleProfileReaderText::read() {
  line_iterator LineIt(*Buffer, /*SkipBlanks=*/true, '#');

  // Read the profile of each function. Since each function may be
  // mentioned more than once, and we are collecting flat profiles,
  // accumulate samples as we parse them.
  Regex HeadRE("^([^0-9].*):([0-9]+):([0-9]+)$");
  Regex LineSampleRE("^([0-9]+)\\.?([0-9]+)?: ([0-9]+)(.*)$");
  Regex CallSampleRE(" +([^0-9 ][^ ]*):([0-9]+)");
  while (!LineIt.is_at_eof()) {
    // Read the header of each function.
    //
    // Note that for function identifiers we are actually expecting
    // mangled names, but we may not always get them. This happens when
    // the compiler decides not to emit the function (e.g., it was inlined
    // and removed). In this case, the binary will not have the linkage
    // name for the function, so the profiler will emit the function's
    // unmangled name, which may contain characters like ':' and '>' in its
    // name (member functions, templates, etc).
    //
    // The only requirement we place on the identifier, then, is that it
    // should not begin with a number.
    SmallVector<StringRef, 4> Matches;
    if (!HeadRE.match(*LineIt, &Matches)) {
      reportParseError(LineIt.line_number(),
                       "Expected 'mangled_name:NUM:NUM', found " + *LineIt);
      return sampleprof_error::malformed;
    }
    assert(Matches.size() == 4);
    StringRef FName = Matches[1];
    unsigned NumSamples, NumHeadSamples;
    Matches[2].getAsInteger(10, NumSamples);
    Matches[3].getAsInteger(10, NumHeadSamples);
    Profiles[FName] = FunctionSamples();
    FunctionSamples &FProfile = Profiles[FName];
    FProfile.addTotalSamples(NumSamples);
    FProfile.addHeadSamples(NumHeadSamples);
    ++LineIt;

    // Now read the body. The body of the function ends when we reach
    // EOF or when we see the start of the next function.
    while (!LineIt.is_at_eof() && isdigit((*LineIt)[0])) {
      if (!LineSampleRE.match(*LineIt, &Matches)) {
        reportParseError(
            LineIt.line_number(),
            "Expected 'NUM[.NUM]: NUM[ mangled_name:NUM]*', found " + *LineIt);
        return sampleprof_error::malformed;
      }
      assert(Matches.size() == 5);
      unsigned LineOffset, NumSamples, Discriminator = 0;
      Matches[1].getAsInteger(10, LineOffset);
      if (Matches[2] != "")
        Matches[2].getAsInteger(10, Discriminator);
      Matches[3].getAsInteger(10, NumSamples);

      // If there are function calls in this line, generate a call sample
      // entry for each call.
      std::string CallsLine(Matches[4]);
      while (CallsLine != "") {
        SmallVector<StringRef, 3> CallSample;
        if (!CallSampleRE.match(CallsLine, &CallSample)) {
          reportParseError(LineIt.line_number(),
                           "Expected 'mangled_name:NUM', found " + CallsLine);
          return sampleprof_error::malformed;
        }
        StringRef CalledFunction = CallSample[1];
        unsigned CalledFunctionSamples;
        CallSample[2].getAsInteger(10, CalledFunctionSamples);
        FProfile.addCalledTargetSamples(LineOffset, Discriminator,
                                        CalledFunction, CalledFunctionSamples);
        CallsLine = CallSampleRE.sub("", CallsLine);
      }

      FProfile.addBodySamples(LineOffset, Discriminator, NumSamples);
      ++LineIt;
    }
  }

  return sampleprof_error::success;
}

template <typename T> ErrorOr<T> SampleProfileReaderBinary::readNumber() {
  unsigned NumBytesRead = 0;
  std::error_code EC;
  uint64_t Val = decodeULEB128(Data, &NumBytesRead);

  if (Val > std::numeric_limits<T>::max())
    EC = sampleprof_error::malformed;
  else if (Data + NumBytesRead > End)
    EC = sampleprof_error::truncated;
  else
    EC = sampleprof_error::success;

  if (EC) {
    reportParseError(0, EC.message());
    return EC;
  }

  Data += NumBytesRead;
  return static_cast<T>(Val);
}

ErrorOr<StringRef> SampleProfileReaderBinary::readString() {
  std::error_code EC;
  StringRef Str(reinterpret_cast<const char *>(Data));
  if (Data + Str.size() + 1 > End) {
    EC = sampleprof_error::truncated;
    reportParseError(0, EC.message());
    return EC;
  }

  Data += Str.size() + 1;
  return Str;
}

std::error_code SampleProfileReaderBinary::read() {
  while (!at_eof()) {
    auto FName(readString());
    if (std::error_code EC = FName.getError())
      return EC;

    Profiles[*FName] = FunctionSamples();
    FunctionSamples &FProfile = Profiles[*FName];

    auto Val = readNumber<unsigned>();
    if (std::error_code EC = Val.getError())
      return EC;
    FProfile.addTotalSamples(*Val);

    Val = readNumber<unsigned>();
    if (std::error_code EC = Val.getError())
      return EC;
    FProfile.addHeadSamples(*Val);

    // Read the samples in the body.
    auto NumRecords = readNumber<unsigned>();
    if (std::error_code EC = NumRecords.getError())
      return EC;
    for (unsigned I = 0; I < *NumRecords; ++I) {
      auto LineOffset = readNumber<uint64_t>();
      if (std::error_code EC = LineOffset.getError())
        return EC;

      auto Discriminator = readNumber<uint64_t>();
      if (std::error_code EC = Discriminator.getError())
        return EC;

      auto NumSamples = readNumber<uint64_t>();
      if (std::error_code EC = NumSamples.getError())
        return EC;

      auto NumCalls = readNumber<unsigned>();
      if (std::error_code EC = NumCalls.getError())
        return EC;

      for (unsigned J = 0; J < *NumCalls; ++J) {
        auto CalledFunction(readString());
        if (std::error_code EC = CalledFunction.getError())
          return EC;

        auto CalledFunctionSamples = readNumber<uint64_t>();
        if (std::error_code EC = CalledFunctionSamples.getError())
          return EC;

        FProfile.addCalledTargetSamples(*LineOffset, *Discriminator,
                                        *CalledFunction,
                                        *CalledFunctionSamples);
      }

      FProfile.addBodySamples(*LineOffset, *Discriminator, *NumSamples);
    }
  }

  return sampleprof_error::success;
}

std::error_code SampleProfileReaderBinary::readHeader() {
  Data = reinterpret_cast<const uint8_t *>(Buffer->getBufferStart());
  End = Data + Buffer->getBufferSize();

  // Read and check the magic identifier.
  auto Magic = readNumber<uint64_t>();
  if (std::error_code EC = Magic.getError())
    return EC;
  else if (*Magic != SPMagic())
    return sampleprof_error::bad_magic;

  // Read the version number.
  auto Version = readNumber<uint64_t>();
  if (std::error_code EC = Version.getError())
    return EC;
  else if (*Version != SPVersion())
    return sampleprof_error::unsupported_version;

  return sampleprof_error::success;
}

bool SampleProfileReaderBinary::hasFormat(const MemoryBuffer &Buffer) {
  const uint8_t *Data =
      reinterpret_cast<const uint8_t *>(Buffer.getBufferStart());
  uint64_t Magic = decodeULEB128(Data);
  return Magic == SPMagic();
}

/// \brief Prepare a memory buffer for the contents of \p Filename.
///
/// \returns an error code indicating the status of the buffer.
static ErrorOr<std::unique_ptr<MemoryBuffer>>
setupMemoryBuffer(std::string Filename) {
  auto BufferOrErr = MemoryBuffer::getFileOrSTDIN(Filename);
  if (std::error_code EC = BufferOrErr.getError())
    return EC;
  auto Buffer = std::move(BufferOrErr.get());

  // Sanity check the file.
  if (Buffer->getBufferSize() > std::numeric_limits<unsigned>::max())
    return sampleprof_error::too_large;

  return std::move(Buffer);
}

/// \brief Create a sample profile reader based on the format of the input file.
///
/// \param Filename The file to open.
///
/// \param Reader The reader to instantiate according to \p Filename's format.
///
/// \param C The LLVM context to use to emit diagnostics.
///
/// \returns an error code indicating the status of the created reader.
ErrorOr<std::unique_ptr<SampleProfileReader>>
SampleProfileReader::create(StringRef Filename, LLVMContext &C) {
  auto BufferOrError = setupMemoryBuffer(Filename);
  if (std::error_code EC = BufferOrError.getError())
    return EC;

  auto Buffer = std::move(BufferOrError.get());
  std::unique_ptr<SampleProfileReader> Reader;
  if (SampleProfileReaderBinary::hasFormat(*Buffer))
    Reader.reset(new SampleProfileReaderBinary(std::move(Buffer), C));
  else
    Reader.reset(new SampleProfileReaderText(std::move(Buffer), C));

  if (std::error_code EC = Reader->readHeader())
    return EC;

  return std::move(Reader);
}
