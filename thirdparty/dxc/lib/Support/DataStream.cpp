//===--- llvm/Support/DataStream.cpp - Lazy streamed data -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements DataStreamer, which fetches bytes of Data from
// a stream source. It provides support for streaming (lazy reading) of
// bitcode. An example implementation of streaming from a file or stdin
// is included.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/DataStream.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include <string>
#include <system_error>
#if !defined(_MSC_VER) && !defined(__MINGW32__)
#include <unistd.h>
#else
#include <io.h>
#endif
using namespace llvm;

#define DEBUG_TYPE "Data-stream"

// Interface goals:
// * StreamingMemoryObject doesn't care about complexities like using
//   threads/async callbacks to actually overlap download+compile
// * Don't want to duplicate Data in memory
// * Don't need to know total Data len in advance
// Non-goals:
// StreamingMemoryObject already has random access so this interface only does
// in-order streaming (no arbitrary seeking, else we'd have to buffer all the
// Data here in addition to MemoryObject).  This also means that if we want
// to be able to to free Data, BitstreamBytes/BitcodeReader will implement it

STATISTIC(NumStreamFetches, "Number of calls to Data stream fetch");

namespace llvm {
DataStreamer::~DataStreamer() {}
}

namespace {

// Very simple stream backed by a file. Mostly useful for stdin and debugging;
// actual file access is probably still best done with mmap.
class DataFileStreamer : public DataStreamer {
 int Fd;
public:
  DataFileStreamer() : Fd(0) {}
  virtual ~DataFileStreamer() {
    llvm::sys::fs::msf_close(Fd);  // HLSL Change - use msf_close
  }
  size_t GetBytes(unsigned char *buf, size_t len) override {
    NumStreamFetches++;
    return llvm::sys::fs::msf_read(Fd, buf, len);
  }

  std::error_code OpenFile(const std::string &Filename) {
    if (Filename == "-") {
      Fd = 0;
      sys::ChangeStdinToBinary();
      return std::error_code();
    }

    return sys::fs::openFileForRead(Filename, Fd);
  }
};

}

std::unique_ptr<DataStreamer>
llvm::getDataFileStreamer(const std::string &Filename, std::string *StrError) {
  std::unique_ptr<DataFileStreamer> s = make_unique<DataFileStreamer>();
  if (std::error_code e = s->OpenFile(Filename)) {
    *StrError = std::string("Could not open ") + Filename + ": " +
        e.message() + "\n";
    return nullptr;
  }
  return std::move(s);
}
