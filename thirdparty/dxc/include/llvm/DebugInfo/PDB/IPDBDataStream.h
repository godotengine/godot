//===- IPDBDataStream.h - base interface for child enumerator -*- C++ ---*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DEBUGINFO_PDB_IPDBDATASTREAM_H
#define LLVM_DEBUGINFO_PDB_IPDBDATASTREAM_H

#include "PDBTypes.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"

namespace llvm {

/// IPDBDataStream defines an interface used to represent a stream consisting
/// of a name and a series of records whose formats depend on the particular
/// stream type.
class IPDBDataStream {
public:
  typedef llvm::SmallVector<uint8_t, 32> RecordType;

  virtual ~IPDBDataStream();

  virtual uint32_t getRecordCount() const = 0;
  virtual std::string getName() const = 0;
  virtual llvm::Optional<RecordType> getItemAtIndex(uint32_t Index) const = 0;
  virtual bool getNext(RecordType &Record) = 0;
  virtual void reset() = 0;
  virtual IPDBDataStream *clone() const = 0;
};
}

#endif
