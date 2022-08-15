//===- ELFObjectFile.cpp - ELF object file implementation -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Part of the ELFObjectFile class implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Support/MathExtras.h"

namespace llvm {
using namespace object;

ELFObjectFileBase::ELFObjectFileBase(unsigned int Type, MemoryBufferRef Source)
    : ObjectFile(Type, Source) {}

ErrorOr<std::unique_ptr<ObjectFile>>
ObjectFile::createELFObjectFile(MemoryBufferRef Obj) {
  std::pair<unsigned char, unsigned char> Ident =
      getElfArchType(Obj.getBuffer());
  std::size_t MaxAlignment =
      1ULL << countTrailingZeros(uintptr_t(Obj.getBufferStart()));

  if (MaxAlignment < 2)
    return object_error::parse_failed;

  std::error_code EC;
  std::unique_ptr<ObjectFile> R;
  if (Ident.first == ELF::ELFCLASS32) {
    if (Ident.second == ELF::ELFDATA2LSB)
      R.reset(new ELFObjectFile<ELFType<support::little, false>>(Obj, EC));
    else if (Ident.second == ELF::ELFDATA2MSB)
      R.reset(new ELFObjectFile<ELFType<support::big, false>>(Obj, EC));
    else
      return object_error::parse_failed;
  } else if (Ident.first == ELF::ELFCLASS64) {
    if (Ident.second == ELF::ELFDATA2LSB)
      R.reset(new ELFObjectFile<ELFType<support::little, true>>(Obj, EC));
    else if (Ident.second == ELF::ELFDATA2MSB)
      R.reset(new ELFObjectFile<ELFType<support::big, true>>(Obj, EC));
    else
      return object_error::parse_failed;
  } else {
    return object_error::parse_failed;
  }

  if (EC)
    return EC;
  return std::move(R);
}

} // end namespace llvm
