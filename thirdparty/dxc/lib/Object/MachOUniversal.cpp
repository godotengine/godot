//===- MachOUniversal.cpp - Mach-O universal binary -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the MachOUniversalBinary class.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/MachOUniversal.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/MachO.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"

using namespace llvm;
using namespace object;

template<typename T>
static void SwapStruct(T &Value);

template<>
void SwapStruct(MachO::fat_header &H) {
  sys::swapByteOrder(H.magic);
  sys::swapByteOrder(H.nfat_arch);
}

template<>
void SwapStruct(MachO::fat_arch &H) {
  sys::swapByteOrder(H.cputype);
  sys::swapByteOrder(H.cpusubtype);
  sys::swapByteOrder(H.offset);
  sys::swapByteOrder(H.size);
  sys::swapByteOrder(H.align);
}

template<typename T>
static T getUniversalBinaryStruct(const char *Ptr) {
  T Res;
  memcpy(&Res, Ptr, sizeof(T));
  // Universal binary headers have big-endian byte order.
  if (sys::IsLittleEndianHost)
    SwapStruct(Res);
  return Res;
}

MachOUniversalBinary::ObjectForArch::ObjectForArch(
    const MachOUniversalBinary *Parent, uint32_t Index)
    : Parent(Parent), Index(Index) {
  if (!Parent || Index >= Parent->getNumberOfObjects()) {
    clear();
  } else {
    // Parse object header.
    StringRef ParentData = Parent->getData();
    const char *HeaderPos = ParentData.begin() + sizeof(MachO::fat_header) +
                            Index * sizeof(MachO::fat_arch);
    Header = getUniversalBinaryStruct<MachO::fat_arch>(HeaderPos);
    if (ParentData.size() < Header.offset + Header.size) {
      clear();
    }
  }
}

ErrorOr<std::unique_ptr<MachOObjectFile>>
MachOUniversalBinary::ObjectForArch::getAsObjectFile() const {
  if (Parent) {
    StringRef ParentData = Parent->getData();
    StringRef ObjectData = ParentData.substr(Header.offset, Header.size);
    StringRef ObjectName = Parent->getFileName();
    MemoryBufferRef ObjBuffer(ObjectData, ObjectName);
    return ObjectFile::createMachOObjectFile(ObjBuffer);
  }
  return object_error::parse_failed;
}

ErrorOr<std::unique_ptr<Archive>>
MachOUniversalBinary::ObjectForArch::getAsArchive() const {
  if (!Parent)
    return object_error::parse_failed;

  StringRef ParentData = Parent->getData();
  StringRef ObjectData = ParentData.substr(Header.offset, Header.size);
  StringRef ObjectName = Parent->getFileName();
  MemoryBufferRef ObjBuffer(ObjectData, ObjectName);
  return Archive::create(ObjBuffer);
}

void MachOUniversalBinary::anchor() { }

ErrorOr<std::unique_ptr<MachOUniversalBinary>>
MachOUniversalBinary::create(MemoryBufferRef Source) {
  std::error_code EC;
  std::unique_ptr<MachOUniversalBinary> Ret(
      new MachOUniversalBinary(Source, EC));
  if (EC)
    return EC;
  return std::move(Ret);
}

MachOUniversalBinary::MachOUniversalBinary(MemoryBufferRef Source,
                                           std::error_code &ec)
    : Binary(Binary::ID_MachOUniversalBinary, Source), NumberOfObjects(0) {
  if (Data.getBufferSize() < sizeof(MachO::fat_header)) {
    ec = object_error::invalid_file_type;
    return;
  }
  // Check for magic value and sufficient header size.
  StringRef Buf = getData();
  MachO::fat_header H= getUniversalBinaryStruct<MachO::fat_header>(Buf.begin());
  NumberOfObjects = H.nfat_arch;
  uint32_t MinSize = sizeof(MachO::fat_header) +
                     sizeof(MachO::fat_arch) * NumberOfObjects;
  if (H.magic != MachO::FAT_MAGIC || Buf.size() < MinSize) {
    ec = object_error::parse_failed;
    return;
  }
  ec = std::error_code();
}

ErrorOr<std::unique_ptr<MachOObjectFile>>
MachOUniversalBinary::getObjectForArch(StringRef ArchName) const {
  if (Triple(ArchName).getArch() == Triple::ArchType::UnknownArch)
    return object_error::arch_not_found;

  for (object_iterator I = begin_objects(), E = end_objects(); I != E; ++I) {
    if (I->getArchTypeName() == ArchName)
      return I->getAsObjectFile();
  }
  return object_error::arch_not_found;
}
