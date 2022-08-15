//===- MachOUniversal.h - Mach-O universal binaries -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares Mach-O fat/universal binaries.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_MACHOUNIVERSAL_H
#define LLVM_OBJECT_MACHOUNIVERSAL_H

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Object/Archive.h"
#include "llvm/Object/Binary.h"
#include "llvm/Object/MachO.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MachO.h"

namespace llvm {
namespace object {

class MachOUniversalBinary : public Binary {
  virtual void anchor();

  uint32_t NumberOfObjects;
public:
  class ObjectForArch {
    const MachOUniversalBinary *Parent;
    /// \brief Index of object in the universal binary.
    uint32_t Index;
    /// \brief Descriptor of the object.
    MachO::fat_arch Header;

  public:
    ObjectForArch(const MachOUniversalBinary *Parent, uint32_t Index);

    void clear() {
      Parent = nullptr;
      Index = 0;
    }

    bool operator==(const ObjectForArch &Other) const {
      return (Parent == Other.Parent) && (Index == Other.Index);
    }

    ObjectForArch getNext() const { return ObjectForArch(Parent, Index + 1); }
    uint32_t getCPUType() const { return Header.cputype; }
    uint32_t getCPUSubType() const { return Header.cpusubtype; }
    uint32_t getOffset() const { return Header.offset; }
    uint32_t getSize() const { return Header.size; }
    uint32_t getAlign() const { return Header.align; }
    std::string getArchTypeName() const {
      Triple T = MachOObjectFile::getArch(Header.cputype, Header.cpusubtype);
      return T.getArchName();
    }

    ErrorOr<std::unique_ptr<MachOObjectFile>> getAsObjectFile() const;

    ErrorOr<std::unique_ptr<Archive>> getAsArchive() const;
  };

  class object_iterator {
    ObjectForArch Obj;
  public:
    object_iterator(const ObjectForArch &Obj) : Obj(Obj) {}
    const ObjectForArch *operator->() const { return &Obj; }
    const ObjectForArch &operator*() const { return Obj; }

    bool operator==(const object_iterator &Other) const {
      return Obj == Other.Obj;
    }
    bool operator!=(const object_iterator &Other) const {
      return !(*this == Other);
    }

    object_iterator& operator++() {  // Preincrement
      Obj = Obj.getNext();
      return *this;
    }
  };

  MachOUniversalBinary(MemoryBufferRef Souce, std::error_code &EC);
  static ErrorOr<std::unique_ptr<MachOUniversalBinary>>
  create(MemoryBufferRef Source);

  object_iterator begin_objects() const {
    return ObjectForArch(this, 0);
  }
  object_iterator end_objects() const {
    return ObjectForArch(nullptr, 0);
  }

  iterator_range<object_iterator> objects() const {
    return make_range(begin_objects(), end_objects());
  }

  uint32_t getNumberOfObjects() const { return NumberOfObjects; }

  // Cast methods.
  static inline bool classof(Binary const *V) {
    return V->isMachOUniversalBinary();
  }

  ErrorOr<std::unique_ptr<MachOObjectFile>>
  getObjectForArch(StringRef ArchName) const;
};

}
}

#endif
