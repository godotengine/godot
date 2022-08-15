//===- ObjectFile.h - File format independent object file -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares a file format independent ObjectFile class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OBJECT_OBJECTFILE_H
#define LLVM_OBJECT_OBJECTFILE_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Object/SymbolicFile.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include <cstring>
#include <vector>

namespace llvm {
namespace object {

class ObjectFile;
class COFFObjectFile;
class MachOObjectFile;

class SymbolRef;
class symbol_iterator;
class SectionRef;
typedef content_iterator<SectionRef> section_iterator;

/// This is a value type class that represents a single relocation in the list
/// of relocations in the object file.
class RelocationRef {
  DataRefImpl RelocationPimpl;
  const ObjectFile *OwningObject;

public:
  RelocationRef() : OwningObject(nullptr) { }

  RelocationRef(DataRefImpl RelocationP, const ObjectFile *Owner);

  bool operator==(const RelocationRef &Other) const;

  void moveNext();

  uint64_t getOffset() const;
  symbol_iterator getSymbol() const;
  uint64_t getType() const;

  /// @brief Get a string that represents the type of this relocation.
  ///
  /// This is for display purposes only.
  void getTypeName(SmallVectorImpl<char> &Result) const;

  DataRefImpl getRawDataRefImpl() const;
  const ObjectFile *getObject() const;
};
typedef content_iterator<RelocationRef> relocation_iterator;

/// This is a value type class that represents a single section in the list of
/// sections in the object file.
class SectionRef {
  friend class SymbolRef;
  DataRefImpl SectionPimpl;
  const ObjectFile *OwningObject;

public:
  SectionRef() : OwningObject(nullptr) { }

  SectionRef(DataRefImpl SectionP, const ObjectFile *Owner);

  bool operator==(const SectionRef &Other) const;
  bool operator!=(const SectionRef &Other) const;
  bool operator<(const SectionRef &Other) const;

  void moveNext();

  std::error_code getName(StringRef &Result) const;
  uint64_t getAddress() const;
  uint64_t getSize() const;
  std::error_code getContents(StringRef &Result) const;

  /// @brief Get the alignment of this section as the actual value (not log 2).
  uint64_t getAlignment() const;

  bool isText() const;
  bool isData() const;
  bool isBSS() const;
  bool isVirtual() const;

  bool containsSymbol(SymbolRef S) const;

  relocation_iterator relocation_begin() const;
  relocation_iterator relocation_end() const;
  iterator_range<relocation_iterator> relocations() const {
    return iterator_range<relocation_iterator>(relocation_begin(),
                                               relocation_end());
  }
  section_iterator getRelocatedSection() const;

  DataRefImpl getRawDataRefImpl() const;
  const ObjectFile *getObject() const;
};

/// This is a value type class that represents a single symbol in the list of
/// symbols in the object file.
class SymbolRef : public BasicSymbolRef {
  friend class SectionRef;

public:
  SymbolRef() : BasicSymbolRef() {}

  enum Type {
    ST_Unknown, // Type not specified
    ST_Data,
    ST_Debug,
    ST_File,
    ST_Function,
    ST_Other
  };

  SymbolRef(DataRefImpl SymbolP, const ObjectFile *Owner);
  SymbolRef(const BasicSymbolRef &B) : BasicSymbolRef(B) {
    assert(isa<ObjectFile>(BasicSymbolRef::getObject()));
  }

  ErrorOr<StringRef> getName() const;
  /// Returns the symbol virtual address (i.e. address at which it will be
  /// mapped).
  ErrorOr<uint64_t> getAddress() const;

  /// Return the value of the symbol depending on the object this can be an
  /// offset or a virtual address.
  uint64_t getValue() const;

  /// @brief Get the alignment of this symbol as the actual value (not log 2).
  uint32_t getAlignment() const;
  uint64_t getCommonSize() const;
  SymbolRef::Type getType() const;

  /// @brief Get section this symbol is defined in reference to. Result is
  /// end_sections() if it is undefined or is an absolute symbol.
  std::error_code getSection(section_iterator &Result) const;

  const ObjectFile *getObject() const;
};

class symbol_iterator : public basic_symbol_iterator {
public:
  symbol_iterator(SymbolRef Sym) : basic_symbol_iterator(Sym) {}
  symbol_iterator(const basic_symbol_iterator &B)
      : basic_symbol_iterator(SymbolRef(B->getRawDataRefImpl(),
                                        cast<ObjectFile>(B->getObject()))) {}

  const SymbolRef *operator->() const {
    const BasicSymbolRef &P = basic_symbol_iterator::operator *();
    return static_cast<const SymbolRef*>(&P);
  }

  const SymbolRef &operator*() const {
    const BasicSymbolRef &P = basic_symbol_iterator::operator *();
    return static_cast<const SymbolRef&>(P);
  }
};

/// This class is the base class for all object file types. Concrete instances
/// of this object are created by createObjectFile, which figures out which type
/// to create.
class ObjectFile : public SymbolicFile {
  virtual void anchor();
  ObjectFile() = delete;
  ObjectFile(const ObjectFile &other) = delete;

protected:
  ObjectFile(unsigned int Type, MemoryBufferRef Source);

  const uint8_t *base() const {
    return reinterpret_cast<const uint8_t *>(Data.getBufferStart());
  }

  // These functions are for SymbolRef to call internally. The main goal of
  // this is to allow SymbolRef::SymbolPimpl to point directly to the symbol
  // entry in the memory mapped object file. SymbolPimpl cannot contain any
  // virtual functions because then it could not point into the memory mapped
  // file.
  //
  // Implementations assume that the DataRefImpl is valid and has not been
  // modified externally. It's UB otherwise.
  friend class SymbolRef;
  virtual ErrorOr<StringRef> getSymbolName(DataRefImpl Symb) const = 0;
  std::error_code printSymbolName(raw_ostream &OS,
                                  DataRefImpl Symb) const override;
  virtual ErrorOr<uint64_t> getSymbolAddress(DataRefImpl Symb) const = 0;
  virtual uint64_t getSymbolValueImpl(DataRefImpl Symb) const = 0;
  virtual uint32_t getSymbolAlignment(DataRefImpl Symb) const;
  virtual uint64_t getCommonSymbolSizeImpl(DataRefImpl Symb) const = 0;
  virtual SymbolRef::Type getSymbolType(DataRefImpl Symb) const = 0;
  virtual std::error_code getSymbolSection(DataRefImpl Symb,
                                           section_iterator &Res) const = 0;

  // Same as above for SectionRef.
  friend class SectionRef;
  virtual void moveSectionNext(DataRefImpl &Sec) const = 0;
  virtual std::error_code getSectionName(DataRefImpl Sec,
                                         StringRef &Res) const = 0;
  virtual uint64_t getSectionAddress(DataRefImpl Sec) const = 0;
  virtual uint64_t getSectionSize(DataRefImpl Sec) const = 0;
  virtual std::error_code getSectionContents(DataRefImpl Sec,
                                             StringRef &Res) const = 0;
  virtual uint64_t getSectionAlignment(DataRefImpl Sec) const = 0;
  virtual bool isSectionText(DataRefImpl Sec) const = 0;
  virtual bool isSectionData(DataRefImpl Sec) const = 0;
  virtual bool isSectionBSS(DataRefImpl Sec) const = 0;
  // A section is 'virtual' if its contents aren't present in the object image.
  virtual bool isSectionVirtual(DataRefImpl Sec) const = 0;
  virtual relocation_iterator section_rel_begin(DataRefImpl Sec) const = 0;
  virtual relocation_iterator section_rel_end(DataRefImpl Sec) const = 0;
  virtual section_iterator getRelocatedSection(DataRefImpl Sec) const;

  // Same as above for RelocationRef.
  friend class RelocationRef;
  virtual void moveRelocationNext(DataRefImpl &Rel) const = 0;
  virtual uint64_t getRelocationOffset(DataRefImpl Rel) const = 0;
  virtual symbol_iterator getRelocationSymbol(DataRefImpl Rel) const = 0;
  virtual uint64_t getRelocationType(DataRefImpl Rel) const = 0;
  virtual void getRelocationTypeName(DataRefImpl Rel,
                                     SmallVectorImpl<char> &Result) const = 0;

  uint64_t getSymbolValue(DataRefImpl Symb) const;

public:
  uint64_t getCommonSymbolSize(DataRefImpl Symb) const {
    assert(getSymbolFlags(Symb) & SymbolRef::SF_Common);
    return getCommonSymbolSizeImpl(Symb);
  }

  typedef iterator_range<symbol_iterator> symbol_iterator_range;
  symbol_iterator_range symbols() const {
    return symbol_iterator_range(symbol_begin(), symbol_end());
  }

  virtual section_iterator section_begin() const = 0;
  virtual section_iterator section_end() const = 0;

  typedef iterator_range<section_iterator> section_iterator_range;
  section_iterator_range sections() const {
    return section_iterator_range(section_begin(), section_end());
  }

  /// @brief The number of bytes used to represent an address in this object
  ///        file format.
  virtual uint8_t getBytesInAddress() const = 0;

  virtual StringRef getFileFormatName() const = 0;
  virtual /* Triple::ArchType */ unsigned getArch() const = 0;

  /// Returns platform-specific object flags, if any.
  virtual std::error_code getPlatformFlags(unsigned &Result) const {
    Result = 0;
    return object_error::invalid_file_type;
  }

  /// True if this is a relocatable object (.o/.obj).
  virtual bool isRelocatableObject() const = 0;

  /// @returns Pointer to ObjectFile subclass to handle this type of object.
  /// @param ObjectPath The path to the object file. ObjectPath.isObject must
  ///        return true.
  /// @brief Create ObjectFile from path.
  static ErrorOr<OwningBinary<ObjectFile>>
  createObjectFile(StringRef ObjectPath);

  static ErrorOr<std::unique_ptr<ObjectFile>>
  createObjectFile(MemoryBufferRef Object, sys::fs::file_magic Type);
  static ErrorOr<std::unique_ptr<ObjectFile>>
  createObjectFile(MemoryBufferRef Object) {
    return createObjectFile(Object, sys::fs::file_magic::unknown);
  }


  static inline bool classof(const Binary *v) {
    return v->isObject();
  }

  static ErrorOr<std::unique_ptr<COFFObjectFile>>
  createCOFFObjectFile(MemoryBufferRef Object);

  static ErrorOr<std::unique_ptr<ObjectFile>>
  createELFObjectFile(MemoryBufferRef Object);

  static ErrorOr<std::unique_ptr<MachOObjectFile>>
  createMachOObjectFile(MemoryBufferRef Object);
};

// Inline function definitions.
inline SymbolRef::SymbolRef(DataRefImpl SymbolP, const ObjectFile *Owner)
    : BasicSymbolRef(SymbolP, Owner) {}

inline ErrorOr<StringRef> SymbolRef::getName() const {
  return getObject()->getSymbolName(getRawDataRefImpl());
}

inline ErrorOr<uint64_t> SymbolRef::getAddress() const {
  return getObject()->getSymbolAddress(getRawDataRefImpl());
}

inline uint64_t SymbolRef::getValue() const {
  return getObject()->getSymbolValue(getRawDataRefImpl());
}

inline uint32_t SymbolRef::getAlignment() const {
  return getObject()->getSymbolAlignment(getRawDataRefImpl());
}

inline uint64_t SymbolRef::getCommonSize() const {
  return getObject()->getCommonSymbolSize(getRawDataRefImpl());
}

inline std::error_code SymbolRef::getSection(section_iterator &Result) const {
  return getObject()->getSymbolSection(getRawDataRefImpl(), Result);
}

inline SymbolRef::Type SymbolRef::getType() const {
  return getObject()->getSymbolType(getRawDataRefImpl());
}

inline const ObjectFile *SymbolRef::getObject() const {
  const SymbolicFile *O = BasicSymbolRef::getObject();
  return cast<ObjectFile>(O);
}


/// SectionRef
inline SectionRef::SectionRef(DataRefImpl SectionP,
                              const ObjectFile *Owner)
  : SectionPimpl(SectionP)
  , OwningObject(Owner) {}

inline bool SectionRef::operator==(const SectionRef &Other) const {
  return SectionPimpl == Other.SectionPimpl;
}

inline bool SectionRef::operator!=(const SectionRef &Other) const {
  return SectionPimpl != Other.SectionPimpl;
}

inline bool SectionRef::operator<(const SectionRef &Other) const {
  return SectionPimpl < Other.SectionPimpl;
}

inline void SectionRef::moveNext() {
  return OwningObject->moveSectionNext(SectionPimpl);
}

inline std::error_code SectionRef::getName(StringRef &Result) const {
  return OwningObject->getSectionName(SectionPimpl, Result);
}

inline uint64_t SectionRef::getAddress() const {
  return OwningObject->getSectionAddress(SectionPimpl);
}

inline uint64_t SectionRef::getSize() const {
  return OwningObject->getSectionSize(SectionPimpl);
}

inline std::error_code SectionRef::getContents(StringRef &Result) const {
  return OwningObject->getSectionContents(SectionPimpl, Result);
}

inline uint64_t SectionRef::getAlignment() const {
  return OwningObject->getSectionAlignment(SectionPimpl);
}

inline bool SectionRef::isText() const {
  return OwningObject->isSectionText(SectionPimpl);
}

inline bool SectionRef::isData() const {
  return OwningObject->isSectionData(SectionPimpl);
}

inline bool SectionRef::isBSS() const {
  return OwningObject->isSectionBSS(SectionPimpl);
}

inline bool SectionRef::isVirtual() const {
  return OwningObject->isSectionVirtual(SectionPimpl);
}

inline relocation_iterator SectionRef::relocation_begin() const {
  return OwningObject->section_rel_begin(SectionPimpl);
}

inline relocation_iterator SectionRef::relocation_end() const {
  return OwningObject->section_rel_end(SectionPimpl);
}

inline section_iterator SectionRef::getRelocatedSection() const {
  return OwningObject->getRelocatedSection(SectionPimpl);
}

inline DataRefImpl SectionRef::getRawDataRefImpl() const {
  return SectionPimpl;
}

inline const ObjectFile *SectionRef::getObject() const {
  return OwningObject;
}

/// RelocationRef
inline RelocationRef::RelocationRef(DataRefImpl RelocationP,
                              const ObjectFile *Owner)
  : RelocationPimpl(RelocationP)
  , OwningObject(Owner) {}

inline bool RelocationRef::operator==(const RelocationRef &Other) const {
  return RelocationPimpl == Other.RelocationPimpl;
}

inline void RelocationRef::moveNext() {
  return OwningObject->moveRelocationNext(RelocationPimpl);
}

inline uint64_t RelocationRef::getOffset() const {
  return OwningObject->getRelocationOffset(RelocationPimpl);
}

inline symbol_iterator RelocationRef::getSymbol() const {
  return OwningObject->getRelocationSymbol(RelocationPimpl);
}

inline uint64_t RelocationRef::getType() const {
  return OwningObject->getRelocationType(RelocationPimpl);
}

inline void RelocationRef::getTypeName(SmallVectorImpl<char> &Result) const {
  return OwningObject->getRelocationTypeName(RelocationPimpl, Result);
}

inline DataRefImpl RelocationRef::getRawDataRefImpl() const {
  return RelocationPimpl;
}

inline const ObjectFile *RelocationRef::getObject() const {
  return OwningObject;
}


} // end namespace object
} // end namespace llvm

#endif
