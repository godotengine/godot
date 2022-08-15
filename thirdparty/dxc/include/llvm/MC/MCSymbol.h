//===- MCSymbol.h - Machine Code Symbols ------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MCSymbol class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCSYMBOL_H
#define LLVM_MC_MCSYMBOL_H

#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/MC/MCAssembler.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class MCAsmInfo;
class MCExpr;
class MCSymbol;
class MCFragment;
class MCSection;
class MCContext;
class raw_ostream;

/// MCSymbol - Instances of this class represent a symbol name in the MC file,
/// and MCSymbols are created and uniqued by the MCContext class.  MCSymbols
/// should only be constructed with valid names for the object file.
///
/// If the symbol is defined/emitted into the current translation unit, the
/// Section member is set to indicate what section it lives in.  Otherwise, if
/// it is a reference to an external entity, it has a null section.
class MCSymbol {
protected:
  /// The kind of the symbol.  If it is any value other than unset then this
  /// class is actually one of the appropriate subclasses of MCSymbol.
  enum SymbolKind {
    SymbolKindUnset,
    SymbolKindCOFF,
    SymbolKindELF,
    SymbolKindMachO,
  };

  /// A symbol can contain an Offset, or Value, or be Common, but never more
  /// than one of these.
  enum Contents : uint8_t {
    SymContentsUnset,
    SymContentsOffset,
    SymContentsVariable,
    SymContentsCommon,
  };

  // Special sentinal value for the absolute pseudo section.
  //
  // FIXME: Use a PointerInt wrapper for this?
  static MCSection *AbsolutePseudoSection;

  /// If a symbol has a Fragment, the section is implied, so we only need
  /// one pointer.
  /// FIXME: We might be able to simplify this by having the asm streamer create
  /// dummy fragments.
  /// If this is a section, then it gives the symbol is defined in. This is null
  /// for undefined symbols, and the special AbsolutePseudoSection value for
  /// absolute symbols. If this is a variable symbol, this caches the variable
  /// value's section.
  ///
  /// If this is a fragment, then it gives the fragment this symbol's value is
  /// relative to, if any.
  ///
  /// For the 'HasName' integer, this is true if this symbol is named.
  /// A named symbol will have a pointer to the name allocated in the bytes
  /// immediately prior to the MCSymbol.
  mutable PointerIntPair<PointerUnion<MCSection *, MCFragment *>, 1>
      SectionOrFragmentAndHasName;

  /// IsTemporary - True if this is an assembler temporary label, which
  /// typically does not survive in the .o file's symbol table.  Usually
  /// "Lfoo" or ".foo".
  unsigned IsTemporary : 1;

  /// \brief True if this symbol can be redefined.
  unsigned IsRedefinable : 1;

  /// IsUsed - True if this symbol has been used.
  mutable unsigned IsUsed : 1;

  mutable bool IsRegistered : 1;

  /// This symbol is visible outside this translation unit.
  mutable unsigned IsExternal : 1;

  /// This symbol is private extern.
  mutable unsigned IsPrivateExtern : 1;

  /// LLVM RTTI discriminator. This is actually a SymbolKind enumerator, but is
  /// unsigned to avoid sign extension and achieve better bitpacking with MSVC.
  unsigned Kind : 2;

  /// True if we have created a relocation that uses this symbol.
  mutable unsigned IsUsedInReloc : 1;

  /// This is actually a Contents enumerator, but is unsigned to avoid sign
  /// extension and achieve better bitpacking with MSVC.
  unsigned SymbolContents : 2;

  /// The alignment of the symbol, if it is 'common', or -1.
  ///
  /// The alignment is stored as log2(align) + 1.  This allows all values from
  /// 0 to 2^31 to be stored which is every power of 2 representable by an
  /// unsigned.
  enum : unsigned { NumCommonAlignmentBits = 5 };
  unsigned CommonAlignLog2 : NumCommonAlignmentBits;

  /// The Flags field is used by object file implementations to store
  /// additional per symbol information which is not easily classified.
  enum : unsigned { NumFlagsBits = 16 };
  mutable uint32_t Flags : NumFlagsBits;

  /// Index field, for use by the object file implementation.
  mutable uint32_t Index = 0;

  union {
    /// The offset to apply to the fragment address to form this symbol's value.
    uint64_t Offset;

    /// The size of the symbol, if it is 'common'.
    uint64_t CommonSize;

    /// If non-null, the value for a variable symbol.
    const MCExpr *Value;
  };

protected: // MCContext creates and uniques these.
  friend class MCExpr;
  friend class MCContext;

  /// \brief The name for a symbol.
  /// MCSymbol contains a uint64_t so is probably aligned to 8.  On a 32-bit
  /// system, the name is a pointer so isn't going to satisfy the 8 byte
  /// alignment of uint64_t.  Account for that here.
  typedef union {
    const StringMapEntry<bool> *NameEntry;
    uint64_t AlignmentPadding;
  } NameEntryStorageTy;

  MCSymbol(SymbolKind Kind, const StringMapEntry<bool> *Name, bool isTemporary)
      : IsTemporary(isTemporary), IsRedefinable(false), IsUsed(false),
        IsRegistered(false), IsExternal(false), IsPrivateExtern(false),
        Kind(Kind), IsUsedInReloc(false), SymbolContents(SymContentsUnset),
        CommonAlignLog2(0), Flags(0) {
    Offset = 0;
    SectionOrFragmentAndHasName.setInt(!!Name);
    if (Name)
      getNameEntryPtr() = Name;
  }

  // Provide custom new/delete as we will only allocate space for a name
  // if we need one.
  void *operator new(size_t s, const StringMapEntry<bool> *Name,
                     MCContext &Ctx);

private:

  void operator delete(void *);
  /// \brief Placement delete - required by std, but never called.
  void operator delete(void*, unsigned) {
    llvm_unreachable("Constructor throws?");
  }
  /// \brief Placement delete - required by std, but never called.
  void operator delete(void*, unsigned, bool) {
    llvm_unreachable("Constructor throws?");
  }

  MCSymbol(const MCSymbol &) = delete;
  void operator=(const MCSymbol &) = delete;
  MCSection *getSectionPtr() const {
    if (MCFragment *F = getFragment())
      return F->getParent();
    const auto &SectionOrFragment = SectionOrFragmentAndHasName.getPointer();
    assert(!SectionOrFragment.is<MCFragment *>() && "Section or null expected");
    MCSection *Section = SectionOrFragment.dyn_cast<MCSection *>();
    if (Section || !isVariable())
      return Section;
    return Section = getVariableValue()->findAssociatedSection();
  }

  /// \brief Get a reference to the name field.  Requires that we have a name
  const StringMapEntry<bool> *&getNameEntryPtr() {
    assert(SectionOrFragmentAndHasName.getInt() && "Name is required");
    NameEntryStorageTy *Name = reinterpret_cast<NameEntryStorageTy *>(this);
    return (*(Name - 1)).NameEntry;
  }
  const StringMapEntry<bool> *&getNameEntryPtr() const {
    return const_cast<MCSymbol*>(this)->getNameEntryPtr();
  }

public:
  /// getName - Get the symbol name.
  StringRef getName() const {
    if (!SectionOrFragmentAndHasName.getInt())
      return StringRef();

    return getNameEntryPtr()->first();
  }

  bool isRegistered() const { return IsRegistered; }
  void setIsRegistered(bool Value) const { IsRegistered = Value; }

  void setUsedInReloc() const { IsUsedInReloc = true; }
  bool isUsedInReloc() const { return IsUsedInReloc; }

  /// \name Accessors
  /// @{

  /// isTemporary - Check if this is an assembler temporary symbol.
  bool isTemporary() const { return IsTemporary; }

  /// isUsed - Check if this is used.
  bool isUsed() const { return IsUsed; }
  void setUsed(bool Value) const { IsUsed = Value; }

  /// \brief Check if this symbol is redefinable.
  bool isRedefinable() const { return IsRedefinable; }
  /// \brief Mark this symbol as redefinable.
  void setRedefinable(bool Value) { IsRedefinable = Value; }
  /// \brief Prepare this symbol to be redefined.
  void redefineIfPossible() {
    if (IsRedefinable) {
      if (SymbolContents == SymContentsVariable) {
        Value = nullptr;
        SymbolContents = SymContentsUnset;
      }
      setUndefined();
      IsRedefinable = false;
    }
  }

  /// @}
  /// \name Associated Sections
  /// @{

  /// isDefined - Check if this symbol is defined (i.e., it has an address).
  ///
  /// Defined symbols are either absolute or in some section.
  bool isDefined() const { return getSectionPtr() != nullptr; }

  /// isInSection - Check if this symbol is defined in some section (i.e., it
  /// is defined but not absolute).
  bool isInSection() const { return isDefined() && !isAbsolute(); }

  /// isUndefined - Check if this symbol undefined (i.e., implicitly defined).
  bool isUndefined() const { return !isDefined(); }

  /// isAbsolute - Check if this is an absolute symbol.
  bool isAbsolute() const { return getSectionPtr() == AbsolutePseudoSection; }

  /// Get the section associated with a defined, non-absolute symbol.
  MCSection &getSection() const {
    assert(isInSection() && "Invalid accessor!");
    return *getSectionPtr();
  }

  /// Mark the symbol as defined in the section \p S.
  void setSection(MCSection &S) {
    assert(!isVariable() && "Cannot set section of variable");
    assert(!SectionOrFragmentAndHasName.getPointer().is<MCFragment *>() &&
           "Section or null expected");
    SectionOrFragmentAndHasName.setPointer(&S);
  }

  /// Mark the symbol as undefined.
  void setUndefined() {
    SectionOrFragmentAndHasName.setPointer(
        PointerUnion<MCSection *, MCFragment *>());
  }

  bool isELF() const { return Kind == SymbolKindELF; }

  bool isCOFF() const { return Kind == SymbolKindCOFF; }

  bool isMachO() const { return Kind == SymbolKindMachO; }

  /// @}
  /// \name Variable Symbols
  /// @{

  /// isVariable - Check if this is a variable symbol.
  bool isVariable() const {
    return SymbolContents == SymContentsVariable;
  }

  /// getVariableValue() - Get the value for variable symbols.
  const MCExpr *getVariableValue() const {
    assert(isVariable() && "Invalid accessor!");
    IsUsed = true;
    return Value;
  }

  void setVariableValue(const MCExpr *Value);

  /// @}

  /// Get the (implementation defined) index.
  uint32_t getIndex() const {
    return Index;
  }

  /// Set the (implementation defined) index.
  void setIndex(uint32_t Value) const {
    Index = Value;
  }

  uint64_t getOffset() const {
    assert((SymbolContents == SymContentsUnset ||
            SymbolContents == SymContentsOffset) &&
           "Cannot get offset for a common/variable symbol");
    return Offset;
  }
  void setOffset(uint64_t Value) {
    assert((SymbolContents == SymContentsUnset ||
            SymbolContents == SymContentsOffset) &&
           "Cannot set offset for a common/variable symbol");
    Offset = Value;
    SymbolContents = SymContentsOffset;
  }

  /// Return the size of a 'common' symbol.
  uint64_t getCommonSize() const {
    assert(isCommon() && "Not a 'common' symbol!");
    return CommonSize;
  }

  /// Mark this symbol as being 'common'.
  ///
  /// \param Size - The size of the symbol.
  /// \param Align - The alignment of the symbol.
  void setCommon(uint64_t Size, unsigned Align) {
    assert(getOffset() == 0);
    CommonSize = Size;
    SymbolContents = SymContentsCommon;

    assert((!Align || isPowerOf2_32(Align)) &&
           "Alignment must be a power of 2");
    unsigned Log2Align = Log2_32(Align) + 1;
    assert(Log2Align < (1U << NumCommonAlignmentBits) &&
           "Out of range alignment");
    CommonAlignLog2 = Log2Align;
  }

  ///  Return the alignment of a 'common' symbol.
  unsigned getCommonAlignment() const {
    assert(isCommon() && "Not a 'common' symbol!");
    return CommonAlignLog2 ? (1U << (CommonAlignLog2 - 1)) : 0;
  }

  /// Declare this symbol as being 'common'.
  ///
  /// \param Size - The size of the symbol.
  /// \param Align - The alignment of the symbol.
  /// \return True if symbol was already declared as a different type
  bool declareCommon(uint64_t Size, unsigned Align) {
    assert(isCommon() || getOffset() == 0);
    if(isCommon()) {
      if(CommonSize != Size || getCommonAlignment() != Align)
       return true;
    } else
      setCommon(Size, Align);
    return false;
  }

  /// Is this a 'common' symbol.
  bool isCommon() const {
    return SymbolContents == SymContentsCommon;
  }

  MCFragment *getFragment() const {
    return SectionOrFragmentAndHasName.getPointer().dyn_cast<MCFragment *>();
  }
  void setFragment(MCFragment *Value) const {
    SectionOrFragmentAndHasName.setPointer(Value);
  }

  bool isExternal() const { return IsExternal; }
  void setExternal(bool Value) const { IsExternal = Value; }

  bool isPrivateExtern() const { return IsPrivateExtern; }
  void setPrivateExtern(bool Value) { IsPrivateExtern = Value; }

  /// print - Print the value to the stream \p OS.
  void print(raw_ostream &OS, const MCAsmInfo *MAI) const;

  /// dump - Print the value to stderr.
  void dump() const;

protected:
  /// Get the (implementation defined) symbol flags.
  uint32_t getFlags() const { return Flags; }

  /// Set the (implementation defined) symbol flags.
  void setFlags(uint32_t Value) const {
    assert(Value < (1U << NumFlagsBits) && "Out of range flags");
    Flags = Value;
  }

  /// Modify the flags via a mask
  void modifyFlags(uint32_t Value, uint32_t Mask) const {
    assert(Value < (1U << NumFlagsBits) && "Out of range flags");
    Flags = (Flags & ~Mask) | Value;
  }
};

inline raw_ostream &operator<<(raw_ostream &OS, const MCSymbol &Sym) {
  Sym.print(OS, nullptr);
  return OS;
}
} // end namespace llvm

#endif
