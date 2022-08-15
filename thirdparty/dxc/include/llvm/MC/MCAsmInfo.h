//===-- llvm/MC/MCAsmInfo.h - Asm info --------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains a class to be used as the basis for target specific
// asm writers.  This class primarily takes care of global printing constants,
// which are used in very similar ways across all targets.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_MC_MCASMINFO_H
#define LLVM_MC_MCASMINFO_H

#include "llvm/MC/MCDirectives.h"
#include "llvm/MC/MCDwarf.h"
#include <cassert>
#include <vector>

namespace llvm {
class MCExpr;
class MCSection;
class MCStreamer;
class MCSymbol;
class MCContext;

namespace WinEH {
enum class EncodingType {
  Invalid, /// Invalid
  Alpha,   /// Windows Alpha
  Alpha64, /// Windows AXP64
  ARM,     /// Windows NT (Windows on ARM)
  CE,      /// Windows CE ARM, PowerPC, SH3, SH4
  Itanium, /// Windows x64, Windows Itanium (IA-64)
  X86,     /// Windows x86, uses no CFI, just EH tables
  MIPS = Alpha,
};
}

enum class ExceptionHandling {
  None,     /// No exception support
  DwarfCFI, /// DWARF-like instruction based exceptions
  SjLj,     /// setjmp/longjmp based exceptions
  ARM,      /// ARM EHABI
  WinEH,    /// Windows Exception Handling
};

namespace LCOMM {
enum LCOMMType { NoAlignment, ByteAlignment, Log2Alignment };
}

/// This class is intended to be used as a base class for asm
/// properties and features specific to the target.
class MCAsmInfo {
protected:
  //===------------------------------------------------------------------===//
  // Properties to be set by the target writer, used to configure asm printer.
  //

  /// Pointer size in bytes.  Default is 4.
  unsigned PointerSize;

  /// Size of the stack slot reserved for callee-saved registers, in bytes.
  /// Default is same as pointer size.
  unsigned CalleeSaveStackSlotSize;

  /// True if target is little endian.  Default is true.
  bool IsLittleEndian;

  /// True if target stack grow up.  Default is false.
  bool StackGrowsUp;

  /// True if this target has the MachO .subsections_via_symbols directive.
  /// Default is false.
  bool HasSubsectionsViaSymbols;

  /// True if this is a MachO target that supports the macho-specific .zerofill
  /// directive for emitting BSS Symbols.  Default is false.
  bool HasMachoZeroFillDirective;

  /// True if this is a MachO target that supports the macho-specific .tbss
  /// directive for emitting thread local BSS Symbols.  Default is false.
  bool HasMachoTBSSDirective;

  /// True if the compiler should emit a ".reference .constructors_used" or
  /// ".reference .destructors_used" directive after the static ctor/dtor
  /// list.  This directive is only emitted in Static relocation model.  Default
  /// is false.
  bool HasStaticCtorDtorReferenceInStaticMode;

  /// This is the maximum possible length of an instruction, which is needed to
  /// compute the size of an inline asm.  Defaults to 4.
  unsigned MaxInstLength;

  /// Every possible instruction length is a multiple of this value.  Factored
  /// out in .debug_frame and .debug_line.  Defaults to 1.
  unsigned MinInstAlignment;

  /// The '$' token, when not referencing an identifier or constant, refers to
  /// the current PC.  Defaults to false.
  bool DollarIsPC;

  /// This string, if specified, is used to separate instructions from each
  /// other when on the same line.  Defaults to ';'
  const char *SeparatorString;

  /// This indicates the comment character used by the assembler.  Defaults to
  /// "#"
  const char *CommentString;

  /// This is appended to emitted labels.  Defaults to ":"
  const char *LabelSuffix;

  // Print the EH begin symbol with an assignment. Defaults to false.
  bool UseAssignmentForEHBegin;

  // Do we need to create a local symbol for .size?
  bool NeedsLocalForSize;

  /// This prefix is used for globals like constant pool entries that are
  /// completely private to the .s file and should not have names in the .o
  /// file.  Defaults to "L"
  const char *PrivateGlobalPrefix;

  /// This prefix is used for labels for basic blocks. Defaults to the same as
  /// PrivateGlobalPrefix.
  const char *PrivateLabelPrefix;

  /// This prefix is used for symbols that should be passed through the
  /// assembler but be removed by the linker.  This is 'l' on Darwin, currently
  /// used for some ObjC metadata.  The default of "" meast that for this system
  /// a plain private symbol should be used.  Defaults to "".
  const char *LinkerPrivateGlobalPrefix;

  /// If these are nonempty, they contain a directive to emit before and after
  /// an inline assembly statement.  Defaults to "#APP\n", "#NO_APP\n"
  const char *InlineAsmStart;
  const char *InlineAsmEnd;

  /// These are assembly directives that tells the assembler to interpret the
  /// following instructions differently.  Defaults to ".code16", ".code32",
  /// ".code64".
  const char *Code16Directive;
  const char *Code32Directive;
  const char *Code64Directive;

  /// Which dialect of an assembler variant to use.  Defaults to 0
  unsigned AssemblerDialect;

  /// This is true if the assembler allows @ characters in symbol names.
  /// Defaults to false.
  bool AllowAtInName;

  /// If this is true, symbol names with invalid characters will be printed in
  /// quotes.
  bool SupportsQuotedNames;

  /// This is true if data region markers should be printed as
  /// ".data_region/.end_data_region" directives. If false, use "$d/$a" labels
  /// instead.
  bool UseDataRegionDirectives;

  //===--- Data Emission Directives -------------------------------------===//

  /// This should be set to the directive used to get some number of zero bytes
  /// emitted to the current section.  Common cases are "\t.zero\t" and
  /// "\t.space\t".  If this is set to null, the Data*bitsDirective's will be
  /// used to emit zero bytes.  Defaults to "\t.zero\t"
  const char *ZeroDirective;

  /// This directive allows emission of an ascii string with the standard C
  /// escape characters embedded into it.  Defaults to "\t.ascii\t"
  const char *AsciiDirective;

  /// If not null, this allows for special handling of zero terminated strings
  /// on this target.  This is commonly supported as ".asciz".  If a target
  /// doesn't support this, it can be set to null.  Defaults to "\t.asciz\t"
  const char *AscizDirective;

  /// These directives are used to output some unit of integer data to the
  /// current section.  If a data directive is set to null, smaller data
  /// directives will be used to emit the large sizes.  Defaults to "\t.byte\t",
  /// "\t.short\t", "\t.long\t", "\t.quad\t"
  const char *Data8bitsDirective;
  const char *Data16bitsDirective;
  const char *Data32bitsDirective;
  const char *Data64bitsDirective;

  /// If non-null, a directive that is used to emit a word which should be
  /// relocated as a 64-bit GP-relative offset, e.g. .gpdword on Mips.  Defaults
  /// to NULL.
  const char *GPRel64Directive;

  /// If non-null, a directive that is used to emit a word which should be
  /// relocated as a 32-bit GP-relative offset, e.g. .gpword on Mips or .gprel32
  /// on Alpha.  Defaults to NULL.
  const char *GPRel32Directive;

  /// This is true if this target uses "Sun Style" syntax for section switching
  /// ("#alloc,#write" etc) instead of the normal ELF syntax (,"a,w") in
  /// .section directives.  Defaults to false.
  bool SunStyleELFSectionSwitchSyntax;

  /// This is true if this target uses ELF '.section' directive before the
  /// '.bss' one. It's used for PPC/Linux which doesn't support the '.bss'
  /// directive only.  Defaults to false.
  bool UsesELFSectionDirectiveForBSS;

  bool NeedsDwarfSectionOffsetDirective;

  //===--- Alignment Information ----------------------------------------===//

  /// If this is true (the default) then the asmprinter emits ".align N"
  /// directives, where N is the number of bytes to align to.  Otherwise, it
  /// emits ".align log2(N)", e.g. 3 to align to an 8 byte boundary.  Defaults
  /// to true.
  bool AlignmentIsInBytes;

  /// If non-zero, this is used to fill the executable space created as the
  /// result of a alignment directive.  Defaults to 0
  unsigned TextAlignFillValue;

  //===--- Global Variable Emission Directives --------------------------===//

  /// This is the directive used to declare a global entity. Defaults to
  /// ".globl".
  const char *GlobalDirective;

  /// True if the expression
  ///   .long f - g
  /// uses a relocation but it can be suppressed by writing
  ///   a = f - g
  ///   .long a
  bool SetDirectiveSuppressesReloc;

  /// False if the assembler requires that we use
  /// \code
  ///   Lc = a - b
  ///   .long Lc
  /// \endcode
  //
  /// instead of
  //
  /// \code
  ///   .long a - b
  /// \endcode
  ///
  ///  Defaults to true.
  bool HasAggressiveSymbolFolding;

  /// True is .comm's and .lcomms optional alignment is to be specified in bytes
  /// instead of log2(n).  Defaults to true.
  bool COMMDirectiveAlignmentIsInBytes;

  /// Describes if the .lcomm directive for the target supports an alignment
  /// argument and how it is interpreted.  Defaults to NoAlignment.
  LCOMM::LCOMMType LCOMMDirectiveAlignmentType;

  // True if the target allows .align directives on functions. This is true for
  // most targets, so defaults to true.
  bool HasFunctionAlignment;

  /// True if the target has .type and .size directives, this is true for most
  /// ELF targets.  Defaults to true.
  bool HasDotTypeDotSizeDirective;

  /// True if the target has a single parameter .file directive, this is true
  /// for ELF targets.  Defaults to true.
  bool HasSingleParameterDotFile;

  /// True if the target has a .ident directive, this is true for ELF targets.
  /// Defaults to false.
  bool HasIdentDirective;

  /// True if this target supports the MachO .no_dead_strip directive.  Defaults
  /// to false.
  bool HasNoDeadStrip;

  /// Used to declare a global as being a weak symbol. Defaults to ".weak".
  const char *WeakDirective;

  /// This directive, if non-null, is used to declare a global as being a weak
  /// undefined symbol.  Defaults to NULL.
  const char *WeakRefDirective;

  /// True if we have a directive to declare a global as being a weak defined
  /// symbol.  Defaults to false.
  bool HasWeakDefDirective;

  /// True if we have a directive to declare a global as being a weak defined
  /// symbol that can be hidden (unexported).  Defaults to false.
  bool HasWeakDefCanBeHiddenDirective;

  /// True if we have a .linkonce directive.  This is used on cygwin/mingw.
  /// Defaults to false.
  bool HasLinkOnceDirective;

  /// This attribute, if not MCSA_Invalid, is used to declare a symbol as having
  /// hidden visibility.  Defaults to MCSA_Hidden.
  MCSymbolAttr HiddenVisibilityAttr;

  /// This attribute, if not MCSA_Invalid, is used to declare an undefined
  /// symbol as having hidden visibility. Defaults to MCSA_Hidden.
  MCSymbolAttr HiddenDeclarationVisibilityAttr;

  /// This attribute, if not MCSA_Invalid, is used to declare a symbol as having
  /// protected visibility.  Defaults to MCSA_Protected
  MCSymbolAttr ProtectedVisibilityAttr;

  //===--- Dwarf Emission Directives -----------------------------------===//

  /// True if target supports emission of debugging information.  Defaults to
  /// false.
  bool SupportsDebugInformation;

  /// Exception handling format for the target.  Defaults to None.
  ExceptionHandling ExceptionsType;

  /// Windows exception handling data (.pdata) encoding.  Defaults to Invalid.
  WinEH::EncodingType WinEHEncodingType;

  /// True if Dwarf2 output generally uses relocations for references to other
  /// .debug_* sections.
  bool DwarfUsesRelocationsAcrossSections;

  /// True if DWARF FDE symbol reference relocations should be replaced by an
  /// absolute difference.
  bool DwarfFDESymbolsUseAbsDiff;

  /// True if dwarf register numbers are printed instead of symbolic register
  /// names in .cfi_* directives.  Defaults to false.
  bool DwarfRegNumForCFI;

  /// True if target uses parens to indicate the symbol variant instead of @.
  /// For example, foo(plt) instead of foo@plt.  Defaults to false.
  bool UseParensForSymbolVariant;

  //===--- Prologue State ----------------------------------------------===//

  std::vector<MCCFIInstruction> InitialFrameState;

  //===--- Integrated Assembler Information ----------------------------===//

  /// Should we use the integrated assembler?
  /// The integrated assembler should be enabled by default (by the
  /// constructors) when failing to parse a valid piece of assembly (inline
  /// or otherwise) is considered a bug. It may then be overridden after
  /// construction (see LLVMTargetMachine::initAsmInfo()).
  bool UseIntegratedAssembler;

  /// Compress DWARF debug sections. Defaults to false.
  bool CompressDebugSections;

  /// True if the integrated assembler should interpret 'a >> b' constant
  /// expressions as logical rather than arithmetic.
  bool UseLogicalShr;

public:
  explicit MCAsmInfo();
  virtual ~MCAsmInfo();

  /// Get the pointer size in bytes.
  unsigned getPointerSize() const { return PointerSize; }

  /// Get the callee-saved register stack slot
  /// size in bytes.
  unsigned getCalleeSaveStackSlotSize() const {
    return CalleeSaveStackSlotSize;
  }

  /// True if the target is little endian.
  bool isLittleEndian() const { return IsLittleEndian; }

  /// True if target stack grow up.
  bool isStackGrowthDirectionUp() const { return StackGrowsUp; }

  bool hasSubsectionsViaSymbols() const { return HasSubsectionsViaSymbols; }

  // Data directive accessors.

  const char *getData8bitsDirective() const { return Data8bitsDirective; }
  const char *getData16bitsDirective() const { return Data16bitsDirective; }
  const char *getData32bitsDirective() const { return Data32bitsDirective; }
  const char *getData64bitsDirective() const { return Data64bitsDirective; }
  const char *getGPRel64Directive() const { return GPRel64Directive; }
  const char *getGPRel32Directive() const { return GPRel32Directive; }

  /// Targets can implement this method to specify a section to switch to if the
  /// translation unit doesn't have any trampolines that require an executable
  /// stack.
  virtual MCSection *getNonexecutableStackSection(MCContext &Ctx) const {
    return nullptr;
  }

  /// \brief True if the section is atomized using the symbols in it.
  /// This is false if the section is not atomized at all (most ELF sections) or
  /// if it is atomized based on its contents (MachO' __TEXT,__cstring for
  /// example).
  virtual bool isSectionAtomizableBySymbols(const MCSection &Section) const;

  virtual const MCExpr *getExprForPersonalitySymbol(const MCSymbol *Sym,
                                                    unsigned Encoding,
                                                    MCStreamer &Streamer) const;

  virtual const MCExpr *getExprForFDESymbol(const MCSymbol *Sym,
                                            unsigned Encoding,
                                            MCStreamer &Streamer) const;

  /// Return true if the identifier \p Name does not need quotes to be
  /// syntactically correct.
  virtual bool isValidUnquotedName(StringRef Name) const;

  bool usesSunStyleELFSectionSwitchSyntax() const {
    return SunStyleELFSectionSwitchSyntax;
  }

  bool usesELFSectionDirectiveForBSS() const {
    return UsesELFSectionDirectiveForBSS;
  }

  bool needsDwarfSectionOffsetDirective() const {
    return NeedsDwarfSectionOffsetDirective;
  }

  // Accessors.

  bool hasMachoZeroFillDirective() const { return HasMachoZeroFillDirective; }
  bool hasMachoTBSSDirective() const { return HasMachoTBSSDirective; }
  bool hasStaticCtorDtorReferenceInStaticMode() const {
    return HasStaticCtorDtorReferenceInStaticMode;
  }
  unsigned getMaxInstLength() const { return MaxInstLength; }
  unsigned getMinInstAlignment() const { return MinInstAlignment; }
  bool getDollarIsPC() const { return DollarIsPC; }
  const char *getSeparatorString() const { return SeparatorString; }

  /// This indicates the column (zero-based) at which asm comments should be
  /// printed.
  unsigned getCommentColumn() const { return 40; }

  const char *getCommentString() const { return CommentString; }
  const char *getLabelSuffix() const { return LabelSuffix; }

  bool useAssignmentForEHBegin() const { return UseAssignmentForEHBegin; }
  bool needsLocalForSize() const { return NeedsLocalForSize; }
  const char *getPrivateGlobalPrefix() const { return PrivateGlobalPrefix; }
  const char *getPrivateLabelPrefix() const { return PrivateLabelPrefix; }
  bool hasLinkerPrivateGlobalPrefix() const {
    return LinkerPrivateGlobalPrefix[0] != '\0';
  }
  const char *getLinkerPrivateGlobalPrefix() const {
    if (hasLinkerPrivateGlobalPrefix())
      return LinkerPrivateGlobalPrefix;
    return getPrivateGlobalPrefix();
  }
  const char *getInlineAsmStart() const { return InlineAsmStart; }
  const char *getInlineAsmEnd() const { return InlineAsmEnd; }
  const char *getCode16Directive() const { return Code16Directive; }
  const char *getCode32Directive() const { return Code32Directive; }
  const char *getCode64Directive() const { return Code64Directive; }
  unsigned getAssemblerDialect() const { return AssemblerDialect; }
  bool doesAllowAtInName() const { return AllowAtInName; }
  bool supportsNameQuoting() const { return SupportsQuotedNames; }
  bool doesSupportDataRegionDirectives() const {
    return UseDataRegionDirectives;
  }
  const char *getZeroDirective() const { return ZeroDirective; }
  const char *getAsciiDirective() const { return AsciiDirective; }
  const char *getAscizDirective() const { return AscizDirective; }
  bool getAlignmentIsInBytes() const { return AlignmentIsInBytes; }
  unsigned getTextAlignFillValue() const { return TextAlignFillValue; }
  const char *getGlobalDirective() const { return GlobalDirective; }
  bool doesSetDirectiveSuppressesReloc() const {
    return SetDirectiveSuppressesReloc;
  }
  bool hasAggressiveSymbolFolding() const { return HasAggressiveSymbolFolding; }
  bool getCOMMDirectiveAlignmentIsInBytes() const {
    return COMMDirectiveAlignmentIsInBytes;
  }
  LCOMM::LCOMMType getLCOMMDirectiveAlignmentType() const {
    return LCOMMDirectiveAlignmentType;
  }
  bool hasFunctionAlignment() const { return HasFunctionAlignment; }
  bool hasDotTypeDotSizeDirective() const { return HasDotTypeDotSizeDirective; }
  bool hasSingleParameterDotFile() const { return HasSingleParameterDotFile; }
  bool hasIdentDirective() const { return HasIdentDirective; }
  bool hasNoDeadStrip() const { return HasNoDeadStrip; }
  const char *getWeakDirective() const { return WeakDirective; }
  const char *getWeakRefDirective() const { return WeakRefDirective; }
  bool hasWeakDefDirective() const { return HasWeakDefDirective; }
  bool hasWeakDefCanBeHiddenDirective() const {
    return HasWeakDefCanBeHiddenDirective;
  }
  bool hasLinkOnceDirective() const { return HasLinkOnceDirective; }

  MCSymbolAttr getHiddenVisibilityAttr() const { return HiddenVisibilityAttr; }
  MCSymbolAttr getHiddenDeclarationVisibilityAttr() const {
    return HiddenDeclarationVisibilityAttr;
  }
  MCSymbolAttr getProtectedVisibilityAttr() const {
    return ProtectedVisibilityAttr;
  }
  bool doesSupportDebugInformation() const { return SupportsDebugInformation; }
  bool doesSupportExceptionHandling() const {
    return ExceptionsType != ExceptionHandling::None;
  }
  ExceptionHandling getExceptionHandlingType() const { return ExceptionsType; }
  WinEH::EncodingType getWinEHEncodingType() const { return WinEHEncodingType; }

  /// Returns true if the exception handling method for the platform uses call
  /// frame information to unwind.
  bool usesCFIForEH() const {
    return (ExceptionsType == ExceptionHandling::DwarfCFI ||
            ExceptionsType == ExceptionHandling::ARM || usesWindowsCFI());
  }

  bool usesWindowsCFI() const {
    return ExceptionsType == ExceptionHandling::WinEH &&
           (WinEHEncodingType != WinEH::EncodingType::Invalid &&
            WinEHEncodingType != WinEH::EncodingType::X86);
  }

  bool doesDwarfUseRelocationsAcrossSections() const {
    return DwarfUsesRelocationsAcrossSections;
  }
  bool doDwarfFDESymbolsUseAbsDiff() const { return DwarfFDESymbolsUseAbsDiff; }
  bool useDwarfRegNumForCFI() const { return DwarfRegNumForCFI; }
  bool useParensForSymbolVariant() const { return UseParensForSymbolVariant; }

  void addInitialFrameState(const MCCFIInstruction &Inst) {
    InitialFrameState.push_back(Inst);
  }

  const std::vector<MCCFIInstruction> &getInitialFrameState() const {
    return InitialFrameState;
  }

  /// Return true if assembly (inline or otherwise) should be parsed.
  bool useIntegratedAssembler() const { return UseIntegratedAssembler; }

  /// Set whether assembly (inline or otherwise) should be parsed.
  virtual void setUseIntegratedAssembler(bool Value) {
    UseIntegratedAssembler = Value;
  }

  bool compressDebugSections() const { return CompressDebugSections; }

  void setCompressDebugSections(bool CompressDebugSections) {
    this->CompressDebugSections = CompressDebugSections;
  }

  bool shouldUseLogicalShr() const { return UseLogicalShr; }
};
}

#endif
