//===- lib/MC/MCAssembler.cpp - Assembler Backend Implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/MCAssembler.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCAsmLayout.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCDwarf.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCFixupKindInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LEB128.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
#include <tuple>
using namespace llvm;

#define DEBUG_TYPE "assembler"

namespace {
namespace stats {
STATISTIC(EmittedFragments, "Number of emitted assembler fragments - total");
STATISTIC(EmittedRelaxableFragments,
          "Number of emitted assembler fragments - relaxable");
STATISTIC(EmittedDataFragments,
          "Number of emitted assembler fragments - data");
STATISTIC(EmittedCompactEncodedInstFragments,
          "Number of emitted assembler fragments - compact encoded inst");
STATISTIC(EmittedAlignFragments,
          "Number of emitted assembler fragments - align");
STATISTIC(EmittedFillFragments,
          "Number of emitted assembler fragments - fill");
STATISTIC(EmittedOrgFragments,
          "Number of emitted assembler fragments - org");
STATISTIC(evaluateFixup, "Number of evaluated fixups");
STATISTIC(FragmentLayouts, "Number of fragment layouts");
STATISTIC(ObjectBytes, "Number of emitted object file bytes");
STATISTIC(RelaxationSteps, "Number of assembler layout and relaxation steps");
STATISTIC(RelaxedInstructions, "Number of relaxed instructions");
}
}

// FIXME FIXME FIXME: There are number of places in this file where we convert
// what is a 64-bit assembler value used for computation into a value in the
// object file, which may truncate it. We should detect that truncation where
// invalid and report errors back.

/* *** */

MCAsmLayout::MCAsmLayout(MCAssembler &Asm)
  : Assembler(Asm), LastValidFragment()
 {
  // Compute the section layout order. Virtual sections must go last.
  for (MCAssembler::iterator it = Asm.begin(), ie = Asm.end(); it != ie; ++it)
    if (!it->isVirtualSection())
      SectionOrder.push_back(&*it);
  for (MCAssembler::iterator it = Asm.begin(), ie = Asm.end(); it != ie; ++it)
    if (it->isVirtualSection())
      SectionOrder.push_back(&*it);
}

bool MCAsmLayout::isFragmentValid(const MCFragment *F) const {
  const MCSection *Sec = F->getParent();
  const MCFragment *LastValid = LastValidFragment.lookup(Sec);
  if (!LastValid)
    return false;
  assert(LastValid->getParent() == Sec);
  return F->getLayoutOrder() <= LastValid->getLayoutOrder();
}

void MCAsmLayout::invalidateFragmentsFrom(MCFragment *F) {
  // If this fragment wasn't already valid, we don't need to do anything.
  if (!isFragmentValid(F))
    return;

  // Otherwise, reset the last valid fragment to the previous fragment
  // (if this is the first fragment, it will be NULL).
  LastValidFragment[F->getParent()] = F->getPrevNode();
}

void MCAsmLayout::ensureValid(const MCFragment *F) const {
  MCSection *Sec = F->getParent();
  MCFragment *Cur = LastValidFragment[Sec];
  if (!Cur)
    Cur = Sec->begin();
  else
    Cur = Cur->getNextNode();

  // Advance the layout position until the fragment is valid.
  while (!isFragmentValid(F)) {
    assert(Cur && "Layout bookkeeping error");
    const_cast<MCAsmLayout*>(this)->layoutFragment(Cur);
    Cur = Cur->getNextNode();
  }
}

uint64_t MCAsmLayout::getFragmentOffset(const MCFragment *F) const {
  ensureValid(F);
  assert(F->Offset != ~UINT64_C(0) && "Address not set!");
  return F->Offset;
}

// Simple getSymbolOffset helper for the non-varibale case.
static bool getLabelOffset(const MCAsmLayout &Layout, const MCSymbol &S,
                           bool ReportError, uint64_t &Val) {
  if (!S.getFragment()) {
    if (ReportError)
      report_fatal_error("unable to evaluate offset to undefined symbol '" +
                         S.getName() + "'");
    return false;
  }
  Val = Layout.getFragmentOffset(S.getFragment()) + S.getOffset();
  return true;
}

static bool getSymbolOffsetImpl(const MCAsmLayout &Layout, const MCSymbol &S,
                                bool ReportError, uint64_t &Val) {
  if (!S.isVariable())
    return getLabelOffset(Layout, S, ReportError, Val);

  // If SD is a variable, evaluate it.
  MCValue Target;
  if (!S.getVariableValue()->evaluateAsRelocatable(Target, &Layout, nullptr))
    report_fatal_error("unable to evaluate offset for variable '" +
                       S.getName() + "'");

  uint64_t Offset = Target.getConstant();

  const MCSymbolRefExpr *A = Target.getSymA();
  if (A) {
    uint64_t ValA;
    if (!getLabelOffset(Layout, A->getSymbol(), ReportError, ValA))
      return false;
    Offset += ValA;
  }

  const MCSymbolRefExpr *B = Target.getSymB();
  if (B) {
    uint64_t ValB;
    if (!getLabelOffset(Layout, B->getSymbol(), ReportError, ValB))
      return false;
    Offset -= ValB;
  }

  Val = Offset;
  return true;
}

bool MCAsmLayout::getSymbolOffset(const MCSymbol &S, uint64_t &Val) const {
  return getSymbolOffsetImpl(*this, S, false, Val);
}

uint64_t MCAsmLayout::getSymbolOffset(const MCSymbol &S) const {
  uint64_t Val;
  getSymbolOffsetImpl(*this, S, true, Val);
  return Val;
}

const MCSymbol *MCAsmLayout::getBaseSymbol(const MCSymbol &Symbol) const {
  if (!Symbol.isVariable())
    return &Symbol;

  const MCExpr *Expr = Symbol.getVariableValue();
  MCValue Value;
  if (!Expr->evaluateAsValue(Value, *this))
    llvm_unreachable("Invalid Expression");

  const MCSymbolRefExpr *RefB = Value.getSymB();
  if (RefB)
    Assembler.getContext().reportFatalError(
        SMLoc(), Twine("symbol '") + RefB->getSymbol().getName() +
                     "' could not be evaluated in a subtraction expression");

  const MCSymbolRefExpr *A = Value.getSymA();
  if (!A)
    return nullptr;

  const MCSymbol &ASym = A->getSymbol();
  const MCAssembler &Asm = getAssembler();
  if (ASym.isCommon()) {
    // FIXME: we should probably add a SMLoc to MCExpr.
    Asm.getContext().reportFatalError(SMLoc(),
                                "Common symbol " + ASym.getName() +
                                    " cannot be used in assignment expr");
  }

  return &ASym;
}

uint64_t MCAsmLayout::getSectionAddressSize(const MCSection *Sec) const {
  // The size is the last fragment's end offset.
  const MCFragment &F = Sec->getFragmentList().back();
  return getFragmentOffset(&F) + getAssembler().computeFragmentSize(*this, F);
}

uint64_t MCAsmLayout::getSectionFileSize(const MCSection *Sec) const {
  // Virtual sections have no file size.
  if (Sec->isVirtualSection())
    return 0;

  // Otherwise, the file size is the same as the address space size.
  return getSectionAddressSize(Sec);
}

uint64_t llvm::computeBundlePadding(const MCAssembler &Assembler,
                                    const MCFragment *F,
                                    uint64_t FOffset, uint64_t FSize) {
  uint64_t BundleSize = Assembler.getBundleAlignSize();
  assert(BundleSize > 0 &&
         "computeBundlePadding should only be called if bundling is enabled");
  uint64_t BundleMask = BundleSize - 1;
  uint64_t OffsetInBundle = FOffset & BundleMask;
  uint64_t EndOfFragment = OffsetInBundle + FSize;

  // There are two kinds of bundling restrictions:
  //
  // 1) For alignToBundleEnd(), add padding to ensure that the fragment will
  //    *end* on a bundle boundary.
  // 2) Otherwise, check if the fragment would cross a bundle boundary. If it
  //    would, add padding until the end of the bundle so that the fragment
  //    will start in a new one.
  if (F->alignToBundleEnd()) {
    // Three possibilities here:
    //
    // A) The fragment just happens to end at a bundle boundary, so we're good.
    // B) The fragment ends before the current bundle boundary: pad it just
    //    enough to reach the boundary.
    // C) The fragment ends after the current bundle boundary: pad it until it
    //    reaches the end of the next bundle boundary.
    //
    // Note: this code could be made shorter with some modulo trickery, but it's
    // intentionally kept in its more explicit form for simplicity.
    if (EndOfFragment == BundleSize)
      return 0;
    else if (EndOfFragment < BundleSize)
      return BundleSize - EndOfFragment;
    else { // EndOfFragment > BundleSize
      return 2 * BundleSize - EndOfFragment;
    }
  } else if (OffsetInBundle > 0 && EndOfFragment > BundleSize)
    return BundleSize - OffsetInBundle;
  else
    return 0;
}

/* *** */

void ilist_node_traits<MCFragment>::deleteNode(MCFragment *V) {
  V->destroy();
}

MCFragment::MCFragment() : Kind(FragmentType(~0)), HasInstructions(false),
                           AlignToBundleEnd(false), BundlePadding(0) {
}

MCFragment::~MCFragment() { }

MCFragment::MCFragment(FragmentType Kind, bool HasInstructions,
                       uint8_t BundlePadding, MCSection *Parent)
    : Kind(Kind), HasInstructions(HasInstructions), AlignToBundleEnd(false),
      BundlePadding(BundlePadding), Parent(Parent), Atom(nullptr),
      Offset(~UINT64_C(0)) {
  if (Parent)
    Parent->getFragmentList().push_back(this);
}

void MCFragment::destroy() {
  // First check if we are the sentinal.
  if (Kind == FragmentType(~0)) {
    delete this;
    return;
  }

  switch (Kind) {
    case FT_Align:
      delete cast<MCAlignFragment>(this);
      return;
    case FT_Data:
      delete cast<MCDataFragment>(this);
      return;
    case FT_CompactEncodedInst:
      delete cast<MCCompactEncodedInstFragment>(this);
      return;
    case FT_Fill:
      delete cast<MCFillFragment>(this);
      return;
    case FT_Relaxable:
      delete cast<MCRelaxableFragment>(this);
      return;
    case FT_Org:
      delete cast<MCOrgFragment>(this);
      return;
    case FT_Dwarf:
      delete cast<MCDwarfLineAddrFragment>(this);
      return;
    case FT_DwarfFrame:
      delete cast<MCDwarfCallFrameFragment>(this);
      return;
    case FT_LEB:
      delete cast<MCLEBFragment>(this);
      return;
    case FT_SafeSEH:
      delete cast<MCSafeSEHFragment>(this);
      return;
  }
}

/* *** */

MCAssembler::MCAssembler(MCContext &Context_, MCAsmBackend &Backend_,
                         MCCodeEmitter &Emitter_, MCObjectWriter &Writer_,
                         raw_ostream &OS_)
    : Context(Context_), Backend(Backend_), Emitter(Emitter_), Writer(Writer_),
      OS(OS_), BundleAlignSize(0), RelaxAll(false),
      SubsectionsViaSymbols(false), ELFHeaderEFlags(0) {
  VersionMinInfo.Major = 0; // Major version == 0 for "none specified"
}

MCAssembler::~MCAssembler() {
}

void MCAssembler::reset() {
  Sections.clear();
  Symbols.clear();
  IndirectSymbols.clear();
  DataRegions.clear();
  LinkerOptions.clear();
  FileNames.clear();
  ThumbFuncs.clear();
  BundleAlignSize = 0;
  RelaxAll = false;
  SubsectionsViaSymbols = false;
  ELFHeaderEFlags = 0;
  LOHContainer.reset();
  VersionMinInfo.Major = 0;

  // reset objects owned by us
  getBackend().reset();
  getEmitter().reset();
  getWriter().reset();
  getLOHContainer().reset();
}

bool MCAssembler::isThumbFunc(const MCSymbol *Symbol) const {
  if (ThumbFuncs.count(Symbol))
    return true;

  if (!Symbol->isVariable())
    return false;

  // FIXME: It looks like gas supports some cases of the form "foo + 2". It
  // is not clear if that is a bug or a feature.
  const MCExpr *Expr = Symbol->getVariableValue();
  const MCSymbolRefExpr *Ref = dyn_cast<MCSymbolRefExpr>(Expr);
  if (!Ref)
    return false;

  if (Ref->getKind() != MCSymbolRefExpr::VK_None)
    return false;

  const MCSymbol &Sym = Ref->getSymbol();
  if (!isThumbFunc(&Sym))
    return false;

  ThumbFuncs.insert(Symbol); // Cache it.
  return true;
}

bool MCAssembler::isSymbolLinkerVisible(const MCSymbol &Symbol) const {
  // Non-temporary labels should always be visible to the linker.
  if (!Symbol.isTemporary())
    return true;

  // Absolute temporary labels are never visible.
  if (!Symbol.isInSection())
    return false;

  if (Symbol.isUsedInReloc())
    return true;

  return false;
}

const MCSymbol *MCAssembler::getAtom(const MCSymbol &S) const {
  // Linker visible symbols define atoms.
  if (isSymbolLinkerVisible(S))
    return &S;

  // Absolute and undefined symbols have no defining atom.
  if (!S.getFragment())
    return nullptr;

  // Non-linker visible symbols in sections which can't be atomized have no
  // defining atom.
  if (!getContext().getAsmInfo()->isSectionAtomizableBySymbols(
          *S.getFragment()->getParent()))
    return nullptr;

  // Otherwise, return the atom for the containing fragment.
  return S.getFragment()->getAtom();
}

bool MCAssembler::evaluateFixup(const MCAsmLayout &Layout,
                                const MCFixup &Fixup, const MCFragment *DF,
                                MCValue &Target, uint64_t &Value) const {
  ++stats::evaluateFixup;

  // FIXME: This code has some duplication with recordRelocation. We should
  // probably merge the two into a single callback that tries to evaluate a
  // fixup and records a relocation if one is needed.
  const MCExpr *Expr = Fixup.getValue();
  if (!Expr->evaluateAsRelocatable(Target, &Layout, &Fixup))
    getContext().reportFatalError(Fixup.getLoc(), "expected relocatable expression");

  bool IsPCRel = Backend.getFixupKindInfo(
    Fixup.getKind()).Flags & MCFixupKindInfo::FKF_IsPCRel;

  bool IsResolved;
  if (IsPCRel) {
    if (Target.getSymB()) {
      IsResolved = false;
    } else if (!Target.getSymA()) {
      IsResolved = false;
    } else {
      const MCSymbolRefExpr *A = Target.getSymA();
      const MCSymbol &SA = A->getSymbol();
      if (A->getKind() != MCSymbolRefExpr::VK_None || SA.isUndefined()) {
        IsResolved = false;
      } else {
        IsResolved = getWriter().isSymbolRefDifferenceFullyResolvedImpl(
            *this, SA, *DF, false, true);
      }
    }
  } else {
    IsResolved = Target.isAbsolute();
  }

  Value = Target.getConstant();

  if (const MCSymbolRefExpr *A = Target.getSymA()) {
    const MCSymbol &Sym = A->getSymbol();
    if (Sym.isDefined())
      Value += Layout.getSymbolOffset(Sym);
  }
  if (const MCSymbolRefExpr *B = Target.getSymB()) {
    const MCSymbol &Sym = B->getSymbol();
    if (Sym.isDefined())
      Value -= Layout.getSymbolOffset(Sym);
  }


  bool ShouldAlignPC = Backend.getFixupKindInfo(Fixup.getKind()).Flags &
                         MCFixupKindInfo::FKF_IsAlignedDownTo32Bits;
  assert((ShouldAlignPC ? IsPCRel : true) &&
    "FKF_IsAlignedDownTo32Bits is only allowed on PC-relative fixups!");

  if (IsPCRel) {
    uint32_t Offset = Layout.getFragmentOffset(DF) + Fixup.getOffset();

    // A number of ARM fixups in Thumb mode require that the effective PC
    // address be determined as the 32-bit aligned version of the actual offset.
    if (ShouldAlignPC) Offset &= ~0x3;
    Value -= Offset;
  }

  // Let the backend adjust the fixup value if necessary, including whether
  // we need a relocation.
  Backend.processFixupValue(*this, Layout, Fixup, DF, Target, Value,
                            IsResolved);

  return IsResolved;
}

uint64_t MCAssembler::computeFragmentSize(const MCAsmLayout &Layout,
                                          const MCFragment &F) const {
  switch (F.getKind()) {
  case MCFragment::FT_Data:
    return cast<MCDataFragment>(F).getContents().size();
  case MCFragment::FT_Relaxable:
    return cast<MCRelaxableFragment>(F).getContents().size();
  case MCFragment::FT_CompactEncodedInst:
    return cast<MCCompactEncodedInstFragment>(F).getContents().size();
  case MCFragment::FT_Fill:
    return cast<MCFillFragment>(F).getSize();

  case MCFragment::FT_LEB:
    return cast<MCLEBFragment>(F).getContents().size();

  case MCFragment::FT_SafeSEH:
    return 4;

  case MCFragment::FT_Align: {
    const MCAlignFragment &AF = cast<MCAlignFragment>(F);
    unsigned Offset = Layout.getFragmentOffset(&AF);
    unsigned Size = OffsetToAlignment(Offset, AF.getAlignment());
    // If we are padding with nops, force the padding to be larger than the
    // minimum nop size.
    if (Size > 0 && AF.hasEmitNops()) {
      while (Size % getBackend().getMinimumNopSize())
        Size += AF.getAlignment();
    }
    if (Size > AF.getMaxBytesToEmit())
      return 0;
    return Size;
  }

  case MCFragment::FT_Org: {
    const MCOrgFragment &OF = cast<MCOrgFragment>(F);
    int64_t TargetLocation;
    if (!OF.getOffset().evaluateAsAbsolute(TargetLocation, Layout))
      report_fatal_error("expected assembly-time absolute expression");

    // FIXME: We need a way to communicate this error.
    uint64_t FragmentOffset = Layout.getFragmentOffset(&OF);
    int64_t Size = TargetLocation - FragmentOffset;
    if (Size < 0 || Size >= 0x40000000)
      report_fatal_error("invalid .org offset '" + Twine(TargetLocation) +
                         "' (at offset '" + Twine(FragmentOffset) + "')");
    return Size;
  }

  case MCFragment::FT_Dwarf:
    return cast<MCDwarfLineAddrFragment>(F).getContents().size();
  case MCFragment::FT_DwarfFrame:
    return cast<MCDwarfCallFrameFragment>(F).getContents().size();
  }

  llvm_unreachable("invalid fragment kind");
}

void MCAsmLayout::layoutFragment(MCFragment *F) {
  MCFragment *Prev = F->getPrevNode();

  // We should never try to recompute something which is valid.
  assert(!isFragmentValid(F) && "Attempt to recompute a valid fragment!");
  // We should never try to compute the fragment layout if its predecessor
  // isn't valid.
  assert((!Prev || isFragmentValid(Prev)) &&
         "Attempt to compute fragment before its predecessor!");

  ++stats::FragmentLayouts;

  // Compute fragment offset and size.
  if (Prev)
    F->Offset = Prev->Offset + getAssembler().computeFragmentSize(*this, *Prev);
  else
    F->Offset = 0;
  LastValidFragment[F->getParent()] = F;

  // If bundling is enabled and this fragment has instructions in it, it has to
  // obey the bundling restrictions. With padding, we'll have:
  //
  //
  //        BundlePadding
  //             |||
  // -------------------------------------
  //   Prev  |##########|       F        |
  // -------------------------------------
  //                    ^
  //                    |
  //                    F->Offset
  //
  // The fragment's offset will point to after the padding, and its computed
  // size won't include the padding.
  //
  // When the -mc-relax-all flag is used, we optimize bundling by writting the
  // padding directly into fragments when the instructions are emitted inside
  // the streamer. When the fragment is larger than the bundle size, we need to
  // ensure that it's bundle aligned. This means that if we end up with
  // multiple fragments, we must emit bundle padding between fragments.
  //
  // ".align N" is an example of a directive that introduces multiple
  // fragments. We could add a special case to handle ".align N" by emitting
  // within-fragment padding (which would produce less padding when N is less
  // than the bundle size), but for now we don't.
  //
  if (Assembler.isBundlingEnabled() && F->hasInstructions()) {
    assert(isa<MCEncodedFragment>(F) &&
           "Only MCEncodedFragment implementations have instructions");
    uint64_t FSize = Assembler.computeFragmentSize(*this, *F);

    if (!Assembler.getRelaxAll() && FSize > Assembler.getBundleAlignSize())
      report_fatal_error("Fragment can't be larger than a bundle size");

    uint64_t RequiredBundlePadding = computeBundlePadding(Assembler, F,
                                                          F->Offset, FSize);
    if (RequiredBundlePadding > UINT8_MAX)
      report_fatal_error("Padding cannot exceed 255 bytes");
    F->setBundlePadding(static_cast<uint8_t>(RequiredBundlePadding));
    F->Offset += RequiredBundlePadding;
  }
}

void MCAssembler::registerSymbol(const MCSymbol &Symbol, bool *Created) {
  bool New = !Symbol.isRegistered();
  if (Created)
    *Created = New;
  if (New) {
    Symbol.setIsRegistered(true);
    Symbols.push_back(&Symbol);
  }
}

void MCAssembler::writeFragmentPadding(const MCFragment &F, uint64_t FSize,
                                       MCObjectWriter *OW) const {
  // Should NOP padding be written out before this fragment?
  unsigned BundlePadding = F.getBundlePadding();
  if (BundlePadding > 0) {
    assert(isBundlingEnabled() &&
           "Writing bundle padding with disabled bundling");
    assert(F.hasInstructions() &&
           "Writing bundle padding for a fragment without instructions");

    unsigned TotalLength = BundlePadding + static_cast<unsigned>(FSize);
    if (F.alignToBundleEnd() && TotalLength > getBundleAlignSize()) {
      // If the padding itself crosses a bundle boundary, it must be emitted
      // in 2 pieces, since even nop instructions must not cross boundaries.
      //             v--------------v   <- BundleAlignSize
      //        v---------v             <- BundlePadding
      // ----------------------------
      // | Prev |####|####|    F    |
      // ----------------------------
      //        ^-------------------^   <- TotalLength
      unsigned DistanceToBoundary = TotalLength - getBundleAlignSize();
      if (!getBackend().writeNopData(DistanceToBoundary, OW))
          report_fatal_error("unable to write NOP sequence of " +
                             Twine(DistanceToBoundary) + " bytes");
      BundlePadding -= DistanceToBoundary;
    }
    if (!getBackend().writeNopData(BundlePadding, OW))
      report_fatal_error("unable to write NOP sequence of " +
                         Twine(BundlePadding) + " bytes");
  }
}

/// \brief Write the fragment \p F to the output file.
static void writeFragment(const MCAssembler &Asm, const MCAsmLayout &Layout,
                          const MCFragment &F) {
  MCObjectWriter *OW = &Asm.getWriter();

  // FIXME: Embed in fragments instead?
  uint64_t FragmentSize = Asm.computeFragmentSize(Layout, F);

  Asm.writeFragmentPadding(F, FragmentSize, OW);

  // This variable (and its dummy usage) is to participate in the assert at
  // the end of the function.
  uint64_t Start = OW->getStream().tell();
  (void) Start;

  ++stats::EmittedFragments;

  switch (F.getKind()) {
  case MCFragment::FT_Align: {
    ++stats::EmittedAlignFragments;
    const MCAlignFragment &AF = cast<MCAlignFragment>(F);
    assert(AF.getValueSize() && "Invalid virtual align in concrete fragment!");

    uint64_t Count = FragmentSize / AF.getValueSize();

    // FIXME: This error shouldn't actually occur (the front end should emit
    // multiple .align directives to enforce the semantics it wants), but is
    // severe enough that we want to report it. How to handle this?
    if (Count * AF.getValueSize() != FragmentSize)
      report_fatal_error("undefined .align directive, value size '" +
                        Twine(AF.getValueSize()) +
                        "' is not a divisor of padding size '" +
                        Twine(FragmentSize) + "'");

    // See if we are aligning with nops, and if so do that first to try to fill
    // the Count bytes.  Then if that did not fill any bytes or there are any
    // bytes left to fill use the Value and ValueSize to fill the rest.
    // If we are aligning with nops, ask that target to emit the right data.
    if (AF.hasEmitNops()) {
      if (!Asm.getBackend().writeNopData(Count, OW))
        report_fatal_error("unable to write nop sequence of " +
                          Twine(Count) + " bytes");
      break;
    }

    // Otherwise, write out in multiples of the value size.
    for (uint64_t i = 0; i != Count; ++i) {
      switch (AF.getValueSize()) {
      default: llvm_unreachable("Invalid size!");
      case 1: OW->write8 (uint8_t (AF.getValue())); break;
      case 2: OW->write16(uint16_t(AF.getValue())); break;
      case 4: OW->write32(uint32_t(AF.getValue())); break;
      case 8: OW->write64(uint64_t(AF.getValue())); break;
      }
    }
    break;
  }

  case MCFragment::FT_Data: 
    ++stats::EmittedDataFragments;
    OW->writeBytes(cast<MCDataFragment>(F).getContents());
    break;

  case MCFragment::FT_Relaxable:
    ++stats::EmittedRelaxableFragments;
    OW->writeBytes(cast<MCRelaxableFragment>(F).getContents());
    break;

  case MCFragment::FT_CompactEncodedInst:
    ++stats::EmittedCompactEncodedInstFragments;
    OW->writeBytes(cast<MCCompactEncodedInstFragment>(F).getContents());
    break;

  case MCFragment::FT_Fill: {
    ++stats::EmittedFillFragments;
    const MCFillFragment &FF = cast<MCFillFragment>(F);

    assert(FF.getValueSize() && "Invalid virtual align in concrete fragment!");

    for (uint64_t i = 0, e = FF.getSize() / FF.getValueSize(); i != e; ++i) {
      switch (FF.getValueSize()) {
      default: llvm_unreachable("Invalid size!");
      case 1: OW->write8 (uint8_t (FF.getValue())); break;
      case 2: OW->write16(uint16_t(FF.getValue())); break;
      case 4: OW->write32(uint32_t(FF.getValue())); break;
      case 8: OW->write64(uint64_t(FF.getValue())); break;
      }
    }
    break;
  }

  case MCFragment::FT_LEB: {
    const MCLEBFragment &LF = cast<MCLEBFragment>(F);
    OW->writeBytes(LF.getContents());
    break;
  }

  case MCFragment::FT_SafeSEH: {
    const MCSafeSEHFragment &SF = cast<MCSafeSEHFragment>(F);
    OW->write32(SF.getSymbol()->getIndex());
    break;
  }

  case MCFragment::FT_Org: {
    ++stats::EmittedOrgFragments;
    const MCOrgFragment &OF = cast<MCOrgFragment>(F);

    for (uint64_t i = 0, e = FragmentSize; i != e; ++i)
      OW->write8(uint8_t(OF.getValue()));

    break;
  }

  case MCFragment::FT_Dwarf: {
    const MCDwarfLineAddrFragment &OF = cast<MCDwarfLineAddrFragment>(F);
    OW->writeBytes(OF.getContents());
    break;
  }
  case MCFragment::FT_DwarfFrame: {
    const MCDwarfCallFrameFragment &CF = cast<MCDwarfCallFrameFragment>(F);
    OW->writeBytes(CF.getContents());
    break;
  }
  }

  assert(OW->getStream().tell() - Start == FragmentSize &&
         "The stream should advance by fragment size");
}

void MCAssembler::writeSectionData(const MCSection *Sec,
                                   const MCAsmLayout &Layout) const {
  // Ignore virtual sections.
  if (Sec->isVirtualSection()) {
    assert(Layout.getSectionFileSize(Sec) == 0 && "Invalid size for section!");

    // Check that contents are only things legal inside a virtual section.
    for (MCSection::const_iterator it = Sec->begin(), ie = Sec->end(); it != ie;
         ++it) {
      switch (it->getKind()) {
      default: llvm_unreachable("Invalid fragment in virtual section!");
      case MCFragment::FT_Data: {
        // Check that we aren't trying to write a non-zero contents (or fixups)
        // into a virtual section. This is to support clients which use standard
        // directives to fill the contents of virtual sections.
        const MCDataFragment &DF = cast<MCDataFragment>(*it);
        assert(DF.fixup_begin() == DF.fixup_end() &&
               "Cannot have fixups in virtual section!");
        for (unsigned i = 0, e = DF.getContents().size(); i != e; ++i)
          if (DF.getContents()[i]) {
            if (auto *ELFSec = dyn_cast<const MCSectionELF>(Sec))
              report_fatal_error("non-zero initializer found in section '" +
                  ELFSec->getSectionName() + "'");
            else
              report_fatal_error("non-zero initializer found in virtual section");
          }
        break;
      }
      case MCFragment::FT_Align:
        // Check that we aren't trying to write a non-zero value into a virtual
        // section.
        assert((cast<MCAlignFragment>(it)->getValueSize() == 0 ||
                cast<MCAlignFragment>(it)->getValue() == 0) &&
               "Invalid align in virtual section!");
        break;
      case MCFragment::FT_Fill:
        assert((cast<MCFillFragment>(it)->getValueSize() == 0 ||
                cast<MCFillFragment>(it)->getValue() == 0) &&
               "Invalid fill in virtual section!");
        break;
      }
    }

    return;
  }

  uint64_t Start = getWriter().getStream().tell();
  (void)Start;

  for (MCSection::const_iterator it = Sec->begin(), ie = Sec->end(); it != ie;
       ++it)
    writeFragment(*this, Layout, *it);

  assert(getWriter().getStream().tell() - Start ==
         Layout.getSectionAddressSize(Sec));
}

std::pair<uint64_t, bool> MCAssembler::handleFixup(const MCAsmLayout &Layout,
                                                   MCFragment &F,
                                                   const MCFixup &Fixup) {
  // Evaluate the fixup.
  MCValue Target;
  uint64_t FixedValue;
  bool IsPCRel = Backend.getFixupKindInfo(Fixup.getKind()).Flags &
                 MCFixupKindInfo::FKF_IsPCRel;
  if (!evaluateFixup(Layout, Fixup, &F, Target, FixedValue)) {
    // The fixup was unresolved, we need a relocation. Inform the object
    // writer of the relocation, and give it an opportunity to adjust the
    // fixup value if need be.
    getWriter().recordRelocation(*this, Layout, &F, Fixup, Target, IsPCRel,
                                 FixedValue);
  }
  return std::make_pair(FixedValue, IsPCRel);
}

void MCAssembler::Finish() {
  DEBUG_WITH_TYPE("mc-dump", {
      llvm::errs() << "assembler backend - pre-layout\n--\n";
      dump(); });

  // Create the layout object.
  MCAsmLayout Layout(*this);

  // Create dummy fragments and assign section ordinals.
  unsigned SectionIndex = 0;
  for (MCAssembler::iterator it = begin(), ie = end(); it != ie; ++it) {
    // Create dummy fragments to eliminate any empty sections, this simplifies
    // layout.
    if (it->getFragmentList().empty())
      new MCDataFragment(&*it);

    it->setOrdinal(SectionIndex++);
  }

  // Assign layout order indices to sections and fragments.
  for (unsigned i = 0, e = Layout.getSectionOrder().size(); i != e; ++i) {
    MCSection *Sec = Layout.getSectionOrder()[i];
    Sec->setLayoutOrder(i);

    unsigned FragmentIndex = 0;
    for (MCSection::iterator iFrag = Sec->begin(), iFragEnd = Sec->end();
         iFrag != iFragEnd; ++iFrag)
      iFrag->setLayoutOrder(FragmentIndex++);
  }

  // Layout until everything fits.
  while (layoutOnce(Layout))
    continue;

  DEBUG_WITH_TYPE("mc-dump", {
      llvm::errs() << "assembler backend - post-relaxation\n--\n";
      dump(); });

  // Finalize the layout, including fragment lowering.
  finishLayout(Layout);

  DEBUG_WITH_TYPE("mc-dump", {
      llvm::errs() << "assembler backend - final-layout\n--\n";
      dump(); });

  uint64_t StartOffset = OS.tell();

  // Allow the object writer a chance to perform post-layout binding (for
  // example, to set the index fields in the symbol data).
  getWriter().executePostLayoutBinding(*this, Layout);

  // Evaluate and apply the fixups, generating relocation entries as necessary.
  for (MCAssembler::iterator it = begin(), ie = end(); it != ie; ++it) {
    for (MCSection::iterator it2 = it->begin(), ie2 = it->end(); it2 != ie2;
         ++it2) {
      MCEncodedFragment *F = dyn_cast<MCEncodedFragment>(it2);
      // Data and relaxable fragments both have fixups.  So only process
      // those here.
      // FIXME: Is there a better way to do this?  MCEncodedFragmentWithFixups
      // being templated makes this tricky.
      if (!F || isa<MCCompactEncodedInstFragment>(F))
        continue;
      ArrayRef<MCFixup> Fixups;
      MutableArrayRef<char> Contents;
      if (auto *FragWithFixups = dyn_cast<MCDataFragment>(F)) {
        Fixups = FragWithFixups->getFixups();
        Contents = FragWithFixups->getContents();
      } else if (auto *FragWithFixups = dyn_cast<MCRelaxableFragment>(F)) {
        Fixups = FragWithFixups->getFixups();
        Contents = FragWithFixups->getContents();
      } else
        llvm_unreachable("Unknown fragment with fixups!");
      for (const MCFixup &Fixup : Fixups) {
        uint64_t FixedValue;
        bool IsPCRel;
        std::tie(FixedValue, IsPCRel) = handleFixup(Layout, *F, Fixup);
        getBackend().applyFixup(Fixup, Contents.data(),
                                Contents.size(), FixedValue, IsPCRel);
      }
    }
  }

  // Write the object file.
  getWriter().writeObject(*this, Layout);

  stats::ObjectBytes += OS.tell() - StartOffset;
}

bool MCAssembler::fixupNeedsRelaxation(const MCFixup &Fixup,
                                       const MCRelaxableFragment *DF,
                                       const MCAsmLayout &Layout) const {
  MCValue Target;
  uint64_t Value;
  bool Resolved = evaluateFixup(Layout, Fixup, DF, Target, Value);
  return getBackend().fixupNeedsRelaxationAdvanced(Fixup, Resolved, Value, DF,
                                                   Layout);
}

bool MCAssembler::fragmentNeedsRelaxation(const MCRelaxableFragment *F,
                                          const MCAsmLayout &Layout) const {
  // If this inst doesn't ever need relaxation, ignore it. This occurs when we
  // are intentionally pushing out inst fragments, or because we relaxed a
  // previous instruction to one that doesn't need relaxation.
  if (!getBackend().mayNeedRelaxation(F->getInst()))
    return false;

  for (MCRelaxableFragment::const_fixup_iterator it = F->fixup_begin(),
       ie = F->fixup_end(); it != ie; ++it)
    if (fixupNeedsRelaxation(*it, F, Layout))
      return true;

  return false;
}

bool MCAssembler::relaxInstruction(MCAsmLayout &Layout,
                                   MCRelaxableFragment &F) {
  if (!fragmentNeedsRelaxation(&F, Layout))
    return false;

  ++stats::RelaxedInstructions;

  // FIXME-PERF: We could immediately lower out instructions if we can tell
  // they are fully resolved, to avoid retesting on later passes.

  // Relax the fragment.

  MCInst Relaxed;
  getBackend().relaxInstruction(F.getInst(), Relaxed);

  // Encode the new instruction.
  //
  // FIXME-PERF: If it matters, we could let the target do this. It can
  // probably do so more efficiently in many cases.
  SmallVector<MCFixup, 4> Fixups;
  SmallString<256> Code;
  raw_svector_ostream VecOS(Code);
  getEmitter().encodeInstruction(Relaxed, VecOS, Fixups, F.getSubtargetInfo());
  VecOS.flush();

  // Update the fragment.
  F.setInst(Relaxed);
  F.getContents() = Code;
  F.getFixups() = Fixups;

  return true;
}

bool MCAssembler::relaxLEB(MCAsmLayout &Layout, MCLEBFragment &LF) {
  uint64_t OldSize = LF.getContents().size();
  int64_t Value;
  bool Abs = LF.getValue().evaluateKnownAbsolute(Value, Layout);
  if (!Abs)
    report_fatal_error("sleb128 and uleb128 expressions must be absolute");
  SmallString<8> &Data = LF.getContents();
  Data.clear();
  raw_svector_ostream OSE(Data);
  if (LF.isSigned())
    encodeSLEB128(Value, OSE);
  else
    encodeULEB128(Value, OSE);
  OSE.flush();
  return OldSize != LF.getContents().size();
}

bool MCAssembler::relaxDwarfLineAddr(MCAsmLayout &Layout,
                                     MCDwarfLineAddrFragment &DF) {
  MCContext &Context = Layout.getAssembler().getContext();
  uint64_t OldSize = DF.getContents().size();
  int64_t AddrDelta;
  bool Abs = DF.getAddrDelta().evaluateKnownAbsolute(AddrDelta, Layout);
  assert(Abs && "We created a line delta with an invalid expression");
  (void) Abs;
  int64_t LineDelta;
  LineDelta = DF.getLineDelta();
  SmallString<8> &Data = DF.getContents();
  Data.clear();
  raw_svector_ostream OSE(Data);
  MCDwarfLineAddr::Encode(Context, LineDelta, AddrDelta, OSE);
  OSE.flush();
  return OldSize != Data.size();
}

bool MCAssembler::relaxDwarfCallFrameFragment(MCAsmLayout &Layout,
                                              MCDwarfCallFrameFragment &DF) {
  MCContext &Context = Layout.getAssembler().getContext();
  uint64_t OldSize = DF.getContents().size();
  int64_t AddrDelta;
  bool Abs = DF.getAddrDelta().evaluateKnownAbsolute(AddrDelta, Layout);
  assert(Abs && "We created call frame with an invalid expression");
  (void) Abs;
  SmallString<8> &Data = DF.getContents();
  Data.clear();
  raw_svector_ostream OSE(Data);
  MCDwarfFrameEmitter::EncodeAdvanceLoc(Context, AddrDelta, OSE);
  OSE.flush();
  return OldSize != Data.size();
}

bool MCAssembler::layoutSectionOnce(MCAsmLayout &Layout, MCSection &Sec) {
  // Holds the first fragment which needed relaxing during this layout. It will
  // remain NULL if none were relaxed.
  // When a fragment is relaxed, all the fragments following it should get
  // invalidated because their offset is going to change.
  MCFragment *FirstRelaxedFragment = nullptr;

  // Attempt to relax all the fragments in the section.
  for (MCSection::iterator I = Sec.begin(), IE = Sec.end(); I != IE; ++I) {
    // Check if this is a fragment that needs relaxation.
    bool RelaxedFrag = false;
    switch(I->getKind()) {
    default:
      break;
    case MCFragment::FT_Relaxable:
      assert(!getRelaxAll() &&
             "Did not expect a MCRelaxableFragment in RelaxAll mode");
      RelaxedFrag = relaxInstruction(Layout, *cast<MCRelaxableFragment>(I));
      break;
    case MCFragment::FT_Dwarf:
      RelaxedFrag = relaxDwarfLineAddr(Layout,
                                       *cast<MCDwarfLineAddrFragment>(I));
      break;
    case MCFragment::FT_DwarfFrame:
      RelaxedFrag =
        relaxDwarfCallFrameFragment(Layout,
                                    *cast<MCDwarfCallFrameFragment>(I));
      break;
    case MCFragment::FT_LEB:
      RelaxedFrag = relaxLEB(Layout, *cast<MCLEBFragment>(I));
      break;
    }
    if (RelaxedFrag && !FirstRelaxedFragment)
      FirstRelaxedFragment = I;
  }
  if (FirstRelaxedFragment) {
    Layout.invalidateFragmentsFrom(FirstRelaxedFragment);
    return true;
  }
  return false;
}

bool MCAssembler::layoutOnce(MCAsmLayout &Layout) {
  ++stats::RelaxationSteps;

  bool WasRelaxed = false;
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    MCSection &Sec = *it;
    while (layoutSectionOnce(Layout, Sec))
      WasRelaxed = true;
  }

  return WasRelaxed;
}

void MCAssembler::finishLayout(MCAsmLayout &Layout) {
  // The layout is done. Mark every fragment as valid.
  for (unsigned int i = 0, n = Layout.getSectionOrder().size(); i != n; ++i) {
    Layout.getFragmentOffset(&*Layout.getSectionOrder()[i]->rbegin());
  }
}

// Debugging methods

namespace llvm {

raw_ostream &operator<<(raw_ostream &OS, const MCFixup &AF) {
  OS << "<MCFixup" << " Offset:" << AF.getOffset()
     << " Value:" << *AF.getValue()
     << " Kind:" << AF.getKind() << ">";
  return OS;
}

}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void MCFragment::dump() {
  raw_ostream &OS = llvm::errs();

  OS << "<";
  switch (getKind()) {
  case MCFragment::FT_Align: OS << "MCAlignFragment"; break;
  case MCFragment::FT_Data:  OS << "MCDataFragment"; break;
  case MCFragment::FT_CompactEncodedInst:
    OS << "MCCompactEncodedInstFragment"; break;
  case MCFragment::FT_Fill:  OS << "MCFillFragment"; break;
  case MCFragment::FT_Relaxable:  OS << "MCRelaxableFragment"; break;
  case MCFragment::FT_Org:   OS << "MCOrgFragment"; break;
  case MCFragment::FT_Dwarf: OS << "MCDwarfFragment"; break;
  case MCFragment::FT_DwarfFrame: OS << "MCDwarfCallFrameFragment"; break;
  case MCFragment::FT_LEB:   OS << "MCLEBFragment"; break;
  case MCFragment::FT_SafeSEH:    OS << "MCSafeSEHFragment"; break;
  }

  OS << "<MCFragment " << (void*) this << " LayoutOrder:" << LayoutOrder
     << " Offset:" << Offset
     << " HasInstructions:" << hasInstructions() 
     << " BundlePadding:" << static_cast<unsigned>(getBundlePadding()) << ">";

  switch (getKind()) {
  case MCFragment::FT_Align: {
    const MCAlignFragment *AF = cast<MCAlignFragment>(this);
    if (AF->hasEmitNops())
      OS << " (emit nops)";
    OS << "\n       ";
    OS << " Alignment:" << AF->getAlignment()
       << " Value:" << AF->getValue() << " ValueSize:" << AF->getValueSize()
       << " MaxBytesToEmit:" << AF->getMaxBytesToEmit() << ">";
    break;
  }
  case MCFragment::FT_Data:  {
    const MCDataFragment *DF = cast<MCDataFragment>(this);
    OS << "\n       ";
    OS << " Contents:[";
    const SmallVectorImpl<char> &Contents = DF->getContents();
    for (unsigned i = 0, e = Contents.size(); i != e; ++i) {
      if (i) OS << ",";
      OS << hexdigit((Contents[i] >> 4) & 0xF) << hexdigit(Contents[i] & 0xF);
    }
    OS << "] (" << Contents.size() << " bytes)";

    if (DF->fixup_begin() != DF->fixup_end()) {
      OS << ",\n       ";
      OS << " Fixups:[";
      for (MCDataFragment::const_fixup_iterator it = DF->fixup_begin(),
             ie = DF->fixup_end(); it != ie; ++it) {
        if (it != DF->fixup_begin()) OS << ",\n                ";
        OS << *it;
      }
      OS << "]";
    }
    break;
  }
  case MCFragment::FT_CompactEncodedInst: {
    const MCCompactEncodedInstFragment *CEIF =
      cast<MCCompactEncodedInstFragment>(this);
    OS << "\n       ";
    OS << " Contents:[";
    const SmallVectorImpl<char> &Contents = CEIF->getContents();
    for (unsigned i = 0, e = Contents.size(); i != e; ++i) {
      if (i) OS << ",";
      OS << hexdigit((Contents[i] >> 4) & 0xF) << hexdigit(Contents[i] & 0xF);
    }
    OS << "] (" << Contents.size() << " bytes)";
    break;
  }
  case MCFragment::FT_Fill:  {
    const MCFillFragment *FF = cast<MCFillFragment>(this);
    OS << " Value:" << FF->getValue() << " ValueSize:" << FF->getValueSize()
       << " Size:" << FF->getSize();
    break;
  }
  case MCFragment::FT_Relaxable:  {
    const MCRelaxableFragment *F = cast<MCRelaxableFragment>(this);
    OS << "\n       ";
    OS << " Inst:";
    F->getInst().dump_pretty(OS);
    break;
  }
  case MCFragment::FT_Org:  {
    const MCOrgFragment *OF = cast<MCOrgFragment>(this);
    OS << "\n       ";
    OS << " Offset:" << OF->getOffset() << " Value:" << OF->getValue();
    break;
  }
  case MCFragment::FT_Dwarf:  {
    const MCDwarfLineAddrFragment *OF = cast<MCDwarfLineAddrFragment>(this);
    OS << "\n       ";
    OS << " AddrDelta:" << OF->getAddrDelta()
       << " LineDelta:" << OF->getLineDelta();
    break;
  }
  case MCFragment::FT_DwarfFrame:  {
    const MCDwarfCallFrameFragment *CF = cast<MCDwarfCallFrameFragment>(this);
    OS << "\n       ";
    OS << " AddrDelta:" << CF->getAddrDelta();
    break;
  }
  case MCFragment::FT_LEB: {
    const MCLEBFragment *LF = cast<MCLEBFragment>(this);
    OS << "\n       ";
    OS << " Value:" << LF->getValue() << " Signed:" << LF->isSigned();
    break;
  }
  case MCFragment::FT_SafeSEH: {
    const MCSafeSEHFragment *F = cast<MCSafeSEHFragment>(this);
    OS << "\n       ";
    OS << " Sym:" << F->getSymbol();
    break;
  }
  }
  OS << ">";
}

void MCAssembler::dump() {
  raw_ostream &OS = llvm::errs();

  OS << "<MCAssembler\n";
  OS << "  Sections:[\n    ";
  for (iterator it = begin(), ie = end(); it != ie; ++it) {
    if (it != begin()) OS << ",\n    ";
    it->dump();
  }
  OS << "],\n";
  OS << "  Symbols:[";

  for (symbol_iterator it = symbol_begin(), ie = symbol_end(); it != ie; ++it) {
    if (it != symbol_begin()) OS << ",\n           ";
    OS << "(";
    it->dump();
    OS << ", Index:" << it->getIndex() << ", ";
    OS << ")";
  }
  OS << "]>\n";
}
#endif
