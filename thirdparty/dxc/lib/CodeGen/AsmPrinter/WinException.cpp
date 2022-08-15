//===-- CodeGen/AsmPrinter/WinException.cpp - Dwarf Exception Impl ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing Win64 exception info into asm files.
//
//===----------------------------------------------------------------------===//

#include "WinException.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/WinEHFuncInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCWin64EH.h"
#include "llvm/Support/Dwarf.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Target/TargetFrameLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/TargetRegisterInfo.h"
using namespace llvm;

WinException::WinException(AsmPrinter *A) : EHStreamer(A) {
  // MSVC's EH tables are always composed of 32-bit words.  All known 64-bit
  // platforms use an imagerel32 relocation to refer to symbols.
  useImageRel32 = (A->getDataLayout().getPointerSizeInBits() == 64);
}

WinException::~WinException() {}

/// endModule - Emit all exception information that should come after the
/// content.
void WinException::endModule() {
  auto &OS = *Asm->OutStreamer;
  const Module *M = MMI->getModule();
  for (const Function &F : *M)
    if (F.hasFnAttribute("safeseh"))
      OS.EmitCOFFSafeSEH(Asm->getSymbol(&F));
}

void WinException::beginFunction(const MachineFunction *MF) {
  shouldEmitMoves = shouldEmitPersonality = shouldEmitLSDA = false;

  // If any landing pads survive, we need an EH table.
  bool hasLandingPads = !MMI->getLandingPads().empty();

  const Function *F = MF->getFunction();
  const Function *ParentF = MMI->getWinEHParent(F);

  shouldEmitMoves = Asm->needsSEHMoves();

  const TargetLoweringObjectFile &TLOF = Asm->getObjFileLowering();
  unsigned PerEncoding = TLOF.getPersonalityEncoding();
  const Function *Per = nullptr;
  if (F->hasPersonalityFn())
    Per = dyn_cast<Function>(F->getPersonalityFn()->stripPointerCasts());

  bool forceEmitPersonality =
    F->hasPersonalityFn() && !isNoOpWithoutInvoke(classifyEHPersonality(Per)) &&
    F->needsUnwindTableEntry();

  shouldEmitPersonality = forceEmitPersonality || (hasLandingPads &&
    PerEncoding != dwarf::DW_EH_PE_omit && Per);

  unsigned LSDAEncoding = TLOF.getLSDAEncoding();
  shouldEmitLSDA = shouldEmitPersonality &&
    LSDAEncoding != dwarf::DW_EH_PE_omit;

  // If we're not using CFI, we don't want the CFI or the personality. If
  // WinEHPrepare outlined something, we should emit the LSDA.
  if (!Asm->MAI->usesWindowsCFI()) {
    bool HasOutlinedChildren =
        F->hasFnAttribute("wineh-parent") && F == ParentF;
    shouldEmitLSDA = HasOutlinedChildren;
    shouldEmitPersonality = false;
    return;
  }

  // If this was an outlined handler, we need to define the label corresponding
  // to the offset of the parent frame relative to the stack pointer after the
  // prologue.
  if (F != ParentF) {
    WinEHFuncInfo &FuncInfo = MMI->getWinEHFuncInfo(ParentF);
    auto I = FuncInfo.CatchHandlerParentFrameObjOffset.find(F);
    if (I != FuncInfo.CatchHandlerParentFrameObjOffset.end()) {
      MCSymbol *HandlerTypeParentFrameOffset =
          Asm->OutContext.getOrCreateParentFrameOffsetSymbol(
              GlobalValue::getRealLinkageName(F->getName()));

      // Emit a symbol assignment.
      Asm->OutStreamer->EmitAssignment(
          HandlerTypeParentFrameOffset,
          MCConstantExpr::create(I->second, Asm->OutContext));
    }
  }

  if (shouldEmitMoves || shouldEmitPersonality)
    Asm->OutStreamer->EmitWinCFIStartProc(Asm->CurrentFnSym);

  if (shouldEmitPersonality) {
    const MCSymbol *PersHandlerSym =
        TLOF.getCFIPersonalitySymbol(Per, *Asm->Mang, Asm->TM, MMI);
    Asm->OutStreamer->EmitWinEHHandler(PersHandlerSym, true, true);
  }
}

/// endFunction - Gather and emit post-function exception information.
///
void WinException::endFunction(const MachineFunction *MF) {
  if (!shouldEmitPersonality && !shouldEmitMoves && !shouldEmitLSDA)
    return;

  const Function *F = MF->getFunction();
  EHPersonality Per = EHPersonality::Unknown;
  if (F->hasPersonalityFn())
    Per = classifyEHPersonality(F->getPersonalityFn());

  // Get rid of any dead landing pads if we're not using a Windows EH scheme. In
  // Windows EH schemes, the landing pad is not actually reachable. It only
  // exists so that we can emit the right table data.
  if (!isMSVCEHPersonality(Per))
    MMI->TidyLandingPads();

  if (shouldEmitPersonality || shouldEmitLSDA) {
    Asm->OutStreamer->PushSection();

    if (shouldEmitMoves || shouldEmitPersonality) {
      // Emit an UNWIND_INFO struct describing the prologue.
      Asm->OutStreamer->EmitWinEHHandlerData();
    } else {
      // Just switch sections to the right xdata section. This use of
      // CurrentFnSym assumes that we only emit the LSDA when ending the parent
      // function.
      MCSection *XData = WinEH::UnwindEmitter::getXDataSection(
          Asm->CurrentFnSym, Asm->OutContext);
      Asm->OutStreamer->SwitchSection(XData);
    }

    // Emit the tables appropriate to the personality function in use. If we
    // don't recognize the personality, assume it uses an Itanium-style LSDA.
    if (Per == EHPersonality::MSVC_Win64SEH)
      emitCSpecificHandlerTable();
    else if (Per == EHPersonality::MSVC_X86SEH)
      emitExceptHandlerTable(MF);
    else if (Per == EHPersonality::MSVC_CXX)
      emitCXXFrameHandler3Table(MF);
    else
      emitExceptionTable();

    Asm->OutStreamer->PopSection();
  }

  if (shouldEmitMoves)
    Asm->OutStreamer->EmitWinCFIEndProc();
}

const MCExpr *WinException::create32bitRef(const MCSymbol *Value) {
  if (!Value)
    return MCConstantExpr::create(0, Asm->OutContext);
  return MCSymbolRefExpr::create(Value, useImageRel32
                                            ? MCSymbolRefExpr::VK_COFF_IMGREL32
                                            : MCSymbolRefExpr::VK_None,
                                 Asm->OutContext);
}

const MCExpr *WinException::create32bitRef(const GlobalValue *GV) {
  if (!GV)
    return MCConstantExpr::create(0, Asm->OutContext);
  return create32bitRef(Asm->getSymbol(GV));
}

/// Emit the language-specific data that __C_specific_handler expects.  This
/// handler lives in the x64 Microsoft C runtime and allows catching or cleaning
/// up after faults with __try, __except, and __finally.  The typeinfo values
/// are not really RTTI data, but pointers to filter functions that return an
/// integer (1, 0, or -1) indicating how to handle the exception. For __finally
/// blocks and other cleanups, the landing pad label is zero, and the filter
/// function is actually a cleanup handler with the same prototype.  A catch-all
/// entry is modeled with a null filter function field and a non-zero landing
/// pad label.
///
/// Possible filter function return values:
///   EXCEPTION_EXECUTE_HANDLER (1):
///     Jump to the landing pad label after cleanups.
///   EXCEPTION_CONTINUE_SEARCH (0):
///     Continue searching this table or continue unwinding.
///   EXCEPTION_CONTINUE_EXECUTION (-1):
///     Resume execution at the trapping PC.
///
/// Inferred table structure:
///   struct Table {
///     int NumEntries;
///     struct Entry {
///       imagerel32 LabelStart;
///       imagerel32 LabelEnd;
///       imagerel32 FilterOrFinally;  // One means catch-all.
///       imagerel32 LabelLPad;        // Zero means __finally.
///     } Entries[NumEntries];
///   };
void WinException::emitCSpecificHandlerTable() {
  const std::vector<LandingPadInfo> &PadInfos = MMI->getLandingPads();

  // Simplifying assumptions for first implementation:
  // - Cleanups are not implemented.
  // - Filters are not implemented.

  // The Itanium LSDA table sorts similar landing pads together to simplify the
  // actions table, but we don't need that.
  SmallVector<const LandingPadInfo *, 64> LandingPads;
  LandingPads.reserve(PadInfos.size());
  for (const auto &LP : PadInfos)
    LandingPads.push_back(&LP);

  // Compute label ranges for call sites as we would for the Itanium LSDA, but
  // use an all zero action table because we aren't using these actions.
  SmallVector<unsigned, 64> FirstActions;
  FirstActions.resize(LandingPads.size());
  SmallVector<CallSiteEntry, 64> CallSites;
  computeCallSiteTable(CallSites, LandingPads, FirstActions);

  MCSymbol *EHFuncBeginSym = Asm->getFunctionBegin();
  MCSymbol *EHFuncEndSym = Asm->getFunctionEnd();

  // Emit the number of table entries.
  unsigned NumEntries = 0;
  for (const CallSiteEntry &CSE : CallSites) {
    if (!CSE.LPad)
      continue; // Ignore gaps.
    NumEntries += CSE.LPad->SEHHandlers.size();
  }
  Asm->OutStreamer->EmitIntValue(NumEntries, 4);

  // If there are no actions, we don't need to iterate again.
  if (NumEntries == 0)
    return;

  // Emit the four-label records for each call site entry. The table has to be
  // sorted in layout order, and the call sites should already be sorted.
  for (const CallSiteEntry &CSE : CallSites) {
    // Ignore gaps. Unlike the Itanium model, unwinding through a frame without
    // an EH table entry will propagate the exception rather than terminating
    // the program.
    if (!CSE.LPad)
      continue;
    const LandingPadInfo *LPad = CSE.LPad;

    // Compute the label range. We may reuse the function begin and end labels
    // rather than forming new ones.
    const MCExpr *Begin =
        create32bitRef(CSE.BeginLabel ? CSE.BeginLabel : EHFuncBeginSym);
    const MCExpr *End;
    if (CSE.EndLabel) {
      // The interval is half-open, so we have to add one to include the return
      // address of the last invoke in the range.
      End = MCBinaryExpr::createAdd(create32bitRef(CSE.EndLabel),
                                    MCConstantExpr::create(1, Asm->OutContext),
                                    Asm->OutContext);
    } else {
      End = create32bitRef(EHFuncEndSym);
    }

    // Emit an entry for each action.
    for (SEHHandler Handler : LPad->SEHHandlers) {
      Asm->OutStreamer->EmitValue(Begin, 4);
      Asm->OutStreamer->EmitValue(End, 4);

      // Emit the filter or finally function pointer, if present. Otherwise,
      // emit '1' to indicate a catch-all.
      const Function *F = Handler.FilterOrFinally;
      if (F)
        Asm->OutStreamer->EmitValue(create32bitRef(Asm->getSymbol(F)), 4);
      else
        Asm->OutStreamer->EmitIntValue(1, 4);

      // Emit the recovery address, if present. Otherwise, this must be a
      // finally.
      const BlockAddress *BA = Handler.RecoverBA;
      if (BA)
        Asm->OutStreamer->EmitValue(
            create32bitRef(Asm->GetBlockAddressSymbol(BA)), 4);
      else
        Asm->OutStreamer->EmitIntValue(0, 4);
    }
  }
}

void WinException::emitCXXFrameHandler3Table(const MachineFunction *MF) {
  const Function *F = MF->getFunction();
  const Function *ParentF = MMI->getWinEHParent(F);
  auto &OS = *Asm->OutStreamer;
  WinEHFuncInfo &FuncInfo = MMI->getWinEHFuncInfo(ParentF);

  StringRef ParentLinkageName =
      GlobalValue::getRealLinkageName(ParentF->getName());

  MCSymbol *FuncInfoXData = nullptr;
  if (shouldEmitPersonality) {
    FuncInfoXData = Asm->OutContext.getOrCreateSymbol(
        Twine("$cppxdata$", ParentLinkageName));
    OS.EmitValue(create32bitRef(FuncInfoXData), 4);

    extendIP2StateTable(MF, ParentF, FuncInfo);

    // Defer emission until we've visited the parent function and all the catch
    // handlers.  Cleanups don't contribute to the ip2state table, so don't count
    // them.
    if (ParentF != F && !FuncInfo.CatchHandlerMaxState.count(F))
      return;
    ++FuncInfo.NumIPToStateFuncsVisited;
    if (FuncInfo.NumIPToStateFuncsVisited != FuncInfo.CatchHandlerMaxState.size())
      return;
  } else {
    FuncInfoXData = Asm->OutContext.getOrCreateLSDASymbol(ParentLinkageName);
    emitEHRegistrationOffsetLabel(FuncInfo, ParentLinkageName);
  }

  MCSymbol *UnwindMapXData = nullptr;
  MCSymbol *TryBlockMapXData = nullptr;
  MCSymbol *IPToStateXData = nullptr;
  if (!FuncInfo.UnwindMap.empty())
    UnwindMapXData = Asm->OutContext.getOrCreateSymbol(
        Twine("$stateUnwindMap$", ParentLinkageName));
  if (!FuncInfo.TryBlockMap.empty())
    TryBlockMapXData = Asm->OutContext.getOrCreateSymbol(
        Twine("$tryMap$", ParentLinkageName));
  if (!FuncInfo.IPToStateList.empty())
    IPToStateXData = Asm->OutContext.getOrCreateSymbol(
        Twine("$ip2state$", ParentLinkageName));

  // FuncInfo {
  //   uint32_t           MagicNumber
  //   int32_t            MaxState;
  //   UnwindMapEntry    *UnwindMap;
  //   uint32_t           NumTryBlocks;
  //   TryBlockMapEntry  *TryBlockMap;
  //   uint32_t           IPMapEntries; // always 0 for x86
  //   IPToStateMapEntry *IPToStateMap; // always 0 for x86
  //   uint32_t           UnwindHelp;   // non-x86 only
  //   ESTypeList        *ESTypeList;
  //   int32_t            EHFlags;
  // }
  // EHFlags & 1 -> Synchronous exceptions only, no async exceptions.
  // EHFlags & 2 -> ???
  // EHFlags & 4 -> The function is noexcept(true), unwinding can't continue.
  OS.EmitValueToAlignment(4);
  OS.EmitLabel(FuncInfoXData);
  OS.EmitIntValue(0x19930522, 4);                      // MagicNumber
  OS.EmitIntValue(FuncInfo.UnwindMap.size(), 4);       // MaxState
  OS.EmitValue(create32bitRef(UnwindMapXData), 4);     // UnwindMap
  OS.EmitIntValue(FuncInfo.TryBlockMap.size(), 4);     // NumTryBlocks
  OS.EmitValue(create32bitRef(TryBlockMapXData), 4);   // TryBlockMap
  OS.EmitIntValue(FuncInfo.IPToStateList.size(), 4);   // IPMapEntries
  OS.EmitValue(create32bitRef(IPToStateXData), 4);     // IPToStateMap
  if (Asm->MAI->usesWindowsCFI())
    OS.EmitIntValue(FuncInfo.UnwindHelpFrameOffset, 4); // UnwindHelp
  OS.EmitIntValue(0, 4);                               // ESTypeList
  OS.EmitIntValue(1, 4);                               // EHFlags

  // UnwindMapEntry {
  //   int32_t ToState;
  //   void  (*Action)();
  // };
  if (UnwindMapXData) {
    OS.EmitLabel(UnwindMapXData);
    for (const WinEHUnwindMapEntry &UME : FuncInfo.UnwindMap) {
      OS.EmitIntValue(UME.ToState, 4);                // ToState
      OS.EmitValue(create32bitRef(UME.Cleanup), 4);   // Action
    }
  }

  // TryBlockMap {
  //   int32_t      TryLow;
  //   int32_t      TryHigh;
  //   int32_t      CatchHigh;
  //   int32_t      NumCatches;
  //   HandlerType *HandlerArray;
  // };
  if (TryBlockMapXData) {
    OS.EmitLabel(TryBlockMapXData);
    SmallVector<MCSymbol *, 1> HandlerMaps;
    for (size_t I = 0, E = FuncInfo.TryBlockMap.size(); I != E; ++I) {
      WinEHTryBlockMapEntry &TBME = FuncInfo.TryBlockMap[I];
      MCSymbol *HandlerMapXData = nullptr;

      if (!TBME.HandlerArray.empty())
        HandlerMapXData =
            Asm->OutContext.getOrCreateSymbol(Twine("$handlerMap$")
                                                  .concat(Twine(I))
                                                  .concat("$")
                                                  .concat(ParentLinkageName));

      HandlerMaps.push_back(HandlerMapXData);

      int CatchHigh = -1;
      for (WinEHHandlerType &HT : TBME.HandlerArray)
        CatchHigh =
            std::max(CatchHigh, FuncInfo.CatchHandlerMaxState[HT.Handler]);

      assert(TBME.TryLow <= TBME.TryHigh);
      OS.EmitIntValue(TBME.TryLow, 4);                    // TryLow
      OS.EmitIntValue(TBME.TryHigh, 4);                   // TryHigh
      OS.EmitIntValue(CatchHigh, 4);                      // CatchHigh
      OS.EmitIntValue(TBME.HandlerArray.size(), 4);       // NumCatches
      OS.EmitValue(create32bitRef(HandlerMapXData), 4);   // HandlerArray
    }

    for (size_t I = 0, E = FuncInfo.TryBlockMap.size(); I != E; ++I) {
      WinEHTryBlockMapEntry &TBME = FuncInfo.TryBlockMap[I];
      MCSymbol *HandlerMapXData = HandlerMaps[I];
      if (!HandlerMapXData)
        continue;
      // HandlerType {
      //   int32_t         Adjectives;
      //   TypeDescriptor *Type;
      //   int32_t         CatchObjOffset;
      //   void          (*Handler)();
      //   int32_t         ParentFrameOffset; // x64 only
      // };
      OS.EmitLabel(HandlerMapXData);
      for (const WinEHHandlerType &HT : TBME.HandlerArray) {
        // Get the frame escape label with the offset of the catch object. If
        // the index is -1, then there is no catch object, and we should emit an
        // offset of zero, indicating that no copy will occur.
        const MCExpr *FrameAllocOffsetRef = nullptr;
        if (HT.CatchObjRecoverIdx >= 0) {
          MCSymbol *FrameAllocOffset =
              Asm->OutContext.getOrCreateFrameAllocSymbol(
                  GlobalValue::getRealLinkageName(ParentF->getName()),
                  HT.CatchObjRecoverIdx);
          FrameAllocOffsetRef = MCSymbolRefExpr::create(
              FrameAllocOffset, MCSymbolRefExpr::VK_None, Asm->OutContext);
        } else {
          FrameAllocOffsetRef = MCConstantExpr::create(0, Asm->OutContext);
        }

        OS.EmitIntValue(HT.Adjectives, 4);                    // Adjectives
        OS.EmitValue(create32bitRef(HT.TypeDescriptor), 4);   // Type
        OS.EmitValue(FrameAllocOffsetRef, 4);                 // CatchObjOffset
        OS.EmitValue(create32bitRef(HT.Handler), 4);          // Handler

        if (shouldEmitPersonality) {
          MCSymbol *ParentFrameOffset =
              Asm->OutContext.getOrCreateParentFrameOffsetSymbol(
                  GlobalValue::getRealLinkageName(HT.Handler->getName()));
          const MCSymbolRefExpr *ParentFrameOffsetRef = MCSymbolRefExpr::create(
              ParentFrameOffset, Asm->OutContext);
          OS.EmitValue(ParentFrameOffsetRef, 4); // ParentFrameOffset
        }
      }
    }
  }

  // IPToStateMapEntry {
  //   void   *IP;
  //   int32_t State;
  // };
  if (IPToStateXData) {
    OS.EmitLabel(IPToStateXData);
    for (auto &IPStatePair : FuncInfo.IPToStateList) {
      OS.EmitValue(create32bitRef(IPStatePair.first), 4);   // IP
      OS.EmitIntValue(IPStatePair.second, 4);               // State
    }
  }
}

void WinException::extendIP2StateTable(const MachineFunction *MF,
                                       const Function *ParentF,
                                       WinEHFuncInfo &FuncInfo) {
  const Function *F = MF->getFunction();

  // The Itanium LSDA table sorts similar landing pads together to simplify the
  // actions table, but we don't need that.
  SmallVector<const LandingPadInfo *, 64> LandingPads;
  const std::vector<LandingPadInfo> &PadInfos = MMI->getLandingPads();
  LandingPads.reserve(PadInfos.size());
  for (const auto &LP : PadInfos)
    LandingPads.push_back(&LP);

  RangeMapType PadMap;
  computePadMap(LandingPads, PadMap);

  // The end label of the previous invoke or nounwind try-range.
  MCSymbol *LastLabel = Asm->getFunctionBegin();

  // Whether there is a potentially throwing instruction (currently this means
  // an ordinary call) between the end of the previous try-range and now.
  bool SawPotentiallyThrowing = false;

  int LastEHState = -2;

  // The parent function and the catch handlers contribute to the 'ip2state'
  // table.

  // Include ip2state entries for the beginning of the main function and
  // for catch handler functions.
  if (F == ParentF) {
    FuncInfo.IPToStateList.push_back(std::make_pair(LastLabel, -1));
    LastEHState = -1;
  } else if (FuncInfo.HandlerBaseState.count(F)) {
    FuncInfo.IPToStateList.push_back(
        std::make_pair(LastLabel, FuncInfo.HandlerBaseState[F]));
    LastEHState = FuncInfo.HandlerBaseState[F];
  }
  for (const auto &MBB : *MF) {
    for (const auto &MI : MBB) {
      if (!MI.isEHLabel()) {
        if (MI.isCall())
          SawPotentiallyThrowing |= !callToNoUnwindFunction(&MI);
        continue;
      }

      // End of the previous try-range?
      MCSymbol *BeginLabel = MI.getOperand(0).getMCSymbol();
      if (BeginLabel == LastLabel)
        SawPotentiallyThrowing = false;

      // Beginning of a new try-range?
      RangeMapType::const_iterator L = PadMap.find(BeginLabel);
      if (L == PadMap.end())
        // Nope, it was just some random label.
        continue;

      const PadRange &P = L->second;
      const LandingPadInfo *LandingPad = LandingPads[P.PadIndex];
      assert(BeginLabel == LandingPad->BeginLabels[P.RangeIndex] &&
             "Inconsistent landing pad map!");

      // FIXME: Should this be using FuncInfo.HandlerBaseState?
      if (SawPotentiallyThrowing && LastEHState != -1) {
        FuncInfo.IPToStateList.push_back(std::make_pair(LastLabel, -1));
        SawPotentiallyThrowing = false;
        LastEHState = -1;
      }

      if (LandingPad->WinEHState != LastEHState)
        FuncInfo.IPToStateList.push_back(
            std::make_pair(BeginLabel, LandingPad->WinEHState));
      LastEHState = LandingPad->WinEHState;
      LastLabel = LandingPad->EndLabels[P.RangeIndex];
    }
  }
}

void WinException::emitEHRegistrationOffsetLabel(const WinEHFuncInfo &FuncInfo,
                                                 StringRef FLinkageName) {
  // Outlined helpers called by the EH runtime need to know the offset of the EH
  // registration in order to recover the parent frame pointer. Now that we know
  // we've code generated the parent, we can emit the label assignment that
  // those helpers use to get the offset of the registration node.
  assert(FuncInfo.EHRegNodeEscapeIndex != INT_MAX &&
         "no EH reg node localescape index");
  MCSymbol *ParentFrameOffset =
      Asm->OutContext.getOrCreateParentFrameOffsetSymbol(FLinkageName);
  MCSymbol *RegistrationOffsetSym = Asm->OutContext.getOrCreateFrameAllocSymbol(
      FLinkageName, FuncInfo.EHRegNodeEscapeIndex);
  const MCExpr *RegistrationOffsetSymRef =
      MCSymbolRefExpr::create(RegistrationOffsetSym, Asm->OutContext);
  Asm->OutStreamer->EmitAssignment(ParentFrameOffset, RegistrationOffsetSymRef);
}

/// Emit the language-specific data that _except_handler3 and 4 expect. This is
/// functionally equivalent to the __C_specific_handler table, except it is
/// indexed by state number instead of IP.
void WinException::emitExceptHandlerTable(const MachineFunction *MF) {
  MCStreamer &OS = *Asm->OutStreamer;
  const Function *F = MF->getFunction();
  StringRef FLinkageName = GlobalValue::getRealLinkageName(F->getName());

  WinEHFuncInfo &FuncInfo = MMI->getWinEHFuncInfo(F);
  emitEHRegistrationOffsetLabel(FuncInfo, FLinkageName);

  // Emit the __ehtable label that we use for llvm.x86.seh.lsda.
  MCSymbol *LSDALabel = Asm->OutContext.getOrCreateLSDASymbol(FLinkageName);
  OS.EmitValueToAlignment(4);
  OS.EmitLabel(LSDALabel);

  const Function *Per =
      dyn_cast<Function>(F->getPersonalityFn()->stripPointerCasts());
  StringRef PerName = Per->getName();
  int BaseState = -1;
  if (PerName == "_except_handler4") {
    // The LSDA for _except_handler4 starts with this struct, followed by the
    // scope table:
    //
    // struct EH4ScopeTable {
    //   int32_t GSCookieOffset;
    //   int32_t GSCookieXOROffset;
    //   int32_t EHCookieOffset;
    //   int32_t EHCookieXOROffset;
    //   ScopeTableEntry ScopeRecord[];
    // };
    //
    // Only the EHCookieOffset field appears to vary, and it appears to be the
    // offset from the final saved SP value to the retaddr.
    OS.EmitIntValue(-2, 4);
    OS.EmitIntValue(0, 4);
    // FIXME: Calculate.
    OS.EmitIntValue(9999, 4);
    OS.EmitIntValue(0, 4);
    BaseState = -2;
  }

  // Build a list of pointers to LandingPadInfos and then sort by WinEHState.
  const std::vector<LandingPadInfo> &PadInfos = MMI->getLandingPads();
  SmallVector<const LandingPadInfo *, 4> LPads;
  LPads.reserve((PadInfos.size()));
  for (const LandingPadInfo &LPInfo : PadInfos)
    LPads.push_back(&LPInfo);
  std::sort(LPads.begin(), LPads.end(),
            [](const LandingPadInfo *L, const LandingPadInfo *R) {
              return L->WinEHState < R->WinEHState;
            });

  // For each action in each lpad, emit one of these:
  // struct ScopeTableEntry {
  //   int32_t EnclosingLevel;
  //   int32_t (__cdecl *Filter)();
  //   void *HandlerOrFinally;
  // };
  //
  // The "outermost" action will use BaseState as its enclosing level. Each
  // other action will refer to the previous state as its enclosing level.
  int CurState = 0;
  for (const LandingPadInfo *LPInfo : LPads) {
    int EnclosingLevel = BaseState;
    assert(CurState + int(LPInfo->SEHHandlers.size()) - 1 ==
               LPInfo->WinEHState &&
           "gaps in the SEH scope table");
    for (auto I = LPInfo->SEHHandlers.rbegin(), E = LPInfo->SEHHandlers.rend();
         I != E; ++I) {
      const SEHHandler &Handler = *I;
      const BlockAddress *BA = Handler.RecoverBA;
      const Function *F = Handler.FilterOrFinally;
      assert(F && "cannot catch all in 32-bit SEH without filter function");
      const MCExpr *FilterOrNull =
          create32bitRef(BA ? Asm->getSymbol(F) : nullptr);
      const MCExpr *ExceptOrFinally = create32bitRef(
          BA ? Asm->GetBlockAddressSymbol(BA) : Asm->getSymbol(F));

      OS.EmitIntValue(EnclosingLevel, 4);
      OS.EmitValue(FilterOrNull, 4);
      OS.EmitValue(ExceptOrFinally, 4);

      // The next state unwinds to this state.
      EnclosingLevel = CurState;
      CurState++;
    }
  }
}
