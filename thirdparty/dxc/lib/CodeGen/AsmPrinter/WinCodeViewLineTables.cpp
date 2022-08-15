//===-- llvm/lib/CodeGen/AsmPrinter/WinCodeViewLineTables.cpp --*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains support for writing line tables info into COFF files.
//
//===----------------------------------------------------------------------===//

#include "WinCodeViewLineTables.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Support/COFF.h"

namespace llvm {

StringRef WinCodeViewLineTables::getFullFilepath(const MDNode *S) {
  assert(S);
  assert((isa<DICompileUnit>(S) || isa<DIFile>(S) || isa<DISubprogram>(S) ||
          isa<DILexicalBlockBase>(S)) &&
         "Unexpected scope info");

  auto *Scope = cast<DIScope>(S);
  StringRef Dir = Scope->getDirectory(),
            Filename = Scope->getFilename();
  char *&Result = DirAndFilenameToFilepathMap[std::make_pair(Dir, Filename)];
  if (Result)
    return Result;

  // Clang emits directory and relative filename info into the IR, but CodeView
  // operates on full paths.  We could change Clang to emit full paths too, but
  // that would increase the IR size and probably not needed for other users.
  // For now, just concatenate and canonicalize the path here.
  std::string Filepath;
  if (Filename.find(':') == 1)
    Filepath = Filename;
  else
    Filepath = (Dir + "\\" + Filename).str();

  // Canonicalize the path.  We have to do it textually because we may no longer
  // have access the file in the filesystem.
  // First, replace all slashes with backslashes.
  std::replace(Filepath.begin(), Filepath.end(), '/', '\\');

  // Remove all "\.\" with "\".
  size_t Cursor = 0;
  while ((Cursor = Filepath.find("\\.\\", Cursor)) != std::string::npos)
    Filepath.erase(Cursor, 2);

  // Replace all "\XXX\..\" with "\".  Don't try too hard though as the original
  // path should be well-formatted, e.g. start with a drive letter, etc.
  Cursor = 0;
  while ((Cursor = Filepath.find("\\..\\", Cursor)) != std::string::npos) {
    // Something's wrong if the path starts with "\..\", abort.
    if (Cursor == 0)
      break;

    size_t PrevSlash = Filepath.rfind('\\', Cursor - 1);
    if (PrevSlash == std::string::npos)
      // Something's wrong, abort.
      break;

    Filepath.erase(PrevSlash, Cursor + 3 - PrevSlash);
    // The next ".." might be following the one we've just erased.
    Cursor = PrevSlash;
  }

  // Remove all duplicate backslashes.
  Cursor = 0;
  while ((Cursor = Filepath.find("\\\\", Cursor)) != std::string::npos)
    Filepath.erase(Cursor, 1);

  Result = strdup(Filepath.c_str());
  return StringRef(Result);
}

void WinCodeViewLineTables::maybeRecordLocation(DebugLoc DL,
                                                const MachineFunction *MF) {
  const MDNode *Scope = DL.getScope();
  if (!Scope)
    return;
  StringRef Filename = getFullFilepath(Scope);

  // Skip this instruction if it has the same file:line as the previous one.
  assert(CurFn);
  if (!CurFn->Instrs.empty()) {
    const InstrInfoTy &LastInstr = InstrInfo[CurFn->Instrs.back()];
    if (LastInstr.Filename == Filename && LastInstr.LineNumber == DL.getLine())
      return;
  }
  FileNameRegistry.add(Filename);

  MCSymbol *MCL = Asm->MMI->getContext().createTempSymbol();
  Asm->OutStreamer->EmitLabel(MCL);
  CurFn->Instrs.push_back(MCL);
  InstrInfo[MCL] = InstrInfoTy(Filename, DL.getLine(), DL.getCol());
}

WinCodeViewLineTables::WinCodeViewLineTables(AsmPrinter *AP)
    : Asm(nullptr), CurFn(nullptr) {
  MachineModuleInfo *MMI = AP->MMI;

  // If module doesn't have named metadata anchors or COFF debug section
  // is not available, skip any debug info related stuff.
  if (!MMI->getModule()->getNamedMetadata("llvm.dbg.cu") ||
      !AP->getObjFileLowering().getCOFFDebugSymbolsSection())
    return;

  // Tell MMI that we have debug info.
  MMI->setDebugInfoAvailability(true);
  Asm = AP;
}

void WinCodeViewLineTables::endModule() {
  if (FnDebugInfo.empty())
    return;

  assert(Asm != nullptr);
  Asm->OutStreamer->SwitchSection(
      Asm->getObjFileLowering().getCOFFDebugSymbolsSection());
  Asm->EmitInt32(COFF::DEBUG_SECTION_MAGIC);

  // The COFF .debug$S section consists of several subsections, each starting
  // with a 4-byte control code (e.g. 0xF1, 0xF2, etc) and then a 4-byte length
  // of the payload followed by the payload itself.  The subsections are 4-byte
  // aligned.

  // Emit per-function debug information.  This code is extracted into a
  // separate function for readability.
  for (size_t I = 0, E = VisitedFunctions.size(); I != E; ++I)
    emitDebugInfoForFunction(VisitedFunctions[I]);

  // This subsection holds a file index to offset in string table table.
  Asm->OutStreamer->AddComment("File index to string table offset subsection");
  Asm->EmitInt32(COFF::DEBUG_INDEX_SUBSECTION);
  size_t NumFilenames = FileNameRegistry.Infos.size();
  Asm->EmitInt32(8 * NumFilenames);
  for (size_t I = 0, E = FileNameRegistry.Filenames.size(); I != E; ++I) {
    StringRef Filename = FileNameRegistry.Filenames[I];
    // For each unique filename, just write its offset in the string table.
    Asm->EmitInt32(FileNameRegistry.Infos[Filename].StartOffset);
    // The function name offset is not followed by any additional data.
    Asm->EmitInt32(0);
  }

  // This subsection holds the string table.
  Asm->OutStreamer->AddComment("String table");
  Asm->EmitInt32(COFF::DEBUG_STRING_TABLE_SUBSECTION);
  Asm->EmitInt32(FileNameRegistry.LastOffset);
  // The payload starts with a null character.
  Asm->EmitInt8(0);

  for (size_t I = 0, E = FileNameRegistry.Filenames.size(); I != E; ++I) {
    // Just emit unique filenames one by one, separated by a null character.
    Asm->OutStreamer->EmitBytes(FileNameRegistry.Filenames[I]);
    Asm->EmitInt8(0);
  }

  // No more subsections. Fill with zeros to align the end of the section by 4.
  Asm->OutStreamer->EmitFill((-FileNameRegistry.LastOffset) % 4, 0);

  clear();
}

static void EmitLabelDiff(MCStreamer &Streamer,
                          const MCSymbol *From, const MCSymbol *To,
                          unsigned int Size = 4) {
  MCSymbolRefExpr::VariantKind Variant = MCSymbolRefExpr::VK_None;
  MCContext &Context = Streamer.getContext();
  const MCExpr *FromRef = MCSymbolRefExpr::create(From, Variant, Context),
               *ToRef   = MCSymbolRefExpr::create(To, Variant, Context);
  const MCExpr *AddrDelta =
      MCBinaryExpr::create(MCBinaryExpr::Sub, ToRef, FromRef, Context);
  Streamer.EmitValue(AddrDelta, Size);
}

void WinCodeViewLineTables::emitDebugInfoForFunction(const Function *GV) {
  // For each function there is a separate subsection
  // which holds the PC to file:line table.
  const MCSymbol *Fn = Asm->getSymbol(GV);
  assert(Fn);

  const FunctionInfo &FI = FnDebugInfo[GV];
  if (FI.Instrs.empty())
    return;
  assert(FI.End && "Don't know where the function ends?");

  StringRef GVName = GV->getName();
  StringRef FuncName;
  if (auto *SP = getDISubprogram(GV))
    FuncName = SP->getDisplayName();

  // FIXME Clang currently sets DisplayName to "bar" for a C++
  // "namespace_foo::bar" function, see PR21528.  Luckily, dbghelp.dll is trying
  // to demangle display names anyways, so let's just put a mangled name into
  // the symbols subsection until Clang gives us what we need.
  if (GVName.startswith("\01?"))
    FuncName = GVName.substr(1);
  // Emit a symbol subsection, required by VS2012+ to find function boundaries.
  MCSymbol *SymbolsBegin = Asm->MMI->getContext().createTempSymbol(),
           *SymbolsEnd = Asm->MMI->getContext().createTempSymbol();
  Asm->OutStreamer->AddComment("Symbol subsection for " + Twine(FuncName));
  Asm->EmitInt32(COFF::DEBUG_SYMBOL_SUBSECTION);
  EmitLabelDiff(*Asm->OutStreamer, SymbolsBegin, SymbolsEnd);
  Asm->OutStreamer->EmitLabel(SymbolsBegin);
  {
    MCSymbol *ProcSegmentBegin = Asm->MMI->getContext().createTempSymbol(),
             *ProcSegmentEnd = Asm->MMI->getContext().createTempSymbol();
    EmitLabelDiff(*Asm->OutStreamer, ProcSegmentBegin, ProcSegmentEnd, 2);
    Asm->OutStreamer->EmitLabel(ProcSegmentBegin);

    Asm->EmitInt16(COFF::DEBUG_SYMBOL_TYPE_PROC_START);
    // Some bytes of this segment don't seem to be required for basic debugging,
    // so just fill them with zeroes.
    Asm->OutStreamer->EmitFill(12, 0);
    // This is the important bit that tells the debugger where the function
    // code is located and what's its size:
    EmitLabelDiff(*Asm->OutStreamer, Fn, FI.End);
    Asm->OutStreamer->EmitFill(12, 0);
    Asm->OutStreamer->EmitCOFFSecRel32(Fn);
    Asm->OutStreamer->EmitCOFFSectionIndex(Fn);
    Asm->EmitInt8(0);
    // Emit the function display name as a null-terminated string.
    Asm->OutStreamer->EmitBytes(FuncName);
    Asm->EmitInt8(0);
    Asm->OutStreamer->EmitLabel(ProcSegmentEnd);

    // We're done with this function.
    Asm->EmitInt16(0x0002);
    Asm->EmitInt16(COFF::DEBUG_SYMBOL_TYPE_PROC_END);
  }
  Asm->OutStreamer->EmitLabel(SymbolsEnd);
  // Every subsection must be aligned to a 4-byte boundary.
  Asm->OutStreamer->EmitFill((-FuncName.size()) % 4, 0);

  // PCs/Instructions are grouped into segments sharing the same filename.
  // Pre-calculate the lengths (in instructions) of these segments and store
  // them in a map for convenience.  Each index in the map is the sequential
  // number of the respective instruction that starts a new segment.
  DenseMap<size_t, size_t> FilenameSegmentLengths;
  size_t LastSegmentEnd = 0;
  StringRef PrevFilename = InstrInfo[FI.Instrs[0]].Filename;
  for (size_t J = 1, F = FI.Instrs.size(); J != F; ++J) {
    if (PrevFilename == InstrInfo[FI.Instrs[J]].Filename)
      continue;
    FilenameSegmentLengths[LastSegmentEnd] = J - LastSegmentEnd;
    LastSegmentEnd = J;
    PrevFilename = InstrInfo[FI.Instrs[J]].Filename;
  }
  FilenameSegmentLengths[LastSegmentEnd] = FI.Instrs.size() - LastSegmentEnd;

  // Emit a line table subsection, requred to do PC-to-file:line lookup.
  Asm->OutStreamer->AddComment("Line table subsection for " + Twine(FuncName));
  Asm->EmitInt32(COFF::DEBUG_LINE_TABLE_SUBSECTION);
  MCSymbol *LineTableBegin = Asm->MMI->getContext().createTempSymbol(),
           *LineTableEnd = Asm->MMI->getContext().createTempSymbol();
  EmitLabelDiff(*Asm->OutStreamer, LineTableBegin, LineTableEnd);
  Asm->OutStreamer->EmitLabel(LineTableBegin);

  // Identify the function this subsection is for.
  Asm->OutStreamer->EmitCOFFSecRel32(Fn);
  Asm->OutStreamer->EmitCOFFSectionIndex(Fn);
  // Insert flags after a 16-bit section index.
  Asm->EmitInt16(COFF::DEBUG_LINE_TABLES_HAVE_COLUMN_RECORDS);

  // Length of the function's code, in bytes.
  EmitLabelDiff(*Asm->OutStreamer, Fn, FI.End);

  // PC-to-linenumber lookup table:
  MCSymbol *FileSegmentEnd = nullptr;

  // The start of the last segment:
  size_t LastSegmentStart = 0;

  auto FinishPreviousChunk = [&] {
    if (!FileSegmentEnd)
      return;
    for (size_t ColSegI = LastSegmentStart,
                ColSegEnd = ColSegI + FilenameSegmentLengths[LastSegmentStart];
         ColSegI != ColSegEnd; ++ColSegI) {
      unsigned ColumnNumber = InstrInfo[FI.Instrs[ColSegI]].ColumnNumber;
      Asm->EmitInt16(ColumnNumber); // Start column
      Asm->EmitInt16(ColumnNumber); // End column
    }
    Asm->OutStreamer->EmitLabel(FileSegmentEnd);
  };

  for (size_t J = 0, F = FI.Instrs.size(); J != F; ++J) {
    MCSymbol *Instr = FI.Instrs[J];
    assert(InstrInfo.count(Instr));

    if (FilenameSegmentLengths.count(J)) {
      // We came to a beginning of a new filename segment.
      FinishPreviousChunk();
      StringRef CurFilename = InstrInfo[FI.Instrs[J]].Filename;
      assert(FileNameRegistry.Infos.count(CurFilename));
      size_t IndexInStringTable =
          FileNameRegistry.Infos[CurFilename].FilenameID;
      // Each segment starts with the offset of the filename
      // in the string table.
      Asm->OutStreamer->AddComment(
          "Segment for file '" + Twine(CurFilename) + "' begins");
      MCSymbol *FileSegmentBegin = Asm->MMI->getContext().createTempSymbol();
      Asm->OutStreamer->EmitLabel(FileSegmentBegin);
      Asm->EmitInt32(8 * IndexInStringTable);

      // Number of PC records in the lookup table.
      size_t SegmentLength = FilenameSegmentLengths[J];
      Asm->EmitInt32(SegmentLength);

      // Full size of the segment for this filename, including the prev two
      // records.
      FileSegmentEnd = Asm->MMI->getContext().createTempSymbol();
      EmitLabelDiff(*Asm->OutStreamer, FileSegmentBegin, FileSegmentEnd);
      LastSegmentStart = J;
    }

    // The first PC with the given linenumber and the linenumber itself.
    EmitLabelDiff(*Asm->OutStreamer, Fn, Instr);
    Asm->EmitInt32(InstrInfo[Instr].LineNumber);
  }

  FinishPreviousChunk();
  Asm->OutStreamer->EmitLabel(LineTableEnd);
}

void WinCodeViewLineTables::beginFunction(const MachineFunction *MF) {
  assert(!CurFn && "Can't process two functions at once!");

  if (!Asm || !Asm->MMI->hasDebugInfo())
    return;

  const Function *GV = MF->getFunction();
  assert(FnDebugInfo.count(GV) == false);
  VisitedFunctions.push_back(GV);
  CurFn = &FnDebugInfo[GV];

  // Find the end of the function prolog.
  // FIXME: is there a simpler a way to do this? Can we just search
  // for the first instruction of the function, not the last of the prolog?
  DebugLoc PrologEndLoc;
  bool EmptyPrologue = true;
  for (const auto &MBB : *MF) {
    if (PrologEndLoc)
      break;
    for (const auto &MI : MBB) {
      if (MI.isDebugValue())
        continue;

      // First known non-DBG_VALUE and non-frame setup location marks
      // the beginning of the function body.
      // FIXME: do we need the first subcondition?
      if (!MI.getFlag(MachineInstr::FrameSetup) && MI.getDebugLoc()) {
        PrologEndLoc = MI.getDebugLoc();
        break;
      }
      EmptyPrologue = false;
    }
  }
  // Record beginning of function if we have a non-empty prologue.
  if (PrologEndLoc && !EmptyPrologue) {
    DebugLoc FnStartDL = PrologEndLoc.getFnDebugLoc();
    maybeRecordLocation(FnStartDL, MF);
  }
}

void WinCodeViewLineTables::endFunction(const MachineFunction *MF) {
  if (!Asm || !CurFn)  // We haven't created any debug info for this function.
    return;

  const Function *GV = MF->getFunction();
  assert(FnDebugInfo.count(GV));
  assert(CurFn == &FnDebugInfo[GV]);

  if (CurFn->Instrs.empty()) {
    FnDebugInfo.erase(GV);
    VisitedFunctions.pop_back();
  } else {
    CurFn->End = Asm->getFunctionEnd();
  }
  CurFn = nullptr;
}

void WinCodeViewLineTables::beginInstruction(const MachineInstr *MI) {
  // Ignore DBG_VALUE locations and function prologue.
  if (!Asm || MI->isDebugValue() || MI->getFlag(MachineInstr::FrameSetup))
    return;
  DebugLoc DL = MI->getDebugLoc();
  if (DL == PrevInstLoc || !DL)
    return;
  maybeRecordLocation(DL, Asm->MF);
}
}
