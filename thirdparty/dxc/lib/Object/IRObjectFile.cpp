//===- IRObjectFile.cpp - IR object file implementation ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Part of the IRObjectFile class implementation.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/IRObjectFile.h"
#include "RecordStreamer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/GVMaterializer.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;
using namespace object;

IRObjectFile::IRObjectFile(MemoryBufferRef Object, std::unique_ptr<Module> Mod)
    : SymbolicFile(Binary::ID_IR, Object), M(std::move(Mod)) {
  Mang.reset(new Mangler());
#if 1 // HLSL Change - remove dependency on machine-specific concepts here
  return;
#else
  const std::string &InlineAsm = M->getModuleInlineAsm();
  if (InlineAsm.empty())
    return;

  Triple TT(M->getTargetTriple());
  std::string Err;
  const Target *T = TargetRegistry::lookupTarget(TT.str(), Err);
  if (!T)
    return;

  std::unique_ptr<MCRegisterInfo> MRI(T->createMCRegInfo(TT.str()));
  if (!MRI)
    return;

  std::unique_ptr<MCAsmInfo> MAI(T->createMCAsmInfo(*MRI, TT.str()));
  if (!MAI)
    return;

  std::unique_ptr<MCSubtargetInfo> STI(
      T->createMCSubtargetInfo(TT.str(), "", ""));
  if (!STI)
    return;

  std::unique_ptr<MCInstrInfo> MCII(T->createMCInstrInfo());
  if (!MCII)
    return;

  MCObjectFileInfo MOFI;
  MCContext MCCtx(MAI.get(), MRI.get(), &MOFI);
  MOFI.InitMCObjectFileInfo(TT, Reloc::Default, CodeModel::Default, MCCtx);
  std::unique_ptr<RecordStreamer> Streamer(new RecordStreamer(MCCtx));
  T->createNullTargetStreamer(*Streamer);

  std::unique_ptr<MemoryBuffer> Buffer(MemoryBuffer::getMemBuffer(InlineAsm));
  SourceMgr SrcMgr;
  SrcMgr.AddNewSourceBuffer(std::move(Buffer), SMLoc());
  std::unique_ptr<MCAsmParser> Parser(
      createMCAsmParser(SrcMgr, MCCtx, *Streamer, *MAI));

  MCTargetOptions MCOptions;
  std::unique_ptr<MCTargetAsmParser> TAP(
      T->createMCAsmParser(*STI, *Parser, *MCII, MCOptions));
  if (!TAP)
    return;

  Parser->setTargetParser(*TAP);
  if (Parser->Run(false))
    return;

  for (auto &KV : *Streamer) {
    StringRef Key = KV.first();
    RecordStreamer::State Value = KV.second;
    uint32_t Res = BasicSymbolRef::SF_None;
    switch (Value) {
    case RecordStreamer::NeverSeen:
      llvm_unreachable("foo");
    case RecordStreamer::DefinedGlobal:
      Res |= BasicSymbolRef::SF_Global;
      break;
    case RecordStreamer::Defined:
      break;
    case RecordStreamer::Global:
    case RecordStreamer::Used:
      Res |= BasicSymbolRef::SF_Undefined;
      Res |= BasicSymbolRef::SF_Global;
      break;
    }
    AsmSymbols.push_back(
        std::make_pair<std::string, uint32_t>(Key, std::move(Res)));
  }
#endif // HLSL Change - remove dependency on machine-specific concepts here
}

IRObjectFile::~IRObjectFile() {
 }

static GlobalValue *getGV(DataRefImpl &Symb) {
  if ((Symb.p & 3) == 3)
    return nullptr;

  return reinterpret_cast<GlobalValue*>(Symb.p & ~uintptr_t(3));
}

static uintptr_t skipEmpty(Module::const_alias_iterator I, const Module &M) {
  if (I == M.alias_end())
    return 3;
  const GlobalValue *GV = &*I;
  return reinterpret_cast<uintptr_t>(GV) | 2;
}

static uintptr_t skipEmpty(Module::const_global_iterator I, const Module &M) {
  if (I == M.global_end())
    return skipEmpty(M.alias_begin(), M);
  const GlobalValue *GV = &*I;
  return reinterpret_cast<uintptr_t>(GV) | 1;
}

static uintptr_t skipEmpty(Module::const_iterator I, const Module &M) {
  if (I == M.end())
    return skipEmpty(M.global_begin(), M);
  const GlobalValue *GV = &*I;
  return reinterpret_cast<uintptr_t>(GV) | 0;
}

static unsigned getAsmSymIndex(DataRefImpl Symb) {
  assert((Symb.p & uintptr_t(3)) == 3);
  uintptr_t Index = Symb.p & ~uintptr_t(3);
  Index >>= 2;
  return Index;
}

void IRObjectFile::moveSymbolNext(DataRefImpl &Symb) const {
  const GlobalValue *GV = getGV(Symb);
  uintptr_t Res;

  switch (Symb.p & 3) {
  case 0: {
    Module::const_iterator Iter(static_cast<const Function*>(GV));
    ++Iter;
    Res = skipEmpty(Iter, *M);
    break;
  }
  case 1: {
    Module::const_global_iterator Iter(static_cast<const GlobalVariable*>(GV));
    ++Iter;
    Res = skipEmpty(Iter, *M);
    break;
  }
  case 2: {
    Module::const_alias_iterator Iter(static_cast<const GlobalAlias*>(GV));
    ++Iter;
    Res = skipEmpty(Iter, *M);
    break;
  }
  case 3: {
    unsigned Index = getAsmSymIndex(Symb);
    assert(Index < AsmSymbols.size());
    ++Index;
    Res = (Index << 2) | 3;
    break;
  }
  default:
    llvm_unreachable("unreachable case");
  }

  Symb.p = Res;
}

std::error_code IRObjectFile::printSymbolName(raw_ostream &OS,
                                              DataRefImpl Symb) const {
  const GlobalValue *GV = getGV(Symb);
  if (!GV) {
    unsigned Index = getAsmSymIndex(Symb);
    assert(Index <= AsmSymbols.size());
    OS << AsmSymbols[Index].first;
    return std::error_code();
  }

  if (GV->hasDLLImportStorageClass())
    OS << "__imp_";

  if (Mang)
    Mang->getNameWithPrefix(OS, GV, false);
  else
    OS << GV->getName();

  return std::error_code();
}

uint32_t IRObjectFile::getSymbolFlags(DataRefImpl Symb) const {
  const GlobalValue *GV = getGV(Symb);

  if (!GV) {
    unsigned Index = getAsmSymIndex(Symb);
    assert(Index <= AsmSymbols.size());
    return AsmSymbols[Index].second;
  }

  uint32_t Res = BasicSymbolRef::SF_None;
  if (GV->isDeclarationForLinker())
    Res |= BasicSymbolRef::SF_Undefined;
  if (GV->hasPrivateLinkage())
    Res |= BasicSymbolRef::SF_FormatSpecific;
  if (!GV->hasLocalLinkage())
    Res |= BasicSymbolRef::SF_Global;
  if (GV->hasCommonLinkage())
    Res |= BasicSymbolRef::SF_Common;
  if (GV->hasLinkOnceLinkage() || GV->hasWeakLinkage())
    Res |= BasicSymbolRef::SF_Weak;

  if (GV->getName().startswith("llvm."))
    Res |= BasicSymbolRef::SF_FormatSpecific;
  else if (auto *Var = dyn_cast<GlobalVariable>(GV)) {
    if (Var->getSection() == StringRef("llvm.metadata"))
      Res |= BasicSymbolRef::SF_FormatSpecific;
  }

  return Res;
}

GlobalValue *IRObjectFile::getSymbolGV(DataRefImpl Symb) { return getGV(Symb); }

std::unique_ptr<Module> IRObjectFile::takeModule() { return std::move(M); }

basic_symbol_iterator IRObjectFile::symbol_begin_impl() const {
  Module::const_iterator I = M->begin();
  DataRefImpl Ret;
  Ret.p = skipEmpty(I, *M);
  return basic_symbol_iterator(BasicSymbolRef(Ret, this));
}

basic_symbol_iterator IRObjectFile::symbol_end_impl() const {
  DataRefImpl Ret;
  uint64_t NumAsm = AsmSymbols.size();
  NumAsm <<= 2;
  Ret.p = 3 | NumAsm;
  return basic_symbol_iterator(BasicSymbolRef(Ret, this));
}

ErrorOr<MemoryBufferRef> IRObjectFile::findBitcodeInObject(const ObjectFile &Obj) {
  for (const SectionRef &Sec : Obj.sections()) {
    StringRef SecName;
    if (std::error_code EC = Sec.getName(SecName))
      return EC;
    if (SecName == ".llvmbc") {
      StringRef SecContents;
      if (std::error_code EC = Sec.getContents(SecContents))
        return EC;
      return MemoryBufferRef(SecContents, Obj.getFileName());
    }
  }

  return object_error::bitcode_section_not_found;
}

ErrorOr<MemoryBufferRef> IRObjectFile::findBitcodeInMemBuffer(MemoryBufferRef Object) {
  sys::fs::file_magic Type = sys::fs::identify_magic(Object.getBuffer());
  switch (Type) {
  case sys::fs::file_magic::bitcode:
    return Object;
  case sys::fs::file_magic::elf_relocatable:
  case sys::fs::file_magic::macho_object:
  case sys::fs::file_magic::coff_object: {
    ErrorOr<std::unique_ptr<ObjectFile>> ObjFile =
        ObjectFile::createObjectFile(Object, Type);
    if (!ObjFile)
      return ObjFile.getError();
    return findBitcodeInObject(*ObjFile->get());
  }
  default:
    return object_error::invalid_file_type;
  }
}

ErrorOr<std::unique_ptr<IRObjectFile>>
llvm::object::IRObjectFile::create(MemoryBufferRef Object,
                                   LLVMContext &Context) {
  ErrorOr<MemoryBufferRef> BCOrErr = findBitcodeInMemBuffer(Object);
  if (!BCOrErr)
    return BCOrErr.getError();

  std::unique_ptr<MemoryBuffer> Buff(
      MemoryBuffer::getMemBuffer(BCOrErr.get(), false));

  ErrorOr<std::unique_ptr<Module>> MOrErr =
      getLazyBitcodeModule(std::move(Buff), Context, nullptr,
                           /*ShouldLazyLoadMetadata*/ true);
  if (std::error_code EC = MOrErr.getError())
    return EC;

  std::unique_ptr<Module> &M = MOrErr.get();
  return llvm::make_unique<IRObjectFile>(Object, std::move(M));
}
