//===-- LTOModule.cpp - LLVM Link Time Optimizer --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Link Time Optimization library. This library is
// intended to be used by linker to optimize code at link time.
//
//===----------------------------------------------------------------------===//

#include "llvm/LTO/LTOModule.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/CodeGen/Analysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Mangler.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCSection.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCTargetAsmParser.h"
#include "llvm/MC/SubtargetFeature.h"
#include "llvm/Object/IRObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetLowering.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetRegisterInfo.h"
#include "llvm/Target/TargetSubtargetInfo.h"
#include "llvm/Transforms/Utils/GlobalStatus.h"
#include <system_error>
using namespace llvm;
using namespace llvm::object;

LTOModule::LTOModule(std::unique_ptr<object::IRObjectFile> Obj,
                     llvm::TargetMachine *TM)
    : IRFile(std::move(Obj)), _target(TM) {}

LTOModule::LTOModule(std::unique_ptr<object::IRObjectFile> Obj,
                     llvm::TargetMachine *TM,
                     std::unique_ptr<LLVMContext> Context)
    : OwnedContext(std::move(Context)), IRFile(std::move(Obj)), _target(TM) {}

LTOModule::~LTOModule() {}

/// isBitcodeFile - Returns 'true' if the file (or memory contents) is LLVM
/// bitcode.
bool LTOModule::isBitcodeFile(const void *Mem, size_t Length) {
  ErrorOr<MemoryBufferRef> BCData = IRObjectFile::findBitcodeInMemBuffer(
      MemoryBufferRef(StringRef((const char *)Mem, Length), "<mem>"));
  return bool(BCData);
}

bool LTOModule::isBitcodeFile(const char *Path) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(Path);
  if (!BufferOrErr)
    return false;

  ErrorOr<MemoryBufferRef> BCData = IRObjectFile::findBitcodeInMemBuffer(
      BufferOrErr.get()->getMemBufferRef());
  return bool(BCData);
}

bool LTOModule::isBitcodeForTarget(MemoryBuffer *Buffer,
                                   StringRef TriplePrefix) {
  ErrorOr<MemoryBufferRef> BCOrErr =
      IRObjectFile::findBitcodeInMemBuffer(Buffer->getMemBufferRef());
  if (!BCOrErr)
    return false;
  LLVMContext Context;
  std::string Triple = getBitcodeTargetTriple(*BCOrErr, Context);
  return StringRef(Triple).startswith(TriplePrefix);
}

LTOModule *LTOModule::createFromFile(const char *path, TargetOptions options,
                                     std::string &errMsg) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(path);
  if (std::error_code EC = BufferOrErr.getError()) {
    errMsg = EC.message();
    return nullptr;
  }
  std::unique_ptr<MemoryBuffer> Buffer = std::move(BufferOrErr.get());
  return makeLTOModule(Buffer->getMemBufferRef(), options, errMsg,
                       &getGlobalContext());
}

LTOModule *LTOModule::createFromOpenFile(int fd, const char *path, size_t size,
                                         TargetOptions options,
                                         std::string &errMsg) {
  return createFromOpenFileSlice(fd, path, size, 0, options, errMsg);
}

LTOModule *LTOModule::createFromOpenFileSlice(int fd, const char *path,
                                              size_t map_size, off_t offset,
                                              TargetOptions options,
                                              std::string &errMsg) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getOpenFileSlice(fd, path, map_size, offset);
  if (std::error_code EC = BufferOrErr.getError()) {
    errMsg = EC.message();
    return nullptr;
  }
  std::unique_ptr<MemoryBuffer> Buffer = std::move(BufferOrErr.get());
  return makeLTOModule(Buffer->getMemBufferRef(), options, errMsg,
                       &getGlobalContext());
}

LTOModule *LTOModule::createFromBuffer(const void *mem, size_t length,
                                       TargetOptions options,
                                       std::string &errMsg, StringRef path) {
  return createInContext(mem, length, options, errMsg, path,
                         &getGlobalContext());
}

LTOModule *LTOModule::createInLocalContext(const void *mem, size_t length,
                                           TargetOptions options,
                                           std::string &errMsg,
                                           StringRef path) {
  return createInContext(mem, length, options, errMsg, path, nullptr);
}

LTOModule *LTOModule::createInContext(const void *mem, size_t length,
                                      TargetOptions options,
                                      std::string &errMsg, StringRef path,
                                      LLVMContext *Context) {
  StringRef Data((const char *)mem, length);
  MemoryBufferRef Buffer(Data, path);
  return makeLTOModule(Buffer, options, errMsg, Context);
}

static std::unique_ptr<Module> parseBitcodeFileImpl(MemoryBufferRef Buffer,
                                                    LLVMContext &Context,
                                                    bool ShouldBeLazy,
                                                    std::string &ErrMsg) {

  // Find the buffer.
  ErrorOr<MemoryBufferRef> MBOrErr =
      IRObjectFile::findBitcodeInMemBuffer(Buffer);
  if (std::error_code EC = MBOrErr.getError()) {
    ErrMsg = EC.message();
    return nullptr;
  }

  std::function<void(const DiagnosticInfo &)> DiagnosticHandler =
      [&ErrMsg](const DiagnosticInfo &DI) {
        raw_string_ostream Stream(ErrMsg);
        DiagnosticPrinterRawOStream DP(Stream);
        DI.print(DP);
      };

  if (!ShouldBeLazy) {
    // Parse the full file.
    ErrorOr<std::unique_ptr<Module>> M =
        parseBitcodeFile(*MBOrErr, Context, DiagnosticHandler);
    if (!M)
      return nullptr;
    return std::move(*M);
  }

  // Parse lazily.
  std::unique_ptr<MemoryBuffer> LightweightBuf =
      MemoryBuffer::getMemBuffer(*MBOrErr, false);
  ErrorOr<std::unique_ptr<Module>> M =
      getLazyBitcodeModule(std::move(LightweightBuf), Context,
                           DiagnosticHandler, true /*ShouldLazyLoadMetadata*/);
  if (!M)
    return nullptr;
  return std::move(*M);
}

LTOModule *LTOModule::makeLTOModule(MemoryBufferRef Buffer,
                                    TargetOptions options, std::string &errMsg,
                                    LLVMContext *Context) {
  std::unique_ptr<LLVMContext> OwnedContext;
  if (!Context) {
    OwnedContext = llvm::make_unique<LLVMContext>();
    Context = OwnedContext.get();
  }

  // If we own a context, we know this is being used only for symbol
  // extraction, not linking.  Be lazy in that case.
  std::unique_ptr<Module> M = parseBitcodeFileImpl(
      Buffer, *Context,
      /* ShouldBeLazy */ static_cast<bool>(OwnedContext), errMsg);
  if (!M)
    return nullptr;

  std::string TripleStr = M->getTargetTriple();
  if (TripleStr.empty())
    TripleStr = sys::getDefaultTargetTriple();
  llvm::Triple Triple(TripleStr);

  // find machine architecture for this module
  const Target *march = TargetRegistry::lookupTarget(TripleStr, errMsg);
  if (!march)
    return nullptr;

  // construct LTOModule, hand over ownership of module and target
  SubtargetFeatures Features;
  Features.getDefaultSubtargetFeatures(Triple);
  std::string FeatureStr = Features.getString();
  // Set a default CPU for Darwin triples.
  std::string CPU;
  if (Triple.isOSDarwin()) {
    if (Triple.getArch() == llvm::Triple::x86_64)
      CPU = "core2";
    else if (Triple.getArch() == llvm::Triple::x86)
      CPU = "yonah";
    else if (Triple.getArch() == llvm::Triple::aarch64)
      CPU = "cyclone";
  }

  TargetMachine *target = march->createTargetMachine(TripleStr, CPU, FeatureStr,
                                                     options);
  M->setDataLayout(*target->getDataLayout());

  std::unique_ptr<object::IRObjectFile> IRObj(
      new object::IRObjectFile(Buffer, std::move(M)));

  LTOModule *Ret;
  if (OwnedContext)
    Ret = new LTOModule(std::move(IRObj), target, std::move(OwnedContext));
  else
    Ret = new LTOModule(std::move(IRObj), target);

  if (Ret->parseSymbols(errMsg)) {
    delete Ret;
    return nullptr;
  }

  Ret->parseMetadata();

  return Ret;
}

/// Create a MemoryBuffer from a memory range with an optional name.
std::unique_ptr<MemoryBuffer>
LTOModule::makeBuffer(const void *mem, size_t length, StringRef name) {
  const char *startPtr = (const char*)mem;
  return MemoryBuffer::getMemBuffer(StringRef(startPtr, length), name, false);
}

/// objcClassNameFromExpression - Get string that the data pointer points to.
bool
LTOModule::objcClassNameFromExpression(const Constant *c, std::string &name) {
  if (const ConstantExpr *ce = dyn_cast<ConstantExpr>(c)) {
    Constant *op = ce->getOperand(0);
    if (GlobalVariable *gvn = dyn_cast<GlobalVariable>(op)) {
      Constant *cn = gvn->getInitializer();
      if (ConstantDataArray *ca = dyn_cast<ConstantDataArray>(cn)) {
        if (ca->isCString()) {
          name = (".objc_class_name_" + ca->getAsCString()).str();
          return true;
        }
      }
    }
  }
  return false;
}

/// addObjCClass - Parse i386/ppc ObjC class data structure.
void LTOModule::addObjCClass(const GlobalVariable *clgv) {
  const ConstantStruct *c = dyn_cast<ConstantStruct>(clgv->getInitializer());
  if (!c) return;

  // second slot in __OBJC,__class is pointer to superclass name
  std::string superclassName;
  if (objcClassNameFromExpression(c->getOperand(1), superclassName)) {
    auto IterBool =
        _undefines.insert(std::make_pair(superclassName, NameAndAttributes()));
    if (IterBool.second) {
      NameAndAttributes &info = IterBool.first->second;
      info.name = IterBool.first->first().data();
      info.attributes = LTO_SYMBOL_DEFINITION_UNDEFINED;
      info.isFunction = false;
      info.symbol = clgv;
    }
  }

  // third slot in __OBJC,__class is pointer to class name
  std::string className;
  if (objcClassNameFromExpression(c->getOperand(2), className)) {
    auto Iter = _defines.insert(className).first;

    NameAndAttributes info;
    info.name = Iter->first().data();
    info.attributes = LTO_SYMBOL_PERMISSIONS_DATA |
      LTO_SYMBOL_DEFINITION_REGULAR | LTO_SYMBOL_SCOPE_DEFAULT;
    info.isFunction = false;
    info.symbol = clgv;
    _symbols.push_back(info);
  }
}

/// addObjCCategory - Parse i386/ppc ObjC category data structure.
void LTOModule::addObjCCategory(const GlobalVariable *clgv) {
  const ConstantStruct *c = dyn_cast<ConstantStruct>(clgv->getInitializer());
  if (!c) return;

  // second slot in __OBJC,__category is pointer to target class name
  std::string targetclassName;
  if (!objcClassNameFromExpression(c->getOperand(1), targetclassName))
    return;

  auto IterBool =
      _undefines.insert(std::make_pair(targetclassName, NameAndAttributes()));

  if (!IterBool.second)
    return;

  NameAndAttributes &info = IterBool.first->second;
  info.name = IterBool.first->first().data();
  info.attributes = LTO_SYMBOL_DEFINITION_UNDEFINED;
  info.isFunction = false;
  info.symbol = clgv;
}

/// addObjCClassRef - Parse i386/ppc ObjC class list data structure.
void LTOModule::addObjCClassRef(const GlobalVariable *clgv) {
  std::string targetclassName;
  if (!objcClassNameFromExpression(clgv->getInitializer(), targetclassName))
    return;

  auto IterBool =
      _undefines.insert(std::make_pair(targetclassName, NameAndAttributes()));

  if (!IterBool.second)
    return;

  NameAndAttributes &info = IterBool.first->second;
  info.name = IterBool.first->first().data();
  info.attributes = LTO_SYMBOL_DEFINITION_UNDEFINED;
  info.isFunction = false;
  info.symbol = clgv;
}

void LTOModule::addDefinedDataSymbol(const object::BasicSymbolRef &Sym) {
  SmallString<64> Buffer;
  {
    raw_svector_ostream OS(Buffer);
    Sym.printName(OS);
  }

  const GlobalValue *V = IRFile->getSymbolGV(Sym.getRawDataRefImpl());
  addDefinedDataSymbol(Buffer.c_str(), V);
}

void LTOModule::addDefinedDataSymbol(const char *Name, const GlobalValue *v) {
  // Add to list of defined symbols.
  addDefinedSymbol(Name, v, false);

  if (!v->hasSection() /* || !isTargetDarwin */)
    return;

  // Special case i386/ppc ObjC data structures in magic sections:
  // The issue is that the old ObjC object format did some strange
  // contortions to avoid real linker symbols.  For instance, the
  // ObjC class data structure is allocated statically in the executable
  // that defines that class.  That data structures contains a pointer to
  // its superclass.  But instead of just initializing that part of the
  // struct to the address of its superclass, and letting the static and
  // dynamic linkers do the rest, the runtime works by having that field
  // instead point to a C-string that is the name of the superclass.
  // At runtime the objc initialization updates that pointer and sets
  // it to point to the actual super class.  As far as the linker
  // knows it is just a pointer to a string.  But then someone wanted the
  // linker to issue errors at build time if the superclass was not found.
  // So they figured out a way in mach-o object format to use an absolute
  // symbols (.objc_class_name_Foo = 0) and a floating reference
  // (.reference .objc_class_name_Bar) to cause the linker into erroring when
  // a class was missing.
  // The following synthesizes the implicit .objc_* symbols for the linker
  // from the ObjC data structures generated by the front end.

  // special case if this data blob is an ObjC class definition
  std::string Section = v->getSection();
  if (Section.compare(0, 15, "__OBJC,__class,") == 0) {
    if (const GlobalVariable *gv = dyn_cast<GlobalVariable>(v)) {
      addObjCClass(gv);
    }
  }

  // special case if this data blob is an ObjC category definition
  else if (Section.compare(0, 18, "__OBJC,__category,") == 0) {
    if (const GlobalVariable *gv = dyn_cast<GlobalVariable>(v)) {
      addObjCCategory(gv);
    }
  }

  // special case if this data blob is the list of referenced classes
  else if (Section.compare(0, 18, "__OBJC,__cls_refs,") == 0) {
    if (const GlobalVariable *gv = dyn_cast<GlobalVariable>(v)) {
      addObjCClassRef(gv);
    }
  }
}

void LTOModule::addDefinedFunctionSymbol(const object::BasicSymbolRef &Sym) {
  SmallString<64> Buffer;
  {
    raw_svector_ostream OS(Buffer);
    Sym.printName(OS);
  }

  const Function *F =
      cast<Function>(IRFile->getSymbolGV(Sym.getRawDataRefImpl()));
  addDefinedFunctionSymbol(Buffer.c_str(), F);
}

void LTOModule::addDefinedFunctionSymbol(const char *Name, const Function *F) {
  // add to list of defined symbols
  addDefinedSymbol(Name, F, true);
}

void LTOModule::addDefinedSymbol(const char *Name, const GlobalValue *def,
                                 bool isFunction) {
  // set alignment part log2() can have rounding errors
  uint32_t align = def->getAlignment();
  uint32_t attr = align ? countTrailingZeros(align) : 0;

  // set permissions part
  if (isFunction) {
    attr |= LTO_SYMBOL_PERMISSIONS_CODE;
  } else {
    const GlobalVariable *gv = dyn_cast<GlobalVariable>(def);
    if (gv && gv->isConstant())
      attr |= LTO_SYMBOL_PERMISSIONS_RODATA;
    else
      attr |= LTO_SYMBOL_PERMISSIONS_DATA;
  }

  // set definition part
  if (def->hasWeakLinkage() || def->hasLinkOnceLinkage())
    attr |= LTO_SYMBOL_DEFINITION_WEAK;
  else if (def->hasCommonLinkage())
    attr |= LTO_SYMBOL_DEFINITION_TENTATIVE;
  else
    attr |= LTO_SYMBOL_DEFINITION_REGULAR;

  // set scope part
  if (def->hasLocalLinkage())
    // Ignore visibility if linkage is local.
    attr |= LTO_SYMBOL_SCOPE_INTERNAL;
  else if (def->hasHiddenVisibility())
    attr |= LTO_SYMBOL_SCOPE_HIDDEN;
  else if (def->hasProtectedVisibility())
    attr |= LTO_SYMBOL_SCOPE_PROTECTED;
  else if (canBeOmittedFromSymbolTable(def))
    attr |= LTO_SYMBOL_SCOPE_DEFAULT_CAN_BE_HIDDEN;
  else
    attr |= LTO_SYMBOL_SCOPE_DEFAULT;

  if (def->hasComdat())
    attr |= LTO_SYMBOL_COMDAT;

  if (isa<GlobalAlias>(def))
    attr |= LTO_SYMBOL_ALIAS;

  auto Iter = _defines.insert(Name).first;

  // fill information structure
  NameAndAttributes info;
  StringRef NameRef = Iter->first();
  info.name = NameRef.data();
  assert(info.name[NameRef.size()] == '\0');
  info.attributes = attr;
  info.isFunction = isFunction;
  info.symbol = def;

  // add to table of symbols
  _symbols.push_back(info);
}

/// addAsmGlobalSymbol - Add a global symbol from module-level ASM to the
/// defined list.
void LTOModule::addAsmGlobalSymbol(const char *name,
                                   lto_symbol_attributes scope) {
  auto IterBool = _defines.insert(name);

  // only add new define if not already defined
  if (!IterBool.second)
    return;

  NameAndAttributes &info = _undefines[IterBool.first->first().data()];

  if (info.symbol == nullptr) {
    // FIXME: This is trying to take care of module ASM like this:
    //
    //   module asm ".zerofill __FOO, __foo, _bar_baz_qux, 0"
    //
    // but is gross and its mother dresses it funny. Have the ASM parser give us
    // more details for this type of situation so that we're not guessing so
    // much.

    // fill information structure
    info.name = IterBool.first->first().data();
    info.attributes =
      LTO_SYMBOL_PERMISSIONS_DATA | LTO_SYMBOL_DEFINITION_REGULAR | scope;
    info.isFunction = false;
    info.symbol = nullptr;

    // add to table of symbols
    _symbols.push_back(info);
    return;
  }

  if (info.isFunction)
    addDefinedFunctionSymbol(info.name, cast<Function>(info.symbol));
  else
    addDefinedDataSymbol(info.name, info.symbol);

  _symbols.back().attributes &= ~LTO_SYMBOL_SCOPE_MASK;
  _symbols.back().attributes |= scope;
}

/// addAsmGlobalSymbolUndef - Add a global symbol from module-level ASM to the
/// undefined list.
void LTOModule::addAsmGlobalSymbolUndef(const char *name) {
  auto IterBool = _undefines.insert(std::make_pair(name, NameAndAttributes()));

  _asm_undefines.push_back(IterBool.first->first().data());

  // we already have the symbol
  if (!IterBool.second)
    return;

  uint32_t attr = LTO_SYMBOL_DEFINITION_UNDEFINED;
  attr |= LTO_SYMBOL_SCOPE_DEFAULT;
  NameAndAttributes &info = IterBool.first->second;
  info.name = IterBool.first->first().data();
  info.attributes = attr;
  info.isFunction = false;
  info.symbol = nullptr;
}

/// Add a symbol which isn't defined just yet to a list to be resolved later.
void LTOModule::addPotentialUndefinedSymbol(const object::BasicSymbolRef &Sym,
                                            bool isFunc) {
  SmallString<64> name;
  {
    raw_svector_ostream OS(name);
    Sym.printName(OS);
  }

  auto IterBool = _undefines.insert(std::make_pair(name, NameAndAttributes()));

  // we already have the symbol
  if (!IterBool.second)
    return;

  NameAndAttributes &info = IterBool.first->second;

  info.name = IterBool.first->first().data();

  const GlobalValue *decl = IRFile->getSymbolGV(Sym.getRawDataRefImpl());

  if (decl->hasExternalWeakLinkage())
    info.attributes = LTO_SYMBOL_DEFINITION_WEAKUNDEF;
  else
    info.attributes = LTO_SYMBOL_DEFINITION_UNDEFINED;

  info.isFunction = isFunc;
  info.symbol = decl;
}

/// parseSymbols - Parse the symbols from the module and model-level ASM and add
/// them to either the defined or undefined lists.
bool LTOModule::parseSymbols(std::string &errMsg) {
  for (auto &Sym : IRFile->symbols()) {
    const GlobalValue *GV = IRFile->getSymbolGV(Sym.getRawDataRefImpl());
    uint32_t Flags = Sym.getFlags();
    if (Flags & object::BasicSymbolRef::SF_FormatSpecific)
      continue;

    bool IsUndefined = Flags & object::BasicSymbolRef::SF_Undefined;

    if (!GV) {
      SmallString<64> Buffer;
      {
        raw_svector_ostream OS(Buffer);
        Sym.printName(OS);
      }
      const char *Name = Buffer.c_str();

      if (IsUndefined)
        addAsmGlobalSymbolUndef(Name);
      else if (Flags & object::BasicSymbolRef::SF_Global)
        addAsmGlobalSymbol(Name, LTO_SYMBOL_SCOPE_DEFAULT);
      else
        addAsmGlobalSymbol(Name, LTO_SYMBOL_SCOPE_INTERNAL);
      continue;
    }

    auto *F = dyn_cast<Function>(GV);
    if (IsUndefined) {
      addPotentialUndefinedSymbol(Sym, F != nullptr);
      continue;
    }

    if (F) {
      addDefinedFunctionSymbol(Sym);
      continue;
    }

    if (isa<GlobalVariable>(GV)) {
      addDefinedDataSymbol(Sym);
      continue;
    }

    assert(isa<GlobalAlias>(GV));
    addDefinedDataSymbol(Sym);
  }

  // make symbols for all undefines
  for (StringMap<NameAndAttributes>::iterator u =_undefines.begin(),
         e = _undefines.end(); u != e; ++u) {
    // If this symbol also has a definition, then don't make an undefine because
    // it is a tentative definition.
    if (_defines.count(u->getKey())) continue;
    NameAndAttributes info = u->getValue();
    _symbols.push_back(info);
  }

  return false;
}

/// parseMetadata - Parse metadata from the module
void LTOModule::parseMetadata() {
  raw_string_ostream OS(LinkerOpts);

  // Linker Options
  if (Metadata *Val = getModule().getModuleFlag("Linker Options")) {
    MDNode *LinkerOptions = cast<MDNode>(Val);
    for (unsigned i = 0, e = LinkerOptions->getNumOperands(); i != e; ++i) {
      MDNode *MDOptions = cast<MDNode>(LinkerOptions->getOperand(i));
      for (unsigned ii = 0, ie = MDOptions->getNumOperands(); ii != ie; ++ii) {
        MDString *MDOption = cast<MDString>(MDOptions->getOperand(ii));
        OS << " " << MDOption->getString();
      }
    }
  }

  // Globals
  Mangler Mang;
  for (const NameAndAttributes &Sym : _symbols) {
    if (!Sym.symbol)
      continue;
    _target->getObjFileLowering()->emitLinkerFlagsForGlobal(OS, Sym.symbol,
                                                            Mang);
  }

  // Add other interesting metadata here.
}
