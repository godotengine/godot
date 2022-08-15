//===- BitcodeReader.cpp - Internal BitcodeReader implementation ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Bitcode/BitstreamReader.h"
#include "llvm/Bitcode/LLVMBitCodes.h"
#include "llvm/IR/AutoUpgrade.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/GVMaterializer.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/OperandTraits.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/DataStream.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <deque>
#include <unordered_set> // HLSL Change
#include "dxc/DXIL/DxilOperations.h"   // HLSL Change
using namespace llvm;

namespace {
enum {
  SWITCH_INST_MAGIC = 0x4B5 // May 2012 => 1205 => Hex
};

class BitcodeReaderValueList {
  std::vector<WeakVH> ValuePtrs;

  /// As we resolve forward-referenced constants, we add information about them
  /// to this vector.  This allows us to resolve them in bulk instead of
  /// resolving each reference at a time.  See the code in
  /// ResolveConstantForwardRefs for more information about this.
  ///
  /// The key of this vector is the placeholder constant, the value is the slot
  /// number that holds the resolved value.
  typedef std::vector<std::pair<Constant*, unsigned> > ResolveConstantsTy;
  ResolveConstantsTy ResolveConstants;
  LLVMContext &Context;
public:
  BitcodeReaderValueList(LLVMContext &C) : Context(C) {}
  ~BitcodeReaderValueList() {
    assert(ResolveConstants.empty() && "Constants not resolved?");
  }

  // vector compatibility methods
  unsigned size() const { return ValuePtrs.size(); }
  void resize(unsigned N) { ValuePtrs.resize(N); }
  void push_back(Value *V) { ValuePtrs.emplace_back(V); }

  void clear() {
    assert(ResolveConstants.empty() && "Constants not resolved?");
    ValuePtrs.clear();
  }

  Value *operator[](unsigned i) const {
    assert(i < ValuePtrs.size());
    return ValuePtrs[i];
  }

  Value *back() const { return ValuePtrs.back(); }
    void pop_back() { ValuePtrs.pop_back(); }
  bool empty() const { return ValuePtrs.empty(); }
  void shrinkTo(unsigned N) {
    assert(N <= size() && "Invalid shrinkTo request!");
    ValuePtrs.resize(N);
  }

  Constant *getConstantFwdRef(unsigned Idx, Type *Ty);
  Value *getValueFwdRef(unsigned Idx, Type *Ty);

  void assignValue(Value *V, unsigned Idx);

  /// Once all constants are read, this method bulk resolves any forward
  /// references.
  void resolveConstantForwardRefs();
};

class BitcodeReaderMDValueList {
  unsigned NumFwdRefs;
  bool AnyFwdRefs;
  unsigned MinFwdRef;
  unsigned MaxFwdRef;
  std::vector<TrackingMDRef> MDValuePtrs;

  LLVMContext &Context;
public:
  BitcodeReaderMDValueList(LLVMContext &C)
      : NumFwdRefs(0), AnyFwdRefs(false), Context(C) {}

  // vector compatibility methods
  unsigned size() const       { return MDValuePtrs.size(); }
  void resize(unsigned N)     { MDValuePtrs.resize(N); }
  void push_back(Metadata *MD) { MDValuePtrs.emplace_back(MD); }
  void clear()                { MDValuePtrs.clear();  }
  Metadata *back() const      { return MDValuePtrs.back(); }
  void pop_back()             { MDValuePtrs.pop_back(); }
  bool empty() const          { return MDValuePtrs.empty(); }

  Metadata *operator[](unsigned i) const {
    assert(i < MDValuePtrs.size());
    return MDValuePtrs[i];
  }

  void shrinkTo(unsigned N) {
    assert(N <= size() && "Invalid shrinkTo request!");
    MDValuePtrs.resize(N);
  }

  Metadata *getValueFwdRef(unsigned Idx);
  void assignValue(Metadata *MD, unsigned Idx);
  void tryToResolveCycles();
};

class BitcodeReader : public GVMaterializer {
  LLVMContext &Context;
  DiagnosticHandlerFunction DiagnosticHandler;
  Module *TheModule = nullptr;
  std::unique_ptr<MemoryBuffer> Buffer;
  std::unique_ptr<BitstreamReader> StreamFile;
  BitstreamCursor Stream;
  uint64_t NextUnreadBit = 0;
  bool SeenValueSymbolTable = false;

  std::vector<Type*> TypeList;
  BitcodeReaderValueList ValueList;
  BitcodeReaderMDValueList MDValueList;
  std::vector<Comdat *> ComdatList;
  SmallVector<Instruction *, 64> InstructionList;

  std::vector<std::pair<GlobalVariable*, unsigned> > GlobalInits;
  std::vector<std::pair<GlobalAlias*, unsigned> > AliasInits;
  std::vector<std::pair<Function*, unsigned> > FunctionPrefixes;
  std::vector<std::pair<Function*, unsigned> > FunctionPrologues;
  std::vector<std::pair<Function*, unsigned> > FunctionPersonalityFns;

  SmallVector<Instruction*, 64> InstsWithTBAATag;

  /// The set of attributes by index.  Index zero in the file is for null, and
  /// is thus not represented here.  As such all indices are off by one.
  std::vector<AttributeSet> MAttributes;

  /// \brief The set of attribute groups.
  std::map<unsigned, AttributeSet> MAttributeGroups;

  /// While parsing a function body, this is a list of the basic blocks for the
  /// function.
  std::vector<BasicBlock*> FunctionBBs;

  // When reading the module header, this list is populated with functions that
  // have bodies later in the file.
  std::vector<Function*> FunctionsWithBodies;

  // When intrinsic functions are encountered which require upgrading they are
  // stored here with their replacement function.
  typedef DenseMap<Function*, Function*> UpgradedIntrinsicMap;
  UpgradedIntrinsicMap UpgradedIntrinsics;

  // Map the bitcode's custom MDKind ID to the Module's MDKind ID.
  DenseMap<unsigned, unsigned> MDKindMap;

  // Several operations happen after the module header has been read, but
  // before function bodies are processed. This keeps track of whether
  // we've done this yet.
  bool SeenFirstFunctionBody = false;

  /// When function bodies are initially scanned, this map contains info about
  /// where to find deferred function body in the stream.
  DenseMap<Function*, uint64_t> DeferredFunctionInfo;

  /// When Metadata block is initially scanned when parsing the module, we may
  /// choose to defer parsing of the metadata. This vector contains info about
  /// which Metadata blocks are deferred.
  std::vector<uint64_t> DeferredMetadataInfo;

  /// These are basic blocks forward-referenced by block addresses.  They are
  /// inserted lazily into functions when they're loaded.  The basic block ID is
  /// its index into the vector.
  DenseMap<Function *, std::vector<BasicBlock *>> BasicBlockFwdRefs;
  std::deque<Function *> BasicBlockFwdRefQueue;

  /// Indicates that we are using a new encoding for instruction operands where
  /// most operands in the current FUNCTION_BLOCK are encoded relative to the
  /// instruction number, for a more compact encoding.  Some instruction
  /// operands are not relative to the instruction ID: basic block numbers, and
  /// types. Once the old style function blocks have been phased out, we would
  /// not need this flag.
  bool UseRelativeIDs = false;

  /// True if all functions will be materialized, negating the need to process
  /// (e.g.) blockaddress forward references.
  bool WillMaterializeAllForwardRefs = false;

  /// Functions that have block addresses taken.  This is usually empty.
  SmallPtrSet<const Function *, 4> BlockAddressesTaken;

  /// True if any Metadata block has been materialized.
  bool IsMetadataMaterialized = false;

  bool StripDebugInfo = false;

public:
  std::error_code error(BitcodeError E, const Twine &Message);
  std::error_code error(BitcodeError E);
  std::error_code error(const Twine &Message);

  BitcodeReader(std::unique_ptr<MemoryBuffer> &&Buffer, LLVMContext &Context, // HLSL Change: unique_ptr
                DiagnosticHandlerFunction DiagnosticHandler);
  BitcodeReader(LLVMContext &Context,
                DiagnosticHandlerFunction DiagnosticHandler);
  ~BitcodeReader() override { freeState(); }

  std::error_code materializeForwardReferencedFunctions();

  void freeState();

  void releaseBuffer();

  bool ShouldTrackBitstreamUsage = false; // HLSL Change
  BitstreamUseTracker Tracker; // HLSL Change

  bool isDematerializable(const GlobalValue *GV) const override;
  std::error_code materialize(GlobalValue *GV) override;
  std::error_code materializeModule(Module *M) override;
  std::vector<StructType *> getIdentifiedStructTypes() const override;
  void dematerialize(GlobalValue *GV) override;

  /// \brief Main interface to parsing a bitcode buffer.
  /// \returns true if an error occurred.
  std::error_code parseBitcodeInto(std::unique_ptr<DataStreamer> Streamer,
                                   Module *M,
                                   bool ShouldLazyLoadMetadata = false);

  /// \brief Cheap mechanism to just extract module triple
  /// \returns true if an error occurred.
  ErrorOr<std::string> parseTriple();

  static uint64_t decodeSignRotatedValue(uint64_t V);

  /// Materialize any deferred Metadata block.
  std::error_code materializeMetadata() override;

  void setStripDebugInfo() override;

private:
  std::vector<StructType *> IdentifiedStructTypes;
  StructType *createIdentifiedStructType(LLVMContext &Context, StringRef Name);
  StructType *createIdentifiedStructType(LLVMContext &Context);

  Type *getTypeByID(unsigned ID);
  Value *getFnValueByID(unsigned ID, Type *Ty) {
    if (Ty && Ty->isMetadataTy())
      return MetadataAsValue::get(Ty->getContext(), getFnMetadataByID(ID));
    return ValueList.getValueFwdRef(ID, Ty);
  }
  Metadata *getFnMetadataByID(unsigned ID) {
    return MDValueList.getValueFwdRef(ID);
  }
  BasicBlock *getBasicBlock(unsigned ID) const {
    if (ID >= FunctionBBs.size()) return nullptr; // Invalid ID
    return FunctionBBs[ID];
  }
  AttributeSet getAttributes(unsigned i) const {
    if (i-1 < MAttributes.size())
      return MAttributes[i-1];
    return AttributeSet();
  }

  /// Read a value/type pair out of the specified record from slot 'Slot'.
  /// Increment Slot past the number of slots used in the record. Return true on
  /// failure.
  bool getValueTypePair(SmallVectorImpl<uint64_t> &Record, unsigned &Slot,
                        unsigned InstNum, Value *&ResVal) {
    if (Slot == Record.size()) return true;
    unsigned ValNo = (unsigned)Record[Slot++];
    // Adjust the ValNo, if it was encoded relative to the InstNum.
    if (UseRelativeIDs)
      ValNo = InstNum - ValNo;
    if (ValNo < InstNum) {
      // If this is not a forward reference, just return the value we already
      // have.
      ResVal = getFnValueByID(ValNo, nullptr);
      return ResVal == nullptr;
    }
    if (Slot == Record.size())
      return true;

    unsigned TypeNo = (unsigned)Record[Slot++];
    ResVal = getFnValueByID(ValNo, getTypeByID(TypeNo));
    return ResVal == nullptr;
  }

  /// Read a value out of the specified record from slot 'Slot'. Increment Slot
  /// past the number of slots used by the value in the record. Return true if
  /// there is an error.
  bool popValue(SmallVectorImpl<uint64_t> &Record, unsigned &Slot,
                unsigned InstNum, Type *Ty, Value *&ResVal) {
    if (getValue(Record, Slot, InstNum, Ty, ResVal))
      return true;
    // All values currently take a single record slot.
    ++Slot;
    return false;
  }

  /// Like popValue, but does not increment the Slot number.
  bool getValue(SmallVectorImpl<uint64_t> &Record, unsigned Slot,
                unsigned InstNum, Type *Ty, Value *&ResVal) {
    ResVal = getValue(Record, Slot, InstNum, Ty);
    return ResVal == nullptr;
  }

  /// Version of getValue that returns ResVal directly, or 0 if there is an
  /// error.
  Value *getValue(SmallVectorImpl<uint64_t> &Record, unsigned Slot,
                  unsigned InstNum, Type *Ty) {
    if (Slot == Record.size()) return nullptr;
    unsigned ValNo = (unsigned)Record[Slot];
    // Adjust the ValNo, if it was encoded relative to the InstNum.
    if (UseRelativeIDs)
      ValNo = InstNum - ValNo;
    return getFnValueByID(ValNo, Ty);
  }

  /// Like getValue, but decodes signed VBRs.
  Value *getValueSigned(SmallVectorImpl<uint64_t> &Record, unsigned Slot,
                        unsigned InstNum, Type *Ty) {
    if (Slot == Record.size()) return nullptr;
    unsigned ValNo = (unsigned)decodeSignRotatedValue(Record[Slot]);
    // Adjust the ValNo, if it was encoded relative to the InstNum.
    if (UseRelativeIDs)
      ValNo = InstNum - ValNo;
    return getFnValueByID(ValNo, Ty);
  }

  /// Converts alignment exponent (i.e. power of two (or zero)) to the
  /// corresponding alignment to use. If alignment is too large, returns
  /// a corresponding error code.
  std::error_code parseAlignmentValue(uint64_t Exponent, unsigned &Alignment);
  std::error_code parseAttrKind(uint64_t Code, Attribute::AttrKind *Kind);
  std::error_code parseModule(bool Resume, bool ShouldLazyLoadMetadata = false);
  std::error_code parseAttributeBlock();
  std::error_code parseAttributeGroupBlock();
  std::error_code parseTypeTable();
  std::error_code parseTypeTableBody();

  std::error_code parseValueSymbolTable();
  std::error_code parseConstants();
  std::error_code rememberAndSkipFunctionBody();
  /// Save the positions of the Metadata blocks and skip parsing the blocks.
  std::error_code rememberAndSkipMetadata();
  std::error_code parseFunctionBody(Function *F);
  std::error_code globalCleanup();
  std::error_code resolveGlobalAndAliasInits();
  std::error_code parseMetadata();
  std::error_code parseSelectNamedMetadata(ArrayRef<StringRef> NamedMetadata); // HLSL Change
  std::error_code materializeSelectNamedMetadata(ArrayRef<StringRef> NamedMetadata) override; // HLSL Change
  std::error_code parseMetadataAttachment(Function &F);
  ErrorOr<std::string> parseModuleTriple();
  std::error_code parseUseLists();
  std::error_code initStream(std::unique_ptr<DataStreamer> Streamer);
  std::error_code initStreamFromBuffer();
  std::error_code initLazyStream(std::unique_ptr<DataStreamer> Streamer);
  std::error_code findFunctionInStream(
      Function *F,
      DenseMap<Function *, uint64_t>::iterator DeferredFunctionInfoIterator);
};
} // namespace

BitcodeDiagnosticInfo::BitcodeDiagnosticInfo(std::error_code EC,
                                             DiagnosticSeverity Severity,
                                             const Twine &Msg)
    : DiagnosticInfo(DK_Bitcode, Severity), Msg(Msg), EC(EC) {}

void BitcodeDiagnosticInfo::print(DiagnosticPrinter &DP) const { DP << Msg; }

static std::error_code error(DiagnosticHandlerFunction DiagnosticHandler,
                             std::error_code EC, const Twine &Message) {
  BitcodeDiagnosticInfo DI(EC, DS_Error, Message);
  DiagnosticHandler(DI);
  return EC;
}

static std::error_code error(DiagnosticHandlerFunction DiagnosticHandler,
                             std::error_code EC) {
  return error(DiagnosticHandler, EC, EC.message());
}

static std::error_code error(DiagnosticHandlerFunction DiagnosticHandler,
                             const Twine &Message) {
  return error(DiagnosticHandler,
               make_error_code(BitcodeError::CorruptedBitcode), Message);
}

std::error_code BitcodeReader::error(BitcodeError E, const Twine &Message) {
  return ::error(DiagnosticHandler, make_error_code(E), Message);
}

std::error_code BitcodeReader::error(const Twine &Message) {
  return ::error(DiagnosticHandler,
                 make_error_code(BitcodeError::CorruptedBitcode), Message);
}

std::error_code BitcodeReader::error(BitcodeError E) {
  return ::error(DiagnosticHandler, make_error_code(E));
}

static DiagnosticHandlerFunction getDiagHandler(DiagnosticHandlerFunction F,
                                                LLVMContext &C) {
  if (F)
    return F;
  return [&C](const DiagnosticInfo &DI) { C.diagnose(DI); };
}

// HLSL Change Starts
static void ReportWarning(DiagnosticHandlerFunction F, const char *Msg) {
  Twine tmsg(Msg);
  BitcodeDiagnosticInfo BDI(std::error_code(), DiagnosticSeverity::DS_Warning,
                            tmsg);
  F(BDI);
}
// HLSL Change Ends

BitcodeReader::BitcodeReader(std::unique_ptr<MemoryBuffer> &&Buffer, LLVMContext &Context, // HLSL Change: unique_ptr
                             DiagnosticHandlerFunction DiagnosticHandler)
    : Context(Context),
      DiagnosticHandler(getDiagHandler(DiagnosticHandler, Context)),
      Buffer(std::move(Buffer)), ValueList(Context), MDValueList(Context) {} // HLSL Change: std::move

BitcodeReader::BitcodeReader(LLVMContext &Context,
                             DiagnosticHandlerFunction DiagnosticHandler)
    : Context(Context),
      DiagnosticHandler(getDiagHandler(DiagnosticHandler, Context)),
      Buffer(nullptr), ValueList(Context), MDValueList(Context) {}

std::error_code BitcodeReader::materializeForwardReferencedFunctions() {
  if (WillMaterializeAllForwardRefs)
    return std::error_code();

  // Prevent recursion.
  WillMaterializeAllForwardRefs = true;

  while (!BasicBlockFwdRefQueue.empty()) {
    Function *F = BasicBlockFwdRefQueue.front();
    BasicBlockFwdRefQueue.pop_front();
    assert(F && "Expected valid function");
    if (!BasicBlockFwdRefs.count(F))
      // Already materialized.
      continue;

    // Check for a function that isn't materializable to prevent an infinite
    // loop.  When parsing a blockaddress stored in a global variable, there
    // isn't a trivial way to check if a function will have a body without a
    // linear search through FunctionsWithBodies, so just check it here.
    if (!F->isMaterializable())
      return error("Never resolved function from blockaddress");

    // Try to materialize F.
    if (std::error_code EC = materialize(F))
      return EC;
  }
  assert(BasicBlockFwdRefs.empty() && "Function missing from queue");

  // Reset state.
  WillMaterializeAllForwardRefs = false;
  return std::error_code();
}

void BitcodeReader::freeState() {
  Buffer = nullptr;
  std::vector<Type*>().swap(TypeList);
  ValueList.clear();
  MDValueList.clear();
  std::vector<Comdat *>().swap(ComdatList);

  std::vector<AttributeSet>().swap(MAttributes);
  std::vector<BasicBlock*>().swap(FunctionBBs);
  std::vector<Function*>().swap(FunctionsWithBodies);
  DeferredFunctionInfo.clear();
  DeferredMetadataInfo.clear();
  MDKindMap.clear();

  assert(BasicBlockFwdRefs.empty() && "Unresolved blockaddress fwd references");
  BasicBlockFwdRefQueue.clear();
}

//===----------------------------------------------------------------------===//
//  Helper functions to implement forward reference resolution, etc.
//===----------------------------------------------------------------------===//

/// Convert a string from a record into an std::string, return true on failure.
template <typename StrTy>
static bool convertToString(ArrayRef<uint64_t> Record, unsigned Idx,
                            StrTy &Result) {
  if (Idx > Record.size())
    return true;

  for (unsigned i = Idx, e = Record.size(); i != e; ++i)
    Result += (char)Record[i];
  return false;
}

static bool hasImplicitComdat(size_t Val) {
  switch (Val) {
  default:
    return false;
  case 1:  // Old WeakAnyLinkage
  case 4:  // Old LinkOnceAnyLinkage
  case 10: // Old WeakODRLinkage
  case 11: // Old LinkOnceODRLinkage
    return true;
  }
}

static GlobalValue::LinkageTypes getDecodedLinkage(unsigned Val) {
  switch (Val) {
  default: // Map unknown/new linkages to external
  case 0:
    return GlobalValue::ExternalLinkage;
  case 2:
    return GlobalValue::AppendingLinkage;
  case 3:
    return GlobalValue::InternalLinkage;
  case 5:
    return GlobalValue::ExternalLinkage; // Obsolete DLLImportLinkage
  case 6:
    return GlobalValue::ExternalLinkage; // Obsolete DLLExportLinkage
  case 7:
    return GlobalValue::ExternalWeakLinkage;
  case 8:
    return GlobalValue::CommonLinkage;
  case 9:
    return GlobalValue::PrivateLinkage;
  case 12:
    return GlobalValue::AvailableExternallyLinkage;
  case 13:
    return GlobalValue::PrivateLinkage; // Obsolete LinkerPrivateLinkage
  case 14:
    return GlobalValue::PrivateLinkage; // Obsolete LinkerPrivateWeakLinkage
  case 15:
    return GlobalValue::ExternalLinkage; // Obsolete LinkOnceODRAutoHideLinkage
  case 1: // Old value with implicit comdat.
  case 16:
    return GlobalValue::WeakAnyLinkage;
  case 10: // Old value with implicit comdat.
  case 17:
    return GlobalValue::WeakODRLinkage;
  case 4: // Old value with implicit comdat.
  case 18:
    return GlobalValue::LinkOnceAnyLinkage;
  case 11: // Old value with implicit comdat.
  case 19:
    return GlobalValue::LinkOnceODRLinkage;
  }
}

static GlobalValue::VisibilityTypes getDecodedVisibility(unsigned Val) {
  switch (Val) {
  default: // Map unknown visibilities to default.
  case 0: return GlobalValue::DefaultVisibility;
  case 1: return GlobalValue::HiddenVisibility;
  case 2: return GlobalValue::ProtectedVisibility;
  }
}

static GlobalValue::DLLStorageClassTypes
getDecodedDLLStorageClass(unsigned Val) {
  switch (Val) {
  default: // Map unknown values to default.
  case 0: return GlobalValue::DefaultStorageClass;
  case 1: return GlobalValue::DLLImportStorageClass;
  case 2: return GlobalValue::DLLExportStorageClass;
  }
}

static GlobalVariable::ThreadLocalMode getDecodedThreadLocalMode(unsigned Val) {
  switch (Val) {
    case 0: return GlobalVariable::NotThreadLocal;
    default: // Map unknown non-zero value to general dynamic.
    case 1: return GlobalVariable::GeneralDynamicTLSModel;
    case 2: return GlobalVariable::LocalDynamicTLSModel;
    case 3: return GlobalVariable::InitialExecTLSModel;
    case 4: return GlobalVariable::LocalExecTLSModel;
  }
}

static int getDecodedCastOpcode(unsigned Val) {
  switch (Val) {
  default: return -1;
  case bitc::CAST_TRUNC   : return Instruction::Trunc;
  case bitc::CAST_ZEXT    : return Instruction::ZExt;
  case bitc::CAST_SEXT    : return Instruction::SExt;
  case bitc::CAST_FPTOUI  : return Instruction::FPToUI;
  case bitc::CAST_FPTOSI  : return Instruction::FPToSI;
  case bitc::CAST_UITOFP  : return Instruction::UIToFP;
  case bitc::CAST_SITOFP  : return Instruction::SIToFP;
  case bitc::CAST_FPTRUNC : return Instruction::FPTrunc;
  case bitc::CAST_FPEXT   : return Instruction::FPExt;
  case bitc::CAST_PTRTOINT: return Instruction::PtrToInt;
  case bitc::CAST_INTTOPTR: return Instruction::IntToPtr;
  case bitc::CAST_BITCAST : return Instruction::BitCast;
  case bitc::CAST_ADDRSPACECAST: return Instruction::AddrSpaceCast;
  }
}

static int getDecodedBinaryOpcode(unsigned Val, Type *Ty) {
  bool IsFP = Ty->isFPOrFPVectorTy();
  // BinOps are only valid for int/fp or vector of int/fp types
  if (!IsFP && !Ty->isIntOrIntVectorTy())
    return -1;

  switch (Val) {
  default:
    return -1;
  case bitc::BINOP_ADD:
    return IsFP ? Instruction::FAdd : Instruction::Add;
  case bitc::BINOP_SUB:
    return IsFP ? Instruction::FSub : Instruction::Sub;
  case bitc::BINOP_MUL:
    return IsFP ? Instruction::FMul : Instruction::Mul;
  case bitc::BINOP_UDIV:
    return IsFP ? -1 : Instruction::UDiv;
  case bitc::BINOP_SDIV:
    return IsFP ? Instruction::FDiv : Instruction::SDiv;
  case bitc::BINOP_UREM:
    return IsFP ? -1 : Instruction::URem;
  case bitc::BINOP_SREM:
    return IsFP ? Instruction::FRem : Instruction::SRem;
  case bitc::BINOP_SHL:
    return IsFP ? -1 : Instruction::Shl;
  case bitc::BINOP_LSHR:
    return IsFP ? -1 : Instruction::LShr;
  case bitc::BINOP_ASHR:
    return IsFP ? -1 : Instruction::AShr;
  case bitc::BINOP_AND:
    return IsFP ? -1 : Instruction::And;
  case bitc::BINOP_OR:
    return IsFP ? -1 : Instruction::Or;
  case bitc::BINOP_XOR:
    return IsFP ? -1 : Instruction::Xor;
  }
}

static AtomicRMWInst::BinOp getDecodedRMWOperation(unsigned Val) {
  switch (Val) {
  default: return AtomicRMWInst::BAD_BINOP;
  case bitc::RMW_XCHG: return AtomicRMWInst::Xchg;
  case bitc::RMW_ADD: return AtomicRMWInst::Add;
  case bitc::RMW_SUB: return AtomicRMWInst::Sub;
  case bitc::RMW_AND: return AtomicRMWInst::And;
  case bitc::RMW_NAND: return AtomicRMWInst::Nand;
  case bitc::RMW_OR: return AtomicRMWInst::Or;
  case bitc::RMW_XOR: return AtomicRMWInst::Xor;
  case bitc::RMW_MAX: return AtomicRMWInst::Max;
  case bitc::RMW_MIN: return AtomicRMWInst::Min;
  case bitc::RMW_UMAX: return AtomicRMWInst::UMax;
  case bitc::RMW_UMIN: return AtomicRMWInst::UMin;
  }
}

static AtomicOrdering getDecodedOrdering(unsigned Val) {
  switch (Val) {
  case bitc::ORDERING_NOTATOMIC: return NotAtomic;
  case bitc::ORDERING_UNORDERED: return Unordered;
  case bitc::ORDERING_MONOTONIC: return Monotonic;
  case bitc::ORDERING_ACQUIRE: return Acquire;
  case bitc::ORDERING_RELEASE: return Release;
  case bitc::ORDERING_ACQREL: return AcquireRelease;
  default: // Map unknown orderings to sequentially-consistent.
  case bitc::ORDERING_SEQCST: return SequentiallyConsistent;
  }
}

static SynchronizationScope getDecodedSynchScope(unsigned Val) {
  switch (Val) {
  case bitc::SYNCHSCOPE_SINGLETHREAD: return SingleThread;
  default: // Map unknown scopes to cross-thread.
  case bitc::SYNCHSCOPE_CROSSTHREAD: return CrossThread;
  }
}

static Comdat::SelectionKind getDecodedComdatSelectionKind(unsigned Val) {
  switch (Val) {
  default: // Map unknown selection kinds to any.
  case bitc::COMDAT_SELECTION_KIND_ANY:
    return Comdat::Any;
  case bitc::COMDAT_SELECTION_KIND_EXACT_MATCH:
    return Comdat::ExactMatch;
  case bitc::COMDAT_SELECTION_KIND_LARGEST:
    return Comdat::Largest;
  case bitc::COMDAT_SELECTION_KIND_NO_DUPLICATES:
    return Comdat::NoDuplicates;
  case bitc::COMDAT_SELECTION_KIND_SAME_SIZE:
    return Comdat::SameSize;
  }
}

static FastMathFlags getDecodedFastMathFlags(unsigned Val) {
  FastMathFlags FMF;
  if (0 != (Val & FastMathFlags::UnsafeAlgebra))
    FMF.setUnsafeAlgebra();
  if (0 != (Val & FastMathFlags::NoNaNs))
    FMF.setNoNaNs();
  if (0 != (Val & FastMathFlags::NoInfs))
    FMF.setNoInfs();
  if (0 != (Val & FastMathFlags::NoSignedZeros))
    FMF.setNoSignedZeros();
  if (0 != (Val & FastMathFlags::AllowReciprocal))
    FMF.setAllowReciprocal();
  return FMF;
}

static void upgradeDLLImportExportLinkage(llvm::GlobalValue *GV, unsigned Val) {
  switch (Val) {
  case 5: GV->setDLLStorageClass(GlobalValue::DLLImportStorageClass); break;
  case 6: GV->setDLLStorageClass(GlobalValue::DLLExportStorageClass); break;
  }
}

namespace llvm {
namespace {
/// \brief A class for maintaining the slot number definition
/// as a placeholder for the actual definition for forward constants defs.
class ConstantPlaceHolder : public ConstantExpr {
  void operator=(const ConstantPlaceHolder &) = delete;

public:
  // allocate space for exactly one operand
  void *operator new(size_t s) { return User::operator new(s, 1); }
  explicit ConstantPlaceHolder(Type *Ty, LLVMContext &Context)
      : ConstantExpr(Ty, Instruction::UserOp1, &Op<0>(), 1) {
    Op<0>() = UndefValue::get(Type::getInt32Ty(Context));
  }

  /// \brief Methods to support type inquiry through isa, cast, and dyn_cast.
  static bool classof(const Value *V) {
    return isa<ConstantExpr>(V) &&
           cast<ConstantExpr>(V)->getOpcode() == Instruction::UserOp1;
  }

  /// Provide fast operand accessors
  DECLARE_TRANSPARENT_OPERAND_ACCESSORS(Value);
};
}

// FIXME: can we inherit this from ConstantExpr?
template <>
struct OperandTraits<ConstantPlaceHolder> :
  public FixedNumOperandTraits<ConstantPlaceHolder, 1> {
};
DEFINE_TRANSPARENT_OPERAND_ACCESSORS(ConstantPlaceHolder, Value)
}

void BitcodeReaderValueList::assignValue(Value *V, unsigned Idx) {
  if (Idx == size()) {
    push_back(V);
    return;
  }

  if (Idx >= size())
    resize(Idx+1);

  WeakVH &OldV = ValuePtrs[Idx];
  if (!OldV) {
    OldV = V;
    return;
  }

  // Handle constants and non-constants (e.g. instrs) differently for
  // efficiency.
  if (Constant *PHC = dyn_cast<Constant>(&*OldV)) {
    ResolveConstants.push_back(std::make_pair(PHC, Idx));
    OldV = V;
  } else {
    // If there was a forward reference to this value, replace it.
    Value *PrevVal = OldV;
    OldV->replaceAllUsesWith(V);
    delete PrevVal;
  }
}


Constant *BitcodeReaderValueList::getConstantFwdRef(unsigned Idx,
                                                    Type *Ty) {
  if (Idx >= size())
    resize(Idx + 1);

  if (Value *V = ValuePtrs[Idx]) {
    if (Ty != V->getType())
      report_fatal_error("Type mismatch in constant table!");
    return cast<Constant>(V);
  }

  // Create and return a placeholder, which will later be RAUW'd.
  Constant *C = new ConstantPlaceHolder(Ty, Context);
  ValuePtrs[Idx] = C;
  return C;
}

Value *BitcodeReaderValueList::getValueFwdRef(unsigned Idx, Type *Ty) {
  // Bail out for a clearly invalid value. This would make us call resize(0)
  if (Idx == UINT_MAX)
    return nullptr;

  if (Idx >= size())
    resize(Idx + 1);

  if (Value *V = ValuePtrs[Idx]) {
    // If the types don't match, it's invalid.
    if (Ty && Ty != V->getType())
      return nullptr;
    return V;
  }

  // No type specified, must be invalid reference.
  if (!Ty) return nullptr;

  // Create and return a placeholder, which will later be RAUW'd.
  Value *V = new Argument(Ty);
  ValuePtrs[Idx] = V;
  return V;
}

/// Once all constants are read, this method bulk resolves any forward
/// references.  The idea behind this is that we sometimes get constants (such
/// as large arrays) which reference *many* forward ref constants.  Replacing
/// each of these causes a lot of thrashing when building/reuniquing the
/// constant.  Instead of doing this, we look at all the uses and rewrite all
/// the place holders at once for any constant that uses a placeholder.
void BitcodeReaderValueList::resolveConstantForwardRefs() {
  // Sort the values by-pointer so that they are efficient to look up with a
  // binary search.
  std::sort(ResolveConstants.begin(), ResolveConstants.end());

  SmallVector<Constant*, 64> NewOps;

  while (!ResolveConstants.empty()) {
    Value *RealVal = operator[](ResolveConstants.back().second);
    Constant *Placeholder = ResolveConstants.back().first;
    ResolveConstants.pop_back();

    // Loop over all users of the placeholder, updating them to reference the
    // new value.  If they reference more than one placeholder, update them all
    // at once.
    while (!Placeholder->use_empty()) {
      auto UI = Placeholder->user_begin();
      User *U = *UI;

      // If the using object isn't uniqued, just update the operands.  This
      // handles instructions and initializers for global variables.
      if (!isa<Constant>(U) || isa<GlobalValue>(U)) {
        UI.getUse().set(RealVal);
        continue;
      }

      // Otherwise, we have a constant that uses the placeholder.  Replace that
      // constant with a new constant that has *all* placeholder uses updated.
      Constant *UserC = cast<Constant>(U);
      for (User::op_iterator I = UserC->op_begin(), E = UserC->op_end();
           I != E; ++I) {
        Value *NewOp;
        if (!isa<ConstantPlaceHolder>(*I)) {
          // Not a placeholder reference.
          NewOp = *I;
        } else if (*I == Placeholder) {
          // Common case is that it just references this one placeholder.
          NewOp = RealVal;
        } else {
          // Otherwise, look up the placeholder in ResolveConstants.
          ResolveConstantsTy::iterator It =
            std::lower_bound(ResolveConstants.begin(), ResolveConstants.end(),
                             std::pair<Constant*, unsigned>(cast<Constant>(*I),
                                                            0));
          assert(It != ResolveConstants.end() && It->first == *I);
          NewOp = operator[](It->second);
        }

        NewOps.push_back(cast<Constant>(NewOp));
      }

      // Make the new constant.
      Constant *NewC;
      if (ConstantArray *UserCA = dyn_cast<ConstantArray>(UserC)) {
        NewC = ConstantArray::get(UserCA->getType(), NewOps);
      } else if (ConstantStruct *UserCS = dyn_cast<ConstantStruct>(UserC)) {
        NewC = ConstantStruct::get(UserCS->getType(), NewOps);
      } else if (isa<ConstantVector>(UserC)) {
        NewC = ConstantVector::get(NewOps);
      } else {
        assert(isa<ConstantExpr>(UserC) && "Must be a ConstantExpr.");
        NewC = cast<ConstantExpr>(UserC)->getWithOperands(NewOps);
      }

      UserC->replaceAllUsesWith(NewC);
      UserC->destroyConstant();
      NewOps.clear();
    }

    // Update all ValueHandles, they should be the only users at this point.
    Placeholder->replaceAllUsesWith(RealVal);
    delete Placeholder;
  }
}

void BitcodeReaderMDValueList::assignValue(Metadata *MD, unsigned Idx) {
  if (Idx == size()) {
    push_back(MD);
    return;
  }

  if (Idx >= size())
    resize(Idx+1);

  TrackingMDRef &OldMD = MDValuePtrs[Idx];
  if (!OldMD) {
    OldMD.reset(MD);
    return;
  }

  // If there was a forward reference to this value, replace it.
  TempMDTuple PrevMD(cast<MDTuple>(OldMD.get()));
  PrevMD->replaceAllUsesWith(MD);
  --NumFwdRefs;
}

Metadata *BitcodeReaderMDValueList::getValueFwdRef(unsigned Idx) {
  if (Idx >= size())
    resize(Idx + 1);

  if (Metadata *MD = MDValuePtrs[Idx])
    return MD;

  // Track forward refs to be resolved later.
  if (AnyFwdRefs) {
    MinFwdRef = std::min(MinFwdRef, Idx);
    MaxFwdRef = std::max(MaxFwdRef, Idx);
  } else {
    AnyFwdRefs = true;
    MinFwdRef = MaxFwdRef = Idx;
  }
  ++NumFwdRefs;

  // Create and return a placeholder, which will later be RAUW'd.
  Metadata *MD = MDNode::getTemporary(Context, None).release();
  MDValuePtrs[Idx].reset(MD);
  return MD;
}

void BitcodeReaderMDValueList::tryToResolveCycles() {
  if (!AnyFwdRefs)
    // Nothing to do.
    return;

  if (NumFwdRefs)
    // Still forward references... can't resolve cycles.
    return;

  // Resolve any cycles.
  for (unsigned I = MinFwdRef, E = MaxFwdRef + 1; I != E; ++I) {
    auto &MD = MDValuePtrs[I];
    auto *N = dyn_cast_or_null<MDNode>(MD);
    if (!N)
      continue;

    assert(!N->isTemporary() && "Unexpected forward reference");
    N->resolveCycles();
  }

  // Make sure we return early again until there's another forward ref.
  AnyFwdRefs = false;
}

Type *BitcodeReader::getTypeByID(unsigned ID) {
  // The type table size is always specified correctly.
  if (ID >= TypeList.size())
    return nullptr;

  if (Type *Ty = TypeList[ID])
    return Ty;

  // If we have a forward reference, the only possible case is when it is to a
  // named struct.  Just create a placeholder for now.
  return TypeList[ID] = createIdentifiedStructType(Context);
}

StructType *BitcodeReader::createIdentifiedStructType(LLVMContext &Context,
                                                      StringRef Name) {
  auto *Ret = StructType::create(Context, Name);
  IdentifiedStructTypes.push_back(Ret);
  return Ret;
}

StructType *BitcodeReader::createIdentifiedStructType(LLVMContext &Context) {
  auto *Ret = StructType::create(Context);
  IdentifiedStructTypes.push_back(Ret);
  return Ret;
}


//===----------------------------------------------------------------------===//
//  Functions for parsing blocks from the bitcode file
//===----------------------------------------------------------------------===//


/// \brief This fills an AttrBuilder object with the LLVM attributes that have
/// been decoded from the given integer. This function must stay in sync with
/// 'encodeLLVMAttributesForBitcode'.
static void decodeLLVMAttributesForBitcode(AttrBuilder &B,
                                           uint64_t EncodedAttrs) {
  // FIXME: Remove in 4.0.

  // The alignment is stored as a 16-bit raw value from bits 31--16.  We shift
  // the bits above 31 down by 11 bits.
  unsigned Alignment = (EncodedAttrs & (0xffffULL << 16)) >> 16;
  assert((!Alignment || isPowerOf2_32(Alignment)) &&
         "Alignment must be a power of two.");

  if (Alignment)
    B.addAlignmentAttr(Alignment);
  B.addRawValue(((EncodedAttrs & (0xfffffULL << 32)) >> 11) |
                (EncodedAttrs & 0xffff));
}

std::error_code BitcodeReader::parseAttributeBlock() {
  if (Stream.EnterSubBlock(bitc::PARAMATTR_BLOCK_ID))
    return error("Invalid record");

  if (!MAttributes.empty())
    return error("Invalid multiple blocks");

  SmallVector<uint64_t, 64> Record;

  SmallVector<AttributeSet, 8> Attrs;

  // Read all the records.
  while (1) {
    // HLSL Change Starts - count skipped blocks
    unsigned skipCount = 0;
    BitstreamEntry Entry = Stream.advanceSkippingSubblocks(0, &skipCount);
    if (skipCount) ReportWarning(DiagnosticHandler, "Unrecognized subblock");
    // HLSL Change End

    switch (Entry.Kind) {
    case BitstreamEntry::SubBlock: // Handled for us already.
    case BitstreamEntry::Error:
      return error("Malformed block");
    case BitstreamEntry::EndBlock:
      return std::error_code();
    case BitstreamEntry::Record:
      // The interesting case.
      break;
    }

    // Read a record.
    Record.clear();
    switch (Stream.readRecord(Entry.ID, Record)) {
    default:  // Default behavior: ignore.
      break;
    case bitc::PARAMATTR_CODE_ENTRY_OLD: { // ENTRY: [paramidx0, attr0, ...]
      // FIXME: Remove in 4.0.
      if (Record.size() & 1)
        return error("Invalid record");

      for (unsigned i = 0, e = Record.size(); i != e; i += 2) {
        AttrBuilder B;
        decodeLLVMAttributesForBitcode(B, Record[i+1]);
        Attrs.push_back(AttributeSet::get(Context, Record[i], B));
      }

      MAttributes.push_back(AttributeSet::get(Context, Attrs));
      Attrs.clear();
      break;
    }
    case bitc::PARAMATTR_CODE_ENTRY: { // ENTRY: [attrgrp0, attrgrp1, ...]
      for (unsigned i = 0, e = Record.size(); i != e; ++i)
        Attrs.push_back(MAttributeGroups[Record[i]]);

      MAttributes.push_back(AttributeSet::get(Context, Attrs));
      Attrs.clear();
      break;
    }
    }
  }
}

// Returns Attribute::None on unrecognized codes.
static Attribute::AttrKind getAttrFromCode(uint64_t Code) {
  switch (Code) {
  default:
    return Attribute::None;
  case bitc::ATTR_KIND_ALIGNMENT:
    return Attribute::Alignment;
  case bitc::ATTR_KIND_ALWAYS_INLINE:
    return Attribute::AlwaysInline;
  case bitc::ATTR_KIND_ARGMEMONLY:
    return Attribute::ArgMemOnly;
  case bitc::ATTR_KIND_BUILTIN:
    return Attribute::Builtin;
  case bitc::ATTR_KIND_BY_VAL:
    return Attribute::ByVal;
  case bitc::ATTR_KIND_IN_ALLOCA:
    return Attribute::InAlloca;
  case bitc::ATTR_KIND_COLD:
    return Attribute::Cold;
  case bitc::ATTR_KIND_CONVERGENT:
    return Attribute::Convergent;
  case bitc::ATTR_KIND_INLINE_HINT:
    return Attribute::InlineHint;
  case bitc::ATTR_KIND_IN_REG:
    return Attribute::InReg;
  case bitc::ATTR_KIND_JUMP_TABLE:
    return Attribute::JumpTable;
  case bitc::ATTR_KIND_MIN_SIZE:
    return Attribute::MinSize;
  case bitc::ATTR_KIND_NAKED:
    return Attribute::Naked;
  case bitc::ATTR_KIND_NEST:
    return Attribute::Nest;
  case bitc::ATTR_KIND_NO_ALIAS:
    return Attribute::NoAlias;
  case bitc::ATTR_KIND_NO_BUILTIN:
    return Attribute::NoBuiltin;
  case bitc::ATTR_KIND_NO_CAPTURE:
    return Attribute::NoCapture;
  case bitc::ATTR_KIND_NO_DUPLICATE:
    return Attribute::NoDuplicate;
  case bitc::ATTR_KIND_NO_IMPLICIT_FLOAT:
    return Attribute::NoImplicitFloat;
  case bitc::ATTR_KIND_NO_INLINE:
    return Attribute::NoInline;
  case bitc::ATTR_KIND_NON_LAZY_BIND:
    return Attribute::NonLazyBind;
  case bitc::ATTR_KIND_NON_NULL:
    return Attribute::NonNull;
  case bitc::ATTR_KIND_DEREFERENCEABLE:
    return Attribute::Dereferenceable;
  case bitc::ATTR_KIND_DEREFERENCEABLE_OR_NULL:
    return Attribute::DereferenceableOrNull;
  case bitc::ATTR_KIND_NO_RED_ZONE:
    return Attribute::NoRedZone;
  case bitc::ATTR_KIND_NO_RETURN:
    return Attribute::NoReturn;
  case bitc::ATTR_KIND_NO_UNWIND:
    return Attribute::NoUnwind;
  case bitc::ATTR_KIND_OPTIMIZE_FOR_SIZE:
    return Attribute::OptimizeForSize;
  case bitc::ATTR_KIND_OPTIMIZE_NONE:
    return Attribute::OptimizeNone;
  case bitc::ATTR_KIND_READ_NONE:
    return Attribute::ReadNone;
  case bitc::ATTR_KIND_READ_ONLY:
    return Attribute::ReadOnly;
  case bitc::ATTR_KIND_RETURNED:
    return Attribute::Returned;
  case bitc::ATTR_KIND_RETURNS_TWICE:
    return Attribute::ReturnsTwice;
  case bitc::ATTR_KIND_S_EXT:
    return Attribute::SExt;
  case bitc::ATTR_KIND_STACK_ALIGNMENT:
    return Attribute::StackAlignment;
  case bitc::ATTR_KIND_STACK_PROTECT:
    return Attribute::StackProtect;
  case bitc::ATTR_KIND_STACK_PROTECT_REQ:
    return Attribute::StackProtectReq;
  case bitc::ATTR_KIND_STACK_PROTECT_STRONG:
    return Attribute::StackProtectStrong;
  case bitc::ATTR_KIND_SAFESTACK:
    return Attribute::SafeStack;
  case bitc::ATTR_KIND_STRUCT_RET:
    return Attribute::StructRet;
  case bitc::ATTR_KIND_SANITIZE_ADDRESS:
    return Attribute::SanitizeAddress;
  case bitc::ATTR_KIND_SANITIZE_THREAD:
    return Attribute::SanitizeThread;
  case bitc::ATTR_KIND_SANITIZE_MEMORY:
    return Attribute::SanitizeMemory;
  case bitc::ATTR_KIND_UW_TABLE:
    return Attribute::UWTable;
  case bitc::ATTR_KIND_Z_EXT:
    return Attribute::ZExt;
  }
}

std::error_code BitcodeReader::parseAlignmentValue(uint64_t Exponent,
                                                   unsigned &Alignment) {
  // Note: Alignment in bitcode files is incremented by 1, so that zero
  // can be used for default alignment.
  if (Exponent > Value::MaxAlignmentExponent + 1)
    return error("Invalid alignment value");
  Alignment = (1 << static_cast<unsigned>(Exponent)) >> 1;
  return std::error_code();
}

std::error_code BitcodeReader::parseAttrKind(uint64_t Code,
                                             Attribute::AttrKind *Kind) {
  *Kind = getAttrFromCode(Code);
  if (*Kind == Attribute::None)
    return error(BitcodeError::CorruptedBitcode,
                 "Unknown attribute kind (" + Twine(Code) + ")");
  return std::error_code();
}

std::error_code BitcodeReader::parseAttributeGroupBlock() {
  if (Stream.EnterSubBlock(bitc::PARAMATTR_GROUP_BLOCK_ID))
    return error("Invalid record");

  if (!MAttributeGroups.empty())
    return error("Invalid multiple blocks");

  SmallVector<uint64_t, 64> Record;

  // Read all the records.
  while (1) {
    // HLSL Change Starts - count skipped blocks
    unsigned skipCount = 0;
    BitstreamEntry Entry = Stream.advanceSkippingSubblocks(0, &skipCount);
    if (skipCount) ReportWarning(DiagnosticHandler, "Unrecognized subblock");
    // HLSL Change End

    switch (Entry.Kind) {
    case BitstreamEntry::SubBlock: // Handled for us already.
    case BitstreamEntry::Error:
      return error("Malformed block");
    case BitstreamEntry::EndBlock:
      return std::error_code();
    case BitstreamEntry::Record:
      // The interesting case.
      break;
    }

    // Read a record.
    Record.clear();
    switch (Stream.readRecord(Entry.ID, Record)) {
    default:  // Default behavior: ignore.
      break;
    case bitc::PARAMATTR_GRP_CODE_ENTRY: { // ENTRY: [grpid, idx, a0, a1, ...]
      if (Record.size() < 3)
        return error("Invalid record");

      uint64_t GrpID = Record[0];
      uint64_t Idx = Record[1]; // Index of the object this attribute refers to.

      AttrBuilder B;
      for (unsigned i = 2, e = Record.size(); i != e; ++i) {
        if (Record[i] == 0) {        // Enum attribute
          Attribute::AttrKind Kind;
          if (std::error_code EC = parseAttrKind(Record[++i], &Kind))
            return EC;

          B.addAttribute(Kind);
        } else if (Record[i] == 1) { // Integer attribute
          Attribute::AttrKind Kind;
          if (std::error_code EC = parseAttrKind(Record[++i], &Kind))
            return EC;
          if (Kind == Attribute::Alignment)
            B.addAlignmentAttr(Record[++i]);
          else if (Kind == Attribute::StackAlignment)
            B.addStackAlignmentAttr(Record[++i]);
          else if (Kind == Attribute::Dereferenceable)
            B.addDereferenceableAttr(Record[++i]);
          else if (Kind == Attribute::DereferenceableOrNull)
            B.addDereferenceableOrNullAttr(Record[++i]);
        } else {                     // String attribute
          assert((Record[i] == 3 || Record[i] == 4) &&
                 "Invalid attribute group entry");
          bool HasValue = (Record[i++] == 4);
          SmallString<64> KindStr;
          SmallString<64> ValStr;

          while (Record[i] != 0 && i != e)
            KindStr += Record[i++];
          assert(Record[i] == 0 && "Kind string not null terminated");

          if (HasValue) {
            // Has a value associated with it.
            ++i; // Skip the '0' that terminates the "kind" string.
            while (Record[i] != 0 && i != e)
              ValStr += Record[i++];
            assert(Record[i] == 0 && "Value string not null terminated");
          }

          B.addAttribute(KindStr.str(), ValStr.str());
        }
      }

      MAttributeGroups[GrpID] = AttributeSet::get(Context, Idx, B);
      break;
    }
    }
  }
}

std::error_code BitcodeReader::parseTypeTable() {
  if (Stream.EnterSubBlock(bitc::TYPE_BLOCK_ID_NEW))
    return error("Invalid record");

  return parseTypeTableBody();
}

std::error_code BitcodeReader::parseTypeTableBody() {
  if (!TypeList.empty())
    return error("Invalid multiple blocks");

  SmallVector<uint64_t, 64> Record;
  unsigned NumRecords = 0;

  SmallString<64> TypeName;

  // Read all the records for this type table.
  while (1) {
    // HLSL Change Starts - count skipped blocks
    unsigned skipCount = 0;
    BitstreamEntry Entry = Stream.advanceSkippingSubblocks(0, &skipCount);
    if (skipCount) ReportWarning(DiagnosticHandler, "Unrecognized subblock");
    // HLSL Change End

    switch (Entry.Kind) {
    case BitstreamEntry::SubBlock: // Handled for us already.
    case BitstreamEntry::Error:
      return error("Malformed block");
    case BitstreamEntry::EndBlock:
      if (NumRecords != TypeList.size())
        return error("Malformed block");
      return std::error_code();
    case BitstreamEntry::Record:
      // The interesting case.
      break;
    }

    // Read a record.
    Record.clear();
    Type *ResultTy = nullptr;
    switch (Stream.readRecord(Entry.ID, Record)) {
    default:
      return error("Invalid value");
    case bitc::TYPE_CODE_NUMENTRY: // TYPE_CODE_NUMENTRY: [numentries]
      // TYPE_CODE_NUMENTRY contains a count of the number of types in the
      // type list.  This allows us to reserve space.
      if (Record.size() < 1)
        return error("Invalid record");
      TypeList.resize(Record[0]);
      continue;
    case bitc::TYPE_CODE_VOID:      // VOID
      ResultTy = Type::getVoidTy(Context);
      break;
    case bitc::TYPE_CODE_HALF:     // HALF
      ResultTy = Type::getHalfTy(Context);
      break;
    case bitc::TYPE_CODE_FLOAT:     // FLOAT
      ResultTy = Type::getFloatTy(Context);
      break;
    case bitc::TYPE_CODE_DOUBLE:    // DOUBLE
      ResultTy = Type::getDoubleTy(Context);
      break;
    case bitc::TYPE_CODE_X86_FP80:  // X86_FP80
      ResultTy = Type::getX86_FP80Ty(Context);
      break;
    case bitc::TYPE_CODE_FP128:     // FP128
      ResultTy = Type::getFP128Ty(Context);
      break;
    case bitc::TYPE_CODE_PPC_FP128: // PPC_FP128
      ResultTy = Type::getPPC_FP128Ty(Context);
      break;
    case bitc::TYPE_CODE_LABEL:     // LABEL
      ResultTy = Type::getLabelTy(Context);
      break;
    case bitc::TYPE_CODE_METADATA:  // METADATA
      ResultTy = Type::getMetadataTy(Context);
      break;
    case bitc::TYPE_CODE_X86_MMX:   // X86_MMX
      ResultTy = Type::getX86_MMXTy(Context);
      break;
    case bitc::TYPE_CODE_INTEGER: { // INTEGER: [width]
      if (Record.size() < 1)
        return error("Invalid record");

      uint64_t NumBits = Record[0];
      if (NumBits < IntegerType::MIN_INT_BITS ||
          NumBits > IntegerType::MAX_INT_BITS)
        return error("Bitwidth for integer type out of range");
      ResultTy = IntegerType::get(Context, NumBits);
      break;
    }
    case bitc::TYPE_CODE_POINTER: { // POINTER: [pointee type] or
                                    //          [pointee type, address space]
      if (Record.size() < 1)
        return error("Invalid record");
      unsigned AddressSpace = 0;
      if (Record.size() == 2)
        AddressSpace = Record[1];
      ResultTy = getTypeByID(Record[0]);
      if (!ResultTy ||
          !PointerType::isValidElementType(ResultTy))
        return error("Invalid type");
      ResultTy = PointerType::get(ResultTy, AddressSpace);
      break;
    }
    case bitc::TYPE_CODE_FUNCTION_OLD: {
      // FIXME: attrid is dead, remove it in LLVM 4.0
      // FUNCTION: [vararg, attrid, retty, paramty x N]
      if (Record.size() < 3)
        return error("Invalid record");
      SmallVector<Type*, 8> ArgTys;
      for (unsigned i = 3, e = Record.size(); i != e; ++i) {
        if (Type *T = getTypeByID(Record[i]))
          ArgTys.push_back(T);
        else
          break;
      }

      ResultTy = getTypeByID(Record[2]);
      if (!ResultTy || ArgTys.size() < Record.size()-3)
        return error("Invalid type");

      ResultTy = FunctionType::get(ResultTy, ArgTys, Record[0]);
      break;
    }
    case bitc::TYPE_CODE_FUNCTION: {
      // FUNCTION: [vararg, retty, paramty x N]
      if (Record.size() < 2)
        return error("Invalid record");
      SmallVector<Type*, 8> ArgTys;
      for (unsigned i = 2, e = Record.size(); i != e; ++i) {
        if (Type *T = getTypeByID(Record[i])) {
          if (!FunctionType::isValidArgumentType(T))
            return error("Invalid function argument type");
          ArgTys.push_back(T);
        }
        else
          break;
      }

      ResultTy = getTypeByID(Record[1]);
      if (!ResultTy || ArgTys.size() < Record.size()-2)
        return error("Invalid type");

      ResultTy = FunctionType::get(ResultTy, ArgTys, Record[0]);
      break;
    }
    case bitc::TYPE_CODE_STRUCT_ANON: {  // STRUCT: [ispacked, eltty x N]
      if (Record.size() < 1)
        return error("Invalid record");
      SmallVector<Type*, 8> EltTys;
      for (unsigned i = 1, e = Record.size(); i != e; ++i) {
        if (Type *T = getTypeByID(Record[i]))
          EltTys.push_back(T);
        else
          break;
      }
      if (EltTys.size() != Record.size()-1)
        return error("Invalid type");
      ResultTy = StructType::get(Context, EltTys, Record[0]);
      break;
    }
    case bitc::TYPE_CODE_STRUCT_NAME:   // STRUCT_NAME: [strchr x N]
      if (convertToString(Record, 0, TypeName))
        return error("Invalid record");
      continue;

    case bitc::TYPE_CODE_STRUCT_NAMED: { // STRUCT: [ispacked, eltty x N]
      if (Record.size() < 1)
        return error("Invalid record");

      if (NumRecords >= TypeList.size())
        return error("Invalid TYPE table");

      // Check to see if this was forward referenced, if so fill in the temp.
      StructType *Res = cast_or_null<StructType>(TypeList[NumRecords]);
      if (Res) {
        Res->setName(TypeName);
        TypeList[NumRecords] = nullptr;
      } else  // Otherwise, create a new struct.
        Res = createIdentifiedStructType(Context, TypeName);
      // HLSL Change Begin - avoid name collision for dxil types.
      bool bNameCollision = Res->getName().size() > TypeName.size();
      //TypeName.clear();
      // HLSL Change End.

      SmallVector<Type*, 8> EltTys;
      for (unsigned i = 1, e = Record.size(); i != e; ++i) {
        if (Type *T = getTypeByID(Record[i]))
          EltTys.push_back(T);
        else
          break;
      }
      if (EltTys.size() != Record.size()-1)
        return error("Invalid record");
      Res->setBody(EltTys, Record[0]);
      // HLSL Change Begin - avoid name collision for dxil types.
      if (bNameCollision) {
        StructType *otherType = TheModule->getTypeByName(TypeName);
        if (otherType->isLayoutIdentical(Res)) {
          Res = otherType;
        }
      }
      TypeName.clear();
      // HLSL Change End.
      ResultTy = Res;
      break;
    }
    case bitc::TYPE_CODE_OPAQUE: {       // OPAQUE: []
      if (Record.size() != 1)
        return error("Invalid record");

      if (NumRecords >= TypeList.size())
        return error("Invalid TYPE table");

      // Check to see if this was forward referenced, if so fill in the temp.
      StructType *Res = cast_or_null<StructType>(TypeList[NumRecords]);
      if (Res) {
        Res->setName(TypeName);
        TypeList[NumRecords] = nullptr;
      } else  // Otherwise, create a new struct with no body.
        Res = createIdentifiedStructType(Context, TypeName);
      TypeName.clear();
      ResultTy = Res;
      break;
    }
    case bitc::TYPE_CODE_ARRAY:     // ARRAY: [numelts, eltty]
      if (Record.size() < 2)
        return error("Invalid record");
      ResultTy = getTypeByID(Record[1]);
      if (!ResultTy || !ArrayType::isValidElementType(ResultTy))
        return error("Invalid type");
      ResultTy = ArrayType::get(ResultTy, Record[0]);
      break;
    case bitc::TYPE_CODE_VECTOR:    // VECTOR: [numelts, eltty]
      if (Record.size() < 2)
        return error("Invalid record");
      if (Record[0] == 0)
        return error("Invalid vector length");
      ResultTy = getTypeByID(Record[1]);
      if (!ResultTy || !StructType::isValidElementType(ResultTy))
        return error("Invalid type");
      ResultTy = VectorType::get(ResultTy, Record[0]);
      break;
    }

    if (NumRecords >= TypeList.size())
      return error("Invalid TYPE table");
    if (TypeList[NumRecords])
      return error(
          "Invalid TYPE table: Only named structs can be forward referenced");
    assert(ResultTy && "Didn't read a type?");
    TypeList[NumRecords++] = ResultTy;
  }
}

std::error_code BitcodeReader::parseValueSymbolTable() {
  if (Stream.EnterSubBlock(bitc::VALUE_SYMTAB_BLOCK_ID))
    return error("Invalid record");

  SmallVector<uint64_t, 64> Record;

  Triple TT(TheModule->getTargetTriple());

  // Read all the records for this value table.
  SmallString<128> ValueName;
  while (1) {
    // HLSL Change Starts - count skipped blocks
    unsigned skipCount = 0;
    BitstreamEntry Entry = Stream.advanceSkippingSubblocks(0, &skipCount);
    if (skipCount) ReportWarning(DiagnosticHandler, "Unrecognized subblock");
    // HLSL Change End

    switch (Entry.Kind) {
    case BitstreamEntry::SubBlock: // Handled for us already.
    case BitstreamEntry::Error:
      return error("Malformed block");
    case BitstreamEntry::EndBlock:
      return std::error_code();
    case BitstreamEntry::Record:
      // The interesting case.
      break;
    }

    // Read a record.
    Record.clear();
    switch (Stream.readRecord(Entry.ID, Record)) {
    default:  // Default behavior: unknown type.
      break;
    case bitc::VST_CODE_ENTRY: {  // VST_ENTRY: [valueid, namechar x N]
      if (convertToString(Record, 1, ValueName))
        return error("Invalid record");
      unsigned ValueID = Record[0];
      if (ValueID >= ValueList.size() || !ValueList[ValueID])
        return error("Invalid record");
      Value *V = ValueList[ValueID];

      V->setName(StringRef(ValueName.data(), ValueName.size()));
      if (auto *GO = dyn_cast<GlobalObject>(V)) {
        if (GO->getComdat() == reinterpret_cast<Comdat *>(1)) {
          if (TT.isOSBinFormatMachO())
            GO->setComdat(nullptr);
          else
            GO->setComdat(TheModule->getOrInsertComdat(V->getName()));
        }
      }
      ValueName.clear();
      break;
    }
    case bitc::VST_CODE_BBENTRY: {
      if (convertToString(Record, 1, ValueName))
        return error("Invalid record");
      BasicBlock *BB = getBasicBlock(Record[0]);
      if (!BB)
        return error("Invalid record");

      BB->setName(StringRef(ValueName.data(), ValueName.size()));
      ValueName.clear();
      break;
    }
    }
  }
}

static int64_t unrotateSign(uint64_t U) { return U & 1 ? ~(U >> 1) : U >> 1; }

// HLSL Change - Begin
// This function takes a list of strings that corresponds to the list of named
// metadata that we want to materialize, and materialize them efficiently.
//
// Note: This function will only materialize metadata that are the following
// types:
//
//    MDString        e.g. !"my metadata string"
//    MDNode          e.g. !10 = !{ !"my node", !32, !48 }
//    distinct MDNode e.g. !10 = distinct !{ !"my node", !32, !48 }
//    ValueAsMetadata e.g. true, 0, 10
//
// Everything else will appear as !<temporary>
//
// We first skip through the whole METADATA_BLOCK_ID block. As we do, we take
// note of the named metadata we want, and push their operands into a queue.
// We also record the bit offsets where all String, Node, and Value metadata.
//
// Next, we go through the queue, and skip to their bit offsets and load their
// data (but only if they're the types listed above). If the metadata has their
// own operands, we insert them into the queue as well.
//
//
std::error_code BitcodeReader::parseSelectNamedMetadata(ArrayRef<StringRef> NamedMetadata) {
  // Remember our bit position right at the start, because we're going to
  // jump back to it later.
  uint64_t OriginalBitPos = Stream.GetCurrentBitNo();

  // Buffer used to read record operands.
  SmallVector<uint64_t, 64> Record;
  SmallVector<uint8_t, 32> Uint8Record;

  // A map that we use to remember where we saw each value number
  struct Info {
    uint64_t BitPos;
    uint64_t ID;
    bool IsString;
  };
  std::vector<Info> NodePositions;

  unsigned NextMDValueNo = MDValueList.size();

  if (Stream.EnterSubBlock(bitc::METADATA_BLOCK_ID))
    return error("Invalid record");

  SmallVector<uint64_t, 1> AbbrevDefines;
  std::vector<uint64_t> NodeQueue;
  std::unordered_set<uint64_t> NodeQueueSet;

  auto add_to_queue = [&NodeQueueSet, &NodeQueue](uint64_t Val) {
    if (NodeQueueSet.insert(Val).second)
      NodeQueue.push_back(Val);
  };

  // Read all the records.
  while (1) {
    // If we encounter a DEFINE_ABBREV record, record where it is.
    // We need to go back to them to recover the abbreviation list.
    // There shouldn't be more than one... but just in case.
    if (Stream.PeekCode() == bitc::DEFINE_ABBREV) {
      AbbrevDefines.push_back(Stream.GetCurrentBitNo());
    }

    // HLSL Change Starts - count skipped blocks
    unsigned skipCount = 0;
    BitstreamEntry Entry = Stream.advanceSkippingSubblocks(0, &skipCount);
    if (skipCount) ReportWarning(DiagnosticHandler, "Unrecognized subblock");
    // HLSL Change End

    bool Stop = false;
    switch (Entry.Kind) {
    case BitstreamEntry::SubBlock: // Handled for us already.
    case BitstreamEntry::Error:
      return error("Malformed block");
    case BitstreamEntry::EndBlock:
      MDValueList.tryToResolveCycles();
      Stop = true;
      break;
    case BitstreamEntry::Record:
      // The interesting case.
      break;
    }
    if (Stop)
      break;

    // Peek the record type without changing bit-position and loading
    // the record's operands.
    unsigned PeekCode = Stream.peekRecord(Entry.ID);

    // For the first pass, we're only interested in named metadata.
    if (PeekCode != bitc::METADATA_NAME) {
      // If it's one of these types of things, remember where we actually
      // found it.
      switch (PeekCode) {
      case bitc::METADATA_DISTINCT_NODE:
      case bitc::METADATA_NODE:
      case bitc::METADATA_STRING:
      case bitc::METADATA_VALUE:
        auto old_size = NodePositions.size();
        NodePositions.resize(NextMDValueNo + 1);
        memset(NodePositions.data() + old_size, 0, sizeof(NodePositions[0]) * (NextMDValueNo + 1 - old_size));
        NodePositions[NextMDValueNo] = { Stream.GetCurrentBitNo(), Entry.ID, PeekCode == bitc::METADATA_STRING };
        break;
      }
      // Skip the record without loading anything.
      Stream.skipRecord(Entry.ID);
      NextMDValueNo++;
      continue;
    }

    // Read a record.
    Record.clear();
    unsigned Code = Stream.readRecord(Entry.ID, Record);

    // Read name of the named metadata.
    SmallString<8> Name(Record.begin(), Record.end());
    Record.clear();
    Code = Stream.ReadCode();

    // Figure out if it's one of the named metadata that we actually want.
    bool found = false;
    for (unsigned i = 0; i < NamedMetadata.size(); i++) {
      if (Name == NamedMetadata[i]) {
        found = true;
        break;
      }
    }

    // If it's not interesting to us, then just skip.
    if (!found) {
      Stream.skipRecord(Code);
    }
    else {
      unsigned NextBitCode = Stream.readRecord(Code, Record);
      if (NextBitCode != bitc::METADATA_NAMED_NODE)
        return error("METADATA_NAME not followed by METADATA_NAMED_NODE");

      // Read named metadata elements.
      unsigned Size = Record.size();
      NamedMDNode *NMD = TheModule->getOrInsertNamedMetadata(Name);
      for (unsigned i = 0; i != Size; ++i) {
        MDNode *MD = dyn_cast_or_null<MDNode>(MDValueList.getValueFwdRef(Record[i]));
        if (!MD)
          return error("Invalid record");

        add_to_queue(Record[i]); // Add this MD number to our queue, so we know to try to read it later.
        NMD->addOperand(MD);
      }
    }
  }

  // Now that we have gathered all the metadata operands that the named
  // metadata need...

  // Go back to the beginning.
  Stream.JumpToBit(OriginalBitPos);

  // Re-enter the metadata block.
  if (Stream.EnterSubBlock(bitc::METADATA_BLOCK_ID))
    return error("Invalid record");

  // Load all the abbreviations's again, since exiting the block and re-entering
  // the block has wiped them clean.
  for (unsigned i = 0; i < AbbrevDefines.size(); i++) {
    Stream.JumpToBit(AbbrevDefines[i]);
    while (1) {
      unsigned Code = Stream.ReadCode();
      if (Code == bitc::DEFINE_ABBREV) {
        Stream.ReadAbbrevRecord();
        continue;
      }
      else {
        break;
      }
    }
  }

  std::string String; // String buffer used to read MD string

  // Go through the queue and read all the metadata we want.
  for (unsigned i = 0; i < NodeQueue.size(); i++) {
    uint64_t MDNumber = NodeQueue[i];

    // If we never memorized the location for this MD No, it means
    // it wasn't one of the MD types that we care about.
    if (MDNumber >= NodePositions.size())
      continue;
    Info I = NodePositions[MDNumber];
    if (I.BitPos == 0)
      continue;

    // Go back to the bit where we read the record
    Stream.JumpToBit(I.BitPos);

    // Read a record
    Record.clear();
    unsigned Code = 0;

    // If it's a string, use our special Uint8Buffer to speed up the reading.
    if (I.IsString) {
      Uint8Record.clear();
      Code = Stream.readRecord(I.ID, Record, nullptr, &Uint8Record);
      assert(!Uint8Record.empty() || (Record.empty() && Uint8Record.empty()));
    }
    else {
      Code = Stream.readRecord(I.ID, Record);
    }

    // Read the actual data. This code is largely copied from parseMetadata
    bool IsDistinct = false;
    switch (Code) {
    default:
      llvm_unreachable("Can't actually be anything else.");
      break;
    case bitc::METADATA_VALUE: {
      if (Record.size() != 2)
        return error("Invalid record");

      Type *Ty = getTypeByID(Record[0]);
      if (Ty->isMetadataTy() || Ty->isVoidTy())
        return error("Invalid record");

      MDValueList.assignValue(
          ValueAsMetadata::get(ValueList.getValueFwdRef(Record[1], Ty)),
          MDNumber);
      break;
    }
    case bitc::METADATA_DISTINCT_NODE:
      IsDistinct = true;
      // fallthrough...
    case bitc::METADATA_NODE: {
      SmallVector<Metadata *, 8> Elts;
      Elts.reserve(Record.size());
      for (unsigned ID : Record) {
        Elts.push_back(ID ? MDValueList.getValueFwdRef(ID - 1) : nullptr);
        // If this ID is not a null MD, add to queue.
        if (ID)
          add_to_queue(ID - 1);
      }
      MDValueList.assignValue(IsDistinct ? MDNode::getDistinct(Context, Elts)
                                         : MDNode::get(Context, Elts),
                              MDNumber);
      break;
    }
    case bitc::METADATA_STRING: {
      String.clear();
      String.resize(Uint8Record.size());
      memcpy(&String[0], Uint8Record.data(), Uint8Record.size());
      llvm::UpgradeMDStringConstant(String);
      Metadata *MD = MDString::get(Context, String);
      MDValueList.assignValue(MD, MDNumber);
      break;
    }
    }
  }

  return std::error_code();
}

std::error_code BitcodeReader::materializeSelectNamedMetadata(ArrayRef<StringRef> NamedMetadata) {
  for (uint64_t BitPos : DeferredMetadataInfo) {
    // Move the bit stream to the saved position.
    Stream.JumpToBit(BitPos);
    if (std::error_code EC = parseSelectNamedMetadata(NamedMetadata))
      return EC;
  }
  DeferredMetadataInfo.clear();
  return std::error_code();
}

// HLSL Change - end

std::error_code BitcodeReader::parseMetadata() {
  IsMetadataMaterialized = true;
  unsigned NextMDValueNo = MDValueList.size();

  if (Stream.EnterSubBlock(bitc::METADATA_BLOCK_ID))
    return error("Invalid record");

  SmallVector<uint64_t, 64> Record;
  SmallVector<uint8_t, 64> Uint8Record; // HLSL Change

  auto getMD =
      [&](unsigned ID) -> Metadata *{ return MDValueList.getValueFwdRef(ID); };
  auto getMDOrNull = [&](unsigned ID) -> Metadata *{
    if (ID)
      return getMD(ID - 1);
    return nullptr;
  };
  auto getMDString = [&](unsigned ID) -> MDString *{
    // This requires that the ID is not really a forward reference.  In
    // particular, the MDString must already have been resolved.
    return cast_or_null<MDString>(getMDOrNull(ID));
  };

#define GET_OR_DISTINCT(CLASS, DISTINCT, ARGS)                                 \
  (DISTINCT ? CLASS::getDistinct ARGS : CLASS::get ARGS)

  // Read all the records.
  while (1) {
    // HLSL Change Starts - count skipped blocks
    unsigned skipCount = 0;
    BitstreamEntry Entry = Stream.advanceSkippingSubblocks(0, &skipCount);
    if (skipCount) ReportWarning(DiagnosticHandler, "Unrecognized subblock");
    // HLSL Change End

    switch (Entry.Kind) {
    case BitstreamEntry::SubBlock: // Handled for us already.
    case BitstreamEntry::Error:
      return error("Malformed block");
    case BitstreamEntry::EndBlock:
      MDValueList.tryToResolveCycles();
      return std::error_code();
    case BitstreamEntry::Record:
      // The interesting case.
      break;
    }

#if 1 // HLSL Change
    // If it's a string metadata, use our special Uint8Record to speed
    // up reading.
    unsigned PeekCode = Stream.peekRecord(Entry.ID);
    unsigned Code = 0;
    Record.clear();
    if (PeekCode == bitc::METADATA_STRING) {
      Uint8Record.clear();
      Code = Stream.readRecord(Entry.ID, Record, nullptr, &Uint8Record);
      assert(!Uint8Record.empty() || (Record.empty() && Uint8Record.empty()));
    }
    else {
      Code = Stream.readRecord(Entry.ID, Record);
    }
#else // HLSL Change
    // Read a record.
    Record.clear();
    unsigned Code = Stream.readRecord(Entry.ID, Record);
#endif // HLSL Change

    std::string String; // HLSL Change - Reuse buffer for loading string.
    bool IsDistinct = false;
    switch (Code) {
    default:  // Default behavior: ignore.
      break;
    case bitc::METADATA_NAME: {
      // Read name of the named metadata.
      SmallString<8> Name(Record.begin(), Record.end());
      Record.clear();
      Code = Stream.ReadCode();

      unsigned NextBitCode = Stream.readRecord(Code, Record);
      if (NextBitCode != bitc::METADATA_NAMED_NODE)
        return error("METADATA_NAME not followed by METADATA_NAMED_NODE");

      // Read named metadata elements.
      unsigned Size = Record.size();
      NamedMDNode *NMD = TheModule->getOrInsertNamedMetadata(Name);
      for (unsigned i = 0; i != Size; ++i) {
        MDNode *MD = dyn_cast_or_null<MDNode>(MDValueList.getValueFwdRef(Record[i]));
        if (!MD)
          return error("Invalid record");
        NMD->addOperand(MD);
      }
      break;
    }
    case bitc::METADATA_OLD_FN_NODE: {
      // FIXME: Remove in 4.0.
      // This is a LocalAsMetadata record, the only type of function-local
      // metadata.
      if (Record.size() % 2 == 1)
        return error("Invalid record");

      // If this isn't a LocalAsMetadata record, we're dropping it.  This used
      // to be legal, but there's no upgrade path.
      auto dropRecord = [&] {
        MDValueList.assignValue(MDNode::get(Context, None), NextMDValueNo++);
      };
      if (Record.size() != 2) {
        dropRecord();
        break;
      }

      Type *Ty = getTypeByID(Record[0]);
      if (Ty->isMetadataTy() || Ty->isVoidTy()) {
        dropRecord();
        break;
      }

      MDValueList.assignValue(
          LocalAsMetadata::get(ValueList.getValueFwdRef(Record[1], Ty)),
          NextMDValueNo++);
      break;
    }
    case bitc::METADATA_OLD_NODE: {
      // FIXME: Remove in 4.0.
      if (Record.size() % 2 == 1)
        return error("Invalid record");

      unsigned Size = Record.size();
      SmallVector<Metadata *, 8> Elts;
      for (unsigned i = 0; i != Size; i += 2) {
        Type *Ty = getTypeByID(Record[i]);
        if (!Ty)
          return error("Invalid record");
        if (Ty->isMetadataTy())
          Elts.push_back(MDValueList.getValueFwdRef(Record[i+1]));
        else if (!Ty->isVoidTy()) {
          auto *MD =
              ValueAsMetadata::get(ValueList.getValueFwdRef(Record[i + 1], Ty));
          assert(isa<ConstantAsMetadata>(MD) &&
                 "Expected non-function-local metadata");
          Elts.push_back(MD);
        } else
          Elts.push_back(nullptr);
      }
      MDValueList.assignValue(MDNode::get(Context, Elts), NextMDValueNo++);
      break;
    }
    case bitc::METADATA_VALUE: {
      if (Record.size() != 2)
        return error("Invalid record");

      Type *Ty = getTypeByID(Record[0]);
      if (Ty->isMetadataTy() || Ty->isVoidTy())
        return error("Invalid record");

      MDValueList.assignValue(
          ValueAsMetadata::get(ValueList.getValueFwdRef(Record[1], Ty)),
          NextMDValueNo++);
      break;
    }
    case bitc::METADATA_DISTINCT_NODE:
      IsDistinct = true;
      // fallthrough...
    case bitc::METADATA_NODE: {
      SmallVector<Metadata *, 8> Elts;
      Elts.reserve(Record.size());
      for (unsigned ID : Record)
        Elts.push_back(ID ? MDValueList.getValueFwdRef(ID - 1) : nullptr);
      MDValueList.assignValue(IsDistinct ? MDNode::getDistinct(Context, Elts)
                                         : MDNode::get(Context, Elts),
                              NextMDValueNo++);
      break;
    }
    case bitc::METADATA_LOCATION: {
      if (Record.size() != 5)
        return error("Invalid record");

      unsigned Line = Record[1];
      unsigned Column = Record[2];
      MDNode *Scope = cast<MDNode>(MDValueList.getValueFwdRef(Record[3]));
      Metadata *InlinedAt =
          Record[4] ? MDValueList.getValueFwdRef(Record[4] - 1) : nullptr;
      MDValueList.assignValue(
          GET_OR_DISTINCT(DILocation, Record[0],
                          (Context, Line, Column, Scope, InlinedAt)),
          NextMDValueNo++);
      break;
    }
    case bitc::METADATA_GENERIC_DEBUG: {
      if (Record.size() < 4)
        return error("Invalid record");

      unsigned Tag = Record[1];
      unsigned Version = Record[2];

      if (Tag >= 1u << 16 || Version != 0)
        return error("Invalid record");

      auto *Header = getMDString(Record[3]);
      SmallVector<Metadata *, 8> DwarfOps;
      for (unsigned I = 4, E = Record.size(); I != E; ++I)
        DwarfOps.push_back(Record[I] ? MDValueList.getValueFwdRef(Record[I] - 1)
                                     : nullptr);
      MDValueList.assignValue(GET_OR_DISTINCT(GenericDINode, Record[0],
                                              (Context, Tag, Header, DwarfOps)),
                              NextMDValueNo++);
      break;
    }
    case bitc::METADATA_SUBRANGE: {
      if (Record.size() != 3)
        return error("Invalid record");

      MDValueList.assignValue(
          GET_OR_DISTINCT(DISubrange, Record[0],
                          (Context, Record[1], unrotateSign(Record[2]))),
          NextMDValueNo++);
      break;
    }
    case bitc::METADATA_ENUMERATOR: {
      if (Record.size() != 3)
        return error("Invalid record");

      MDValueList.assignValue(GET_OR_DISTINCT(DIEnumerator, Record[0],
                                              (Context, unrotateSign(Record[1]),
                                               getMDString(Record[2]))),
                              NextMDValueNo++);
      break;
    }
    case bitc::METADATA_BASIC_TYPE: {
      if (Record.size() != 6)
        return error("Invalid record");

      MDValueList.assignValue(
          GET_OR_DISTINCT(DIBasicType, Record[0],
                          (Context, Record[1], getMDString(Record[2]),
                           Record[3], Record[4], Record[5])),
          NextMDValueNo++);
      break;
    }
    case bitc::METADATA_DERIVED_TYPE: {
      if (Record.size() != 12)
        return error("Invalid record");

      MDValueList.assignValue(
          GET_OR_DISTINCT(DIDerivedType, Record[0],
                          (Context, Record[1], getMDString(Record[2]),
                           getMDOrNull(Record[3]), Record[4],
                           getMDOrNull(Record[5]), getMDOrNull(Record[6]),
                           Record[7], Record[8], Record[9], Record[10],
                           getMDOrNull(Record[11]))),
          NextMDValueNo++);
      break;
    }
    case bitc::METADATA_COMPOSITE_TYPE: {
      if (Record.size() != 16)
        return error("Invalid record");

      MDValueList.assignValue(
          GET_OR_DISTINCT(DICompositeType, Record[0],
                          (Context, Record[1], getMDString(Record[2]),
                           getMDOrNull(Record[3]), Record[4],
                           getMDOrNull(Record[5]), getMDOrNull(Record[6]),
                           Record[7], Record[8], Record[9], Record[10],
                           getMDOrNull(Record[11]), Record[12],
                           getMDOrNull(Record[13]), getMDOrNull(Record[14]),
                           getMDString(Record[15]))),
          NextMDValueNo++);
      break;
    }
    case bitc::METADATA_SUBROUTINE_TYPE: {
      if (Record.size() != 3)
        return error("Invalid record");

      MDValueList.assignValue(
          GET_OR_DISTINCT(DISubroutineType, Record[0],
                          (Context, Record[1], getMDOrNull(Record[2]))),
          NextMDValueNo++);
      break;
    }

    case bitc::METADATA_MODULE: {
      if (Record.size() != 6)
        return error("Invalid record");

      MDValueList.assignValue(
          GET_OR_DISTINCT(DIModule, Record[0],
                          (Context, getMDOrNull(Record[1]),
                          getMDString(Record[2]), getMDString(Record[3]),
                          getMDString(Record[4]), getMDString(Record[5]))),
          NextMDValueNo++);
      break;
    }

    case bitc::METADATA_FILE: {
      if (Record.size() != 3)
        return error("Invalid record");

      MDValueList.assignValue(
          GET_OR_DISTINCT(DIFile, Record[0], (Context, getMDString(Record[1]),
                                              getMDString(Record[2]))),
          NextMDValueNo++);
      break;
    }
    case bitc::METADATA_COMPILE_UNIT: {
      if (Record.size() < 14 || Record.size() > 15)
        return error("Invalid record");

      MDValueList.assignValue(
          GET_OR_DISTINCT(
              DICompileUnit, Record[0],
              (Context, Record[1], getMDOrNull(Record[2]),
               getMDString(Record[3]), Record[4], getMDString(Record[5]),
               Record[6], getMDString(Record[7]), Record[8],
               getMDOrNull(Record[9]), getMDOrNull(Record[10]),
               getMDOrNull(Record[11]), getMDOrNull(Record[12]),
               getMDOrNull(Record[13]), Record.size() == 14 ? 0 : Record[14])),
          NextMDValueNo++);
      break;
    }
    case bitc::METADATA_SUBPROGRAM: {
      if (Record.size() != 19)
        return error("Invalid record");

      MDValueList.assignValue(
          GET_OR_DISTINCT(
              DISubprogram, Record[0],
              (Context, getMDOrNull(Record[1]), getMDString(Record[2]),
               getMDString(Record[3]), getMDOrNull(Record[4]), Record[5],
               getMDOrNull(Record[6]), Record[7], Record[8], Record[9],
               getMDOrNull(Record[10]), Record[11], Record[12], Record[13],
               Record[14], getMDOrNull(Record[15]), getMDOrNull(Record[16]),
               getMDOrNull(Record[17]), getMDOrNull(Record[18]))),
          NextMDValueNo++);
      break;
    }
    case bitc::METADATA_LEXICAL_BLOCK: {
      if (Record.size() != 5)
        return error("Invalid record");

      MDValueList.assignValue(
          GET_OR_DISTINCT(DILexicalBlock, Record[0],
                          (Context, getMDOrNull(Record[1]),
                           getMDOrNull(Record[2]), Record[3], Record[4])),
          NextMDValueNo++);
      break;
    }
    case bitc::METADATA_LEXICAL_BLOCK_FILE: {
      if (Record.size() != 4)
        return error("Invalid record");

      MDValueList.assignValue(
          GET_OR_DISTINCT(DILexicalBlockFile, Record[0],
                          (Context, getMDOrNull(Record[1]),
                           getMDOrNull(Record[2]), Record[3])),
          NextMDValueNo++);
      break;
    }
    case bitc::METADATA_NAMESPACE: {
      if (Record.size() != 5)
        return error("Invalid record");

      MDValueList.assignValue(
          GET_OR_DISTINCT(DINamespace, Record[0],
                          (Context, getMDOrNull(Record[1]),
                           getMDOrNull(Record[2]), getMDString(Record[3]),
                           Record[4])),
          NextMDValueNo++);
      break;
    }
    case bitc::METADATA_TEMPLATE_TYPE: {
      if (Record.size() != 3)
        return error("Invalid record");

      MDValueList.assignValue(GET_OR_DISTINCT(DITemplateTypeParameter,
                                              Record[0],
                                              (Context, getMDString(Record[1]),
                                               getMDOrNull(Record[2]))),
                              NextMDValueNo++);
      break;
    }
    case bitc::METADATA_TEMPLATE_VALUE: {
      if (Record.size() != 5)
        return error("Invalid record");

      MDValueList.assignValue(
          GET_OR_DISTINCT(DITemplateValueParameter, Record[0],
                          (Context, Record[1], getMDString(Record[2]),
                           getMDOrNull(Record[3]), getMDOrNull(Record[4]))),
          NextMDValueNo++);
      break;
    }
    case bitc::METADATA_GLOBAL_VAR: {
      if (Record.size() != 11)
        return error("Invalid record");

      MDValueList.assignValue(
          GET_OR_DISTINCT(DIGlobalVariable, Record[0],
                          (Context, getMDOrNull(Record[1]),
                           getMDString(Record[2]), getMDString(Record[3]),
                           getMDOrNull(Record[4]), Record[5],
                           getMDOrNull(Record[6]), Record[7], Record[8],
                           getMDOrNull(Record[9]), getMDOrNull(Record[10]))),
          NextMDValueNo++);
      break;
    }
    case bitc::METADATA_LOCAL_VAR: {
      // 10th field is for the obseleted 'inlinedAt:' field.
      if (Record.size() != 9 && Record.size() != 10)
        return error("Invalid record");

      MDValueList.assignValue(
          GET_OR_DISTINCT(DILocalVariable, Record[0],
                          (Context, Record[1], getMDOrNull(Record[2]),
                           getMDString(Record[3]), getMDOrNull(Record[4]),
                           Record[5], getMDOrNull(Record[6]), Record[7],
                           Record[8])),
          NextMDValueNo++);
      break;
    }
    case bitc::METADATA_EXPRESSION: {
      if (Record.size() < 1)
        return error("Invalid record");

      MDValueList.assignValue(
          GET_OR_DISTINCT(DIExpression, Record[0],
                          (Context, makeArrayRef(Record).slice(1))),
          NextMDValueNo++);
      break;
    }
    case bitc::METADATA_OBJC_PROPERTY: {
      if (Record.size() != 8)
        return error("Invalid record");

      MDValueList.assignValue(
          GET_OR_DISTINCT(DIObjCProperty, Record[0],
                          (Context, getMDString(Record[1]),
                           getMDOrNull(Record[2]), Record[3],
                           getMDString(Record[4]), getMDString(Record[5]),
                           Record[6], getMDOrNull(Record[7]))),
          NextMDValueNo++);
      break;
    }
    case bitc::METADATA_IMPORTED_ENTITY: {
      if (Record.size() != 6)
        return error("Invalid record");

      MDValueList.assignValue(
          GET_OR_DISTINCT(DIImportedEntity, Record[0],
                          (Context, Record[1], getMDOrNull(Record[2]),
                           getMDOrNull(Record[3]), Record[4],
                           getMDString(Record[5]))),
          NextMDValueNo++);
      break;
    }
    case bitc::METADATA_STRING: {
#if 0
      std::string String(Record.begin(), Record.end());
#else
      String.resize(Uint8Record.size());
      memcpy(&String[0], Uint8Record.data(), Uint8Record.size());
#endif
      llvm::UpgradeMDStringConstant(String);
      Metadata *MD = MDString::get(Context, String);
      MDValueList.assignValue(MD, NextMDValueNo++);
      break;
    }
    case bitc::METADATA_KIND: {
      if (Record.size() < 2)
        return error("Invalid record");

      unsigned Kind = Record[0];
      SmallString<8> Name(Record.begin()+1, Record.end());

      unsigned NewKind = TheModule->getMDKindID(Name.str());
      if (!MDKindMap.insert(std::make_pair(Kind, NewKind)).second)
        return error("Conflicting METADATA_KIND records");
      break;
    }
    }
  }
#undef GET_OR_DISTINCT
}

/// Decode a signed value stored with the sign bit in the LSB for dense VBR
/// encoding.
uint64_t BitcodeReader::decodeSignRotatedValue(uint64_t V) {
  if ((V & 1) == 0)
    return V >> 1;
  if (V != 1)
    return -(V >> 1);
  // There is no such thing as -0 with integers.  "-0" really means MININT.
  return 1ULL << 63;
}

/// Resolve all of the initializers for global values and aliases that we can.
std::error_code BitcodeReader::resolveGlobalAndAliasInits() {
  std::vector<std::pair<GlobalVariable*, unsigned> > GlobalInitWorklist;
  std::vector<std::pair<GlobalAlias*, unsigned> > AliasInitWorklist;
  std::vector<std::pair<Function*, unsigned> > FunctionPrefixWorklist;
  std::vector<std::pair<Function*, unsigned> > FunctionPrologueWorklist;
  std::vector<std::pair<Function*, unsigned> > FunctionPersonalityFnWorklist;

  GlobalInitWorklist.swap(GlobalInits);
  AliasInitWorklist.swap(AliasInits);
  FunctionPrefixWorklist.swap(FunctionPrefixes);
  FunctionPrologueWorklist.swap(FunctionPrologues);
  FunctionPersonalityFnWorklist.swap(FunctionPersonalityFns);

  while (!GlobalInitWorklist.empty()) {
    unsigned ValID = GlobalInitWorklist.back().second;
    if (ValID >= ValueList.size()) {
      // Not ready to resolve this yet, it requires something later in the file.
      GlobalInits.push_back(GlobalInitWorklist.back());
    } else {
      if (Constant *C = dyn_cast_or_null<Constant>(ValueList[ValID]))
        GlobalInitWorklist.back().first->setInitializer(C);
      else
        return error("Expected a constant");
    }
    GlobalInitWorklist.pop_back();
  }

  while (!AliasInitWorklist.empty()) {
    unsigned ValID = AliasInitWorklist.back().second;
    if (ValID >= ValueList.size()) {
      AliasInits.push_back(AliasInitWorklist.back());
    } else {
      Constant *C = dyn_cast_or_null<Constant>(ValueList[ValID]);
      if (!C)
        return error("Expected a constant");
      GlobalAlias *Alias = AliasInitWorklist.back().first;
      if (C->getType() != Alias->getType())
        return error("Alias and aliasee types don't match");
      Alias->setAliasee(C);
    }
    AliasInitWorklist.pop_back();
  }

  while (!FunctionPrefixWorklist.empty()) {
    unsigned ValID = FunctionPrefixWorklist.back().second;
    if (ValID >= ValueList.size()) {
      FunctionPrefixes.push_back(FunctionPrefixWorklist.back());
    } else {
      if (Constant *C = dyn_cast_or_null<Constant>(ValueList[ValID]))
        FunctionPrefixWorklist.back().first->setPrefixData(C);
      else
        return error("Expected a constant");
    }
    FunctionPrefixWorklist.pop_back();
  }

  while (!FunctionPrologueWorklist.empty()) {
    unsigned ValID = FunctionPrologueWorklist.back().second;
    if (ValID >= ValueList.size()) {
      FunctionPrologues.push_back(FunctionPrologueWorklist.back());
    } else {
      if (Constant *C = dyn_cast_or_null<Constant>(ValueList[ValID]))
        FunctionPrologueWorklist.back().first->setPrologueData(C);
      else
        return error("Expected a constant");
    }
    FunctionPrologueWorklist.pop_back();
  }

  while (!FunctionPersonalityFnWorklist.empty()) {
    unsigned ValID = FunctionPersonalityFnWorklist.back().second;
    if (ValID >= ValueList.size()) {
      FunctionPersonalityFns.push_back(FunctionPersonalityFnWorklist.back());
    } else {
      if (Constant *C = dyn_cast_or_null<Constant>(ValueList[ValID]))
        FunctionPersonalityFnWorklist.back().first->setPersonalityFn(C);
      else
        return error("Expected a constant");
    }
    FunctionPersonalityFnWorklist.pop_back();
  }

  return std::error_code();
}

static APInt readWideAPInt(ArrayRef<uint64_t> Vals, unsigned TypeBits) {
  SmallVector<uint64_t, 8> Words(Vals.size());
  std::transform(Vals.begin(), Vals.end(), Words.begin(),
                 BitcodeReader::decodeSignRotatedValue);

  return APInt(TypeBits, Words);
}

std::error_code BitcodeReader::parseConstants() {
  if (Stream.EnterSubBlock(bitc::CONSTANTS_BLOCK_ID))
    return error("Invalid record");

  SmallVector<uint64_t, 64> Record;

  // Read all the records for this value table.
  Type *CurTy = Type::getInt32Ty(Context);
  unsigned NextCstNo = ValueList.size();
  while (1) {
    // HLSL Change Starts - count skipped blocks
    unsigned skipCount = 0;
    BitstreamEntry Entry = Stream.advanceSkippingSubblocks(0, &skipCount);
    if (skipCount) ReportWarning(DiagnosticHandler, "Unrecognized subblock");
    // HLSL Change End

    switch (Entry.Kind) {
    case BitstreamEntry::SubBlock: // Handled for us already.
    case BitstreamEntry::Error:
      return error("Malformed block");
    case BitstreamEntry::EndBlock:
      if (NextCstNo != ValueList.size())
        return error("Invalid ronstant reference");

      // Once all the constants have been read, go through and resolve forward
      // references.
      ValueList.resolveConstantForwardRefs();
      return std::error_code();
    case BitstreamEntry::Record:
      // The interesting case.
      break;
    }

    // Read a record.
    Record.clear();
    Value *V = nullptr;
    unsigned BitCode = Stream.readRecord(Entry.ID, Record);
    switch (BitCode) {
    default:  // Default behavior: unknown constant
    case bitc::CST_CODE_UNDEF:     // UNDEF
      V = UndefValue::get(CurTy);
      break;
    case bitc::CST_CODE_SETTYPE:   // SETTYPE: [typeid]
      if (Record.empty())
        return error("Invalid record");
      if (Record[0] >= TypeList.size() || !TypeList[Record[0]])
        return error("Invalid record");
      CurTy = TypeList[Record[0]];
      continue;  // Skip the ValueList manipulation.
    case bitc::CST_CODE_NULL:      // NULL
      V = Constant::getNullValue(CurTy);
      break;
    case bitc::CST_CODE_INTEGER:   // INTEGER: [intval]
      if (!CurTy->isIntegerTy() || Record.empty())
        return error("Invalid record");
      V = ConstantInt::get(CurTy, decodeSignRotatedValue(Record[0]));
      break;
    case bitc::CST_CODE_WIDE_INTEGER: {// WIDE_INTEGER: [n x intval]
      if (!CurTy->isIntegerTy() || Record.empty())
        return error("Invalid record");

      APInt VInt =
          readWideAPInt(Record, cast<IntegerType>(CurTy)->getBitWidth());
      V = ConstantInt::get(Context, VInt);

      break;
    }
    case bitc::CST_CODE_FLOAT: {    // FLOAT: [fpval]
      if (Record.empty())
        return error("Invalid record");
      if (CurTy->isHalfTy())
        V = ConstantFP::get(Context, APFloat(APFloat::IEEEhalf,
                                             APInt(16, (uint16_t)Record[0])));
      else if (CurTy->isFloatTy())
        V = ConstantFP::get(Context, APFloat(APFloat::IEEEsingle,
                                             APInt(32, (uint32_t)Record[0])));
      else if (CurTy->isDoubleTy())
        V = ConstantFP::get(Context, APFloat(APFloat::IEEEdouble,
                                             APInt(64, Record[0])));
      else if (CurTy->isX86_FP80Ty()) {
        // Bits are not stored the same way as a normal i80 APInt, compensate.
        uint64_t Rearrange[2];
        Rearrange[0] = (Record[1] & 0xffffLL) | (Record[0] << 16);
        Rearrange[1] = Record[0] >> 48;
        V = ConstantFP::get(Context, APFloat(APFloat::x87DoubleExtended,
                                             APInt(80, Rearrange)));
      } else if (CurTy->isFP128Ty())
        V = ConstantFP::get(Context, APFloat(APFloat::IEEEquad,
                                             APInt(128, Record)));
      else if (CurTy->isPPC_FP128Ty())
        V = ConstantFP::get(Context, APFloat(APFloat::PPCDoubleDouble,
                                             APInt(128, Record)));
      else
        V = UndefValue::get(CurTy);
      break;
    }

    case bitc::CST_CODE_AGGREGATE: {// AGGREGATE: [n x value number]
      if (Record.empty())
        return error("Invalid record");

      unsigned Size = Record.size();
      SmallVector<Constant*, 16> Elts;

      if (StructType *STy = dyn_cast<StructType>(CurTy)) {
        for (unsigned i = 0; i != Size; ++i)
          Elts.push_back(ValueList.getConstantFwdRef(Record[i],
                                                     STy->getElementType(i)));
        V = ConstantStruct::get(STy, Elts);
      } else if (ArrayType *ATy = dyn_cast<ArrayType>(CurTy)) {
        Type *EltTy = ATy->getElementType();
        for (unsigned i = 0; i != Size; ++i)
          Elts.push_back(ValueList.getConstantFwdRef(Record[i], EltTy));
        V = ConstantArray::get(ATy, Elts);
      } else if (VectorType *VTy = dyn_cast<VectorType>(CurTy)) {
        Type *EltTy = VTy->getElementType();
        for (unsigned i = 0; i != Size; ++i)
          Elts.push_back(ValueList.getConstantFwdRef(Record[i], EltTy));
        V = ConstantVector::get(Elts);
      } else {
        V = UndefValue::get(CurTy);
      }
      break;
    }
    case bitc::CST_CODE_STRING:    // STRING: [values]
    case bitc::CST_CODE_CSTRING: { // CSTRING: [values]
      if (Record.empty())
        return error("Invalid record");

      SmallString<16> Elts(Record.begin(), Record.end());
      V = ConstantDataArray::getString(Context, Elts,
                                       BitCode == bitc::CST_CODE_CSTRING);
      break;
    }
    case bitc::CST_CODE_DATA: {// DATA: [n x value]
      if (Record.empty())
        return error("Invalid record");

      Type *EltTy = cast<SequentialType>(CurTy)->getElementType();
      unsigned Size = Record.size();

      if (EltTy->isIntegerTy(8)) {
        SmallVector<uint8_t, 16> Elts(Record.begin(), Record.end());
        if (isa<VectorType>(CurTy))
          V = ConstantDataVector::get(Context, Elts);
        else
          V = ConstantDataArray::get(Context, Elts);
      } else if (EltTy->isIntegerTy(16)) {
        SmallVector<uint16_t, 16> Elts(Record.begin(), Record.end());
        if (isa<VectorType>(CurTy))
          V = ConstantDataVector::get(Context, Elts);
        else
          V = ConstantDataArray::get(Context, Elts);
      } else if (EltTy->isIntegerTy(32)) {
        SmallVector<uint32_t, 16> Elts(Record.begin(), Record.end());
        if (isa<VectorType>(CurTy))
          V = ConstantDataVector::get(Context, Elts);
        else
          V = ConstantDataArray::get(Context, Elts);
      } else if (EltTy->isIntegerTy(64)) {
        SmallVector<uint64_t, 16> Elts(Record.begin(), Record.end());
        if (isa<VectorType>(CurTy))
          V = ConstantDataVector::get(Context, Elts);
        else
          V = ConstantDataArray::get(Context, Elts);
      } else if (EltTy->isFloatTy()) {
        SmallVector<float, 16> Elts(Size);
        std::transform(Record.begin(), Record.end(), Elts.begin(), BitsToFloat);
        if (isa<VectorType>(CurTy))
          V = ConstantDataVector::get(Context, Elts);
        else
          V = ConstantDataArray::get(Context, Elts);
      } else if (EltTy->isDoubleTy()) {
        SmallVector<double, 16> Elts(Size);
        std::transform(Record.begin(), Record.end(), Elts.begin(),
                       BitsToDouble);
        if (isa<VectorType>(CurTy))
          V = ConstantDataVector::get(Context, Elts);
        else
          V = ConstantDataArray::get(Context, Elts);
      } else {
        return error("Invalid type for value");
      }
      break;
    }

    case bitc::CST_CODE_CE_BINOP: {  // CE_BINOP: [opcode, opval, opval]
      if (Record.size() < 3)
        return error("Invalid record");
      int Opc = getDecodedBinaryOpcode(Record[0], CurTy);
      if (Opc < 0) {
        V = UndefValue::get(CurTy);  // Unknown binop.
      } else {
        Constant *LHS = ValueList.getConstantFwdRef(Record[1], CurTy);
        Constant *RHS = ValueList.getConstantFwdRef(Record[2], CurTy);
        unsigned Flags = 0;
        if (Record.size() >= 4) {
          if (Opc == Instruction::Add ||
              Opc == Instruction::Sub ||
              Opc == Instruction::Mul ||
              Opc == Instruction::Shl) {
            if (Record[3] & (1 << bitc::OBO_NO_SIGNED_WRAP))
              Flags |= OverflowingBinaryOperator::NoSignedWrap;
            if (Record[3] & (1 << bitc::OBO_NO_UNSIGNED_WRAP))
              Flags |= OverflowingBinaryOperator::NoUnsignedWrap;
          } else if (Opc == Instruction::SDiv ||
                     Opc == Instruction::UDiv ||
                     Opc == Instruction::LShr ||
                     Opc == Instruction::AShr) {
            if (Record[3] & (1 << bitc::PEO_EXACT))
              Flags |= SDivOperator::IsExact;
          }
        }
        V = ConstantExpr::get(Opc, LHS, RHS, Flags);
      }
      break;
    }
    case bitc::CST_CODE_CE_CAST: {  // CE_CAST: [opcode, opty, opval]
      if (Record.size() < 3)
        return error("Invalid record");
      int Opc = getDecodedCastOpcode(Record[0]);
      if (Opc < 0) {
        V = UndefValue::get(CurTy);  // Unknown cast.
      } else {
        Type *OpTy = getTypeByID(Record[1]);
        if (!OpTy)
          return error("Invalid record");
        Constant *Op = ValueList.getConstantFwdRef(Record[2], OpTy);
        V = UpgradeBitCastExpr(Opc, Op, CurTy);
        if (!V) V = ConstantExpr::getCast(Opc, Op, CurTy);
      }
      break;
    }
    case bitc::CST_CODE_CE_INBOUNDS_GEP:
    case bitc::CST_CODE_CE_GEP: {  // CE_GEP:        [n x operands]
      unsigned OpNum = 0;
      Type *PointeeType = nullptr;
      if (Record.size() % 2)
        PointeeType = getTypeByID(Record[OpNum++]);
      SmallVector<Constant*, 16> Elts;
      while (OpNum != Record.size()) {
        Type *ElTy = getTypeByID(Record[OpNum++]);
        if (!ElTy)
          return error("Invalid record");
        Elts.push_back(ValueList.getConstantFwdRef(Record[OpNum++], ElTy));
      }

      if (PointeeType &&
          PointeeType !=
              cast<SequentialType>(Elts[0]->getType()->getScalarType())
                  ->getElementType())
        return error("Explicit gep operator type does not match pointee type "
                     "of pointer operand");

      ArrayRef<Constant *> Indices(Elts.begin() + 1, Elts.end());
      V = ConstantExpr::getGetElementPtr(PointeeType, Elts[0], Indices,
                                         BitCode ==
                                             bitc::CST_CODE_CE_INBOUNDS_GEP);
      break;
    }
    case bitc::CST_CODE_CE_SELECT: {  // CE_SELECT: [opval#, opval#, opval#]
      if (Record.size() < 3)
        return error("Invalid record");

      Type *SelectorTy = Type::getInt1Ty(Context);

      // If CurTy is a vector of length n, then Record[0] must be a <n x i1>
      // vector. Otherwise, it must be a single bit.
      if (VectorType *VTy = dyn_cast<VectorType>(CurTy))
        SelectorTy = VectorType::get(Type::getInt1Ty(Context),
                                     VTy->getNumElements());

      V = ConstantExpr::getSelect(ValueList.getConstantFwdRef(Record[0],
                                                              SelectorTy),
                                  ValueList.getConstantFwdRef(Record[1],CurTy),
                                  ValueList.getConstantFwdRef(Record[2],CurTy));
      break;
    }
    case bitc::CST_CODE_CE_EXTRACTELT
        : { // CE_EXTRACTELT: [opty, opval, opty, opval]
      if (Record.size() < 3)
        return error("Invalid record");
      VectorType *OpTy =
        dyn_cast_or_null<VectorType>(getTypeByID(Record[0]));
      if (!OpTy)
        return error("Invalid record");
      Constant *Op0 = ValueList.getConstantFwdRef(Record[1], OpTy);
      Constant *Op1 = nullptr;
      if (Record.size() == 4) {
        Type *IdxTy = getTypeByID(Record[2]);
        if (!IdxTy)
          return error("Invalid record");
        Op1 = ValueList.getConstantFwdRef(Record[3], IdxTy);
      } else // TODO: Remove with llvm 4.0
        Op1 = ValueList.getConstantFwdRef(Record[2], Type::getInt32Ty(Context));
      if (!Op1)
        return error("Invalid record");
      V = ConstantExpr::getExtractElement(Op0, Op1);
      break;
    }
    case bitc::CST_CODE_CE_INSERTELT
        : { // CE_INSERTELT: [opval, opval, opty, opval]
      VectorType *OpTy = dyn_cast<VectorType>(CurTy);
      if (Record.size() < 3 || !OpTy)
        return error("Invalid record");
      Constant *Op0 = ValueList.getConstantFwdRef(Record[0], OpTy);
      Constant *Op1 = ValueList.getConstantFwdRef(Record[1],
                                                  OpTy->getElementType());
      Constant *Op2 = nullptr;
      if (Record.size() == 4) {
        Type *IdxTy = getTypeByID(Record[2]);
        if (!IdxTy)
          return error("Invalid record");
        Op2 = ValueList.getConstantFwdRef(Record[3], IdxTy);
      } else // TODO: Remove with llvm 4.0
        Op2 = ValueList.getConstantFwdRef(Record[2], Type::getInt32Ty(Context));
      if (!Op2)
        return error("Invalid record");
      V = ConstantExpr::getInsertElement(Op0, Op1, Op2);
      break;
    }
    case bitc::CST_CODE_CE_SHUFFLEVEC: { // CE_SHUFFLEVEC: [opval, opval, opval]
      VectorType *OpTy = dyn_cast<VectorType>(CurTy);
      if (Record.size() < 3 || !OpTy)
        return error("Invalid record");
      Constant *Op0 = ValueList.getConstantFwdRef(Record[0], OpTy);
      Constant *Op1 = ValueList.getConstantFwdRef(Record[1], OpTy);
      Type *ShufTy = VectorType::get(Type::getInt32Ty(Context),
                                                 OpTy->getNumElements());
      Constant *Op2 = ValueList.getConstantFwdRef(Record[2], ShufTy);
      V = ConstantExpr::getShuffleVector(Op0, Op1, Op2);
      break;
    }
    case bitc::CST_CODE_CE_SHUFVEC_EX: { // [opty, opval, opval, opval]
      VectorType *RTy = dyn_cast<VectorType>(CurTy);
      VectorType *OpTy =
        dyn_cast_or_null<VectorType>(getTypeByID(Record[0]));
      if (Record.size() < 4 || !RTy || !OpTy)
        return error("Invalid record");
      Constant *Op0 = ValueList.getConstantFwdRef(Record[1], OpTy);
      Constant *Op1 = ValueList.getConstantFwdRef(Record[2], OpTy);
      Type *ShufTy = VectorType::get(Type::getInt32Ty(Context),
                                                 RTy->getNumElements());
      Constant *Op2 = ValueList.getConstantFwdRef(Record[3], ShufTy);
      V = ConstantExpr::getShuffleVector(Op0, Op1, Op2);
      break;
    }
    case bitc::CST_CODE_CE_CMP: {     // CE_CMP: [opty, opval, opval, pred]
      if (Record.size() < 4)
        return error("Invalid record");
      Type *OpTy = getTypeByID(Record[0]);
      if (!OpTy)
        return error("Invalid record");
      Constant *Op0 = ValueList.getConstantFwdRef(Record[1], OpTy);
      Constant *Op1 = ValueList.getConstantFwdRef(Record[2], OpTy);

      if (OpTy->isFPOrFPVectorTy())
        V = ConstantExpr::getFCmp(Record[3], Op0, Op1);
      else
        V = ConstantExpr::getICmp(Record[3], Op0, Op1);
      break;
    }
    // This maintains backward compatibility, pre-asm dialect keywords.
    // FIXME: Remove with the 4.0 release.
    case bitc::CST_CODE_INLINEASM_OLD: {
      if (Record.size() < 2)
        return error("Invalid record");
      std::string AsmStr, ConstrStr;
      bool HasSideEffects = Record[0] & 1;
      bool IsAlignStack = Record[0] >> 1;
      unsigned AsmStrSize = Record[1];
      if (2+AsmStrSize >= Record.size())
        return error("Invalid record");
      unsigned ConstStrSize = Record[2+AsmStrSize];
      if (3+AsmStrSize+ConstStrSize > Record.size())
        return error("Invalid record");

      for (unsigned i = 0; i != AsmStrSize; ++i)
        AsmStr += (char)Record[2+i];
      for (unsigned i = 0; i != ConstStrSize; ++i)
        ConstrStr += (char)Record[3+AsmStrSize+i];
      PointerType *PTy = cast<PointerType>(CurTy);
      V = InlineAsm::get(cast<FunctionType>(PTy->getElementType()),
                         AsmStr, ConstrStr, HasSideEffects, IsAlignStack);
      break;
    }
    // This version adds support for the asm dialect keywords (e.g.,
    // inteldialect).
    case bitc::CST_CODE_INLINEASM: {
      if (Record.size() < 2)
        return error("Invalid record");
      std::string AsmStr, ConstrStr;
      bool HasSideEffects = Record[0] & 1;
      bool IsAlignStack = (Record[0] >> 1) & 1;
      unsigned AsmDialect = Record[0] >> 2;
      unsigned AsmStrSize = Record[1];
      if (2+AsmStrSize >= Record.size())
        return error("Invalid record");
      unsigned ConstStrSize = Record[2+AsmStrSize];
      if (3+AsmStrSize+ConstStrSize > Record.size())
        return error("Invalid record");

      for (unsigned i = 0; i != AsmStrSize; ++i)
        AsmStr += (char)Record[2+i];
      for (unsigned i = 0; i != ConstStrSize; ++i)
        ConstrStr += (char)Record[3+AsmStrSize+i];
      PointerType *PTy = cast<PointerType>(CurTy);
      V = InlineAsm::get(cast<FunctionType>(PTy->getElementType()),
                         AsmStr, ConstrStr, HasSideEffects, IsAlignStack,
                         InlineAsm::AsmDialect(AsmDialect));
      break;
    }
    case bitc::CST_CODE_BLOCKADDRESS:{
      if (Record.size() < 3)
        return error("Invalid record");
      Type *FnTy = getTypeByID(Record[0]);
      if (!FnTy)
        return error("Invalid record");
      Function *Fn =
        dyn_cast_or_null<Function>(ValueList.getConstantFwdRef(Record[1],FnTy));
      if (!Fn)
        return error("Invalid record");

      // Don't let Fn get dematerialized.
      BlockAddressesTaken.insert(Fn);

      // If the function is already parsed we can insert the block address right
      // away.
      BasicBlock *BB;
      unsigned BBID = Record[2];
      if (!BBID)
        // Invalid reference to entry block.
        return error("Invalid ID");
      if (!Fn->empty()) {
        Function::iterator BBI = Fn->begin(), BBE = Fn->end();
        for (size_t I = 0, E = BBID; I != E; ++I) {
          if (BBI == BBE)
            return error("Invalid ID");
          ++BBI;
        }
        BB = BBI;
      } else {
        // Otherwise insert a placeholder and remember it so it can be inserted
        // when the function is parsed.
        auto &FwdBBs = BasicBlockFwdRefs[Fn];
        if (FwdBBs.empty())
          BasicBlockFwdRefQueue.push_back(Fn);
        if (FwdBBs.size() < BBID + 1)
          FwdBBs.resize(BBID + 1);
        if (!FwdBBs[BBID])
          FwdBBs[BBID] = BasicBlock::Create(Context);
        BB = FwdBBs[BBID];
      }
      V = BlockAddress::get(Fn, BB);
      break;
    }
    }

    ValueList.assignValue(V, NextCstNo);
    ++NextCstNo;
  }
}

std::error_code BitcodeReader::parseUseLists() {
  if (Stream.EnterSubBlock(bitc::USELIST_BLOCK_ID))
    return error("Invalid record");

  // Read all the records.
  SmallVector<uint64_t, 64> Record;
  while (1) {
    // HLSL Change Starts - count skipped blocks
    unsigned skipCount = 0;
    BitstreamEntry Entry = Stream.advanceSkippingSubblocks(0, &skipCount);
    if (skipCount) ReportWarning(DiagnosticHandler, "Unrecognized subblock");
    // HLSL Change End

    switch (Entry.Kind) {
    case BitstreamEntry::SubBlock: // Handled for us already.
    case BitstreamEntry::Error:
      return error("Malformed block");
    case BitstreamEntry::EndBlock:
      return std::error_code();
    case BitstreamEntry::Record:
      // The interesting case.
      break;
    }

    // Read a use list record.
    Record.clear();
    bool IsBB = false;
    switch (Stream.readRecord(Entry.ID, Record)) {
    default:  // Default behavior: unknown type.
      break;
    case bitc::USELIST_CODE_BB:
      IsBB = true;
      // fallthrough
    case bitc::USELIST_CODE_DEFAULT: {
      unsigned RecordLength = Record.size();
      if (RecordLength < 3)
        // Records should have at least an ID and two indexes.
        return error("Invalid record");
      unsigned ID = Record.back();
      Record.pop_back();

      Value *V;
      if (IsBB) {
        assert(ID < FunctionBBs.size() && "Basic block not found");
        V = FunctionBBs[ID];
      } else
        V = ValueList[ID];
      unsigned NumUses = 0;
      SmallDenseMap<const Use *, unsigned, 16> Order;
      for (const Use &U : V->uses()) {
        if (++NumUses > Record.size())
          break;
        Order[&U] = Record[NumUses - 1];
      }
      if (Order.size() != Record.size() || NumUses > Record.size())
        // Mismatches can happen if the functions are being materialized lazily
        // (out-of-order), or a value has been upgraded.
        break;

      V->sortUseList([&](const Use &L, const Use &R) {
        return Order.lookup(&L) < Order.lookup(&R);
      });
      break;
    }
    }
  }
}

/// When we see the block for metadata, remember where it is and then skip it.
/// This lets us lazily deserialize the metadata.
std::error_code BitcodeReader::rememberAndSkipMetadata() {
  // Save the current stream state.
  uint64_t CurBit = Stream.GetCurrentBitNo();
  DeferredMetadataInfo.push_back(CurBit);

  // Skip over the block for now.
  if (Stream.SkipBlock())
    return error("Invalid record");
  return std::error_code();
}

std::error_code BitcodeReader::materializeMetadata() {
  for (uint64_t BitPos : DeferredMetadataInfo) {
    // Move the bit stream to the saved position.
    Stream.JumpToBit(BitPos);
    if (std::error_code EC = parseMetadata())
      return EC;
  }
  DeferredMetadataInfo.clear();
  return std::error_code();
}

void BitcodeReader::setStripDebugInfo() { StripDebugInfo = true; }

/// When we see the block for a function body, remember where it is and then
/// skip it.  This lets us lazily deserialize the functions.
std::error_code BitcodeReader::rememberAndSkipFunctionBody() {
  // Get the function we are talking about.
  if (FunctionsWithBodies.empty())
    return error("Insufficient function protos");

  Function *Fn = FunctionsWithBodies.back();
  FunctionsWithBodies.pop_back();

  // Save the current stream state.
  uint64_t CurBit = Stream.GetCurrentBitNo();
  DeferredFunctionInfo[Fn] = CurBit;

  // Skip over the function block for now.
  if (Stream.SkipBlock())
    return error("Invalid record");
  return std::error_code();
}

std::error_code BitcodeReader::globalCleanup() {
  // Patch the initializers for globals and aliases up.
  resolveGlobalAndAliasInits();
  if (!GlobalInits.empty() || !AliasInits.empty())
    return error("Malformed global initializer set");

  // Look for intrinsic functions which need to be upgraded at some point
  for (Function &F : *TheModule) {
    Function *NewFn;
    if (UpgradeIntrinsicFunction(&F, NewFn))
      UpgradedIntrinsics[&F] = NewFn;
  }

  // Look for global variables which need to be renamed.
  for (GlobalVariable &GV : TheModule->globals())
    UpgradeGlobalVariable(&GV);

  // Force deallocation of memory for these vectors to favor the client that
  // want lazy deserialization.
  std::vector<std::pair<GlobalVariable*, unsigned> >().swap(GlobalInits);
  std::vector<std::pair<GlobalAlias*, unsigned> >().swap(AliasInits);
  return std::error_code();
}

std::error_code BitcodeReader::parseModule(bool Resume,
                                           bool ShouldLazyLoadMetadata) {
  if (Resume)
    Stream.JumpToBit(NextUnreadBit);
  else if (Stream.EnterSubBlock(bitc::MODULE_BLOCK_ID))
    return error("Invalid record");

  SmallVector<uint64_t, 64> Record;
  std::vector<std::string> SectionTable;
  std::vector<std::string> GCTable;

  // Read all the records for this module.
  while (1) {
    BitstreamEntry Entry = Stream.advance();

    switch (Entry.Kind) {
    case BitstreamEntry::Error:
      return error("Malformed block");
    case BitstreamEntry::EndBlock:
      return globalCleanup();

    case BitstreamEntry::SubBlock:
      switch (Entry.ID) {
      default:  // Skip unknown content.
        if (Stream.SkipBlock())
          return error("Invalid record");
        ReportWarning(DiagnosticHandler, "Unrecognized block found"); // HLSL Change - check for skipped blocks
        break;
      case bitc::BLOCKINFO_BLOCK_ID:
        // HLSL Changes Start -- check for skipped blocks
        unsigned count;
        count = 0;
        if (Stream.ReadBlockInfoBlock(&count))
          return error("Malformed block");
        if (count > 0)
          ReportWarning(DiagnosticHandler, "Unrecognized block found");
        // HLSL Changes End
        break;
      case bitc::PARAMATTR_BLOCK_ID:
        if (std::error_code EC = parseAttributeBlock())
          return EC;
        break;
      case bitc::PARAMATTR_GROUP_BLOCK_ID:
        if (std::error_code EC = parseAttributeGroupBlock())
          return EC;
        break;
      case bitc::TYPE_BLOCK_ID_NEW:
        if (std::error_code EC = parseTypeTable())
          return EC;
        break;
      case bitc::VALUE_SYMTAB_BLOCK_ID:
        if (std::error_code EC = parseValueSymbolTable())
          return EC;
        SeenValueSymbolTable = true;
        break;
      case bitc::CONSTANTS_BLOCK_ID:
        if (std::error_code EC = parseConstants())
          return EC;
        if (std::error_code EC = resolveGlobalAndAliasInits())
          return EC;
        break;
      case bitc::METADATA_BLOCK_ID:
        if (ShouldLazyLoadMetadata && !IsMetadataMaterialized) {
          if (std::error_code EC = rememberAndSkipMetadata())
            return EC;
          break;
        }
        assert(DeferredMetadataInfo.empty() && "Unexpected deferred metadata");
        if (std::error_code EC = parseMetadata())
          return EC;
        break;
      case bitc::FUNCTION_BLOCK_ID:
        // If this is the first function body we've seen, reverse the
        // FunctionsWithBodies list.
        if (!SeenFirstFunctionBody) {
          std::reverse(FunctionsWithBodies.begin(), FunctionsWithBodies.end());
          if (std::error_code EC = globalCleanup())
            return EC;
          SeenFirstFunctionBody = true;
        }

        if (std::error_code EC = rememberAndSkipFunctionBody())
          return EC;
        // Suspend parsing when we reach the function bodies. Subsequent
        // materialization calls will resume it when necessary. If the bitcode
        // file is old, the symbol table will be at the end instead and will not
        // have been seen yet. In this case, just finish the parse now.
        if (SeenValueSymbolTable) {
          NextUnreadBit = Stream.GetCurrentBitNo();
          return std::error_code();
        }
        break;
      case bitc::USELIST_BLOCK_ID:
        if (std::error_code EC = parseUseLists())
          return EC;
        break;
      }
      continue;

    case BitstreamEntry::Record:
      // The interesting case.
      break;
    }


    // Read a record.
    switch (Stream.readRecord(Entry.ID, Record)) {
    default: break;  // Default behavior, ignore unknown content.
    case bitc::MODULE_CODE_VERSION: {  // VERSION: [version#]
      if (Record.size() < 1)
        return error("Invalid record");
      // Only version #0 and #1 are supported so far.
      unsigned module_version = Record[0];
      switch (module_version) {
        default:
          return error("Invalid value");
        case 0:
          UseRelativeIDs = false;
          break;
        case 1:
          UseRelativeIDs = true;
          break;
      }
      break;
    }
    case bitc::MODULE_CODE_TRIPLE: {  // TRIPLE: [strchr x N]
      std::string S;
      if (convertToString(Record, 0, S))
        return error("Invalid record");
      TheModule->setTargetTriple(S);
      break;
    }
    case bitc::MODULE_CODE_DATALAYOUT: {  // DATALAYOUT: [strchr x N]
      std::string S;
      if (convertToString(Record, 0, S))
        return error("Invalid record");
      TheModule->setDataLayout(S);
      break;
    }
    case bitc::MODULE_CODE_ASM: {  // ASM: [strchr x N]
      std::string S;
      if (convertToString(Record, 0, S))
        return error("Invalid record");
      TheModule->setModuleInlineAsm(S);
      break;
    }
    case bitc::MODULE_CODE_DEPLIB: {  // DEPLIB: [strchr x N]
      // FIXME: Remove in 4.0.
      std::string S;
      if (convertToString(Record, 0, S))
        return error("Invalid record");
      // Ignore value.
      break;
    }
    case bitc::MODULE_CODE_SECTIONNAME: {  // SECTIONNAME: [strchr x N]
      std::string S;
      if (convertToString(Record, 0, S))
        return error("Invalid record");
      SectionTable.push_back(S);
      break;
    }
    case bitc::MODULE_CODE_GCNAME: {  // SECTIONNAME: [strchr x N]
      std::string S;
      if (convertToString(Record, 0, S))
        return error("Invalid record");
      GCTable.push_back(S);
      break;
    }
    case bitc::MODULE_CODE_COMDAT: { // COMDAT: [selection_kind, name]
      if (Record.size() < 2)
        return error("Invalid record");
      Comdat::SelectionKind SK = getDecodedComdatSelectionKind(Record[0]);
      unsigned ComdatNameSize = Record[1];
      std::string ComdatName;
      ComdatName.reserve(ComdatNameSize);
      for (unsigned i = 0; i != ComdatNameSize; ++i)
        ComdatName += (char)Record[2 + i];
      Comdat *C = TheModule->getOrInsertComdat(ComdatName);
      C->setSelectionKind(SK);
      ComdatList.push_back(C);
      break;
    }
    // GLOBALVAR: [pointer type, isconst, initid,
    //             linkage, alignment, section, visibility, threadlocal,
    //             unnamed_addr, externally_initialized, dllstorageclass,
    //             comdat]
    case bitc::MODULE_CODE_GLOBALVAR: {
      if (Record.size() < 6)
        return error("Invalid record");
      Type *Ty = getTypeByID(Record[0]);
      if (!Ty)
        return error("Invalid record");
      bool isConstant = Record[1] & 1;
      bool explicitType = Record[1] & 2;
      unsigned AddressSpace;
      if (explicitType) {
        AddressSpace = Record[1] >> 2;
      } else {
        if (!Ty->isPointerTy())
          return error("Invalid type for value");
        AddressSpace = cast<PointerType>(Ty)->getAddressSpace();
        Ty = cast<PointerType>(Ty)->getElementType();
      }

      uint64_t RawLinkage = Record[3];
      GlobalValue::LinkageTypes Linkage = getDecodedLinkage(RawLinkage);
      unsigned Alignment;
      if (std::error_code EC = parseAlignmentValue(Record[4], Alignment))
        return EC;
      std::string Section;
      if (Record[5]) {
        if (Record[5]-1 >= SectionTable.size())
          return error("Invalid ID");
        Section = SectionTable[Record[5]-1];
      }
      GlobalValue::VisibilityTypes Visibility = GlobalValue::DefaultVisibility;
      // Local linkage must have default visibility.
      if (Record.size() > 6 && !GlobalValue::isLocalLinkage(Linkage))
        // FIXME: Change to an error if non-default in 4.0.
        Visibility = getDecodedVisibility(Record[6]);

      GlobalVariable::ThreadLocalMode TLM = GlobalVariable::NotThreadLocal;
      if (Record.size() > 7)
        TLM = getDecodedThreadLocalMode(Record[7]);

      bool UnnamedAddr = false;
      if (Record.size() > 8)
        UnnamedAddr = Record[8];

      bool ExternallyInitialized = false;
      if (Record.size() > 9)
        ExternallyInitialized = Record[9];

      GlobalVariable *NewGV =
        new GlobalVariable(*TheModule, Ty, isConstant, Linkage, nullptr, "", nullptr,
                           TLM, AddressSpace, ExternallyInitialized);
      NewGV->setAlignment(Alignment);
      if (!Section.empty())
        NewGV->setSection(Section);
      NewGV->setVisibility(Visibility);
      NewGV->setUnnamedAddr(UnnamedAddr);

      if (Record.size() > 10)
        NewGV->setDLLStorageClass(getDecodedDLLStorageClass(Record[10]));
      else
        upgradeDLLImportExportLinkage(NewGV, RawLinkage);

      ValueList.push_back(NewGV);

      // Remember which value to use for the global initializer.
      if (unsigned InitID = Record[2])
        GlobalInits.push_back(std::make_pair(NewGV, InitID-1));

      if (Record.size() > 11) {
        if (unsigned ComdatID = Record[11]) {
          if (ComdatID > ComdatList.size())
            return error("Invalid global variable comdat ID");
          NewGV->setComdat(ComdatList[ComdatID - 1]);
        }
      } else if (hasImplicitComdat(RawLinkage)) {
        NewGV->setComdat(reinterpret_cast<Comdat *>(1));
      }
      break;
    }
    // FUNCTION:  [type, callingconv, isproto, linkage, paramattr,
    //             alignment, section, visibility, gc, unnamed_addr,
    //             prologuedata, dllstorageclass, comdat, prefixdata]
    case bitc::MODULE_CODE_FUNCTION: {
      if (Record.size() < 8)
        return error("Invalid record");
      Type *Ty = getTypeByID(Record[0]);
      if (!Ty)
        return error("Invalid record");
      if (auto *PTy = dyn_cast<PointerType>(Ty))
        Ty = PTy->getElementType();
      auto *FTy = dyn_cast<FunctionType>(Ty);
      if (!FTy)
        return error("Invalid type for value");

      Function *Func = Function::Create(FTy, GlobalValue::ExternalLinkage,
                                        "", TheModule);

      Func->setCallingConv(static_cast<CallingConv::ID>(Record[1]));
      bool isProto = Record[2];
      uint64_t RawLinkage = Record[3];
      Func->setLinkage(getDecodedLinkage(RawLinkage));
      Func->setAttributes(getAttributes(Record[4]));

      unsigned Alignment;
      if (std::error_code EC = parseAlignmentValue(Record[5], Alignment))
        return EC;
      Func->setAlignment(Alignment);
      if (Record[6]) {
        if (Record[6]-1 >= SectionTable.size())
          return error("Invalid ID");
        Func->setSection(SectionTable[Record[6]-1]);
      }
      // Local linkage must have default visibility.
      if (!Func->hasLocalLinkage())
        // FIXME: Change to an error if non-default in 4.0.
        Func->setVisibility(getDecodedVisibility(Record[7]));
      if (Record.size() > 8 && Record[8]) {
        if (Record[8]-1 >= GCTable.size())
          return error("Invalid ID");
        Func->setGC(GCTable[Record[8]-1].c_str());
      }
      bool UnnamedAddr = false;
      if (Record.size() > 9)
        UnnamedAddr = Record[9];
      Func->setUnnamedAddr(UnnamedAddr);
      if (Record.size() > 10 && Record[10] != 0)
        FunctionPrologues.push_back(std::make_pair(Func, Record[10]-1));

      if (Record.size() > 11)
        Func->setDLLStorageClass(getDecodedDLLStorageClass(Record[11]));
      else
        upgradeDLLImportExportLinkage(Func, RawLinkage);

      if (Record.size() > 12) {
        if (unsigned ComdatID = Record[12]) {
          if (ComdatID > ComdatList.size())
            return error("Invalid function comdat ID");
          Func->setComdat(ComdatList[ComdatID - 1]);
        }
      } else if (hasImplicitComdat(RawLinkage)) {
        Func->setComdat(reinterpret_cast<Comdat *>(1));
      }

      if (Record.size() > 13 && Record[13] != 0)
        FunctionPrefixes.push_back(std::make_pair(Func, Record[13]-1));

      if (Record.size() > 14 && Record[14] != 0)
        FunctionPersonalityFns.push_back(std::make_pair(Func, Record[14] - 1));

      ValueList.push_back(Func);

      // If this is a function with a body, remember the prototype we are
      // creating now, so that we can match up the body with them later.
      if (!isProto) {
        Func->setIsMaterializable(true);
        FunctionsWithBodies.push_back(Func);
        DeferredFunctionInfo[Func] = 0;
      }
      break;
    }
    // ALIAS: [alias type, aliasee val#, linkage]
    // ALIAS: [alias type, aliasee val#, linkage, visibility, dllstorageclass]
    case bitc::MODULE_CODE_ALIAS: {
      if (Record.size() < 3)
        return error("Invalid record");
      Type *Ty = getTypeByID(Record[0]);
      if (!Ty)
        return error("Invalid record");
      auto *PTy = dyn_cast<PointerType>(Ty);
      if (!PTy)
        return error("Invalid type for value");

      auto *NewGA =
          GlobalAlias::create(PTy, getDecodedLinkage(Record[2]), "", TheModule);
      // Old bitcode files didn't have visibility field.
      // Local linkage must have default visibility.
      if (Record.size() > 3 && !NewGA->hasLocalLinkage())
        // FIXME: Change to an error if non-default in 4.0.
        NewGA->setVisibility(getDecodedVisibility(Record[3]));
      if (Record.size() > 4)
        NewGA->setDLLStorageClass(getDecodedDLLStorageClass(Record[4]));
      else
        upgradeDLLImportExportLinkage(NewGA, Record[2]);
      if (Record.size() > 5)
        NewGA->setThreadLocalMode(getDecodedThreadLocalMode(Record[5]));
      if (Record.size() > 6)
        NewGA->setUnnamedAddr(Record[6]);
      ValueList.push_back(NewGA);
      AliasInits.push_back(std::make_pair(NewGA, Record[1]));
      break;
    }
    /// MODULE_CODE_PURGEVALS: [numvals]
    case bitc::MODULE_CODE_PURGEVALS:
      // Trim down the value list to the specified size.
      if (Record.size() < 1 || Record[0] > ValueList.size())
        return error("Invalid record");
      ValueList.shrinkTo(Record[0]);
      break;
    }
    Record.clear();
  }
}

std::error_code
BitcodeReader::parseBitcodeInto(std::unique_ptr<DataStreamer> Streamer,
                                Module *M, bool ShouldLazyLoadMetadata) {
  TheModule = M;

  if (std::error_code EC = initStream(std::move(Streamer)))
    return EC;

  // Sniff for the signature.
  if (Stream.Read(8) != 'B' ||
      Stream.Read(8) != 'C' ||
      Stream.Read(4) != 0x0 ||
      Stream.Read(4) != 0xC ||
      Stream.Read(4) != 0xE ||
      Stream.Read(4) != 0xD)
    return error("Invalid bitcode signature");

  // We expect a number of well-defined blocks, though we don't necessarily
  // need to understand them all.
  while (1) {
    if (Stream.AtEndOfStream()) {
      return error("Malformed IR file");
    }

    BitstreamEntry Entry =
      Stream.advance(BitstreamCursor::AF_DontAutoprocessAbbrevs);

    if (Entry.Kind != BitstreamEntry::SubBlock)
      return error("Malformed block");

    // HLSL Change - process the complete stream.
    if (Entry.ID == bitc::MODULE_BLOCK_ID) {
      return parseModule(false, ShouldLazyLoadMetadata);
    }
    else {
      ReportWarning(DiagnosticHandler, "Unrecognized block found");
      if (Stream.SkipBlock())
        return error("Invalid record");
    }
    // HLSL Change Ends - process the complete stream.
  }
}

ErrorOr<std::string> BitcodeReader::parseModuleTriple() {
  if (Stream.EnterSubBlock(bitc::MODULE_BLOCK_ID))
    return error("Invalid record");

  SmallVector<uint64_t, 64> Record;

  std::string Triple;
  // Read all the records for this module.
  while (1) {
    // HLSL Change Starts - count skipped blocks
    unsigned skipCount = 0;
    BitstreamEntry Entry = Stream.advanceSkippingSubblocks(0, &skipCount);
    if (skipCount) ReportWarning(DiagnosticHandler, "Unrecognized subblock");
    // HLSL Change End

    switch (Entry.Kind) {
    case BitstreamEntry::SubBlock: // Handled for us already.
    case BitstreamEntry::Error:
      return error("Malformed block");
    case BitstreamEntry::EndBlock:
      return Triple;
    case BitstreamEntry::Record:
      // The interesting case.
      break;
    }

    // Read a record.
    switch (Stream.readRecord(Entry.ID, Record)) {
    default: break;  // Default behavior, ignore unknown content.
    case bitc::MODULE_CODE_TRIPLE: {  // TRIPLE: [strchr x N]
      std::string S;
      if (convertToString(Record, 0, S))
        return error("Invalid record");
      Triple = S;
      break;
    }
    }
    Record.clear();
  }
  llvm_unreachable("Exit infinite loop");
}

ErrorOr<std::string> BitcodeReader::parseTriple() {
  if (std::error_code EC = initStream(nullptr))
    return EC;

  // Sniff for the signature.
  if (Stream.Read(8) != 'B' ||
      Stream.Read(8) != 'C' ||
      Stream.Read(4) != 0x0 ||
      Stream.Read(4) != 0xC ||
      Stream.Read(4) != 0xE ||
      Stream.Read(4) != 0xD)
    return error("Invalid bitcode signature");

  // We expect a number of well-defined blocks, though we don't necessarily
  // need to understand them all.
  while (1) {
    BitstreamEntry Entry = Stream.advance();

    switch (Entry.Kind) {
    case BitstreamEntry::Error:
      return error("Malformed block");
    case BitstreamEntry::EndBlock:
      return std::error_code();

    case BitstreamEntry::SubBlock:
      if (Entry.ID == bitc::MODULE_BLOCK_ID)
        return parseModuleTriple();

      // Ignore other sub-blocks.
      if (Stream.SkipBlock())
        return error("Malformed block");
      ReportWarning(DiagnosticHandler, "Unrecognized block found");
      continue;

    case BitstreamEntry::Record:
      Stream.skipRecord(Entry.ID);
      continue;
    }
  }
}

/// Parse metadata attachments.
std::error_code BitcodeReader::parseMetadataAttachment(Function &F) {
  if (Stream.EnterSubBlock(bitc::METADATA_ATTACHMENT_ID))
    return error("Invalid record");

  SmallVector<uint64_t, 64> Record;
  while (1) {
    // HLSL Change Starts - count skipped blocks
    unsigned skipCount = 0;
    BitstreamEntry Entry = Stream.advanceSkippingSubblocks(0, &skipCount);
    if (skipCount) ReportWarning(DiagnosticHandler, "Unrecognized subblock");
    // HLSL Change End

    switch (Entry.Kind) {
    case BitstreamEntry::SubBlock: // Handled for us already.
    case BitstreamEntry::Error:
      return error("Malformed block");
    case BitstreamEntry::EndBlock:
      return std::error_code();
    case BitstreamEntry::Record:
      // The interesting case.
      break;
    }

    // Read a metadata attachment record.
    Record.clear();
    switch (Stream.readRecord(Entry.ID, Record)) {
    default:  // Default behavior: ignore.
      break;
    case bitc::METADATA_ATTACHMENT: {
      unsigned RecordLength = Record.size();
      if (Record.empty())
        return error("Invalid record");
      if (RecordLength % 2 == 0) {
        // A function attachment.
        for (unsigned I = 0; I != RecordLength; I += 2) {
          auto K = MDKindMap.find(Record[I]);
          if (K == MDKindMap.end())
            return error("Invalid ID");
          Metadata *MD = MDValueList.getValueFwdRef(Record[I + 1]);
          F.setMetadata(K->second, cast<MDNode>(MD));
        }
        continue;
      }

      // An instruction attachment.
      Instruction *Inst = InstructionList[Record[0]];
      for (unsigned i = 1; i != RecordLength; i = i+2) {
        unsigned Kind = Record[i];
        DenseMap<unsigned, unsigned>::iterator I =
          MDKindMap.find(Kind);
        if (I == MDKindMap.end())
          return error("Invalid ID");
        Metadata *Node = MDValueList.getValueFwdRef(Record[i + 1]);
        if (isa<LocalAsMetadata>(Node))
          // Drop the attachment.  This used to be legal, but there's no
          // upgrade path.
          break;
        Inst->setMetadata(I->second, cast<MDNode>(Node));
        if (I->second == LLVMContext::MD_tbaa)
          InstsWithTBAATag.push_back(Inst);
      }
      break;
    }
    }
  }
}

static std::error_code typeCheckLoadStoreInst(DiagnosticHandlerFunction DH,
                                              Type *ValType, Type *PtrType) {
  if (!isa<PointerType>(PtrType))
    return error(DH, "Load/Store operand is not a pointer type");
  Type *ElemType = cast<PointerType>(PtrType)->getElementType();

  if (ValType && ValType != ElemType)
    return error(DH, "Explicit load/store type does not match pointee type of "
                     "pointer operand");
  if (!PointerType::isLoadableOrStorableType(ElemType))
    return error(DH, "Cannot load/store from pointer");
  return std::error_code();
}

/// Lazily parse the specified function body block.
std::error_code BitcodeReader::parseFunctionBody(Function *F) {
  if (Stream.EnterSubBlock(bitc::FUNCTION_BLOCK_ID))
    return error("Invalid record");

  InstructionList.clear();
  unsigned ModuleValueListSize = ValueList.size();
  unsigned ModuleMDValueListSize = MDValueList.size();

  // Add all the function arguments to the value table.
  for(Function::arg_iterator I = F->arg_begin(), E = F->arg_end(); I != E; ++I)
    ValueList.push_back(I);

  unsigned NextValueNo = ValueList.size();
  BasicBlock *CurBB = nullptr;
  unsigned CurBBNo = 0;

  DebugLoc LastLoc;
  auto getLastInstruction = [&]() -> Instruction * {
    if (CurBB && !CurBB->empty())
      return &CurBB->back();
    else if (CurBBNo && FunctionBBs[CurBBNo - 1] &&
             !FunctionBBs[CurBBNo - 1]->empty())
      return &FunctionBBs[CurBBNo - 1]->back();
    return nullptr;
  };

  // Read all the records.
  SmallVector<uint64_t, 64> Record;
  while (1) {
    BitstreamEntry Entry = Stream.advance();

    switch (Entry.Kind) {
    case BitstreamEntry::Error:
      return error("Malformed block");
    case BitstreamEntry::EndBlock:
      goto OutOfRecordLoop;

    case BitstreamEntry::SubBlock:
      switch (Entry.ID) {
      default:  // Skip unknown content.
        if (Stream.SkipBlock())
          return error("Invalid record");
        ReportWarning(DiagnosticHandler, "Unrecognized block found");
        break;
      case bitc::CONSTANTS_BLOCK_ID:
        if (std::error_code EC = parseConstants())
          return EC;
        NextValueNo = ValueList.size();
        break;
      case bitc::VALUE_SYMTAB_BLOCK_ID:
        if (std::error_code EC = parseValueSymbolTable())
          return EC;
        break;
      case bitc::METADATA_ATTACHMENT_ID:
        if (std::error_code EC = parseMetadataAttachment(*F))
          return EC;
        break;
      case bitc::METADATA_BLOCK_ID:
        if (std::error_code EC = parseMetadata())
          return EC;
        break;
      case bitc::USELIST_BLOCK_ID:
        if (std::error_code EC = parseUseLists())
          return EC;
        break;
      }
      continue;

    case BitstreamEntry::Record:
      // The interesting case.
      break;
    }

    // Read a record.
    Record.clear();
    Instruction *I = nullptr;
    unsigned BitCode = Stream.readRecord(Entry.ID, Record);
    switch (BitCode) {
    default: // Default behavior: reject
      return error("Invalid value");
    case bitc::FUNC_CODE_DECLAREBLOCKS: {   // DECLAREBLOCKS: [nblocks]
      if (Record.size() < 1 || Record[0] == 0)
        return error("Invalid record");
      // Create all the basic blocks for the function.
      FunctionBBs.resize(Record[0]);

      // See if anything took the address of blocks in this function.
      auto BBFRI = BasicBlockFwdRefs.find(F);
      if (BBFRI == BasicBlockFwdRefs.end()) {
        for (unsigned i = 0, e = FunctionBBs.size(); i != e; ++i)
          FunctionBBs[i] = BasicBlock::Create(Context, "", F);
      } else {
        auto &BBRefs = BBFRI->second;
        // Check for invalid basic block references.
        if (BBRefs.size() > FunctionBBs.size())
          return error("Invalid ID");
        assert(!BBRefs.empty() && "Unexpected empty array");
        assert(!BBRefs.front() && "Invalid reference to entry block");
        for (unsigned I = 0, E = FunctionBBs.size(), RE = BBRefs.size(); I != E;
             ++I)
          if (I < RE && BBRefs[I]) {
            BBRefs[I]->insertInto(F);
            FunctionBBs[I] = BBRefs[I];
          } else {
            FunctionBBs[I] = BasicBlock::Create(Context, "", F);
          }

        // Erase from the table.
        BasicBlockFwdRefs.erase(BBFRI);
      }

      CurBB = FunctionBBs[0];
      continue;
    }

    case bitc::FUNC_CODE_DEBUG_LOC_AGAIN:  // DEBUG_LOC_AGAIN
      // This record indicates that the last instruction is at the same
      // location as the previous instruction with a location.
      I = getLastInstruction();

      if (!I)
        return error("Invalid record");
      I->setDebugLoc(LastLoc);
      I = nullptr;
      continue;

    case bitc::FUNC_CODE_DEBUG_LOC: {      // DEBUG_LOC: [line, col, scope, ia]
      I = getLastInstruction();
      if (!I || Record.size() < 4)
        return error("Invalid record");

      unsigned Line = Record[0], Col = Record[1];
      unsigned ScopeID = Record[2], IAID = Record[3];

      MDNode *Scope = nullptr, *IA = nullptr;
      if (ScopeID) Scope = cast<MDNode>(MDValueList.getValueFwdRef(ScopeID-1));
      if (IAID)    IA = cast<MDNode>(MDValueList.getValueFwdRef(IAID-1));
      LastLoc = DebugLoc::get(Line, Col, Scope, IA);
      I->setDebugLoc(LastLoc);
      I = nullptr;
      continue;
    }

    case bitc::FUNC_CODE_INST_BINOP: {    // BINOP: [opval, ty, opval, opcode]
      unsigned OpNum = 0;
      Value *LHS, *RHS;
      if (getValueTypePair(Record, OpNum, NextValueNo, LHS) ||
          popValue(Record, OpNum, NextValueNo, LHS->getType(), RHS) ||
          OpNum+1 > Record.size())
        return error("Invalid record");

      int Opc = getDecodedBinaryOpcode(Record[OpNum++], LHS->getType());
      if (Opc == -1)
        return error("Invalid record");
      I = BinaryOperator::Create((Instruction::BinaryOps)Opc, LHS, RHS);
      InstructionList.push_back(I);
      if (OpNum < Record.size()) {
        if (Opc == Instruction::Add ||
            Opc == Instruction::Sub ||
            Opc == Instruction::Mul ||
            Opc == Instruction::Shl) {
          if (Record[OpNum] & (1 << bitc::OBO_NO_SIGNED_WRAP))
            cast<BinaryOperator>(I)->setHasNoSignedWrap(true);
          if (Record[OpNum] & (1 << bitc::OBO_NO_UNSIGNED_WRAP))
            cast<BinaryOperator>(I)->setHasNoUnsignedWrap(true);
        } else if (Opc == Instruction::SDiv ||
                   Opc == Instruction::UDiv ||
                   Opc == Instruction::LShr ||
                   Opc == Instruction::AShr) {
          if (Record[OpNum] & (1 << bitc::PEO_EXACT))
            cast<BinaryOperator>(I)->setIsExact(true);
        } else if (isa<FPMathOperator>(I)) {
          FastMathFlags FMF = getDecodedFastMathFlags(Record[OpNum]);
          if (FMF.any())
            I->setFastMathFlags(FMF);
        }

      }
      break;
    }
    case bitc::FUNC_CODE_INST_CAST: {    // CAST: [opval, opty, destty, castopc]
      unsigned OpNum = 0;
      Value *Op;
      if (getValueTypePair(Record, OpNum, NextValueNo, Op) ||
          OpNum+2 != Record.size())
        return error("Invalid record");

      Type *ResTy = getTypeByID(Record[OpNum]);
      int Opc = getDecodedCastOpcode(Record[OpNum + 1]);
      if (Opc == -1 || !ResTy)
        return error("Invalid record");
      Instruction *Temp = nullptr;
      if ((I = UpgradeBitCastInst(Opc, Op, ResTy, Temp))) {
        if (Temp) {
          InstructionList.push_back(Temp);
          CurBB->getInstList().push_back(Temp);
        }
      } else {
        I = CastInst::Create((Instruction::CastOps)Opc, Op, ResTy);
      }
      InstructionList.push_back(I);
      break;
    }
    case bitc::FUNC_CODE_INST_INBOUNDS_GEP_OLD:
    case bitc::FUNC_CODE_INST_GEP_OLD:
    case bitc::FUNC_CODE_INST_GEP: { // GEP: type, [n x operands]
      unsigned OpNum = 0;

      Type *Ty;
      bool InBounds;

      if (BitCode == bitc::FUNC_CODE_INST_GEP) {
        InBounds = Record[OpNum++];
        Ty = getTypeByID(Record[OpNum++]);
      } else {
        InBounds = BitCode == bitc::FUNC_CODE_INST_INBOUNDS_GEP_OLD;
        Ty = nullptr;
      }

      Value *BasePtr;
      if (getValueTypePair(Record, OpNum, NextValueNo, BasePtr))
        return error("Invalid record");

      if (!Ty)
        Ty = cast<SequentialType>(BasePtr->getType()->getScalarType())
                 ->getElementType();
      else if (Ty !=
               cast<SequentialType>(BasePtr->getType()->getScalarType())
                   ->getElementType())
        return error(
            "Explicit gep type does not match pointee type of pointer operand");

      SmallVector<Value*, 16> GEPIdx;
      while (OpNum != Record.size()) {
        Value *Op;
        if (getValueTypePair(Record, OpNum, NextValueNo, Op))
          return error("Invalid record");
        GEPIdx.push_back(Op);
      }

      I = GetElementPtrInst::Create(Ty, BasePtr, GEPIdx);

      InstructionList.push_back(I);
      if (InBounds)
        cast<GetElementPtrInst>(I)->setIsInBounds(true);
      break;
    }

    case bitc::FUNC_CODE_INST_EXTRACTVAL: {
                                       // EXTRACTVAL: [opty, opval, n x indices]
      unsigned OpNum = 0;
      Value *Agg;
      if (getValueTypePair(Record, OpNum, NextValueNo, Agg))
        return error("Invalid record");

      unsigned RecSize = Record.size();
      if (OpNum == RecSize)
        return error("EXTRACTVAL: Invalid instruction with 0 indices");

      SmallVector<unsigned, 4> EXTRACTVALIdx;
      Type *CurTy = Agg->getType();
      for (; OpNum != RecSize; ++OpNum) {
        bool IsArray = CurTy->isArrayTy();
        bool IsStruct = CurTy->isStructTy();
        uint64_t Index = Record[OpNum];

        if (!IsStruct && !IsArray)
          return error("EXTRACTVAL: Invalid type");
        if ((unsigned)Index != Index)
          return error("Invalid value");
        if (IsStruct && Index >= CurTy->subtypes().size())
          return error("EXTRACTVAL: Invalid struct index");
        if (IsArray && Index >= CurTy->getArrayNumElements())
          return error("EXTRACTVAL: Invalid array index");
        EXTRACTVALIdx.push_back((unsigned)Index);

        if (IsStruct)
          CurTy = CurTy->subtypes()[Index];
        else
          CurTy = CurTy->subtypes()[0];
      }

      I = ExtractValueInst::Create(Agg, EXTRACTVALIdx);
      InstructionList.push_back(I);
      break;
    }

    case bitc::FUNC_CODE_INST_INSERTVAL: {
                           // INSERTVAL: [opty, opval, opty, opval, n x indices]
      unsigned OpNum = 0;
      Value *Agg;
      if (getValueTypePair(Record, OpNum, NextValueNo, Agg))
        return error("Invalid record");
      Value *Val;
      if (getValueTypePair(Record, OpNum, NextValueNo, Val))
        return error("Invalid record");

      unsigned RecSize = Record.size();
      if (OpNum == RecSize)
        return error("INSERTVAL: Invalid instruction with 0 indices");

      SmallVector<unsigned, 4> INSERTVALIdx;
      Type *CurTy = Agg->getType();
      for (; OpNum != RecSize; ++OpNum) {
        bool IsArray = CurTy->isArrayTy();
        bool IsStruct = CurTy->isStructTy();
        uint64_t Index = Record[OpNum];

        if (!IsStruct && !IsArray)
          return error("INSERTVAL: Invalid type");
        if ((unsigned)Index != Index)
          return error("Invalid value");
        if (IsStruct && Index >= CurTy->subtypes().size())
          return error("INSERTVAL: Invalid struct index");
        if (IsArray && Index >= CurTy->getArrayNumElements())
          return error("INSERTVAL: Invalid array index");

        INSERTVALIdx.push_back((unsigned)Index);
        if (IsStruct)
          CurTy = CurTy->subtypes()[Index];
        else
          CurTy = CurTy->subtypes()[0];
      }

      if (CurTy != Val->getType())
        return error("Inserted value type doesn't match aggregate type");

      I = InsertValueInst::Create(Agg, Val, INSERTVALIdx);
      InstructionList.push_back(I);
      break;
    }

    case bitc::FUNC_CODE_INST_SELECT: { // SELECT: [opval, ty, opval, opval]
      // obsolete form of select
      // handles select i1 ... in old bitcode
      unsigned OpNum = 0;
      Value *TrueVal, *FalseVal, *Cond;
      if (getValueTypePair(Record, OpNum, NextValueNo, TrueVal) ||
          popValue(Record, OpNum, NextValueNo, TrueVal->getType(), FalseVal) ||
          popValue(Record, OpNum, NextValueNo, Type::getInt1Ty(Context), Cond))
        return error("Invalid record");

      I = SelectInst::Create(Cond, TrueVal, FalseVal);
      InstructionList.push_back(I);
      break;
    }

    case bitc::FUNC_CODE_INST_VSELECT: {// VSELECT: [ty,opval,opval,predty,pred]
      // new form of select
      // handles select i1 or select [N x i1]
      unsigned OpNum = 0;
      Value *TrueVal, *FalseVal, *Cond;
      if (getValueTypePair(Record, OpNum, NextValueNo, TrueVal) ||
          popValue(Record, OpNum, NextValueNo, TrueVal->getType(), FalseVal) ||
          getValueTypePair(Record, OpNum, NextValueNo, Cond))
        return error("Invalid record");

      // select condition can be either i1 or [N x i1]
      if (VectorType* vector_type =
          dyn_cast<VectorType>(Cond->getType())) {
        // expect <n x i1>
        if (vector_type->getElementType() != Type::getInt1Ty(Context))
          return error("Invalid type for value");
      } else {
        // expect i1
        if (Cond->getType() != Type::getInt1Ty(Context))
          return error("Invalid type for value");
      }

      I = SelectInst::Create(Cond, TrueVal, FalseVal);
      InstructionList.push_back(I);
      break;
    }

    case bitc::FUNC_CODE_INST_EXTRACTELT: { // EXTRACTELT: [opty, opval, opval]
      unsigned OpNum = 0;
      Value *Vec, *Idx;
      if (getValueTypePair(Record, OpNum, NextValueNo, Vec) ||
          getValueTypePair(Record, OpNum, NextValueNo, Idx))
        return error("Invalid record");
      if (!Vec->getType()->isVectorTy())
        return error("Invalid type for value");
      I = ExtractElementInst::Create(Vec, Idx);
      InstructionList.push_back(I);
      break;
    }

    case bitc::FUNC_CODE_INST_INSERTELT: { // INSERTELT: [ty, opval,opval,opval]
      unsigned OpNum = 0;
      Value *Vec, *Elt, *Idx;
      if (getValueTypePair(Record, OpNum, NextValueNo, Vec))
        return error("Invalid record");
      if (!Vec->getType()->isVectorTy())
        return error("Invalid type for value");
      if (popValue(Record, OpNum, NextValueNo,
                   cast<VectorType>(Vec->getType())->getElementType(), Elt) ||
          getValueTypePair(Record, OpNum, NextValueNo, Idx))
        return error("Invalid record");
      I = InsertElementInst::Create(Vec, Elt, Idx);
      InstructionList.push_back(I);
      break;
    }

    case bitc::FUNC_CODE_INST_SHUFFLEVEC: {// SHUFFLEVEC: [opval,ty,opval,opval]
      unsigned OpNum = 0;
      Value *Vec1, *Vec2, *Mask;
      if (getValueTypePair(Record, OpNum, NextValueNo, Vec1) ||
          popValue(Record, OpNum, NextValueNo, Vec1->getType(), Vec2))
        return error("Invalid record");

      if (getValueTypePair(Record, OpNum, NextValueNo, Mask))
        return error("Invalid record");
      if (!Vec1->getType()->isVectorTy() || !Vec2->getType()->isVectorTy())
        return error("Invalid type for value");
      I = new ShuffleVectorInst(Vec1, Vec2, Mask);
      InstructionList.push_back(I);
      break;
    }

    case bitc::FUNC_CODE_INST_CMP:   // CMP: [opty, opval, opval, pred]
      // Old form of ICmp/FCmp returning bool
      // Existed to differentiate between icmp/fcmp and vicmp/vfcmp which were
      // both legal on vectors but had different behaviour.
    case bitc::FUNC_CODE_INST_CMP2: { // CMP2: [opty, opval, opval, pred]
      // FCmp/ICmp returning bool or vector of bool

      unsigned OpNum = 0;
      Value *LHS, *RHS;
      if (getValueTypePair(Record, OpNum, NextValueNo, LHS) ||
          popValue(Record, OpNum, NextValueNo, LHS->getType(), RHS))
        return error("Invalid record");

      unsigned PredVal = Record[OpNum];
      bool IsFP = LHS->getType()->isFPOrFPVectorTy();
      FastMathFlags FMF;
      if (IsFP && Record.size() > OpNum+1)
        FMF = getDecodedFastMathFlags(Record[++OpNum]);

      if (OpNum+1 != Record.size())
        return error("Invalid record");

      if (LHS->getType()->isFPOrFPVectorTy())
        I = new FCmpInst((FCmpInst::Predicate)PredVal, LHS, RHS);
      else
        I = new ICmpInst((ICmpInst::Predicate)PredVal, LHS, RHS);

      if (FMF.any())
        I->setFastMathFlags(FMF);
      InstructionList.push_back(I);
      break;
    }

    case bitc::FUNC_CODE_INST_RET: // RET: [opty,opval<optional>]
      {
        unsigned Size = Record.size();
        if (Size == 0) {
          I = ReturnInst::Create(Context);
          InstructionList.push_back(I);
          break;
        }

        unsigned OpNum = 0;
        Value *Op = nullptr;
        if (getValueTypePair(Record, OpNum, NextValueNo, Op))
          return error("Invalid record");
        if (OpNum != Record.size())
          return error("Invalid record");

        I = ReturnInst::Create(Context, Op);
        InstructionList.push_back(I);
        break;
      }
    case bitc::FUNC_CODE_INST_BR: { // BR: [bb#, bb#, opval] or [bb#]
      if (Record.size() != 1 && Record.size() != 3)
        return error("Invalid record");
      BasicBlock *TrueDest = getBasicBlock(Record[0]);
      if (!TrueDest)
        return error("Invalid record");

      if (Record.size() == 1) {
        I = BranchInst::Create(TrueDest);
        InstructionList.push_back(I);
      }
      else {
        BasicBlock *FalseDest = getBasicBlock(Record[1]);
        Value *Cond = getValue(Record, 2, NextValueNo,
                               Type::getInt1Ty(Context));
        if (!FalseDest || !Cond)
          return error("Invalid record");
        I = BranchInst::Create(TrueDest, FalseDest, Cond);
        InstructionList.push_back(I);
      }
      break;
    }
    case bitc::FUNC_CODE_INST_SWITCH: { // SWITCH: [opty, op0, op1, ...]
      // Check magic
      if ((Record[0] >> 16) == SWITCH_INST_MAGIC) {
        // "New" SwitchInst format with case ranges. The changes to write this
        // format were reverted but we still recognize bitcode that uses it.
        // Hopefully someday we will have support for case ranges and can use
        // this format again.

        Type *OpTy = getTypeByID(Record[1]);
        unsigned ValueBitWidth = cast<IntegerType>(OpTy)->getBitWidth();

        Value *Cond = getValue(Record, 2, NextValueNo, OpTy);
        BasicBlock *Default = getBasicBlock(Record[3]);
        if (!OpTy || !Cond || !Default)
          return error("Invalid record");

        unsigned NumCases = Record[4];

        SwitchInst *SI = SwitchInst::Create(Cond, Default, NumCases);
        InstructionList.push_back(SI);

        unsigned CurIdx = 5;
        for (unsigned i = 0; i != NumCases; ++i) {
          SmallVector<ConstantInt*, 1> CaseVals;
          unsigned NumItems = Record[CurIdx++];
          for (unsigned ci = 0; ci != NumItems; ++ci) {
            bool isSingleNumber = Record[CurIdx++];

            APInt Low;
            unsigned ActiveWords = 1;
            if (ValueBitWidth > 64)
              ActiveWords = Record[CurIdx++];
            Low = readWideAPInt(makeArrayRef(&Record[CurIdx], ActiveWords),
                                ValueBitWidth);
            CurIdx += ActiveWords;

            if (!isSingleNumber) {
              ActiveWords = 1;
              if (ValueBitWidth > 64)
                ActiveWords = Record[CurIdx++];
              APInt High = readWideAPInt(
                  makeArrayRef(&Record[CurIdx], ActiveWords), ValueBitWidth);
              CurIdx += ActiveWords;

              // FIXME: It is not clear whether values in the range should be
              // compared as signed or unsigned values. The partially
              // implemented changes that used this format in the past used
              // unsigned comparisons.
              for ( ; Low.ule(High); ++Low)
                CaseVals.push_back(ConstantInt::get(Context, Low));
            } else
              CaseVals.push_back(ConstantInt::get(Context, Low));
          }
          BasicBlock *DestBB = getBasicBlock(Record[CurIdx++]);
          for (SmallVector<ConstantInt*, 1>::iterator cvi = CaseVals.begin(),
                 cve = CaseVals.end(); cvi != cve; ++cvi)
            SI->addCase(*cvi, DestBB);
        }
        I = SI;
        break;
      }

      // Old SwitchInst format without case ranges.

      if (Record.size() < 3 || (Record.size() & 1) == 0)
        return error("Invalid record");
      Type *OpTy = getTypeByID(Record[0]);
      Value *Cond = getValue(Record, 1, NextValueNo, OpTy);
      BasicBlock *Default = getBasicBlock(Record[2]);
      if (!OpTy || !Cond || !Default)
        return error("Invalid record");
      unsigned NumCases = (Record.size()-3)/2;
      SwitchInst *SI = SwitchInst::Create(Cond, Default, NumCases);
      InstructionList.push_back(SI);
      for (unsigned i = 0, e = NumCases; i != e; ++i) {
        ConstantInt *CaseVal =
          dyn_cast_or_null<ConstantInt>(getFnValueByID(Record[3+i*2], OpTy));
        BasicBlock *DestBB = getBasicBlock(Record[1+3+i*2]);
        if (!CaseVal || !DestBB) {
          delete SI;
          return error("Invalid record");
        }
        SI->addCase(CaseVal, DestBB);
      }
      I = SI;
      break;
    }
    case bitc::FUNC_CODE_INST_INDIRECTBR: { // INDIRECTBR: [opty, op0, op1, ...]
      if (Record.size() < 2)
        return error("Invalid record");
      Type *OpTy = getTypeByID(Record[0]);
      Value *Address = getValue(Record, 1, NextValueNo, OpTy);
      if (!OpTy || !Address)
        return error("Invalid record");
      unsigned NumDests = Record.size()-2;
      IndirectBrInst *IBI = IndirectBrInst::Create(Address, NumDests);
      InstructionList.push_back(IBI);
      for (unsigned i = 0, e = NumDests; i != e; ++i) {
        if (BasicBlock *DestBB = getBasicBlock(Record[2+i])) {
          IBI->addDestination(DestBB);
        } else {
          delete IBI;
          return error("Invalid record");
        }
      }
      I = IBI;
      break;
    }

    case bitc::FUNC_CODE_INST_INVOKE: {
      // INVOKE: [attrs, cc, normBB, unwindBB, fnty, op0,op1,op2, ...]
      if (Record.size() < 4)
        return error("Invalid record");
      unsigned OpNum = 0;
      AttributeSet PAL = getAttributes(Record[OpNum++]);
      unsigned CCInfo = Record[OpNum++];
      BasicBlock *NormalBB = getBasicBlock(Record[OpNum++]);
      BasicBlock *UnwindBB = getBasicBlock(Record[OpNum++]);

      FunctionType *FTy = nullptr;
      if (CCInfo >> 13 & 1 &&
          !(FTy = dyn_cast<FunctionType>(getTypeByID(Record[OpNum++]))))
        return error("Explicit invoke type is not a function type");

      Value *Callee;
      if (getValueTypePair(Record, OpNum, NextValueNo, Callee))
        return error("Invalid record");

      PointerType *CalleeTy = dyn_cast<PointerType>(Callee->getType());
      if (!CalleeTy)
        return error("Callee is not a pointer");
      if (!FTy) {
        FTy = dyn_cast<FunctionType>(CalleeTy->getElementType());
        if (!FTy)
          return error("Callee is not of pointer to function type");
      } else if (CalleeTy->getElementType() != FTy)
        return error("Explicit invoke type does not match pointee type of "
                     "callee operand");
      if (Record.size() < FTy->getNumParams() + OpNum)
        return error("Insufficient operands to call");

      SmallVector<Value*, 16> Ops;
      for (unsigned i = 0, e = FTy->getNumParams(); i != e; ++i, ++OpNum) {
        Ops.push_back(getValue(Record, OpNum, NextValueNo,
                               FTy->getParamType(i)));
        if (!Ops.back())
          return error("Invalid record");
      }

      if (!FTy->isVarArg()) {
        if (Record.size() != OpNum)
          return error("Invalid record");
      } else {
        // Read type/value pairs for varargs params.
        while (OpNum != Record.size()) {
          Value *Op;
          if (getValueTypePair(Record, OpNum, NextValueNo, Op))
            return error("Invalid record");
          Ops.push_back(Op);
        }
      }

      I = InvokeInst::Create(Callee, NormalBB, UnwindBB, Ops);
      InstructionList.push_back(I);
      cast<InvokeInst>(I)
          ->setCallingConv(static_cast<CallingConv::ID>(~(1U << 13) & CCInfo));
      cast<InvokeInst>(I)->setAttributes(PAL);
      break;
    }
    case bitc::FUNC_CODE_INST_RESUME: { // RESUME: [opval]
      unsigned Idx = 0;
      Value *Val = nullptr;
      if (getValueTypePair(Record, Idx, NextValueNo, Val))
        return error("Invalid record");
      I = ResumeInst::Create(Val);
      InstructionList.push_back(I);
      break;
    }
    case bitc::FUNC_CODE_INST_UNREACHABLE: // UNREACHABLE
      I = new UnreachableInst(Context);
      InstructionList.push_back(I);
      break;
    case bitc::FUNC_CODE_INST_PHI: { // PHI: [ty, val0,bb0, ...]
      if (Record.size() < 1 || ((Record.size()-1)&1))
        return error("Invalid record");
      Type *Ty = getTypeByID(Record[0]);
      if (!Ty)
        return error("Invalid record");

      PHINode *PN = PHINode::Create(Ty, (Record.size()-1)/2);
      InstructionList.push_back(PN);

      for (unsigned i = 0, e = Record.size()-1; i != e; i += 2) {
        Value *V;
        // With the new function encoding, it is possible that operands have
        // negative IDs (for forward references).  Use a signed VBR
        // representation to keep the encoding small.
        if (UseRelativeIDs)
          V = getValueSigned(Record, 1+i, NextValueNo, Ty);
        else
          V = getValue(Record, 1+i, NextValueNo, Ty);
        BasicBlock *BB = getBasicBlock(Record[2+i]);
        if (!V || !BB)
          return error("Invalid record");
        PN->addIncoming(V, BB);
      }
      I = PN;
      break;
    }

    case bitc::FUNC_CODE_INST_LANDINGPAD:
    case bitc::FUNC_CODE_INST_LANDINGPAD_OLD: {
      // LANDINGPAD: [ty, val, val, num, (id0,val0 ...)?]
      unsigned Idx = 0;
      if (BitCode == bitc::FUNC_CODE_INST_LANDINGPAD) {
        if (Record.size() < 3)
          return error("Invalid record");
      } else {
        assert(BitCode == bitc::FUNC_CODE_INST_LANDINGPAD_OLD);
        if (Record.size() < 4)
          return error("Invalid record");
      }
      Type *Ty = getTypeByID(Record[Idx++]);
      if (!Ty)
        return error("Invalid record");
      if (BitCode == bitc::FUNC_CODE_INST_LANDINGPAD_OLD) {
        Value *PersFn = nullptr;
        if (getValueTypePair(Record, Idx, NextValueNo, PersFn))
          return error("Invalid record");

        if (!F->hasPersonalityFn())
          F->setPersonalityFn(cast<Constant>(PersFn));
        else if (F->getPersonalityFn() != cast<Constant>(PersFn))
          return error("Personality function mismatch");
      }

      bool IsCleanup = !!Record[Idx++];
      unsigned NumClauses = Record[Idx++];
      LandingPadInst *LP = LandingPadInst::Create(Ty, NumClauses);
      LP->setCleanup(IsCleanup);
      for (unsigned J = 0; J != NumClauses; ++J) {
        LandingPadInst::ClauseType CT =
          LandingPadInst::ClauseType(Record[Idx++]); (void)CT;
        Value *Val;

        if (getValueTypePair(Record, Idx, NextValueNo, Val)) {
          delete LP;
          return error("Invalid record");
        }

        assert((CT != LandingPadInst::Catch ||
                !isa<ArrayType>(Val->getType())) &&
               "Catch clause has a invalid type!");
        assert((CT != LandingPadInst::Filter ||
                isa<ArrayType>(Val->getType())) &&
               "Filter clause has invalid type!");
        LP->addClause(cast<Constant>(Val));
      }

      I = LP;
      InstructionList.push_back(I);
      break;
    }

    case bitc::FUNC_CODE_INST_ALLOCA: { // ALLOCA: [instty, opty, op, align]
      if (Record.size() != 4)
        return error("Invalid record");
      uint64_t AlignRecord = Record[3];
      const uint64_t InAllocaMask = uint64_t(1) << 5;
      const uint64_t ExplicitTypeMask = uint64_t(1) << 6;
      const uint64_t FlagMask = InAllocaMask | ExplicitTypeMask;
      bool InAlloca = AlignRecord & InAllocaMask;
      Type *Ty = getTypeByID(Record[0]);
      if ((AlignRecord & ExplicitTypeMask) == 0) {
        auto *PTy = dyn_cast_or_null<PointerType>(Ty);
        if (!PTy)
          return error("Old-style alloca with a non-pointer type");
        Ty = PTy->getElementType();
      }
      Type *OpTy = getTypeByID(Record[1]);
      Value *Size = getFnValueByID(Record[2], OpTy);
      unsigned Align;
      if (std::error_code EC =
              parseAlignmentValue(AlignRecord & ~FlagMask, Align)) {
        return EC;
      }
      if (!Ty || !Size)
        return error("Invalid record");
      AllocaInst *AI = new AllocaInst(Ty, Size, Align);
      AI->setUsedWithInAlloca(InAlloca);
      I = AI;
      InstructionList.push_back(I);
      break;
    }
    case bitc::FUNC_CODE_INST_LOAD: { // LOAD: [opty, op, align, vol]
      unsigned OpNum = 0;
      Value *Op;
      if (getValueTypePair(Record, OpNum, NextValueNo, Op) ||
          (OpNum + 2 != Record.size() && OpNum + 3 != Record.size()))
        return error("Invalid record");

      Type *Ty = nullptr;
      if (OpNum + 3 == Record.size())
        Ty = getTypeByID(Record[OpNum++]);
      if (std::error_code EC =
              typeCheckLoadStoreInst(DiagnosticHandler, Ty, Op->getType()))
        return EC;
      if (!Ty)
        Ty = cast<PointerType>(Op->getType())->getElementType();

      unsigned Align;
      if (std::error_code EC = parseAlignmentValue(Record[OpNum], Align))
        return EC;
      I = new LoadInst(Ty, Op, "", Record[OpNum + 1], Align);

      InstructionList.push_back(I);
      break;
    }
    case bitc::FUNC_CODE_INST_LOADATOMIC: {
       // LOADATOMIC: [opty, op, align, vol, ordering, synchscope]
      unsigned OpNum = 0;
      Value *Op;
      if (getValueTypePair(Record, OpNum, NextValueNo, Op) ||
          (OpNum + 4 != Record.size() && OpNum + 5 != Record.size()))
        return error("Invalid record");

      Type *Ty = nullptr;
      if (OpNum + 5 == Record.size())
        Ty = getTypeByID(Record[OpNum++]);
      if (std::error_code EC =
              typeCheckLoadStoreInst(DiagnosticHandler, Ty, Op->getType()))
        return EC;
      if (!Ty)
        Ty = cast<PointerType>(Op->getType())->getElementType();

      AtomicOrdering Ordering = getDecodedOrdering(Record[OpNum + 2]);
      if (Ordering == NotAtomic || Ordering == Release ||
          Ordering == AcquireRelease)
        return error("Invalid record");
      if (Ordering != NotAtomic && Record[OpNum] == 0)
        return error("Invalid record");
      SynchronizationScope SynchScope = getDecodedSynchScope(Record[OpNum + 3]);

      unsigned Align;
      if (std::error_code EC = parseAlignmentValue(Record[OpNum], Align))
        return EC;
      I = new LoadInst(Op, "", Record[OpNum+1], Align, Ordering, SynchScope);

      InstructionList.push_back(I);
      break;
    }
    case bitc::FUNC_CODE_INST_STORE:
    case bitc::FUNC_CODE_INST_STORE_OLD: { // STORE2:[ptrty, ptr, val, align, vol]
      unsigned OpNum = 0;
      Value *Val, *Ptr;
      if (getValueTypePair(Record, OpNum, NextValueNo, Ptr) ||
          (BitCode == bitc::FUNC_CODE_INST_STORE
               ? getValueTypePair(Record, OpNum, NextValueNo, Val)
               : popValue(Record, OpNum, NextValueNo,
                          cast<PointerType>(Ptr->getType())->getElementType(),
                          Val)) ||
          OpNum + 2 != Record.size())
        return error("Invalid record");

      if (std::error_code EC = typeCheckLoadStoreInst(
              DiagnosticHandler, Val->getType(), Ptr->getType()))
        return EC;
      unsigned Align;
      if (std::error_code EC = parseAlignmentValue(Record[OpNum], Align))
        return EC;
      I = new StoreInst(Val, Ptr, Record[OpNum+1], Align);
      InstructionList.push_back(I);
      break;
    }
    case bitc::FUNC_CODE_INST_STOREATOMIC:
    case bitc::FUNC_CODE_INST_STOREATOMIC_OLD: {
      // STOREATOMIC: [ptrty, ptr, val, align, vol, ordering, synchscope]
      unsigned OpNum = 0;
      Value *Val, *Ptr;
      if (getValueTypePair(Record, OpNum, NextValueNo, Ptr) ||
          (BitCode == bitc::FUNC_CODE_INST_STOREATOMIC
               ? getValueTypePair(Record, OpNum, NextValueNo, Val)
               : popValue(Record, OpNum, NextValueNo,
                          cast<PointerType>(Ptr->getType())->getElementType(),
                          Val)) ||
          OpNum + 4 != Record.size())
        return error("Invalid record");

      if (std::error_code EC = typeCheckLoadStoreInst(
              DiagnosticHandler, Val->getType(), Ptr->getType()))
        return EC;
      AtomicOrdering Ordering = getDecodedOrdering(Record[OpNum + 2]);
      if (Ordering == NotAtomic || Ordering == Acquire ||
          Ordering == AcquireRelease)
        return error("Invalid record");
      SynchronizationScope SynchScope = getDecodedSynchScope(Record[OpNum + 3]);
      if (Ordering != NotAtomic && Record[OpNum] == 0)
        return error("Invalid record");

      unsigned Align;
      if (std::error_code EC = parseAlignmentValue(Record[OpNum], Align))
        return EC;
      I = new StoreInst(Val, Ptr, Record[OpNum+1], Align, Ordering, SynchScope);
      InstructionList.push_back(I);
      break;
    }
    case bitc::FUNC_CODE_INST_CMPXCHG_OLD:
    case bitc::FUNC_CODE_INST_CMPXCHG: {
      // CMPXCHG:[ptrty, ptr, cmp, new, vol, successordering, synchscope,
      //          failureordering?, isweak?]
      unsigned OpNum = 0;
      Value *Ptr, *Cmp, *New;
      if (getValueTypePair(Record, OpNum, NextValueNo, Ptr) ||
          (BitCode == bitc::FUNC_CODE_INST_CMPXCHG
               ? getValueTypePair(Record, OpNum, NextValueNo, Cmp)
               : popValue(Record, OpNum, NextValueNo,
                          cast<PointerType>(Ptr->getType())->getElementType(),
                          Cmp)) ||
          popValue(Record, OpNum, NextValueNo, Cmp->getType(), New) ||
          Record.size() < OpNum + 3 || Record.size() > OpNum + 5)
        return error("Invalid record");
      AtomicOrdering SuccessOrdering = getDecodedOrdering(Record[OpNum + 1]);
      if (SuccessOrdering == NotAtomic || SuccessOrdering == Unordered)
        return error("Invalid record");
      SynchronizationScope SynchScope = getDecodedSynchScope(Record[OpNum + 2]);

      if (std::error_code EC = typeCheckLoadStoreInst(
              DiagnosticHandler, Cmp->getType(), Ptr->getType()))
        return EC;
      AtomicOrdering FailureOrdering;
      if (Record.size() < 7)
        FailureOrdering =
            AtomicCmpXchgInst::getStrongestFailureOrdering(SuccessOrdering);
      else
        FailureOrdering = getDecodedOrdering(Record[OpNum + 3]);

      I = new AtomicCmpXchgInst(Ptr, Cmp, New, SuccessOrdering, FailureOrdering,
                                SynchScope);
      cast<AtomicCmpXchgInst>(I)->setVolatile(Record[OpNum]);

      if (Record.size() < 8) {
        // Before weak cmpxchgs existed, the instruction simply returned the
        // value loaded from memory, so bitcode files from that era will be
        // expecting the first component of a modern cmpxchg.
        CurBB->getInstList().push_back(I);
        I = ExtractValueInst::Create(I, 0);
      } else {
        cast<AtomicCmpXchgInst>(I)->setWeak(Record[OpNum+4]);
      }

      InstructionList.push_back(I);
      break;
    }
    case bitc::FUNC_CODE_INST_ATOMICRMW: {
      // ATOMICRMW:[ptrty, ptr, val, op, vol, ordering, synchscope]
      unsigned OpNum = 0;
      Value *Ptr, *Val;
      if (getValueTypePair(Record, OpNum, NextValueNo, Ptr) ||
          popValue(Record, OpNum, NextValueNo,
                    cast<PointerType>(Ptr->getType())->getElementType(), Val) ||
          OpNum+4 != Record.size())
        return error("Invalid record");
      AtomicRMWInst::BinOp Operation = getDecodedRMWOperation(Record[OpNum]);
      if (Operation < AtomicRMWInst::FIRST_BINOP ||
          Operation > AtomicRMWInst::LAST_BINOP)
        return error("Invalid record");
      AtomicOrdering Ordering = getDecodedOrdering(Record[OpNum + 2]);
      if (Ordering == NotAtomic || Ordering == Unordered)
        return error("Invalid record");
      SynchronizationScope SynchScope = getDecodedSynchScope(Record[OpNum + 3]);
      I = new AtomicRMWInst(Operation, Ptr, Val, Ordering, SynchScope);
      cast<AtomicRMWInst>(I)->setVolatile(Record[OpNum+1]);
      InstructionList.push_back(I);
      break;
    }
    case bitc::FUNC_CODE_INST_FENCE: { // FENCE:[ordering, synchscope]
      if (2 != Record.size())
        return error("Invalid record");
      AtomicOrdering Ordering = getDecodedOrdering(Record[0]);
      if (Ordering == NotAtomic || Ordering == Unordered ||
          Ordering == Monotonic)
        return error("Invalid record");
      SynchronizationScope SynchScope = getDecodedSynchScope(Record[1]);
      I = new FenceInst(Context, Ordering, SynchScope);
      InstructionList.push_back(I);
      break;
    }
    case bitc::FUNC_CODE_INST_CALL: {
      // CALL: [paramattrs, cc, fnty, fnid, arg0, arg1...]
      if (Record.size() < 3)
        return error("Invalid record");

      unsigned OpNum = 0;
      AttributeSet PAL = getAttributes(Record[OpNum++]);
      unsigned CCInfo = Record[OpNum++];

      FunctionType *FTy = nullptr;
      if (CCInfo >> 15 & 1 &&
          !(FTy = dyn_cast<FunctionType>(getTypeByID(Record[OpNum++]))))
        return error("Explicit call type is not a function type");

      Value *Callee;
      if (getValueTypePair(Record, OpNum, NextValueNo, Callee))
        return error("Invalid record");

      PointerType *OpTy = dyn_cast<PointerType>(Callee->getType());
      if (!OpTy)
        return error("Callee is not a pointer type");
      if (!FTy) {
        FTy = dyn_cast<FunctionType>(OpTy->getElementType());
        if (!FTy)
          return error("Callee is not of pointer to function type");
      } else if (OpTy->getElementType() != FTy)
        return error("Explicit call type does not match pointee type of "
                     "callee operand");
      if (Record.size() < FTy->getNumParams() + OpNum)
        return error("Insufficient operands to call");

      SmallVector<Value*, 16> Args;
      // Read the fixed params.
      for (unsigned i = 0, e = FTy->getNumParams(); i != e; ++i, ++OpNum) {
        if (FTy->getParamType(i)->isLabelTy())
          Args.push_back(getBasicBlock(Record[OpNum]));
        else
          Args.push_back(getValue(Record, OpNum, NextValueNo,
                                  FTy->getParamType(i)));
        if (!Args.back())
          return error("Invalid record");
      }

      // Read type/value pairs for varargs params.
      if (!FTy->isVarArg()) {
        if (OpNum != Record.size())
          return error("Invalid record");
      } else {
        while (OpNum != Record.size()) {
          Value *Op;
          if (getValueTypePair(Record, OpNum, NextValueNo, Op))
            return error("Invalid record");
          Args.push_back(Op);
        }
      }

      I = CallInst::Create(FTy, Callee, Args);
      InstructionList.push_back(I);
      cast<CallInst>(I)->setCallingConv(
          static_cast<CallingConv::ID>((~(1U << 14) & CCInfo) >> 1));
      CallInst::TailCallKind TCK = CallInst::TCK_None;
      if (CCInfo & 1)
        TCK = CallInst::TCK_Tail;
      if (CCInfo & (1 << 14))
        TCK = CallInst::TCK_MustTail;
      cast<CallInst>(I)->setTailCallKind(TCK);
      cast<CallInst>(I)->setAttributes(PAL);
      break;
    }
    case bitc::FUNC_CODE_INST_VAARG: { // VAARG: [valistty, valist, instty]
      if (Record.size() < 3)
        return error("Invalid record");
      Type *OpTy = getTypeByID(Record[0]);
      Value *Op = getValue(Record, 1, NextValueNo, OpTy);
      Type *ResTy = getTypeByID(Record[2]);
      if (!OpTy || !Op || !ResTy)
        return error("Invalid record");
      I = new VAArgInst(Op, ResTy);
      InstructionList.push_back(I);
      break;
    }
    }

    // Add instruction to end of current BB.  If there is no current BB, reject
    // this file.
    if (!CurBB) {
      delete I;
      return error("Invalid instruction with no BB");
    }
    CurBB->getInstList().push_back(I);

    // If this was a terminator instruction, move to the next block.
    if (isa<TerminatorInst>(I)) {
      ++CurBBNo;
      CurBB = CurBBNo < FunctionBBs.size() ? FunctionBBs[CurBBNo] : nullptr;
    }

    // Non-void values get registered in the value table for future use.
    if (I && !I->getType()->isVoidTy())
      ValueList.assignValue(I, NextValueNo++);
  }

OutOfRecordLoop:

  // Check the function list for unresolved values.
  if (Argument *A = dyn_cast<Argument>(ValueList.back())) {
    if (!A->getParent()) {
      // We found at least one unresolved value.  Nuke them all to avoid leaks.
      for (unsigned i = ModuleValueListSize, e = ValueList.size(); i != e; ++i){
        if ((A = dyn_cast_or_null<Argument>(ValueList[i])) && !A->getParent()) {
          A->replaceAllUsesWith(UndefValue::get(A->getType()));
          delete A;
        }
      }
      return error("Never resolved value found in function");
    }
  }

  // FIXME: Check for unresolved forward-declared metadata references
  // and clean up leaks.

  // Trim the value list down to the size it was before we parsed this function.
  ValueList.shrinkTo(ModuleValueListSize);
  MDValueList.shrinkTo(ModuleMDValueListSize);
  std::vector<BasicBlock*>().swap(FunctionBBs);
  return std::error_code();
}

/// Find the function body in the bitcode stream
std::error_code BitcodeReader::findFunctionInStream(
    Function *F,
    DenseMap<Function *, uint64_t>::iterator DeferredFunctionInfoIterator) {
  while (DeferredFunctionInfoIterator->second == 0) {
    if (Stream.AtEndOfStream())
      return error("Could not find function in stream");
    // ParseModule will parse the next body in the stream and set its
    // position in the DeferredFunctionInfo map.
    if (std::error_code EC = parseModule(true))
      return EC;
  }
  return std::error_code();
}

//===----------------------------------------------------------------------===//
// GVMaterializer implementation
//===----------------------------------------------------------------------===//

void BitcodeReader::releaseBuffer() { Buffer.release(); }

std::error_code BitcodeReader::materialize(GlobalValue *GV) {
  if (std::error_code EC = materializeMetadata())
    return EC;

  Function *F = dyn_cast<Function>(GV);
  // If it's not a function or is already material, ignore the request.
  if (!F || !F->isMaterializable())
    return std::error_code();

  DenseMap<Function*, uint64_t>::iterator DFII = DeferredFunctionInfo.find(F);
  assert(DFII != DeferredFunctionInfo.end() && "Deferred function not found!");
  // If its position is recorded as 0, its body is somewhere in the stream
  // but we haven't seen it yet.
  if (DFII->second == 0)
    if (std::error_code EC = findFunctionInStream(F, DFII))
      return EC;

  // Move the bit stream to the saved position of the deferred function body.
  Stream.JumpToBit(DFII->second);

  if (std::error_code EC = parseFunctionBody(F))
    return EC;
  F->setIsMaterializable(false);

  if (StripDebugInfo)
    stripDebugInfo(*F);

  // Upgrade any old intrinsic calls in the function.
  for (auto &I : UpgradedIntrinsics) {
    for (auto UI = I.first->user_begin(), UE = I.first->user_end(); UI != UE;) {
      User *U = *UI;
      ++UI;
      if (CallInst *CI = dyn_cast<CallInst>(U))
        UpgradeIntrinsicCall(CI, I.second);
    }
  }

  // Bring in any functions that this function forward-referenced via
  // blockaddresses.
  return materializeForwardReferencedFunctions();
}

bool BitcodeReader::isDematerializable(const GlobalValue *GV) const {
  const Function *F = dyn_cast<Function>(GV);
  if (!F || F->isDeclaration())
    return false;

  // Dematerializing F would leave dangling references that wouldn't be
  // reconnected on re-materialization.
  if (BlockAddressesTaken.count(F))
    return false;

  return DeferredFunctionInfo.count(const_cast<Function*>(F));
}

void BitcodeReader::dematerialize(GlobalValue *GV) {
  Function *F = dyn_cast<Function>(GV);
  // If this function isn't dematerializable, this is a noop.
  if (!F || !isDematerializable(F))
    return;

  assert(DeferredFunctionInfo.count(F) && "No info to read function later?");

  // Just forget the function body, we can remat it later.
  F->dropAllReferences();
  F->setIsMaterializable(true);
}

std::error_code BitcodeReader::materializeModule(Module *M) {
  assert(M == TheModule &&
         "Can only Materialize the Module this BitcodeReader is attached to.");

  if (std::error_code EC = materializeMetadata())
    return EC;

  // Promise to materialize all forward references.
  WillMaterializeAllForwardRefs = true;

  // Iterate over the module, deserializing any functions that are still on
  // disk.
  for (Module::iterator F = TheModule->begin(), E = TheModule->end();
       F != E; ++F) {
    if (std::error_code EC = materialize(F))
      return EC;
  }
  // At this point, if there are any function bodies, the current bit is
  // pointing to the END_BLOCK record after them. Now make sure the rest
  // of the bits in the module have been read.
  if (NextUnreadBit)
    parseModule(true);

  // Check that all block address forward references got resolved (as we
  // promised above).
  if (!BasicBlockFwdRefs.empty())
    return error("Never resolved function from blockaddress");

  // Upgrade any intrinsic calls that slipped through (should not happen!) and
  // delete the old functions to clean up. We can't do this unless the entire
  // module is materialized because there could always be another function body
  // with calls to the old function.
  for (auto &I : UpgradedIntrinsics) {
    for (auto *U : I.first->users()) {
      if (CallInst *CI = dyn_cast<CallInst>(U))
        UpgradeIntrinsicCall(CI, I.second);
    }
    if (!I.first->use_empty())
      I.first->replaceAllUsesWith(I.second);
    I.first->eraseFromParent();
  }
  UpgradedIntrinsics.clear();

  for (unsigned I = 0, E = InstsWithTBAATag.size(); I < E; I++)
    UpgradeInstWithTBAATag(InstsWithTBAATag[I]);

  UpgradeDebugInfo(*M);

  // HLSL Change Starts
  if (ShouldTrackBitstreamUsage && !Tracker.isDense((uint64_t)(Buffer->getBufferSize()) * 8)) {
    ReportWarning(DiagnosticHandler, "Unused bits in buffer.");
  }
  // HLSL Change Ends

  return std::error_code();
}

std::vector<StructType *> BitcodeReader::getIdentifiedStructTypes() const {
  return IdentifiedStructTypes;
}

std::error_code
BitcodeReader::initStream(std::unique_ptr<DataStreamer> Streamer) {
  if (Streamer)
    return initLazyStream(std::move(Streamer));
  return initStreamFromBuffer();
}

std::error_code BitcodeReader::initStreamFromBuffer() {
  const unsigned char *BufPtr = (const unsigned char*)Buffer->getBufferStart();
  const unsigned char *BufEnd = BufPtr+Buffer->getBufferSize();

  if (Buffer->getBufferSize() & 3)
    return error("Invalid bitcode size"); // HLSL Change - bitcode size is the problem, not the signature per se

  // If we have a wrapper header, parse it and ignore the non-bc file contents.
  // The magic number is 0x0B17C0DE stored in little endian.
  if (isBitcodeWrapper(BufPtr, BufEnd))
    if (SkipBitcodeWrapperHeader(BufPtr, BufEnd, true))
      return error("Invalid bitcode wrapper header");

  StreamFile.reset(new BitstreamReader(BufPtr, BufEnd));
  if (ShouldTrackBitstreamUsage) StreamFile->Tracker = &Tracker; // HLSL Change
  Stream.init(&*StreamFile);

  return std::error_code();
}

std::error_code
BitcodeReader::initLazyStream(std::unique_ptr<DataStreamer> Streamer) {
  // Check and strip off the bitcode wrapper; BitstreamReader expects never to
  // see it.
  auto OwnedBytes =
      llvm::make_unique<StreamingMemoryObject>(std::move(Streamer));
  StreamingMemoryObject &Bytes = *OwnedBytes;
  StreamFile = llvm::make_unique<BitstreamReader>(std::move(OwnedBytes));
  Stream.init(&*StreamFile);

  unsigned char buf[16];
  if (Bytes.readBytes(buf, 16, 0) != 16)
    return error("Invalid bitcode signature");

  if (!isBitcode(buf, buf + 16))
    return error("Invalid bitcode signature");

  if (isBitcodeWrapper(buf, buf + 4)) {
    const unsigned char *bitcodeStart = buf;
    const unsigned char *bitcodeEnd = buf + 16;
    SkipBitcodeWrapperHeader(bitcodeStart, bitcodeEnd, false);
    Bytes.dropLeadingBytes(bitcodeStart - buf);
    Bytes.setKnownObjectSize(bitcodeEnd - bitcodeStart);
  }
  return std::error_code();
}

namespace {
class BitcodeErrorCategoryType : public std::error_category {
  const char *name() const LLVM_NOEXCEPT override {
    return "llvm.bitcode";
  }
  std::string message(int IE) const override {
    BitcodeError E = static_cast<BitcodeError>(IE);
    switch (E) {
    case BitcodeError::InvalidBitcodeSignature:
      return "Invalid bitcode signature";
    case BitcodeError::CorruptedBitcode:
      return "Corrupted bitcode";
    }
    llvm_unreachable("Unknown error type!");
  }
};
}

static BitcodeErrorCategoryType g_ErrorCategory; // HLSL Change - not a ManagedStatic

const std::error_category &llvm::BitcodeErrorCategory() {
  return g_ErrorCategory; // HLSL Change - simple global
}

//===----------------------------------------------------------------------===//
// External interface
//===----------------------------------------------------------------------===//

static ErrorOr<std::unique_ptr<Module>>
getBitcodeModuleImpl(std::unique_ptr<DataStreamer> Streamer, StringRef Name,
                     std::unique_ptr<BitcodeReader> RPtr, LLVMContext &Context, // HLSL Change: unique_ptr
                     bool MaterializeAll, bool ShouldLazyLoadMetadata) {
  std::unique_ptr<Module> M = make_unique<Module>(Name, Context);
  // HLSL Change Begin: Transfer ownership of R to M, but keep a raw pointer
  BitcodeReader* R = RPtr.release();
  M->setMaterializer(R);
  // HLSL Change End

  // Delay parsing Metadata if ShouldLazyLoadMetadata is true.
  if (std::error_code EC = R->parseBitcodeInto(std::move(Streamer), M.get(),
                                               ShouldLazyLoadMetadata))
    return EC; // HLSL Change: Correct memory management of BitcodeReader.buffer

  if (MaterializeAll) {
    // Read in the entire module, and destroy the BitcodeReader.
    if (std::error_code EC = M->materializeAllPermanently())
      return EC; // HLSL Change: Correct memory management of BitcodeReader.buffer
  } else {
    // Resolve forward references from blockaddresses.
    if (std::error_code EC = R->materializeForwardReferencedFunctions())
      return EC; // HLSL Change: Correct memory management of BitcodeReader.buffer
  }
  return std::move(M);
}

/// \brief Get a lazy one-at-time loading module from bitcode.
///
/// This isn't always used in a lazy context.  In particular, it's also used by
/// \a parseBitcodeFile().  If this is truly lazy, then we need to eagerly pull
/// in forward-referenced functions from block address references.
///
/// \param[in] MaterializeAll Set to \c true if we should materialize
/// everything.
static ErrorOr<std::unique_ptr<Module>>
getLazyBitcodeModuleImpl(std::unique_ptr<MemoryBuffer> &&Buffer,
                         LLVMContext &Context, bool MaterializeAll,
                         DiagnosticHandlerFunction DiagnosticHandler,
                         bool ShouldLazyLoadMetadata = false,
                         bool ShouldTrackBitstreamUsage = false) // HLSL Change
{
  // HLSL Change Begin: Proper memory management with unique_ptr
  // Get the buffer identifier before we transfer the ownership to the bitcode reader,
  // this is ugly but safe as long as it keeps the buffer, and hence identifier string, alive.
  const char* BufferIdentifier = Buffer->getBufferIdentifier();
  std::unique_ptr<BitcodeReader> R = llvm::make_unique<BitcodeReader>(
    std::move(Buffer), Context, DiagnosticHandler);

  if (R) R->ShouldTrackBitstreamUsage = ShouldTrackBitstreamUsage; // HLSL Change
  ErrorOr<std::unique_ptr<Module>> Ret =
      getBitcodeModuleImpl(nullptr, BufferIdentifier, std::move(R), Context,
                           MaterializeAll, ShouldLazyLoadMetadata);
  // HLSL Change End
  if (!Ret)
    return Ret;

  return Ret;
}

ErrorOr<std::unique_ptr<Module>> llvm::getLazyBitcodeModule(
    std::unique_ptr<MemoryBuffer> &&Buffer, LLVMContext &Context,
    DiagnosticHandlerFunction DiagnosticHandler, bool ShouldLazyLoadMetadata,
    bool ShouldTrackBitstreamUsage) {
  return getLazyBitcodeModuleImpl(std::move(Buffer), Context, false,
                                  DiagnosticHandler, ShouldLazyLoadMetadata,
                                  ShouldTrackBitstreamUsage); // HLSL Change
}

ErrorOr<std::unique_ptr<Module>> llvm::getStreamedBitcodeModule(
    StringRef Name, std::unique_ptr<DataStreamer> Streamer,
    LLVMContext &Context, DiagnosticHandlerFunction DiagnosticHandler) {
  std::unique_ptr<Module> M = make_unique<Module>(Name, Context);
  std::unique_ptr<BitcodeReader> R = llvm::make_unique<BitcodeReader>(Context, DiagnosticHandler); // HLSL Change: unique_ptr

  return getBitcodeModuleImpl(std::move(Streamer), Name, std::move(R), Context, false, // HLSL Change: std::move
                              false);
}

// HLSL Change Starts
struct report_fatal_error_data {
  report_fatal_error_data(DiagnosticHandlerFunction DH)
      : DiagnosticHandler(DH) {}
  DiagnosticHandlerFunction DiagnosticHandler;
};
void report_fatal_error_handler(void *user_datam, const std::string &reason,
                                bool gen_crash_diag) {
  report_fatal_error_data *data = (report_fatal_error_data *)user_datam;
  BitcodeDiagnosticInfo BDI(std::error_code(EINVAL, std::system_category()),
                            DiagnosticSeverity::DS_Error, reason);
  data->DiagnosticHandler(BDI);
  throw std::runtime_error("Invalid bitcode");
}
// HLSL Change Ends

ErrorOr<std::unique_ptr<Module>>
llvm::parseBitcodeFile(MemoryBufferRef Buffer, LLVMContext &Context,
                       DiagnosticHandlerFunction DiagnosticHandler,
                       bool ShouldTrackBitstreamUsage) // HLSL Change
{
  // HLSL Change Starts - introduce a ScopedFatalErrorHandler to handle
  // report_fatal_error from readers.
  report_fatal_error_data data(DiagnosticHandler);
  ScopedFatalErrorHandler SFE(report_fatal_error_handler, &data);
  // HLSL Change Ends
  std::unique_ptr<MemoryBuffer> Buf = MemoryBuffer::getMemBuffer(Buffer, false);
  return getLazyBitcodeModuleImpl(std::move(Buf), Context, true,
                                  DiagnosticHandler,
                                  false, ShouldTrackBitstreamUsage); // HLSL Change
  // TODO: Restore the use-lists to the in-memory state when the bitcode was
  // written.  We must defer until the Module has been fully materialized.
}

std::string
llvm::getBitcodeTargetTriple(MemoryBufferRef Buffer, LLVMContext &Context,
                             DiagnosticHandlerFunction DiagnosticHandler) {
  std::unique_ptr<MemoryBuffer> Buf = MemoryBuffer::getMemBuffer(Buffer, false);
  auto R = llvm::make_unique<BitcodeReader>(std::move(Buf), Context, // HLSL Change: std::move
                                            DiagnosticHandler);
  ErrorOr<std::string> Triple = R->parseTriple();
  if (Triple.getError())
    return "";
  return Triple.get();
}
