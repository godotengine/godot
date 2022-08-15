//===-- MemorySanitizer.cpp - detector of uninitialized reads -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
/// \file
/// This file is a part of MemorySanitizer, a detector of uninitialized
/// reads.
///
/// The algorithm of the tool is similar to Memcheck
/// (http://goo.gl/QKbem). We associate a few shadow bits with every
/// byte of the application memory, poison the shadow of the malloc-ed
/// or alloca-ed memory, load the shadow bits on every memory read,
/// propagate the shadow bits through some of the arithmetic
/// instruction (including MOV), store the shadow bits on every memory
/// write, report a bug on some other instructions (e.g. JMP) if the
/// associated shadow is poisoned.
///
/// But there are differences too. The first and the major one:
/// compiler instrumentation instead of binary instrumentation. This
/// gives us much better register allocation, possible compiler
/// optimizations and a fast start-up. But this brings the major issue
/// as well: msan needs to see all program events, including system
/// calls and reads/writes in system libraries, so we either need to
/// compile *everything* with msan or use a binary translation
/// component (e.g. DynamoRIO) to instrument pre-built libraries.
/// Another difference from Memcheck is that we use 8 shadow bits per
/// byte of application memory and use a direct shadow mapping. This
/// greatly simplifies the instrumentation code and avoids races on
/// shadow updates (Memcheck is single-threaded so races are not a
/// concern there. Memcheck uses 2 shadow bits per byte with a slow
/// path storage that uses 8 bits per byte).
///
/// The default value of shadow is 0, which means "clean" (not poisoned).
///
/// Every module initializer should call __msan_init to ensure that the
/// shadow memory is ready. On error, __msan_warning is called. Since
/// parameters and return values may be passed via registers, we have a
/// specialized thread-local shadow for return values
/// (__msan_retval_tls) and parameters (__msan_param_tls).
///
///                           Origin tracking.
///
/// MemorySanitizer can track origins (allocation points) of all uninitialized
/// values. This behavior is controlled with a flag (msan-track-origins) and is
/// disabled by default.
///
/// Origins are 4-byte values created and interpreted by the runtime library.
/// They are stored in a second shadow mapping, one 4-byte value for 4 bytes
/// of application memory. Propagation of origins is basically a bunch of
/// "select" instructions that pick the origin of a dirty argument, if an
/// instruction has one.
///
/// Every 4 aligned, consecutive bytes of application memory have one origin
/// value associated with them. If these bytes contain uninitialized data
/// coming from 2 different allocations, the last store wins. Because of this,
/// MemorySanitizer reports can show unrelated origins, but this is unlikely in
/// practice.
///
/// Origins are meaningless for fully initialized values, so MemorySanitizer
/// avoids storing origin to memory when a fully initialized value is stored.
/// This way it avoids needless overwritting origin of the 4-byte region on
/// a short (i.e. 1 byte) clean store, and it is also good for performance.
///
///                            Atomic handling.
///
/// Ideally, every atomic store of application value should update the
/// corresponding shadow location in an atomic way. Unfortunately, atomic store
/// of two disjoint locations can not be done without severe slowdown.
///
/// Therefore, we implement an approximation that may err on the safe side.
/// In this implementation, every atomically accessed location in the program
/// may only change from (partially) uninitialized to fully initialized, but
/// not the other way around. We load the shadow _after_ the application load,
/// and we store the shadow _before_ the app store. Also, we always store clean
/// shadow (if the application store is atomic). This way, if the store-load
/// pair constitutes a happens-before arc, shadow store and load are correctly
/// ordered such that the load will get either the value that was stored, or
/// some later value (which is always clean).
///
/// This does not work very well with Compare-And-Swap (CAS) and
/// Read-Modify-Write (RMW) operations. To follow the above logic, CAS and RMW
/// must store the new shadow before the app operation, and load the shadow
/// after the app operation. Computers don't work this way. Current
/// implementation ignores the load aspect of CAS/RMW, always returning a clean
/// value. It implements the store part as a simple atomic store by storing a
/// clean shadow.

#include "llvm/Transforms/Instrumentation.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Triple.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/ValueMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Local.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "msan"

static const unsigned kOriginSize = 4;
static const unsigned kMinOriginAlignment = 4;
static const unsigned kShadowTLSAlignment = 8;

// These constants must be kept in sync with the ones in msan.h.
static const unsigned kParamTLSSize = 800;
static const unsigned kRetvalTLSSize = 800;

// Accesses sizes are powers of two: 1, 2, 4, 8.
static const size_t kNumberOfAccessSizes = 4;

/// \brief Track origins of uninitialized values.
///
/// Adds a section to MemorySanitizer report that points to the allocation
/// (stack or heap) the uninitialized bits came from originally.
static cl::opt<int> ClTrackOrigins("msan-track-origins",
       cl::desc("Track origins (allocation sites) of poisoned memory"),
       cl::Hidden, cl::init(0));
static cl::opt<bool> ClKeepGoing("msan-keep-going",
       cl::desc("keep going after reporting a UMR"),
       cl::Hidden, cl::init(false));
static cl::opt<bool> ClPoisonStack("msan-poison-stack",
       cl::desc("poison uninitialized stack variables"),
       cl::Hidden, cl::init(true));
static cl::opt<bool> ClPoisonStackWithCall("msan-poison-stack-with-call",
       cl::desc("poison uninitialized stack variables with a call"),
       cl::Hidden, cl::init(false));
static cl::opt<int> ClPoisonStackPattern("msan-poison-stack-pattern",
       cl::desc("poison uninitialized stack variables with the given patter"),
       cl::Hidden, cl::init(0xff));
static cl::opt<bool> ClPoisonUndef("msan-poison-undef",
       cl::desc("poison undef temps"),
       cl::Hidden, cl::init(true));

static cl::opt<bool> ClHandleICmp("msan-handle-icmp",
       cl::desc("propagate shadow through ICmpEQ and ICmpNE"),
       cl::Hidden, cl::init(true));

static cl::opt<bool> ClHandleICmpExact("msan-handle-icmp-exact",
       cl::desc("exact handling of relational integer ICmp"),
       cl::Hidden, cl::init(false));

// This flag controls whether we check the shadow of the address
// operand of load or store. Such bugs are very rare, since load from
// a garbage address typically results in SEGV, but still happen
// (e.g. only lower bits of address are garbage, or the access happens
// early at program startup where malloc-ed memory is more likely to
// be zeroed. As of 2012-08-28 this flag adds 20% slowdown.
static cl::opt<bool> ClCheckAccessAddress("msan-check-access-address",
       cl::desc("report accesses through a pointer which has poisoned shadow"),
       cl::Hidden, cl::init(true));

static cl::opt<bool> ClDumpStrictInstructions("msan-dump-strict-instructions",
       cl::desc("print out instructions with default strict semantics"),
       cl::Hidden, cl::init(false));

static cl::opt<int> ClInstrumentationWithCallThreshold(
    "msan-instrumentation-with-call-threshold",
    cl::desc(
        "If the function being instrumented requires more than "
        "this number of checks and origin stores, use callbacks instead of "
        "inline checks (-1 means never use callbacks)."),
    cl::Hidden, cl::init(3500));

// This is an experiment to enable handling of cases where shadow is a non-zero
// compile-time constant. For some unexplainable reason they were silently
// ignored in the instrumentation.
static cl::opt<bool> ClCheckConstantShadow("msan-check-constant-shadow",
       cl::desc("Insert checks for constant shadow values"),
       cl::Hidden, cl::init(false));

static const char *const kMsanModuleCtorName = "msan.module_ctor";
static const char *const kMsanInitName = "__msan_init";

namespace {

// Memory map parameters used in application-to-shadow address calculation.
// Offset = (Addr & ~AndMask) ^ XorMask
// Shadow = ShadowBase + Offset
// Origin = OriginBase + Offset
struct MemoryMapParams {
  uint64_t AndMask;
  uint64_t XorMask;
  uint64_t ShadowBase;
  uint64_t OriginBase;
};

struct PlatformMemoryMapParams {
  const MemoryMapParams *bits32;
  const MemoryMapParams *bits64;
};

// i386 Linux
static const MemoryMapParams Linux_I386_MemoryMapParams = {
  0x000080000000,  // AndMask
  0,               // XorMask (not used)
  0,               // ShadowBase (not used)
  0x000040000000,  // OriginBase
};

// x86_64 Linux
static const MemoryMapParams Linux_X86_64_MemoryMapParams = {
  0x400000000000,  // AndMask
  0,               // XorMask (not used)
  0,               // ShadowBase (not used)
  0x200000000000,  // OriginBase
};

// mips64 Linux
static const MemoryMapParams Linux_MIPS64_MemoryMapParams = {
  0x004000000000,  // AndMask
  0,               // XorMask (not used)
  0,               // ShadowBase (not used)
  0x002000000000,  // OriginBase
};

// ppc64 Linux
static const MemoryMapParams Linux_PowerPC64_MemoryMapParams = {
  0x200000000000,  // AndMask
  0x100000000000,  // XorMask
  0x080000000000,  // ShadowBase
  0x1C0000000000,  // OriginBase
};

// i386 FreeBSD
static const MemoryMapParams FreeBSD_I386_MemoryMapParams = {
  0x000180000000,  // AndMask
  0x000040000000,  // XorMask
  0x000020000000,  // ShadowBase
  0x000700000000,  // OriginBase
};

// x86_64 FreeBSD
static const MemoryMapParams FreeBSD_X86_64_MemoryMapParams = {
  0xc00000000000,  // AndMask
  0x200000000000,  // XorMask
  0x100000000000,  // ShadowBase
  0x380000000000,  // OriginBase
};

static const PlatformMemoryMapParams Linux_X86_MemoryMapParams = {
  &Linux_I386_MemoryMapParams,
  &Linux_X86_64_MemoryMapParams,
};

static const PlatformMemoryMapParams Linux_MIPS_MemoryMapParams = {
  NULL,
  &Linux_MIPS64_MemoryMapParams,
};

static const PlatformMemoryMapParams Linux_PowerPC_MemoryMapParams = {
  NULL,
  &Linux_PowerPC64_MemoryMapParams,
};

static const PlatformMemoryMapParams FreeBSD_X86_MemoryMapParams = {
  &FreeBSD_I386_MemoryMapParams,
  &FreeBSD_X86_64_MemoryMapParams,
};

/// \brief An instrumentation pass implementing detection of uninitialized
/// reads.
///
/// MemorySanitizer: instrument the code in module to find
/// uninitialized reads.
class MemorySanitizer : public FunctionPass {
 public:
  MemorySanitizer(int TrackOrigins = 0)
      : FunctionPass(ID),
        TrackOrigins(std::max(TrackOrigins, (int)ClTrackOrigins)),
        WarningFn(nullptr) {}
  StringRef getPassName() const override { return "MemorySanitizer"; }
  bool runOnFunction(Function &F) override;
  bool doInitialization(Module &M) override;
  static char ID;  // Pass identification, replacement for typeid.

 private:
  void initializeCallbacks(Module &M);

  /// \brief Track origins (allocation points) of uninitialized values.
  int TrackOrigins;

  LLVMContext *C;
  Type *IntptrTy;
  Type *OriginTy;
  /// \brief Thread-local shadow storage for function parameters.
  GlobalVariable *ParamTLS;
  /// \brief Thread-local origin storage for function parameters.
  GlobalVariable *ParamOriginTLS;
  /// \brief Thread-local shadow storage for function return value.
  GlobalVariable *RetvalTLS;
  /// \brief Thread-local origin storage for function return value.
  GlobalVariable *RetvalOriginTLS;
  /// \brief Thread-local shadow storage for in-register va_arg function
  /// parameters (x86_64-specific).
  GlobalVariable *VAArgTLS;
  /// \brief Thread-local shadow storage for va_arg overflow area
  /// (x86_64-specific).
  GlobalVariable *VAArgOverflowSizeTLS;
  /// \brief Thread-local space used to pass origin value to the UMR reporting
  /// function.
  GlobalVariable *OriginTLS;

  /// \brief The run-time callback to print a warning.
  Value *WarningFn;
  // These arrays are indexed by log2(AccessSize).
  Value *MaybeWarningFn[kNumberOfAccessSizes];
  Value *MaybeStoreOriginFn[kNumberOfAccessSizes];

  /// \brief Run-time helper that generates a new origin value for a stack
  /// allocation.
  Value *MsanSetAllocaOrigin4Fn;
  /// \brief Run-time helper that poisons stack on function entry.
  Value *MsanPoisonStackFn;
  /// \brief Run-time helper that records a store (or any event) of an
  /// uninitialized value and returns an updated origin id encoding this info.
  Value *MsanChainOriginFn;
  /// \brief MSan runtime replacements for memmove, memcpy and memset.
  Value *MemmoveFn, *MemcpyFn, *MemsetFn;

  /// \brief Memory map parameters used in application-to-shadow calculation.
  const MemoryMapParams *MapParams;

  MDNode *ColdCallWeights;
  /// \brief Branch weights for origin store.
  MDNode *OriginStoreWeights;
  /// \brief An empty volatile inline asm that prevents callback merge.
  InlineAsm *EmptyAsm;
  Function *MsanCtorFunction;

  friend struct MemorySanitizerVisitor;
  friend struct VarArgAMD64Helper;
  friend struct VarArgMIPS64Helper;
};
}  // namespace

char MemorySanitizer::ID = 0;
INITIALIZE_PASS(MemorySanitizer, "msan",
                "MemorySanitizer: detects uninitialized reads.",
                false, false)

FunctionPass *llvm::createMemorySanitizerPass(int TrackOrigins) {
  return new MemorySanitizer(TrackOrigins);
}

/// \brief Create a non-const global initialized with the given string.
///
/// Creates a writable global for Str so that we can pass it to the
/// run-time lib. Runtime uses first 4 bytes of the string to store the
/// frame ID, so the string needs to be mutable.
static GlobalVariable *createPrivateNonConstGlobalForString(Module &M,
                                                            StringRef Str) {
  Constant *StrConst = ConstantDataArray::getString(M.getContext(), Str);
  return new GlobalVariable(M, StrConst->getType(), /*isConstant=*/false,
                            GlobalValue::PrivateLinkage, StrConst, "");
}


/// \brief Insert extern declaration of runtime-provided functions and globals.
void MemorySanitizer::initializeCallbacks(Module &M) {
  // Only do this once.
  if (WarningFn)
    return;

  IRBuilder<> IRB(*C);
  // Create the callback.
  // FIXME: this function should have "Cold" calling conv,
  // which is not yet implemented.
  StringRef WarningFnName = ClKeepGoing ? "__msan_warning"
                                        : "__msan_warning_noreturn";
  WarningFn = M.getOrInsertFunction(WarningFnName, IRB.getVoidTy(), nullptr);

  for (size_t AccessSizeIndex = 0; AccessSizeIndex < kNumberOfAccessSizes;
       AccessSizeIndex++) {
    unsigned AccessSize = 1 << AccessSizeIndex;
    std::string FunctionName = "__msan_maybe_warning_" + itostr(AccessSize);
    MaybeWarningFn[AccessSizeIndex] = M.getOrInsertFunction(
        FunctionName, IRB.getVoidTy(), IRB.getIntNTy(AccessSize * 8),
        IRB.getInt32Ty(), nullptr);

    FunctionName = "__msan_maybe_store_origin_" + itostr(AccessSize);
    MaybeStoreOriginFn[AccessSizeIndex] = M.getOrInsertFunction(
        FunctionName, IRB.getVoidTy(), IRB.getIntNTy(AccessSize * 8),
        IRB.getInt8PtrTy(), IRB.getInt32Ty(), nullptr);
  }

  MsanSetAllocaOrigin4Fn = M.getOrInsertFunction(
    "__msan_set_alloca_origin4", IRB.getVoidTy(), IRB.getInt8PtrTy(), IntptrTy,
    IRB.getInt8PtrTy(), IntptrTy, nullptr);
  MsanPoisonStackFn =
      M.getOrInsertFunction("__msan_poison_stack", IRB.getVoidTy(),
                            IRB.getInt8PtrTy(), IntptrTy, nullptr);
  MsanChainOriginFn = M.getOrInsertFunction(
    "__msan_chain_origin", IRB.getInt32Ty(), IRB.getInt32Ty(), nullptr);
  MemmoveFn = M.getOrInsertFunction(
    "__msan_memmove", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
    IRB.getInt8PtrTy(), IntptrTy, nullptr);
  MemcpyFn = M.getOrInsertFunction(
    "__msan_memcpy", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
    IntptrTy, nullptr);
  MemsetFn = M.getOrInsertFunction(
    "__msan_memset", IRB.getInt8PtrTy(), IRB.getInt8PtrTy(), IRB.getInt32Ty(),
    IntptrTy, nullptr);

  // Create globals.
  RetvalTLS = new GlobalVariable(
    M, ArrayType::get(IRB.getInt64Ty(), kRetvalTLSSize / 8), false,
    GlobalVariable::ExternalLinkage, nullptr, "__msan_retval_tls", nullptr,
    GlobalVariable::InitialExecTLSModel);
  RetvalOriginTLS = new GlobalVariable(
    M, OriginTy, false, GlobalVariable::ExternalLinkage, nullptr,
    "__msan_retval_origin_tls", nullptr, GlobalVariable::InitialExecTLSModel);

  ParamTLS = new GlobalVariable(
    M, ArrayType::get(IRB.getInt64Ty(), kParamTLSSize / 8), false,
    GlobalVariable::ExternalLinkage, nullptr, "__msan_param_tls", nullptr,
    GlobalVariable::InitialExecTLSModel);
  ParamOriginTLS = new GlobalVariable(
    M, ArrayType::get(OriginTy, kParamTLSSize / 4), false,
    GlobalVariable::ExternalLinkage, nullptr, "__msan_param_origin_tls",
    nullptr, GlobalVariable::InitialExecTLSModel);

  VAArgTLS = new GlobalVariable(
    M, ArrayType::get(IRB.getInt64Ty(), kParamTLSSize / 8), false,
    GlobalVariable::ExternalLinkage, nullptr, "__msan_va_arg_tls", nullptr,
    GlobalVariable::InitialExecTLSModel);
  VAArgOverflowSizeTLS = new GlobalVariable(
    M, IRB.getInt64Ty(), false, GlobalVariable::ExternalLinkage, nullptr,
    "__msan_va_arg_overflow_size_tls", nullptr,
    GlobalVariable::InitialExecTLSModel);
  OriginTLS = new GlobalVariable(
    M, IRB.getInt32Ty(), false, GlobalVariable::ExternalLinkage, nullptr,
    "__msan_origin_tls", nullptr, GlobalVariable::InitialExecTLSModel);

  // We insert an empty inline asm after __msan_report* to avoid callback merge.
  EmptyAsm = InlineAsm::get(FunctionType::get(IRB.getVoidTy(), false),
                            StringRef(""), StringRef(""),
                            /*hasSideEffects=*/true);
}

/// \brief Module-level initialization.
///
/// inserts a call to __msan_init to the module's constructor list.
bool MemorySanitizer::doInitialization(Module &M) {
  auto &DL = M.getDataLayout();

  Triple TargetTriple(M.getTargetTriple());
  switch (TargetTriple.getOS()) {
    case Triple::FreeBSD:
      switch (TargetTriple.getArch()) {
        case Triple::x86_64:
          MapParams = FreeBSD_X86_MemoryMapParams.bits64;
          break;
        case Triple::x86:
          MapParams = FreeBSD_X86_MemoryMapParams.bits32;
          break;
        default:
          report_fatal_error("unsupported architecture");
      }
      break;
    case Triple::Linux:
      switch (TargetTriple.getArch()) {
        case Triple::x86_64:
          MapParams = Linux_X86_MemoryMapParams.bits64;
          break;
        case Triple::x86:
          MapParams = Linux_X86_MemoryMapParams.bits32;
          break;
        case Triple::mips64:
        case Triple::mips64el:
          MapParams = Linux_MIPS_MemoryMapParams.bits64;
          break;
        case Triple::ppc64:
        case Triple::ppc64le:
          MapParams = Linux_PowerPC_MemoryMapParams.bits64;
          break;
        default:
          report_fatal_error("unsupported architecture");
      }
      break;
    default:
      report_fatal_error("unsupported operating system");
  }

  C = &(M.getContext());
  IRBuilder<> IRB(*C);
  IntptrTy = IRB.getIntPtrTy(DL);
  OriginTy = IRB.getInt32Ty();

  ColdCallWeights = MDBuilder(*C).createBranchWeights(1, 1000);
  OriginStoreWeights = MDBuilder(*C).createBranchWeights(1, 1000);

  std::tie(MsanCtorFunction, std::ignore) =
      createSanitizerCtorAndInitFunctions(M, kMsanModuleCtorName, kMsanInitName,
                                          /*InitArgTypes=*/{},
                                          /*InitArgs=*/{});

  appendToGlobalCtors(M, MsanCtorFunction, 0);

  if (TrackOrigins)
    new GlobalVariable(M, IRB.getInt32Ty(), true, GlobalValue::WeakODRLinkage,
                       IRB.getInt32(TrackOrigins), "__msan_track_origins");

  if (ClKeepGoing)
    new GlobalVariable(M, IRB.getInt32Ty(), true, GlobalValue::WeakODRLinkage,
                       IRB.getInt32(ClKeepGoing), "__msan_keep_going");

  return true;
}

namespace {

/// \brief A helper class that handles instrumentation of VarArg
/// functions on a particular platform.
///
/// Implementations are expected to insert the instrumentation
/// necessary to propagate argument shadow through VarArg function
/// calls. Visit* methods are called during an InstVisitor pass over
/// the function, and should avoid creating new basic blocks. A new
/// instance of this class is created for each instrumented function.
struct VarArgHelper {
  /// \brief Visit a CallSite.
  virtual void visitCallSite(CallSite &CS, IRBuilder<> &IRB) = 0;

  /// \brief Visit a va_start call.
  virtual void visitVAStartInst(VAStartInst &I) = 0;

  /// \brief Visit a va_copy call.
  virtual void visitVACopyInst(VACopyInst &I) = 0;

  /// \brief Finalize function instrumentation.
  ///
  /// This method is called after visiting all interesting (see above)
  /// instructions in a function.
  virtual void finalizeInstrumentation() = 0;

  virtual ~VarArgHelper() {}
};

struct MemorySanitizerVisitor;

VarArgHelper*
CreateVarArgHelper(Function &Func, MemorySanitizer &Msan,
                   MemorySanitizerVisitor &Visitor);

unsigned TypeSizeToSizeIndex(unsigned TypeSize) {
  if (TypeSize <= 8) return 0;
  return Log2_32_Ceil(TypeSize / 8);
}

/// This class does all the work for a given function. Store and Load
/// instructions store and load corresponding shadow and origin
/// values. Most instructions propagate shadow from arguments to their
/// return values. Certain instructions (most importantly, BranchInst)
/// test their argument shadow and print reports (with a runtime call) if it's
/// non-zero.
struct MemorySanitizerVisitor : public InstVisitor<MemorySanitizerVisitor> {
  Function &F;
  MemorySanitizer &MS;
  SmallVector<PHINode *, 16> ShadowPHINodes, OriginPHINodes;
  ValueMap<Value*, Value*> ShadowMap, OriginMap;
  std::unique_ptr<VarArgHelper> VAHelper;

  // The following flags disable parts of MSan instrumentation based on
  // blacklist contents and command-line options.
  bool InsertChecks;
  bool PropagateShadow;
  bool PoisonStack;
  bool PoisonUndef;
  bool CheckReturnValue;

  struct ShadowOriginAndInsertPoint {
    Value *Shadow;
    Value *Origin;
    Instruction *OrigIns;
    ShadowOriginAndInsertPoint(Value *S, Value *O, Instruction *I)
      : Shadow(S), Origin(O), OrigIns(I) { }
  };
  SmallVector<ShadowOriginAndInsertPoint, 16> InstrumentationList;
  SmallVector<Instruction*, 16> StoreList;

  MemorySanitizerVisitor(Function &F, MemorySanitizer &MS)
      : F(F), MS(MS), VAHelper(CreateVarArgHelper(F, MS, *this)) {
    bool SanitizeFunction = F.hasFnAttribute(Attribute::SanitizeMemory);
    InsertChecks = SanitizeFunction;
    PropagateShadow = SanitizeFunction;
    PoisonStack = SanitizeFunction && ClPoisonStack;
    PoisonUndef = SanitizeFunction && ClPoisonUndef;
    // FIXME: Consider using SpecialCaseList to specify a list of functions that
    // must always return fully initialized values. For now, we hardcode "main".
    CheckReturnValue = SanitizeFunction && (F.getName() == "main");

    DEBUG(if (!InsertChecks)
          dbgs() << "MemorySanitizer is not inserting checks into '"
                 << F.getName() << "'\n");
  }

  Value *updateOrigin(Value *V, IRBuilder<> &IRB) {
    if (MS.TrackOrigins <= 1) return V;
    return IRB.CreateCall(MS.MsanChainOriginFn, V);
  }

  Value *originToIntptr(IRBuilder<> &IRB, Value *Origin) {
    const DataLayout &DL = F.getParent()->getDataLayout();
    unsigned IntptrSize = DL.getTypeStoreSize(MS.IntptrTy);
    if (IntptrSize == kOriginSize) return Origin;
    assert(IntptrSize == kOriginSize * 2);
    Origin = IRB.CreateIntCast(Origin, MS.IntptrTy, /* isSigned */ false);
    return IRB.CreateOr(Origin, IRB.CreateShl(Origin, kOriginSize * 8));
  }

  /// \brief Fill memory range with the given origin value.
  void paintOrigin(IRBuilder<> &IRB, Value *Origin, Value *OriginPtr,
                   unsigned Size, unsigned Alignment) {
    const DataLayout &DL = F.getParent()->getDataLayout();
    unsigned IntptrAlignment = DL.getABITypeAlignment(MS.IntptrTy);
    unsigned IntptrSize = DL.getTypeStoreSize(MS.IntptrTy);
    assert(IntptrAlignment >= kMinOriginAlignment);
    assert(IntptrSize >= kOriginSize);

    unsigned Ofs = 0;
    unsigned CurrentAlignment = Alignment;
    if (Alignment >= IntptrAlignment && IntptrSize > kOriginSize) {
      Value *IntptrOrigin = originToIntptr(IRB, Origin);
      Value *IntptrOriginPtr =
          IRB.CreatePointerCast(OriginPtr, PointerType::get(MS.IntptrTy, 0));
      for (unsigned i = 0; i < Size / IntptrSize; ++i) {
        Value *Ptr = i ? IRB.CreateConstGEP1_32(MS.IntptrTy, IntptrOriginPtr, i)
                       : IntptrOriginPtr;
        IRB.CreateAlignedStore(IntptrOrigin, Ptr, CurrentAlignment);
        Ofs += IntptrSize / kOriginSize;
        CurrentAlignment = IntptrAlignment;
      }
    }

    for (unsigned i = Ofs; i < (Size + kOriginSize - 1) / kOriginSize; ++i) {
      Value *GEP =
          i ? IRB.CreateConstGEP1_32(nullptr, OriginPtr, i) : OriginPtr;
      IRB.CreateAlignedStore(Origin, GEP, CurrentAlignment);
      CurrentAlignment = kMinOriginAlignment;
    }
  }

  void storeOrigin(IRBuilder<> &IRB, Value *Addr, Value *Shadow, Value *Origin,
                   unsigned Alignment, bool AsCall) {
    const DataLayout &DL = F.getParent()->getDataLayout();
    unsigned OriginAlignment = std::max(kMinOriginAlignment, Alignment);
    unsigned StoreSize = DL.getTypeStoreSize(Shadow->getType());
    if (isa<StructType>(Shadow->getType())) {
      paintOrigin(IRB, updateOrigin(Origin, IRB),
                  getOriginPtr(Addr, IRB, Alignment), StoreSize,
                  OriginAlignment);
    } else {
      Value *ConvertedShadow = convertToShadowTyNoVec(Shadow, IRB);
      Constant *ConstantShadow = dyn_cast_or_null<Constant>(ConvertedShadow);
      if (ConstantShadow) {
        if (ClCheckConstantShadow && !ConstantShadow->isZeroValue())
          paintOrigin(IRB, updateOrigin(Origin, IRB),
                      getOriginPtr(Addr, IRB, Alignment), StoreSize,
                      OriginAlignment);
        return;
      }

      unsigned TypeSizeInBits =
          DL.getTypeSizeInBits(ConvertedShadow->getType());
      unsigned SizeIndex = TypeSizeToSizeIndex(TypeSizeInBits);
      if (AsCall && SizeIndex < kNumberOfAccessSizes) {
        Value *Fn = MS.MaybeStoreOriginFn[SizeIndex];
        Value *ConvertedShadow2 = IRB.CreateZExt(
            ConvertedShadow, IRB.getIntNTy(8 * (1 << SizeIndex)));
        IRB.CreateCall(Fn, {ConvertedShadow2,
                            IRB.CreatePointerCast(Addr, IRB.getInt8PtrTy()),
                            Origin});
      } else {
        Value *Cmp = IRB.CreateICmpNE(
            ConvertedShadow, getCleanShadow(ConvertedShadow), "_mscmp");
        Instruction *CheckTerm = SplitBlockAndInsertIfThen(
            Cmp, IRB.GetInsertPoint(), false, MS.OriginStoreWeights);
        IRBuilder<> IRBNew(CheckTerm);
        paintOrigin(IRBNew, updateOrigin(Origin, IRBNew),
                    getOriginPtr(Addr, IRBNew, Alignment), StoreSize,
                    OriginAlignment);
      }
    }
  }

  void materializeStores(bool InstrumentWithCalls) {
    for (auto Inst : StoreList) {
      StoreInst &SI = *dyn_cast<StoreInst>(Inst);

      IRBuilder<> IRB(&SI);
      Value *Val = SI.getValueOperand();
      Value *Addr = SI.getPointerOperand();
      Value *Shadow = SI.isAtomic() ? getCleanShadow(Val) : getShadow(Val);
      Value *ShadowPtr = getShadowPtr(Addr, Shadow->getType(), IRB);

      StoreInst *NewSI =
          IRB.CreateAlignedStore(Shadow, ShadowPtr, SI.getAlignment());
      DEBUG(dbgs() << "  STORE: " << *NewSI << "\n");
      (void)NewSI;

      if (ClCheckAccessAddress) insertShadowCheck(Addr, &SI);

      if (SI.isAtomic()) SI.setOrdering(addReleaseOrdering(SI.getOrdering()));

      if (MS.TrackOrigins && !SI.isAtomic())
        storeOrigin(IRB, Addr, Shadow, getOrigin(Val), SI.getAlignment(),
                    InstrumentWithCalls);
    }
  }

  void materializeOneCheck(Instruction *OrigIns, Value *Shadow, Value *Origin,
                           bool AsCall) {
    IRBuilder<> IRB(OrigIns);
    DEBUG(dbgs() << "  SHAD0 : " << *Shadow << "\n");
    Value *ConvertedShadow = convertToShadowTyNoVec(Shadow, IRB);
    DEBUG(dbgs() << "  SHAD1 : " << *ConvertedShadow << "\n");

    Constant *ConstantShadow = dyn_cast_or_null<Constant>(ConvertedShadow);
    if (ConstantShadow) {
      if (ClCheckConstantShadow && !ConstantShadow->isZeroValue()) {
        if (MS.TrackOrigins) {
          IRB.CreateStore(Origin ? (Value *)Origin : (Value *)IRB.getInt32(0),
                          MS.OriginTLS);
        }
        IRB.CreateCall(MS.WarningFn, {});
        IRB.CreateCall(MS.EmptyAsm, {});
        // FIXME: Insert UnreachableInst if !ClKeepGoing?
        // This may invalidate some of the following checks and needs to be done
        // at the very end.
      }
      return;
    }

    const DataLayout &DL = OrigIns->getModule()->getDataLayout();

    unsigned TypeSizeInBits = DL.getTypeSizeInBits(ConvertedShadow->getType());
    unsigned SizeIndex = TypeSizeToSizeIndex(TypeSizeInBits);
    if (AsCall && SizeIndex < kNumberOfAccessSizes) {
      Value *Fn = MS.MaybeWarningFn[SizeIndex];
      Value *ConvertedShadow2 =
          IRB.CreateZExt(ConvertedShadow, IRB.getIntNTy(8 * (1 << SizeIndex)));
      IRB.CreateCall(Fn, {ConvertedShadow2, MS.TrackOrigins && Origin
                                                ? Origin
                                                : (Value *)IRB.getInt32(0)});
    } else {
      Value *Cmp = IRB.CreateICmpNE(ConvertedShadow,
                                    getCleanShadow(ConvertedShadow), "_mscmp");
      Instruction *CheckTerm = SplitBlockAndInsertIfThen(
          Cmp, OrigIns,
          /* Unreachable */ !ClKeepGoing, MS.ColdCallWeights);

      IRB.SetInsertPoint(CheckTerm);
      if (MS.TrackOrigins) {
        IRB.CreateStore(Origin ? (Value *)Origin : (Value *)IRB.getInt32(0),
                        MS.OriginTLS);
      }
      IRB.CreateCall(MS.WarningFn, {});
      IRB.CreateCall(MS.EmptyAsm, {});
      DEBUG(dbgs() << "  CHECK: " << *Cmp << "\n");
    }
  }

  void materializeChecks(bool InstrumentWithCalls) {
    for (const auto &ShadowData : InstrumentationList) {
      Instruction *OrigIns = ShadowData.OrigIns;
      Value *Shadow = ShadowData.Shadow;
      Value *Origin = ShadowData.Origin;
      materializeOneCheck(OrigIns, Shadow, Origin, InstrumentWithCalls);
    }
    DEBUG(dbgs() << "DONE:\n" << F);
  }

  /// \brief Add MemorySanitizer instrumentation to a function.
  bool runOnFunction() {
    MS.initializeCallbacks(*F.getParent());

    // In the presence of unreachable blocks, we may see Phi nodes with
    // incoming nodes from such blocks. Since InstVisitor skips unreachable
    // blocks, such nodes will not have any shadow value associated with them.
    // It's easier to remove unreachable blocks than deal with missing shadow.
    removeUnreachableBlocks(F);

    // Iterate all BBs in depth-first order and create shadow instructions
    // for all instructions (where applicable).
    // For PHI nodes we create dummy shadow PHIs which will be finalized later.
    for (BasicBlock *BB : depth_first(&F.getEntryBlock()))
      visit(*BB);


    // Finalize PHI nodes.
    for (PHINode *PN : ShadowPHINodes) {
      PHINode *PNS = cast<PHINode>(getShadow(PN));
      PHINode *PNO = MS.TrackOrigins ? cast<PHINode>(getOrigin(PN)) : nullptr;
      size_t NumValues = PN->getNumIncomingValues();
      for (size_t v = 0; v < NumValues; v++) {
        PNS->addIncoming(getShadow(PN, v), PN->getIncomingBlock(v));
        if (PNO) PNO->addIncoming(getOrigin(PN, v), PN->getIncomingBlock(v));
      }
    }

    VAHelper->finalizeInstrumentation();

    bool InstrumentWithCalls = ClInstrumentationWithCallThreshold >= 0 &&
                               InstrumentationList.size() + StoreList.size() >
                                   (unsigned)ClInstrumentationWithCallThreshold;

    // Delayed instrumentation of StoreInst.
    // This may add new checks to be inserted later.
    materializeStores(InstrumentWithCalls);

    // Insert shadow value checks.
    materializeChecks(InstrumentWithCalls);

    return true;
  }

  /// \brief Compute the shadow type that corresponds to a given Value.
  Type *getShadowTy(Value *V) {
    return getShadowTy(V->getType());
  }

  /// \brief Compute the shadow type that corresponds to a given Type.
  Type *getShadowTy(Type *OrigTy) {
    if (!OrigTy->isSized()) {
      return nullptr;
    }
    // For integer type, shadow is the same as the original type.
    // This may return weird-sized types like i1.
    if (IntegerType *IT = dyn_cast<IntegerType>(OrigTy))
      return IT;
    const DataLayout &DL = F.getParent()->getDataLayout();
    if (VectorType *VT = dyn_cast<VectorType>(OrigTy)) {
      uint32_t EltSize = DL.getTypeSizeInBits(VT->getElementType());
      return VectorType::get(IntegerType::get(*MS.C, EltSize),
                             VT->getNumElements());
    }
    if (ArrayType *AT = dyn_cast<ArrayType>(OrigTy)) {
      return ArrayType::get(getShadowTy(AT->getElementType()),
                            AT->getNumElements());
    }
    if (StructType *ST = dyn_cast<StructType>(OrigTy)) {
      SmallVector<Type*, 4> Elements;
      for (unsigned i = 0, n = ST->getNumElements(); i < n; i++)
        Elements.push_back(getShadowTy(ST->getElementType(i)));
      StructType *Res = StructType::get(*MS.C, Elements, ST->isPacked());
      DEBUG(dbgs() << "getShadowTy: " << *ST << " ===> " << *Res << "\n");
      return Res;
    }
    uint32_t TypeSize = DL.getTypeSizeInBits(OrigTy);
    return IntegerType::get(*MS.C, TypeSize);
  }

  /// \brief Flatten a vector type.
  Type *getShadowTyNoVec(Type *ty) {
    if (VectorType *vt = dyn_cast<VectorType>(ty))
      return IntegerType::get(*MS.C, vt->getBitWidth());
    return ty;
  }

  /// \brief Convert a shadow value to it's flattened variant.
  Value *convertToShadowTyNoVec(Value *V, IRBuilder<> &IRB) {
    Type *Ty = V->getType();
    Type *NoVecTy = getShadowTyNoVec(Ty);
    if (Ty == NoVecTy) return V;
    return IRB.CreateBitCast(V, NoVecTy);
  }

  /// \brief Compute the integer shadow offset that corresponds to a given
  /// application address.
  ///
  /// Offset = (Addr & ~AndMask) ^ XorMask
  Value *getShadowPtrOffset(Value *Addr, IRBuilder<> &IRB) {
    uint64_t AndMask = MS.MapParams->AndMask;
    assert(AndMask != 0 && "AndMask shall be specified");
    Value *OffsetLong =
      IRB.CreateAnd(IRB.CreatePointerCast(Addr, MS.IntptrTy),
                    ConstantInt::get(MS.IntptrTy, ~AndMask));

    uint64_t XorMask = MS.MapParams->XorMask;
    if (XorMask != 0)
      OffsetLong = IRB.CreateXor(OffsetLong,
                                 ConstantInt::get(MS.IntptrTy, XorMask));
    return OffsetLong;
  }

  /// \brief Compute the shadow address that corresponds to a given application
  /// address.
  ///
  /// Shadow = ShadowBase + Offset
  Value *getShadowPtr(Value *Addr, Type *ShadowTy,
                      IRBuilder<> &IRB) {
    Value *ShadowLong = getShadowPtrOffset(Addr, IRB);
    uint64_t ShadowBase = MS.MapParams->ShadowBase;
    if (ShadowBase != 0)
      ShadowLong =
        IRB.CreateAdd(ShadowLong,
                      ConstantInt::get(MS.IntptrTy, ShadowBase));
    return IRB.CreateIntToPtr(ShadowLong, PointerType::get(ShadowTy, 0));
  }

  /// \brief Compute the origin address that corresponds to a given application
  /// address.
  ///
  /// OriginAddr = (OriginBase + Offset) & ~3ULL
  Value *getOriginPtr(Value *Addr, IRBuilder<> &IRB, unsigned Alignment) {
    Value *OriginLong = getShadowPtrOffset(Addr, IRB);
    uint64_t OriginBase = MS.MapParams->OriginBase;
    if (OriginBase != 0)
      OriginLong =
        IRB.CreateAdd(OriginLong,
                      ConstantInt::get(MS.IntptrTy, OriginBase));
    if (Alignment < kMinOriginAlignment) {
      uint64_t Mask = kMinOriginAlignment - 1;
      OriginLong = IRB.CreateAnd(OriginLong,
                                 ConstantInt::get(MS.IntptrTy, ~Mask));
    }
    return IRB.CreateIntToPtr(OriginLong,
                              PointerType::get(IRB.getInt32Ty(), 0));
  }

  /// \brief Compute the shadow address for a given function argument.
  ///
  /// Shadow = ParamTLS+ArgOffset.
  Value *getShadowPtrForArgument(Value *A, IRBuilder<> &IRB,
                                 int ArgOffset) {
    Value *Base = IRB.CreatePointerCast(MS.ParamTLS, MS.IntptrTy);
    Base = IRB.CreateAdd(Base, ConstantInt::get(MS.IntptrTy, ArgOffset));
    return IRB.CreateIntToPtr(Base, PointerType::get(getShadowTy(A), 0),
                              "_msarg");
  }

  /// \brief Compute the origin address for a given function argument.
  Value *getOriginPtrForArgument(Value *A, IRBuilder<> &IRB,
                                 int ArgOffset) {
    if (!MS.TrackOrigins) return nullptr;
    Value *Base = IRB.CreatePointerCast(MS.ParamOriginTLS, MS.IntptrTy);
    Base = IRB.CreateAdd(Base, ConstantInt::get(MS.IntptrTy, ArgOffset));
    return IRB.CreateIntToPtr(Base, PointerType::get(MS.OriginTy, 0),
                              "_msarg_o");
  }

  /// \brief Compute the shadow address for a retval.
  Value *getShadowPtrForRetval(Value *A, IRBuilder<> &IRB) {
    Value *Base = IRB.CreatePointerCast(MS.RetvalTLS, MS.IntptrTy);
    return IRB.CreateIntToPtr(Base, PointerType::get(getShadowTy(A), 0),
                              "_msret");
  }

  /// \brief Compute the origin address for a retval.
  Value *getOriginPtrForRetval(IRBuilder<> &IRB) {
    // We keep a single origin for the entire retval. Might be too optimistic.
    return MS.RetvalOriginTLS;
  }

  /// \brief Set SV to be the shadow value for V.
  void setShadow(Value *V, Value *SV) {
    assert(!ShadowMap.count(V) && "Values may only have one shadow");
    ShadowMap[V] = PropagateShadow ? SV : getCleanShadow(V);
  }

  /// \brief Set Origin to be the origin value for V.
  void setOrigin(Value *V, Value *Origin) {
    if (!MS.TrackOrigins) return;
    assert(!OriginMap.count(V) && "Values may only have one origin");
    DEBUG(dbgs() << "ORIGIN: " << *V << "  ==> " << *Origin << "\n");
    OriginMap[V] = Origin;
  }

  /// \brief Create a clean shadow value for a given value.
  ///
  /// Clean shadow (all zeroes) means all bits of the value are defined
  /// (initialized).
  Constant *getCleanShadow(Value *V) {
    Type *ShadowTy = getShadowTy(V);
    if (!ShadowTy)
      return nullptr;
    return Constant::getNullValue(ShadowTy);
  }

  /// \brief Create a dirty shadow of a given shadow type.
  Constant *getPoisonedShadow(Type *ShadowTy) {
    assert(ShadowTy);
    if (isa<IntegerType>(ShadowTy) || isa<VectorType>(ShadowTy))
      return Constant::getAllOnesValue(ShadowTy);
    if (ArrayType *AT = dyn_cast<ArrayType>(ShadowTy)) {
      SmallVector<Constant *, 4> Vals(AT->getNumElements(),
                                      getPoisonedShadow(AT->getElementType()));
      return ConstantArray::get(AT, Vals);
    }
    if (StructType *ST = dyn_cast<StructType>(ShadowTy)) {
      SmallVector<Constant *, 4> Vals;
      for (unsigned i = 0, n = ST->getNumElements(); i < n; i++)
        Vals.push_back(getPoisonedShadow(ST->getElementType(i)));
      return ConstantStruct::get(ST, Vals);
    }
    llvm_unreachable("Unexpected shadow type");
  }

  /// \brief Create a dirty shadow for a given value.
  Constant *getPoisonedShadow(Value *V) {
    Type *ShadowTy = getShadowTy(V);
    if (!ShadowTy)
      return nullptr;
    return getPoisonedShadow(ShadowTy);
  }

  /// \brief Create a clean (zero) origin.
  Value *getCleanOrigin() {
    return Constant::getNullValue(MS.OriginTy);
  }

  /// \brief Get the shadow value for a given Value.
  ///
  /// This function either returns the value set earlier with setShadow,
  /// or extracts if from ParamTLS (for function arguments).
  Value *getShadow(Value *V) {
    if (!PropagateShadow) return getCleanShadow(V);
    if (Instruction *I = dyn_cast<Instruction>(V)) {
      // For instructions the shadow is already stored in the map.
      Value *Shadow = ShadowMap[V];
      if (!Shadow) {
        DEBUG(dbgs() << "No shadow: " << *V << "\n" << *(I->getParent()));
        (void)I;
        assert(Shadow && "No shadow for a value");
      }
      return Shadow;
    }
    if (UndefValue *U = dyn_cast<UndefValue>(V)) {
      Value *AllOnes = PoisonUndef ? getPoisonedShadow(V) : getCleanShadow(V);
      DEBUG(dbgs() << "Undef: " << *U << " ==> " << *AllOnes << "\n");
      (void)U;
      return AllOnes;
    }
    if (Argument *A = dyn_cast<Argument>(V)) {
      // For arguments we compute the shadow on demand and store it in the map.
      Value **ShadowPtr = &ShadowMap[V];
      if (*ShadowPtr)
        return *ShadowPtr;
      Function *F = A->getParent();
      IRBuilder<> EntryIRB(F->getEntryBlock().getFirstNonPHI());
      unsigned ArgOffset = 0;
      const DataLayout &DL = F->getParent()->getDataLayout();
      for (auto &FArg : F->args()) {
        if (!FArg.getType()->isSized()) {
          DEBUG(dbgs() << "Arg is not sized\n");
          continue;
        }
        unsigned Size =
            FArg.hasByValAttr()
                ? DL.getTypeAllocSize(FArg.getType()->getPointerElementType())
                : DL.getTypeAllocSize(FArg.getType());
        if (A == &FArg) {
          bool Overflow = ArgOffset + Size > kParamTLSSize;
          Value *Base = getShadowPtrForArgument(&FArg, EntryIRB, ArgOffset);
          if (FArg.hasByValAttr()) {
            // ByVal pointer itself has clean shadow. We copy the actual
            // argument shadow to the underlying memory.
            // Figure out maximal valid memcpy alignment.
            unsigned ArgAlign = FArg.getParamAlignment();
            if (ArgAlign == 0) {
              Type *EltType = A->getType()->getPointerElementType();
              ArgAlign = DL.getABITypeAlignment(EltType);
            }
            if (Overflow) {
              // ParamTLS overflow.
              EntryIRB.CreateMemSet(
                  getShadowPtr(V, EntryIRB.getInt8Ty(), EntryIRB),
                  Constant::getNullValue(EntryIRB.getInt8Ty()), Size, ArgAlign);
            } else {
              unsigned CopyAlign = std::min(ArgAlign, kShadowTLSAlignment);
              Value *Cpy = EntryIRB.CreateMemCpy(
                  getShadowPtr(V, EntryIRB.getInt8Ty(), EntryIRB), Base, Size,
                  CopyAlign);
              DEBUG(dbgs() << "  ByValCpy: " << *Cpy << "\n");
              (void)Cpy;
            }
            *ShadowPtr = getCleanShadow(V);
          } else {
            if (Overflow) {
              // ParamTLS overflow.
              *ShadowPtr = getCleanShadow(V);
            } else {
              *ShadowPtr =
                  EntryIRB.CreateAlignedLoad(Base, kShadowTLSAlignment);
            }
          }
          DEBUG(dbgs() << "  ARG:    "  << FArg << " ==> " <<
                **ShadowPtr << "\n");
          if (MS.TrackOrigins && !Overflow) {
            Value *OriginPtr =
                getOriginPtrForArgument(&FArg, EntryIRB, ArgOffset);
            setOrigin(A, EntryIRB.CreateLoad(OriginPtr));
          } else {
            setOrigin(A, getCleanOrigin());
          }
        }
        ArgOffset += RoundUpToAlignment(Size, kShadowTLSAlignment);
      }
      assert(*ShadowPtr && "Could not find shadow for an argument");
      return *ShadowPtr;
    }
    // For everything else the shadow is zero.
    return getCleanShadow(V);
  }

  /// \brief Get the shadow for i-th argument of the instruction I.
  Value *getShadow(Instruction *I, int i) {
    return getShadow(I->getOperand(i));
  }

  /// \brief Get the origin for a value.
  Value *getOrigin(Value *V) {
    if (!MS.TrackOrigins) return nullptr;
    if (!PropagateShadow) return getCleanOrigin();
    if (isa<Constant>(V)) return getCleanOrigin();
    assert((isa<Instruction>(V) || isa<Argument>(V)) &&
           "Unexpected value type in getOrigin()");
    Value *Origin = OriginMap[V];
    assert(Origin && "Missing origin");
    return Origin;
  }

  /// \brief Get the origin for i-th argument of the instruction I.
  Value *getOrigin(Instruction *I, int i) {
    return getOrigin(I->getOperand(i));
  }

  /// \brief Remember the place where a shadow check should be inserted.
  ///
  /// This location will be later instrumented with a check that will print a
  /// UMR warning in runtime if the shadow value is not 0.
  void insertShadowCheck(Value *Shadow, Value *Origin, Instruction *OrigIns) {
    assert(Shadow);
    if (!InsertChecks) return;
#ifndef NDEBUG
    Type *ShadowTy = Shadow->getType();
    assert((isa<IntegerType>(ShadowTy) || isa<VectorType>(ShadowTy)) &&
           "Can only insert checks for integer and vector shadow types");
#endif
    InstrumentationList.push_back(
        ShadowOriginAndInsertPoint(Shadow, Origin, OrigIns));
  }

  /// \brief Remember the place where a shadow check should be inserted.
  ///
  /// This location will be later instrumented with a check that will print a
  /// UMR warning in runtime if the value is not fully defined.
  void insertShadowCheck(Value *Val, Instruction *OrigIns) {
    assert(Val);
    Value *Shadow, *Origin;
    if (ClCheckConstantShadow) {
      Shadow = getShadow(Val);
      if (!Shadow) return;
      Origin = getOrigin(Val);
    } else {
      Shadow = dyn_cast_or_null<Instruction>(getShadow(Val));
      if (!Shadow) return;
      Origin = dyn_cast_or_null<Instruction>(getOrigin(Val));
    }
    insertShadowCheck(Shadow, Origin, OrigIns);
  }

  AtomicOrdering addReleaseOrdering(AtomicOrdering a) {
    switch (a) {
      case NotAtomic:
        return NotAtomic;
      case Unordered:
      case Monotonic:
      case Release:
        return Release;
      case Acquire:
      case AcquireRelease:
        return AcquireRelease;
      case SequentiallyConsistent:
        return SequentiallyConsistent;
    }
    llvm_unreachable("Unknown ordering");
  }

  AtomicOrdering addAcquireOrdering(AtomicOrdering a) {
    switch (a) {
      case NotAtomic:
        return NotAtomic;
      case Unordered:
      case Monotonic:
      case Acquire:
        return Acquire;
      case Release:
      case AcquireRelease:
        return AcquireRelease;
      case SequentiallyConsistent:
        return SequentiallyConsistent;
    }
    llvm_unreachable("Unknown ordering");
  }

  // ------------------- Visitors.

  /// \brief Instrument LoadInst
  ///
  /// Loads the corresponding shadow and (optionally) origin.
  /// Optionally, checks that the load address is fully defined.
  void visitLoadInst(LoadInst &I) {
    assert(I.getType()->isSized() && "Load type must have size");
    IRBuilder<> IRB(I.getNextNode());
    Type *ShadowTy = getShadowTy(&I);
    Value *Addr = I.getPointerOperand();
    if (PropagateShadow && !I.getMetadata("nosanitize")) {
      Value *ShadowPtr = getShadowPtr(Addr, ShadowTy, IRB);
      setShadow(&I,
                IRB.CreateAlignedLoad(ShadowPtr, I.getAlignment(), "_msld"));
    } else {
      setShadow(&I, getCleanShadow(&I));
    }

    if (ClCheckAccessAddress)
      insertShadowCheck(I.getPointerOperand(), &I);

    if (I.isAtomic())
      I.setOrdering(addAcquireOrdering(I.getOrdering()));

    if (MS.TrackOrigins) {
      if (PropagateShadow) {
        unsigned Alignment = I.getAlignment();
        unsigned OriginAlignment = std::max(kMinOriginAlignment, Alignment);
        setOrigin(&I, IRB.CreateAlignedLoad(getOriginPtr(Addr, IRB, Alignment),
                                            OriginAlignment));
      } else {
        setOrigin(&I, getCleanOrigin());
      }
    }
  }

  /// \brief Instrument StoreInst
  ///
  /// Stores the corresponding shadow and (optionally) origin.
  /// Optionally, checks that the store address is fully defined.
  void visitStoreInst(StoreInst &I) {
    StoreList.push_back(&I);
  }

  void handleCASOrRMW(Instruction &I) {
    assert(isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I));

    IRBuilder<> IRB(&I);
    Value *Addr = I.getOperand(0);
    Value *ShadowPtr = getShadowPtr(Addr, I.getType(), IRB);

    if (ClCheckAccessAddress)
      insertShadowCheck(Addr, &I);

    // Only test the conditional argument of cmpxchg instruction.
    // The other argument can potentially be uninitialized, but we can not
    // detect this situation reliably without possible false positives.
    if (isa<AtomicCmpXchgInst>(I))
      insertShadowCheck(I.getOperand(1), &I);

    IRB.CreateStore(getCleanShadow(&I), ShadowPtr);

    setShadow(&I, getCleanShadow(&I));
    setOrigin(&I, getCleanOrigin());
  }

  void visitAtomicRMWInst(AtomicRMWInst &I) {
    handleCASOrRMW(I);
    I.setOrdering(addReleaseOrdering(I.getOrdering()));
  }

  void visitAtomicCmpXchgInst(AtomicCmpXchgInst &I) {
    handleCASOrRMW(I);
    I.setSuccessOrdering(addReleaseOrdering(I.getSuccessOrdering()));
  }

  // Vector manipulation.
  void visitExtractElementInst(ExtractElementInst &I) {
    insertShadowCheck(I.getOperand(1), &I);
    IRBuilder<> IRB(&I);
    setShadow(&I, IRB.CreateExtractElement(getShadow(&I, 0), I.getOperand(1),
              "_msprop"));
    setOrigin(&I, getOrigin(&I, 0));
  }

  void visitInsertElementInst(InsertElementInst &I) {
    insertShadowCheck(I.getOperand(2), &I);
    IRBuilder<> IRB(&I);
    setShadow(&I, IRB.CreateInsertElement(getShadow(&I, 0), getShadow(&I, 1),
              I.getOperand(2), "_msprop"));
    setOriginForNaryOp(I);
  }

  void visitShuffleVectorInst(ShuffleVectorInst &I) {
    insertShadowCheck(I.getOperand(2), &I);
    IRBuilder<> IRB(&I);
    setShadow(&I, IRB.CreateShuffleVector(getShadow(&I, 0), getShadow(&I, 1),
              I.getOperand(2), "_msprop"));
    setOriginForNaryOp(I);
  }

  // Casts.
  void visitSExtInst(SExtInst &I) {
    IRBuilder<> IRB(&I);
    setShadow(&I, IRB.CreateSExt(getShadow(&I, 0), I.getType(), "_msprop"));
    setOrigin(&I, getOrigin(&I, 0));
  }

  void visitZExtInst(ZExtInst &I) {
    IRBuilder<> IRB(&I);
    setShadow(&I, IRB.CreateZExt(getShadow(&I, 0), I.getType(), "_msprop"));
    setOrigin(&I, getOrigin(&I, 0));
  }

  void visitTruncInst(TruncInst &I) {
    IRBuilder<> IRB(&I);
    setShadow(&I, IRB.CreateTrunc(getShadow(&I, 0), I.getType(), "_msprop"));
    setOrigin(&I, getOrigin(&I, 0));
  }

  void visitBitCastInst(BitCastInst &I) {
    IRBuilder<> IRB(&I);
    setShadow(&I, IRB.CreateBitCast(getShadow(&I, 0), getShadowTy(&I)));
    setOrigin(&I, getOrigin(&I, 0));
  }

  void visitPtrToIntInst(PtrToIntInst &I) {
    IRBuilder<> IRB(&I);
    setShadow(&I, IRB.CreateIntCast(getShadow(&I, 0), getShadowTy(&I), false,
             "_msprop_ptrtoint"));
    setOrigin(&I, getOrigin(&I, 0));
  }

  void visitIntToPtrInst(IntToPtrInst &I) {
    IRBuilder<> IRB(&I);
    setShadow(&I, IRB.CreateIntCast(getShadow(&I, 0), getShadowTy(&I), false,
             "_msprop_inttoptr"));
    setOrigin(&I, getOrigin(&I, 0));
  }

  void visitFPToSIInst(CastInst& I) { handleShadowOr(I); }
  void visitFPToUIInst(CastInst& I) { handleShadowOr(I); }
  void visitSIToFPInst(CastInst& I) { handleShadowOr(I); }
  void visitUIToFPInst(CastInst& I) { handleShadowOr(I); }
  void visitFPExtInst(CastInst& I) { handleShadowOr(I); }
  void visitFPTruncInst(CastInst& I) { handleShadowOr(I); }

  /// \brief Propagate shadow for bitwise AND.
  ///
  /// This code is exact, i.e. if, for example, a bit in the left argument
  /// is defined and 0, then neither the value not definedness of the
  /// corresponding bit in B don't affect the resulting shadow.
  void visitAnd(BinaryOperator &I) {
    IRBuilder<> IRB(&I);
    //  "And" of 0 and a poisoned value results in unpoisoned value.
    //  1&1 => 1;     0&1 => 0;     p&1 => p;
    //  1&0 => 0;     0&0 => 0;     p&0 => 0;
    //  1&p => p;     0&p => 0;     p&p => p;
    //  S = (S1 & S2) | (V1 & S2) | (S1 & V2)
    Value *S1 = getShadow(&I, 0);
    Value *S2 = getShadow(&I, 1);
    Value *V1 = I.getOperand(0);
    Value *V2 = I.getOperand(1);
    if (V1->getType() != S1->getType()) {
      V1 = IRB.CreateIntCast(V1, S1->getType(), false);
      V2 = IRB.CreateIntCast(V2, S2->getType(), false);
    }
    Value *S1S2 = IRB.CreateAnd(S1, S2);
    Value *V1S2 = IRB.CreateAnd(V1, S2);
    Value *S1V2 = IRB.CreateAnd(S1, V2);
    setShadow(&I, IRB.CreateOr(S1S2, IRB.CreateOr(V1S2, S1V2)));
    setOriginForNaryOp(I);
  }

  void visitOr(BinaryOperator &I) {
    IRBuilder<> IRB(&I);
    //  "Or" of 1 and a poisoned value results in unpoisoned value.
    //  1|1 => 1;     0|1 => 1;     p|1 => 1;
    //  1|0 => 1;     0|0 => 0;     p|0 => p;
    //  1|p => 1;     0|p => p;     p|p => p;
    //  S = (S1 & S2) | (~V1 & S2) | (S1 & ~V2)
    Value *S1 = getShadow(&I, 0);
    Value *S2 = getShadow(&I, 1);
    Value *V1 = IRB.CreateNot(I.getOperand(0));
    Value *V2 = IRB.CreateNot(I.getOperand(1));
    if (V1->getType() != S1->getType()) {
      V1 = IRB.CreateIntCast(V1, S1->getType(), false);
      V2 = IRB.CreateIntCast(V2, S2->getType(), false);
    }
    Value *S1S2 = IRB.CreateAnd(S1, S2);
    Value *V1S2 = IRB.CreateAnd(V1, S2);
    Value *S1V2 = IRB.CreateAnd(S1, V2);
    setShadow(&I, IRB.CreateOr(S1S2, IRB.CreateOr(V1S2, S1V2)));
    setOriginForNaryOp(I);
  }

  /// \brief Default propagation of shadow and/or origin.
  ///
  /// This class implements the general case of shadow propagation, used in all
  /// cases where we don't know and/or don't care about what the operation
  /// actually does. It converts all input shadow values to a common type
  /// (extending or truncating as necessary), and bitwise OR's them.
  ///
  /// This is much cheaper than inserting checks (i.e. requiring inputs to be
  /// fully initialized), and less prone to false positives.
  ///
  /// This class also implements the general case of origin propagation. For a
  /// Nary operation, result origin is set to the origin of an argument that is
  /// not entirely initialized. If there is more than one such arguments, the
  /// rightmost of them is picked. It does not matter which one is picked if all
  /// arguments are initialized.
  template <bool CombineShadow>
  class Combiner {
    Value *Shadow;
    Value *Origin;
    IRBuilder<> &IRB;
    MemorySanitizerVisitor *MSV;

  public:
    Combiner(MemorySanitizerVisitor *MSV, IRBuilder<> &IRB) :
      Shadow(nullptr), Origin(nullptr), IRB(IRB), MSV(MSV) {}

    /// \brief Add a pair of shadow and origin values to the mix.
    Combiner &Add(Value *OpShadow, Value *OpOrigin) {
      if (CombineShadow) {
        assert(OpShadow);
        if (!Shadow)
          Shadow = OpShadow;
        else {
          OpShadow = MSV->CreateShadowCast(IRB, OpShadow, Shadow->getType());
          Shadow = IRB.CreateOr(Shadow, OpShadow, "_msprop");
        }
      }

      if (MSV->MS.TrackOrigins) {
        assert(OpOrigin);
        if (!Origin) {
          Origin = OpOrigin;
        } else {
          Constant *ConstOrigin = dyn_cast<Constant>(OpOrigin);
          // No point in adding something that might result in 0 origin value.
          if (!ConstOrigin || !ConstOrigin->isNullValue()) {
            Value *FlatShadow = MSV->convertToShadowTyNoVec(OpShadow, IRB);
            Value *Cond =
                IRB.CreateICmpNE(FlatShadow, MSV->getCleanShadow(FlatShadow));
            Origin = IRB.CreateSelect(Cond, OpOrigin, Origin);
          }
        }
      }
      return *this;
    }

    /// \brief Add an application value to the mix.
    Combiner &Add(Value *V) {
      Value *OpShadow = MSV->getShadow(V);
      Value *OpOrigin = MSV->MS.TrackOrigins ? MSV->getOrigin(V) : nullptr;
      return Add(OpShadow, OpOrigin);
    }

    /// \brief Set the current combined values as the given instruction's shadow
    /// and origin.
    void Done(Instruction *I) {
      if (CombineShadow) {
        assert(Shadow);
        Shadow = MSV->CreateShadowCast(IRB, Shadow, MSV->getShadowTy(I));
        MSV->setShadow(I, Shadow);
      }
      if (MSV->MS.TrackOrigins) {
        assert(Origin);
        MSV->setOrigin(I, Origin);
      }
    }
  };

  typedef Combiner<true> ShadowAndOriginCombiner;
  typedef Combiner<false> OriginCombiner;

  /// \brief Propagate origin for arbitrary operation.
  void setOriginForNaryOp(Instruction &I) {
    if (!MS.TrackOrigins) return;
    IRBuilder<> IRB(&I);
    OriginCombiner OC(this, IRB);
    for (Instruction::op_iterator OI = I.op_begin(); OI != I.op_end(); ++OI)
      OC.Add(OI->get());
    OC.Done(&I);
  }

  size_t VectorOrPrimitiveTypeSizeInBits(Type *Ty) {
    assert(!(Ty->isVectorTy() && Ty->getScalarType()->isPointerTy()) &&
           "Vector of pointers is not a valid shadow type");
    return Ty->isVectorTy() ?
      Ty->getVectorNumElements() * Ty->getScalarSizeInBits() :
      Ty->getPrimitiveSizeInBits();
  }

  /// \brief Cast between two shadow types, extending or truncating as
  /// necessary.
  Value *CreateShadowCast(IRBuilder<> &IRB, Value *V, Type *dstTy,
                          bool Signed = false) {
    Type *srcTy = V->getType();
    if (dstTy->isIntegerTy() && srcTy->isIntegerTy())
      return IRB.CreateIntCast(V, dstTy, Signed);
    if (dstTy->isVectorTy() && srcTy->isVectorTy() &&
        dstTy->getVectorNumElements() == srcTy->getVectorNumElements())
      return IRB.CreateIntCast(V, dstTy, Signed);
    size_t srcSizeInBits = VectorOrPrimitiveTypeSizeInBits(srcTy);
    size_t dstSizeInBits = VectorOrPrimitiveTypeSizeInBits(dstTy);
    Value *V1 = IRB.CreateBitCast(V, Type::getIntNTy(*MS.C, srcSizeInBits));
    Value *V2 =
      IRB.CreateIntCast(V1, Type::getIntNTy(*MS.C, dstSizeInBits), Signed);
    return IRB.CreateBitCast(V2, dstTy);
    // TODO: handle struct types.
  }

  /// \brief Cast an application value to the type of its own shadow.
  Value *CreateAppToShadowCast(IRBuilder<> &IRB, Value *V) {
    Type *ShadowTy = getShadowTy(V);
    if (V->getType() == ShadowTy)
      return V;
    if (V->getType()->isPtrOrPtrVectorTy())
      return IRB.CreatePtrToInt(V, ShadowTy);
    else
      return IRB.CreateBitCast(V, ShadowTy);
  }

  /// \brief Propagate shadow for arbitrary operation.
  void handleShadowOr(Instruction &I) {
    IRBuilder<> IRB(&I);
    ShadowAndOriginCombiner SC(this, IRB);
    for (Instruction::op_iterator OI = I.op_begin(); OI != I.op_end(); ++OI)
      SC.Add(OI->get());
    SC.Done(&I);
  }

  // \brief Handle multiplication by constant.
  //
  // Handle a special case of multiplication by constant that may have one or
  // more zeros in the lower bits. This makes corresponding number of lower bits
  // of the result zero as well. We model it by shifting the other operand
  // shadow left by the required number of bits. Effectively, we transform
  // (X * (A * 2**B)) to ((X << B) * A) and instrument (X << B) as (Sx << B).
  // We use multiplication by 2**N instead of shift to cover the case of
  // multiplication by 0, which may occur in some elements of a vector operand.
  void handleMulByConstant(BinaryOperator &I, Constant *ConstArg,
                           Value *OtherArg) {
    Constant *ShadowMul;
    Type *Ty = ConstArg->getType();
    if (Ty->isVectorTy()) {
      unsigned NumElements = Ty->getVectorNumElements();
      Type *EltTy = Ty->getSequentialElementType();
      SmallVector<Constant *, 16> Elements;
      for (unsigned Idx = 0; Idx < NumElements; ++Idx) {
        ConstantInt *Elt =
            dyn_cast<ConstantInt>(ConstArg->getAggregateElement(Idx));
        APInt V = Elt->getValue();
        APInt V2 = APInt(V.getBitWidth(), 1) << V.countTrailingZeros();
        Elements.push_back(ConstantInt::get(EltTy, V2));
      }
      ShadowMul = ConstantVector::get(Elements);
    } else {
      ConstantInt *Elt = dyn_cast<ConstantInt>(ConstArg);
      APInt V = Elt->getValue();
      APInt V2 = APInt(V.getBitWidth(), 1) << V.countTrailingZeros();
      ShadowMul = ConstantInt::get(Elt->getType(), V2);
    }

    IRBuilder<> IRB(&I);
    setShadow(&I,
              IRB.CreateMul(getShadow(OtherArg), ShadowMul, "msprop_mul_cst"));
    setOrigin(&I, getOrigin(OtherArg));
  }

  void visitMul(BinaryOperator &I) {
    Constant *constOp0 = dyn_cast<Constant>(I.getOperand(0));
    Constant *constOp1 = dyn_cast<Constant>(I.getOperand(1));
    if (constOp0 && !constOp1)
      handleMulByConstant(I, constOp0, I.getOperand(1));
    else if (constOp1 && !constOp0)
      handleMulByConstant(I, constOp1, I.getOperand(0));
    else
      handleShadowOr(I);
  }

  void visitFAdd(BinaryOperator &I) { handleShadowOr(I); }
  void visitFSub(BinaryOperator &I) { handleShadowOr(I); }
  void visitFMul(BinaryOperator &I) { handleShadowOr(I); }
  void visitAdd(BinaryOperator &I) { handleShadowOr(I); }
  void visitSub(BinaryOperator &I) { handleShadowOr(I); }
  void visitXor(BinaryOperator &I) { handleShadowOr(I); }

  void handleDiv(Instruction &I) {
    IRBuilder<> IRB(&I);
    // Strict on the second argument.
    insertShadowCheck(I.getOperand(1), &I);
    setShadow(&I, getShadow(&I, 0));
    setOrigin(&I, getOrigin(&I, 0));
  }

  void visitUDiv(BinaryOperator &I) { handleDiv(I); }
  void visitSDiv(BinaryOperator &I) { handleDiv(I); }
  void visitFDiv(BinaryOperator &I) { handleDiv(I); }
  void visitURem(BinaryOperator &I) { handleDiv(I); }
  void visitSRem(BinaryOperator &I) { handleDiv(I); }
  void visitFRem(BinaryOperator &I) { handleDiv(I); }

  /// \brief Instrument == and != comparisons.
  ///
  /// Sometimes the comparison result is known even if some of the bits of the
  /// arguments are not.
  void handleEqualityComparison(ICmpInst &I) {
    IRBuilder<> IRB(&I);
    Value *A = I.getOperand(0);
    Value *B = I.getOperand(1);
    Value *Sa = getShadow(A);
    Value *Sb = getShadow(B);

    // Get rid of pointers and vectors of pointers.
    // For ints (and vectors of ints), types of A and Sa match,
    // and this is a no-op.
    A = IRB.CreatePointerCast(A, Sa->getType());
    B = IRB.CreatePointerCast(B, Sb->getType());

    // A == B  <==>  (C = A^B) == 0
    // A != B  <==>  (C = A^B) != 0
    // Sc = Sa | Sb
    Value *C = IRB.CreateXor(A, B);
    Value *Sc = IRB.CreateOr(Sa, Sb);
    // Now dealing with i = (C == 0) comparison (or C != 0, does not matter now)
    // Result is defined if one of the following is true
    // * there is a defined 1 bit in C
    // * C is fully defined
    // Si = !(C & ~Sc) && Sc
    Value *Zero = Constant::getNullValue(Sc->getType());
    Value *MinusOne = Constant::getAllOnesValue(Sc->getType());
    Value *Si =
      IRB.CreateAnd(IRB.CreateICmpNE(Sc, Zero),
                    IRB.CreateICmpEQ(
                      IRB.CreateAnd(IRB.CreateXor(Sc, MinusOne), C), Zero));
    Si->setName("_msprop_icmp");
    setShadow(&I, Si);
    setOriginForNaryOp(I);
  }

  /// \brief Build the lowest possible value of V, taking into account V's
  ///        uninitialized bits.
  Value *getLowestPossibleValue(IRBuilder<> &IRB, Value *A, Value *Sa,
                                bool isSigned) {
    if (isSigned) {
      // Split shadow into sign bit and other bits.
      Value *SaOtherBits = IRB.CreateLShr(IRB.CreateShl(Sa, 1), 1);
      Value *SaSignBit = IRB.CreateXor(Sa, SaOtherBits);
      // Maximise the undefined shadow bit, minimize other undefined bits.
      return
        IRB.CreateOr(IRB.CreateAnd(A, IRB.CreateNot(SaOtherBits)), SaSignBit);
    } else {
      // Minimize undefined bits.
      return IRB.CreateAnd(A, IRB.CreateNot(Sa));
    }
  }

  /// \brief Build the highest possible value of V, taking into account V's
  ///        uninitialized bits.
  Value *getHighestPossibleValue(IRBuilder<> &IRB, Value *A, Value *Sa,
                                bool isSigned) {
    if (isSigned) {
      // Split shadow into sign bit and other bits.
      Value *SaOtherBits = IRB.CreateLShr(IRB.CreateShl(Sa, 1), 1);
      Value *SaSignBit = IRB.CreateXor(Sa, SaOtherBits);
      // Minimise the undefined shadow bit, maximise other undefined bits.
      return
        IRB.CreateOr(IRB.CreateAnd(A, IRB.CreateNot(SaSignBit)), SaOtherBits);
    } else {
      // Maximize undefined bits.
      return IRB.CreateOr(A, Sa);
    }
  }

  /// \brief Instrument relational comparisons.
  ///
  /// This function does exact shadow propagation for all relational
  /// comparisons of integers, pointers and vectors of those.
  /// FIXME: output seems suboptimal when one of the operands is a constant
  void handleRelationalComparisonExact(ICmpInst &I) {
    IRBuilder<> IRB(&I);
    Value *A = I.getOperand(0);
    Value *B = I.getOperand(1);
    Value *Sa = getShadow(A);
    Value *Sb = getShadow(B);

    // Get rid of pointers and vectors of pointers.
    // For ints (and vectors of ints), types of A and Sa match,
    // and this is a no-op.
    A = IRB.CreatePointerCast(A, Sa->getType());
    B = IRB.CreatePointerCast(B, Sb->getType());

    // Let [a0, a1] be the interval of possible values of A, taking into account
    // its undefined bits. Let [b0, b1] be the interval of possible values of B.
    // Then (A cmp B) is defined iff (a0 cmp b1) == (a1 cmp b0).
    bool IsSigned = I.isSigned();
    Value *S1 = IRB.CreateICmp(I.getPredicate(),
                               getLowestPossibleValue(IRB, A, Sa, IsSigned),
                               getHighestPossibleValue(IRB, B, Sb, IsSigned));
    Value *S2 = IRB.CreateICmp(I.getPredicate(),
                               getHighestPossibleValue(IRB, A, Sa, IsSigned),
                               getLowestPossibleValue(IRB, B, Sb, IsSigned));
    Value *Si = IRB.CreateXor(S1, S2);
    setShadow(&I, Si);
    setOriginForNaryOp(I);
  }

  /// \brief Instrument signed relational comparisons.
  ///
  /// Handle (x<0) and (x>=0) comparisons (essentially, sign bit tests) by
  /// propagating the highest bit of the shadow. Everything else is delegated
  /// to handleShadowOr().
  void handleSignedRelationalComparison(ICmpInst &I) {
    Constant *constOp0 = dyn_cast<Constant>(I.getOperand(0));
    Constant *constOp1 = dyn_cast<Constant>(I.getOperand(1));
    Value* op = nullptr;
    CmpInst::Predicate pre = I.getPredicate();
    if (constOp0 && constOp0->isNullValue() &&
        (pre == CmpInst::ICMP_SGT || pre == CmpInst::ICMP_SLE)) {
      op = I.getOperand(1);
    } else if (constOp1 && constOp1->isNullValue() &&
               (pre == CmpInst::ICMP_SLT || pre == CmpInst::ICMP_SGE)) {
      op = I.getOperand(0);
    }
    if (op) {
      IRBuilder<> IRB(&I);
      Value* Shadow =
        IRB.CreateICmpSLT(getShadow(op), getCleanShadow(op), "_msprop_icmpslt");
      setShadow(&I, Shadow);
      setOrigin(&I, getOrigin(op));
    } else {
      handleShadowOr(I);
    }
  }

  void visitICmpInst(ICmpInst &I) {
    if (!ClHandleICmp) {
      handleShadowOr(I);
      return;
    }
    if (I.isEquality()) {
      handleEqualityComparison(I);
      return;
    }

    assert(I.isRelational());
    if (ClHandleICmpExact) {
      handleRelationalComparisonExact(I);
      return;
    }
    if (I.isSigned()) {
      handleSignedRelationalComparison(I);
      return;
    }

    assert(I.isUnsigned());
    if ((isa<Constant>(I.getOperand(0)) || isa<Constant>(I.getOperand(1)))) {
      handleRelationalComparisonExact(I);
      return;
    }

    handleShadowOr(I);
  }

  void visitFCmpInst(FCmpInst &I) {
    handleShadowOr(I);
  }

  void handleShift(BinaryOperator &I) {
    IRBuilder<> IRB(&I);
    // If any of the S2 bits are poisoned, the whole thing is poisoned.
    // Otherwise perform the same shift on S1.
    Value *S1 = getShadow(&I, 0);
    Value *S2 = getShadow(&I, 1);
    Value *S2Conv = IRB.CreateSExt(IRB.CreateICmpNE(S2, getCleanShadow(S2)),
                                   S2->getType());
    Value *V2 = I.getOperand(1);
    Value *Shift = IRB.CreateBinOp(I.getOpcode(), S1, V2);
    setShadow(&I, IRB.CreateOr(Shift, S2Conv));
    setOriginForNaryOp(I);
  }

  void visitShl(BinaryOperator &I) { handleShift(I); }
  void visitAShr(BinaryOperator &I) { handleShift(I); }
  void visitLShr(BinaryOperator &I) { handleShift(I); }

  /// \brief Instrument llvm.memmove
  ///
  /// At this point we don't know if llvm.memmove will be inlined or not.
  /// If we don't instrument it and it gets inlined,
  /// our interceptor will not kick in and we will lose the memmove.
  /// If we instrument the call here, but it does not get inlined,
  /// we will memove the shadow twice: which is bad in case
  /// of overlapping regions. So, we simply lower the intrinsic to a call.
  ///
  /// Similar situation exists for memcpy and memset.
  void visitMemMoveInst(MemMoveInst &I) {
    IRBuilder<> IRB(&I);
    IRB.CreateCall(
        MS.MemmoveFn,
        {IRB.CreatePointerCast(I.getArgOperand(0), IRB.getInt8PtrTy()),
         IRB.CreatePointerCast(I.getArgOperand(1), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(I.getArgOperand(2), MS.IntptrTy, false)});
    I.eraseFromParent();
  }

  // Similar to memmove: avoid copying shadow twice.
  // This is somewhat unfortunate as it may slowdown small constant memcpys.
  // FIXME: consider doing manual inline for small constant sizes and proper
  // alignment.
  void visitMemCpyInst(MemCpyInst &I) {
    IRBuilder<> IRB(&I);
    IRB.CreateCall(
        MS.MemcpyFn,
        {IRB.CreatePointerCast(I.getArgOperand(0), IRB.getInt8PtrTy()),
         IRB.CreatePointerCast(I.getArgOperand(1), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(I.getArgOperand(2), MS.IntptrTy, false)});
    I.eraseFromParent();
  }

  // Same as memcpy.
  void visitMemSetInst(MemSetInst &I) {
    IRBuilder<> IRB(&I);
    IRB.CreateCall(
        MS.MemsetFn,
        {IRB.CreatePointerCast(I.getArgOperand(0), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(I.getArgOperand(1), IRB.getInt32Ty(), false),
         IRB.CreateIntCast(I.getArgOperand(2), MS.IntptrTy, false)});
    I.eraseFromParent();
  }

  void visitVAStartInst(VAStartInst &I) {
    VAHelper->visitVAStartInst(I);
  }

  void visitVACopyInst(VACopyInst &I) {
    VAHelper->visitVACopyInst(I);
  }

  enum IntrinsicKind {
    IK_DoesNotAccessMemory,
    IK_OnlyReadsMemory,
    IK_WritesMemory
  };

  static IntrinsicKind getIntrinsicKind(Intrinsic::ID iid) {
    const int DoesNotAccessMemory = IK_DoesNotAccessMemory;
    const int OnlyReadsArgumentPointees = IK_OnlyReadsMemory;
    const int OnlyReadsMemory = IK_OnlyReadsMemory;
    const int OnlyAccessesArgumentPointees = IK_WritesMemory;
    const int UnknownModRefBehavior = IK_WritesMemory;
#define GET_INTRINSIC_MODREF_BEHAVIOR
#define ModRefBehavior IntrinsicKind
#include "llvm/IR/Intrinsics.gen"
#undef ModRefBehavior
#undef GET_INTRINSIC_MODREF_BEHAVIOR
  }

  /// \brief Handle vector store-like intrinsics.
  ///
  /// Instrument intrinsics that look like a simple SIMD store: writes memory,
  /// has 1 pointer argument and 1 vector argument, returns void.
  bool handleVectorStoreIntrinsic(IntrinsicInst &I) {
    IRBuilder<> IRB(&I);
    Value* Addr = I.getArgOperand(0);
    Value *Shadow = getShadow(&I, 1);
    Value *ShadowPtr = getShadowPtr(Addr, Shadow->getType(), IRB);

    // We don't know the pointer alignment (could be unaligned SSE store!).
    // Have to assume to worst case.
    IRB.CreateAlignedStore(Shadow, ShadowPtr, 1);

    if (ClCheckAccessAddress)
      insertShadowCheck(Addr, &I);

    // FIXME: use ClStoreCleanOrigin
    // FIXME: factor out common code from materializeStores
    if (MS.TrackOrigins)
      IRB.CreateStore(getOrigin(&I, 1), getOriginPtr(Addr, IRB, 1));
    return true;
  }

  /// \brief Handle vector load-like intrinsics.
  ///
  /// Instrument intrinsics that look like a simple SIMD load: reads memory,
  /// has 1 pointer argument, returns a vector.
  bool handleVectorLoadIntrinsic(IntrinsicInst &I) {
    IRBuilder<> IRB(&I);
    Value *Addr = I.getArgOperand(0);

    Type *ShadowTy = getShadowTy(&I);
    if (PropagateShadow) {
      Value *ShadowPtr = getShadowPtr(Addr, ShadowTy, IRB);
      // We don't know the pointer alignment (could be unaligned SSE load!).
      // Have to assume to worst case.
      setShadow(&I, IRB.CreateAlignedLoad(ShadowPtr, 1, "_msld"));
    } else {
      setShadow(&I, getCleanShadow(&I));
    }

    if (ClCheckAccessAddress)
      insertShadowCheck(Addr, &I);

    if (MS.TrackOrigins) {
      if (PropagateShadow)
        setOrigin(&I, IRB.CreateLoad(getOriginPtr(Addr, IRB, 1)));
      else
        setOrigin(&I, getCleanOrigin());
    }
    return true;
  }

  /// \brief Handle (SIMD arithmetic)-like intrinsics.
  ///
  /// Instrument intrinsics with any number of arguments of the same type,
  /// equal to the return type. The type should be simple (no aggregates or
  /// pointers; vectors are fine).
  /// Caller guarantees that this intrinsic does not access memory.
  bool maybeHandleSimpleNomemIntrinsic(IntrinsicInst &I) {
    Type *RetTy = I.getType();
    if (!(RetTy->isIntOrIntVectorTy() ||
          RetTy->isFPOrFPVectorTy() ||
          RetTy->isX86_MMXTy()))
      return false;

    unsigned NumArgOperands = I.getNumArgOperands();

    for (unsigned i = 0; i < NumArgOperands; ++i) {
      Type *Ty = I.getArgOperand(i)->getType();
      if (Ty != RetTy)
        return false;
    }

    IRBuilder<> IRB(&I);
    ShadowAndOriginCombiner SC(this, IRB);
    for (unsigned i = 0; i < NumArgOperands; ++i)
      SC.Add(I.getArgOperand(i));
    SC.Done(&I);

    return true;
  }

  /// \brief Heuristically instrument unknown intrinsics.
  ///
  /// The main purpose of this code is to do something reasonable with all
  /// random intrinsics we might encounter, most importantly - SIMD intrinsics.
  /// We recognize several classes of intrinsics by their argument types and
  /// ModRefBehaviour and apply special intrumentation when we are reasonably
  /// sure that we know what the intrinsic does.
  ///
  /// We special-case intrinsics where this approach fails. See llvm.bswap
  /// handling as an example of that.
  bool handleUnknownIntrinsic(IntrinsicInst &I) {
    unsigned NumArgOperands = I.getNumArgOperands();
    if (NumArgOperands == 0)
      return false;

    Intrinsic::ID iid = I.getIntrinsicID();
    IntrinsicKind IK = getIntrinsicKind(iid);
    bool OnlyReadsMemory = IK == IK_OnlyReadsMemory;
    bool WritesMemory = IK == IK_WritesMemory;
    assert(!(OnlyReadsMemory && WritesMemory));

    if (NumArgOperands == 2 &&
        I.getArgOperand(0)->getType()->isPointerTy() &&
        I.getArgOperand(1)->getType()->isVectorTy() &&
        I.getType()->isVoidTy() &&
        WritesMemory) {
      // This looks like a vector store.
      return handleVectorStoreIntrinsic(I);
    }

    if (NumArgOperands == 1 &&
        I.getArgOperand(0)->getType()->isPointerTy() &&
        I.getType()->isVectorTy() &&
        OnlyReadsMemory) {
      // This looks like a vector load.
      return handleVectorLoadIntrinsic(I);
    }

    if (!OnlyReadsMemory && !WritesMemory)
      if (maybeHandleSimpleNomemIntrinsic(I))
        return true;

    // FIXME: detect and handle SSE maskstore/maskload
    return false;
  }

  void handleBswap(IntrinsicInst &I) {
    IRBuilder<> IRB(&I);
    Value *Op = I.getArgOperand(0);
    Type *OpType = Op->getType();
    Function *BswapFunc = Intrinsic::getDeclaration(
      F.getParent(), Intrinsic::bswap, makeArrayRef(&OpType, 1));
    setShadow(&I, IRB.CreateCall(BswapFunc, getShadow(Op)));
    setOrigin(&I, getOrigin(Op));
  }

  // \brief Instrument vector convert instrinsic.
  //
  // This function instruments intrinsics like cvtsi2ss:
  // %Out = int_xxx_cvtyyy(%ConvertOp)
  // or
  // %Out = int_xxx_cvtyyy(%CopyOp, %ConvertOp)
  // Intrinsic converts \p NumUsedElements elements of \p ConvertOp to the same
  // number \p Out elements, and (if has 2 arguments) copies the rest of the
  // elements from \p CopyOp.
  // In most cases conversion involves floating-point value which may trigger a
  // hardware exception when not fully initialized. For this reason we require
  // \p ConvertOp[0:NumUsedElements] to be fully initialized and trap otherwise.
  // We copy the shadow of \p CopyOp[NumUsedElements:] to \p
  // Out[NumUsedElements:]. This means that intrinsics without \p CopyOp always
  // return a fully initialized value.
  void handleVectorConvertIntrinsic(IntrinsicInst &I, int NumUsedElements) {
    IRBuilder<> IRB(&I);
    Value *CopyOp, *ConvertOp;

    switch (I.getNumArgOperands()) {
    case 3:
      assert(isa<ConstantInt>(I.getArgOperand(2)) && "Invalid rounding mode");
    case 2:
      CopyOp = I.getArgOperand(0);
      ConvertOp = I.getArgOperand(1);
      break;
    case 1:
      ConvertOp = I.getArgOperand(0);
      CopyOp = nullptr;
      break;
    default:
      llvm_unreachable("Cvt intrinsic with unsupported number of arguments.");
    }

    // The first *NumUsedElements* elements of ConvertOp are converted to the
    // same number of output elements. The rest of the output is copied from
    // CopyOp, or (if not available) filled with zeroes.
    // Combine shadow for elements of ConvertOp that are used in this operation,
    // and insert a check.
    // FIXME: consider propagating shadow of ConvertOp, at least in the case of
    // int->any conversion.
    Value *ConvertShadow = getShadow(ConvertOp);
    Value *AggShadow = nullptr;
    if (ConvertOp->getType()->isVectorTy()) {
      AggShadow = IRB.CreateExtractElement(
          ConvertShadow, ConstantInt::get(IRB.getInt32Ty(), 0));
      for (int i = 1; i < NumUsedElements; ++i) {
        Value *MoreShadow = IRB.CreateExtractElement(
            ConvertShadow, ConstantInt::get(IRB.getInt32Ty(), i));
        AggShadow = IRB.CreateOr(AggShadow, MoreShadow);
      }
    } else {
      AggShadow = ConvertShadow;
    }
    assert(AggShadow->getType()->isIntegerTy());
    insertShadowCheck(AggShadow, getOrigin(ConvertOp), &I);

    // Build result shadow by zero-filling parts of CopyOp shadow that come from
    // ConvertOp.
    if (CopyOp) {
      assert(CopyOp->getType() == I.getType());
      assert(CopyOp->getType()->isVectorTy());
      Value *ResultShadow = getShadow(CopyOp);
      Type *EltTy = ResultShadow->getType()->getVectorElementType();
      for (int i = 0; i < NumUsedElements; ++i) {
        ResultShadow = IRB.CreateInsertElement(
            ResultShadow, ConstantInt::getNullValue(EltTy),
            ConstantInt::get(IRB.getInt32Ty(), i));
      }
      setShadow(&I, ResultShadow);
      setOrigin(&I, getOrigin(CopyOp));
    } else {
      setShadow(&I, getCleanShadow(&I));
      setOrigin(&I, getCleanOrigin());
    }
  }

  // Given a scalar or vector, extract lower 64 bits (or less), and return all
  // zeroes if it is zero, and all ones otherwise.
  Value *Lower64ShadowExtend(IRBuilder<> &IRB, Value *S, Type *T) {
    if (S->getType()->isVectorTy())
      S = CreateShadowCast(IRB, S, IRB.getInt64Ty(), /* Signed */ true);
    assert(S->getType()->getPrimitiveSizeInBits() <= 64);
    Value *S2 = IRB.CreateICmpNE(S, getCleanShadow(S));
    return CreateShadowCast(IRB, S2, T, /* Signed */ true);
  }

  Value *VariableShadowExtend(IRBuilder<> &IRB, Value *S) {
    Type *T = S->getType();
    assert(T->isVectorTy());
    Value *S2 = IRB.CreateICmpNE(S, getCleanShadow(S));
    return IRB.CreateSExt(S2, T);
  }

  // \brief Instrument vector shift instrinsic.
  //
  // This function instruments intrinsics like int_x86_avx2_psll_w.
  // Intrinsic shifts %In by %ShiftSize bits.
  // %ShiftSize may be a vector. In that case the lower 64 bits determine shift
  // size, and the rest is ignored. Behavior is defined even if shift size is
  // greater than register (or field) width.
  void handleVectorShiftIntrinsic(IntrinsicInst &I, bool Variable) {
    assert(I.getNumArgOperands() == 2);
    IRBuilder<> IRB(&I);
    // If any of the S2 bits are poisoned, the whole thing is poisoned.
    // Otherwise perform the same shift on S1.
    Value *S1 = getShadow(&I, 0);
    Value *S2 = getShadow(&I, 1);
    Value *S2Conv = Variable ? VariableShadowExtend(IRB, S2)
                             : Lower64ShadowExtend(IRB, S2, getShadowTy(&I));
    Value *V1 = I.getOperand(0);
    Value *V2 = I.getOperand(1);
    Value *Shift = IRB.CreateCall(I.getCalledValue(),
                                  {IRB.CreateBitCast(S1, V1->getType()), V2});
    Shift = IRB.CreateBitCast(Shift, getShadowTy(&I));
    setShadow(&I, IRB.CreateOr(Shift, S2Conv));
    setOriginForNaryOp(I);
  }

  // \brief Get an X86_MMX-sized vector type.
  Type *getMMXVectorTy(unsigned EltSizeInBits) {
    const unsigned X86_MMXSizeInBits = 64;
    return VectorType::get(IntegerType::get(*MS.C, EltSizeInBits),
                           X86_MMXSizeInBits / EltSizeInBits);
  }

  // \brief Returns a signed counterpart for an (un)signed-saturate-and-pack
  // intrinsic.
  Intrinsic::ID getSignedPackIntrinsic(Intrinsic::ID id) {
    switch (id) {
      case llvm::Intrinsic::x86_sse2_packsswb_128:
      case llvm::Intrinsic::x86_sse2_packuswb_128:
        return llvm::Intrinsic::x86_sse2_packsswb_128;

      case llvm::Intrinsic::x86_sse2_packssdw_128:
      case llvm::Intrinsic::x86_sse41_packusdw:
        return llvm::Intrinsic::x86_sse2_packssdw_128;

      case llvm::Intrinsic::x86_avx2_packsswb:
      case llvm::Intrinsic::x86_avx2_packuswb:
        return llvm::Intrinsic::x86_avx2_packsswb;

      case llvm::Intrinsic::x86_avx2_packssdw:
      case llvm::Intrinsic::x86_avx2_packusdw:
        return llvm::Intrinsic::x86_avx2_packssdw;

      case llvm::Intrinsic::x86_mmx_packsswb:
      case llvm::Intrinsic::x86_mmx_packuswb:
        return llvm::Intrinsic::x86_mmx_packsswb;

      case llvm::Intrinsic::x86_mmx_packssdw:
        return llvm::Intrinsic::x86_mmx_packssdw;
      default:
        llvm_unreachable("unexpected intrinsic id");
    }
  }

  // \brief Instrument vector pack instrinsic.
  //
  // This function instruments intrinsics like x86_mmx_packsswb, that
  // packs elements of 2 input vectors into half as many bits with saturation.
  // Shadow is propagated with the signed variant of the same intrinsic applied
  // to sext(Sa != zeroinitializer), sext(Sb != zeroinitializer).
  // EltSizeInBits is used only for x86mmx arguments.
  void handleVectorPackIntrinsic(IntrinsicInst &I, unsigned EltSizeInBits = 0) {
    assert(I.getNumArgOperands() == 2);
    bool isX86_MMX = I.getOperand(0)->getType()->isX86_MMXTy();
    IRBuilder<> IRB(&I);
    Value *S1 = getShadow(&I, 0);
    Value *S2 = getShadow(&I, 1);
    assert(isX86_MMX || S1->getType()->isVectorTy());

    // SExt and ICmpNE below must apply to individual elements of input vectors.
    // In case of x86mmx arguments, cast them to appropriate vector types and
    // back.
    Type *T = isX86_MMX ? getMMXVectorTy(EltSizeInBits) : S1->getType();
    if (isX86_MMX) {
      S1 = IRB.CreateBitCast(S1, T);
      S2 = IRB.CreateBitCast(S2, T);
    }
    Value *S1_ext = IRB.CreateSExt(
        IRB.CreateICmpNE(S1, llvm::Constant::getNullValue(T)), T);
    Value *S2_ext = IRB.CreateSExt(
        IRB.CreateICmpNE(S2, llvm::Constant::getNullValue(T)), T);
    if (isX86_MMX) {
      Type *X86_MMXTy = Type::getX86_MMXTy(*MS.C);
      S1_ext = IRB.CreateBitCast(S1_ext, X86_MMXTy);
      S2_ext = IRB.CreateBitCast(S2_ext, X86_MMXTy);
    }

    Function *ShadowFn = Intrinsic::getDeclaration(
        F.getParent(), getSignedPackIntrinsic(I.getIntrinsicID()));

    Value *S =
        IRB.CreateCall(ShadowFn, {S1_ext, S2_ext}, "_msprop_vector_pack");
    if (isX86_MMX) S = IRB.CreateBitCast(S, getShadowTy(&I));
    setShadow(&I, S);
    setOriginForNaryOp(I);
  }

  // \brief Instrument sum-of-absolute-differencies intrinsic.
  void handleVectorSadIntrinsic(IntrinsicInst &I) {
    const unsigned SignificantBitsPerResultElement = 16;
    bool isX86_MMX = I.getOperand(0)->getType()->isX86_MMXTy();
    Type *ResTy = isX86_MMX ? IntegerType::get(*MS.C, 64) : I.getType();
    unsigned ZeroBitsPerResultElement =
        ResTy->getScalarSizeInBits() - SignificantBitsPerResultElement;

    IRBuilder<> IRB(&I);
    Value *S = IRB.CreateOr(getShadow(&I, 0), getShadow(&I, 1));
    S = IRB.CreateBitCast(S, ResTy);
    S = IRB.CreateSExt(IRB.CreateICmpNE(S, Constant::getNullValue(ResTy)),
                       ResTy);
    S = IRB.CreateLShr(S, ZeroBitsPerResultElement);
    S = IRB.CreateBitCast(S, getShadowTy(&I));
    setShadow(&I, S);
    setOriginForNaryOp(I);
  }

  // \brief Instrument multiply-add intrinsic.
  void handleVectorPmaddIntrinsic(IntrinsicInst &I,
                                  unsigned EltSizeInBits = 0) {
    bool isX86_MMX = I.getOperand(0)->getType()->isX86_MMXTy();
    Type *ResTy = isX86_MMX ? getMMXVectorTy(EltSizeInBits * 2) : I.getType();
    IRBuilder<> IRB(&I);
    Value *S = IRB.CreateOr(getShadow(&I, 0), getShadow(&I, 1));
    S = IRB.CreateBitCast(S, ResTy);
    S = IRB.CreateSExt(IRB.CreateICmpNE(S, Constant::getNullValue(ResTy)),
                       ResTy);
    S = IRB.CreateBitCast(S, getShadowTy(&I));
    setShadow(&I, S);
    setOriginForNaryOp(I);
  }

  void visitIntrinsicInst(IntrinsicInst &I) {
    switch (I.getIntrinsicID()) {
    case llvm::Intrinsic::bswap:
      handleBswap(I);
      break;
    case llvm::Intrinsic::x86_avx512_cvtsd2usi64:
    case llvm::Intrinsic::x86_avx512_cvtsd2usi:
    case llvm::Intrinsic::x86_avx512_cvtss2usi64:
    case llvm::Intrinsic::x86_avx512_cvtss2usi:
    case llvm::Intrinsic::x86_avx512_cvttss2usi64:
    case llvm::Intrinsic::x86_avx512_cvttss2usi:
    case llvm::Intrinsic::x86_avx512_cvttsd2usi64:
    case llvm::Intrinsic::x86_avx512_cvttsd2usi:
    case llvm::Intrinsic::x86_avx512_cvtusi2sd:
    case llvm::Intrinsic::x86_avx512_cvtusi2ss:
    case llvm::Intrinsic::x86_avx512_cvtusi642sd:
    case llvm::Intrinsic::x86_avx512_cvtusi642ss:
    case llvm::Intrinsic::x86_sse2_cvtsd2si64:
    case llvm::Intrinsic::x86_sse2_cvtsd2si:
    case llvm::Intrinsic::x86_sse2_cvtsd2ss:
    case llvm::Intrinsic::x86_sse2_cvtsi2sd:
    case llvm::Intrinsic::x86_sse2_cvtsi642sd:
    case llvm::Intrinsic::x86_sse2_cvtss2sd:
    case llvm::Intrinsic::x86_sse2_cvttsd2si64:
    case llvm::Intrinsic::x86_sse2_cvttsd2si:
    case llvm::Intrinsic::x86_sse_cvtsi2ss:
    case llvm::Intrinsic::x86_sse_cvtsi642ss:
    case llvm::Intrinsic::x86_sse_cvtss2si64:
    case llvm::Intrinsic::x86_sse_cvtss2si:
    case llvm::Intrinsic::x86_sse_cvttss2si64:
    case llvm::Intrinsic::x86_sse_cvttss2si:
      handleVectorConvertIntrinsic(I, 1);
      break;
    case llvm::Intrinsic::x86_sse2_cvtdq2pd:
    case llvm::Intrinsic::x86_sse2_cvtps2pd:
    case llvm::Intrinsic::x86_sse_cvtps2pi:
    case llvm::Intrinsic::x86_sse_cvttps2pi:
      handleVectorConvertIntrinsic(I, 2);
      break;
    case llvm::Intrinsic::x86_avx2_psll_w:
    case llvm::Intrinsic::x86_avx2_psll_d:
    case llvm::Intrinsic::x86_avx2_psll_q:
    case llvm::Intrinsic::x86_avx2_pslli_w:
    case llvm::Intrinsic::x86_avx2_pslli_d:
    case llvm::Intrinsic::x86_avx2_pslli_q:
    case llvm::Intrinsic::x86_avx2_psrl_w:
    case llvm::Intrinsic::x86_avx2_psrl_d:
    case llvm::Intrinsic::x86_avx2_psrl_q:
    case llvm::Intrinsic::x86_avx2_psra_w:
    case llvm::Intrinsic::x86_avx2_psra_d:
    case llvm::Intrinsic::x86_avx2_psrli_w:
    case llvm::Intrinsic::x86_avx2_psrli_d:
    case llvm::Intrinsic::x86_avx2_psrli_q:
    case llvm::Intrinsic::x86_avx2_psrai_w:
    case llvm::Intrinsic::x86_avx2_psrai_d:
    case llvm::Intrinsic::x86_sse2_psll_w:
    case llvm::Intrinsic::x86_sse2_psll_d:
    case llvm::Intrinsic::x86_sse2_psll_q:
    case llvm::Intrinsic::x86_sse2_pslli_w:
    case llvm::Intrinsic::x86_sse2_pslli_d:
    case llvm::Intrinsic::x86_sse2_pslli_q:
    case llvm::Intrinsic::x86_sse2_psrl_w:
    case llvm::Intrinsic::x86_sse2_psrl_d:
    case llvm::Intrinsic::x86_sse2_psrl_q:
    case llvm::Intrinsic::x86_sse2_psra_w:
    case llvm::Intrinsic::x86_sse2_psra_d:
    case llvm::Intrinsic::x86_sse2_psrli_w:
    case llvm::Intrinsic::x86_sse2_psrli_d:
    case llvm::Intrinsic::x86_sse2_psrli_q:
    case llvm::Intrinsic::x86_sse2_psrai_w:
    case llvm::Intrinsic::x86_sse2_psrai_d:
    case llvm::Intrinsic::x86_mmx_psll_w:
    case llvm::Intrinsic::x86_mmx_psll_d:
    case llvm::Intrinsic::x86_mmx_psll_q:
    case llvm::Intrinsic::x86_mmx_pslli_w:
    case llvm::Intrinsic::x86_mmx_pslli_d:
    case llvm::Intrinsic::x86_mmx_pslli_q:
    case llvm::Intrinsic::x86_mmx_psrl_w:
    case llvm::Intrinsic::x86_mmx_psrl_d:
    case llvm::Intrinsic::x86_mmx_psrl_q:
    case llvm::Intrinsic::x86_mmx_psra_w:
    case llvm::Intrinsic::x86_mmx_psra_d:
    case llvm::Intrinsic::x86_mmx_psrli_w:
    case llvm::Intrinsic::x86_mmx_psrli_d:
    case llvm::Intrinsic::x86_mmx_psrli_q:
    case llvm::Intrinsic::x86_mmx_psrai_w:
    case llvm::Intrinsic::x86_mmx_psrai_d:
      handleVectorShiftIntrinsic(I, /* Variable */ false);
      break;
    case llvm::Intrinsic::x86_avx2_psllv_d:
    case llvm::Intrinsic::x86_avx2_psllv_d_256:
    case llvm::Intrinsic::x86_avx2_psllv_q:
    case llvm::Intrinsic::x86_avx2_psllv_q_256:
    case llvm::Intrinsic::x86_avx2_psrlv_d:
    case llvm::Intrinsic::x86_avx2_psrlv_d_256:
    case llvm::Intrinsic::x86_avx2_psrlv_q:
    case llvm::Intrinsic::x86_avx2_psrlv_q_256:
    case llvm::Intrinsic::x86_avx2_psrav_d:
    case llvm::Intrinsic::x86_avx2_psrav_d_256:
      handleVectorShiftIntrinsic(I, /* Variable */ true);
      break;

    case llvm::Intrinsic::x86_sse2_packsswb_128:
    case llvm::Intrinsic::x86_sse2_packssdw_128:
    case llvm::Intrinsic::x86_sse2_packuswb_128:
    case llvm::Intrinsic::x86_sse41_packusdw:
    case llvm::Intrinsic::x86_avx2_packsswb:
    case llvm::Intrinsic::x86_avx2_packssdw:
    case llvm::Intrinsic::x86_avx2_packuswb:
    case llvm::Intrinsic::x86_avx2_packusdw:
      handleVectorPackIntrinsic(I);
      break;

    case llvm::Intrinsic::x86_mmx_packsswb:
    case llvm::Intrinsic::x86_mmx_packuswb:
      handleVectorPackIntrinsic(I, 16);
      break;

    case llvm::Intrinsic::x86_mmx_packssdw:
      handleVectorPackIntrinsic(I, 32);
      break;

    case llvm::Intrinsic::x86_mmx_psad_bw:
    case llvm::Intrinsic::x86_sse2_psad_bw:
    case llvm::Intrinsic::x86_avx2_psad_bw:
      handleVectorSadIntrinsic(I);
      break;

    case llvm::Intrinsic::x86_sse2_pmadd_wd:
    case llvm::Intrinsic::x86_avx2_pmadd_wd:
    case llvm::Intrinsic::x86_ssse3_pmadd_ub_sw_128:
    case llvm::Intrinsic::x86_avx2_pmadd_ub_sw:
      handleVectorPmaddIntrinsic(I);
      break;

    case llvm::Intrinsic::x86_ssse3_pmadd_ub_sw:
      handleVectorPmaddIntrinsic(I, 8);
      break;

    case llvm::Intrinsic::x86_mmx_pmadd_wd:
      handleVectorPmaddIntrinsic(I, 16);
      break;

    default:
      if (!handleUnknownIntrinsic(I))
        visitInstruction(I);
      break;
    }
  }

  void visitCallSite(CallSite CS) {
    Instruction &I = *CS.getInstruction();
    assert((CS.isCall() || CS.isInvoke()) && "Unknown type of CallSite");
    if (CS.isCall()) {
      CallInst *Call = cast<CallInst>(&I);

      // For inline asm, do the usual thing: check argument shadow and mark all
      // outputs as clean. Note that any side effects of the inline asm that are
      // not immediately visible in its constraints are not handled.
      if (Call->isInlineAsm()) {
        visitInstruction(I);
        return;
      }

      assert(!isa<IntrinsicInst>(&I) && "intrinsics are handled elsewhere");

      // We are going to insert code that relies on the fact that the callee
      // will become a non-readonly function after it is instrumented by us. To
      // prevent this code from being optimized out, mark that function
      // non-readonly in advance.
      if (Function *Func = Call->getCalledFunction()) {
        // Clear out readonly/readnone attributes.
        AttrBuilder B;
        B.addAttribute(Attribute::ReadOnly)
          .addAttribute(Attribute::ReadNone);
        Func->removeAttributes(AttributeSet::FunctionIndex,
                               AttributeSet::get(Func->getContext(),
                                                 AttributeSet::FunctionIndex,
                                                 B));
      }
    }
    IRBuilder<> IRB(&I);

    unsigned ArgOffset = 0;
    DEBUG(dbgs() << "  CallSite: " << I << "\n");
    for (CallSite::arg_iterator ArgIt = CS.arg_begin(), End = CS.arg_end();
         ArgIt != End; ++ArgIt) {
      Value *A = *ArgIt;
      unsigned i = ArgIt - CS.arg_begin();
      if (!A->getType()->isSized()) {
        DEBUG(dbgs() << "Arg " << i << " is not sized: " << I << "\n");
        continue;
      }
      unsigned Size = 0;
      Value *Store = nullptr;
      // Compute the Shadow for arg even if it is ByVal, because
      // in that case getShadow() will copy the actual arg shadow to
      // __msan_param_tls.
      Value *ArgShadow = getShadow(A);
      Value *ArgShadowBase = getShadowPtrForArgument(A, IRB, ArgOffset);
      DEBUG(dbgs() << "  Arg#" << i << ": " << *A <<
            " Shadow: " << *ArgShadow << "\n");
      bool ArgIsInitialized = false;
      const DataLayout &DL = F.getParent()->getDataLayout();
      if (CS.paramHasAttr(i + 1, Attribute::ByVal)) {
        assert(A->getType()->isPointerTy() &&
               "ByVal argument is not a pointer!");
        Size = DL.getTypeAllocSize(A->getType()->getPointerElementType());
        if (ArgOffset + Size > kParamTLSSize) break;
        unsigned ParamAlignment = CS.getParamAlignment(i + 1);
        unsigned Alignment = std::min(ParamAlignment, kShadowTLSAlignment);
        Store = IRB.CreateMemCpy(ArgShadowBase,
                                 getShadowPtr(A, Type::getInt8Ty(*MS.C), IRB),
                                 Size, Alignment);
      } else {
        Size = DL.getTypeAllocSize(A->getType());
        if (ArgOffset + Size > kParamTLSSize) break;
        Store = IRB.CreateAlignedStore(ArgShadow, ArgShadowBase,
                                       kShadowTLSAlignment);
        Constant *Cst = dyn_cast<Constant>(ArgShadow);
        if (Cst && Cst->isNullValue()) ArgIsInitialized = true;
      }
      if (MS.TrackOrigins && !ArgIsInitialized)
        IRB.CreateStore(getOrigin(A),
                        getOriginPtrForArgument(A, IRB, ArgOffset));
      (void)Store;
      assert(Size != 0 && Store != nullptr);
      DEBUG(dbgs() << "  Param:" << *Store << "\n");
      ArgOffset += RoundUpToAlignment(Size, 8);
    }
    DEBUG(dbgs() << "  done with call args\n");

    FunctionType *FT =
      cast<FunctionType>(CS.getCalledValue()->getType()->getContainedType(0));
    if (FT->isVarArg()) {
      VAHelper->visitCallSite(CS, IRB);
    }

    // Now, get the shadow for the RetVal.
    if (!I.getType()->isSized()) return;
    IRBuilder<> IRBBefore(&I);
    // Until we have full dynamic coverage, make sure the retval shadow is 0.
    Value *Base = getShadowPtrForRetval(&I, IRBBefore);
    IRBBefore.CreateAlignedStore(getCleanShadow(&I), Base, kShadowTLSAlignment);
    Instruction *NextInsn = nullptr;
    if (CS.isCall()) {
      NextInsn = I.getNextNode();
    } else {
      BasicBlock *NormalDest = cast<InvokeInst>(&I)->getNormalDest();
      if (!NormalDest->getSinglePredecessor()) {
        // FIXME: this case is tricky, so we are just conservative here.
        // Perhaps we need to split the edge between this BB and NormalDest,
        // but a naive attempt to use SplitEdge leads to a crash.
        setShadow(&I, getCleanShadow(&I));
        setOrigin(&I, getCleanOrigin());
        return;
      }
      NextInsn = NormalDest->getFirstInsertionPt();
      assert(NextInsn &&
             "Could not find insertion point for retval shadow load");
    }
    IRBuilder<> IRBAfter(NextInsn);
    Value *RetvalShadow =
      IRBAfter.CreateAlignedLoad(getShadowPtrForRetval(&I, IRBAfter),
                                 kShadowTLSAlignment, "_msret");
    setShadow(&I, RetvalShadow);
    if (MS.TrackOrigins)
      setOrigin(&I, IRBAfter.CreateLoad(getOriginPtrForRetval(IRBAfter)));
  }

  void visitReturnInst(ReturnInst &I) {
    IRBuilder<> IRB(&I);
    Value *RetVal = I.getReturnValue();
    if (!RetVal) return;
    Value *ShadowPtr = getShadowPtrForRetval(RetVal, IRB);
    if (CheckReturnValue) {
      insertShadowCheck(RetVal, &I);
      Value *Shadow = getCleanShadow(RetVal);
      IRB.CreateAlignedStore(Shadow, ShadowPtr, kShadowTLSAlignment);
    } else {
      Value *Shadow = getShadow(RetVal);
      IRB.CreateAlignedStore(Shadow, ShadowPtr, kShadowTLSAlignment);
      // FIXME: make it conditional if ClStoreCleanOrigin==0
      if (MS.TrackOrigins)
        IRB.CreateStore(getOrigin(RetVal), getOriginPtrForRetval(IRB));
    }
  }

  void visitPHINode(PHINode &I) {
    IRBuilder<> IRB(&I);
    if (!PropagateShadow) {
      setShadow(&I, getCleanShadow(&I));
      setOrigin(&I, getCleanOrigin());
      return;
    }

    ShadowPHINodes.push_back(&I);
    setShadow(&I, IRB.CreatePHI(getShadowTy(&I), I.getNumIncomingValues(),
                                "_msphi_s"));
    if (MS.TrackOrigins)
      setOrigin(&I, IRB.CreatePHI(MS.OriginTy, I.getNumIncomingValues(),
                                  "_msphi_o"));
  }

  void visitAllocaInst(AllocaInst &I) {
    setShadow(&I, getCleanShadow(&I));
    setOrigin(&I, getCleanOrigin());
    IRBuilder<> IRB(I.getNextNode());
    const DataLayout &DL = F.getParent()->getDataLayout();
    uint64_t Size = DL.getTypeAllocSize(I.getAllocatedType());
    if (PoisonStack && ClPoisonStackWithCall) {
      IRB.CreateCall(MS.MsanPoisonStackFn,
                     {IRB.CreatePointerCast(&I, IRB.getInt8PtrTy()),
                      ConstantInt::get(MS.IntptrTy, Size)});
    } else {
      Value *ShadowBase = getShadowPtr(&I, Type::getInt8PtrTy(*MS.C), IRB);
      Value *PoisonValue = IRB.getInt8(PoisonStack ? ClPoisonStackPattern : 0);
      IRB.CreateMemSet(ShadowBase, PoisonValue, Size, I.getAlignment());
    }

    if (PoisonStack && MS.TrackOrigins) {
      SmallString<2048> StackDescriptionStorage;
      raw_svector_ostream StackDescription(StackDescriptionStorage);
      // We create a string with a description of the stack allocation and
      // pass it into __msan_set_alloca_origin.
      // It will be printed by the run-time if stack-originated UMR is found.
      // The first 4 bytes of the string are set to '----' and will be replaced
      // by __msan_va_arg_overflow_size_tls at the first call.
      StackDescription << "----" << I.getName() << "@" << F.getName();
      Value *Descr =
          createPrivateNonConstGlobalForString(*F.getParent(),
                                               StackDescription.str());

      IRB.CreateCall(MS.MsanSetAllocaOrigin4Fn,
                     {IRB.CreatePointerCast(&I, IRB.getInt8PtrTy()),
                      ConstantInt::get(MS.IntptrTy, Size),
                      IRB.CreatePointerCast(Descr, IRB.getInt8PtrTy()),
                      IRB.CreatePointerCast(&F, MS.IntptrTy)});
    }
  }

  void visitSelectInst(SelectInst& I) {
    IRBuilder<> IRB(&I);
    // a = select b, c, d
    Value *B = I.getCondition();
    Value *C = I.getTrueValue();
    Value *D = I.getFalseValue();
    Value *Sb = getShadow(B);
    Value *Sc = getShadow(C);
    Value *Sd = getShadow(D);

    // Result shadow if condition shadow is 0.
    Value *Sa0 = IRB.CreateSelect(B, Sc, Sd);
    Value *Sa1;
    if (I.getType()->isAggregateType()) {
      // To avoid "sign extending" i1 to an arbitrary aggregate type, we just do
      // an extra "select". This results in much more compact IR.
      // Sa = select Sb, poisoned, (select b, Sc, Sd)
      Sa1 = getPoisonedShadow(getShadowTy(I.getType()));
    } else {
      // Sa = select Sb, [ (c^d) | Sc | Sd ], [ b ? Sc : Sd ]
      // If Sb (condition is poisoned), look for bits in c and d that are equal
      // and both unpoisoned.
      // If !Sb (condition is unpoisoned), simply pick one of Sc and Sd.

      // Cast arguments to shadow-compatible type.
      C = CreateAppToShadowCast(IRB, C);
      D = CreateAppToShadowCast(IRB, D);

      // Result shadow if condition shadow is 1.
      Sa1 = IRB.CreateOr(IRB.CreateXor(C, D), IRB.CreateOr(Sc, Sd));
    }
    Value *Sa = IRB.CreateSelect(Sb, Sa1, Sa0, "_msprop_select");
    setShadow(&I, Sa);
    if (MS.TrackOrigins) {
      // Origins are always i32, so any vector conditions must be flattened.
      // FIXME: consider tracking vector origins for app vectors?
      if (B->getType()->isVectorTy()) {
        Type *FlatTy = getShadowTyNoVec(B->getType());
        B = IRB.CreateICmpNE(IRB.CreateBitCast(B, FlatTy),
                                ConstantInt::getNullValue(FlatTy));
        Sb = IRB.CreateICmpNE(IRB.CreateBitCast(Sb, FlatTy),
                                      ConstantInt::getNullValue(FlatTy));
      }
      // a = select b, c, d
      // Oa = Sb ? Ob : (b ? Oc : Od)
      setOrigin(
          &I, IRB.CreateSelect(Sb, getOrigin(I.getCondition()),
                               IRB.CreateSelect(B, getOrigin(I.getTrueValue()),
                                                getOrigin(I.getFalseValue()))));
    }
  }

  void visitLandingPadInst(LandingPadInst &I) {
    // Do nothing.
    // See http://code.google.com/p/memory-sanitizer/issues/detail?id=1
    setShadow(&I, getCleanShadow(&I));
    setOrigin(&I, getCleanOrigin());
  }

  void visitGetElementPtrInst(GetElementPtrInst &I) {
    handleShadowOr(I);
  }

  void visitExtractValueInst(ExtractValueInst &I) {
    IRBuilder<> IRB(&I);
    Value *Agg = I.getAggregateOperand();
    DEBUG(dbgs() << "ExtractValue:  " << I << "\n");
    Value *AggShadow = getShadow(Agg);
    DEBUG(dbgs() << "   AggShadow:  " << *AggShadow << "\n");
    Value *ResShadow = IRB.CreateExtractValue(AggShadow, I.getIndices());
    DEBUG(dbgs() << "   ResShadow:  " << *ResShadow << "\n");
    setShadow(&I, ResShadow);
    setOriginForNaryOp(I);
  }

  void visitInsertValueInst(InsertValueInst &I) {
    IRBuilder<> IRB(&I);
    DEBUG(dbgs() << "InsertValue:  " << I << "\n");
    Value *AggShadow = getShadow(I.getAggregateOperand());
    Value *InsShadow = getShadow(I.getInsertedValueOperand());
    DEBUG(dbgs() << "   AggShadow:  " << *AggShadow << "\n");
    DEBUG(dbgs() << "   InsShadow:  " << *InsShadow << "\n");
    Value *Res = IRB.CreateInsertValue(AggShadow, InsShadow, I.getIndices());
    DEBUG(dbgs() << "   Res:        " << *Res << "\n");
    setShadow(&I, Res);
    setOriginForNaryOp(I);
  }

  void dumpInst(Instruction &I) {
    if (CallInst *CI = dyn_cast<CallInst>(&I)) {
      errs() << "ZZZ call " << CI->getCalledFunction()->getName() << "\n";
    } else {
      errs() << "ZZZ " << I.getOpcodeName() << "\n";
    }
    errs() << "QQQ " << I << "\n";
  }

  void visitResumeInst(ResumeInst &I) {
    DEBUG(dbgs() << "Resume: " << I << "\n");
    // Nothing to do here.
  }

  void visitInstruction(Instruction &I) {
    // Everything else: stop propagating and check for poisoned shadow.
    if (ClDumpStrictInstructions)
      dumpInst(I);
    DEBUG(dbgs() << "DEFAULT: " << I << "\n");
    for (size_t i = 0, n = I.getNumOperands(); i < n; i++)
      insertShadowCheck(I.getOperand(i), &I);
    setShadow(&I, getCleanShadow(&I));
    setOrigin(&I, getCleanOrigin());
  }
};

/// \brief AMD64-specific implementation of VarArgHelper.
struct VarArgAMD64Helper : public VarArgHelper {
  // An unfortunate workaround for asymmetric lowering of va_arg stuff.
  // See a comment in visitCallSite for more details.
  static const unsigned AMD64GpEndOffset = 48;  // AMD64 ABI Draft 0.99.6 p3.5.7
  static const unsigned AMD64FpEndOffset = 176;

  Function &F;
  MemorySanitizer &MS;
  MemorySanitizerVisitor &MSV;
  Value *VAArgTLSCopy;
  Value *VAArgOverflowSize;

  SmallVector<CallInst*, 16> VAStartInstrumentationList;

  VarArgAMD64Helper(Function &F, MemorySanitizer &MS,
                    MemorySanitizerVisitor &MSV)
    : F(F), MS(MS), MSV(MSV), VAArgTLSCopy(nullptr),
      VAArgOverflowSize(nullptr) {}

  enum ArgKind { AK_GeneralPurpose, AK_FloatingPoint, AK_Memory };

  ArgKind classifyArgument(Value* arg) {
    // A very rough approximation of X86_64 argument classification rules.
    Type *T = arg->getType();
    if (T->isFPOrFPVectorTy() || T->isX86_MMXTy())
      return AK_FloatingPoint;
    if (T->isIntegerTy() && T->getPrimitiveSizeInBits() <= 64)
      return AK_GeneralPurpose;
    if (T->isPointerTy())
      return AK_GeneralPurpose;
    return AK_Memory;
  }

  // For VarArg functions, store the argument shadow in an ABI-specific format
  // that corresponds to va_list layout.
  // We do this because Clang lowers va_arg in the frontend, and this pass
  // only sees the low level code that deals with va_list internals.
  // A much easier alternative (provided that Clang emits va_arg instructions)
  // would have been to associate each live instance of va_list with a copy of
  // MSanParamTLS, and extract shadow on va_arg() call in the argument list
  // order.
  void visitCallSite(CallSite &CS, IRBuilder<> &IRB) override {
    unsigned GpOffset = 0;
    unsigned FpOffset = AMD64GpEndOffset;
    unsigned OverflowOffset = AMD64FpEndOffset;
    const DataLayout &DL = F.getParent()->getDataLayout();
    for (CallSite::arg_iterator ArgIt = CS.arg_begin(), End = CS.arg_end();
         ArgIt != End; ++ArgIt) {
      Value *A = *ArgIt;
      unsigned ArgNo = CS.getArgumentNo(ArgIt);
      bool IsByVal = CS.paramHasAttr(ArgNo + 1, Attribute::ByVal);
      if (IsByVal) {
        // ByVal arguments always go to the overflow area.
        assert(A->getType()->isPointerTy());
        Type *RealTy = A->getType()->getPointerElementType();
        uint64_t ArgSize = DL.getTypeAllocSize(RealTy);
        Value *Base = getShadowPtrForVAArgument(RealTy, IRB, OverflowOffset);
        OverflowOffset += RoundUpToAlignment(ArgSize, 8);
        IRB.CreateMemCpy(Base, MSV.getShadowPtr(A, IRB.getInt8Ty(), IRB),
                         ArgSize, kShadowTLSAlignment);
      } else {
        ArgKind AK = classifyArgument(A);
        if (AK == AK_GeneralPurpose && GpOffset >= AMD64GpEndOffset)
          AK = AK_Memory;
        if (AK == AK_FloatingPoint && FpOffset >= AMD64FpEndOffset)
          AK = AK_Memory;
        Value *Base;
        switch (AK) {
          case AK_GeneralPurpose:
            Base = getShadowPtrForVAArgument(A->getType(), IRB, GpOffset);
            GpOffset += 8;
            break;
          case AK_FloatingPoint:
            Base = getShadowPtrForVAArgument(A->getType(), IRB, FpOffset);
            FpOffset += 16;
            break;
          default: assert(AK == AK_Memory); // HLSL Change - set as default case rather than case AK_Memory
            uint64_t ArgSize = DL.getTypeAllocSize(A->getType());
            Base = getShadowPtrForVAArgument(A->getType(), IRB, OverflowOffset);
            OverflowOffset += RoundUpToAlignment(ArgSize, 8);
        }
        IRB.CreateAlignedStore(MSV.getShadow(A), Base, kShadowTLSAlignment);
      }
    }
    Constant *OverflowSize =
      ConstantInt::get(IRB.getInt64Ty(), OverflowOffset - AMD64FpEndOffset);
    IRB.CreateStore(OverflowSize, MS.VAArgOverflowSizeTLS);
  }

  /// \brief Compute the shadow address for a given va_arg.
  Value *getShadowPtrForVAArgument(Type *Ty, IRBuilder<> &IRB,
                                   int ArgOffset) {
    Value *Base = IRB.CreatePointerCast(MS.VAArgTLS, MS.IntptrTy);
    Base = IRB.CreateAdd(Base, ConstantInt::get(MS.IntptrTy, ArgOffset));
    return IRB.CreateIntToPtr(Base, PointerType::get(MSV.getShadowTy(Ty), 0),
                              "_msarg");
  }

  void visitVAStartInst(VAStartInst &I) override {
    IRBuilder<> IRB(&I);
    VAStartInstrumentationList.push_back(&I);
    Value *VAListTag = I.getArgOperand(0);
    Value *ShadowPtr = MSV.getShadowPtr(VAListTag, IRB.getInt8Ty(), IRB);

    // Unpoison the whole __va_list_tag.
    // FIXME: magic ABI constants.
    IRB.CreateMemSet(ShadowPtr, Constant::getNullValue(IRB.getInt8Ty()),
                     /* size */24, /* alignment */8, false);
  }

  void visitVACopyInst(VACopyInst &I) override {
    IRBuilder<> IRB(&I);
    Value *VAListTag = I.getArgOperand(0);
    Value *ShadowPtr = MSV.getShadowPtr(VAListTag, IRB.getInt8Ty(), IRB);

    // Unpoison the whole __va_list_tag.
    // FIXME: magic ABI constants.
    IRB.CreateMemSet(ShadowPtr, Constant::getNullValue(IRB.getInt8Ty()),
                     /* size */24, /* alignment */8, false);
  }

  void finalizeInstrumentation() override {
    assert(!VAArgOverflowSize && !VAArgTLSCopy &&
           "finalizeInstrumentation called twice");
    if (!VAStartInstrumentationList.empty()) {
      // If there is a va_start in this function, make a backup copy of
      // va_arg_tls somewhere in the function entry block.
      IRBuilder<> IRB(F.getEntryBlock().getFirstNonPHI());
      VAArgOverflowSize = IRB.CreateLoad(MS.VAArgOverflowSizeTLS);
      Value *CopySize =
        IRB.CreateAdd(ConstantInt::get(MS.IntptrTy, AMD64FpEndOffset),
                      VAArgOverflowSize);
      VAArgTLSCopy = IRB.CreateAlloca(Type::getInt8Ty(*MS.C), CopySize);
      IRB.CreateMemCpy(VAArgTLSCopy, MS.VAArgTLS, CopySize, 8);
    }

    // Instrument va_start.
    // Copy va_list shadow from the backup copy of the TLS contents.
    for (size_t i = 0, n = VAStartInstrumentationList.size(); i < n; i++) {
      CallInst *OrigInst = VAStartInstrumentationList[i];
      IRBuilder<> IRB(OrigInst->getNextNode());
      Value *VAListTag = OrigInst->getArgOperand(0);

      Value *RegSaveAreaPtrPtr =
        IRB.CreateIntToPtr(
          IRB.CreateAdd(IRB.CreatePtrToInt(VAListTag, MS.IntptrTy),
                        ConstantInt::get(MS.IntptrTy, 16)),
          Type::getInt64PtrTy(*MS.C));
      Value *RegSaveAreaPtr = IRB.CreateLoad(RegSaveAreaPtrPtr);
      Value *RegSaveAreaShadowPtr =
        MSV.getShadowPtr(RegSaveAreaPtr, IRB.getInt8Ty(), IRB);
      IRB.CreateMemCpy(RegSaveAreaShadowPtr, VAArgTLSCopy,
                       AMD64FpEndOffset, 16);

      Value *OverflowArgAreaPtrPtr =
        IRB.CreateIntToPtr(
          IRB.CreateAdd(IRB.CreatePtrToInt(VAListTag, MS.IntptrTy),
                        ConstantInt::get(MS.IntptrTy, 8)),
          Type::getInt64PtrTy(*MS.C));
      Value *OverflowArgAreaPtr = IRB.CreateLoad(OverflowArgAreaPtrPtr);
      Value *OverflowArgAreaShadowPtr =
        MSV.getShadowPtr(OverflowArgAreaPtr, IRB.getInt8Ty(), IRB);
      Value *SrcPtr = IRB.CreateConstGEP1_32(IRB.getInt8Ty(), VAArgTLSCopy,
                                             AMD64FpEndOffset);
      IRB.CreateMemCpy(OverflowArgAreaShadowPtr, SrcPtr, VAArgOverflowSize, 16);
    }
  }
};

/// \brief MIPS64-specific implementation of VarArgHelper.
struct VarArgMIPS64Helper : public VarArgHelper {
  Function &F;
  MemorySanitizer &MS;
  MemorySanitizerVisitor &MSV;
  Value *VAArgTLSCopy;
  Value *VAArgSize;

  SmallVector<CallInst*, 16> VAStartInstrumentationList;

  VarArgMIPS64Helper(Function &F, MemorySanitizer &MS,
                    MemorySanitizerVisitor &MSV)
    : F(F), MS(MS), MSV(MSV), VAArgTLSCopy(nullptr),
      VAArgSize(nullptr) {}

  void visitCallSite(CallSite &CS, IRBuilder<> &IRB) override {
    unsigned VAArgOffset = 0;
    const DataLayout &DL = F.getParent()->getDataLayout();
    for (CallSite::arg_iterator ArgIt = CS.arg_begin() + 1, End = CS.arg_end();
         ArgIt != End; ++ArgIt) {
      Value *A = *ArgIt;
      Value *Base;
      uint64_t ArgSize = DL.getTypeAllocSize(A->getType());
#if defined(__MIPSEB__) || defined(MIPSEB)
      // Adjusting the shadow for argument with size < 8 to match the placement
      // of bits in big endian system
      if (ArgSize < 8)
        VAArgOffset += (8 - ArgSize);
#endif
      Base = getShadowPtrForVAArgument(A->getType(), IRB, VAArgOffset);
      VAArgOffset += ArgSize;
      VAArgOffset = RoundUpToAlignment(VAArgOffset, 8);
      IRB.CreateAlignedStore(MSV.getShadow(A), Base, kShadowTLSAlignment);
    }

    Constant *TotalVAArgSize = ConstantInt::get(IRB.getInt64Ty(), VAArgOffset);
    // Here using VAArgOverflowSizeTLS as VAArgSizeTLS to avoid creation of
    // a new class member i.e. it is the total size of all VarArgs.
    IRB.CreateStore(TotalVAArgSize, MS.VAArgOverflowSizeTLS);
  }

  /// \brief Compute the shadow address for a given va_arg.
  Value *getShadowPtrForVAArgument(Type *Ty, IRBuilder<> &IRB,
                                   int ArgOffset) {
    Value *Base = IRB.CreatePointerCast(MS.VAArgTLS, MS.IntptrTy);
    Base = IRB.CreateAdd(Base, ConstantInt::get(MS.IntptrTy, ArgOffset));
    return IRB.CreateIntToPtr(Base, PointerType::get(MSV.getShadowTy(Ty), 0),
                              "_msarg");
  }

  void visitVAStartInst(VAStartInst &I) override {
    IRBuilder<> IRB(&I);
    VAStartInstrumentationList.push_back(&I);
    Value *VAListTag = I.getArgOperand(0);
    Value *ShadowPtr = MSV.getShadowPtr(VAListTag, IRB.getInt8Ty(), IRB);
    IRB.CreateMemSet(ShadowPtr, Constant::getNullValue(IRB.getInt8Ty()),
                     /* size */8, /* alignment */8, false);
  }

  void visitVACopyInst(VACopyInst &I) override {
    IRBuilder<> IRB(&I);
    Value *VAListTag = I.getArgOperand(0);
    Value *ShadowPtr = MSV.getShadowPtr(VAListTag, IRB.getInt8Ty(), IRB);
    // Unpoison the whole __va_list_tag.
    // FIXME: magic ABI constants.
    IRB.CreateMemSet(ShadowPtr, Constant::getNullValue(IRB.getInt8Ty()),
                     /* size */8, /* alignment */8, false);
  }

  void finalizeInstrumentation() override {
    assert(!VAArgSize && !VAArgTLSCopy &&
           "finalizeInstrumentation called twice");
    IRBuilder<> IRB(F.getEntryBlock().getFirstNonPHI());
    VAArgSize = IRB.CreateLoad(MS.VAArgOverflowSizeTLS);
    Value *CopySize = IRB.CreateAdd(ConstantInt::get(MS.IntptrTy, 0),
                                    VAArgSize);

    if (!VAStartInstrumentationList.empty()) {
      // If there is a va_start in this function, make a backup copy of
      // va_arg_tls somewhere in the function entry block.
      VAArgTLSCopy = IRB.CreateAlloca(Type::getInt8Ty(*MS.C), CopySize);
      IRB.CreateMemCpy(VAArgTLSCopy, MS.VAArgTLS, CopySize, 8);
    }

    // Instrument va_start.
    // Copy va_list shadow from the backup copy of the TLS contents.
    for (size_t i = 0, n = VAStartInstrumentationList.size(); i < n; i++) {
      CallInst *OrigInst = VAStartInstrumentationList[i];
      IRBuilder<> IRB(OrigInst->getNextNode());
      Value *VAListTag = OrigInst->getArgOperand(0);
      Value *RegSaveAreaPtrPtr =
        IRB.CreateIntToPtr(IRB.CreatePtrToInt(VAListTag, MS.IntptrTy),
                        Type::getInt64PtrTy(*MS.C));
      Value *RegSaveAreaPtr = IRB.CreateLoad(RegSaveAreaPtrPtr);
      Value *RegSaveAreaShadowPtr =
      MSV.getShadowPtr(RegSaveAreaPtr, IRB.getInt8Ty(), IRB);
      IRB.CreateMemCpy(RegSaveAreaShadowPtr, VAArgTLSCopy, CopySize, 8);
    }
  }
};

/// \brief A no-op implementation of VarArgHelper.
struct VarArgNoOpHelper : public VarArgHelper {
  VarArgNoOpHelper(Function &F, MemorySanitizer &MS,
                   MemorySanitizerVisitor &MSV) {}

  void visitCallSite(CallSite &CS, IRBuilder<> &IRB) override {}

  void visitVAStartInst(VAStartInst &I) override {}

  void visitVACopyInst(VACopyInst &I) override {}

  void finalizeInstrumentation() override {}
};

VarArgHelper *CreateVarArgHelper(Function &Func, MemorySanitizer &Msan,
                                 MemorySanitizerVisitor &Visitor) {
  // VarArg handling is only implemented on AMD64. False positives are possible
  // on other platforms.
  llvm::Triple TargetTriple(Func.getParent()->getTargetTriple());
  if (TargetTriple.getArch() == llvm::Triple::x86_64)
    return new VarArgAMD64Helper(Func, Msan, Visitor);
  else if (TargetTriple.getArch() == llvm::Triple::mips64 ||
           TargetTriple.getArch() == llvm::Triple::mips64el)
    return new VarArgMIPS64Helper(Func, Msan, Visitor);
  else
    return new VarArgNoOpHelper(Func, Msan, Visitor);
}

}  // namespace

bool MemorySanitizer::runOnFunction(Function &F) {
  if (&F == MsanCtorFunction)
    return false;
  MemorySanitizerVisitor Visitor(F, *this);

  // Clear out readonly/readnone attributes.
  AttrBuilder B;
  B.addAttribute(Attribute::ReadOnly)
    .addAttribute(Attribute::ReadNone);
  F.removeAttributes(AttributeSet::FunctionIndex,
                     AttributeSet::get(F.getContext(),
                                       AttributeSet::FunctionIndex, B));

  return Visitor.runOnFunction();
}
