//===-- SanitizerCoverage.cpp - coverage instrumentation for sanitizers ---===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Coverage instrumentation that works with AddressSanitizer
// and potentially with other Sanitizers.
//
// We create a Guard variable with the same linkage
// as the function and inject this code into the entry block (SCK_Function)
// or all blocks (SCK_BB):
// if (Guard < 0) {
//    __sanitizer_cov(&Guard);
// }
// The accesses to Guard are atomic. The rest of the logic is
// in __sanitizer_cov (it's fine to call it more than once).
//
// With SCK_Edge we also split critical edges this effectively
// instrumenting all edges.
//
// This coverage implementation provides very limited data:
// it only tells if a given function (block) was ever executed. No counters.
// But for many use cases this is what we need and the added slowdown small.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "sancov"

static const char *const kSanCovModuleInitName = "__sanitizer_cov_module_init";
static const char *const kSanCovName = "__sanitizer_cov";
static const char *const kSanCovWithCheckName = "__sanitizer_cov_with_check";
static const char *const kSanCovIndirCallName = "__sanitizer_cov_indir_call16";
static const char *const kSanCovTraceEnter = "__sanitizer_cov_trace_func_enter";
static const char *const kSanCovTraceBB = "__sanitizer_cov_trace_basic_block";
static const char *const kSanCovTraceCmp = "__sanitizer_cov_trace_cmp";
static const char *const kSanCovModuleCtorName = "sancov.module_ctor";
static const uint64_t    kSanCtorAndDtorPriority = 2;

static cl::opt<int> ClCoverageLevel("sanitizer-coverage-level",
       cl::desc("Sanitizer Coverage. 0: none, 1: entry block, 2: all blocks, "
                "3: all blocks and critical edges, "
                "4: above plus indirect calls"),
       cl::Hidden, cl::init(0));

static cl::opt<unsigned> ClCoverageBlockThreshold(
    "sanitizer-coverage-block-threshold",
    cl::desc("Use a callback with a guard check inside it if there are"
             " more than this number of blocks."),
    cl::Hidden, cl::init(500));

static cl::opt<bool>
    ClExperimentalTracing("sanitizer-coverage-experimental-tracing",
                          cl::desc("Experimental basic-block tracing: insert "
                                   "callbacks at every basic block"),
                          cl::Hidden, cl::init(false));

static cl::opt<bool>
    ClExperimentalCMPTracing("sanitizer-coverage-experimental-trace-compares",
                             cl::desc("Experimental tracing of CMP and similar "
                                      "instructions"),
                             cl::Hidden, cl::init(false));

// Experimental 8-bit counters used as an additional search heuristic during
// coverage-guided fuzzing.
// The counters are not thread-friendly:
//   - contention on these counters may cause significant slowdown;
//   - the counter updates are racy and the results may be inaccurate.
// They are also inaccurate due to 8-bit integer overflow.
static cl::opt<bool> ClUse8bitCounters("sanitizer-coverage-8bit-counters",
                                       cl::desc("Experimental 8-bit counters"),
                                       cl::Hidden, cl::init(false));

namespace {

SanitizerCoverageOptions getOptions(int LegacyCoverageLevel) {
  SanitizerCoverageOptions Res;
  switch (LegacyCoverageLevel) {
  case 0:
    Res.CoverageType = SanitizerCoverageOptions::SCK_None;
    break;
  case 1:
    Res.CoverageType = SanitizerCoverageOptions::SCK_Function;
    break;
  case 2:
    Res.CoverageType = SanitizerCoverageOptions::SCK_BB;
    break;
  case 3:
    Res.CoverageType = SanitizerCoverageOptions::SCK_Edge;
    break;
  case 4:
    Res.CoverageType = SanitizerCoverageOptions::SCK_Edge;
    Res.IndirectCalls = true;
    break;
  }
  return Res;
}

SanitizerCoverageOptions OverrideFromCL(SanitizerCoverageOptions Options) {
  // Sets CoverageType and IndirectCalls.
  SanitizerCoverageOptions CLOpts = getOptions(ClCoverageLevel);
  Options.CoverageType = std::max(Options.CoverageType, CLOpts.CoverageType);
  Options.IndirectCalls |= CLOpts.IndirectCalls;
  Options.TraceBB |= ClExperimentalTracing;
  Options.TraceCmp |= ClExperimentalCMPTracing;
  Options.Use8bitCounters |= ClUse8bitCounters;
  return Options;
}

class SanitizerCoverageModule : public ModulePass {
 public:
  SanitizerCoverageModule(
      const SanitizerCoverageOptions &Options = SanitizerCoverageOptions())
      : ModulePass(ID), Options(OverrideFromCL(Options)) {}
  bool runOnModule(Module &M) override;
  bool runOnFunction(Function &F);
  static char ID;  // Pass identification, replacement for typeid
  StringRef getPassName() const override {
    return "SanitizerCoverageModule";
  }

 private:
  void InjectCoverageForIndirectCalls(Function &F,
                                      ArrayRef<Instruction *> IndirCalls);
  void InjectTraceForCmp(Function &F, ArrayRef<Instruction *> CmpTraceTargets);
  bool InjectCoverage(Function &F, ArrayRef<BasicBlock *> AllBlocks);
  void SetNoSanitizeMetadata(Instruction *I);
  void InjectCoverageAtBlock(Function &F, BasicBlock &BB, bool UseCalls);
  unsigned NumberOfInstrumentedBlocks() {
    return SanCovFunction->getNumUses() + SanCovWithCheckFunction->getNumUses();
  }
  Function *SanCovFunction;
  Function *SanCovWithCheckFunction;
  Function *SanCovIndirCallFunction;
  Function *SanCovTraceEnter, *SanCovTraceBB;
  Function *SanCovTraceCmpFunction;
  InlineAsm *EmptyAsm;
  Type *IntptrTy, *Int64Ty;
  LLVMContext *C;
  const DataLayout *DL;

  GlobalVariable *GuardArray;
  GlobalVariable *EightBitCounterArray;

  SanitizerCoverageOptions Options;
};

}  // namespace

bool SanitizerCoverageModule::runOnModule(Module &M) {
  if (Options.CoverageType == SanitizerCoverageOptions::SCK_None)
    return false;
  C = &(M.getContext());
  DL = &M.getDataLayout();
  IntptrTy = Type::getIntNTy(*C, DL->getPointerSizeInBits());
  Type *VoidTy = Type::getVoidTy(*C);
  IRBuilder<> IRB(*C);
  Type *Int8PtrTy = PointerType::getUnqual(IRB.getInt8Ty());
  Type *Int32PtrTy = PointerType::getUnqual(IRB.getInt32Ty());
  Int64Ty = IRB.getInt64Ty();

  SanCovFunction = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction(kSanCovName, VoidTy, Int32PtrTy, nullptr));
  SanCovWithCheckFunction = checkSanitizerInterfaceFunction(
      M.getOrInsertFunction(kSanCovWithCheckName, VoidTy, Int32PtrTy, nullptr));
  SanCovIndirCallFunction =
      checkSanitizerInterfaceFunction(M.getOrInsertFunction(
          kSanCovIndirCallName, VoidTy, IntptrTy, IntptrTy, nullptr));
  SanCovTraceCmpFunction =
      checkSanitizerInterfaceFunction(M.getOrInsertFunction(
          kSanCovTraceCmp, VoidTy, Int64Ty, Int64Ty, Int64Ty, nullptr));

  // We insert an empty inline asm after cov callbacks to avoid callback merge.
  EmptyAsm = InlineAsm::get(FunctionType::get(IRB.getVoidTy(), false),
                            StringRef(""), StringRef(""),
                            /*hasSideEffects=*/true);

  if (Options.TraceBB) {
    SanCovTraceEnter = checkSanitizerInterfaceFunction(
        M.getOrInsertFunction(kSanCovTraceEnter, VoidTy, Int32PtrTy, nullptr));
    SanCovTraceBB = checkSanitizerInterfaceFunction(
        M.getOrInsertFunction(kSanCovTraceBB, VoidTy, Int32PtrTy, nullptr));
  }

  // At this point we create a dummy array of guards because we don't
  // know how many elements we will need.
  Type *Int32Ty = IRB.getInt32Ty();
  Type *Int8Ty = IRB.getInt8Ty();

  GuardArray =
      new GlobalVariable(M, Int32Ty, false, GlobalValue::ExternalLinkage,
                         nullptr, "__sancov_gen_cov_tmp");
  if (Options.Use8bitCounters)
    EightBitCounterArray =
        new GlobalVariable(M, Int8Ty, false, GlobalVariable::ExternalLinkage,
                           nullptr, "__sancov_gen_cov_tmp");

  for (auto &F : M)
    runOnFunction(F);

  auto N = NumberOfInstrumentedBlocks();

  // Now we know how many elements we need. Create an array of guards
  // with one extra element at the beginning for the size.
  Type *Int32ArrayNTy = ArrayType::get(Int32Ty, N + 1);
  GlobalVariable *RealGuardArray = new GlobalVariable(
      M, Int32ArrayNTy, false, GlobalValue::PrivateLinkage,
      Constant::getNullValue(Int32ArrayNTy), "__sancov_gen_cov");


  // Replace the dummy array with the real one.
  GuardArray->replaceAllUsesWith(
      IRB.CreatePointerCast(RealGuardArray, Int32PtrTy));
  GuardArray->eraseFromParent();

  GlobalVariable *RealEightBitCounterArray;
  if (Options.Use8bitCounters) {
    // Make sure the array is 16-aligned.
    static const int kCounterAlignment = 16;
    Type *Int8ArrayNTy =
        ArrayType::get(Int8Ty, RoundUpToAlignment(N, kCounterAlignment));
    RealEightBitCounterArray = new GlobalVariable(
        M, Int8ArrayNTy, false, GlobalValue::PrivateLinkage,
        Constant::getNullValue(Int8ArrayNTy), "__sancov_gen_cov_counter");
    RealEightBitCounterArray->setAlignment(kCounterAlignment);
    EightBitCounterArray->replaceAllUsesWith(
        IRB.CreatePointerCast(RealEightBitCounterArray, Int8PtrTy));
    EightBitCounterArray->eraseFromParent();
  }

  // Create variable for module (compilation unit) name
  Constant *ModNameStrConst =
      ConstantDataArray::getString(M.getContext(), M.getName(), true);
  GlobalVariable *ModuleName =
      new GlobalVariable(M, ModNameStrConst->getType(), true,
                         GlobalValue::PrivateLinkage, ModNameStrConst);

  Function *CtorFunc;
  std::tie(CtorFunc, std::ignore) = createSanitizerCtorAndInitFunctions(
      M, kSanCovModuleCtorName, kSanCovModuleInitName,
      {Int32PtrTy, IntptrTy, Int8PtrTy, Int8PtrTy},
      {IRB.CreatePointerCast(RealGuardArray, Int32PtrTy),
       ConstantInt::get(IntptrTy, N),
       Options.Use8bitCounters
           ? IRB.CreatePointerCast(RealEightBitCounterArray, Int8PtrTy)
           : Constant::getNullValue(Int8PtrTy),
       IRB.CreatePointerCast(ModuleName, Int8PtrTy)});

  appendToGlobalCtors(M, CtorFunc, kSanCtorAndDtorPriority);

  return true;
}

bool SanitizerCoverageModule::runOnFunction(Function &F) {
  if (F.empty()) return false;
  if (F.getName().find(".module_ctor") != std::string::npos)
    return false;  // Should not instrument sanitizer init functions.
  if (Options.CoverageType >= SanitizerCoverageOptions::SCK_Edge)
    SplitAllCriticalEdges(F);
  SmallVector<Instruction*, 8> IndirCalls;
  SmallVector<BasicBlock*, 16> AllBlocks;
  SmallVector<Instruction*, 8> CmpTraceTargets;
  for (auto &BB : F) {
    AllBlocks.push_back(&BB);
    for (auto &Inst : BB) {
      if (Options.IndirectCalls) {
        CallSite CS(&Inst);
        if (CS && !CS.getCalledFunction())
          IndirCalls.push_back(&Inst);
      }
      if (Options.TraceCmp && isa<ICmpInst>(&Inst))
        CmpTraceTargets.push_back(&Inst);
    }
  }
  InjectCoverage(F, AllBlocks);
  InjectCoverageForIndirectCalls(F, IndirCalls);
  InjectTraceForCmp(F, CmpTraceTargets);
  return true;
}

bool SanitizerCoverageModule::InjectCoverage(Function &F,
                                             ArrayRef<BasicBlock *> AllBlocks) {
  switch (Options.CoverageType) {
  case SanitizerCoverageOptions::SCK_None:
    return false;
  case SanitizerCoverageOptions::SCK_Function:
    InjectCoverageAtBlock(F, F.getEntryBlock(), false);
    return true;
  default: {
    bool UseCalls = ClCoverageBlockThreshold < AllBlocks.size();
    for (auto BB : AllBlocks)
      InjectCoverageAtBlock(F, *BB, UseCalls);
    return true;
  }
  }
}

// On every indirect call we call a run-time function
// __sanitizer_cov_indir_call* with two parameters:
//   - callee address,
//   - global cache array that contains kCacheSize pointers (zero-initialized).
//     The cache is used to speed up recording the caller-callee pairs.
// The address of the caller is passed implicitly via caller PC.
// kCacheSize is encoded in the name of the run-time function.
void SanitizerCoverageModule::InjectCoverageForIndirectCalls(
    Function &F, ArrayRef<Instruction *> IndirCalls) {
  if (IndirCalls.empty()) return;
  const int kCacheSize = 16;
  const int kCacheAlignment = 64;  // Align for better performance.
  Type *Ty = ArrayType::get(IntptrTy, kCacheSize);
  for (auto I : IndirCalls) {
    IRBuilder<> IRB(I);
    CallSite CS(I);
    Value *Callee = CS.getCalledValue();
    if (isa<InlineAsm>(Callee)) continue;
    GlobalVariable *CalleeCache = new GlobalVariable(
        *F.getParent(), Ty, false, GlobalValue::PrivateLinkage,
        Constant::getNullValue(Ty), "__sancov_gen_callee_cache");
    CalleeCache->setAlignment(kCacheAlignment);
    IRB.CreateCall(SanCovIndirCallFunction,
                   {IRB.CreatePointerCast(Callee, IntptrTy),
                    IRB.CreatePointerCast(CalleeCache, IntptrTy)});
  }
}

void SanitizerCoverageModule::InjectTraceForCmp(
    Function &F, ArrayRef<Instruction *> CmpTraceTargets) {
  for (auto I : CmpTraceTargets) {
    if (ICmpInst *ICMP = dyn_cast<ICmpInst>(I)) {
      IRBuilder<> IRB(ICMP);
      Value *A0 = ICMP->getOperand(0);
      Value *A1 = ICMP->getOperand(1);
      if (!A0->getType()->isIntegerTy()) continue;
      uint64_t TypeSize = DL->getTypeStoreSizeInBits(A0->getType());
      // __sanitizer_cov_trace_cmp((type_size << 32) | predicate, A0, A1);
      IRB.CreateCall(
          SanCovTraceCmpFunction,
          {ConstantInt::get(Int64Ty, (TypeSize << 32) | ICMP->getPredicate()),
           IRB.CreateIntCast(A0, Int64Ty, true),
           IRB.CreateIntCast(A1, Int64Ty, true)});
    }
  }
}

void SanitizerCoverageModule::SetNoSanitizeMetadata(Instruction *I) {
  I->setMetadata(
      I->getParent()->getParent()->getParent()->getMDKindID("nosanitize"),
      MDNode::get(*C, None));
}

void SanitizerCoverageModule::InjectCoverageAtBlock(Function &F, BasicBlock &BB,
                                                    bool UseCalls) {
  // Don't insert coverage for unreachable blocks: we will never call
  // __sanitizer_cov() for them, so counting them in
  // NumberOfInstrumentedBlocks() might complicate calculation of code coverage
  // percentage. Also, unreachable instructions frequently have no debug
  // locations.
  if (isa<UnreachableInst>(BB.getTerminator()))
    return;
  BasicBlock::iterator IP = BB.getFirstInsertionPt(), BE = BB.end();
  // Skip static allocas at the top of the entry block so they don't become
  // dynamic when we split the block.  If we used our optimized stack layout,
  // then there will only be one alloca and it will come first.
  for (; IP != BE; ++IP) {
    AllocaInst *AI = dyn_cast<AllocaInst>(IP);
    if (!AI || !AI->isStaticAlloca())
      break;
  }

  bool IsEntryBB = &BB == &F.getEntryBlock();
  DebugLoc EntryLoc;
  if (IsEntryBB) {
    if (auto SP = getDISubprogram(&F))
      EntryLoc = DebugLoc::get(SP->getScopeLine(), 0, SP);
  } else {
    EntryLoc = IP->getDebugLoc();
  }

  IRBuilder<> IRB(IP);
  IRB.SetCurrentDebugLocation(EntryLoc);
  SmallVector<Value *, 1> Indices;
  Value *GuardP = IRB.CreateAdd(
      IRB.CreatePointerCast(GuardArray, IntptrTy),
      ConstantInt::get(IntptrTy, (1 + NumberOfInstrumentedBlocks()) * 4));
  Type *Int32PtrTy = PointerType::getUnqual(IRB.getInt32Ty());
  GuardP = IRB.CreateIntToPtr(GuardP, Int32PtrTy);
  if (UseCalls) {
    IRB.CreateCall(SanCovWithCheckFunction, GuardP);
  } else {
    LoadInst *Load = IRB.CreateLoad(GuardP);
    Load->setAtomic(Monotonic);
    Load->setAlignment(4);
    SetNoSanitizeMetadata(Load);
    Value *Cmp = IRB.CreateICmpSGE(Constant::getNullValue(Load->getType()), Load);
    Instruction *Ins = SplitBlockAndInsertIfThen(
        Cmp, IP, false, MDBuilder(*C).createBranchWeights(1, 100000));
    IRB.SetInsertPoint(Ins);
    IRB.SetCurrentDebugLocation(EntryLoc);
    // __sanitizer_cov gets the PC of the instruction using GET_CALLER_PC.
    IRB.CreateCall(SanCovFunction, GuardP);
    IRB.CreateCall(EmptyAsm, {}); // Avoids callback merge.
  }

  if (Options.Use8bitCounters) {
    IRB.SetInsertPoint(IP);
    Value *P = IRB.CreateAdd(
        IRB.CreatePointerCast(EightBitCounterArray, IntptrTy),
        ConstantInt::get(IntptrTy, NumberOfInstrumentedBlocks() - 1));
    P = IRB.CreateIntToPtr(P, IRB.getInt8PtrTy());
    LoadInst *LI = IRB.CreateLoad(P);
    Value *Inc = IRB.CreateAdd(LI, ConstantInt::get(IRB.getInt8Ty(), 1));
    StoreInst *SI = IRB.CreateStore(Inc, P);
    SetNoSanitizeMetadata(LI);
    SetNoSanitizeMetadata(SI);
  }

  if (Options.TraceBB) {
    // Experimental support for tracing.
    // Insert a callback with the same guard variable as used for coverage.
    IRB.SetInsertPoint(IP);
    IRB.CreateCall(IsEntryBB ? SanCovTraceEnter : SanCovTraceBB, GuardP);
  }
}

char SanitizerCoverageModule::ID = 0;
INITIALIZE_PASS(SanitizerCoverageModule, "sancov",
    "SanitizerCoverage: TODO."
    "ModulePass", false, false)
ModulePass *llvm::createSanitizerCoverageModulePass(
    const SanitizerCoverageOptions &Options) {
  return new SanitizerCoverageModule(Options);
}
