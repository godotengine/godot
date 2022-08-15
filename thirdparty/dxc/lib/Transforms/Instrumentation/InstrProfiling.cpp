//===-- InstrProfiling.cpp - Frontend instrumentation based profiling -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass lowers instrprof_increment intrinsics emitted by a frontend for
// profiling. It also builds the data structures and initialization code needed
// for updating execution counts and emitting the profile at runtime.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation.h"

#include "llvm/ADT/Triple.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "instrprof"

namespace {

class InstrProfiling : public ModulePass {
public:
  static char ID;

  InstrProfiling() : ModulePass(ID) {}

  InstrProfiling(const InstrProfOptions &Options)
      : ModulePass(ID), Options(Options) {}

  StringRef getPassName() const override {
    return "Frontend instrumentation-based coverage lowering";
  }

  bool runOnModule(Module &M) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
  }

private:
  InstrProfOptions Options;
  Module *M;
  DenseMap<GlobalVariable *, GlobalVariable *> RegionCounters;
  std::vector<Value *> UsedVars;

  bool isMachO() const {
    return Triple(M->getTargetTriple()).isOSBinFormatMachO();
  }

  /// Get the section name for the counter variables.
  StringRef getCountersSection() const {
    return isMachO() ? "__DATA,__llvm_prf_cnts" : "__llvm_prf_cnts";
  }

  /// Get the section name for the name variables.
  StringRef getNameSection() const {
    return isMachO() ? "__DATA,__llvm_prf_names" : "__llvm_prf_names";
  }

  /// Get the section name for the profile data variables.
  StringRef getDataSection() const {
    return isMachO() ? "__DATA,__llvm_prf_data" : "__llvm_prf_data";
  }

  /// Get the section name for the coverage mapping data.
  StringRef getCoverageSection() const {
    return isMachO() ? "__DATA,__llvm_covmap" : "__llvm_covmap";
  }

  /// Replace instrprof_increment with an increment of the appropriate value.
  void lowerIncrement(InstrProfIncrementInst *Inc);

  /// Set up the section and uses for coverage data and its references.
  void lowerCoverageData(GlobalVariable *CoverageData);

  /// Get the region counters for an increment, creating them if necessary.
  ///
  /// If the counter array doesn't yet exist, the profile data variables
  /// referring to them will also be created.
  GlobalVariable *getOrCreateRegionCounters(InstrProfIncrementInst *Inc);

  /// Emit runtime registration functions for each profile data variable.
  void emitRegistration();

  /// Emit the necessary plumbing to pull in the runtime initialization.
  void emitRuntimeHook();

  /// Add uses of our data variables and runtime hook.
  void emitUses();

  /// Create a static initializer for our data, on platforms that need it,
  /// and for any profile output file that was specified.
  void emitInitialization();
};

} // anonymous namespace

char InstrProfiling::ID = 0;
INITIALIZE_PASS(InstrProfiling, "instrprof",
                "Frontend instrumentation-based coverage lowering.", false,
                false)

ModulePass *llvm::createInstrProfilingPass(const InstrProfOptions &Options) {
  return new InstrProfiling(Options);
}

bool InstrProfiling::runOnModule(Module &M) {
  bool MadeChange = false;

  this->M = &M;
  RegionCounters.clear();
  UsedVars.clear();

  for (Function &F : M)
    for (BasicBlock &BB : F)
      for (auto I = BB.begin(), E = BB.end(); I != E;)
        if (auto *Inc = dyn_cast<InstrProfIncrementInst>(I++)) {
          lowerIncrement(Inc);
          MadeChange = true;
        }
  if (GlobalVariable *Coverage = M.getNamedGlobal("__llvm_coverage_mapping")) {
    lowerCoverageData(Coverage);
    MadeChange = true;
  }
  if (!MadeChange)
    return false;

  emitRegistration();
  emitRuntimeHook();
  emitUses();
  emitInitialization();
  return true;
}

void InstrProfiling::lowerIncrement(InstrProfIncrementInst *Inc) {
  GlobalVariable *Counters = getOrCreateRegionCounters(Inc);

  IRBuilder<> Builder(Inc->getParent(), *Inc);
  uint64_t Index = Inc->getIndex()->getZExtValue();
  Value *Addr = Builder.CreateConstInBoundsGEP2_64(Counters, 0, Index);
  Value *Count = Builder.CreateLoad(Addr, "pgocount");
  Count = Builder.CreateAdd(Count, Builder.getInt64(1));
  Inc->replaceAllUsesWith(Builder.CreateStore(Count, Addr));
  Inc->eraseFromParent();
}

void InstrProfiling::lowerCoverageData(GlobalVariable *CoverageData) {
  CoverageData->setSection(getCoverageSection());
  CoverageData->setAlignment(8);

  Constant *Init = CoverageData->getInitializer();
  // We're expecting { i32, i32, i32, i32, [n x { i8*, i32, i32 }], [m x i8] }
  // for some C. If not, the frontend's given us something broken.
  assert(Init->getNumOperands() == 6 && "bad number of fields in coverage map");
  assert(isa<ConstantArray>(Init->getAggregateElement(4)) &&
         "invalid function list in coverage map");
  ConstantArray *Records = cast<ConstantArray>(Init->getAggregateElement(4));
  for (unsigned I = 0, E = Records->getNumOperands(); I < E; ++I) {
    Constant *Record = Records->getOperand(I);
    Value *V = const_cast<Value *>(Record->getOperand(0))->stripPointerCasts();

    assert(isa<GlobalVariable>(V) && "Missing reference to function name");
    GlobalVariable *Name = cast<GlobalVariable>(V);

    // If we have region counters for this name, we've already handled it.
    auto It = RegionCounters.find(Name);
    if (It != RegionCounters.end())
      continue;

    // Move the name variable to the right section.
    Name->setSection(getNameSection());
    Name->setAlignment(1);
  }
}

/// Get the name of a profiling variable for a particular function.
static std::string getVarName(InstrProfIncrementInst *Inc, StringRef VarName) {
  auto *Arr = cast<ConstantDataArray>(Inc->getName()->getInitializer());
  StringRef Name = Arr->isCString() ? Arr->getAsCString() : Arr->getAsString();
  return ("__llvm_profile_" + VarName + "_" + Name).str();
}

GlobalVariable *
InstrProfiling::getOrCreateRegionCounters(InstrProfIncrementInst *Inc) {
  GlobalVariable *Name = Inc->getName();
  auto It = RegionCounters.find(Name);
  if (It != RegionCounters.end())
    return It->second;

  // Move the name variable to the right section. Make sure it is placed in the
  // same comdat as its associated function. Otherwise, we may get multiple
  // counters for the same function in certain cases.
  Function *Fn = Inc->getParent()->getParent();
  Name->setSection(getNameSection());
  Name->setAlignment(1);
  Name->setComdat(Fn->getComdat());

  uint64_t NumCounters = Inc->getNumCounters()->getZExtValue();
  LLVMContext &Ctx = M->getContext();
  ArrayType *CounterTy = ArrayType::get(Type::getInt64Ty(Ctx), NumCounters);

  // Create the counters variable.
  auto *Counters = new GlobalVariable(*M, CounterTy, false, Name->getLinkage(),
                                      Constant::getNullValue(CounterTy),
                                      getVarName(Inc, "counters"));
  Counters->setVisibility(Name->getVisibility());
  Counters->setSection(getCountersSection());
  Counters->setAlignment(8);
  Counters->setComdat(Fn->getComdat());

  RegionCounters[Inc->getName()] = Counters;

  // Create data variable.
  auto *NameArrayTy = Name->getType()->getPointerElementType();
  auto *Int32Ty = Type::getInt32Ty(Ctx);
  auto *Int64Ty = Type::getInt64Ty(Ctx);
  auto *Int8PtrTy = Type::getInt8PtrTy(Ctx);
  auto *Int64PtrTy = Type::getInt64PtrTy(Ctx);

  Type *DataTypes[] = {Int32Ty, Int32Ty, Int64Ty, Int8PtrTy, Int64PtrTy};
  auto *DataTy = StructType::get(Ctx, makeArrayRef(DataTypes));
  Constant *DataVals[] = {
      ConstantInt::get(Int32Ty, NameArrayTy->getArrayNumElements()),
      ConstantInt::get(Int32Ty, NumCounters),
      ConstantInt::get(Int64Ty, Inc->getHash()->getZExtValue()),
      ConstantExpr::getBitCast(Name, Int8PtrTy),
      ConstantExpr::getBitCast(Counters, Int64PtrTy)};
  auto *Data = new GlobalVariable(*M, DataTy, true, Name->getLinkage(),
                                  ConstantStruct::get(DataTy, DataVals),
                                  getVarName(Inc, "data"));
  Data->setVisibility(Name->getVisibility());
  Data->setSection(getDataSection());
  Data->setAlignment(8);
  Data->setComdat(Fn->getComdat());

  // Mark the data variable as used so that it isn't stripped out.
  UsedVars.push_back(Data);

  return Counters;
}

void InstrProfiling::emitRegistration() {
  // Don't do this for Darwin.  compiler-rt uses linker magic.
  if (Triple(M->getTargetTriple()).isOSDarwin())
    return;

  // Construct the function.
  auto *VoidTy = Type::getVoidTy(M->getContext());
  auto *VoidPtrTy = Type::getInt8PtrTy(M->getContext());
  auto *RegisterFTy = FunctionType::get(VoidTy, false);
  auto *RegisterF = Function::Create(RegisterFTy, GlobalValue::InternalLinkage,
                                     "__llvm_profile_register_functions", M);
  RegisterF->setUnnamedAddr(true);
  if (Options.NoRedZone)
    RegisterF->addFnAttr(Attribute::NoRedZone);

  auto *RuntimeRegisterTy = FunctionType::get(VoidTy, VoidPtrTy, false);
  auto *RuntimeRegisterF =
      Function::Create(RuntimeRegisterTy, GlobalVariable::ExternalLinkage,
                       "__llvm_profile_register_function", M);

  IRBuilder<> IRB(BasicBlock::Create(M->getContext(), "", RegisterF));
  for (Value *Data : UsedVars)
    IRB.CreateCall(RuntimeRegisterF, IRB.CreateBitCast(Data, VoidPtrTy));
  IRB.CreateRetVoid();
}

void InstrProfiling::emitRuntimeHook() {
  const char *const RuntimeVarName = "__llvm_profile_runtime";
  const char *const RuntimeUserName = "__llvm_profile_runtime_user";

  // If the module's provided its own runtime, we don't need to do anything.
  if (M->getGlobalVariable(RuntimeVarName))
    return;

  // Declare an external variable that will pull in the runtime initialization.
  auto *Int32Ty = Type::getInt32Ty(M->getContext());
  auto *Var =
      new GlobalVariable(*M, Int32Ty, false, GlobalValue::ExternalLinkage,
                         nullptr, RuntimeVarName);

  // Make a function that uses it.
  auto *User =
      Function::Create(FunctionType::get(Int32Ty, false),
                       GlobalValue::LinkOnceODRLinkage, RuntimeUserName, M);
  User->addFnAttr(Attribute::NoInline);
  if (Options.NoRedZone)
    User->addFnAttr(Attribute::NoRedZone);
  User->setVisibility(GlobalValue::HiddenVisibility);

  IRBuilder<> IRB(BasicBlock::Create(M->getContext(), "", User));
  auto *Load = IRB.CreateLoad(Var);
  IRB.CreateRet(Load);

  // Mark the user variable as used so that it isn't stripped out.
  UsedVars.push_back(User);
}

void InstrProfiling::emitUses() {
  if (UsedVars.empty())
    return;

  GlobalVariable *LLVMUsed = M->getGlobalVariable("llvm.used");
  std::vector<Constant *> MergedVars;
  if (LLVMUsed) {
    // Collect the existing members of llvm.used.
    ConstantArray *Inits = cast<ConstantArray>(LLVMUsed->getInitializer());
    for (unsigned I = 0, E = Inits->getNumOperands(); I != E; ++I)
      MergedVars.push_back(Inits->getOperand(I));
    LLVMUsed->eraseFromParent();
  }

  Type *i8PTy = Type::getInt8PtrTy(M->getContext());
  // Add uses for our data.
  for (auto *Value : UsedVars)
    MergedVars.push_back(
        ConstantExpr::getBitCast(cast<Constant>(Value), i8PTy));

  // Recreate llvm.used.
  ArrayType *ATy = ArrayType::get(i8PTy, MergedVars.size());
  LLVMUsed =
      new GlobalVariable(*M, ATy, false, GlobalValue::AppendingLinkage,
                         ConstantArray::get(ATy, MergedVars), "llvm.used");

  LLVMUsed->setSection("llvm.metadata");
}

void InstrProfiling::emitInitialization() {
  std::string InstrProfileOutput = Options.InstrProfileOutput;

  Constant *RegisterF = M->getFunction("__llvm_profile_register_functions");
  if (!RegisterF && InstrProfileOutput.empty())
    return;

  // Create the initialization function.
  auto *VoidTy = Type::getVoidTy(M->getContext());
  auto *F =
      Function::Create(FunctionType::get(VoidTy, false),
                       GlobalValue::InternalLinkage, "__llvm_profile_init", M);
  F->setUnnamedAddr(true);
  F->addFnAttr(Attribute::NoInline);
  if (Options.NoRedZone)
    F->addFnAttr(Attribute::NoRedZone);

  // Add the basic block and the necessary calls.
  IRBuilder<> IRB(BasicBlock::Create(M->getContext(), "", F));
  if (RegisterF)
    IRB.CreateCall(RegisterF, {});
  if (!InstrProfileOutput.empty()) {
    auto *Int8PtrTy = Type::getInt8PtrTy(M->getContext());
    auto *SetNameTy = FunctionType::get(VoidTy, Int8PtrTy, false);
    auto *SetNameF =
        Function::Create(SetNameTy, GlobalValue::ExternalLinkage,
                         "__llvm_profile_override_default_filename", M);

    // Create variable for profile name.
    Constant *ProfileNameConst =
        ConstantDataArray::getString(M->getContext(), InstrProfileOutput, true);
    GlobalVariable *ProfileName =
        new GlobalVariable(*M, ProfileNameConst->getType(), true,
                           GlobalValue::PrivateLinkage, ProfileNameConst);

    IRB.CreateCall(SetNameF, IRB.CreatePointerCast(ProfileName, Int8PtrTy));
  }
  IRB.CreateRetVoid();

  appendToGlobalCtors(*M, F, 0);
}
