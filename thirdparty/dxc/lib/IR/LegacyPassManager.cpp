//===- LegacyPassManager.cpp - LLVM Pass Infrastructure Implementation ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the legacy LLVM Pass Manager infrastructure.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/IRPrintingPasses.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/LegacyPassManagers.h"
#include "llvm/IR/LegacyPassNameParser.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/Mutex.h"
#include "llvm/Support/TimeValue.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <map>
using namespace llvm;
using namespace llvm::legacy;

// See PassManagers.h for Pass Manager infrastructure overview.

//===----------------------------------------------------------------------===//
// Pass debugging information.  Often it is useful to find out what pass is
// running when a crash occurs in a utility.  When this library is compiled with
// debugging on, a command line option (--debug-pass) is enabled that causes the
// pass name to be printed before it executes.
//

namespace {
// Different debug levels that can be enabled...
enum PassDebugLevel {
  Disabled, Arguments, Structure, Executions, Details
};
}

#if 0 // HLSL Change Starts - option pending
static cl::opt<enum PassDebugLevel>
PassDebugging("debug-pass", cl::Hidden,
                  cl::desc("Print PassManager debugging information"),
                  cl::values(
  clEnumVal(Disabled  , "disable debug output"),
  clEnumVal(Arguments , "print pass arguments to pass to 'opt'"),
  clEnumVal(Structure , "print pass structure before run()"),
  clEnumVal(Executions, "print pass name before it is executed"),
  clEnumVal(Details   , "print pass details when it is executed"),
                             clEnumValEnd));
#else
static const PassDebugLevel PassDebugging  = PassDebugLevel::Disabled;
#endif // HLSL Change Ends

namespace {
typedef llvm::cl::list<const llvm::PassInfo *, bool, PassNameParser>
PassOptionList;
}

#if 0 // HLSL Change Starts - option pending
// Print IR out before/after specified passes.
static PassOptionList
PrintBefore("print-before",
            llvm::cl::desc("Print IR before specified passes"),
            cl::Hidden);

static PassOptionList
PrintAfter("print-after",
           llvm::cl::desc("Print IR after specified passes"),
           cl::Hidden);

static cl::opt<bool>
PrintBeforeAll("print-before-all",
               llvm::cl::desc("Print IR before each pass"),
               cl::init(false));
static cl::opt<bool>
PrintAfterAll("print-after-all",
              llvm::cl::desc("Print IR after each pass"),
              cl::init(false));
#endif // HLSL Change Ends

/// This is a helper to determine whether to print IR before or
/// after a pass.

#if 0 // HLSL Change
static bool ShouldPrintBeforeOrAfterPass(const PassInfo *PI,
                                         PassOptionList &PassesToPrint) {
  for (auto *PassInf : PassesToPrint) {
    if (PassInf)
      if (PassInf->getPassArgument() == PI->getPassArgument()) {
        return true;
      }
  }
  return false;
}
#endif

/// This is a utility to check whether a pass should have IR dumped
/// before it.
static bool ShouldPrintBeforePass(const PassInfo *PI) {
  return false; // HLSL Change - return PrintBeforeAll || ShouldPrintBeforeOrAfterPass(PI, PrintBefore);
}

/// This is a utility to check whether a pass should have IR dumped
/// after it.
static bool ShouldPrintAfterPass(const PassInfo *PI) {
  return false; // HLSL Change -PrintAfterAll || ShouldPrintBeforeOrAfterPass(PI, PrintAfter);
}

/// isPassDebuggingExecutionsOrMore - Return true if -debug-pass=Executions
/// or higher is specified.
bool PMDataManager::isPassDebuggingExecutionsOrMore() const {
  return PassDebugging >= Executions;
}




void PassManagerPrettyStackEntry::print(raw_ostream &OS) const {
  if (!V && !M)
    OS << "Releasing pass '";
  else
    OS << "Running pass '";

  OS << P->getPassName() << "'";

  if (M) {
    OS << " on module '" << M->getModuleIdentifier() << "'.\n";
    return;
  }
  if (!V) {
    OS << '\n';
    return;
  }

  OS << " on ";
  if (isa<Function>(V))
    OS << "function";
  else if (isa<BasicBlock>(V))
    OS << "basic block";
  else
    OS << "value";

  OS << " '";
  V->printAsOperand(OS, /*PrintTy=*/false, M);
  OS << "'\n";
}


namespace {
//===----------------------------------------------------------------------===//
// BBPassManager
//
/// BBPassManager manages BasicBlockPass. It batches all the
/// pass together and sequence them to process one basic block before
/// processing next basic block.
class BBPassManager : public PMDataManager, public FunctionPass {

public:
  static char ID;
  explicit BBPassManager()
    : PMDataManager(), FunctionPass(ID) {}

  /// Execute all of the passes scheduled for execution.  Keep track of
  /// whether any of the passes modifies the function, and if so, return true.
  bool runOnFunction(Function &F) override;

  /// Pass Manager itself does not invalidate any analysis info.
  void getAnalysisUsage(AnalysisUsage &Info) const override {
    Info.setPreservesAll();
  }

  bool doInitialization(Module &M) override;
  bool doInitialization(Function &F);
  bool doFinalization(Module &M) override;
  bool doFinalization(Function &F);

  PMDataManager *getAsPMDataManager() override { return this; }
  Pass *getAsPass() override { return this; }

  StringRef getPassName() const override {
    return "BasicBlock Pass Manager";
  }

  // Print passes managed by this manager
  void dumpPassStructure(unsigned Offset) override {
    dbgs().indent(Offset*2) << "BasicBlockPass Manager\n";
    for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
      BasicBlockPass *BP = getContainedPass(Index);
      BP->dumpPassStructure(Offset + 1);
      dumpLastUses(BP, Offset+1);
    }
  }

  BasicBlockPass *getContainedPass(unsigned N) {
    assert(N < PassVector.size() && "Pass number out of range!");
    BasicBlockPass *BP = static_cast<BasicBlockPass *>(PassVector[N]);
    return BP;
  }

  PassManagerType getPassManagerType() const override {
    return PMT_BasicBlockPassManager;
  }
};

char BBPassManager::ID = 0;
} // End anonymous namespace

namespace llvm {
namespace legacy {
//===----------------------------------------------------------------------===//
// FunctionPassManagerImpl
//
/// FunctionPassManagerImpl manages FPPassManagers
class FunctionPassManagerImpl : public Pass,
                                public PMDataManager,
                                public PMTopLevelManager {
  virtual void anchor();
private:
  bool wasRun;
public:
  static char ID;
  explicit FunctionPassManagerImpl() :
    Pass(PT_PassManager, ID), PMDataManager(),
    PMTopLevelManager(new FPPassManager()), wasRun(false) {}

  /// \copydoc FunctionPassManager::add()
  void add(Pass *P) {
    schedulePass(P);
  }

  /// createPrinterPass - Get a function printer pass.
  Pass *createPrinterPass(raw_ostream &O,
                          const std::string &Banner) const override {
    return createPrintFunctionPass(O, Banner);
  }

  // Prepare for running an on the fly pass, freeing memory if needed
  // from a previous run.
  void releaseMemoryOnTheFly();

  /// run - Execute all of the passes scheduled for execution.  Keep track of
  /// whether any of the passes modifies the module, and if so, return true.
  bool run(Function &F);

  /// doInitialization - Run all of the initializers for the function passes.
  ///
  bool doInitialization(Module &M) override;

  /// doFinalization - Run all of the finalizers for the function passes.
  ///
  bool doFinalization(Module &M) override;


  PMDataManager *getAsPMDataManager() override { return this; }
  Pass *getAsPass() override { return this; }
  PassManagerType getTopLevelPassManagerType() override {
    return PMT_FunctionPassManager;
  }

  /// Pass Manager itself does not invalidate any analysis info.
  void getAnalysisUsage(AnalysisUsage &Info) const override {
    Info.setPreservesAll();
  }

  FPPassManager *getContainedManager(unsigned N) {
    assert(N < PassManagers.size() && "Pass number out of range!");
    FPPassManager *FP = static_cast<FPPassManager *>(PassManagers[N]);
    return FP;
  }
};

void FunctionPassManagerImpl::anchor() {}

char FunctionPassManagerImpl::ID = 0;
} // End of legacy namespace
} // End of llvm namespace

namespace {
//===----------------------------------------------------------------------===//
// MPPassManager
//
/// MPPassManager manages ModulePasses and function pass managers.
/// It batches all Module passes and function pass managers together and
/// sequences them to process one module.
class MPPassManager : public Pass, public PMDataManager {
public:
  static char ID;
  explicit MPPassManager() :
    Pass(PT_PassManager, ID), PMDataManager() { }

  // Delete on the fly managers.
  ~MPPassManager() override {
    for (auto &OnTheFlyManager : OnTheFlyManagers) {
      FunctionPassManagerImpl *FPP = OnTheFlyManager.second;
      delete FPP;
    }
  }

  /// createPrinterPass - Get a module printer pass.
  Pass *createPrinterPass(raw_ostream &O,
                          const std::string &Banner) const override {
    return createPrintModulePass(O, Banner);
  }

  /// run - Execute all of the passes scheduled for execution.  Keep track of
  /// whether any of the passes modifies the module, and if so, return true.
  bool runOnModule(Module &M);

  using llvm::Pass::doInitialization;
  using llvm::Pass::doFinalization;

  /// doInitialization - Run all of the initializers for the module passes.
  ///
  bool doInitialization();

  /// doFinalization - Run all of the finalizers for the module passes.
  ///
  bool doFinalization();

  /// Pass Manager itself does not invalidate any analysis info.
  void getAnalysisUsage(AnalysisUsage &Info) const override {
    Info.setPreservesAll();
  }

  /// Add RequiredPass into list of lower level passes required by pass P.
  /// RequiredPass is run on the fly by Pass Manager when P requests it
  /// through getAnalysis interface.
  void addLowerLevelRequiredPass(Pass *P, Pass *RequiredPass) override;

  /// Return function pass corresponding to PassInfo PI, that is
  /// required by module pass MP. Instantiate analysis pass, by using
  /// its runOnFunction() for function F.
  Pass* getOnTheFlyPass(Pass *MP, AnalysisID PI, Function &F) override;

  StringRef getPassName() const override {
    return "Module Pass Manager";
  }

  PMDataManager *getAsPMDataManager() override { return this; }
  Pass *getAsPass() override { return this; }

  // Print passes managed by this manager
  void dumpPassStructure(unsigned Offset) override {
    dbgs().indent(Offset*2) << "ModulePass Manager\n";
    for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
      ModulePass *MP = getContainedPass(Index);
      MP->dumpPassStructure(Offset + 1);
      std::map<Pass *, FunctionPassManagerImpl *>::const_iterator I =
        OnTheFlyManagers.find(MP);
      if (I != OnTheFlyManagers.end())
        I->second->dumpPassStructure(Offset + 2);
      dumpLastUses(MP, Offset+1);
    }
  }

  ModulePass *getContainedPass(unsigned N) {
    assert(N < PassVector.size() && "Pass number out of range!");
    return static_cast<ModulePass *>(PassVector[N]);
  }

  PassManagerType getPassManagerType() const override {
    return PMT_ModulePassManager;
  }

 private:
  /// Collection of on the fly FPPassManagers. These managers manage
  /// function passes that are required by module passes.
  std::map<Pass *, FunctionPassManagerImpl *> OnTheFlyManagers;
};

char MPPassManager::ID = 0;
} // End anonymous namespace

namespace llvm {
namespace legacy {
//===----------------------------------------------------------------------===//
// PassManagerImpl
//

/// PassManagerImpl manages MPPassManagers
class PassManagerImpl : public Pass,
                        public PMDataManager,
                        public PMTopLevelManager {
  virtual void anchor();

public:
  static char ID;
  explicit PassManagerImpl() :
    Pass(PT_PassManager, ID), PMDataManager(),
                              PMTopLevelManager(new MPPassManager()) {}

  /// \copydoc PassManager::add()
  void add(Pass *P) {
    schedulePass(P);
  }

  /// createPrinterPass - Get a module printer pass.
  Pass *createPrinterPass(raw_ostream &O,
                          const std::string &Banner) const override {
    return createPrintModulePass(O, Banner);
  }

  /// run - Execute all of the passes scheduled for execution.  Keep track of
  /// whether any of the passes modifies the module, and if so, return true.
  bool run(Module &M);

  using llvm::Pass::doInitialization;
  using llvm::Pass::doFinalization;

  /// doInitialization - Run all of the initializers for the module passes.
  ///
  bool doInitialization();

  /// doFinalization - Run all of the finalizers for the module passes.
  ///
  bool doFinalization();

  /// Pass Manager itself does not invalidate any analysis info.
  void getAnalysisUsage(AnalysisUsage &Info) const override {
    Info.setPreservesAll();
  }

  PMDataManager *getAsPMDataManager() override { return this; }
  Pass *getAsPass() override { return this; }
  PassManagerType getTopLevelPassManagerType() override {
    return PMT_ModulePassManager;
  }

  MPPassManager *getContainedManager(unsigned N) {
    assert(N < PassManagers.size() && "Pass number out of range!");
    MPPassManager *MP = static_cast<MPPassManager *>(PassManagers[N]);
    return MP;
  }
};

void PassManagerImpl::anchor() {}

char PassManagerImpl::ID = 0;
} // End of legacy namespace
} // End of llvm namespace

namespace {

//===----------------------------------------------------------------------===//
/// TimingInfo Class - This class is used to calculate information about the
/// amount of time each pass takes to execute.  This only happens when
/// -time-passes is enabled on the command line.
///

static ManagedStatic<sys::SmartMutex<true> > TimingInfoMutex;

class TimingInfo {
  DenseMap<Pass*, Timer*> TimingData;
  TimerGroup TG;
public:
  // Use 'create' member to get this.
  TimingInfo() : TG("... Pass execution timing report ...") {}

  // TimingDtor - Print out information about timing information
  ~TimingInfo() {
    // Delete all of the timers, which accumulate their info into the
    // TimerGroup.
    for (auto &I : TimingData)
      delete I.second;
    // TimerGroup is deleted next, printing the report.
  }

  // createTheTimeInfo - This method either initializes the TheTimeInfo pointer
  // to a non-null value (if the -time-passes option is enabled) or it leaves it
  // null.  It may be called multiple times.
  static void createTheTimeInfo();

  /// getPassTimer - Return the timer for the specified pass if it exists.
  Timer *getPassTimer(Pass *P) {
    if (P->getAsPMDataManager())
      return nullptr;

    sys::SmartScopedLock<true> Lock(*TimingInfoMutex);
    Timer *&T = TimingData[P];
    if (!T)
      T = new Timer(P->getPassName(), TG);
    return T;
  }
};

} // End of anon namespace

static TimingInfo *TheTimeInfo;

//===----------------------------------------------------------------------===//
// PMTopLevelManager implementation

/// Initialize top level manager. Create first pass manager.
PMTopLevelManager::PMTopLevelManager(PMDataManager *PMDM) {
  PMDM->setTopLevelManager(this);
  addPassManager(PMDM);
  activeStack.push(PMDM);
}

/// Set pass P as the last user of the given analysis passes.
void
PMTopLevelManager::setLastUser(ArrayRef<Pass*> AnalysisPasses, Pass *P) {
  unsigned PDepth = 0;
  if (P->getResolver())
    PDepth = P->getResolver()->getPMDataManager().getDepth();

  for (Pass *AP : AnalysisPasses) {
    LastUser[AP] = P;

    if (P == AP)
      continue;

    // Update the last users of passes that are required transitive by AP.
    AnalysisUsage *AnUsage = findAnalysisUsage(AP);
    const AnalysisUsage::VectorType &IDs = AnUsage->getRequiredTransitiveSet();
    SmallVector<Pass *, 12> LastUses;
    SmallVector<Pass *, 12> LastPMUses;
    for (AnalysisUsage::VectorType::const_iterator I = IDs.begin(),
         E = IDs.end(); I != E; ++I) {
      Pass *AnalysisPass = findAnalysisPass(*I);
      assert(AnalysisPass && "Expected analysis pass to exist.");
      AnalysisResolver *AR = AnalysisPass->getResolver();
      assert(AR && "Expected analysis resolver to exist.");
      unsigned APDepth = AR->getPMDataManager().getDepth();

      if (PDepth == APDepth)
        LastUses.push_back(AnalysisPass);
      else if (PDepth > APDepth)
        LastPMUses.push_back(AnalysisPass);
    }

    setLastUser(LastUses, P);

    // If this pass has a corresponding pass manager, push higher level
    // analysis to this pass manager.
    if (P->getResolver())
      setLastUser(LastPMUses, P->getResolver()->getPMDataManager().getAsPass());


    // If AP is the last user of other passes then make P last user of
    // such passes.
    for (DenseMap<Pass *, Pass *>::iterator LUI = LastUser.begin(),
           LUE = LastUser.end(); LUI != LUE; ++LUI) {
      if (LUI->second == AP)
        // DenseMap iterator is not invalidated here because
        // this is just updating existing entries.
        LastUser[LUI->first] = P;
    }
  }
}

/// Collect passes whose last user is P
void PMTopLevelManager::collectLastUses(SmallVectorImpl<Pass *> &LastUses,
                                        Pass *P) {
  DenseMap<Pass *, SmallPtrSet<Pass *, 8> >::iterator DMI =
    InversedLastUser.find(P);
  if (DMI == InversedLastUser.end())
    return;

  SmallPtrSet<Pass *, 8> &LU = DMI->second;
  for (Pass *LUP : LU) {
    LastUses.push_back(LUP);
  }

}

AnalysisUsage *PMTopLevelManager::findAnalysisUsage(Pass *P) {
  AnalysisUsage *AnUsage = nullptr;
  DenseMap<Pass *, AnalysisUsage *>::iterator DMI = AnUsageMap.find(P);
  if (DMI != AnUsageMap.end())
    AnUsage = DMI->second;
  else {
    AnUsage = new AnalysisUsage();
    std::unique_ptr<AnalysisUsage> AnUsagePtr(AnUsage); // HLSL Change - unique_ptr until added
    P->getAnalysisUsage(*AnUsage);
    AnUsageMap[P] = AnUsage;
    AnUsagePtr.release(); // HLSL Change
  }
  return AnUsage;
}

/// Schedule pass P for execution. Make sure that passes required by
/// P are run before P is run. Update analysis info maintained by
/// the manager. Remove dead passes. This is a recursive function.
void PMTopLevelManager::schedulePass(Pass *P) {

  // TODO : Allocate function manager for this pass, other wise required set
  // may be inserted into previous function manager

  std::unique_ptr<Pass> PPtr(P); // take ownership locally until we pass it on

  // Give pass a chance to prepare the stage.
  P->preparePassManager(activeStack);

  // If P is an analysis pass and it is available then do not
  // generate the analysis again. Stale analysis info should not be
  // available at this point.
  const PassInfo *PI = findAnalysisPassInfo(P->getPassID());
  if (PI && PI->isAnalysis() && findAnalysisPass(P->getPassID())) {
    // delete P; // HLSL Change - let PPtr take care of this
    return;
  }

  AnalysisUsage *AnUsage = findAnalysisUsage(P);

  bool checkAnalysis = true;
  while (checkAnalysis) {
    checkAnalysis = false;

    const AnalysisUsage::VectorType &RequiredSet = AnUsage->getRequiredSet();
    for (AnalysisUsage::VectorType::const_iterator I = RequiredSet.begin(),
           E = RequiredSet.end(); I != E; ++I) {

      Pass *AnalysisPass = findAnalysisPass(*I);
      if (!AnalysisPass) {
        const PassInfo *PI = findAnalysisPassInfo(*I);

        if (!PI) {
          // Pass P is not in the global PassRegistry
          dbgs() << "Pass '"  << P->getPassName() << "' is not initialized." << "\n";
          dbgs() << "Verify if there is a pass dependency cycle." << "\n";
          dbgs() << "Required Passes:" << "\n";
          for (AnalysisUsage::VectorType::const_iterator I2 = RequiredSet.begin(),
                 E = RequiredSet.end(); I2 != E && I2 != I; ++I2) {
            Pass *AnalysisPass2 = findAnalysisPass(*I2);
            if (AnalysisPass2) {
              dbgs() << "\t" << AnalysisPass2->getPassName() << "\n";
            } else {
              dbgs() << "\t"   << "Error: Required pass not found! Possible causes:"  << "\n";
              dbgs() << "\t\t" << "- Pass misconfiguration (e.g.: missing macros)"    << "\n";
              dbgs() << "\t\t" << "- Corruption of the global PassRegistry"           << "\n";
            }
          }
        }

        assert(PI && "Expected required passes to be initialized");
        AnalysisPass = PI->createPass();
        if (P->getPotentialPassManagerType () ==
            AnalysisPass->getPotentialPassManagerType())
          // Schedule analysis pass that is managed by the same pass manager.
          schedulePass(AnalysisPass);
        else if (P->getPotentialPassManagerType () >
                 AnalysisPass->getPotentialPassManagerType()) {
          // Schedule analysis pass that is managed by a new manager.
          schedulePass(AnalysisPass);
          // Recheck analysis passes to ensure that required analyses that
          // are already checked are still available.
          checkAnalysis = true;
        } else
          // Do not schedule this analysis. Lower level analysis
          // passes are run on the fly.
          delete AnalysisPass;
      }
    }
  }

  // Now all required passes are available.
  if (ImmutablePass *IP = P->getAsImmutablePass()) {
    // P is a immutable pass and it will be managed by this
    // top level manager. Set up analysis resolver to connect them.
    PMDataManager *DM = getAsPMDataManager();
    AnalysisResolver *AR = new AnalysisResolver(*DM);
    P->setResolver(AR); // HLSL Comment - P takes ownership of AR here
    DM->initializeAnalysisImpl(P);
    addImmutablePass(IP);
    PPtr.release(); // HLSL Change
    DM->recordAvailableAnalysis(IP);
    return;
  }

  if (PI && !PI->isAnalysis() && ShouldPrintBeforePass(PI)) {
    Pass *PP = P->createPrinterPass(dbgs(), std::string("*** IR Dump Before ") +
                                         P->getPassName().str() + " ***");
    PP->assignPassManager(activeStack, getTopLevelPassManagerType());
  }

  // Add the requested pass to the best available pass manager.
  PPtr.release(); // HLSL Change - assignPassManager takes ownership
  P->assignPassManager(activeStack, getTopLevelPassManagerType());

  if (PI && !PI->isAnalysis() && ShouldPrintAfterPass(PI)) {
    Pass *PP = P->createPrinterPass(dbgs(), std::string("*** IR Dump After ") +
                                         P->getPassName().str() + " ***");
    PP->assignPassManager(activeStack, getTopLevelPassManagerType());
  }

  // HLSL Change - begin
  if (PI && !PI->isAnalysis() && (this->HLSLPrintAfterAll || (this->HLSLPrintAfter.size() && this->HLSLPrintAfter.count(PI->getPassArgument())))) {
    class direct_stderr_stream : public raw_ostream {
      uint64_t current_pos() const override { return 0; }
      /// See raw_ostream::write_impl.
      void write_impl(const char *Ptr, size_t Size) override {
        fwrite(Ptr, Size, 1, stderr);
      }
    };

    static direct_stderr_stream stderr_stream;

    Pass *PP = P->createPrinterPass(
      stderr_stream, std::string("*** IR Dump After ") + P->getPassName().str() + " (" + PI->getPassArgument() + ") ***");
    PP->assignPassManager(activeStack, getTopLevelPassManagerType());
  }
  // HLSL Change - end
}

/// Find the pass that implements Analysis AID. Search immutable
/// passes and all pass managers. If desired pass is not found
/// then return NULL.
Pass *PMTopLevelManager::findAnalysisPass(AnalysisID AID) {

  // Check pass managers
  for (PMDataManager *PassManager : PassManagers)
    if (Pass *P = PassManager->findAnalysisPass(AID, false))
      return P;

  // Check other pass managers
  for (PMDataManager *IndirectPassManager : IndirectPassManagers)
    if (Pass *P = IndirectPassManager->findAnalysisPass(AID, false))
      return P;

  // Check the immutable passes. Iterate in reverse order so that we find
  // the most recently registered passes first.
  for (auto I = ImmutablePasses.rbegin(), E = ImmutablePasses.rend(); I != E;
       ++I) {
    AnalysisID PI = (*I)->getPassID();
    if (PI == AID)
      return *I;

    // If Pass not found then check the interfaces implemented by Immutable Pass
    const PassInfo *PassInf = findAnalysisPassInfo(PI);
    assert(PassInf && "Expected all immutable passes to be initialized");
    const std::vector<const PassInfo*> &ImmPI =
      PassInf->getInterfacesImplemented();
    for (const PassInfo *PI : ImmPI)
      if (PI->getTypeInfo() == AID)
        return *I;
  }

  return nullptr;
}

const PassInfo *PMTopLevelManager::findAnalysisPassInfo(AnalysisID AID) const {
  const PassInfo *&PI = AnalysisPassInfos[AID];
  if (!PI)
    PI = PassRegistry::getPassRegistry()->getPassInfo(AID);
  else
    assert(PI == PassRegistry::getPassRegistry()->getPassInfo(AID) &&
           "The pass info pointer changed for an analysis ID!");

  return PI;
}

// Print passes managed by this top level manager.
void PMTopLevelManager::dumpPasses() const {

  if (PassDebugging < Structure)
    return;

  // Print out the immutable passes
  for (unsigned i = 0, e = ImmutablePasses.size(); i != e; ++i) {
    ImmutablePasses[i]->dumpPassStructure(0);
  }

  // Every class that derives from PMDataManager also derives from Pass
  // (sometimes indirectly), but there's no inheritance relationship
  // between PMDataManager and Pass, so we have to getAsPass to get
  // from a PMDataManager* to a Pass*.
  for (PMDataManager *Manager : PassManagers)
    Manager->getAsPass()->dumpPassStructure(1);
}

void PMTopLevelManager::dumpArguments() const {

  if (PassDebugging < Arguments)
    return;

  dbgs() << "Pass Arguments: ";
  for (SmallVectorImpl<ImmutablePass *>::const_iterator I =
       ImmutablePasses.begin(), E = ImmutablePasses.end(); I != E; ++I)
    if (const PassInfo *PI = findAnalysisPassInfo((*I)->getPassID())) {
      assert(PI && "Expected all immutable passes to be initialized");
      if (!PI->isAnalysisGroup())
        dbgs() << " -" << PI->getPassArgument();
    }
  for (SmallVectorImpl<PMDataManager *>::const_iterator I =
       PassManagers.begin(), E = PassManagers.end(); I != E; ++I)
    (*I)->dumpPassArguments();
  dbgs() << "\n";
}

void PMTopLevelManager::initializeAllAnalysisInfo() {
  for (SmallVectorImpl<PMDataManager *>::iterator I = PassManagers.begin(),
         E = PassManagers.end(); I != E; ++I)
    (*I)->initializeAnalysisInfo();

  // Initailize other pass managers
  for (SmallVectorImpl<PMDataManager *>::iterator
       I = IndirectPassManagers.begin(), E = IndirectPassManagers.end();
       I != E; ++I)
    (*I)->initializeAnalysisInfo();

  for (DenseMap<Pass *, Pass *>::iterator DMI = LastUser.begin(),
        DME = LastUser.end(); DMI != DME; ++DMI) {
    DenseMap<Pass *, SmallPtrSet<Pass *, 8> >::iterator InvDMI =
      InversedLastUser.find(DMI->second);
    if (InvDMI != InversedLastUser.end()) {
      SmallPtrSet<Pass *, 8> &L = InvDMI->second;
      L.insert(DMI->first);
    } else {
      SmallPtrSet<Pass *, 8> L; L.insert(DMI->first);
      InversedLastUser[DMI->second] = L;
    }
  }
}

/// Destructor
PMTopLevelManager::~PMTopLevelManager() {
  for (SmallVectorImpl<PMDataManager *>::iterator I = PassManagers.begin(),
         E = PassManagers.end(); I != E; ++I)
    delete *I;

  for (SmallVectorImpl<ImmutablePass *>::iterator
         I = ImmutablePasses.begin(), E = ImmutablePasses.end(); I != E; ++I)
    delete *I;

  for (DenseMap<Pass *, AnalysisUsage *>::iterator DMI = AnUsageMap.begin(),
         DME = AnUsageMap.end(); DMI != DME; ++DMI)
    delete DMI->second;
}

//===----------------------------------------------------------------------===//
// PMDataManager implementation

/// Augement AvailableAnalysis by adding analysis made available by pass P.
void PMDataManager::recordAvailableAnalysis(Pass *P) {
  AnalysisID PI = P->getPassID();

  AvailableAnalysis[PI] = P;

  assert(!AvailableAnalysis.empty());

  // This pass is the current implementation of all of the interfaces it
  // implements as well.
  const PassInfo *PInf = TPM->findAnalysisPassInfo(PI);
  if (!PInf) return;
  const std::vector<const PassInfo*> &II = PInf->getInterfacesImplemented();
  for (unsigned i = 0, e = II.size(); i != e; ++i)
    AvailableAnalysis[II[i]->getTypeInfo()] = P;
}

// Return true if P preserves high level analysis used by other
// passes managed by this manager
bool PMDataManager::preserveHigherLevelAnalysis(Pass *P) {
  AnalysisUsage *AnUsage = TPM->findAnalysisUsage(P);
  if (AnUsage->getPreservesAll())
    return true;

  const AnalysisUsage::VectorType &PreservedSet = AnUsage->getPreservedSet();
  for (SmallVectorImpl<Pass *>::iterator I = HigherLevelAnalysis.begin(),
         E = HigherLevelAnalysis.end(); I  != E; ++I) {
    Pass *P1 = *I;
    if (P1->getAsImmutablePass() == nullptr &&
        std::find(PreservedSet.begin(), PreservedSet.end(),
                  P1->getPassID()) ==
           PreservedSet.end())
      return false;
  }

  return true;
}

/// verifyPreservedAnalysis -- Verify analysis preserved by pass P.
void PMDataManager::verifyPreservedAnalysis(Pass *P) {
  // Don't do this unless assertions are enabled.
#ifdef NDEBUG
  return;
#endif
  AnalysisUsage *AnUsage = TPM->findAnalysisUsage(P);
  const AnalysisUsage::VectorType &PreservedSet = AnUsage->getPreservedSet();

  // Verify preserved analysis
  for (AnalysisUsage::VectorType::const_iterator I = PreservedSet.begin(),
         E = PreservedSet.end(); I != E; ++I) {
    AnalysisID AID = *I;
    if (Pass *AP = findAnalysisPass(AID, true)) {
      TimeRegion PassTimer(getPassTimer(AP));
      AP->verifyAnalysis();
    }
  }
}

/// Remove Analysis not preserved by Pass P
void PMDataManager::removeNotPreservedAnalysis(Pass *P) {
  AnalysisUsage *AnUsage = TPM->findAnalysisUsage(P);
  if (AnUsage->getPreservesAll())
    return;

  const AnalysisUsage::VectorType &PreservedSet = AnUsage->getPreservedSet();
  for (DenseMap<AnalysisID, Pass*>::iterator I = AvailableAnalysis.begin(),
         E = AvailableAnalysis.end(); I != E; ) {
    DenseMap<AnalysisID, Pass*>::iterator Info = I++;
    if (Info->second->getAsImmutablePass() == nullptr &&
        std::find(PreservedSet.begin(), PreservedSet.end(), Info->first) ==
        PreservedSet.end()) {
      // Remove this analysis
      if (PassDebugging >= Details) {
        Pass *S = Info->second;
        dbgs() << " -- '" <<  P->getPassName() << "' is not preserving '";
        dbgs() << S->getPassName() << "'\n";
      }
      AvailableAnalysis.erase(Info);
    }
  }

  // Check inherited analysis also. If P is not preserving analysis
  // provided by parent manager then remove it here.
  for (unsigned Index = 0; Index < PMT_Last; ++Index) {

    if (!InheritedAnalysis[Index])
      continue;

    for (DenseMap<AnalysisID, Pass*>::iterator
           I = InheritedAnalysis[Index]->begin(),
           E = InheritedAnalysis[Index]->end(); I != E; ) {
      DenseMap<AnalysisID, Pass *>::iterator Info = I++;
      if (Info->second->getAsImmutablePass() == nullptr &&
          std::find(PreservedSet.begin(), PreservedSet.end(), Info->first) ==
             PreservedSet.end()) {
        // Remove this analysis
        if (PassDebugging >= Details) {
          Pass *S = Info->second;
          dbgs() << " -- '" <<  P->getPassName() << "' is not preserving '";
          dbgs() << S->getPassName() << "'\n";
        }
        InheritedAnalysis[Index]->erase(Info);
      }
    }
  }
}

/// Remove analysis passes that are not used any longer
void PMDataManager::removeDeadPasses(Pass *P, StringRef Msg,
                                     enum PassDebuggingString DBG_STR) {

  SmallVector<Pass *, 12> DeadPasses;

  // If this is a on the fly manager then it does not have TPM.
  if (!TPM)
    return;

  TPM->collectLastUses(DeadPasses, P);

  if (PassDebugging >= Details && !DeadPasses.empty()) {
    dbgs() << " -*- '" <<  P->getPassName();
    dbgs() << "' is the last user of following pass instances.";
    dbgs() << " Free these instances\n";
  }

  for (SmallVectorImpl<Pass *>::iterator I = DeadPasses.begin(),
         E = DeadPasses.end(); I != E; ++I)
    freePass(*I, Msg, DBG_STR);
}

void PMDataManager::freePass(Pass *P, StringRef Msg,
                             enum PassDebuggingString DBG_STR) {
  dumpPassInfo(P, FREEING_MSG, DBG_STR, Msg);

  {
    // If the pass crashes releasing memory, remember this.
    PassManagerPrettyStackEntry X(P);
    TimeRegion PassTimer(getPassTimer(P));

    P->releaseMemory();
  }

  AnalysisID PI = P->getPassID();
  if (const PassInfo *PInf = TPM->findAnalysisPassInfo(PI)) {
    // Remove the pass itself (if it is not already removed).
    AvailableAnalysis.erase(PI);

    // Remove all interfaces this pass implements, for which it is also
    // listed as the available implementation.
    const std::vector<const PassInfo*> &II = PInf->getInterfacesImplemented();
    for (unsigned i = 0, e = II.size(); i != e; ++i) {
      DenseMap<AnalysisID, Pass*>::iterator Pos =
        AvailableAnalysis.find(II[i]->getTypeInfo());
      if (Pos != AvailableAnalysis.end() && Pos->second == P)
        AvailableAnalysis.erase(Pos);
    }
  }
}

/// Add pass P into the PassVector. Update
/// AvailableAnalysis appropriately if ProcessAnalysis is true.
void PMDataManager::add(Pass *P, bool ProcessAnalysis) {
  std::unique_ptr<Pass> PPtr(P); // HLSL Change - take ownership of P
  // This manager is going to manage pass P. Set up analysis resolver
  // to connect them.
  AnalysisResolver *AR = new AnalysisResolver(*this);
  P->setResolver(AR); // HLSL Note: setResolver takes onwership of AR

  // If a FunctionPass F is the last user of ModulePass info M
  // then the F's manager, not F, records itself as a last user of M.
  SmallVector<Pass *, 12> TransferLastUses;

  if (!ProcessAnalysis) {
    // Add pass
    PassVector.push_back(P);
    PPtr.release(); // HLSL Change
    return;
  }

  // At the moment, this pass is the last user of all required passes.
  SmallVector<Pass *, 12> LastUses;
  SmallVector<Pass *, 8> RequiredPasses;
  SmallVector<AnalysisID, 8> ReqAnalysisNotAvailable;

  unsigned PDepth = this->getDepth();

  collectRequiredAnalysis(RequiredPasses,
                          ReqAnalysisNotAvailable, P);
  for (SmallVectorImpl<Pass *>::iterator I = RequiredPasses.begin(),
         E = RequiredPasses.end(); I != E; ++I) {
    Pass *PRequired = *I;
    unsigned RDepth = 0;

    assert(PRequired->getResolver() && "Analysis Resolver is not set");
    PMDataManager &DM = PRequired->getResolver()->getPMDataManager();
    RDepth = DM.getDepth();

    if (PDepth == RDepth)
      LastUses.push_back(PRequired);
    else if (PDepth > RDepth) {
      // Let the parent claim responsibility of last use
      TransferLastUses.push_back(PRequired);
      // Keep track of higher level analysis used by this manager.
      HigherLevelAnalysis.push_back(PRequired);
    } else
      llvm_unreachable("Unable to accommodate Required Pass");
  }

  // Set P as P's last user until someone starts using P.
  // However, if P is a Pass Manager then it does not need
  // to record its last user.
  if (!P->getAsPMDataManager())
    LastUses.push_back(P);
  TPM->setLastUser(LastUses, P);

  if (!TransferLastUses.empty()) {
    Pass *My_PM = getAsPass();
    TPM->setLastUser(TransferLastUses, My_PM);
    TransferLastUses.clear();
  }

  // Now, take care of required analyses that are not available.
  for (SmallVectorImpl<AnalysisID>::iterator
         I = ReqAnalysisNotAvailable.begin(),
         E = ReqAnalysisNotAvailable.end() ;I != E; ++I) {
    const PassInfo *PI = TPM->findAnalysisPassInfo(*I);
    Pass *AnalysisPass = PI->createPass();
    this->addLowerLevelRequiredPass(P, AnalysisPass);
  }

  // Take a note of analysis required and made available by this pass.
  // Remove the analysis not preserved by this pass
  removeNotPreservedAnalysis(P);
  recordAvailableAnalysis(P);

  // Add pass
  PassVector.push_back(P);
  PPtr.release(); // HLSL Change
}


/// Populate RP with analysis pass that are required by
/// pass P and are available. Populate RP_NotAvail with analysis
/// pass that are required by pass P but are not available.
void PMDataManager::collectRequiredAnalysis(SmallVectorImpl<Pass *> &RP,
                                       SmallVectorImpl<AnalysisID> &RP_NotAvail,
                                            Pass *P) {
  AnalysisUsage *AnUsage = TPM->findAnalysisUsage(P);
  const AnalysisUsage::VectorType &RequiredSet = AnUsage->getRequiredSet();
  for (AnalysisUsage::VectorType::const_iterator
         I = RequiredSet.begin(), E = RequiredSet.end(); I != E; ++I) {
    if (Pass *AnalysisPass = findAnalysisPass(*I, true))
      RP.push_back(AnalysisPass);
    else
      RP_NotAvail.push_back(*I);
  }

  const AnalysisUsage::VectorType &IDs = AnUsage->getRequiredTransitiveSet();
  for (AnalysisUsage::VectorType::const_iterator I = IDs.begin(),
         E = IDs.end(); I != E; ++I) {
    if (Pass *AnalysisPass = findAnalysisPass(*I, true))
      RP.push_back(AnalysisPass);
    else
      RP_NotAvail.push_back(*I);
  }
}

// All Required analyses should be available to the pass as it runs!  Here
// we fill in the AnalysisImpls member of the pass so that it can
// successfully use the getAnalysis() method to retrieve the
// implementations it needs.
//
void PMDataManager::initializeAnalysisImpl(Pass *P) {
  AnalysisUsage *AnUsage = TPM->findAnalysisUsage(P);

  for (AnalysisUsage::VectorType::const_iterator
         I = AnUsage->getRequiredSet().begin(),
         E = AnUsage->getRequiredSet().end(); I != E; ++I) {
    Pass *Impl = findAnalysisPass(*I, true);
    if (!Impl)
      // This may be analysis pass that is initialized on the fly.
      // If that is not the case then it will raise an assert when it is used.
      continue;
    AnalysisResolver *AR = P->getResolver();
    assert(AR && "Analysis Resolver is not set");
    AR->addAnalysisImplsPair(*I, Impl);
  }
}

/// Find the pass that implements Analysis AID. If desired pass is not found
/// then return NULL.
Pass *PMDataManager::findAnalysisPass(AnalysisID AID, bool SearchParent) {

  // Check if AvailableAnalysis map has one entry.
  DenseMap<AnalysisID, Pass*>::const_iterator I =  AvailableAnalysis.find(AID);

  if (I != AvailableAnalysis.end())
    return I->second;

  // Search Parents through TopLevelManager
  if (SearchParent)
    return TPM->findAnalysisPass(AID);

  return nullptr;
}

// Print list of passes that are last used by P.
void PMDataManager::dumpLastUses(Pass *P, unsigned Offset) const{

  SmallVector<Pass *, 12> LUses;

  // If this is a on the fly manager then it does not have TPM.
  if (!TPM)
    return;

  TPM->collectLastUses(LUses, P);

  for (SmallVectorImpl<Pass *>::iterator I = LUses.begin(),
         E = LUses.end(); I != E; ++I) {
    dbgs() << "--" << std::string(Offset*2, ' ');
    (*I)->dumpPassStructure(0);
  }
}

void PMDataManager::dumpPassArguments() const {
  for (SmallVectorImpl<Pass *>::const_iterator I = PassVector.begin(),
        E = PassVector.end(); I != E; ++I) {
    if (PMDataManager *PMD = (*I)->getAsPMDataManager())
      PMD->dumpPassArguments();
    else
      if (const PassInfo *PI =
            TPM->findAnalysisPassInfo((*I)->getPassID()))
        if (!PI->isAnalysisGroup())
          dbgs() << " -" << PI->getPassArgument();
  }
}

void PMDataManager::dumpPassInfo(Pass *P, enum PassDebuggingString S1,
                                 enum PassDebuggingString S2,
                                 StringRef Msg) {
  if (PassDebugging < Executions)
    return;
  dbgs() << "[" << sys::TimeValue::now().str() << "] " << (void *)this
         << std::string(getDepth() * 2 + 1, ' ');
  switch (S1) {
  case EXECUTION_MSG:
    dbgs() << "Executing Pass '" << P->getPassName();
    break;
  case MODIFICATION_MSG:
    dbgs() << "Made Modification '" << P->getPassName();
    break;
  case FREEING_MSG:
    dbgs() << " Freeing Pass '" << P->getPassName();
    break;
  default:
    break;
  }
  switch (S2) {
  case ON_BASICBLOCK_MSG:
    dbgs() << "' on BasicBlock '" << Msg << "'...\n";
    break;
  case ON_FUNCTION_MSG:
    dbgs() << "' on Function '" << Msg << "'...\n";
    break;
  case ON_MODULE_MSG:
    dbgs() << "' on Module '"  << Msg << "'...\n";
    break;
  case ON_REGION_MSG:
    dbgs() << "' on Region '"  << Msg << "'...\n";
    break;
  case ON_LOOP_MSG:
    dbgs() << "' on Loop '" << Msg << "'...\n";
    break;
  case ON_CG_MSG:
    dbgs() << "' on Call Graph Nodes '" << Msg << "'...\n";
    break;
  default:
    break;
  }
}

void PMDataManager::dumpRequiredSet(const Pass *P) const {
  if (PassDebugging < Details)
    return;

  AnalysisUsage analysisUsage;
  P->getAnalysisUsage(analysisUsage);
  dumpAnalysisUsage("Required", P, analysisUsage.getRequiredSet());
}

void PMDataManager::dumpPreservedSet(const Pass *P) const {
  if (PassDebugging < Details)
    return;

  AnalysisUsage analysisUsage;
  P->getAnalysisUsage(analysisUsage);
  dumpAnalysisUsage("Preserved", P, analysisUsage.getPreservedSet());
}

void PMDataManager::dumpAnalysisUsage(StringRef Msg, const Pass *P,
                                   const AnalysisUsage::VectorType &Set) const {
  assert(PassDebugging >= Details);
  if (Set.empty())
    return;
  dbgs() << (const void*)P << std::string(getDepth()*2+3, ' ') << Msg << " Analyses:";
  for (unsigned i = 0; i != Set.size(); ++i) {
    if (i) dbgs() << ',';
    const PassInfo *PInf = TPM->findAnalysisPassInfo(Set[i]);
    if (!PInf) {
      // Some preserved passes, such as AliasAnalysis, may not be initialized by
      // all drivers.
      dbgs() << " Uninitialized Pass";
      continue;
    }
    dbgs() << ' ' << PInf->getPassName();
  }
  dbgs() << '\n';
}

/// Add RequiredPass into list of lower level passes required by pass P.
/// RequiredPass is run on the fly by Pass Manager when P requests it
/// through getAnalysis interface.
/// This should be handled by specific pass manager.
void PMDataManager::addLowerLevelRequiredPass(Pass *P, Pass *RequiredPass) {
  if (TPM) {
    TPM->dumpArguments();
    TPM->dumpPasses();
  }

  // Module Level pass may required Function Level analysis info
  // (e.g. dominator info). Pass manager uses on the fly function pass manager
  // to provide this on demand. In that case, in Pass manager terminology,
  // module level pass is requiring lower level analysis info managed by
  // lower level pass manager.

  // When Pass manager is not able to order required analysis info, Pass manager
  // checks whether any lower level manager will be able to provide this
  // analysis info on demand or not.
#ifndef NDEBUG
  dbgs() << "Unable to schedule '" << RequiredPass->getPassName();
  dbgs() << "' required by '" << P->getPassName() << "'\n";
#endif
  llvm_unreachable("Unable to schedule pass");
}

Pass *PMDataManager::getOnTheFlyPass(Pass *P, AnalysisID PI, Function &F) {
  llvm_unreachable("Unable to find on the fly pass");
}

// Destructor
PMDataManager::~PMDataManager() {
  for (SmallVectorImpl<Pass *>::iterator I = PassVector.begin(),
         E = PassVector.end(); I != E; ++I)
    delete *I;
}

//===----------------------------------------------------------------------===//
// NOTE: Is this the right place to define this method ?
// getAnalysisIfAvailable - Return analysis result or null if it doesn't exist.
Pass *AnalysisResolver::getAnalysisIfAvailable(AnalysisID ID, bool dir) const {
  return PM.findAnalysisPass(ID, dir);
}

Pass *AnalysisResolver::findImplPass(Pass *P, AnalysisID AnalysisPI,
                                     Function &F) {
  return PM.getOnTheFlyPass(P, AnalysisPI, F);
}

//===----------------------------------------------------------------------===//
// BBPassManager implementation

/// Execute all of the passes scheduled for execution by invoking
/// runOnBasicBlock method.  Keep track of whether any of the passes modifies
/// the function, and if so, return true.
bool BBPassManager::runOnFunction(Function &F) {
  if (F.isDeclaration())
    return false;

  bool Changed = doInitialization(F);

  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
    for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
      BasicBlockPass *BP = getContainedPass(Index);
      bool LocalChanged = false;

      dumpPassInfo(BP, EXECUTION_MSG, ON_BASICBLOCK_MSG, I->getName());
      dumpRequiredSet(BP);

      initializeAnalysisImpl(BP);

      {
        // If the pass crashes, remember this.
        PassManagerPrettyStackEntry X(BP, *I);
        TimeRegion PassTimer(getPassTimer(BP));

        LocalChanged |= BP->runOnBasicBlock(*I);
      }

      Changed |= LocalChanged;
      if (LocalChanged)
        dumpPassInfo(BP, MODIFICATION_MSG, ON_BASICBLOCK_MSG,
                     I->getName());
      dumpPreservedSet(BP);

      verifyPreservedAnalysis(BP);
      removeNotPreservedAnalysis(BP);
      recordAvailableAnalysis(BP);
      removeDeadPasses(BP, I->getName(), ON_BASICBLOCK_MSG);
    }

  return doFinalization(F) || Changed;
}

// Implement doInitialization and doFinalization
bool BBPassManager::doInitialization(Module &M) {
  bool Changed = false;

  for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index)
    Changed |= getContainedPass(Index)->doInitialization(M);

  return Changed;
}

bool BBPassManager::doFinalization(Module &M) {
  bool Changed = false;

  for (int Index = getNumContainedPasses() - 1; Index >= 0; --Index)
    Changed |= getContainedPass(Index)->doFinalization(M);

  return Changed;
}

bool BBPassManager::doInitialization(Function &F) {
  bool Changed = false;

  for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
    BasicBlockPass *BP = getContainedPass(Index);
    Changed |= BP->doInitialization(F);
  }

  return Changed;
}

bool BBPassManager::doFinalization(Function &F) {
  bool Changed = false;

  for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
    BasicBlockPass *BP = getContainedPass(Index);
    Changed |= BP->doFinalization(F);
  }

  return Changed;
}


//===----------------------------------------------------------------------===//
// FunctionPassManager implementation

/// Create new Function pass manager
FunctionPassManager::FunctionPassManager(Module *m) : M(m) {
  FPM = new FunctionPassManagerImpl();
  // FPM is the top level manager.
  FPM->setTopLevelManager(FPM);

  AnalysisResolver *AR = new (std::nothrow)AnalysisResolver(*FPM); // HLSL Change: nothrow and recover
  if (!AR) {
    delete FPM;
    throw std::bad_alloc();
  }
  FPM->setResolver(AR);
}

FunctionPassManager::~FunctionPassManager() {
  delete FPM;
}

void FunctionPassManager::add(Pass *P) {
  // HLSL Change Starts
  FPM->HLSLPrintAfterAll = this->HLSLPrintAfterAll;
  FPM->HLSLPrintAfter = this->HLSLPrintAfter;
  std::unique_ptr<Pass> PPtr(P); // take ownership of P, even on failure paths
  if (TrackPassOS) {
    P->dumpConfig(*TrackPassOS);
    (*TrackPassOS) << '\n';
  }
  PPtr.release();
  // HLSL Change Ends
  FPM->add(P);
}

/// run - Execute all of the passes scheduled for execution.  Keep
/// track of whether any of the passes modifies the function, and if
/// so, return true.
///
bool FunctionPassManager::run(Function &F) {
  if (std::error_code EC = F.materialize())
    report_fatal_error("Error reading bitcode file: " + EC.message());
  return FPM->run(F);
}


/// doInitialization - Run all of the initializers for the function passes.
///
bool FunctionPassManager::doInitialization() {
  return FPM->doInitialization(*M);
}

/// doFinalization - Run all of the finalizers for the function passes.
///
bool FunctionPassManager::doFinalization() {
  return FPM->doFinalization(*M);
}

//===----------------------------------------------------------------------===//
// FunctionPassManagerImpl implementation
//
bool FunctionPassManagerImpl::doInitialization(Module &M) {
  bool Changed = false;

  dumpArguments();
  dumpPasses();

  for (ImmutablePass *ImPass : getImmutablePasses())
    Changed |= ImPass->doInitialization(M);

  for (unsigned Index = 0; Index < getNumContainedManagers(); ++Index)
    Changed |= getContainedManager(Index)->doInitialization(M);

  return Changed;
}

bool FunctionPassManagerImpl::doFinalization(Module &M) {
  bool Changed = false;

  for (int Index = getNumContainedManagers() - 1; Index >= 0; --Index)
    Changed |= getContainedManager(Index)->doFinalization(M);

  for (ImmutablePass *ImPass : getImmutablePasses())
    Changed |= ImPass->doFinalization(M);

  return Changed;
}

/// cleanup - After running all passes, clean up pass manager cache.
void FPPassManager::cleanup() {
 for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
    FunctionPass *FP = getContainedPass(Index);
    AnalysisResolver *AR = FP->getResolver();
    assert(AR && "Analysis Resolver is not set");
    AR->clearAnalysisImpls();
 }
}

void FunctionPassManagerImpl::releaseMemoryOnTheFly() {
  if (!wasRun)
    return;
  for (unsigned Index = 0; Index < getNumContainedManagers(); ++Index) {
    FPPassManager *FPPM = getContainedManager(Index);
    for (unsigned Index = 0; Index < FPPM->getNumContainedPasses(); ++Index) {
      FPPM->getContainedPass(Index)->releaseMemory();
    }
  }
  wasRun = false;
}

// Execute all the passes managed by this top level manager.
// Return true if any function is modified by a pass.
bool FunctionPassManagerImpl::run(Function &F) {
  bool Changed = false;
  TimingInfo::createTheTimeInfo();

  initializeAllAnalysisInfo();
  for (unsigned Index = 0; Index < getNumContainedManagers(); ++Index) {
    Changed |= getContainedManager(Index)->runOnFunction(F);
    F.getContext().yield();
  }

  for (unsigned Index = 0; Index < getNumContainedManagers(); ++Index)
    getContainedManager(Index)->cleanup();

  wasRun = true;
  return Changed;
}

//===----------------------------------------------------------------------===//
// FPPassManager implementation

char FPPassManager::ID = 0;
/// Print passes managed by this manager
void FPPassManager::dumpPassStructure(unsigned Offset) {
  dbgs().indent(Offset*2) << "FunctionPass Manager\n";
  for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
    FunctionPass *FP = getContainedPass(Index);
    FP->dumpPassStructure(Offset + 1);
    dumpLastUses(FP, Offset+1);
  }
}


/// Execute all of the passes scheduled for execution by invoking
/// runOnFunction method.  Keep track of whether any of the passes modifies
/// the function, and if so, return true.
bool FPPassManager::runOnFunction(Function &F) {
  if (F.isDeclaration())
    return false;

  bool Changed = false;

  // Collect inherited analysis from Module level pass manager.
  populateInheritedAnalysis(TPM->activeStack);

  for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
    FunctionPass *FP = getContainedPass(Index);
    bool LocalChanged = false;

    dumpPassInfo(FP, EXECUTION_MSG, ON_FUNCTION_MSG, F.getName());
    dumpRequiredSet(FP);

    initializeAnalysisImpl(FP);

    {
      PassManagerPrettyStackEntry X(FP, F);
      TimeRegion PassTimer(getPassTimer(FP));

      LocalChanged |= FP->runOnFunction(F);
    }

    Changed |= LocalChanged;
    if (LocalChanged)
      dumpPassInfo(FP, MODIFICATION_MSG, ON_FUNCTION_MSG, F.getName());
    dumpPreservedSet(FP);

    verifyPreservedAnalysis(FP);
    removeNotPreservedAnalysis(FP);
    recordAvailableAnalysis(FP);
    removeDeadPasses(FP, F.getName(), ON_FUNCTION_MSG);
  }
  return Changed;
}

bool FPPassManager::runOnModule(Module &M) {
  bool Changed = false;

  for (Function &F : M)
    Changed |= runOnFunction(F);

  return Changed;
}

bool FPPassManager::doInitialization(Module &M) {
  bool Changed = false;

  for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index)
    Changed |= getContainedPass(Index)->doInitialization(M);

  return Changed;
}

bool FPPassManager::doFinalization(Module &M) {
  bool Changed = false;

  for (int Index = getNumContainedPasses() - 1; Index >= 0; --Index)
    Changed |= getContainedPass(Index)->doFinalization(M);

  return Changed;
}

//===----------------------------------------------------------------------===//
// MPPassManager implementation

/// Execute all of the passes scheduled for execution by invoking
/// runOnModule method.  Keep track of whether any of the passes modifies
/// the module, and if so, return true.
bool
MPPassManager::runOnModule(Module &M) {
  bool Changed = false;

  // Initialize on-the-fly passes
  for (auto &OnTheFlyManager : OnTheFlyManagers) {
    FunctionPassManagerImpl *FPP = OnTheFlyManager.second;
    Changed |= FPP->doInitialization(M);
  }

  // Initialize module passes
  for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index)
    Changed |= getContainedPass(Index)->doInitialization(M);

  for (unsigned Index = 0; Index < getNumContainedPasses(); ++Index) {
    ModulePass *MP = getContainedPass(Index);
    bool LocalChanged = false;

    dumpPassInfo(MP, EXECUTION_MSG, ON_MODULE_MSG, M.getModuleIdentifier());
    dumpRequiredSet(MP);

    initializeAnalysisImpl(MP);

    {
      PassManagerPrettyStackEntry X(MP, M);
      TimeRegion PassTimer(getPassTimer(MP));

      LocalChanged |= MP->runOnModule(M);
    }

    Changed |= LocalChanged;
    if (LocalChanged)
      dumpPassInfo(MP, MODIFICATION_MSG, ON_MODULE_MSG,
                   M.getModuleIdentifier());
    dumpPreservedSet(MP);

    verifyPreservedAnalysis(MP);
    removeNotPreservedAnalysis(MP);
    recordAvailableAnalysis(MP);
    removeDeadPasses(MP, M.getModuleIdentifier(), ON_MODULE_MSG);
  }

  // Finalize module passes
  for (int Index = getNumContainedPasses() - 1; Index >= 0; --Index)
    Changed |= getContainedPass(Index)->doFinalization(M);

  // Finalize on-the-fly passes
  for (auto &OnTheFlyManager : OnTheFlyManagers) {
    FunctionPassManagerImpl *FPP = OnTheFlyManager.second;
    // We don't know when is the last time an on-the-fly pass is run,
    // so we need to releaseMemory / finalize here
    FPP->releaseMemoryOnTheFly();
    Changed |= FPP->doFinalization(M);
  }

  return Changed;
}

/// Add RequiredPass into list of lower level passes required by pass P.
/// RequiredPass is run on the fly by Pass Manager when P requests it
/// through getAnalysis interface.
void MPPassManager::addLowerLevelRequiredPass(Pass *P, Pass *RequiredPass) {
  assert(P->getPotentialPassManagerType() == PMT_ModulePassManager &&
         "Unable to handle Pass that requires lower level Analysis pass");
  assert((P->getPotentialPassManagerType() <
          RequiredPass->getPotentialPassManagerType()) &&
         "Unable to handle Pass that requires lower level Analysis pass");
  if (!RequiredPass)
    return;

  FunctionPassManagerImpl *FPP = OnTheFlyManagers[P];
  if (!FPP) {
    FPP = new FunctionPassManagerImpl();
    // FPP is the top level manager.
    FPP->setTopLevelManager(FPP);

    OnTheFlyManagers[P] = FPP;
  }
  const PassInfo *RequiredPassPI =
      TPM->findAnalysisPassInfo(RequiredPass->getPassID());

  Pass *FoundPass = nullptr;
  if (RequiredPassPI && RequiredPassPI->isAnalysis()) {
    FoundPass =
      ((PMTopLevelManager*)FPP)->findAnalysisPass(RequiredPass->getPassID());
  }
  if (!FoundPass) {
    FoundPass = RequiredPass;
    // This should be guaranteed to add RequiredPass to the passmanager given
    // that we checked for an available analysis above.
    FPP->add(RequiredPass);
  }
  // Register P as the last user of FoundPass or RequiredPass.
  SmallVector<Pass *, 1> LU;
  LU.push_back(FoundPass);
  FPP->setLastUser(LU,  P);
}

/// Return function pass corresponding to PassInfo PI, that is
/// required by module pass MP. Instantiate analysis pass, by using
/// its runOnFunction() for function F.
Pass* MPPassManager::getOnTheFlyPass(Pass *MP, AnalysisID PI, Function &F){
  FunctionPassManagerImpl *FPP = OnTheFlyManagers[MP];
  assert(FPP && "Unable to find on the fly pass");

  FPP->releaseMemoryOnTheFly();
  FPP->run(F);
  return ((PMTopLevelManager*)FPP)->findAnalysisPass(PI);
}


//===----------------------------------------------------------------------===//
// PassManagerImpl implementation

//
/// run - Execute all of the passes scheduled for execution.  Keep track of
/// whether any of the passes modifies the module, and if so, return true.
bool PassManagerImpl::run(Module &M) {
  bool Changed = false;
  TimingInfo::createTheTimeInfo();

  dumpArguments();
  dumpPasses();

  for (ImmutablePass *ImPass : getImmutablePasses())
    Changed |= ImPass->doInitialization(M);

  initializeAllAnalysisInfo();
  for (unsigned Index = 0; Index < getNumContainedManagers(); ++Index) {
    Changed |= getContainedManager(Index)->runOnModule(M);
    M.getContext().yield();
  }

  for (ImmutablePass *ImPass : getImmutablePasses())
    Changed |= ImPass->doFinalization(M);

  return Changed;
}

//===----------------------------------------------------------------------===//
// PassManager implementation

/// Create new pass manager
PassManager::PassManager() {
  PM = new PassManagerImpl();
  // PM is the top level manager
  PM->setTopLevelManager(PM);
}

PassManager::~PassManager() {
  delete PM;
}

void PassManager::add(Pass *P) {
  // HLSL Change Starts
  PM->HLSLPrintAfterAll = this->HLSLPrintAfterAll;
  PM->HLSLPrintAfter = this->HLSLPrintAfter;
  std::unique_ptr<Pass> PPtr(P); // take ownership of P, even on failure paths
  if (TrackPassOS) {
    P->dumpConfig(*TrackPassOS);
    (*TrackPassOS) << '\n';
  }
  PPtr.release();
  // HLSL Change Ends
  PM->add(P);
}

/// run - Execute all of the passes scheduled for execution.  Keep track of
/// whether any of the passes modifies the module, and if so, return true.
bool PassManager::run(Module &M) {
  return PM->run(M);
}

//===----------------------------------------------------------------------===//
// TimingInfo implementation

bool llvm::TimePassesIsEnabled = false;
#if 0 // HLSL Change Starts - option pending
static cl::opt<bool,true>
EnableTiming("time-passes", cl::location(TimePassesIsEnabled),
            cl::desc("Time each pass, printing elapsed time for each on exit"));
#endif

// createTheTimeInfo - This method either initializes the TheTimeInfo pointer to
// a non-null value (if the -time-passes option is enabled) or it leaves it
// null.  It may be called multiple times.
void TimingInfo::createTheTimeInfo() {
  if (!TimePassesIsEnabled || TheTimeInfo) return;

  // Constructed the first time this is called, iff -time-passes is enabled.
  // This guarantees that the object will be constructed before static globals,
  // thus it will be destroyed before them.
  static ManagedStatic<TimingInfo> TTI;
  TheTimeInfo = &*TTI;
}

/// If TimingInfo is enabled then start pass timer.
Timer *llvm::getPassTimer(Pass *P) {
  if (TheTimeInfo)
    return TheTimeInfo->getPassTimer(P);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// PMStack implementation
//

// Pop Pass Manager from the stack and clear its analysis info.
void PMStack::pop() {

  PMDataManager *Top = this->top();
  Top->initializeAnalysisInfo();

  S.pop_back();
}

// Push PM on the stack and set its top level manager.
void PMStack::push(PMDataManager *PM) {
  assert(PM && "Unable to push. Pass Manager expected");
  assert(PM->getDepth()==0 && "Pass Manager depth set too early");

  if (!this->empty()) {
    assert(PM->getPassManagerType() > this->top()->getPassManagerType()
           && "pushing bad pass manager to PMStack");
    PMTopLevelManager *TPM = this->top()->getTopLevelManager();

    assert(TPM && "Unable to find top level manager");
    TPM->addIndirectPassManager(PM);
    PM->setTopLevelManager(TPM);
    PM->setDepth(this->top()->getDepth()+1);
  } else {
    assert((PM->getPassManagerType() == PMT_ModulePassManager
           || PM->getPassManagerType() == PMT_FunctionPassManager)
           && "pushing bad pass manager to PMStack");
    PM->setDepth(1);
  }

  S.push_back(PM);
}

// Dump content of the pass manager stack.
void PMStack::dump() const {
  for (PMDataManager *Manager : S)
    dbgs() << Manager->getAsPass()->getPassName() << ' ';

  if (!S.empty())
    dbgs() << '\n';
}

/// Find appropriate Module Pass Manager in the PM Stack and
/// add self into that manager.
void ModulePass::assignPassManager(PMStack &PMS,
                                   PassManagerType PreferredType) {
  // Find Module Pass Manager
  while (!PMS.empty()) {
    PassManagerType TopPMType = PMS.top()->getPassManagerType();
    if (TopPMType == PreferredType)
      break; // We found desired pass manager
    else if (TopPMType > PMT_ModulePassManager)
      PMS.pop();    // Pop children pass managers
    else
      break;
  }
  assert(!PMS.empty() && "Unable to find appropriate Pass Manager");
  PMS.top()->add(this);
}

/// Find appropriate Function Pass Manager or Call Graph Pass Manager
/// in the PM Stack and add self into that manager.
void FunctionPass::assignPassManager(PMStack &PMS,
                                     PassManagerType PreferredType) {
  std::unique_ptr<FunctionPass> thisPtr(this); // HLSL Change

  // Find Function Pass Manager
  while (!PMS.empty()) {
    if (PMS.top()->getPassManagerType() > PMT_FunctionPassManager)
      PMS.pop();
    else
      break;
  }

  // Create new Function Pass Manager if needed.
  FPPassManager *FPP;
  if (PMS.top()->getPassManagerType() == PMT_FunctionPassManager) {
    FPP = (FPPassManager *)PMS.top();
  } else {
    assert(!PMS.empty() && "Unable to create Function Pass Manager");
    PMDataManager *PMD = PMS.top();

    // [1] Create new Function Pass Manager
    FPP = new FPPassManager();
    std::unique_ptr<FPPassManager> NewFPP(FPP); // HLSL Change
    FPP->populateInheritedAnalysis(PMS);

    // [2] Set up new manager's top level manager
    PMTopLevelManager *TPM = PMD->getTopLevelManager();
    TPM->addIndirectPassManager(FPP);

    // [3] Assign manager to manage this new manager. This may create
    // and push new managers into PMS
    NewFPP.release(); // HLSL Change: assignPassManager transfers ownership of 'this'...
    FPP->assignPassManager(PMS, PMD->getPassManagerType());

    // [4] Push new manager into PMS
    PMS.push(FPP);
  }

  // Assign FPP as the manager of this pass.
  thisPtr.release();
  FPP->add(this);
}

/// Find appropriate Basic Pass Manager or Call Graph Pass Manager
/// in the PM Stack and add self into that manager.
void BasicBlockPass::assignPassManager(PMStack &PMS,
                                       PassManagerType PreferredType) {
  BBPassManager *BBP;

  // Basic Pass Manager is a leaf pass manager. It does not handle
  // any other pass manager.
  if (!PMS.empty() &&
      PMS.top()->getPassManagerType() == PMT_BasicBlockPassManager) {
    BBP = (BBPassManager *)PMS.top();
  } else {
    // If leaf manager is not Basic Block Pass manager then create new
    // basic Block Pass manager.
    assert(!PMS.empty() && "Unable to create BasicBlock Pass Manager");
    PMDataManager *PMD = PMS.top();

    // [1] Create new Basic Block Manager
    BBP = new BBPassManager();

    // [2] Set up new manager's top level manager
    // Basic Block Pass Manager does not live by itself
    PMTopLevelManager *TPM = PMD->getTopLevelManager();
    TPM->addIndirectPassManager(BBP);

    // [3] Assign manager to manage this new manager. This may create
    // and push new managers into PMS
    BBP->assignPassManager(PMS, PreferredType);

    // [4] Push new manager into PMS
    PMS.push(BBP);
  }

  // Assign BBP as the manager of this pass.
  BBP->add(this);
}

PassManagerBase::~PassManagerBase() {}
