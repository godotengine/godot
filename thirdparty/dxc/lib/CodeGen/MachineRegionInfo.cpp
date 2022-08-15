
#include "llvm/CodeGen/MachineRegionInfo.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/RegionInfoImpl.h"
#include "llvm/CodeGen/MachinePostDominators.h"

#define DEBUG_TYPE "region"

using namespace llvm;

STATISTIC(numMachineRegions,       "The # of machine regions");
STATISTIC(numMachineSimpleRegions, "The # of simple machine regions");

namespace llvm {
template class RegionBase<RegionTraits<MachineFunction>>;
template class RegionNodeBase<RegionTraits<MachineFunction>>;
template class RegionInfoBase<RegionTraits<MachineFunction>>;
}

//===----------------------------------------------------------------------===//
// MachineRegion implementation
//

MachineRegion::MachineRegion(MachineBasicBlock *Entry, MachineBasicBlock *Exit,
                             MachineRegionInfo* RI,
                             MachineDominatorTree *DT, MachineRegion *Parent) :
  RegionBase<RegionTraits<MachineFunction>>(Entry, Exit, RI, DT, Parent) {

}

MachineRegion::~MachineRegion() { }

//===----------------------------------------------------------------------===//
// MachineRegionInfo implementation
//

MachineRegionInfo::MachineRegionInfo() :
  RegionInfoBase<RegionTraits<MachineFunction>>() {

}

MachineRegionInfo::~MachineRegionInfo() {

}

void MachineRegionInfo::updateStatistics(MachineRegion *R) {
  ++numMachineRegions;

  // TODO: Slow. Should only be enabled if -stats is used.
  if (R->isSimple())
    ++numMachineSimpleRegions;
}

void MachineRegionInfo::recalculate(MachineFunction &F,
                                    MachineDominatorTree *DT_,
                                    MachinePostDominatorTree *PDT_,
                                    MachineDominanceFrontier *DF_) {
  DT = DT_;
  PDT = PDT_;
  DF = DF_;

  MachineBasicBlock *Entry = GraphTraits<MachineFunction*>::getEntryNode(&F);

  TopLevelRegion = new MachineRegion(Entry, nullptr, this, DT, nullptr);
  updateStatistics(TopLevelRegion);
  calculate(F);
}

//===----------------------------------------------------------------------===//
// MachineRegionInfoPass implementation
//

MachineRegionInfoPass::MachineRegionInfoPass() : MachineFunctionPass(ID) {
  initializeMachineRegionInfoPassPass(*PassRegistry::getPassRegistry());
}

MachineRegionInfoPass::~MachineRegionInfoPass() {

}

bool MachineRegionInfoPass::runOnMachineFunction(MachineFunction &F) {
  releaseMemory();

  auto DT = &getAnalysis<MachineDominatorTree>();
  auto PDT = &getAnalysis<MachinePostDominatorTree>();
  auto DF = &getAnalysis<MachineDominanceFrontier>();

  RI.recalculate(F, DT, PDT, DF);
  return false;
}

void MachineRegionInfoPass::releaseMemory() {
  RI.releaseMemory();
}

void MachineRegionInfoPass::verifyAnalysis() const {
  // Only do verification when user wants to, otherwise this expensive check
  // will be invoked by PMDataManager::verifyPreservedAnalysis when
  // a regionpass (marked PreservedAll) finish.
  if (MachineRegionInfo::VerifyRegionInfo)
    RI.verifyAnalysis();
}

void MachineRegionInfoPass::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequiredTransitive<DominatorTreeWrapperPass>();
  AU.addRequired<PostDominatorTree>();
  AU.addRequired<DominanceFrontier>();
}

void MachineRegionInfoPass::print(raw_ostream &OS, const Module *) const {
  RI.print(OS);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
void MachineRegionInfoPass::dump() const {
  RI.dump();
}
#endif

char MachineRegionInfoPass::ID = 0;

INITIALIZE_PASS_BEGIN(MachineRegionInfoPass, "regions",
                "Detect single entry single exit regions", true, true)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachinePostDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachineDominanceFrontier)
INITIALIZE_PASS_END(MachineRegionInfoPass, "regions",
                "Detect single entry single exit regions", true, true)

// Create methods available outside of this file, to use them
// "include/llvm/LinkAllPasses.h". Otherwise the pass would be deleted by
// the link time optimization.

namespace llvm {
  FunctionPass *createMachineRegionInfoPass() {
    return new MachineRegionInfoPass();
  }
}

