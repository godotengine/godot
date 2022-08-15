//===- MachineBlockFrequencyInfo.cpp - MBB Frequency Analysis -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Loops should be simplified before this analysis.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineBlockFrequencyInfo.h"
#include "llvm/Analysis/BlockFrequencyInfoImpl.h"
#include "llvm/CodeGen/MachineBranchProbabilityInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GraphWriter.h"

using namespace llvm;

#define DEBUG_TYPE "block-freq"

#ifndef NDEBUG
enum GVDAGType {
  GVDT_None,
  GVDT_Fraction,
  GVDT_Integer
};

static cl::opt<GVDAGType>
ViewMachineBlockFreqPropagationDAG("view-machine-block-freq-propagation-dags",
                                   cl::Hidden,
          cl::desc("Pop up a window to show a dag displaying how machine block "
                   "frequencies propagate through the CFG."),
          cl::values(
            clEnumValN(GVDT_None, "none",
                       "do not display graphs."),
            clEnumValN(GVDT_Fraction, "fraction", "display a graph using the "
                       "fractional block frequency representation."),
            clEnumValN(GVDT_Integer, "integer", "display a graph using the raw "
                       "integer fractional block frequency representation."),
            clEnumValEnd));

namespace llvm {

template <>
struct GraphTraits<MachineBlockFrequencyInfo *> {
  typedef const MachineBasicBlock NodeType;
  typedef MachineBasicBlock::const_succ_iterator ChildIteratorType;
  typedef MachineFunction::const_iterator nodes_iterator;

  static inline
  const NodeType *getEntryNode(const MachineBlockFrequencyInfo *G) {
    return G->getFunction()->begin();
  }

  static ChildIteratorType child_begin(const NodeType *N) {
    return N->succ_begin();
  }

  static ChildIteratorType child_end(const NodeType *N) {
    return N->succ_end();
  }

  static nodes_iterator nodes_begin(const MachineBlockFrequencyInfo *G) {
    return G->getFunction()->begin();
  }

  static nodes_iterator nodes_end(const MachineBlockFrequencyInfo *G) {
    return G->getFunction()->end();
  }
};

template<>
struct DOTGraphTraits<MachineBlockFrequencyInfo*> :
    public DefaultDOTGraphTraits {
  explicit DOTGraphTraits(bool isSimple=false) :
    DefaultDOTGraphTraits(isSimple) {}

  static std::string getGraphName(const MachineBlockFrequencyInfo *G) {
    return G->getFunction()->getName();
  }

  std::string getNodeLabel(const MachineBasicBlock *Node,
                           const MachineBlockFrequencyInfo *Graph) {
    std::string Result;
    raw_string_ostream OS(Result);

    OS << Node->getName().str() << ":";
    switch (ViewMachineBlockFreqPropagationDAG) {
    case GVDT_Fraction:
      Graph->printBlockFreq(OS, Node);
      break;
    case GVDT_Integer:
      OS << Graph->getBlockFreq(Node).getFrequency();
      break;
    case GVDT_None:
      llvm_unreachable("If we are not supposed to render a graph we should "
                       "never reach this point.");
    }

    return Result;
  }
};


} // end namespace llvm
#endif

INITIALIZE_PASS_BEGIN(MachineBlockFrequencyInfo, "machine-block-freq",
                      "Machine Block Frequency Analysis", true, true)
INITIALIZE_PASS_DEPENDENCY(MachineBranchProbabilityInfo)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfo)
INITIALIZE_PASS_END(MachineBlockFrequencyInfo, "machine-block-freq",
                    "Machine Block Frequency Analysis", true, true)

char MachineBlockFrequencyInfo::ID = 0;


MachineBlockFrequencyInfo::
MachineBlockFrequencyInfo() :MachineFunctionPass(ID) {
  initializeMachineBlockFrequencyInfoPass(*PassRegistry::getPassRegistry());
}

MachineBlockFrequencyInfo::~MachineBlockFrequencyInfo() {}

void MachineBlockFrequencyInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<MachineBranchProbabilityInfo>();
  AU.addRequired<MachineLoopInfo>();
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

bool MachineBlockFrequencyInfo::runOnMachineFunction(MachineFunction &F) {
  MachineBranchProbabilityInfo &MBPI =
      getAnalysis<MachineBranchProbabilityInfo>();
  MachineLoopInfo &MLI = getAnalysis<MachineLoopInfo>();
  if (!MBFI)
    MBFI.reset(new ImplType);
  MBFI->doFunction(&F, &MBPI, &MLI);
#ifndef NDEBUG
  if (ViewMachineBlockFreqPropagationDAG != GVDT_None) {
    view();
  }
#endif
  return false;
}

void MachineBlockFrequencyInfo::releaseMemory() { MBFI.reset(); }

/// Pop up a ghostview window with the current block frequency propagation
/// rendered using dot.
void MachineBlockFrequencyInfo::view() const {
// This code is only for debugging.
#ifndef NDEBUG
  ViewGraph(const_cast<MachineBlockFrequencyInfo *>(this),
            "MachineBlockFrequencyDAGs");
#else
  errs() << "MachineBlockFrequencyInfo::view is only available in debug builds "
    "on systems with Graphviz or gv!\n";
#endif // NDEBUG
}

BlockFrequency MachineBlockFrequencyInfo::
getBlockFreq(const MachineBasicBlock *MBB) const {
  return MBFI ? MBFI->getBlockFreq(MBB) : 0;
}

const MachineFunction *MachineBlockFrequencyInfo::getFunction() const {
  return MBFI ? MBFI->getFunction() : nullptr;
}

raw_ostream &
MachineBlockFrequencyInfo::printBlockFreq(raw_ostream &OS,
                                          const BlockFrequency Freq) const {
  return MBFI ? MBFI->printBlockFreq(OS, Freq) : OS;
}

raw_ostream &
MachineBlockFrequencyInfo::printBlockFreq(raw_ostream &OS,
                                          const MachineBasicBlock *MBB) const {
  return MBFI ? MBFI->printBlockFreq(OS, MBB) : OS;
}

uint64_t MachineBlockFrequencyInfo::getEntryFreq() const {
  return MBFI ? MBFI->getEntryFreq() : 0;
}
