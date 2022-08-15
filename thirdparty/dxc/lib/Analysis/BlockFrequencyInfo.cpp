//===- BlockFrequencyInfo.cpp - Block Frequency Analysis ------------------===//
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

#include "llvm/Analysis/BlockFrequencyInfo.h"
#include "llvm/Analysis/BlockFrequencyInfoImpl.h"
#include "llvm/Analysis/BranchProbabilityInfo.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/Passes.h"
#include "llvm/IR/CFG.h"
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
ViewBlockFreqPropagationDAG("view-block-freq-propagation-dags", cl::Hidden,
          cl::desc("Pop up a window to show a dag displaying how block "
                   "frequencies propagation through the CFG."),
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
struct GraphTraits<BlockFrequencyInfo *> {
  typedef const BasicBlock NodeType;
  typedef succ_const_iterator ChildIteratorType;
  typedef Function::const_iterator nodes_iterator;

  static inline const NodeType *getEntryNode(const BlockFrequencyInfo *G) {
    return G->getFunction()->begin();
  }
  static ChildIteratorType child_begin(const NodeType *N) {
    return succ_begin(N);
  }
  static ChildIteratorType child_end(const NodeType *N) {
    return succ_end(N);
  }
  static nodes_iterator nodes_begin(const BlockFrequencyInfo *G) {
    return G->getFunction()->begin();
  }
  static nodes_iterator nodes_end(const BlockFrequencyInfo *G) {
    return G->getFunction()->end();
  }
};

template<>
struct DOTGraphTraits<BlockFrequencyInfo*> : public DefaultDOTGraphTraits {
  explicit DOTGraphTraits(bool isSimple=false) :
    DefaultDOTGraphTraits(isSimple) {}

  static std::string getGraphName(const BlockFrequencyInfo *G) {
    return G->getFunction()->getName();
  }

  std::string getNodeLabel(const BasicBlock *Node,
                           const BlockFrequencyInfo *Graph) {
    std::string Result;
    raw_string_ostream OS(Result);

    OS << Node->getName() << ":";
    switch (ViewBlockFreqPropagationDAG) {
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

INITIALIZE_PASS_BEGIN(BlockFrequencyInfo, "block-freq",
                      "Block Frequency Analysis", true, true)
INITIALIZE_PASS_DEPENDENCY(BranchProbabilityInfo)
INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)
INITIALIZE_PASS_END(BlockFrequencyInfo, "block-freq",
                    "Block Frequency Analysis", true, true)

char BlockFrequencyInfo::ID = 0;


BlockFrequencyInfo::BlockFrequencyInfo() : FunctionPass(ID) {
  initializeBlockFrequencyInfoPass(*PassRegistry::getPassRegistry());
}

BlockFrequencyInfo::~BlockFrequencyInfo() {}

void BlockFrequencyInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<BranchProbabilityInfo>();
  AU.addRequired<LoopInfoWrapperPass>();
  AU.setPreservesAll();
}

bool BlockFrequencyInfo::runOnFunction(Function &F) {
  BranchProbabilityInfo &BPI = getAnalysis<BranchProbabilityInfo>();
  LoopInfo &LI = getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  if (!BFI)
    BFI.reset(new ImplType);
  BFI->doFunction(&F, &BPI, &LI);
#ifndef NDEBUG
  if (ViewBlockFreqPropagationDAG != GVDT_None)
    view();
#endif
  return false;
}

void BlockFrequencyInfo::releaseMemory() { BFI.reset(); }

void BlockFrequencyInfo::print(raw_ostream &O, const Module *) const {
  if (BFI) BFI->print(O);
}

BlockFrequency BlockFrequencyInfo::getBlockFreq(const BasicBlock *BB) const {
  return BFI ? BFI->getBlockFreq(BB) : 0;
}

/// Pop up a ghostview window with the current block frequency propagation
/// rendered using dot.
void BlockFrequencyInfo::view() const {
// This code is only for debugging.
#ifndef NDEBUG
  ViewGraph(const_cast<BlockFrequencyInfo *>(this), "BlockFrequencyDAGs");
#else
  errs() << "BlockFrequencyInfo::view is only available in debug builds on "
            "systems with Graphviz or gv!\n";
#endif // NDEBUG
}

const Function *BlockFrequencyInfo::getFunction() const {
  return BFI ? BFI->getFunction() : nullptr;
}

raw_ostream &BlockFrequencyInfo::
printBlockFreq(raw_ostream &OS, const BlockFrequency Freq) const {
  return BFI ? BFI->printBlockFreq(OS, Freq) : OS;
}

raw_ostream &
BlockFrequencyInfo::printBlockFreq(raw_ostream &OS,
                                   const BasicBlock *BB) const {
  return BFI ? BFI->printBlockFreq(OS, BB) : OS;
}

uint64_t BlockFrequencyInfo::getEntryFreq() const {
  return BFI ? BFI->getEntryFreq() : 0;
}
