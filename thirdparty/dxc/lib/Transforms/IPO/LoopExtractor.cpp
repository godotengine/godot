//===- LoopExtractor.cpp - Extract each loop into a new function ----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// A pass wrapper around the ExtractLoop() scalar transformation to extract each
// top-level loop into its own new function. If the loop is the ONLY loop in a
// given function, it is not touched. This is a pass most useful for debugging
// via bugpoint.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/CodeExtractor.h"
#include <fstream>
#include <set>
using namespace llvm;

#define DEBUG_TYPE "loop-extract"

STATISTIC(NumExtracted, "Number of loops extracted");

namespace {
  struct LoopExtractor : public LoopPass {
    static char ID; // Pass identification, replacement for typeid
    unsigned NumLoops;

    explicit LoopExtractor(unsigned numLoops = ~0) 
      : LoopPass(ID), NumLoops(numLoops) {
        initializeLoopExtractorPass(*PassRegistry::getPassRegistry());
      }

    bool runOnLoop(Loop *L, LPPassManager &LPM) override;

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequiredID(BreakCriticalEdgesID);
      AU.addRequiredID(LoopSimplifyID);
      AU.addRequired<DominatorTreeWrapperPass>();
    }
  };
}

char LoopExtractor::ID = 0;
INITIALIZE_PASS_BEGIN(LoopExtractor, "loop-extract",
                      "Extract loops into new functions", false, false)
INITIALIZE_PASS_DEPENDENCY(BreakCriticalEdges)
INITIALIZE_PASS_DEPENDENCY(LoopSimplify)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_END(LoopExtractor, "loop-extract",
                    "Extract loops into new functions", false, false)

namespace {
  /// SingleLoopExtractor - For bugpoint.
  struct SingleLoopExtractor : public LoopExtractor {
    static char ID; // Pass identification, replacement for typeid
    SingleLoopExtractor() : LoopExtractor(1) {}
  };
} // End anonymous namespace

char SingleLoopExtractor::ID = 0;
INITIALIZE_PASS(SingleLoopExtractor, "loop-extract-single",
                "Extract at most one loop into a new function", false, false)

// createLoopExtractorPass - This pass extracts all natural loops from the
// program into a function if it can.
//
Pass *llvm::createLoopExtractorPass() { return new LoopExtractor(); }

bool LoopExtractor::runOnLoop(Loop *L, LPPassManager &LPM) {
  if (skipOptnoneFunction(L))
    return false;

  // Only visit top-level loops.
  if (L->getParentLoop())
    return false;

  // If LoopSimplify form is not available, stay out of trouble.
  if (!L->isLoopSimplifyForm())
    return false;

  DominatorTree &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  bool Changed = false;

  // If there is more than one top-level loop in this function, extract all of
  // the loops. Otherwise there is exactly one top-level loop; in this case if
  // this function is more than a minimal wrapper around the loop, extract
  // the loop.
  bool ShouldExtractLoop = false;

  // Extract the loop if the entry block doesn't branch to the loop header.
  TerminatorInst *EntryTI =
    L->getHeader()->getParent()->getEntryBlock().getTerminator();
  if (!isa<BranchInst>(EntryTI) ||
      !cast<BranchInst>(EntryTI)->isUnconditional() ||
      EntryTI->getSuccessor(0) != L->getHeader()) {
    ShouldExtractLoop = true;
  } else {
    // Check to see if any exits from the loop are more than just return
    // blocks.
    SmallVector<BasicBlock*, 8> ExitBlocks;
    L->getExitBlocks(ExitBlocks);
    for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i)
      if (!isa<ReturnInst>(ExitBlocks[i]->getTerminator())) {
        ShouldExtractLoop = true;
        break;
      }
  }

  if (ShouldExtractLoop) {
    // We must omit landing pads. Landing pads must accompany the invoke
    // instruction. But this would result in a loop in the extracted
    // function. An infinite cycle occurs when it tries to extract that loop as
    // well.
    SmallVector<BasicBlock*, 8> ExitBlocks;
    L->getExitBlocks(ExitBlocks);
    for (unsigned i = 0, e = ExitBlocks.size(); i != e; ++i)
      if (ExitBlocks[i]->isLandingPad()) {
        ShouldExtractLoop = false;
        break;
      }
  }

  if (ShouldExtractLoop) {
    if (NumLoops == 0) return Changed;
    --NumLoops;
    CodeExtractor Extractor(DT, *L);
    if (Extractor.extractCodeRegion() != nullptr) {
      Changed = true;
      // After extraction, the loop is replaced by a function call, so
      // we shouldn't try to run any more loop passes on it.
      LPM.deleteLoopFromQueue(L);
    }
    ++NumExtracted;
  }

  return Changed;
}

// createSingleLoopExtractorPass - This pass extracts one natural loop from the
// program into a function if it can.  This is used by bugpoint.
//
Pass *llvm::createSingleLoopExtractorPass() {
  return new SingleLoopExtractor();
}


// BlockFile - A file which contains a list of blocks that should not be
// extracted.
static cl::opt<std::string>
BlockFile("extract-blocks-file", cl::value_desc("filename"),
          cl::desc("A file containing list of basic blocks to not extract"),
          cl::Hidden);

namespace {
  /// BlockExtractorPass - This pass is used by bugpoint to extract all blocks
  /// from the module into their own functions except for those specified by the
  /// BlocksToNotExtract list.
  class BlockExtractorPass : public ModulePass {
    void LoadFile(const char *Filename);
    void SplitLandingPadPreds(Function *F);

    std::vector<BasicBlock*> BlocksToNotExtract;
    std::vector<std::pair<std::string, std::string> > BlocksToNotExtractByName;
  public:
    static char ID; // Pass identification, replacement for typeid
    BlockExtractorPass() : ModulePass(ID) {
      if (!BlockFile.empty())
        LoadFile(BlockFile.c_str());
    }

    bool runOnModule(Module &M) override;
  };
}

char BlockExtractorPass::ID = 0;
INITIALIZE_PASS(BlockExtractorPass, "extract-blocks",
                "Extract Basic Blocks From Module (for bugpoint use)",
                false, false)

// createBlockExtractorPass - This pass extracts all blocks (except those
// specified in the argument list) from the functions in the module.
//
ModulePass *llvm::createBlockExtractorPass() {
  return new BlockExtractorPass();
}

void BlockExtractorPass::LoadFile(const char *Filename) {
  // Load the BlockFile...
  std::ifstream In(Filename);
  if (!In.good()) {
    errs() << "WARNING: BlockExtractor couldn't load file '" << Filename
           << "'!\n";
    return;
  }
  while (In) {
    std::string FunctionName, BlockName;
    In >> FunctionName;
    In >> BlockName;
    if (!BlockName.empty())
      BlocksToNotExtractByName.push_back(
          std::make_pair(FunctionName, BlockName));
  }
}

/// SplitLandingPadPreds - The landing pad needs to be extracted with the invoke
/// instruction. The critical edge breaker will refuse to break critical edges
/// to a landing pad. So do them here. After this method runs, all landing pads
/// should have only one predecessor.
void BlockExtractorPass::SplitLandingPadPreds(Function *F) {
  for (Function::iterator I = F->begin(), E = F->end(); I != E; ++I) {
    InvokeInst *II = dyn_cast<InvokeInst>(I);
    if (!II) continue;
    BasicBlock *Parent = II->getParent();
    BasicBlock *LPad = II->getUnwindDest();

    // Look through the landing pad's predecessors. If one of them ends in an
    // 'invoke', then we want to split the landing pad.
    bool Split = false;
    for (pred_iterator
           PI = pred_begin(LPad), PE = pred_end(LPad); PI != PE; ++PI) {
      BasicBlock *BB = *PI;
      if (BB->isLandingPad() && BB != Parent &&
          isa<InvokeInst>(Parent->getTerminator())) {
        Split = true;
        break;
      }
    }

    if (!Split) continue;

    SmallVector<BasicBlock*, 2> NewBBs;
    SplitLandingPadPredecessors(LPad, Parent, ".1", ".2", NewBBs);
  }
}

bool BlockExtractorPass::runOnModule(Module &M) {
  std::set<BasicBlock*> TranslatedBlocksToNotExtract;
  for (unsigned i = 0, e = BlocksToNotExtract.size(); i != e; ++i) {
    BasicBlock *BB = BlocksToNotExtract[i];
    Function *F = BB->getParent();

    // Map the corresponding function in this module.
    Function *MF = M.getFunction(F->getName());
    assert(MF->getFunctionType() == F->getFunctionType() && "Wrong function?");

    // Figure out which index the basic block is in its function.
    Function::iterator BBI = MF->begin();
    std::advance(BBI, std::distance(F->begin(), Function::iterator(BB)));
    TranslatedBlocksToNotExtract.insert(BBI);
  }

  while (!BlocksToNotExtractByName.empty()) {
    // There's no way to find BBs by name without looking at every BB inside
    // every Function. Fortunately, this is always empty except when used by
    // bugpoint in which case correctness is more important than performance.

    std::string &FuncName  = BlocksToNotExtractByName.back().first;
    std::string &BlockName = BlocksToNotExtractByName.back().second;

    for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI) {
      Function &F = *FI;
      if (F.getName() != FuncName) continue;

      for (Function::iterator BI = F.begin(), BE = F.end(); BI != BE; ++BI) {
        BasicBlock &BB = *BI;
        if (BB.getName() != BlockName) continue;

        TranslatedBlocksToNotExtract.insert(BI);
      }
    }

    BlocksToNotExtractByName.pop_back();
  }

  // Now that we know which blocks to not extract, figure out which ones we WANT
  // to extract.
  std::vector<BasicBlock*> BlocksToExtract;
  for (Module::iterator F = M.begin(), E = M.end(); F != E; ++F) {
    SplitLandingPadPreds(&*F);
    for (Function::iterator BB = F->begin(), E = F->end(); BB != E; ++BB)
      if (!TranslatedBlocksToNotExtract.count(BB))
        BlocksToExtract.push_back(BB);
  }

  for (unsigned i = 0, e = BlocksToExtract.size(); i != e; ++i) {
    SmallVector<BasicBlock*, 2> BlocksToExtractVec;
    BlocksToExtractVec.push_back(BlocksToExtract[i]);
    if (const InvokeInst *II =
        dyn_cast<InvokeInst>(BlocksToExtract[i]->getTerminator()))
      BlocksToExtractVec.push_back(II->getUnwindDest());
    CodeExtractor(BlocksToExtractVec).extractCodeRegion();
  }

  return !BlocksToExtract.empty();
}
