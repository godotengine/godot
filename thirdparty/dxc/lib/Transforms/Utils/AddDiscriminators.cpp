//===- AddDiscriminators.cpp - Insert DWARF path discriminators -----------===//
//
//                      The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file adds DWARF discriminators to the IR. Path discriminators are
// used to decide what CFG path was taken inside sub-graphs whose instructions
// share the same line and column number information.
//
// The main user of this is the sample profiler. Instruction samples are
// mapped to line number information. Since a single line may be spread
// out over several basic blocks, discriminators add more precise location
// for the samples.
//
// For example,
//
//   1  #define ASSERT(P)
//   2      if (!(P))
//   3        abort()
//   ...
//   100   while (true) {
//   101     ASSERT (sum < 0);
//   102     ...
//   130   }
//
// when converted to IR, this snippet looks something like:
//
// while.body:                                       ; preds = %entry, %if.end
//   %0 = load i32* %sum, align 4, !dbg !15
//   %cmp = icmp slt i32 %0, 0, !dbg !15
//   br i1 %cmp, label %if.end, label %if.then, !dbg !15
//
// if.then:                                          ; preds = %while.body
//   call void @abort(), !dbg !15
//   br label %if.end, !dbg !15
//
// Notice that all the instructions in blocks 'while.body' and 'if.then'
// have exactly the same debug information. When this program is sampled
// at runtime, the profiler will assume that all these instructions are
// equally frequent. This, in turn, will consider the edge while.body->if.then
// to be frequently taken (which is incorrect).
//
// By adding a discriminator value to the instructions in block 'if.then',
// we can distinguish instructions at line 101 with discriminator 0 from
// the instructions at line 101 with discriminator 1.
//
// For more details about DWARF discriminators, please visit
// http://wiki.dwarfstd.org/index.php?title=Path_Discriminators
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Scalar.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;

#define DEBUG_TYPE "add-discriminators"

namespace {
  struct AddDiscriminators : public FunctionPass {
    static char ID; // Pass identification, replacement for typeid
    AddDiscriminators() : FunctionPass(ID) {
      initializeAddDiscriminatorsPass(*PassRegistry::getPassRegistry());
    }
    bool runOnFunction(Function &F) override;
    // HLSL Change Starts
    bool NoDiscriminators = false;
    void applyOptions(PassOptions O) override {
      GetPassOptionBool(O, "no-discriminators", &NoDiscriminators, false);
    }
    void dumpConfig(raw_ostream &OS) override {
      FunctionPass::dumpConfig(OS);
      OS << ",no-discriminators=" << NoDiscriminators;
    }
    // HLSL Change Ends
  };
}

char AddDiscriminators::ID = 0;
INITIALIZE_PASS_BEGIN(AddDiscriminators, "add-discriminators",
                      "Add DWARF path discriminators", false, false)
INITIALIZE_PASS_END(AddDiscriminators, "add-discriminators",
                    "Add DWARF path discriminators", false, false)

#ifdef HLSL_GLOBAL_OPT // HLSL Change Starts
// Command line option to disable discriminator generation even in the
// presence of debug information. This is only needed when debugging
// debug info generation issues.
static cl::opt<bool>
NoDiscriminators("no-discriminators", cl::init(false),
                 cl::desc("Disable generation of discriminator information."));
#endif // HLSL_GLOBAL_OPT HLSL Change Ends

FunctionPass *llvm::createAddDiscriminatorsPass() {
  return new AddDiscriminators();
}

static bool hasDebugInfo(const Function &F) {
  NamedMDNode *CUNodes = F.getParent()->getNamedMetadata("llvm.dbg.cu");
  return CUNodes != nullptr;
}

/// \brief Assign DWARF discriminators.
///
/// To assign discriminators, we examine the boundaries of every
/// basic block and its successors. Suppose there is a basic block B1
/// with successor B2. The last instruction I1 in B1 and the first
/// instruction I2 in B2 are located at the same file and line number.
/// This situation is illustrated in the following code snippet:
///
///       if (i < 10) x = i;
///
///     entry:
///       br i1 %cmp, label %if.then, label %if.end, !dbg !10
///     if.then:
///       %1 = load i32* %i.addr, align 4, !dbg !10
///       store i32 %1, i32* %x, align 4, !dbg !10
///       br label %if.end, !dbg !10
///     if.end:
///       ret void, !dbg !12
///
/// Notice how the branch instruction in block 'entry' and all the
/// instructions in block 'if.then' have the exact same debug location
/// information (!dbg !10).
///
/// To distinguish instructions in block 'entry' from instructions in
/// block 'if.then', we generate a new lexical block for all the
/// instruction in block 'if.then' that share the same file and line
/// location with the last instruction of block 'entry'.
///
/// This new lexical block will have the same location information as
/// the previous one, but with a new DWARF discriminator value.
///
/// One of the main uses of this discriminator value is in runtime
/// sample profilers. It allows the profiler to distinguish instructions
/// at location !dbg !10 that execute on different basic blocks. This is
/// important because while the predicate 'if (x < 10)' may have been
/// executed millions of times, the assignment 'x = i' may have only
/// executed a handful of times (meaning that the entry->if.then edge is
/// seldom taken).
///
/// If we did not have discriminator information, the profiler would
/// assign the same weight to both blocks 'entry' and 'if.then', which
/// in turn will make it conclude that the entry->if.then edge is very
/// hot.
///
/// To decide where to create new discriminator values, this function
/// traverses the CFG and examines instruction at basic block boundaries.
/// If the last instruction I1 of a block B1 is at the same file and line
/// location as instruction I2 of successor B2, then it creates a new
/// lexical block for I2 and all the instruction in B2 that share the same
/// file and line location as I2. This new lexical block will have a
/// different discriminator number than I1.
bool AddDiscriminators::runOnFunction(Function &F) {
  // If the function has debug information, but the user has disabled
  // discriminators, do nothing.
  // Simlarly, if the function has no debug info, do nothing.
  // Finally, if this module is built with dwarf versions earlier than 4,
  // do nothing (discriminator support is a DWARF 4 feature).
  if (NoDiscriminators ||
      !hasDebugInfo(F) ||
      F.getParent()->getDwarfVersion() < 4)
    return false;

  bool Changed = false;
  Module *M = F.getParent();
  LLVMContext &Ctx = M->getContext();
  DIBuilder Builder(*M, /*AllowUnresolved*/ false);

  // Traverse all the blocks looking for instructions in different
  // blocks that are at the same file:line location.
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I) {
    BasicBlock *B = I;
    TerminatorInst *Last = B->getTerminator();
    const DILocation *LastDIL = Last->getDebugLoc();
    if (!LastDIL)
      continue;

    for (unsigned I = 0; I < Last->getNumSuccessors(); ++I) {
      BasicBlock *Succ = Last->getSuccessor(I);
      Instruction *First = Succ->getFirstNonPHIOrDbgOrLifetime();
      const DILocation *FirstDIL = First->getDebugLoc();
      if (!FirstDIL)
        continue;

      // If the first instruction (First) of Succ is at the same file
      // location as B's last instruction (Last), add a new
      // discriminator for First's location and all the instructions
      // in Succ that share the same location with First.
      if (!FirstDIL->canDiscriminate(*LastDIL)) {
        // Create a new lexical scope and compute a new discriminator
        // number for it.
        StringRef Filename = FirstDIL->getFilename();
        auto *Scope = FirstDIL->getScope();
        auto *File = Builder.createFile(Filename, Scope->getDirectory());

        // FIXME: Calculate the discriminator here, based on local information,
        // and delete DILocation::computeNewDiscriminator().  The current
        // solution gives different results depending on other modules in the
        // same context.  All we really need is to discriminate between
        // FirstDIL and LastDIL -- a local map would suffice.
        unsigned Discriminator = FirstDIL->computeNewDiscriminator();
        auto *NewScope =
            Builder.createLexicalBlockFile(Scope, File, Discriminator);
        auto *NewDIL =
            DILocation::get(Ctx, FirstDIL->getLine(), FirstDIL->getColumn(),
                            NewScope, FirstDIL->getInlinedAt());
        DebugLoc newDebugLoc = NewDIL;

        // Attach this new debug location to First and every
        // instruction following First that shares the same location.
        for (BasicBlock::iterator I1(*First), E1 = Succ->end(); I1 != E1;
             ++I1) {
          if (I1->getDebugLoc().get() != FirstDIL)
            break;
          I1->setDebugLoc(newDebugLoc);
          DEBUG(dbgs() << NewDIL->getFilename() << ":" << NewDIL->getLine()
                       << ":" << NewDIL->getColumn() << ":"
                       << NewDIL->getDiscriminator() << *I1 << "\n");
        }
        DEBUG(dbgs() << "\n");
        Changed = true;
      }
    }
  }
  return Changed;
}
