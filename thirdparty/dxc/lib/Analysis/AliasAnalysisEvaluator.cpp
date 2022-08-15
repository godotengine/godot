//===- AliasAnalysisEvaluator.cpp - Alias Analysis Accuracy Evaluator -----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple N^2 alias analysis accuracy evaluator.
// Basically, for each function in the program, it simply queries to see how the
// alias analysis implementation answers alias queries between each pair of
// pointers in the function.
//
// This is inspired and adapted from code by: Naveen Neelakantam, Francesco
// Spadini, and Wojciech Stryjewski.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
using namespace llvm;

static cl::opt<bool> PrintAll("print-all-alias-modref-info", cl::ReallyHidden);

static cl::opt<bool> PrintNoAlias("print-no-aliases", cl::ReallyHidden);
static cl::opt<bool> PrintMayAlias("print-may-aliases", cl::ReallyHidden);
static cl::opt<bool> PrintPartialAlias("print-partial-aliases", cl::ReallyHidden);
static cl::opt<bool> PrintMustAlias("print-must-aliases", cl::ReallyHidden);

static cl::opt<bool> PrintNoModRef("print-no-modref", cl::ReallyHidden);
static cl::opt<bool> PrintMod("print-mod", cl::ReallyHidden);
static cl::opt<bool> PrintRef("print-ref", cl::ReallyHidden);
static cl::opt<bool> PrintModRef("print-modref", cl::ReallyHidden);

static cl::opt<bool> EvalAAMD("evaluate-aa-metadata", cl::ReallyHidden);

namespace {
  class AAEval : public FunctionPass {
    unsigned NoAliasCount, MayAliasCount, PartialAliasCount, MustAliasCount;
    unsigned NoModRefCount, ModCount, RefCount, ModRefCount;

  public:
    static char ID; // Pass identification, replacement for typeid
    AAEval() : FunctionPass(ID) {
      initializeAAEvalPass(*PassRegistry::getPassRegistry());
    }

    void getAnalysisUsage(AnalysisUsage &AU) const override {
      AU.addRequired<AliasAnalysis>();
      AU.setPreservesAll();
    }

    bool doInitialization(Module &M) override {
      NoAliasCount = MayAliasCount = PartialAliasCount = MustAliasCount = 0;
      NoModRefCount = ModCount = RefCount = ModRefCount = 0;

      if (PrintAll) {
        PrintNoAlias = PrintMayAlias = true;
        PrintPartialAlias = PrintMustAlias = true;
        PrintNoModRef = PrintMod = PrintRef = PrintModRef = true;
      }
      return false;
    }

    bool runOnFunction(Function &F) override;
    bool doFinalization(Module &M) override;
  };
}

char AAEval::ID = 0;
INITIALIZE_PASS_BEGIN(AAEval, "aa-eval",
                "Exhaustive Alias Analysis Precision Evaluator", false, true)
INITIALIZE_AG_DEPENDENCY(AliasAnalysis)
INITIALIZE_PASS_END(AAEval, "aa-eval",
                "Exhaustive Alias Analysis Precision Evaluator", false, true)

FunctionPass *llvm::createAAEvalPass() { return new AAEval(); }

static void PrintResults(const char *Msg, bool P, const Value *V1,
                         const Value *V2, const Module *M) {
  if (P) {
    std::string o1, o2;
    {
      raw_string_ostream os1(o1), os2(o2);
      V1->printAsOperand(os1, true, M);
      V2->printAsOperand(os2, true, M);
    }
    
    if (o2 < o1)
      std::swap(o1, o2);
    errs() << "  " << Msg << ":\t"
           << o1 << ", "
           << o2 << "\n";
  }
}

static inline void
PrintModRefResults(const char *Msg, bool P, Instruction *I, Value *Ptr,
                   Module *M) {
  if (P) {
    errs() << "  " << Msg << ":  Ptr: ";
    Ptr->printAsOperand(errs(), true, M);
    errs() << "\t<->" << *I << '\n';
  }
}

static inline void
PrintModRefResults(const char *Msg, bool P, CallSite CSA, CallSite CSB,
                   Module *M) {
  if (P) {
    errs() << "  " << Msg << ": " << *CSA.getInstruction()
           << " <-> " << *CSB.getInstruction() << '\n';
  }
}

static inline void
PrintLoadStoreResults(const char *Msg, bool P, const Value *V1,
                      const Value *V2, const Module *M) {
  if (P) {
    errs() << "  " << Msg << ": " << *V1
           << " <-> " << *V2 << '\n';
  }
}

static inline bool isInterestingPointer(Value *V) {
  return V->getType()->isPointerTy()
      && !isa<ConstantPointerNull>(V);
}

bool AAEval::runOnFunction(Function &F) {
  AliasAnalysis &AA = getAnalysis<AliasAnalysis>();

  SetVector<Value *> Pointers;
  SetVector<CallSite> CallSites;
  SetVector<Value *> Loads;
  SetVector<Value *> Stores;

  for (Function::arg_iterator I = F.arg_begin(), E = F.arg_end(); I != E; ++I)
    if (I->getType()->isPointerTy())    // Add all pointer arguments.
      Pointers.insert(I);

  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
    if (I->getType()->isPointerTy()) // Add all pointer instructions.
      Pointers.insert(&*I);
    if (EvalAAMD && isa<LoadInst>(&*I))
      Loads.insert(&*I);
    if (EvalAAMD && isa<StoreInst>(&*I))
      Stores.insert(&*I);
    Instruction &Inst = *I;
    if (auto CS = CallSite(&Inst)) {
      Value *Callee = CS.getCalledValue();
      // Skip actual functions for direct function calls.
      if (!isa<Function>(Callee) && isInterestingPointer(Callee))
        Pointers.insert(Callee);
      // Consider formals.
      for (CallSite::arg_iterator AI = CS.arg_begin(), AE = CS.arg_end();
           AI != AE; ++AI)
        if (isInterestingPointer(*AI))
          Pointers.insert(*AI);
      CallSites.insert(CS);
    } else {
      // Consider all operands.
      for (Instruction::op_iterator OI = Inst.op_begin(), OE = Inst.op_end();
           OI != OE; ++OI)
        if (isInterestingPointer(*OI))
          Pointers.insert(*OI);
    }
  }

  if (PrintNoAlias || PrintMayAlias || PrintPartialAlias || PrintMustAlias ||
      PrintNoModRef || PrintMod || PrintRef || PrintModRef)
    errs() << "Function: " << F.getName() << ": " << Pointers.size()
           << " pointers, " << CallSites.size() << " call sites\n";

  // iterate over the worklist, and run the full (n^2)/2 disambiguations
  for (SetVector<Value *>::iterator I1 = Pointers.begin(), E = Pointers.end();
       I1 != E; ++I1) {
    uint64_t I1Size = MemoryLocation::UnknownSize;
    Type *I1ElTy = cast<PointerType>((*I1)->getType())->getElementType();
    if (I1ElTy->isSized()) I1Size = AA.getTypeStoreSize(I1ElTy);

    for (SetVector<Value *>::iterator I2 = Pointers.begin(); I2 != I1; ++I2) {
      uint64_t I2Size = MemoryLocation::UnknownSize;
      Type *I2ElTy =cast<PointerType>((*I2)->getType())->getElementType();
      if (I2ElTy->isSized()) I2Size = AA.getTypeStoreSize(I2ElTy);

      switch (AA.alias(*I1, I1Size, *I2, I2Size)) {
      case NoAlias:
        PrintResults("NoAlias", PrintNoAlias, *I1, *I2, F.getParent());
        ++NoAliasCount;
        break;
      case MayAlias:
        PrintResults("MayAlias", PrintMayAlias, *I1, *I2, F.getParent());
        ++MayAliasCount;
        break;
      case PartialAlias:
        PrintResults("PartialAlias", PrintPartialAlias, *I1, *I2,
                     F.getParent());
        ++PartialAliasCount;
        break;
      case MustAlias:
        PrintResults("MustAlias", PrintMustAlias, *I1, *I2, F.getParent());
        ++MustAliasCount;
        break;
      }
    }
  }

  if (EvalAAMD) {
    // iterate over all pairs of load, store
    for (SetVector<Value *>::iterator I1 = Loads.begin(), E = Loads.end();
         I1 != E; ++I1) {
      for (SetVector<Value *>::iterator I2 = Stores.begin(), E2 = Stores.end();
           I2 != E2; ++I2) {
        switch (AA.alias(MemoryLocation::get(cast<LoadInst>(*I1)),
                         MemoryLocation::get(cast<StoreInst>(*I2)))) {
        case NoAlias:
          PrintLoadStoreResults("NoAlias", PrintNoAlias, *I1, *I2,
                                F.getParent());
          ++NoAliasCount;
          break;
        case MayAlias:
          PrintLoadStoreResults("MayAlias", PrintMayAlias, *I1, *I2,
                                F.getParent());
          ++MayAliasCount;
          break;
        case PartialAlias:
          PrintLoadStoreResults("PartialAlias", PrintPartialAlias, *I1, *I2,
                                F.getParent());
          ++PartialAliasCount;
          break;
        case MustAlias:
          PrintLoadStoreResults("MustAlias", PrintMustAlias, *I1, *I2,
                                F.getParent());
          ++MustAliasCount;
          break;
        }
      }
    }

    // iterate over all pairs of store, store
    for (SetVector<Value *>::iterator I1 = Stores.begin(), E = Stores.end();
         I1 != E; ++I1) {
      for (SetVector<Value *>::iterator I2 = Stores.begin(); I2 != I1; ++I2) {
        switch (AA.alias(MemoryLocation::get(cast<StoreInst>(*I1)),
                         MemoryLocation::get(cast<StoreInst>(*I2)))) {
        case NoAlias:
          PrintLoadStoreResults("NoAlias", PrintNoAlias, *I1, *I2,
                                F.getParent());
          ++NoAliasCount;
          break;
        case MayAlias:
          PrintLoadStoreResults("MayAlias", PrintMayAlias, *I1, *I2,
                                F.getParent());
          ++MayAliasCount;
          break;
        case PartialAlias:
          PrintLoadStoreResults("PartialAlias", PrintPartialAlias, *I1, *I2,
                                F.getParent());
          ++PartialAliasCount;
          break;
        case MustAlias:
          PrintLoadStoreResults("MustAlias", PrintMustAlias, *I1, *I2,
                                F.getParent());
          ++MustAliasCount;
          break;
        }
      }
    }
  }

  // Mod/ref alias analysis: compare all pairs of calls and values
  for (SetVector<CallSite>::iterator C = CallSites.begin(),
         Ce = CallSites.end(); C != Ce; ++C) {
    Instruction *I = C->getInstruction();

    for (SetVector<Value *>::iterator V = Pointers.begin(), Ve = Pointers.end();
         V != Ve; ++V) {
      uint64_t Size = MemoryLocation::UnknownSize;
      Type *ElTy = cast<PointerType>((*V)->getType())->getElementType();
      if (ElTy->isSized()) Size = AA.getTypeStoreSize(ElTy);

      switch (AA.getModRefInfo(*C, *V, Size)) {
      case AliasAnalysis::NoModRef:
        PrintModRefResults("NoModRef", PrintNoModRef, I, *V, F.getParent());
        ++NoModRefCount;
        break;
      case AliasAnalysis::Mod:
        PrintModRefResults("Just Mod", PrintMod, I, *V, F.getParent());
        ++ModCount;
        break;
      case AliasAnalysis::Ref:
        PrintModRefResults("Just Ref", PrintRef, I, *V, F.getParent());
        ++RefCount;
        break;
      case AliasAnalysis::ModRef:
        PrintModRefResults("Both ModRef", PrintModRef, I, *V, F.getParent());
        ++ModRefCount;
        break;
      }
    }
  }

  // Mod/ref alias analysis: compare all pairs of calls
  for (SetVector<CallSite>::iterator C = CallSites.begin(),
         Ce = CallSites.end(); C != Ce; ++C) {
    for (SetVector<CallSite>::iterator D = CallSites.begin(); D != Ce; ++D) {
      if (D == C)
        continue;
      switch (AA.getModRefInfo(*C, *D)) {
      case AliasAnalysis::NoModRef:
        PrintModRefResults("NoModRef", PrintNoModRef, *C, *D, F.getParent());
        ++NoModRefCount;
        break;
      case AliasAnalysis::Mod:
        PrintModRefResults("Just Mod", PrintMod, *C, *D, F.getParent());
        ++ModCount;
        break;
      case AliasAnalysis::Ref:
        PrintModRefResults("Just Ref", PrintRef, *C, *D, F.getParent());
        ++RefCount;
        break;
      case AliasAnalysis::ModRef:
        PrintModRefResults("Both ModRef", PrintModRef, *C, *D, F.getParent());
        ++ModRefCount;
        break;
      }
    }
  }

  return false;
}

static void PrintPercent(unsigned Num, unsigned Sum) {
  errs() << "(" << Num*100ULL/Sum << "."
         << ((Num*1000ULL/Sum) % 10) << "%)\n";
}

bool AAEval::doFinalization(Module &M) {
  unsigned AliasSum =
      NoAliasCount + MayAliasCount + PartialAliasCount + MustAliasCount;
  errs() << "===== Alias Analysis Evaluator Report =====\n";
  if (AliasSum == 0) {
    errs() << "  Alias Analysis Evaluator Summary: No pointers!\n";
  } else {
    errs() << "  " << AliasSum << " Total Alias Queries Performed\n";
    errs() << "  " << NoAliasCount << " no alias responses ";
    PrintPercent(NoAliasCount, AliasSum);
    errs() << "  " << MayAliasCount << " may alias responses ";
    PrintPercent(MayAliasCount, AliasSum);
    errs() << "  " << PartialAliasCount << " partial alias responses ";
    PrintPercent(PartialAliasCount, AliasSum);
    errs() << "  " << MustAliasCount << " must alias responses ";
    PrintPercent(MustAliasCount, AliasSum);
    errs() << "  Alias Analysis Evaluator Pointer Alias Summary: "
           << NoAliasCount * 100 / AliasSum << "%/"
           << MayAliasCount * 100 / AliasSum << "%/"
           << PartialAliasCount * 100 / AliasSum << "%/"
           << MustAliasCount * 100 / AliasSum << "%\n";
  }

  // Display the summary for mod/ref analysis
  unsigned ModRefSum = NoModRefCount + ModCount + RefCount + ModRefCount;
  if (ModRefSum == 0) {
    errs() << "  Alias Analysis Mod/Ref Evaluator Summary: no "
              "mod/ref!\n";
  } else {
    errs() << "  " << ModRefSum << " Total ModRef Queries Performed\n";
    errs() << "  " << NoModRefCount << " no mod/ref responses ";
    PrintPercent(NoModRefCount, ModRefSum);
    errs() << "  " << ModCount << " mod responses ";
    PrintPercent(ModCount, ModRefSum);
    errs() << "  " << RefCount << " ref responses ";
    PrintPercent(RefCount, ModRefSum);
    errs() << "  " << ModRefCount << " mod & ref responses ";
    PrintPercent(ModRefCount, ModRefSum);
    errs() << "  Alias Analysis Evaluator Mod/Ref Summary: "
           << NoModRefCount * 100 / ModRefSum << "%/"
           << ModCount * 100 / ModRefSum << "%/" << RefCount * 100 / ModRefSum
           << "%/" << ModRefCount * 100 / ModRefSum << "%\n";
  }

  return false;
}
