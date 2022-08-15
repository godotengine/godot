//===--------- DxilValueCache.h - Dxil Constant Value Cache --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_DXILVALUECACHE_H
#define LLVM_ANALYSIS_DXILVALUECACHE_H

#include "llvm/Pass.h"
#include "llvm/IR/ValueMap.h"

namespace llvm {

class Module;
class DominatorTree;
class Constant;
class ConstantInt;
class PHINode;

struct DxilValueCache : public ImmutablePass {
  static char ID;

  // Special Weak Value to Weak Value map.
  struct WeakValueMap {
    struct ValueVH : public CallbackVH {
      ValueVH(Value *V) : CallbackVH(V) {}
      void allUsesReplacedWith(Value *) override { setValPtr(nullptr); }
    };
    struct ValueEntry {
      WeakVH Value;
      ValueVH Self;
      ValueEntry() : Value(nullptr), Self(nullptr) {}
      inline void Set(llvm::Value *Key, llvm::Value *V) { Self = Key; Value = V; }
      inline bool IsStale() const { return Self == nullptr; }
    };
    ValueMap<const Value *, ValueEntry> Map;
    Value *Get(Value *V);
    void Set(Value *Key, Value *V);
    bool Seen(Value *v);
    void SetSentinel(Value *V);
    void ResetUnknowns();
    void ResetAll();
    void dump() const;
  private:
    Value *GetSentinel(LLVMContext &Ctx);
    std::unique_ptr<PHINode> Sentinel;
  };

private:

  WeakValueMap ValueMap;
  bool (*ShouldSkipCallback)(Value *V) = nullptr;

  void MarkUnreachable(BasicBlock *BB);
  bool IsUnreachable_(BasicBlock *BB);
  bool MayBranchTo(BasicBlock *A, BasicBlock *B);
  Value *TryGetCachedValue(Value *V);
  Value *ProcessValue(Value *V, DominatorTree *DT);

  Value *ProcessAndSimplify_PHI(Instruction *I, DominatorTree *DT);
  Value *ProcessAndSimplify_Br(Instruction *I, DominatorTree *DT);
  Value *ProcessAndSimplify_Switch(Instruction *I, DominatorTree *DT);
  Value *ProcessAndSimplify_Load(Instruction *LI, DominatorTree *DT);
  Value *SimplifyAndCacheResult(Instruction *I, DominatorTree *DT);

public:

  StringRef getPassName() const override;
  DxilValueCache();
  void getAnalysisUsage(AnalysisUsage &) const override;

  void dump() const;
  Value *GetValue(Value *V, DominatorTree *DT=nullptr);
  Constant *GetConstValue(Value *V, DominatorTree *DT = nullptr);
  ConstantInt *GetConstInt(Value *V, DominatorTree *DT = nullptr);
  void ResetUnknowns() { ValueMap.ResetUnknowns(); }
  void ResetAll() { ValueMap.ResetAll(); }
  bool IsUnreachable(BasicBlock *BB, DominatorTree *DT=nullptr);
  void SetShouldSkipCallback(bool (*Callback)(Value *V)) { ShouldSkipCallback = Callback; };
};

void initializeDxilValueCachePass(class llvm::PassRegistry &);
Pass *createDxilValueCachePass();

}

#endif
