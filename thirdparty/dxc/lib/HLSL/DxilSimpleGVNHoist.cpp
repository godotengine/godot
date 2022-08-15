///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// DxilSimpleGVNHoist.cpp                                                    //
// Copyright (C) Microsoft Corporation. All rights reserved.                 //
// This file is distributed under the University of Illinois Open Source     //
// License. See LICENSE.TXT for details.                                     //
//                                                                           //
// A simple version of GVN hoist for DXIL.                                   //
// Based on GVNHoist in LLVM 6.0.                                            //
///////////////////////////////////////////////////////////////////////////////

#include "dxc/HLSL/DxilGenerationPass.h"
#include "dxc/DXIL/DxilOperations.h"

#include "llvm/IR/Dominators.h"
#include "llvm/IR/Instructions.h"

#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/CFG.h"

#include "llvm/Analysis/PostDominators.h"
#include "dxc/HLSL/DxilNoops.h"

using namespace llvm;
using namespace hlsl;

///////////////////////////////////////////////////////////////////////////////
namespace {
struct Expression {
  uint32_t opcode;
  Type *type;
  bool commutative = false;
  SmallVector<uint32_t, 4> varargs;

  Expression(uint32_t o = ~2U) : opcode(o) {}

  bool operator==(const Expression &other) const {
    if (opcode != other.opcode)
      return false;
    if (opcode == ~0U || opcode == ~1U)
      return true;
    if (type != other.type)
      return false;
    if (varargs != other.varargs)
      return false;
    return true;
  }

  friend hash_code hash_value(const Expression &Value) {
    return hash_combine(
        Value.opcode, Value.type,
        hash_combine_range(Value.varargs.begin(), Value.varargs.end()));
  }
};

}

namespace llvm {
template <> struct DenseMapInfo<Expression> {
  static inline Expression getEmptyKey() { return ~0U; }
  static inline Expression getTombstoneKey() { return ~1U; }

  static unsigned getHashValue(const Expression &e) {
    using llvm::hash_value;

    return static_cast<unsigned>(hash_value(e));
  }

  static bool isEqual(const Expression &LHS, const Expression &RHS) {
    return LHS == RHS;
  }
};
} // namespace llvm

namespace {
// Simple Value table which support DXIL operation.
class ValueTable {
  DenseMap<Value *, uint32_t> valueNumbering;
  DenseMap<Expression, uint32_t> expressionNumbering;

  // Expressions is the vector of Expression. ExprIdx is the mapping from
  // value number to the index of Expression in Expressions. We use it
  // instead of a DenseMap because filling such mapping is faster than
  // filling a DenseMap and the compile time is a little better.
  uint32_t nextExprNumber;

  std::vector<Expression> Expressions;
  std::vector<uint32_t> ExprIdx;

  DominatorTree *DT;

  uint32_t nextValueNumber = 1;

  Expression createExpr(Instruction *I);
  Expression createCmpExpr(unsigned Opcode, CmpInst::Predicate Predicate,
                           Value *LHS, Value *RHS);
  Expression createExtractvalueExpr(ExtractValueInst *EI);
  uint32_t lookupOrAddCall(CallInst *C);

  std::pair<uint32_t, bool> assignExpNewValueNum(Expression &exp);

public:
  ValueTable();
  ValueTable(const ValueTable &Arg);
  ValueTable(ValueTable &&Arg);
  ~ValueTable();

  uint32_t lookupOrAdd(Value *V);
  uint32_t lookup(Value *V, bool Verify = true) const;
  uint32_t lookupOrAddCmp(unsigned Opcode, CmpInst::Predicate Pred, Value *LHS,
                          Value *RHS);
  bool exists(Value *V) const;
  void add(Value *V, uint32_t num);
  void clear();
  void erase(Value *v);
  void setDomTree(DominatorTree *D) { DT = D; }
  uint32_t getNextUnusedValueNumber() { return nextValueNumber; }
  void verifyRemoved(const Value *) const;
};

//===----------------------------------------------------------------------===//
//                     ValueTable Internal Functions
//===----------------------------------------------------------------------===//

Expression ValueTable::createExpr(Instruction *I) {
  Expression e;
  e.type = I->getType();
  e.opcode = I->getOpcode();
  for (Instruction::op_iterator OI = I->op_begin(), OE = I->op_end();
       OI != OE; ++OI)
    e.varargs.push_back(lookupOrAdd(*OI));
  if (I->isCommutative()) {
    // Ensure that commutative instructions that only differ by a permutation
    // of their operands get the same value number by sorting the operand value
    // numbers.  Since all commutative instructions have two operands it is more
    // efficient to sort by hand rather than using, say, std::sort.
    assert(I->getNumOperands() == 2 && "Unsupported commutative instruction!");
    if (e.varargs[0] > e.varargs[1])
      std::swap(e.varargs[0], e.varargs[1]);
    e.commutative = true;
  }

  if (CmpInst *C = dyn_cast<CmpInst>(I)) {
    // Sort the operand value numbers so x<y and y>x get the same value number.
    CmpInst::Predicate Predicate = C->getPredicate();
    if (e.varargs[0] > e.varargs[1]) {
      std::swap(e.varargs[0], e.varargs[1]);
Predicate = CmpInst::getSwappedPredicate(Predicate);
    }
    e.opcode = (C->getOpcode() << 8) | Predicate;
    e.commutative = true;
  }
 else if (InsertValueInst *E = dyn_cast<InsertValueInst>(I)) {
 for (InsertValueInst::idx_iterator II = E->idx_begin(), IE = E->idx_end();
     II != IE; ++II)
     e.varargs.push_back(*II);
  }

  return e;
}

Expression ValueTable::createCmpExpr(unsigned Opcode,
    CmpInst::Predicate Predicate,
    Value *LHS, Value *RHS) {
    assert((Opcode == Instruction::ICmp || Opcode == Instruction::FCmp) &&
        "Not a comparison!");
    Expression e;
    e.type = CmpInst::makeCmpResultType(LHS->getType());
    e.varargs.push_back(lookupOrAdd(LHS));
    e.varargs.push_back(lookupOrAdd(RHS));

    // Sort the operand value numbers so x<y and y>x get the same value number.
    if (e.varargs[0] > e.varargs[1]) {
        std::swap(e.varargs[0], e.varargs[1]);
        Predicate = CmpInst::getSwappedPredicate(Predicate);
    }
    e.opcode = (Opcode << 8) | Predicate;
    e.commutative = true;
    return e;
}

Expression ValueTable::createExtractvalueExpr(ExtractValueInst *EI) {
    assert(EI && "Not an ExtractValueInst?");
    Expression e;
    e.type = EI->getType();
    e.opcode = 0;

    IntrinsicInst *I = dyn_cast<IntrinsicInst>(EI->getAggregateOperand());
    if (I != nullptr && EI->getNumIndices() == 1 && *EI->idx_begin() == 0) {
        // EI might be an extract from one of our recognised intrinsics. If it
        // is we'll synthesize a semantically equivalent expression instead on
        // an extract value expression.
        switch (I->getIntrinsicID()) {
        case Intrinsic::sadd_with_overflow:
        case Intrinsic::uadd_with_overflow:
            e.opcode = Instruction::Add;
            break;
        case Intrinsic::ssub_with_overflow:
        case Intrinsic::usub_with_overflow:
            e.opcode = Instruction::Sub;
            break;
        case Intrinsic::smul_with_overflow:
        case Intrinsic::umul_with_overflow:
            e.opcode = Instruction::Mul;
            break;
        default:
            break;
        }

        if (e.opcode != 0) {
            // Intrinsic recognized. Grab its args to finish building the expression.
            assert(I->getNumArgOperands() == 2 &&
                "Expect two args for recognised intrinsics.");
            e.varargs.push_back(lookupOrAdd(I->getArgOperand(0)));
            e.varargs.push_back(lookupOrAdd(I->getArgOperand(1)));
            return e;
        }
    }

    // Not a recognised intrinsic. Fall back to producing an extract value
    // expression.
    e.opcode = EI->getOpcode();
    for (Instruction::op_iterator OI = EI->op_begin(), OE = EI->op_end();
        OI != OE; ++OI)
        e.varargs.push_back(lookupOrAdd(*OI));

    for (ExtractValueInst::idx_iterator II = EI->idx_begin(), IE = EI->idx_end();
        II != IE; ++II)
        e.varargs.push_back(*II);

    return e;
}

//===----------------------------------------------------------------------===//
//                     ValueTable External Functions
//===----------------------------------------------------------------------===//

ValueTable::ValueTable() = default;
ValueTable::ValueTable(const ValueTable &) = default;
ValueTable::ValueTable(ValueTable &&) = default;
ValueTable::~ValueTable() = default;

/// add - Insert a value into the table with a specified value number.
void ValueTable::add(Value *V, uint32_t num) {
    valueNumbering.insert(std::make_pair(V, num));
}

uint32_t ValueTable::lookupOrAddCall(CallInst *C) {
  Function *F = C->getCalledFunction();
  bool bSafe = false;
  if (F) {
    if (F->hasFnAttribute(Attribute::ReadNone)) {
      bSafe = true;
    }
    else if (F->hasFnAttribute(Attribute::ReadOnly)) {
      if (hlsl::OP::IsDxilOpFunc(F)) {
        DXIL::OpCode Opcode = hlsl::OP::GetDxilOpFuncCallInst(C);
        switch (Opcode) {
        default:
          break;
          // TODO: make buffer/texture load on srv safe.
        case DXIL::OpCode::CreateHandleForLib:
        case DXIL::OpCode::AnnotateHandle:
        case DXIL::OpCode::CBufferLoad:
        case DXIL::OpCode::CBufferLoadLegacy:
        case DXIL::OpCode::Sample:
        case DXIL::OpCode::SampleBias:
        case DXIL::OpCode::SampleCmp:
        case DXIL::OpCode::SampleCmpLevel:
        case DXIL::OpCode::SampleCmpLevelZero:
        case DXIL::OpCode::SampleGrad:
        case DXIL::OpCode::CheckAccessFullyMapped:
        case DXIL::OpCode::GetDimensions:
        case DXIL::OpCode::TextureLoad:
        case DXIL::OpCode::TextureGather:
        case DXIL::OpCode::TextureGatherCmp:
        case DXIL::OpCode::Texture2DMSGetSamplePosition:
        case DXIL::OpCode::RenderTargetGetSampleCount:
        case DXIL::OpCode::RenderTargetGetSamplePosition:
        case DXIL::OpCode::CalculateLOD:
          bSafe = true;
          break;
        }
      }
    }
  }
  if (bSafe) {
    Expression exp = createExpr(C);
    uint32_t e = assignExpNewValueNum(exp).first;
    valueNumbering[C] = e;
    return e;
  } else {
    // Not sure safe or not, always use new value number.
    valueNumbering[C] = nextValueNumber;
    return nextValueNumber++;
  }
}

/// Returns true if a value number exists for the specified value.
bool ValueTable::exists(Value *V) const { return valueNumbering.count(V) != 0; }

/// lookup_or_add - Returns the value number for the specified value, assigning
/// it a new number if it did not have one before.
uint32_t ValueTable::lookupOrAdd(Value *V) {
  DenseMap<Value*, uint32_t>::iterator VI = valueNumbering.find(V);
  if (VI != valueNumbering.end())
    return VI->second;

  if (!isa<Instruction>(V)) {
    valueNumbering[V] = nextValueNumber;
    return nextValueNumber++;
  }

  Instruction* I = cast<Instruction>(V);
  Expression exp;
  switch (I->getOpcode()) {
    case Instruction::Call:
      return lookupOrAddCall(cast<CallInst>(I));
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
    case Instruction::ICmp:
    case Instruction::FCmp:
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::BitCast:
    case Instruction::Select:
    case Instruction::ExtractElement:
    case Instruction::InsertElement:
    case Instruction::ShuffleVector:
    case Instruction::InsertValue:
    case Instruction::GetElementPtr:
      exp = createExpr(I);
      break;
    case Instruction::ExtractValue:
      exp = createExtractvalueExpr(cast<ExtractValueInst>(I));
      break;
    case Instruction::PHI:
      valueNumbering[V] = nextValueNumber;
      return nextValueNumber++;
    default:
      valueNumbering[V] = nextValueNumber;
      return nextValueNumber++;
  }

  uint32_t e = assignExpNewValueNum(exp).first;
  valueNumbering[V] = e;
  return e;
}

/// Returns the value number of the specified value. Fails if
/// the value has not yet been numbered.
uint32_t ValueTable::lookup(Value *V, bool Verify) const {
  DenseMap<Value*, uint32_t>::const_iterator VI = valueNumbering.find(V);
  if (Verify) {
    assert(VI != valueNumbering.end() && "Value not numbered?");
    return VI->second;
  }
  return (VI != valueNumbering.end()) ? VI->second : 0;
}

/// Returns the value number of the given comparison,
/// assigning it a new number if it did not have one before.  Useful when
/// we deduced the result of a comparison, but don't immediately have an
/// instruction realizing that comparison to hand.
uint32_t ValueTable::lookupOrAddCmp(unsigned Opcode,
                                         CmpInst::Predicate Predicate,
                                         Value *LHS, Value *RHS) {
  Expression exp = createCmpExpr(Opcode, Predicate, LHS, RHS);
  return assignExpNewValueNum(exp).first;
}

/// Remove all entries from the ValueTable.
void ValueTable::clear() {
  valueNumbering.clear();
  expressionNumbering.clear();
  nextValueNumber = 1;
  Expressions.clear();
  ExprIdx.clear();
  nextExprNumber = 0;
}

/// Remove a value from the value numbering.
void ValueTable::erase(Value *V) {
  valueNumbering.erase(V);
}

/// verifyRemoved - Verify that the value is removed from all internal data
/// structures.
void ValueTable::verifyRemoved(const Value *V) const {
  for (DenseMap<Value*, uint32_t>::const_iterator
         I = valueNumbering.begin(), E = valueNumbering.end(); I != E; ++I) {
    assert(I->first != V && "Inst still occurs in value numbering map!");
  }
}

/// Return a pair the first field showing the value number of \p Exp and the
/// second field showing whether it is a value number newly created.
std::pair<uint32_t, bool>
ValueTable::assignExpNewValueNum(Expression &Exp) {
  uint32_t &e = expressionNumbering[Exp];
  bool CreateNewValNum = !e;
  if (CreateNewValNum) {
    Expressions.push_back(Exp);
    if (ExprIdx.size() < nextValueNumber + 1)
      ExprIdx.resize(nextValueNumber * 2);
    e = nextValueNumber;
    ExprIdx[nextValueNumber++] = nextExprNumber++;
  }
  return {e, CreateNewValNum};
}

} // namespace

namespace {
// Reduce code size for pattern like this:
// if (a.x > 0) {
//  r = tex.Sample(ss, uv)-1;
// } else {
//  if (a.y > 0)
//    r = tex.Sample(ss, uv);
//  else
//    r = tex.Sample(ss, uv) + 3;
// }
class DxilSimpleGVNHoist : public FunctionPass {

public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilSimpleGVNHoist() : FunctionPass(ID) {}

  StringRef getPassName() const override {
    return "DXIL simple GVN hoist";
  }

  bool runOnFunction(Function &F) override;

private:
  bool tryToHoist(BasicBlock *BB, BasicBlock *Succ0, BasicBlock *Succ1);
};

char DxilSimpleGVNHoist::ID = 0;

bool HasOnePred(BasicBlock *BB) {
  if (pred_empty(BB))
    return false;

  auto pred = pred_begin(BB);
  pred++;
  if (pred != pred_end(BB))
    return false;
  return true;
}

bool DxilSimpleGVNHoist::tryToHoist(BasicBlock *BB, BasicBlock *Succ0,
                                    BasicBlock *Succ1) {
  // ValueNumber Succ0 and Succ1.
  ValueTable VT;
  DenseMap<uint32_t, SmallVector<Instruction *, 2>> VNtoInsts;
  for (Instruction &I : *Succ0) {
    uint32_t V = VT.lookupOrAdd(&I);
    VNtoInsts[V].emplace_back(&I);
  }

  std::vector<uint32_t> HoistCandidateVN;

  for (Instruction &I : *Succ1) {
    uint32_t V = VT.lookupOrAdd(&I);
    if (!VNtoInsts.count(V))
      continue;
    VNtoInsts[V].emplace_back(&I);
    HoistCandidateVN.emplace_back(V);
  }

  if (HoistCandidateVN.empty()) {
    return false;
  }

  DenseSet<uint32_t> ProcessedVN;
  Instruction *TI = BB->getTerminator();
  // Hoist need to be in order, so operand could hoist before its users.
  for (uint32_t VN : HoistCandidateVN) {
    // Skip processed VN
    if (ProcessedVN.count(VN))
      continue;
    ProcessedVN.insert(VN);

    auto &Insts = VNtoInsts[VN];
    if (Insts.size() == 1)
      continue;
    bool bHoist = false;
    for (Instruction *I : Insts) {
      if (I->getParent() == Succ1) {
        bHoist = true;
        break;
      }
    }

    Instruction *FirstI = Insts.front();
    if (bHoist) {
      // When operand is different, need to hoist operand.
      auto it = Insts.begin();
      it++;
      bool bHasDifferentOperand = false;
      unsigned NumOps = FirstI->getNumOperands();
      for (; it != Insts.end(); it++) {
        Instruction *I = *it;
        assert(NumOps == I->getNumOperands());
        for (unsigned i = 0; i < NumOps; i++) {
          if (FirstI->getOperand(i) != I->getOperand(i)) {
            bHasDifferentOperand = true;
            break;
          }
        }
        if (bHasDifferentOperand)
          break;
      }
      // TODO: hoist operands.
      if (bHasDifferentOperand)
        continue;
      // Move FirstI to BB.
      FirstI->removeFromParent();
      FirstI->insertBefore(TI);
    }
    // Replace all insts with same value number with firstI.
    auto it = Insts.begin();
    it++;
    for (; it != Insts.end(); it++) {
      Instruction *I = *it;
      I->replaceAllUsesWith(FirstI);
      I->eraseFromParent();
    }
    Insts.clear();
  }
  return true;
}

bool DxilSimpleGVNHoist::runOnFunction(Function &F) {
  BasicBlock &Entry = F.getEntryBlock();
  bool bUpdated = false;
  for (auto it = po_begin(&Entry); it != po_end(&Entry); it++) {
    BasicBlock *BB = *it;
    TerminatorInst *TI = BB->getTerminator();
    if (TI->getNumSuccessors() != 2)
      continue;
    BasicBlock *Succ0 = TI->getSuccessor(0);
    BasicBlock *Succ1 = TI->getSuccessor(1);
    if (BB == Succ0)
      continue;
    if (BB == Succ1)
      continue;

    if (!HasOnePred(Succ0))
      continue;
    if (!HasOnePred(Succ1))
      continue;
    bUpdated |= tryToHoist(BB, Succ0, Succ1);
  }
  return bUpdated;
}

}

FunctionPass *llvm::createDxilSimpleGVNHoistPass() {
  return new DxilSimpleGVNHoist();
}

INITIALIZE_PASS(DxilSimpleGVNHoist, "dxil-gvn-hoist",
                "DXIL simple gvn hoist", false, false)

//================================================================================
//
// This pass tries to turn conditional branches to unconditional branches by
// proving two sides of branch are equivalent using ValueTable and dominator
// trees.
// 
// The algorithm:
//
// - Find any conditional branch 'Br' with successors 'S0' and 'S1', where
//   'Br' is their sole predecessor.
// - Find the common destination 'End' of the branches.
// - Find Find two predecessors 'P0' and 'P1' of 'End' such that 'S0' dominates
//   'P0' and 'P0' post-dominates 'S0', and 'P0' only has a single successor
//   (Same with 'S1' and 'P1').
//   This means if 'Br'->'S0' is taken, then 'End' must be reached via 'P0'; if
//   'End' is reached via 'P0', 'Br'->'S0' must have been taken (Same with 'S1'
//   and 'P1').
// - Using ValueTable, compare if every pair of incoming values from P0 and P1
//   is identical for any PHIs in 'End'
// - Make sure there is no side effect or loop between 'Br' and 'End'
// - If all above checks succeed, replace 'Br' with with an unconditional branch
//   to S0
// 
// The current state of the pass is pretty limited. If incoming values from P0
// and P1 are dependent on any PHIs defined between Br and End, then the pass
// will fail to simplify the branch. If there are any side effects within the
// region, the pass will also fail. It's possible to handling these cases, but
// require proving the two sides of the branch have equivalent control flow,
// which is non-trivial, and will be left to a later date.
//
namespace {

class DxilSimpleGVNEliminateRegion : public FunctionPass {

  bool RegionHasSideEffectsorLoops(BasicBlock *Begin, BasicBlock *End /*Non inclusive*/);

  std::unordered_map<BasicBlock *, bool> BlockHasSideEffects;
  bool MayHaveSideEffects(BasicBlock *BB) {
    auto It = BlockHasSideEffects.find(BB);
    if (It != BlockHasSideEffects.end())
      return It->second;
    bool HasSideEffects = false;
    for (Instruction &I : *BB) {
      if (I.mayHaveSideEffects() && !hlsl::IsNop(&I)) {
        HasSideEffects = true;
        break;
      }
    }
    BlockHasSideEffects[BB] = HasSideEffects;
    return HasSideEffects;
  }
  bool ProcessBB(BasicBlock &BB, ValueTable &VT, DominatorTree *DT, PostDominatorTree *PDT);

public:
  static char ID; // Pass identification, replacement for typeid
  explicit DxilSimpleGVNEliminateRegion() : FunctionPass(ID) {}

  StringRef getPassName() const override {
    return "DXIL simple GVN eliminate region";
  }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<PostDominatorTree>();
    AU.addRequired<DominatorTreeWrapperPass>();
  }
};

char DxilSimpleGVNEliminateRegion::ID = 0;

bool DxilSimpleGVNEliminateRegion::RegionHasSideEffectsorLoops(BasicBlock *Begin, BasicBlock *End /*Non inclusive*/) {
  SmallVector<BasicBlock *, 10> Worklist;
  Worklist.push_back(Begin);
  SmallPtrSet<BasicBlock *, 10> Seen;

  while (Worklist.size()) {
    BasicBlock *BB = Worklist.pop_back_val();
    // Stop before reaching End
    if (BB == End)
      continue;

    if (MayHaveSideEffects(BB))
      return false;

    Seen.insert(BB);
    for (BasicBlock *Succ : successors(BB)) {
      // Goes back into the region. Give up.
      if (Seen.count(Succ))
        return false;
      Worklist.push_back(Succ);
    }
  }
  return true;
}

bool DxilSimpleGVNEliminateRegion::ProcessBB(BasicBlock &BB, ValueTable &VT, DominatorTree *DT, PostDominatorTree *PDT) {
  TerminatorInst *TI = BB.getTerminator();
  if (TI->getNumSuccessors() != 2) {
    return false;
  }

  BasicBlock *S0 = TI->getSuccessor(0);
  BasicBlock *S1 = TI->getSuccessor(1);
  if (&BB == S0)
    return false;
  if (&BB == S1)
    return false;
  if (!HasOnePred(S0))
    return false;
  if (!HasOnePred(S1))
    return false;

  BasicBlock *End = PDT->findNearestCommonDominator(S0, S1);

  // Don't handle this situation for now.
  if (!End || S0 == End || S1 == End)
    return false;

  // If there's no phi node at the beginning of End, then either there's
  // side effects in the body or this is not the right pass to handle it.
  if (!isa<PHINode>(End->front()))
    return false;

  BasicBlock *P0 = nullptr;
  BasicBlock *P1 = nullptr;
  PHINode *FirstPHI = cast<PHINode>(&End->front());

  for (unsigned i = 0; i < FirstPHI->getNumIncomingValues(); i++) {
    BasicBlock *Incoming = FirstPHI->getIncomingBlock(i);
    if (!Incoming->getSingleSuccessor())
      continue;

    if (DT->dominates(S0, Incoming) && PDT->dominates(Incoming, S0)) {
      P0 = Incoming;
    }
    if (DT->dominates(S1, Incoming) && PDT->dominates(Incoming, S1)) {
      P1 = Incoming;
    }
  }

  if (!P0 || !P1 || P0 == P1)
    return false;

  for (Instruction &I : *End) {
    PHINode *Phi = dyn_cast<PHINode>(&I);
    if (!Phi)
      break;

    Value *Incoming0 = Phi->getIncomingValueForBlock(P0);
    Value *Incoming1 = Phi->getIncomingValueForBlock(P1);

    if (VT.lookupOrAdd(Incoming0) != VT.lookupOrAdd(Incoming1)) {
      return false;
    }
  }

  if (!RegionHasSideEffectsorLoops(S0, End))
    return false;
  if (!RegionHasSideEffectsorLoops(S1, End))
    return false;

  BranchInst *Br = BranchInst::Create(S0, &BB);
  Br->setDebugLoc(TI->getDebugLoc());
  TI->eraseFromParent();

  return true;
}

bool DxilSimpleGVNEliminateRegion::runOnFunction(Function &F) {
  PostDominatorTree *PDT = &getAnalysis<PostDominatorTree>();
  DominatorTree *DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();

  bool bChanged = false;
  ValueTable VT;

  for (BasicBlock &BB : F) {
    bChanged |= ProcessBB(BB, VT, DT, PDT);
  }

  return bChanged;
}

}

FunctionPass *llvm::createDxilSimpleGVNEliminateRegionPass() {
  return new DxilSimpleGVNEliminateRegion();
}

INITIALIZE_PASS(DxilSimpleGVNEliminateRegion, "dxil-gvn-eliminate-region",
                "DXIL simple eliminate region", false, false)

