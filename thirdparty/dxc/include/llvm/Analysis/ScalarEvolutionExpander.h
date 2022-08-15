//===---- llvm/Analysis/ScalarEvolutionExpander.h - SCEV Exprs --*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines the classes used to generate code from scalar expressions.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SCALAREVOLUTIONEXPANDER_H
#define LLVM_ANALYSIS_SCALAREVOLUTIONEXPANDER_H

#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ScalarEvolutionNormalization.h"
#include "llvm/Analysis/TargetFolder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/ValueHandle.h"
#include <set>

namespace llvm {
  class TargetTransformInfo;

  /// Return true if the given expression is safe to expand in the sense that
  /// all materialized values are safe to speculate.
  bool isSafeToExpand(const SCEV *S, ScalarEvolution &SE);

  /// This class uses information about analyze scalars to
  /// rewrite expressions in canonical form.
  ///
  /// Clients should create an instance of this class when rewriting is needed,
  /// and destroy it when finished to allow the release of the associated
  /// memory.
  class SCEVExpander : public SCEVVisitor<SCEVExpander, Value*> {
    ScalarEvolution &SE;
    const DataLayout &DL;

    // New instructions receive a name to identifies them with the current pass.
    const char* IVName;

    // InsertedExpressions caches Values for reuse, so must track RAUW.
    std::map<std::pair<const SCEV *, Instruction *>, TrackingVH<Value> >
      InsertedExpressions;
    // InsertedValues only flags inserted instructions so needs no RAUW.
    std::set<AssertingVH<Value> > InsertedValues;
    std::set<AssertingVH<Value> > InsertedPostIncValues;

    /// A memoization of the "relevant" loop for a given SCEV.
    DenseMap<const SCEV *, const Loop *> RelevantLoops;

    /// \brief Addrecs referring to any of the given loops are expanded
    /// in post-inc mode. For example, expanding {1,+,1}<L> in post-inc mode
    /// returns the add instruction that adds one to the phi for {0,+,1}<L>,
    /// as opposed to a new phi starting at 1. This is only supported in
    /// non-canonical mode.
    PostIncLoopSet PostIncLoops;

    /// \brief When this is non-null, addrecs expanded in the loop it indicates
    /// should be inserted with increments at IVIncInsertPos.
    const Loop *IVIncInsertLoop;

    /// \brief When expanding addrecs in the IVIncInsertLoop loop, insert the IV
    /// increment at this position.
    Instruction *IVIncInsertPos;

    /// \brief Phis that complete an IV chain. Reuse
    std::set<AssertingVH<PHINode> > ChainedPhis;

    /// \brief When true, expressions are expanded in "canonical" form. In
    /// particular, addrecs are expanded as arithmetic based on a canonical
    /// induction variable. When false, expression are expanded in a more
    /// literal form.
    bool CanonicalMode;

    /// \brief When invoked from LSR, the expander is in "strength reduction"
    /// mode. The only difference is that phi's are only reused if they are
    /// already in "expanded" form.
    bool LSRMode;

    typedef IRBuilder<true, TargetFolder> BuilderType;
    BuilderType Builder;

#ifndef NDEBUG
    const char *DebugType;
#endif

    friend struct SCEVVisitor<SCEVExpander, Value*>;

  public:
    /// \brief Construct a SCEVExpander in "canonical" mode.
    explicit SCEVExpander(ScalarEvolution &se, const DataLayout &DL,
                          const char *name)
        : SE(se), DL(DL), IVName(name), IVIncInsertLoop(nullptr),
          IVIncInsertPos(nullptr), CanonicalMode(true), LSRMode(false),
          Builder(se.getContext(), TargetFolder(DL)) {
#ifndef NDEBUG
      DebugType = "";
#endif
    }

#ifndef NDEBUG
    void setDebugType(const char* s) { DebugType = s; }
#endif

    /// \brief Erase the contents of the InsertedExpressions map so that users
    /// trying to expand the same expression into multiple BasicBlocks or
    /// different places within the same BasicBlock can do so.
    void clear() {
      InsertedExpressions.clear();
      InsertedValues.clear();
      InsertedPostIncValues.clear();
      ChainedPhis.clear();
    }

    /// \brief Return true for expressions that may incur non-trivial cost to
    /// evaluate at runtime.
    bool isHighCostExpansion(const SCEV *Expr, Loop *L) {
      SmallPtrSet<const SCEV *, 8> Processed;
      return isHighCostExpansionHelper(Expr, L, Processed);
    }

    /// \brief This method returns the canonical induction variable of the
    /// specified type for the specified loop (inserting one if there is none).
    /// A canonical induction variable starts at zero and steps by one on each
    /// iteration.
    PHINode *getOrInsertCanonicalInductionVariable(const Loop *L, Type *Ty);

    /// \brief Return the induction variable increment's IV operand.
    Instruction *getIVIncOperand(Instruction *IncV, Instruction *InsertPos,
                                 bool allowScale);

    /// \brief Utility for hoisting an IV increment.
    bool hoistIVInc(Instruction *IncV, Instruction *InsertPos);

    /// \brief replace congruent phis with their most canonical
    /// representative. Return the number of phis eliminated.
    unsigned replaceCongruentIVs(Loop *L, const DominatorTree *DT,
                                 SmallVectorImpl<WeakVH> &DeadInsts,
                                 const TargetTransformInfo *TTI = nullptr);

    /// \brief Insert code to directly compute the specified SCEV expression
    /// into the program.  The inserted code is inserted into the specified
    /// block.
    Value *expandCodeFor(const SCEV *SH, Type *Ty, Instruction *I);

    /// \brief Set the current IV increment loop and position.
    void setIVIncInsertPos(const Loop *L, Instruction *Pos) {
      assert(!CanonicalMode &&
             "IV increment positions are not supported in CanonicalMode");
      IVIncInsertLoop = L;
      IVIncInsertPos = Pos;
    }

    /// \brief Enable post-inc expansion for addrecs referring to the given
    /// loops. Post-inc expansion is only supported in non-canonical mode.
    void setPostInc(const PostIncLoopSet &L) {
      assert(!CanonicalMode &&
             "Post-inc expansion is not supported in CanonicalMode");
      PostIncLoops = L;
    }

    /// \brief Disable all post-inc expansion.
    void clearPostInc() {
      PostIncLoops.clear();

      // When we change the post-inc loop set, cached expansions may no
      // longer be valid.
      InsertedPostIncValues.clear();
    }

    /// \brief Disable the behavior of expanding expressions in canonical form
    /// rather than in a more literal form. Non-canonical mode is useful for
    /// late optimization passes.
    void disableCanonicalMode() { CanonicalMode = false; }

    void enableLSRMode() { LSRMode = true; }

    /// \brief Clear the current insertion point. This is useful if the
    /// instruction that had been serving as the insertion point may have been
    /// deleted.
    void clearInsertPoint() {
      Builder.ClearInsertionPoint();
    }

    /// \brief Return true if the specified instruction was inserted by the code
    /// rewriter.  If so, the client should not modify the instruction.
    bool isInsertedInstruction(Instruction *I) const {
      return InsertedValues.count(I) || InsertedPostIncValues.count(I);
    }

    void setChainedPhi(PHINode *PN) { ChainedPhis.insert(PN); }

  private:
    LLVMContext &getContext() const { return SE.getContext(); }

    /// \brief Recursive helper function for isHighCostExpansion.
    bool isHighCostExpansionHelper(const SCEV *S, Loop *L,
                                   SmallPtrSetImpl<const SCEV *> &Processed);

    /// \brief Insert the specified binary operator, doing a small amount
    /// of work to avoid inserting an obviously redundant operation.
    Value *InsertBinop(Instruction::BinaryOps Opcode, Value *LHS, Value *RHS);

    /// \brief Arrange for there to be a cast of V to Ty at IP, reusing an
    /// existing cast if a suitable one exists, moving an existing cast if a
    /// suitable one exists but isn't in the right place, or or creating a new
    /// one.
    Value *ReuseOrCreateCast(Value *V, Type *Ty,
                             Instruction::CastOps Op,
                             BasicBlock::iterator IP);

    /// \brief Insert a cast of V to the specified type, which must be possible
    /// with a noop cast, doing what we can to share the casts.
    Value *InsertNoopCastOfTo(Value *V, Type *Ty);

    /// \brief Expand a SCEVAddExpr with a pointer type into a GEP
    /// instead of using ptrtoint+arithmetic+inttoptr.
    Value *expandAddToGEP(const SCEV *const *op_begin,
                          const SCEV *const *op_end,
                          PointerType *PTy, Type *Ty, Value *V);

    Value *expand(const SCEV *S);

    /// \brief Insert code to directly compute the specified SCEV expression
    /// into the program.  The inserted code is inserted into the SCEVExpander's
    /// current insertion point. If a type is specified, the result will be
    /// expanded to have that type, with a cast if necessary.
    Value *expandCodeFor(const SCEV *SH, Type *Ty = nullptr);

    /// \brief Determine the most "relevant" loop for the given SCEV.
    const Loop *getRelevantLoop(const SCEV *);

    Value *visitConstant(const SCEVConstant *S) {
      return S->getValue();
    }

    Value *visitTruncateExpr(const SCEVTruncateExpr *S);

    Value *visitZeroExtendExpr(const SCEVZeroExtendExpr *S);

    Value *visitSignExtendExpr(const SCEVSignExtendExpr *S);

    Value *visitAddExpr(const SCEVAddExpr *S);

    Value *visitMulExpr(const SCEVMulExpr *S);

    Value *visitUDivExpr(const SCEVUDivExpr *S);

    Value *visitAddRecExpr(const SCEVAddRecExpr *S);

    Value *visitSMaxExpr(const SCEVSMaxExpr *S);

    Value *visitUMaxExpr(const SCEVUMaxExpr *S);

    Value *visitUnknown(const SCEVUnknown *S) {
      return S->getValue();
    }

    void rememberInstruction(Value *I);

    bool isNormalAddRecExprPHI(PHINode *PN, Instruction *IncV, const Loop *L);

    bool isExpandedAddRecExprPHI(PHINode *PN, Instruction *IncV, const Loop *L);

    Value *expandAddRecExprLiterally(const SCEVAddRecExpr *);
    PHINode *getAddRecExprPHILiterally(const SCEVAddRecExpr *Normalized,
                                       const Loop *L,
                                       Type *ExpandTy,
                                       Type *IntTy,
                                       Type *&TruncTy,
                                       bool &InvertStep);
    Value *expandIVInc(PHINode *PN, Value *StepV, const Loop *L,
                       Type *ExpandTy, Type *IntTy, bool useSubtract);
  };
}

#endif
