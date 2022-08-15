//===- InstCombineAndOrXor.cpp --------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the visitAnd, visitOr, and visitXor functions.
//
//===----------------------------------------------------------------------===//

#include "InstCombineInternal.h"
#include "llvm/Analysis/InstructionSimplify.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Transforms/Utils/CmpInstAnalysis.h"
using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "instcombine"

static inline Value *dyn_castNotVal(Value *V) {
  // If this is not(not(x)) don't return that this is a not: we want the two
  // not's to be folded first.
  if (BinaryOperator::isNot(V)) {
    Value *Operand = BinaryOperator::getNotArgument(V);
    if (!IsFreeToInvert(Operand, Operand->hasOneUse()))
      return Operand;
  }

  // Constants can be considered to be not'ed values...
  if (ConstantInt *C = dyn_cast<ConstantInt>(V))
    return ConstantInt::get(C->getType(), ~C->getValue());
  return nullptr;
}

/// getFCmpCode - Similar to getICmpCode but for FCmpInst. This encodes a fcmp
/// predicate into a three bit mask. It also returns whether it is an ordered
/// predicate by reference.
static unsigned getFCmpCode(FCmpInst::Predicate CC, bool &isOrdered) {
  isOrdered = false;
  switch (CC) {
  case FCmpInst::FCMP_ORD: isOrdered = true; return 0;  // 000
  case FCmpInst::FCMP_UNO:                   return 0;  // 000
  case FCmpInst::FCMP_OGT: isOrdered = true; return 1;  // 001
  case FCmpInst::FCMP_UGT:                   return 1;  // 001
  case FCmpInst::FCMP_OEQ: isOrdered = true; return 2;  // 010
  case FCmpInst::FCMP_UEQ:                   return 2;  // 010
  case FCmpInst::FCMP_OGE: isOrdered = true; return 3;  // 011
  case FCmpInst::FCMP_UGE:                   return 3;  // 011
  case FCmpInst::FCMP_OLT: isOrdered = true; return 4;  // 100
  case FCmpInst::FCMP_ULT:                   return 4;  // 100
  case FCmpInst::FCMP_ONE: isOrdered = true; return 5;  // 101
  case FCmpInst::FCMP_UNE:                   return 5;  // 101
  case FCmpInst::FCMP_OLE: isOrdered = true; return 6;  // 110
  case FCmpInst::FCMP_ULE:                   return 6;  // 110
    // True -> 7
  default:
    // Not expecting FCMP_FALSE and FCMP_TRUE;
    llvm_unreachable("Unexpected FCmp predicate!");
  }
}

/// getNewICmpValue - This is the complement of getICmpCode, which turns an
/// opcode and two operands into either a constant true or false, or a brand
/// new ICmp instruction. The sign is passed in to determine which kind
/// of predicate to use in the new icmp instruction.
static Value *getNewICmpValue(bool Sign, unsigned Code, Value *LHS, Value *RHS,
                              InstCombiner::BuilderTy *Builder) {
  ICmpInst::Predicate NewPred;
  if (Value *NewConstant = getICmpValue(Sign, Code, LHS, RHS, NewPred))
    return NewConstant;
  return Builder->CreateICmp(NewPred, LHS, RHS);
}

/// getFCmpValue - This is the complement of getFCmpCode, which turns an
/// opcode and two operands into either a FCmp instruction. isordered is passed
/// in to determine which kind of predicate to use in the new fcmp instruction.
static Value *getFCmpValue(bool isordered, unsigned code,
                           Value *LHS, Value *RHS,
                           InstCombiner::BuilderTy *Builder) {
  CmpInst::Predicate Pred;
  switch (code) {
  default: llvm_unreachable("Illegal FCmp code!");
  case 0: Pred = isordered ? FCmpInst::FCMP_ORD : FCmpInst::FCMP_UNO; break;
  case 1: Pred = isordered ? FCmpInst::FCMP_OGT : FCmpInst::FCMP_UGT; break;
  case 2: Pred = isordered ? FCmpInst::FCMP_OEQ : FCmpInst::FCMP_UEQ; break;
  case 3: Pred = isordered ? FCmpInst::FCMP_OGE : FCmpInst::FCMP_UGE; break;
  case 4: Pred = isordered ? FCmpInst::FCMP_OLT : FCmpInst::FCMP_ULT; break;
  case 5: Pred = isordered ? FCmpInst::FCMP_ONE : FCmpInst::FCMP_UNE; break;
  case 6: Pred = isordered ? FCmpInst::FCMP_OLE : FCmpInst::FCMP_ULE; break;
  case 7:
    if (!isordered)
      return ConstantInt::get(CmpInst::makeCmpResultType(LHS->getType()), 1);
    Pred = FCmpInst::FCMP_ORD; break;
  }
  return Builder->CreateFCmp(Pred, LHS, RHS);
}

/// \brief Transform BITWISE_OP(BSWAP(A),BSWAP(B)) to BSWAP(BITWISE_OP(A, B))
/// \param I Binary operator to transform.
/// \return Pointer to node that must replace the original binary operator, or
///         null pointer if no transformation was made.
Value *InstCombiner::SimplifyBSwap(BinaryOperator &I) {
  IntegerType *ITy = dyn_cast<IntegerType>(I.getType());

  // Can't do vectors.
  if (I.getType()->isVectorTy()) return nullptr;

  // Can only do bitwise ops.
  unsigned Op = I.getOpcode();
  if (Op != Instruction::And && Op != Instruction::Or &&
      Op != Instruction::Xor)
    return nullptr;

  Value *OldLHS = I.getOperand(0);
  Value *OldRHS = I.getOperand(1);
  ConstantInt *ConstLHS = dyn_cast<ConstantInt>(OldLHS);
  ConstantInt *ConstRHS = dyn_cast<ConstantInt>(OldRHS);
  IntrinsicInst *IntrLHS = dyn_cast<IntrinsicInst>(OldLHS);
  IntrinsicInst *IntrRHS = dyn_cast<IntrinsicInst>(OldRHS);
  bool IsBswapLHS = (IntrLHS && IntrLHS->getIntrinsicID() == Intrinsic::bswap);
  bool IsBswapRHS = (IntrRHS && IntrRHS->getIntrinsicID() == Intrinsic::bswap);

  if (!IsBswapLHS && !IsBswapRHS)
    return nullptr;

  if (!IsBswapLHS && !ConstLHS)
    return nullptr;

  if (!IsBswapRHS && !ConstRHS)
    return nullptr;

  /// OP( BSWAP(x), BSWAP(y) ) -> BSWAP( OP(x, y) )
  /// OP( BSWAP(x), CONSTANT ) -> BSWAP( OP(x, BSWAP(CONSTANT) ) )
  Value *NewLHS = IsBswapLHS ? IntrLHS->getOperand(0) :
                  Builder->getInt(ConstLHS->getValue().byteSwap());

  Value *NewRHS = IsBswapRHS ? IntrRHS->getOperand(0) :
                  Builder->getInt(ConstRHS->getValue().byteSwap());

  Value *BinOp = nullptr;
  if (Op == Instruction::And)
    BinOp = Builder->CreateAnd(NewLHS, NewRHS);
  else if (Op == Instruction::Or)
    BinOp = Builder->CreateOr(NewLHS, NewRHS);
  else //if (Op == Instruction::Xor)
    BinOp = Builder->CreateXor(NewLHS, NewRHS);

  Module *M = I.getParent()->getParent()->getParent();
  Function *F = Intrinsic::getDeclaration(M, Intrinsic::bswap, ITy);
  return Builder->CreateCall(F, BinOp);
}

// OptAndOp - This handles expressions of the form ((val OP C1) & C2).  Where
// the Op parameter is 'OP', OpRHS is 'C1', and AndRHS is 'C2'.  Op is
// guaranteed to be a binary operator.
Instruction *InstCombiner::OptAndOp(Instruction *Op,
                                    ConstantInt *OpRHS,
                                    ConstantInt *AndRHS,
                                    BinaryOperator &TheAnd) {
  Value *X = Op->getOperand(0);
  Constant *Together = nullptr;
  if (!Op->isShift())
    Together = ConstantExpr::getAnd(AndRHS, OpRHS);

  switch (Op->getOpcode()) {
  case Instruction::Xor:
    if (Op->hasOneUse()) {
      // (X ^ C1) & C2 --> (X & C2) ^ (C1&C2)
      Value *And = Builder->CreateAnd(X, AndRHS);
      And->takeName(Op);
      return BinaryOperator::CreateXor(And, Together);
    }
    break;
  case Instruction::Or:
    if (Op->hasOneUse()){
      if (Together != OpRHS) {
        // (X | C1) & C2 --> (X | (C1&C2)) & C2
        Value *Or = Builder->CreateOr(X, Together);
        Or->takeName(Op);
        return BinaryOperator::CreateAnd(Or, AndRHS);
      }

      ConstantInt *TogetherCI = dyn_cast<ConstantInt>(Together);
      if (TogetherCI && !TogetherCI->isZero()){
        // (X | C1) & C2 --> (X & (C2^(C1&C2))) | C1
        // NOTE: This reduces the number of bits set in the & mask, which
        // can expose opportunities for store narrowing.
        Together = ConstantExpr::getXor(AndRHS, Together);
        Value *And = Builder->CreateAnd(X, Together);
        And->takeName(Op);
        return BinaryOperator::CreateOr(And, OpRHS);
      }
    }

    break;
  case Instruction::Add:
    if (Op->hasOneUse()) {
      // Adding a one to a single bit bit-field should be turned into an XOR
      // of the bit.  First thing to check is to see if this AND is with a
      // single bit constant.
      const APInt &AndRHSV = AndRHS->getValue();

      // If there is only one bit set.
      if (AndRHSV.isPowerOf2()) {
        // Ok, at this point, we know that we are masking the result of the
        // ADD down to exactly one bit.  If the constant we are adding has
        // no bits set below this bit, then we can eliminate the ADD.
        const APInt& AddRHS = OpRHS->getValue();

        // Check to see if any bits below the one bit set in AndRHSV are set.
        if ((AddRHS & (AndRHSV-1)) == 0) {
          // If not, the only thing that can effect the output of the AND is
          // the bit specified by AndRHSV.  If that bit is set, the effect of
          // the XOR is to toggle the bit.  If it is clear, then the ADD has
          // no effect.
          if ((AddRHS & AndRHSV) == 0) { // Bit is not set, noop
            TheAnd.setOperand(0, X);
            return &TheAnd;
          } else {
            // Pull the XOR out of the AND.
            Value *NewAnd = Builder->CreateAnd(X, AndRHS);
            NewAnd->takeName(Op);
            return BinaryOperator::CreateXor(NewAnd, AndRHS);
          }
        }
      }
    }
    break;

  case Instruction::Shl: {
    // We know that the AND will not produce any of the bits shifted in, so if
    // the anded constant includes them, clear them now!
    //
    uint32_t BitWidth = AndRHS->getType()->getBitWidth();
    uint32_t OpRHSVal = OpRHS->getLimitedValue(BitWidth);
    APInt ShlMask(APInt::getHighBitsSet(BitWidth, BitWidth-OpRHSVal));
    ConstantInt *CI = Builder->getInt(AndRHS->getValue() & ShlMask);

    if (CI->getValue() == ShlMask)
      // Masking out bits that the shift already masks.
      return ReplaceInstUsesWith(TheAnd, Op);   // No need for the and.

    if (CI != AndRHS) {                  // Reducing bits set in and.
      TheAnd.setOperand(1, CI);
      return &TheAnd;
    }
    break;
  }
  case Instruction::LShr: {
    // We know that the AND will not produce any of the bits shifted in, so if
    // the anded constant includes them, clear them now!  This only applies to
    // unsigned shifts, because a signed shr may bring in set bits!
    //
    uint32_t BitWidth = AndRHS->getType()->getBitWidth();
    uint32_t OpRHSVal = OpRHS->getLimitedValue(BitWidth);
    APInt ShrMask(APInt::getLowBitsSet(BitWidth, BitWidth - OpRHSVal));
    ConstantInt *CI = Builder->getInt(AndRHS->getValue() & ShrMask);

    if (CI->getValue() == ShrMask)
      // Masking out bits that the shift already masks.
      return ReplaceInstUsesWith(TheAnd, Op);

    if (CI != AndRHS) {
      TheAnd.setOperand(1, CI);  // Reduce bits set in and cst.
      return &TheAnd;
    }
    break;
  }
  case Instruction::AShr:
    // Signed shr.
    // See if this is shifting in some sign extension, then masking it out
    // with an and.
    if (Op->hasOneUse()) {
      uint32_t BitWidth = AndRHS->getType()->getBitWidth();
      uint32_t OpRHSVal = OpRHS->getLimitedValue(BitWidth);
      APInt ShrMask(APInt::getLowBitsSet(BitWidth, BitWidth - OpRHSVal));
      Constant *C = Builder->getInt(AndRHS->getValue() & ShrMask);
      if (C == AndRHS) {          // Masking out bits shifted in.
        // (Val ashr C1) & C2 -> (Val lshr C1) & C2
        // Make the argument unsigned.
        Value *ShVal = Op->getOperand(0);
        ShVal = Builder->CreateLShr(ShVal, OpRHS, Op->getName());
        return BinaryOperator::CreateAnd(ShVal, AndRHS, TheAnd.getName());
      }
    }
    break;
  }
  return nullptr;
}

/// Emit a computation of: (V >= Lo && V < Hi) if Inside is true, otherwise
/// (V < Lo || V >= Hi).  In practice, we emit the more efficient
/// (V-Lo) \<u Hi-Lo.  This method expects that Lo <= Hi. isSigned indicates
/// whether to treat the V, Lo and HI as signed or not. IB is the location to
/// insert new instructions.
Value *InstCombiner::InsertRangeTest(Value *V, Constant *Lo, Constant *Hi,
                                     bool isSigned, bool Inside) {
  assert(cast<ConstantInt>(ConstantExpr::getICmp((isSigned ?
            ICmpInst::ICMP_SLE:ICmpInst::ICMP_ULE), Lo, Hi))->getZExtValue() &&
         "Lo is not <= Hi in range emission code!");

  if (Inside) {
    if (Lo == Hi)  // Trivially false.
      return Builder->getFalse();

    // V >= Min && V < Hi --> V < Hi
    if (cast<ConstantInt>(Lo)->isMinValue(isSigned)) {
      ICmpInst::Predicate pred = (isSigned ?
        ICmpInst::ICMP_SLT : ICmpInst::ICMP_ULT);
      return Builder->CreateICmp(pred, V, Hi);
    }

    // Emit V-Lo <u Hi-Lo
    Constant *NegLo = ConstantExpr::getNeg(Lo);
    Value *Add = Builder->CreateAdd(V, NegLo, V->getName()+".off");
    Constant *UpperBound = ConstantExpr::getAdd(NegLo, Hi);
    return Builder->CreateICmpULT(Add, UpperBound);
  }

  if (Lo == Hi)  // Trivially true.
    return Builder->getTrue();

  // V < Min || V >= Hi -> V > Hi-1
  Hi = SubOne(cast<ConstantInt>(Hi));
  if (cast<ConstantInt>(Lo)->isMinValue(isSigned)) {
    ICmpInst::Predicate pred = (isSigned ?
        ICmpInst::ICMP_SGT : ICmpInst::ICMP_UGT);
    return Builder->CreateICmp(pred, V, Hi);
  }

  // Emit V-Lo >u Hi-1-Lo
  // Note that Hi has already had one subtracted from it, above.
  ConstantInt *NegLo = cast<ConstantInt>(ConstantExpr::getNeg(Lo));
  Value *Add = Builder->CreateAdd(V, NegLo, V->getName()+".off");
  Constant *LowerBound = ConstantExpr::getAdd(NegLo, Hi);
  return Builder->CreateICmpUGT(Add, LowerBound);
}

// isRunOfOnes - Returns true iff Val consists of one contiguous run of 1s with
// any number of 0s on either side.  The 1s are allowed to wrap from LSB to
// MSB, so 0x000FFF0, 0x0000FFFF, and 0xFF0000FF are all runs.  0x0F0F0000 is
// not, since all 1s are not contiguous.
static bool isRunOfOnes(ConstantInt *Val, uint32_t &MB, uint32_t &ME) {
  const APInt& V = Val->getValue();
  uint32_t BitWidth = Val->getType()->getBitWidth();
  if (!APIntOps::isShiftedMask(BitWidth, V)) return false;

  // look for the first zero bit after the run of ones
  MB = BitWidth - ((V - 1) ^ V).countLeadingZeros();
  // look for the first non-zero bit
  ME = V.getActiveBits();
  return true;
}

/// FoldLogicalPlusAnd - This is part of an expression (LHS +/- RHS) & Mask,
/// where isSub determines whether the operator is a sub.  If we can fold one of
/// the following xforms:
///
/// ((A & N) +/- B) & Mask -> (A +/- B) & Mask iff N&Mask == Mask
/// ((A | N) +/- B) & Mask -> (A +/- B) & Mask iff N&Mask == 0
/// ((A ^ N) +/- B) & Mask -> (A +/- B) & Mask iff N&Mask == 0
///
/// return (A +/- B).
///
Value *InstCombiner::FoldLogicalPlusAnd(Value *LHS, Value *RHS,
                                        ConstantInt *Mask, bool isSub,
                                        Instruction &I) {
  Instruction *LHSI = dyn_cast<Instruction>(LHS);
  if (!LHSI || LHSI->getNumOperands() != 2 ||
      !isa<ConstantInt>(LHSI->getOperand(1))) return nullptr;

  ConstantInt *N = cast<ConstantInt>(LHSI->getOperand(1));

  switch (LHSI->getOpcode()) {
  default: return nullptr;
  case Instruction::And:
    if (ConstantExpr::getAnd(N, Mask) == Mask) {
      // If the AndRHS is a power of two minus one (0+1+), this is simple.
      if ((Mask->getValue().countLeadingZeros() +
           Mask->getValue().countPopulation()) ==
          Mask->getValue().getBitWidth())
        break;

      // Otherwise, if Mask is 0+1+0+, and if B is known to have the low 0+
      // part, we don't need any explicit masks to take them out of A.  If that
      // is all N is, ignore it.
      uint32_t MB = 0, ME = 0;
      if (isRunOfOnes(Mask, MB, ME)) {  // begin/end bit of run, inclusive
        uint32_t BitWidth = cast<IntegerType>(RHS->getType())->getBitWidth();
        APInt Mask(APInt::getLowBitsSet(BitWidth, MB-1));
        if (MaskedValueIsZero(RHS, Mask, 0, &I))
          break;
      }
    }
    return nullptr;
  case Instruction::Or:
  case Instruction::Xor:
    // If the AndRHS is a power of two minus one (0+1+), and N&Mask == 0
    if ((Mask->getValue().countLeadingZeros() +
         Mask->getValue().countPopulation()) == Mask->getValue().getBitWidth()
        && ConstantExpr::getAnd(N, Mask)->isNullValue())
      break;
    return nullptr;
  }

  if (isSub)
    return Builder->CreateSub(LHSI->getOperand(0), RHS, "fold");
  return Builder->CreateAdd(LHSI->getOperand(0), RHS, "fold");
}

/// enum for classifying (icmp eq (A & B), C) and (icmp ne (A & B), C)
/// One of A and B is considered the mask, the other the value. This is
/// described as the "AMask" or "BMask" part of the enum. If the enum
/// contains only "Mask", then both A and B can be considered masks.
/// If A is the mask, then it was proven, that (A & C) == C. This
/// is trivial if C == A, or C == 0. If both A and C are constants, this
/// proof is also easy.
/// For the following explanations we assume that A is the mask.
/// The part "AllOnes" declares, that the comparison is true only
/// if (A & B) == A, or all bits of A are set in B.
///   Example: (icmp eq (A & 3), 3) -> FoldMskICmp_AMask_AllOnes
/// The part "AllZeroes" declares, that the comparison is true only
/// if (A & B) == 0, or all bits of A are cleared in B.
///   Example: (icmp eq (A & 3), 0) -> FoldMskICmp_Mask_AllZeroes
/// The part "Mixed" declares, that (A & B) == C and C might or might not
/// contain any number of one bits and zero bits.
///   Example: (icmp eq (A & 3), 1) -> FoldMskICmp_AMask_Mixed
/// The Part "Not" means, that in above descriptions "==" should be replaced
/// by "!=".
///   Example: (icmp ne (A & 3), 3) -> FoldMskICmp_AMask_NotAllOnes
/// If the mask A contains a single bit, then the following is equivalent:
///    (icmp eq (A & B), A) equals (icmp ne (A & B), 0)
///    (icmp ne (A & B), A) equals (icmp eq (A & B), 0)
enum MaskedICmpType {
  FoldMskICmp_AMask_AllOnes           =     1,
  FoldMskICmp_AMask_NotAllOnes        =     2,
  FoldMskICmp_BMask_AllOnes           =     4,
  FoldMskICmp_BMask_NotAllOnes        =     8,
  FoldMskICmp_Mask_AllZeroes          =    16,
  FoldMskICmp_Mask_NotAllZeroes       =    32,
  FoldMskICmp_AMask_Mixed             =    64,
  FoldMskICmp_AMask_NotMixed          =   128,
  FoldMskICmp_BMask_Mixed             =   256,
  FoldMskICmp_BMask_NotMixed          =   512
};

/// return the set of pattern classes (from MaskedICmpType)
/// that (icmp SCC (A & B), C) satisfies
static unsigned getTypeOfMaskedICmp(Value* A, Value* B, Value* C,
                                    ICmpInst::Predicate SCC)
{
  ConstantInt *ACst = dyn_cast<ConstantInt>(A);
  ConstantInt *BCst = dyn_cast<ConstantInt>(B);
  ConstantInt *CCst = dyn_cast<ConstantInt>(C);
  bool icmp_eq = (SCC == ICmpInst::ICMP_EQ);
  bool icmp_abit = (ACst && !ACst->isZero() &&
                    ACst->getValue().isPowerOf2());
  bool icmp_bbit = (BCst && !BCst->isZero() &&
                    BCst->getValue().isPowerOf2());
  unsigned result = 0;
  if (CCst && CCst->isZero()) {
    // if C is zero, then both A and B qualify as mask
    result |= (icmp_eq ? (FoldMskICmp_Mask_AllZeroes |
                          FoldMskICmp_Mask_AllZeroes |
                          FoldMskICmp_AMask_Mixed |
                          FoldMskICmp_BMask_Mixed)
                       : (FoldMskICmp_Mask_NotAllZeroes |
                          FoldMskICmp_Mask_NotAllZeroes |
                          FoldMskICmp_AMask_NotMixed |
                          FoldMskICmp_BMask_NotMixed));
    if (icmp_abit)
      result |= (icmp_eq ? (FoldMskICmp_AMask_NotAllOnes |
                            FoldMskICmp_AMask_NotMixed)
                         : (FoldMskICmp_AMask_AllOnes |
                            FoldMskICmp_AMask_Mixed));
    if (icmp_bbit)
      result |= (icmp_eq ? (FoldMskICmp_BMask_NotAllOnes |
                            FoldMskICmp_BMask_NotMixed)
                         : (FoldMskICmp_BMask_AllOnes |
                            FoldMskICmp_BMask_Mixed));
    return result;
  }
  if (A == C) {
    result |= (icmp_eq ? (FoldMskICmp_AMask_AllOnes |
                          FoldMskICmp_AMask_Mixed)
                       : (FoldMskICmp_AMask_NotAllOnes |
                          FoldMskICmp_AMask_NotMixed));
    if (icmp_abit)
      result |= (icmp_eq ? (FoldMskICmp_Mask_NotAllZeroes |
                            FoldMskICmp_AMask_NotMixed)
                         : (FoldMskICmp_Mask_AllZeroes |
                            FoldMskICmp_AMask_Mixed));
  } else if (ACst && CCst &&
             ConstantExpr::getAnd(ACst, CCst) == CCst) {
    result |= (icmp_eq ? FoldMskICmp_AMask_Mixed
                       : FoldMskICmp_AMask_NotMixed);
  }
  if (B == C) {
    result |= (icmp_eq ? (FoldMskICmp_BMask_AllOnes |
                          FoldMskICmp_BMask_Mixed)
                       : (FoldMskICmp_BMask_NotAllOnes |
                          FoldMskICmp_BMask_NotMixed));
    if (icmp_bbit)
      result |= (icmp_eq ? (FoldMskICmp_Mask_NotAllZeroes |
                            FoldMskICmp_BMask_NotMixed)
                         : (FoldMskICmp_Mask_AllZeroes |
                            FoldMskICmp_BMask_Mixed));
  } else if (BCst && CCst &&
             ConstantExpr::getAnd(BCst, CCst) == CCst) {
    result |= (icmp_eq ? FoldMskICmp_BMask_Mixed
                       : FoldMskICmp_BMask_NotMixed);
  }
  return result;
}

/// Convert an analysis of a masked ICmp into its equivalent if all boolean
/// operations had the opposite sense. Since each "NotXXX" flag (recording !=)
/// is adjacent to the corresponding normal flag (recording ==), this just
/// involves swapping those bits over.
static unsigned conjugateICmpMask(unsigned Mask) {
  unsigned NewMask;
  NewMask = (Mask & (FoldMskICmp_AMask_AllOnes | FoldMskICmp_BMask_AllOnes |
                     FoldMskICmp_Mask_AllZeroes | FoldMskICmp_AMask_Mixed |
                     FoldMskICmp_BMask_Mixed))
            << 1;

  NewMask |=
      (Mask & (FoldMskICmp_AMask_NotAllOnes | FoldMskICmp_BMask_NotAllOnes |
               FoldMskICmp_Mask_NotAllZeroes | FoldMskICmp_AMask_NotMixed |
               FoldMskICmp_BMask_NotMixed))
      >> 1;

  return NewMask;
}

/// decomposeBitTestICmp - Decompose an icmp into the form ((X & Y) pred Z)
/// if possible. The returned predicate is either == or !=. Returns false if
/// decomposition fails.
static bool decomposeBitTestICmp(const ICmpInst *I, ICmpInst::Predicate &Pred,
                                 Value *&X, Value *&Y, Value *&Z) {
  ConstantInt *C = dyn_cast<ConstantInt>(I->getOperand(1));
  if (!C)
    return false;

  switch (I->getPredicate()) {
  default:
    return false;
  case ICmpInst::ICMP_SLT:
    // X < 0 is equivalent to (X & SignBit) != 0.
    if (!C->isZero())
      return false;
    Y = ConstantInt::get(I->getContext(), APInt::getSignBit(C->getBitWidth()));
    Pred = ICmpInst::ICMP_NE;
    break;
  case ICmpInst::ICMP_SGT:
    // X > -1 is equivalent to (X & SignBit) == 0.
    if (!C->isAllOnesValue())
      return false;
    Y = ConstantInt::get(I->getContext(), APInt::getSignBit(C->getBitWidth()));
    Pred = ICmpInst::ICMP_EQ;
    break;
  case ICmpInst::ICMP_ULT:
    // X <u 2^n is equivalent to (X & ~(2^n-1)) == 0.
    if (!C->getValue().isPowerOf2())
      return false;
    Y = ConstantInt::get(I->getContext(), -C->getValue());
    Pred = ICmpInst::ICMP_EQ;
    break;
  case ICmpInst::ICMP_UGT:
    // X >u 2^n-1 is equivalent to (X & ~(2^n-1)) != 0.
    if (!(C->getValue() + 1).isPowerOf2())
      return false;
    Y = ConstantInt::get(I->getContext(), ~C->getValue());
    Pred = ICmpInst::ICMP_NE;
    break;
  }

  X = I->getOperand(0);
  Z = ConstantInt::getNullValue(C->getType());
  return true;
}

/// foldLogOpOfMaskedICmpsHelper:
/// handle (icmp(A & B) ==/!= C) &/| (icmp(A & D) ==/!= E)
/// return the set of pattern classes (from MaskedICmpType)
/// that both LHS and RHS satisfy
static unsigned foldLogOpOfMaskedICmpsHelper(Value*& A,
                                             Value*& B, Value*& C,
                                             Value*& D, Value*& E,
                                             ICmpInst *LHS, ICmpInst *RHS,
                                             ICmpInst::Predicate &LHSCC,
                                             ICmpInst::Predicate &RHSCC) {
  if (LHS->getOperand(0)->getType() != RHS->getOperand(0)->getType()) return 0;
  // vectors are not (yet?) supported
  if (LHS->getOperand(0)->getType()->isVectorTy()) return 0;

  // Here comes the tricky part:
  // LHS might be of the form L11 & L12 == X, X == L21 & L22,
  // and L11 & L12 == L21 & L22. The same goes for RHS.
  // Now we must find those components L** and R**, that are equal, so
  // that we can extract the parameters A, B, C, D, and E for the canonical
  // above.
  Value *L1 = LHS->getOperand(0);
  Value *L2 = LHS->getOperand(1);
  Value *L11,*L12,*L21,*L22;
  // Check whether the icmp can be decomposed into a bit test.
  if (decomposeBitTestICmp(LHS, LHSCC, L11, L12, L2)) {
    L21 = L22 = L1 = nullptr;
  } else {
    // Look for ANDs in the LHS icmp.
    if (!L1->getType()->isIntegerTy()) {
      // You can icmp pointers, for example. They really aren't masks.
      L11 = L12 = nullptr;
    } else if (!match(L1, m_And(m_Value(L11), m_Value(L12)))) {
      // Any icmp can be viewed as being trivially masked; if it allows us to
      // remove one, it's worth it.
      L11 = L1;
      L12 = Constant::getAllOnesValue(L1->getType());
    }

    if (!L2->getType()->isIntegerTy()) {
      // You can icmp pointers, for example. They really aren't masks.
      L21 = L22 = nullptr;
    } else if (!match(L2, m_And(m_Value(L21), m_Value(L22)))) {
      L21 = L2;
      L22 = Constant::getAllOnesValue(L2->getType());
    }
  }

  // Bail if LHS was a icmp that can't be decomposed into an equality.
  if (!ICmpInst::isEquality(LHSCC))
    return 0;

  Value *R1 = RHS->getOperand(0);
  Value *R2 = RHS->getOperand(1);
  Value *R11,*R12;
  bool ok = false;
  if (decomposeBitTestICmp(RHS, RHSCC, R11, R12, R2)) {
    if (R11 == L11 || R11 == L12 || R11 == L21 || R11 == L22) {
      A = R11; D = R12;
    } else if (R12 == L11 || R12 == L12 || R12 == L21 || R12 == L22) {
      A = R12; D = R11;
    } else {
      return 0;
    }
    E = R2; R1 = nullptr; ok = true;
  } else if (R1->getType()->isIntegerTy()) {
    if (!match(R1, m_And(m_Value(R11), m_Value(R12)))) {
      // As before, model no mask as a trivial mask if it'll let us do an
      // optimization.
      R11 = R1;
      R12 = Constant::getAllOnesValue(R1->getType());
    }

    if (R11 == L11 || R11 == L12 || R11 == L21 || R11 == L22) {
      A = R11; D = R12; E = R2; ok = true;
    } else if (R12 == L11 || R12 == L12 || R12 == L21 || R12 == L22) {
      A = R12; D = R11; E = R2; ok = true;
    }
  }

  // Bail if RHS was a icmp that can't be decomposed into an equality.
  if (!ICmpInst::isEquality(RHSCC))
    return 0;

  // Look for ANDs in on the right side of the RHS icmp.
  if (!ok && R2->getType()->isIntegerTy()) {
    if (!match(R2, m_And(m_Value(R11), m_Value(R12)))) {
      R11 = R2;
      R12 = Constant::getAllOnesValue(R2->getType());
    }

    if (R11 == L11 || R11 == L12 || R11 == L21 || R11 == L22) {
      A = R11; D = R12; E = R1; ok = true;
    } else if (R12 == L11 || R12 == L12 || R12 == L21 || R12 == L22) {
      A = R12; D = R11; E = R1; ok = true;
    } else {
      return 0;
    }
  }
  if (!ok)
    return 0;

  if (L11 == A) {
    B = L12; C = L2;
  } else if (L12 == A) {
    B = L11; C = L2;
  } else if (L21 == A) {
    B = L22; C = L1;
  } else if (L22 == A) {
    B = L21; C = L1;
  }

  unsigned left_type = getTypeOfMaskedICmp(A, B, C, LHSCC);
  unsigned right_type = getTypeOfMaskedICmp(A, D, E, RHSCC);
  return left_type & right_type;
}
/// foldLogOpOfMaskedICmps:
/// try to fold (icmp(A & B) ==/!= C) &/| (icmp(A & D) ==/!= E)
/// into a single (icmp(A & X) ==/!= Y)
static Value *foldLogOpOfMaskedICmps(ICmpInst *LHS, ICmpInst *RHS, bool IsAnd,
                                     llvm::InstCombiner::BuilderTy *Builder) {
  Value *A = nullptr, *B = nullptr, *C = nullptr, *D = nullptr, *E = nullptr;
  ICmpInst::Predicate LHSCC = LHS->getPredicate(), RHSCC = RHS->getPredicate();
  unsigned mask = foldLogOpOfMaskedICmpsHelper(A, B, C, D, E, LHS, RHS,
                                               LHSCC, RHSCC);
  if (mask == 0) return nullptr;
  assert(ICmpInst::isEquality(LHSCC) && ICmpInst::isEquality(RHSCC) &&
         "foldLogOpOfMaskedICmpsHelper must return an equality predicate.");

  // In full generality:
  //     (icmp (A & B) Op C) | (icmp (A & D) Op E)
  // ==  ![ (icmp (A & B) !Op C) & (icmp (A & D) !Op E) ]
  //
  // If the latter can be converted into (icmp (A & X) Op Y) then the former is
  // equivalent to (icmp (A & X) !Op Y).
  //
  // Therefore, we can pretend for the rest of this function that we're dealing
  // with the conjunction, provided we flip the sense of any comparisons (both
  // input and output).

  // In most cases we're going to produce an EQ for the "&&" case.
  ICmpInst::Predicate NEWCC = IsAnd ? ICmpInst::ICMP_EQ : ICmpInst::ICMP_NE;
  if (!IsAnd) {
    // Convert the masking analysis into its equivalent with negated
    // comparisons.
    mask = conjugateICmpMask(mask);
  }

  if (mask & FoldMskICmp_Mask_AllZeroes) {
    // (icmp eq (A & B), 0) & (icmp eq (A & D), 0)
    // -> (icmp eq (A & (B|D)), 0)
    Value *newOr = Builder->CreateOr(B, D);
    Value *newAnd = Builder->CreateAnd(A, newOr);
    // we can't use C as zero, because we might actually handle
    //   (icmp ne (A & B), B) & (icmp ne (A & D), D)
    // with B and D, having a single bit set
    Value *zero = Constant::getNullValue(A->getType());
    return Builder->CreateICmp(NEWCC, newAnd, zero);
  }
  if (mask & FoldMskICmp_BMask_AllOnes) {
    // (icmp eq (A & B), B) & (icmp eq (A & D), D)
    // -> (icmp eq (A & (B|D)), (B|D))
    Value *newOr = Builder->CreateOr(B, D);
    Value *newAnd = Builder->CreateAnd(A, newOr);
    return Builder->CreateICmp(NEWCC, newAnd, newOr);
  }
  if (mask & FoldMskICmp_AMask_AllOnes) {
    // (icmp eq (A & B), A) & (icmp eq (A & D), A)
    // -> (icmp eq (A & (B&D)), A)
    Value *newAnd1 = Builder->CreateAnd(B, D);
    Value *newAnd = Builder->CreateAnd(A, newAnd1);
    return Builder->CreateICmp(NEWCC, newAnd, A);
  }

  // Remaining cases assume at least that B and D are constant, and depend on
  // their actual values. This isn't strictly, necessary, just a "handle the
  // easy cases for now" decision.
  ConstantInt *BCst = dyn_cast<ConstantInt>(B);
  if (!BCst) return nullptr;
  ConstantInt *DCst = dyn_cast<ConstantInt>(D);
  if (!DCst) return nullptr;

  if (mask & (FoldMskICmp_Mask_NotAllZeroes | FoldMskICmp_BMask_NotAllOnes)) {
    // (icmp ne (A & B), 0) & (icmp ne (A & D), 0) and
    // (icmp ne (A & B), B) & (icmp ne (A & D), D)
    //     -> (icmp ne (A & B), 0) or (icmp ne (A & D), 0)
    // Only valid if one of the masks is a superset of the other (check "B&D" is
    // the same as either B or D).
    APInt NewMask = BCst->getValue() & DCst->getValue();

    if (NewMask == BCst->getValue())
      return LHS;
    else if (NewMask == DCst->getValue())
      return RHS;
  }
  if (mask & FoldMskICmp_AMask_NotAllOnes) {
    // (icmp ne (A & B), B) & (icmp ne (A & D), D)
    //     -> (icmp ne (A & B), A) or (icmp ne (A & D), A)
    // Only valid if one of the masks is a superset of the other (check "B|D" is
    // the same as either B or D).
    APInt NewMask = BCst->getValue() | DCst->getValue();

    if (NewMask == BCst->getValue())
      return LHS;
    else if (NewMask == DCst->getValue())
      return RHS;
  }
  if (mask & FoldMskICmp_BMask_Mixed) {
    // (icmp eq (A & B), C) & (icmp eq (A & D), E)
    // We already know that B & C == C && D & E == E.
    // If we can prove that (B & D) & (C ^ E) == 0, that is, the bits of
    // C and E, which are shared by both the mask B and the mask D, don't
    // contradict, then we can transform to
    // -> (icmp eq (A & (B|D)), (C|E))
    // Currently, we only handle the case of B, C, D, and E being constant.
    // we can't simply use C and E, because we might actually handle
    //   (icmp ne (A & B), B) & (icmp eq (A & D), D)
    // with B and D, having a single bit set
    ConstantInt *CCst = dyn_cast<ConstantInt>(C);
    if (!CCst) return nullptr;
    ConstantInt *ECst = dyn_cast<ConstantInt>(E);
    if (!ECst) return nullptr;
    if (LHSCC != NEWCC)
      CCst = cast<ConstantInt>(ConstantExpr::getXor(BCst, CCst));
    if (RHSCC != NEWCC)
      ECst = cast<ConstantInt>(ConstantExpr::getXor(DCst, ECst));
    // if there is a conflict we should actually return a false for the
    // whole construct
    if (((BCst->getValue() & DCst->getValue()) &
         (CCst->getValue() ^ ECst->getValue())) != 0)
      return ConstantInt::get(LHS->getType(), !IsAnd);
    Value *newOr1 = Builder->CreateOr(B, D);
    Value *newOr2 = ConstantExpr::getOr(CCst, ECst);
    Value *newAnd = Builder->CreateAnd(A, newOr1);
    return Builder->CreateICmp(NEWCC, newAnd, newOr2);
  }
  return nullptr;
}

/// Try to fold a signed range checked with lower bound 0 to an unsigned icmp.
/// Example: (icmp sge x, 0) & (icmp slt x, n) --> icmp ult x, n
/// If \p Inverted is true then the check is for the inverted range, e.g.
/// (icmp slt x, 0) | (icmp sgt x, n) --> icmp ugt x, n
Value *InstCombiner::simplifyRangeCheck(ICmpInst *Cmp0, ICmpInst *Cmp1,
                                        bool Inverted) {
  // Check the lower range comparison, e.g. x >= 0
  // InstCombine already ensured that if there is a constant it's on the RHS.
  ConstantInt *RangeStart = dyn_cast<ConstantInt>(Cmp0->getOperand(1));
  if (!RangeStart)
    return nullptr;

  ICmpInst::Predicate Pred0 = (Inverted ? Cmp0->getInversePredicate() :
                               Cmp0->getPredicate());

  // Accept x > -1 or x >= 0 (after potentially inverting the predicate).
  if (!((Pred0 == ICmpInst::ICMP_SGT && RangeStart->isMinusOne()) ||
        (Pred0 == ICmpInst::ICMP_SGE && RangeStart->isZero())))
    return nullptr;

  ICmpInst::Predicate Pred1 = (Inverted ? Cmp1->getInversePredicate() :
                               Cmp1->getPredicate());

  Value *Input = Cmp0->getOperand(0);
  Value *RangeEnd;
  if (Cmp1->getOperand(0) == Input) {
    // For the upper range compare we have: icmp x, n
    RangeEnd = Cmp1->getOperand(1);
  } else if (Cmp1->getOperand(1) == Input) {
    // For the upper range compare we have: icmp n, x
    RangeEnd = Cmp1->getOperand(0);
    Pred1 = ICmpInst::getSwappedPredicate(Pred1);
  } else {
    return nullptr;
  }

  // Check the upper range comparison, e.g. x < n
  ICmpInst::Predicate NewPred;
  switch (Pred1) {
    case ICmpInst::ICMP_SLT: NewPred = ICmpInst::ICMP_ULT; break;
    case ICmpInst::ICMP_SLE: NewPred = ICmpInst::ICMP_ULE; break;
    default: return nullptr;
  }

  // This simplification is only valid if the upper range is not negative.
  bool IsNegative, IsNotNegative;
  ComputeSignBit(RangeEnd, IsNotNegative, IsNegative, /*Depth=*/0, Cmp1);
  if (!IsNotNegative)
    return nullptr;

  if (Inverted)
    NewPred = ICmpInst::getInversePredicate(NewPred);

  return Builder->CreateICmp(NewPred, Input, RangeEnd);
}

/// FoldAndOfICmps - Fold (icmp)&(icmp) if possible.
Value *InstCombiner::FoldAndOfICmps(ICmpInst *LHS, ICmpInst *RHS) {
  ICmpInst::Predicate LHSCC = LHS->getPredicate(), RHSCC = RHS->getPredicate();

  // (icmp1 A, B) & (icmp2 A, B) --> (icmp3 A, B)
  if (PredicatesFoldable(LHSCC, RHSCC)) {
    if (LHS->getOperand(0) == RHS->getOperand(1) &&
        LHS->getOperand(1) == RHS->getOperand(0))
      LHS->swapOperands();
    if (LHS->getOperand(0) == RHS->getOperand(0) &&
        LHS->getOperand(1) == RHS->getOperand(1)) {
      Value *Op0 = LHS->getOperand(0), *Op1 = LHS->getOperand(1);
      unsigned Code = getICmpCode(LHS) & getICmpCode(RHS);
      bool isSigned = LHS->isSigned() || RHS->isSigned();
      return getNewICmpValue(isSigned, Code, Op0, Op1, Builder);
    }
  }

  // handle (roughly):  (icmp eq (A & B), C) & (icmp eq (A & D), E)
  if (Value *V = foldLogOpOfMaskedICmps(LHS, RHS, true, Builder))
    return V;

  // E.g. (icmp sge x, 0) & (icmp slt x, n) --> icmp ult x, n
  if (Value *V = simplifyRangeCheck(LHS, RHS, /*Inverted=*/false))
    return V;

  // E.g. (icmp slt x, n) & (icmp sge x, 0) --> icmp ult x, n
  if (Value *V = simplifyRangeCheck(RHS, LHS, /*Inverted=*/false))
    return V;

  // This only handles icmp of constants: (icmp1 A, C1) & (icmp2 B, C2).
  Value *Val = LHS->getOperand(0), *Val2 = RHS->getOperand(0);
  ConstantInt *LHSCst = dyn_cast<ConstantInt>(LHS->getOperand(1));
  ConstantInt *RHSCst = dyn_cast<ConstantInt>(RHS->getOperand(1));
  if (!LHSCst || !RHSCst) return nullptr;

  if (LHSCst == RHSCst && LHSCC == RHSCC) {
    // (icmp ult A, C) & (icmp ult B, C) --> (icmp ult (A|B), C)
    // where C is a power of 2
    if (LHSCC == ICmpInst::ICMP_ULT &&
        LHSCst->getValue().isPowerOf2()) {
      Value *NewOr = Builder->CreateOr(Val, Val2);
      return Builder->CreateICmp(LHSCC, NewOr, LHSCst);
    }

    // (icmp eq A, 0) & (icmp eq B, 0) --> (icmp eq (A|B), 0)
    if (LHSCC == ICmpInst::ICMP_EQ && LHSCst->isZero()) {
      Value *NewOr = Builder->CreateOr(Val, Val2);
      return Builder->CreateICmp(LHSCC, NewOr, LHSCst);
    }
  }

  // (trunc x) == C1 & (and x, CA) == C2 -> (and x, CA|CMAX) == C1|C2
  // where CMAX is the all ones value for the truncated type,
  // iff the lower bits of C2 and CA are zero.
  if (LHSCC == ICmpInst::ICMP_EQ && LHSCC == RHSCC &&
      LHS->hasOneUse() && RHS->hasOneUse()) {
    Value *V;
    ConstantInt *AndCst, *SmallCst = nullptr, *BigCst = nullptr;

    // (trunc x) == C1 & (and x, CA) == C2
    // (and x, CA) == C2 & (trunc x) == C1
    if (match(Val2, m_Trunc(m_Value(V))) &&
        match(Val, m_And(m_Specific(V), m_ConstantInt(AndCst)))) {
      SmallCst = RHSCst;
      BigCst = LHSCst;
    } else if (match(Val, m_Trunc(m_Value(V))) &&
               match(Val2, m_And(m_Specific(V), m_ConstantInt(AndCst)))) {
      SmallCst = LHSCst;
      BigCst = RHSCst;
    }

    if (SmallCst && BigCst) {
      unsigned BigBitSize = BigCst->getType()->getBitWidth();
      unsigned SmallBitSize = SmallCst->getType()->getBitWidth();

      // Check that the low bits are zero.
      APInt Low = APInt::getLowBitsSet(BigBitSize, SmallBitSize);
      if ((Low & AndCst->getValue()) == 0 && (Low & BigCst->getValue()) == 0) {
        Value *NewAnd = Builder->CreateAnd(V, Low | AndCst->getValue());
        APInt N = SmallCst->getValue().zext(BigBitSize) | BigCst->getValue();
        Value *NewVal = ConstantInt::get(AndCst->getType()->getContext(), N);
        return Builder->CreateICmp(LHSCC, NewAnd, NewVal);
      }
    }
  }

  // From here on, we only handle:
  //    (icmp1 A, C1) & (icmp2 A, C2) --> something simpler.
  if (Val != Val2) return nullptr;

  // ICMP_[US][GL]E X, CST is folded to ICMP_[US][GL]T elsewhere.
  if (LHSCC == ICmpInst::ICMP_UGE || LHSCC == ICmpInst::ICMP_ULE ||
      RHSCC == ICmpInst::ICMP_UGE || RHSCC == ICmpInst::ICMP_ULE ||
      LHSCC == ICmpInst::ICMP_SGE || LHSCC == ICmpInst::ICMP_SLE ||
      RHSCC == ICmpInst::ICMP_SGE || RHSCC == ICmpInst::ICMP_SLE)
    return nullptr;

  // Make a constant range that's the intersection of the two icmp ranges.
  // If the intersection is empty, we know that the result is false.
  ConstantRange LHSRange =
      ConstantRange::makeAllowedICmpRegion(LHSCC, LHSCst->getValue());
  ConstantRange RHSRange =
      ConstantRange::makeAllowedICmpRegion(RHSCC, RHSCst->getValue());

  if (LHSRange.intersectWith(RHSRange).isEmptySet())
    return ConstantInt::get(CmpInst::makeCmpResultType(LHS->getType()), 0);

  // We can't fold (ugt x, C) & (sgt x, C2).
  if (!PredicatesFoldable(LHSCC, RHSCC))
    return nullptr;

  // Ensure that the larger constant is on the RHS.
  bool ShouldSwap;
  if (CmpInst::isSigned(LHSCC) ||
      (ICmpInst::isEquality(LHSCC) &&
       CmpInst::isSigned(RHSCC)))
    ShouldSwap = LHSCst->getValue().sgt(RHSCst->getValue());
  else
    ShouldSwap = LHSCst->getValue().ugt(RHSCst->getValue());

  if (ShouldSwap) {
    std::swap(LHS, RHS);
    std::swap(LHSCst, RHSCst);
    std::swap(LHSCC, RHSCC);
  }

  // At this point, we know we have two icmp instructions
  // comparing a value against two constants and and'ing the result
  // together.  Because of the above check, we know that we only have
  // icmp eq, icmp ne, icmp [su]lt, and icmp [SU]gt here. We also know
  // (from the icmp folding check above), that the two constants
  // are not equal and that the larger constant is on the RHS
  assert(LHSCst != RHSCst && "Compares not folded above?");

  switch (LHSCC) {
  default: llvm_unreachable("Unknown integer condition code!");
  case ICmpInst::ICMP_EQ:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_NE:         // (X == 13 & X != 15) -> X == 13
    case ICmpInst::ICMP_ULT:        // (X == 13 & X <  15) -> X == 13
    case ICmpInst::ICMP_SLT:        // (X == 13 & X <  15) -> X == 13
      return LHS;
    }
  case ICmpInst::ICMP_NE:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_ULT:
      if (LHSCst == SubOne(RHSCst)) // (X != 13 & X u< 14) -> X < 13
        return Builder->CreateICmpULT(Val, LHSCst);
      if (LHSCst->isNullValue())    // (X !=  0 & X u< 14) -> X-1 u< 13
        return InsertRangeTest(Val, AddOne(LHSCst), RHSCst, false, true);
      break;                        // (X != 13 & X u< 15) -> no change
    case ICmpInst::ICMP_SLT:
      if (LHSCst == SubOne(RHSCst)) // (X != 13 & X s< 14) -> X < 13
        return Builder->CreateICmpSLT(Val, LHSCst);
      break;                        // (X != 13 & X s< 15) -> no change
    case ICmpInst::ICMP_EQ:         // (X != 13 & X == 15) -> X == 15
    case ICmpInst::ICMP_UGT:        // (X != 13 & X u> 15) -> X u> 15
    case ICmpInst::ICMP_SGT:        // (X != 13 & X s> 15) -> X s> 15
      return RHS;
    case ICmpInst::ICMP_NE:
      // Special case to get the ordering right when the values wrap around
      // zero.
      if (LHSCst->getValue() == 0 && RHSCst->getValue().isAllOnesValue())
        std::swap(LHSCst, RHSCst);
      if (LHSCst == SubOne(RHSCst)){// (X != 13 & X != 14) -> X-13 >u 1
        Constant *AddCST = ConstantExpr::getNeg(LHSCst);
        Value *Add = Builder->CreateAdd(Val, AddCST, Val->getName()+".off");
        return Builder->CreateICmpUGT(Add, ConstantInt::get(Add->getType(), 1),
                                      Val->getName()+".cmp");
      }
      break;                        // (X != 13 & X != 15) -> no change
    }
    break;
  case ICmpInst::ICMP_ULT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X u< 13 & X == 15) -> false
    case ICmpInst::ICMP_UGT:        // (X u< 13 & X u> 15) -> false
      return ConstantInt::get(CmpInst::makeCmpResultType(LHS->getType()), 0);
    case ICmpInst::ICMP_SGT:        // (X u< 13 & X s> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X u< 13 & X != 15) -> X u< 13
    case ICmpInst::ICMP_ULT:        // (X u< 13 & X u< 15) -> X u< 13
      return LHS;
    case ICmpInst::ICMP_SLT:        // (X u< 13 & X s< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_SLT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_UGT:        // (X s< 13 & X u> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X s< 13 & X != 15) -> X < 13
    case ICmpInst::ICMP_SLT:        // (X s< 13 & X s< 15) -> X < 13
      return LHS;
    case ICmpInst::ICMP_ULT:        // (X s< 13 & X u< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_UGT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X u> 13 & X == 15) -> X == 15
    case ICmpInst::ICMP_UGT:        // (X u> 13 & X u> 15) -> X u> 15
      return RHS;
    case ICmpInst::ICMP_SGT:        // (X u> 13 & X s> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:
      if (RHSCst == AddOne(LHSCst)) // (X u> 13 & X != 14) -> X u> 14
        return Builder->CreateICmp(LHSCC, Val, RHSCst);
      break;                        // (X u> 13 & X != 15) -> no change
    case ICmpInst::ICMP_ULT:        // (X u> 13 & X u< 15) -> (X-14) <u 1
      return InsertRangeTest(Val, AddOne(LHSCst), RHSCst, false, true);
    case ICmpInst::ICMP_SLT:        // (X u> 13 & X s< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_SGT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X s> 13 & X == 15) -> X == 15
    case ICmpInst::ICMP_SGT:        // (X s> 13 & X s> 15) -> X s> 15
      return RHS;
    case ICmpInst::ICMP_UGT:        // (X s> 13 & X u> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:
      if (RHSCst == AddOne(LHSCst)) // (X s> 13 & X != 14) -> X s> 14
        return Builder->CreateICmp(LHSCC, Val, RHSCst);
      break;                        // (X s> 13 & X != 15) -> no change
    case ICmpInst::ICMP_SLT:        // (X s> 13 & X s< 15) -> (X-14) s< 1
      return InsertRangeTest(Val, AddOne(LHSCst), RHSCst, true, true);
    case ICmpInst::ICMP_ULT:        // (X s> 13 & X u< 15) -> no change
      break;
    }
    break;
  }

  return nullptr;
}

/// FoldAndOfFCmps - Optimize (fcmp)&(fcmp).  NOTE: Unlike the rest of
/// instcombine, this returns a Value which should already be inserted into the
/// function.
Value *InstCombiner::FoldAndOfFCmps(FCmpInst *LHS, FCmpInst *RHS) {
  if (LHS->getPredicate() == FCmpInst::FCMP_ORD &&
      RHS->getPredicate() == FCmpInst::FCMP_ORD) {
    if (LHS->getOperand(0)->getType() != RHS->getOperand(0)->getType())
      return nullptr;

    // (fcmp ord x, c) & (fcmp ord y, c)  -> (fcmp ord x, y)
    if (ConstantFP *LHSC = dyn_cast<ConstantFP>(LHS->getOperand(1)))
      if (ConstantFP *RHSC = dyn_cast<ConstantFP>(RHS->getOperand(1))) {
        // If either of the constants are nans, then the whole thing returns
        // false.
        if (LHSC->getValueAPF().isNaN() || RHSC->getValueAPF().isNaN())
          return Builder->getFalse();
        return Builder->CreateFCmpORD(LHS->getOperand(0), RHS->getOperand(0));
      }

    // Handle vector zeros.  This occurs because the canonical form of
    // "fcmp ord x,x" is "fcmp ord x, 0".
    if (isa<ConstantAggregateZero>(LHS->getOperand(1)) &&
        isa<ConstantAggregateZero>(RHS->getOperand(1)))
      return Builder->CreateFCmpORD(LHS->getOperand(0), RHS->getOperand(0));
    return nullptr;
  }

  Value *Op0LHS = LHS->getOperand(0), *Op0RHS = LHS->getOperand(1);
  Value *Op1LHS = RHS->getOperand(0), *Op1RHS = RHS->getOperand(1);
  FCmpInst::Predicate Op0CC = LHS->getPredicate(), Op1CC = RHS->getPredicate();


  if (Op0LHS == Op1RHS && Op0RHS == Op1LHS) {
    // Swap RHS operands to match LHS.
    Op1CC = FCmpInst::getSwappedPredicate(Op1CC);
    std::swap(Op1LHS, Op1RHS);
  }

  if (Op0LHS == Op1LHS && Op0RHS == Op1RHS) {
    // Simplify (fcmp cc0 x, y) & (fcmp cc1 x, y).
    if (Op0CC == Op1CC)
      return Builder->CreateFCmp((FCmpInst::Predicate)Op0CC, Op0LHS, Op0RHS);
    if (Op0CC == FCmpInst::FCMP_FALSE || Op1CC == FCmpInst::FCMP_FALSE)
      return ConstantInt::get(CmpInst::makeCmpResultType(LHS->getType()), 0);
    if (Op0CC == FCmpInst::FCMP_TRUE)
      return RHS;
    if (Op1CC == FCmpInst::FCMP_TRUE)
      return LHS;

    bool Op0Ordered;
    bool Op1Ordered;
    unsigned Op0Pred = getFCmpCode(Op0CC, Op0Ordered);
    unsigned Op1Pred = getFCmpCode(Op1CC, Op1Ordered);
    // uno && ord -> false
    if (Op0Pred == 0 && Op1Pred == 0 && Op0Ordered != Op1Ordered)
        return ConstantInt::get(CmpInst::makeCmpResultType(LHS->getType()), 0);
    if (Op1Pred == 0) {
      std::swap(LHS, RHS);
      std::swap(Op0Pred, Op1Pred);
      std::swap(Op0Ordered, Op1Ordered);
    }
    if (Op0Pred == 0) {
      // uno && ueq -> uno && (uno || eq) -> uno
      // ord && olt -> ord && (ord && lt) -> olt
      if (!Op0Ordered && (Op0Ordered == Op1Ordered))
        return LHS;
      if (Op0Ordered && (Op0Ordered == Op1Ordered))
        return RHS;

      // uno && oeq -> uno && (ord && eq) -> false
      if (!Op0Ordered)
        return ConstantInt::get(CmpInst::makeCmpResultType(LHS->getType()), 0);
      // ord && ueq -> ord && (uno || eq) -> oeq
      return getFCmpValue(true, Op1Pred, Op0LHS, Op0RHS, Builder);
    }
  }

  return nullptr;
}

Instruction *InstCombiner::visitAnd(BinaryOperator &I) {
  bool Changed = SimplifyAssociativeOrCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyVectorOp(I))
    return ReplaceInstUsesWith(I, V);

  if (Value *V = SimplifyAndInst(Op0, Op1, DL, TLI, DT, AC))
    return ReplaceInstUsesWith(I, V);

  // (A|B)&(A|C) -> A|(B&C) etc
  if (Value *V = SimplifyUsingDistributiveLaws(I))
    return ReplaceInstUsesWith(I, V);

  // See if we can simplify any instructions used by the instruction whose sole
  // purpose is to compute bits we don't care about.
  if (SimplifyDemandedInstructionBits(I))
    return &I;

  if (Value *V = SimplifyBSwap(I))
    return ReplaceInstUsesWith(I, V);

  if (ConstantInt *AndRHS = dyn_cast<ConstantInt>(Op1)) {
    const APInt &AndRHSMask = AndRHS->getValue();

    // Optimize a variety of ((val OP C1) & C2) combinations...
    if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0)) {
      Value *Op0LHS = Op0I->getOperand(0);
      Value *Op0RHS = Op0I->getOperand(1);
      switch (Op0I->getOpcode()) {
      default: break;
      case Instruction::Xor:
      case Instruction::Or: {
        // If the mask is only needed on one incoming arm, push it up.
        if (!Op0I->hasOneUse()) break;

        APInt NotAndRHS(~AndRHSMask);
        if (MaskedValueIsZero(Op0LHS, NotAndRHS, 0, &I)) {
          // Not masking anything out for the LHS, move to RHS.
          Value *NewRHS = Builder->CreateAnd(Op0RHS, AndRHS,
                                             Op0RHS->getName()+".masked");
          return BinaryOperator::Create(Op0I->getOpcode(), Op0LHS, NewRHS);
        }
        if (!isa<Constant>(Op0RHS) &&
            MaskedValueIsZero(Op0RHS, NotAndRHS, 0, &I)) {
          // Not masking anything out for the RHS, move to LHS.
          Value *NewLHS = Builder->CreateAnd(Op0LHS, AndRHS,
                                             Op0LHS->getName()+".masked");
          return BinaryOperator::Create(Op0I->getOpcode(), NewLHS, Op0RHS);
        }

        break;
      }
      case Instruction::Add:
        // ((A & N) + B) & AndRHS -> (A + B) & AndRHS iff N&AndRHS == AndRHS.
        // ((A | N) + B) & AndRHS -> (A + B) & AndRHS iff N&AndRHS == 0
        // ((A ^ N) + B) & AndRHS -> (A + B) & AndRHS iff N&AndRHS == 0
        if (Value *V = FoldLogicalPlusAnd(Op0LHS, Op0RHS, AndRHS, false, I))
          return BinaryOperator::CreateAnd(V, AndRHS);
        if (Value *V = FoldLogicalPlusAnd(Op0RHS, Op0LHS, AndRHS, false, I))
          return BinaryOperator::CreateAnd(V, AndRHS);  // Add commutes
        break;

      case Instruction::Sub:
        // ((A & N) - B) & AndRHS -> (A - B) & AndRHS iff N&AndRHS == AndRHS.
        // ((A | N) - B) & AndRHS -> (A - B) & AndRHS iff N&AndRHS == 0
        // ((A ^ N) - B) & AndRHS -> (A - B) & AndRHS iff N&AndRHS == 0
        if (Value *V = FoldLogicalPlusAnd(Op0LHS, Op0RHS, AndRHS, true, I))
          return BinaryOperator::CreateAnd(V, AndRHS);

        // (A - N) & AndRHS -> -N & AndRHS iff A&AndRHS==0 and AndRHS
        // has 1's for all bits that the subtraction with A might affect.
        if (Op0I->hasOneUse() && !match(Op0LHS, m_Zero())) {
          uint32_t BitWidth = AndRHSMask.getBitWidth();
          uint32_t Zeros = AndRHSMask.countLeadingZeros();
          APInt Mask = APInt::getLowBitsSet(BitWidth, BitWidth - Zeros);

          if (MaskedValueIsZero(Op0LHS, Mask, 0, &I)) {
            Value *NewNeg = Builder->CreateNeg(Op0RHS);
            return BinaryOperator::CreateAnd(NewNeg, AndRHS);
          }
        }
        break;

      case Instruction::Shl:
      case Instruction::LShr:
        // (1 << x) & 1 --> zext(x == 0)
        // (1 >> x) & 1 --> zext(x == 0)
        if (AndRHSMask == 1 && Op0LHS == AndRHS) {
          Value *NewICmp =
            Builder->CreateICmpEQ(Op0RHS, Constant::getNullValue(I.getType()));
          return new ZExtInst(NewICmp, I.getType());
        }
        break;
      }

      if (ConstantInt *Op0CI = dyn_cast<ConstantInt>(Op0I->getOperand(1)))
        if (Instruction *Res = OptAndOp(Op0I, Op0CI, AndRHS, I))
          return Res;
    }

    // If this is an integer truncation, and if the source is an 'and' with
    // immediate, transform it.  This frequently occurs for bitfield accesses.
    {
      Value *X = nullptr; ConstantInt *YC = nullptr;
      if (match(Op0, m_Trunc(m_And(m_Value(X), m_ConstantInt(YC))))) {
        // Change: and (trunc (and X, YC) to T), C2
        // into  : and (trunc X to T), trunc(YC) & C2
        // This will fold the two constants together, which may allow
        // other simplifications.
        Value *NewCast = Builder->CreateTrunc(X, I.getType(), "and.shrunk");
        Constant *C3 = ConstantExpr::getTrunc(YC, I.getType());
        C3 = ConstantExpr::getAnd(C3, AndRHS);
        return BinaryOperator::CreateAnd(NewCast, C3);
      }
    }

    // Try to fold constant and into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;
    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }


  // (~A & ~B) == (~(A | B)) - De Morgan's Law
  if (Value *Op0NotVal = dyn_castNotVal(Op0))
    if (Value *Op1NotVal = dyn_castNotVal(Op1))
      if (Op0->hasOneUse() && Op1->hasOneUse()) {
        Value *Or = Builder->CreateOr(Op0NotVal, Op1NotVal,
                                      I.getName()+".demorgan");
        return BinaryOperator::CreateNot(Or);
      }

  {
    Value *A = nullptr, *B = nullptr, *C = nullptr, *D = nullptr;
    // (A|B) & ~(A&B) -> A^B
    if (match(Op0, m_Or(m_Value(A), m_Value(B))) &&
        match(Op1, m_Not(m_And(m_Value(C), m_Value(D)))) &&
        ((A == C && B == D) || (A == D && B == C)))
      return BinaryOperator::CreateXor(A, B);

    // ~(A&B) & (A|B) -> A^B
    if (match(Op1, m_Or(m_Value(A), m_Value(B))) &&
        match(Op0, m_Not(m_And(m_Value(C), m_Value(D)))) &&
        ((A == C && B == D) || (A == D && B == C)))
      return BinaryOperator::CreateXor(A, B);

    // A&(A^B) => A & ~B
    {
      Value *tmpOp0 = Op0;
      Value *tmpOp1 = Op1;
      if (Op0->hasOneUse() &&
          match(Op0, m_Xor(m_Value(A), m_Value(B)))) {
        if (A == Op1 || B == Op1 ) {
          tmpOp1 = Op0;
          tmpOp0 = Op1;
          // Simplify below
        }
      }

      if (tmpOp1->hasOneUse() &&
          match(tmpOp1, m_Xor(m_Value(A), m_Value(B)))) {
        if (B == tmpOp0) {
          std::swap(A, B);
        }
        // Notice that the patten (A&(~B)) is actually (A&(-1^B)), so if
        // A is originally -1 (or a vector of -1 and undefs), then we enter
        // an endless loop. By checking that A is non-constant we ensure that
        // we will never get to the loop.
        if (A == tmpOp0 && !isa<Constant>(A)) // A&(A^B) -> A & ~B
          return BinaryOperator::CreateAnd(A, Builder->CreateNot(B));
      }
    }

    // (A&((~A)|B)) -> A&B
    if (match(Op0, m_Or(m_Not(m_Specific(Op1)), m_Value(A))) ||
        match(Op0, m_Or(m_Value(A), m_Not(m_Specific(Op1)))))
      return BinaryOperator::CreateAnd(A, Op1);
    if (match(Op1, m_Or(m_Not(m_Specific(Op0)), m_Value(A))) ||
        match(Op1, m_Or(m_Value(A), m_Not(m_Specific(Op0)))))
      return BinaryOperator::CreateAnd(A, Op0);

    // (A ^ B) & ((B ^ C) ^ A) -> (A ^ B) & ~C
    if (match(Op0, m_Xor(m_Value(A), m_Value(B))))
      if (match(Op1, m_Xor(m_Xor(m_Specific(B), m_Value(C)), m_Specific(A))))
        if (Op1->hasOneUse() || cast<BinaryOperator>(Op1)->hasOneUse())
          return BinaryOperator::CreateAnd(Op0, Builder->CreateNot(C));

    // ((A ^ C) ^ B) & (B ^ A) -> (B ^ A) & ~C
    if (match(Op0, m_Xor(m_Xor(m_Value(A), m_Value(C)), m_Value(B))))
      if (match(Op1, m_Xor(m_Specific(B), m_Specific(A))))
        if (Op0->hasOneUse() || cast<BinaryOperator>(Op0)->hasOneUse())
          return BinaryOperator::CreateAnd(Op1, Builder->CreateNot(C));

    // (A | B) & ((~A) ^ B) -> (A & B)
    if (match(Op0, m_Or(m_Value(A), m_Value(B))) &&
        match(Op1, m_Xor(m_Not(m_Specific(A)), m_Specific(B))))
      return BinaryOperator::CreateAnd(A, B);

    // ((~A) ^ B) & (A | B) -> (A & B)
    if (match(Op0, m_Xor(m_Not(m_Value(A)), m_Value(B))) &&
        match(Op1, m_Or(m_Specific(A), m_Specific(B))))
      return BinaryOperator::CreateAnd(A, B);
  }

  {
    ICmpInst *LHS = dyn_cast<ICmpInst>(Op0);
    ICmpInst *RHS = dyn_cast<ICmpInst>(Op1);
    if (LHS && RHS)
      if (Value *Res = FoldAndOfICmps(LHS, RHS))
        return ReplaceInstUsesWith(I, Res);

    // TODO: Make this recursive; it's a little tricky because an arbitrary
    // number of 'and' instructions might have to be created.
    Value *X, *Y;
    if (LHS && match(Op1, m_OneUse(m_And(m_Value(X), m_Value(Y))))) {
      if (auto *Cmp = dyn_cast<ICmpInst>(X))
        if (Value *Res = FoldAndOfICmps(LHS, Cmp))
          return ReplaceInstUsesWith(I, Builder->CreateAnd(Res, Y));
      if (auto *Cmp = dyn_cast<ICmpInst>(Y))
        if (Value *Res = FoldAndOfICmps(LHS, Cmp))
          return ReplaceInstUsesWith(I, Builder->CreateAnd(Res, X));
    }
    if (RHS && match(Op0, m_OneUse(m_And(m_Value(X), m_Value(Y))))) {
      if (auto *Cmp = dyn_cast<ICmpInst>(X))
        if (Value *Res = FoldAndOfICmps(Cmp, RHS))
          return ReplaceInstUsesWith(I, Builder->CreateAnd(Res, Y));
      if (auto *Cmp = dyn_cast<ICmpInst>(Y))
        if (Value *Res = FoldAndOfICmps(Cmp, RHS))
          return ReplaceInstUsesWith(I, Builder->CreateAnd(Res, X));
    }
  }

  // If and'ing two fcmp, try combine them into one.
  if (FCmpInst *LHS = dyn_cast<FCmpInst>(I.getOperand(0)))
    if (FCmpInst *RHS = dyn_cast<FCmpInst>(I.getOperand(1)))
      if (Value *Res = FoldAndOfFCmps(LHS, RHS))
        return ReplaceInstUsesWith(I, Res);


  // fold (and (cast A), (cast B)) -> (cast (and A, B))
  if (CastInst *Op0C = dyn_cast<CastInst>(Op0))
    if (CastInst *Op1C = dyn_cast<CastInst>(Op1)) {
      Type *SrcTy = Op0C->getOperand(0)->getType();
      if (Op0C->getOpcode() == Op1C->getOpcode() && // same cast kind ?
          SrcTy == Op1C->getOperand(0)->getType() &&
          SrcTy->isIntOrIntVectorTy()) {
        Value *Op0COp = Op0C->getOperand(0), *Op1COp = Op1C->getOperand(0);

        // Only do this if the casts both really cause code to be generated.
        if (ShouldOptimizeCast(Op0C->getOpcode(), Op0COp, I.getType()) &&
            ShouldOptimizeCast(Op1C->getOpcode(), Op1COp, I.getType())) {
          Value *NewOp = Builder->CreateAnd(Op0COp, Op1COp, I.getName());
          return CastInst::Create(Op0C->getOpcode(), NewOp, I.getType());
        }

        // If this is and(cast(icmp), cast(icmp)), try to fold this even if the
        // cast is otherwise not optimizable.  This happens for vector sexts.
        if (ICmpInst *RHS = dyn_cast<ICmpInst>(Op1COp))
          if (ICmpInst *LHS = dyn_cast<ICmpInst>(Op0COp))
            if (Value *Res = FoldAndOfICmps(LHS, RHS))
              return CastInst::Create(Op0C->getOpcode(), Res, I.getType());

        // If this is and(cast(fcmp), cast(fcmp)), try to fold this even if the
        // cast is otherwise not optimizable.  This happens for vector sexts.
        if (FCmpInst *RHS = dyn_cast<FCmpInst>(Op1COp))
          if (FCmpInst *LHS = dyn_cast<FCmpInst>(Op0COp))
            if (Value *Res = FoldAndOfFCmps(LHS, RHS))
              return CastInst::Create(Op0C->getOpcode(), Res, I.getType());
      }
    }

  {
    Value *X = nullptr;
    bool OpsSwapped = false;
    // Canonicalize SExt or Not to the LHS
    if (match(Op1, m_SExt(m_Value())) ||
        match(Op1, m_Not(m_Value()))) {
      std::swap(Op0, Op1);
      OpsSwapped = true;
    }

    // Fold (and (sext bool to A), B) --> (select bool, B, 0)
    if (match(Op0, m_SExt(m_Value(X))) &&
        X->getType()->getScalarType()->isIntegerTy(1)) {
      Value *Zero = Constant::getNullValue(Op1->getType());
      return SelectInst::Create(X, Op1, Zero);
    }

    // Fold (and ~(sext bool to A), B) --> (select bool, 0, B)
    if (match(Op0, m_Not(m_SExt(m_Value(X)))) &&
        X->getType()->getScalarType()->isIntegerTy(1)) {
      Value *Zero = Constant::getNullValue(Op0->getType());
      return SelectInst::Create(X, Zero, Op1);
    }

    if (OpsSwapped)
      std::swap(Op0, Op1);
  }

  return Changed ? &I : nullptr;
}

/// CollectBSwapParts - Analyze the specified subexpression and see if it is
/// capable of providing pieces of a bswap.  The subexpression provides pieces
/// of a bswap if it is proven that each of the non-zero bytes in the output of
/// the expression came from the corresponding "byte swapped" byte in some other
/// value.  For example, if the current subexpression is "(shl i32 %X, 24)" then
/// we know that the expression deposits the low byte of %X into the high byte
/// of the bswap result and that all other bytes are zero.  This expression is
/// accepted, the high byte of ByteValues is set to X to indicate a correct
/// match.
///
/// This function returns true if the match was unsuccessful and false if so.
/// On entry to the function the "OverallLeftShift" is a signed integer value
/// indicating the number of bytes that the subexpression is later shifted.  For
/// example, if the expression is later right shifted by 16 bits, the
/// OverallLeftShift value would be -2 on entry.  This is used to specify which
/// byte of ByteValues is actually being set.
///
/// Similarly, ByteMask is a bitmask where a bit is clear if its corresponding
/// byte is masked to zero by a user.  For example, in (X & 255), X will be
/// processed with a bytemask of 1.  Because bytemask is 32-bits, this limits
/// this function to working on up to 32-byte (256 bit) values.  ByteMask is
/// always in the local (OverallLeftShift) coordinate space.
///
static bool CollectBSwapParts(Value *V, int OverallLeftShift, uint32_t ByteMask,
                              SmallVectorImpl<Value *> &ByteValues) {
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    // If this is an or instruction, it may be an inner node of the bswap.
    if (I->getOpcode() == Instruction::Or) {
      return CollectBSwapParts(I->getOperand(0), OverallLeftShift, ByteMask,
                               ByteValues) ||
             CollectBSwapParts(I->getOperand(1), OverallLeftShift, ByteMask,
                               ByteValues);
    }

    // If this is a logical shift by a constant multiple of 8, recurse with
    // OverallLeftShift and ByteMask adjusted.
    if (I->isLogicalShift() && isa<ConstantInt>(I->getOperand(1))) {
      unsigned ShAmt =
        cast<ConstantInt>(I->getOperand(1))->getLimitedValue(~0U);
      // Ensure the shift amount is defined and of a byte value.
      if ((ShAmt & 7) || (ShAmt > 8*ByteValues.size()))
        return true;

      unsigned ByteShift = ShAmt >> 3;
      if (I->getOpcode() == Instruction::Shl) {
        // X << 2 -> collect(X, +2)
        OverallLeftShift += ByteShift;
        ByteMask >>= ByteShift;
      } else {
        // X >>u 2 -> collect(X, -2)
        OverallLeftShift -= ByteShift;
        ByteMask <<= ByteShift;
        ByteMask &= (~0U >> (32-ByteValues.size()));
      }

      if (OverallLeftShift >= (int)ByteValues.size()) return true;
      if (OverallLeftShift <= -(int)ByteValues.size()) return true;

      return CollectBSwapParts(I->getOperand(0), OverallLeftShift, ByteMask,
                               ByteValues);
    }

    // If this is a logical 'and' with a mask that clears bytes, clear the
    // corresponding bytes in ByteMask.
    if (I->getOpcode() == Instruction::And &&
        isa<ConstantInt>(I->getOperand(1))) {
      // Scan every byte of the and mask, seeing if the byte is either 0 or 255.
      unsigned NumBytes = ByteValues.size();
      APInt Byte(I->getType()->getPrimitiveSizeInBits(), 255);
      const APInt &AndMask = cast<ConstantInt>(I->getOperand(1))->getValue();

      for (unsigned i = 0; i != NumBytes; ++i, Byte <<= 8) {
        // If this byte is masked out by a later operation, we don't care what
        // the and mask is.
        if ((ByteMask & (1 << i)) == 0)
          continue;

        // If the AndMask is all zeros for this byte, clear the bit.
        APInt MaskB = AndMask & Byte;
        if (MaskB == 0) {
          ByteMask &= ~(1U << i);
          continue;
        }

        // If the AndMask is not all ones for this byte, it's not a bytezap.
        if (MaskB != Byte)
          return true;

        // Otherwise, this byte is kept.
      }

      return CollectBSwapParts(I->getOperand(0), OverallLeftShift, ByteMask,
                               ByteValues);
    }
  }

  // Okay, we got to something that isn't a shift, 'or' or 'and'.  This must be
  // the input value to the bswap.  Some observations: 1) if more than one byte
  // is demanded from this input, then it could not be successfully assembled
  // into a byteswap.  At least one of the two bytes would not be aligned with
  // their ultimate destination.
  if (!isPowerOf2_32(ByteMask)) return true;
  unsigned InputByteNo = countTrailingZeros(ByteMask);

  // 2) The input and ultimate destinations must line up: if byte 3 of an i32
  // is demanded, it needs to go into byte 0 of the result.  This means that the
  // byte needs to be shifted until it lands in the right byte bucket.  The
  // shift amount depends on the position: if the byte is coming from the high
  // part of the value (e.g. byte 3) then it must be shifted right.  If from the
  // low part, it must be shifted left.
  unsigned DestByteNo = InputByteNo + OverallLeftShift;
  if (ByteValues.size()-1-DestByteNo != InputByteNo)
    return true;

  // If the destination byte value is already defined, the values are or'd
  // together, which isn't a bswap (unless it's an or of the same bits).
  if (ByteValues[DestByteNo] && ByteValues[DestByteNo] != V)
    return true;
  ByteValues[DestByteNo] = V;
  return false;
}

/// MatchBSwap - Given an OR instruction, check to see if this is a bswap idiom.
/// If so, insert the new bswap intrinsic and return it.
Instruction *InstCombiner::MatchBSwap(BinaryOperator &I) {
  IntegerType *ITy = dyn_cast<IntegerType>(I.getType());
  if (!ITy || ITy->getBitWidth() % 16 ||
      // ByteMask only allows up to 32-byte values.
      ITy->getBitWidth() > 32*8)
    return nullptr;   // Can only bswap pairs of bytes.  Can't do vectors.

  /// ByteValues - For each byte of the result, we keep track of which value
  /// defines each byte.
  SmallVector<Value*, 8> ByteValues;
  ByteValues.resize(ITy->getBitWidth()/8);

  // Try to find all the pieces corresponding to the bswap.
  uint32_t ByteMask = ~0U >> (32-ByteValues.size());
  if (CollectBSwapParts(&I, 0, ByteMask, ByteValues))
    return nullptr;

  // Check to see if all of the bytes come from the same value.
  Value *V = ByteValues[0];
  if (!V) return nullptr;  // Didn't find a byte?  Must be zero.

  // Check to make sure that all of the bytes come from the same value.
  for (unsigned i = 1, e = ByteValues.size(); i != e; ++i)
    if (ByteValues[i] != V)
      return nullptr;
  Module *M = I.getParent()->getParent()->getParent();
  Function *F = Intrinsic::getDeclaration(M, Intrinsic::bswap, ITy);
  return CallInst::Create(F, V);
}

/// MatchSelectFromAndOr - We have an expression of the form (A&C)|(B&D).  Check
/// If A is (cond?-1:0) and either B or D is ~(cond?-1,0) or (cond?0,-1), then
/// we can simplify this expression to "cond ? C : D or B".
static Instruction *MatchSelectFromAndOr(Value *A, Value *B,
                                         Value *C, Value *D) {
  // If A is not a select of -1/0, this cannot match.
  Value *Cond = nullptr;
  if (!match(A, m_SExt(m_Value(Cond))) ||
      !Cond->getType()->isIntegerTy(1))
    return nullptr;

  // ((cond?-1:0)&C) | (B&(cond?0:-1)) -> cond ? C : B.
  if (match(D, m_Not(m_SExt(m_Specific(Cond)))))
    return SelectInst::Create(Cond, C, B);
  if (match(D, m_SExt(m_Not(m_Specific(Cond)))))
    return SelectInst::Create(Cond, C, B);

  // ((cond?-1:0)&C) | ((cond?0:-1)&D) -> cond ? C : D.
  if (match(B, m_Not(m_SExt(m_Specific(Cond)))))
    return SelectInst::Create(Cond, C, D);
  if (match(B, m_SExt(m_Not(m_Specific(Cond)))))
    return SelectInst::Create(Cond, C, D);
  return nullptr;
}

/// FoldOrOfICmps - Fold (icmp)|(icmp) if possible.
Value *InstCombiner::FoldOrOfICmps(ICmpInst *LHS, ICmpInst *RHS,
                                   Instruction *CxtI) {
  ICmpInst::Predicate LHSCC = LHS->getPredicate(), RHSCC = RHS->getPredicate();

  // Fold (iszero(A & K1) | iszero(A & K2)) ->  (A & (K1 | K2)) != (K1 | K2)
  // if K1 and K2 are a one-bit mask.
  ConstantInt *LHSCst = dyn_cast<ConstantInt>(LHS->getOperand(1));
  ConstantInt *RHSCst = dyn_cast<ConstantInt>(RHS->getOperand(1));

  if (LHS->getPredicate() == ICmpInst::ICMP_EQ && LHSCst && LHSCst->isZero() &&
      RHS->getPredicate() == ICmpInst::ICMP_EQ && RHSCst && RHSCst->isZero()) {

    BinaryOperator *LAnd = dyn_cast<BinaryOperator>(LHS->getOperand(0));
    BinaryOperator *RAnd = dyn_cast<BinaryOperator>(RHS->getOperand(0));
    if (LAnd && RAnd && LAnd->hasOneUse() && RHS->hasOneUse() &&
        LAnd->getOpcode() == Instruction::And &&
        RAnd->getOpcode() == Instruction::And) {

      Value *Mask = nullptr;
      Value *Masked = nullptr;
      if (LAnd->getOperand(0) == RAnd->getOperand(0) &&
          isKnownToBeAPowerOfTwo(LAnd->getOperand(1), DL, false, 0, AC, CxtI,
                                 DT) &&
          isKnownToBeAPowerOfTwo(RAnd->getOperand(1), DL, false, 0, AC, CxtI,
                                 DT)) {
        Mask = Builder->CreateOr(LAnd->getOperand(1), RAnd->getOperand(1));
        Masked = Builder->CreateAnd(LAnd->getOperand(0), Mask);
      } else if (LAnd->getOperand(1) == RAnd->getOperand(1) &&
                 isKnownToBeAPowerOfTwo(LAnd->getOperand(0), DL, false, 0, AC,
                                        CxtI, DT) &&
                 isKnownToBeAPowerOfTwo(RAnd->getOperand(0), DL, false, 0, AC,
                                        CxtI, DT)) {
        Mask = Builder->CreateOr(LAnd->getOperand(0), RAnd->getOperand(0));
        Masked = Builder->CreateAnd(LAnd->getOperand(1), Mask);
      }

      if (Masked)
        return Builder->CreateICmp(ICmpInst::ICMP_NE, Masked, Mask);
    }
  }

  // Fold (icmp ult/ule (A + C1), C3) | (icmp ult/ule (A + C2), C3)
  //                   -->  (icmp ult/ule ((A & ~(C1 ^ C2)) + max(C1, C2)), C3)
  // The original condition actually refers to the following two ranges:
  // [MAX_UINT-C1+1, MAX_UINT-C1+1+C3] and [MAX_UINT-C2+1, MAX_UINT-C2+1+C3]
  // We can fold these two ranges if:
  // 1) C1 and C2 is unsigned greater than C3.
  // 2) The two ranges are separated.
  // 3) C1 ^ C2 is one-bit mask.
  // 4) LowRange1 ^ LowRange2 and HighRange1 ^ HighRange2 are one-bit mask.
  // This implies all values in the two ranges differ by exactly one bit.

  if ((LHSCC == ICmpInst::ICMP_ULT || LHSCC == ICmpInst::ICMP_ULE) &&
      LHSCC == RHSCC && LHSCst && RHSCst && LHS->hasOneUse() &&
      RHS->hasOneUse() && LHSCst->getType() == RHSCst->getType() &&
      LHSCst->getValue() == (RHSCst->getValue())) {

    Value *LAdd = LHS->getOperand(0);
    Value *RAdd = RHS->getOperand(0);

    Value *LAddOpnd, *RAddOpnd;
    ConstantInt *LAddCst, *RAddCst;
    if (match(LAdd, m_Add(m_Value(LAddOpnd), m_ConstantInt(LAddCst))) &&
        match(RAdd, m_Add(m_Value(RAddOpnd), m_ConstantInt(RAddCst))) &&
        LAddCst->getValue().ugt(LHSCst->getValue()) &&
        RAddCst->getValue().ugt(LHSCst->getValue())) {

      APInt DiffCst = LAddCst->getValue() ^ RAddCst->getValue();
      if (LAddOpnd == RAddOpnd && DiffCst.isPowerOf2()) {
        ConstantInt *MaxAddCst = nullptr;
        if (LAddCst->getValue().ult(RAddCst->getValue()))
          MaxAddCst = RAddCst;
        else
          MaxAddCst = LAddCst;

        APInt RRangeLow = -RAddCst->getValue();
        APInt RRangeHigh = RRangeLow + LHSCst->getValue();
        APInt LRangeLow = -LAddCst->getValue();
        APInt LRangeHigh = LRangeLow + LHSCst->getValue();
        APInt LowRangeDiff = RRangeLow ^ LRangeLow;
        APInt HighRangeDiff = RRangeHigh ^ LRangeHigh;
        APInt RangeDiff = LRangeLow.sgt(RRangeLow) ? LRangeLow - RRangeLow
                                                   : RRangeLow - LRangeLow;

        if (LowRangeDiff.isPowerOf2() && LowRangeDiff == HighRangeDiff &&
            RangeDiff.ugt(LHSCst->getValue())) {
          Value *MaskCst = ConstantInt::get(LAddCst->getType(), ~DiffCst);

          Value *NewAnd = Builder->CreateAnd(LAddOpnd, MaskCst);
          Value *NewAdd = Builder->CreateAdd(NewAnd, MaxAddCst);
          return (Builder->CreateICmp(LHS->getPredicate(), NewAdd, LHSCst));
        }
      }
    }
  }

  // (icmp1 A, B) | (icmp2 A, B) --> (icmp3 A, B)
  if (PredicatesFoldable(LHSCC, RHSCC)) {
    if (LHS->getOperand(0) == RHS->getOperand(1) &&
        LHS->getOperand(1) == RHS->getOperand(0))
      LHS->swapOperands();
    if (LHS->getOperand(0) == RHS->getOperand(0) &&
        LHS->getOperand(1) == RHS->getOperand(1)) {
      Value *Op0 = LHS->getOperand(0), *Op1 = LHS->getOperand(1);
      unsigned Code = getICmpCode(LHS) | getICmpCode(RHS);
      bool isSigned = LHS->isSigned() || RHS->isSigned();
      return getNewICmpValue(isSigned, Code, Op0, Op1, Builder);
    }
  }

  // handle (roughly):
  // (icmp ne (A & B), C) | (icmp ne (A & D), E)
  if (Value *V = foldLogOpOfMaskedICmps(LHS, RHS, false, Builder))
    return V;

  Value *Val = LHS->getOperand(0), *Val2 = RHS->getOperand(0);
  if (LHS->hasOneUse() || RHS->hasOneUse()) {
    // (icmp eq B, 0) | (icmp ult A, B) -> (icmp ule A, B-1)
    // (icmp eq B, 0) | (icmp ugt B, A) -> (icmp ule A, B-1)
    Value *A = nullptr, *B = nullptr;
    if (LHSCC == ICmpInst::ICMP_EQ && LHSCst && LHSCst->isZero()) {
      B = Val;
      if (RHSCC == ICmpInst::ICMP_ULT && Val == RHS->getOperand(1))
        A = Val2;
      else if (RHSCC == ICmpInst::ICMP_UGT && Val == Val2)
        A = RHS->getOperand(1);
    }
    // (icmp ult A, B) | (icmp eq B, 0) -> (icmp ule A, B-1)
    // (icmp ugt B, A) | (icmp eq B, 0) -> (icmp ule A, B-1)
    else if (RHSCC == ICmpInst::ICMP_EQ && RHSCst && RHSCst->isZero()) {
      B = Val2;
      if (LHSCC == ICmpInst::ICMP_ULT && Val2 == LHS->getOperand(1))
        A = Val;
      else if (LHSCC == ICmpInst::ICMP_UGT && Val2 == Val)
        A = LHS->getOperand(1);
    }
    if (A && B)
      return Builder->CreateICmp(
          ICmpInst::ICMP_UGE,
          Builder->CreateAdd(B, ConstantInt::getSigned(B->getType(), -1)), A);
  }

  // E.g. (icmp slt x, 0) | (icmp sgt x, n) --> icmp ugt x, n
  if (Value *V = simplifyRangeCheck(LHS, RHS, /*Inverted=*/true))
    return V;

  // E.g. (icmp sgt x, n) | (icmp slt x, 0) --> icmp ugt x, n
  if (Value *V = simplifyRangeCheck(RHS, LHS, /*Inverted=*/true))
    return V;
 
  // This only handles icmp of constants: (icmp1 A, C1) | (icmp2 B, C2).
  if (!LHSCst || !RHSCst) return nullptr;

  if (LHSCst == RHSCst && LHSCC == RHSCC) {
    // (icmp ne A, 0) | (icmp ne B, 0) --> (icmp ne (A|B), 0)
    if (LHSCC == ICmpInst::ICMP_NE && LHSCst->isZero()) {
      Value *NewOr = Builder->CreateOr(Val, Val2);
      return Builder->CreateICmp(LHSCC, NewOr, LHSCst);
    }
  }

  // (icmp ult (X + CA), C1) | (icmp eq X, C2) -> (icmp ule (X + CA), C1)
  //   iff C2 + CA == C1.
  if (LHSCC == ICmpInst::ICMP_ULT && RHSCC == ICmpInst::ICMP_EQ) {
    ConstantInt *AddCst;
    if (match(Val, m_Add(m_Specific(Val2), m_ConstantInt(AddCst))))
      if (RHSCst->getValue() + AddCst->getValue() == LHSCst->getValue())
        return Builder->CreateICmpULE(Val, LHSCst);
  }

  // From here on, we only handle:
  //    (icmp1 A, C1) | (icmp2 A, C2) --> something simpler.
  if (Val != Val2) return nullptr;

  // ICMP_[US][GL]E X, CST is folded to ICMP_[US][GL]T elsewhere.
  if (LHSCC == ICmpInst::ICMP_UGE || LHSCC == ICmpInst::ICMP_ULE ||
      RHSCC == ICmpInst::ICMP_UGE || RHSCC == ICmpInst::ICMP_ULE ||
      LHSCC == ICmpInst::ICMP_SGE || LHSCC == ICmpInst::ICMP_SLE ||
      RHSCC == ICmpInst::ICMP_SGE || RHSCC == ICmpInst::ICMP_SLE)
    return nullptr;

  // We can't fold (ugt x, C) | (sgt x, C2).
  if (!PredicatesFoldable(LHSCC, RHSCC))
    return nullptr;

  // Ensure that the larger constant is on the RHS.
  bool ShouldSwap;
  if (CmpInst::isSigned(LHSCC) ||
      (ICmpInst::isEquality(LHSCC) &&
       CmpInst::isSigned(RHSCC)))
    ShouldSwap = LHSCst->getValue().sgt(RHSCst->getValue());
  else
    ShouldSwap = LHSCst->getValue().ugt(RHSCst->getValue());

  if (ShouldSwap) {
    std::swap(LHS, RHS);
    std::swap(LHSCst, RHSCst);
    std::swap(LHSCC, RHSCC);
  }

  // At this point, we know we have two icmp instructions
  // comparing a value against two constants and or'ing the result
  // together.  Because of the above check, we know that we only have
  // ICMP_EQ, ICMP_NE, ICMP_LT, and ICMP_GT here. We also know (from the
  // icmp folding check above), that the two constants are not
  // equal.
  assert(LHSCst != RHSCst && "Compares not folded above?");

  switch (LHSCC) {
  default: llvm_unreachable("Unknown integer condition code!");
  case ICmpInst::ICMP_EQ:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:
      if (LHS->getOperand(0) == RHS->getOperand(0)) {
        // if LHSCst and RHSCst differ only by one bit:
        // (A == C1 || A == C2) -> (A & ~(C1 ^ C2)) == C1
        assert(LHSCst->getValue().ule(LHSCst->getValue()));

        APInt Xor = LHSCst->getValue() ^ RHSCst->getValue();
        if (Xor.isPowerOf2()) {
          Value *NegCst = Builder->getInt(~Xor);
          Value *And = Builder->CreateAnd(LHS->getOperand(0), NegCst);
          return Builder->CreateICmp(ICmpInst::ICMP_EQ, And, LHSCst);
        }
      }

      if (LHSCst == SubOne(RHSCst)) {
        // (X == 13 | X == 14) -> X-13 <u 2
        Constant *AddCST = ConstantExpr::getNeg(LHSCst);
        Value *Add = Builder->CreateAdd(Val, AddCST, Val->getName()+".off");
        AddCST = ConstantExpr::getSub(AddOne(RHSCst), LHSCst);
        return Builder->CreateICmpULT(Add, AddCST);
      }

      break;                         // (X == 13 | X == 15) -> no change
    case ICmpInst::ICMP_UGT:         // (X == 13 | X u> 14) -> no change
    case ICmpInst::ICMP_SGT:         // (X == 13 | X s> 14) -> no change
      break;
    case ICmpInst::ICMP_NE:          // (X == 13 | X != 15) -> X != 15
    case ICmpInst::ICMP_ULT:         // (X == 13 | X u< 15) -> X u< 15
    case ICmpInst::ICMP_SLT:         // (X == 13 | X s< 15) -> X s< 15
      return RHS;
    }
    break;
  case ICmpInst::ICMP_NE:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:          // (X != 13 | X == 15) -> X != 13
    case ICmpInst::ICMP_UGT:         // (X != 13 | X u> 15) -> X != 13
    case ICmpInst::ICMP_SGT:         // (X != 13 | X s> 15) -> X != 13
      return LHS;
    case ICmpInst::ICMP_NE:          // (X != 13 | X != 15) -> true
    case ICmpInst::ICMP_ULT:         // (X != 13 | X u< 15) -> true
    case ICmpInst::ICMP_SLT:         // (X != 13 | X s< 15) -> true
      return Builder->getTrue();
    }
  case ICmpInst::ICMP_ULT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X u< 13 | X == 14) -> no change
      break;
    case ICmpInst::ICMP_UGT:        // (X u< 13 | X u> 15) -> (X-13) u> 2
      // If RHSCst is [us]MAXINT, it is always false.  Not handling
      // this can cause overflow.
      if (RHSCst->isMaxValue(false))
        return LHS;
      return InsertRangeTest(Val, LHSCst, AddOne(RHSCst), false, false);
    case ICmpInst::ICMP_SGT:        // (X u< 13 | X s> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X u< 13 | X != 15) -> X != 15
    case ICmpInst::ICMP_ULT:        // (X u< 13 | X u< 15) -> X u< 15
      return RHS;
    case ICmpInst::ICMP_SLT:        // (X u< 13 | X s< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_SLT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X s< 13 | X == 14) -> no change
      break;
    case ICmpInst::ICMP_SGT:        // (X s< 13 | X s> 15) -> (X-13) s> 2
      // If RHSCst is [us]MAXINT, it is always false.  Not handling
      // this can cause overflow.
      if (RHSCst->isMaxValue(true))
        return LHS;
      return InsertRangeTest(Val, LHSCst, AddOne(RHSCst), true, false);
    case ICmpInst::ICMP_UGT:        // (X s< 13 | X u> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X s< 13 | X != 15) -> X != 15
    case ICmpInst::ICMP_SLT:        // (X s< 13 | X s< 15) -> X s< 15
      return RHS;
    case ICmpInst::ICMP_ULT:        // (X s< 13 | X u< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_UGT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X u> 13 | X == 15) -> X u> 13
    case ICmpInst::ICMP_UGT:        // (X u> 13 | X u> 15) -> X u> 13
      return LHS;
    case ICmpInst::ICMP_SGT:        // (X u> 13 | X s> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X u> 13 | X != 15) -> true
    case ICmpInst::ICMP_ULT:        // (X u> 13 | X u< 15) -> true
      return Builder->getTrue();
    case ICmpInst::ICMP_SLT:        // (X u> 13 | X s< 15) -> no change
      break;
    }
    break;
  case ICmpInst::ICMP_SGT:
    switch (RHSCC) {
    default: llvm_unreachable("Unknown integer condition code!");
    case ICmpInst::ICMP_EQ:         // (X s> 13 | X == 15) -> X > 13
    case ICmpInst::ICMP_SGT:        // (X s> 13 | X s> 15) -> X > 13
      return LHS;
    case ICmpInst::ICMP_UGT:        // (X s> 13 | X u> 15) -> no change
      break;
    case ICmpInst::ICMP_NE:         // (X s> 13 | X != 15) -> true
    case ICmpInst::ICMP_SLT:        // (X s> 13 | X s< 15) -> true
      return Builder->getTrue();
    case ICmpInst::ICMP_ULT:        // (X s> 13 | X u< 15) -> no change
      break;
    }
    break;
  }
  return nullptr;
}

/// FoldOrOfFCmps - Optimize (fcmp)|(fcmp).  NOTE: Unlike the rest of
/// instcombine, this returns a Value which should already be inserted into the
/// function.
Value *InstCombiner::FoldOrOfFCmps(FCmpInst *LHS, FCmpInst *RHS) {
  if (LHS->getPredicate() == FCmpInst::FCMP_UNO &&
      RHS->getPredicate() == FCmpInst::FCMP_UNO &&
      LHS->getOperand(0)->getType() == RHS->getOperand(0)->getType()) {
    if (ConstantFP *LHSC = dyn_cast<ConstantFP>(LHS->getOperand(1)))
      if (ConstantFP *RHSC = dyn_cast<ConstantFP>(RHS->getOperand(1))) {
        // If either of the constants are nans, then the whole thing returns
        // true.
        if (LHSC->getValueAPF().isNaN() || RHSC->getValueAPF().isNaN())
          return Builder->getTrue();

        // Otherwise, no need to compare the two constants, compare the
        // rest.
        return Builder->CreateFCmpUNO(LHS->getOperand(0), RHS->getOperand(0));
      }

    // Handle vector zeros.  This occurs because the canonical form of
    // "fcmp uno x,x" is "fcmp uno x, 0".
    if (isa<ConstantAggregateZero>(LHS->getOperand(1)) &&
        isa<ConstantAggregateZero>(RHS->getOperand(1)))
      return Builder->CreateFCmpUNO(LHS->getOperand(0), RHS->getOperand(0));

    return nullptr;
  }

  Value *Op0LHS = LHS->getOperand(0), *Op0RHS = LHS->getOperand(1);
  Value *Op1LHS = RHS->getOperand(0), *Op1RHS = RHS->getOperand(1);
  FCmpInst::Predicate Op0CC = LHS->getPredicate(), Op1CC = RHS->getPredicate();

  if (Op0LHS == Op1RHS && Op0RHS == Op1LHS) {
    // Swap RHS operands to match LHS.
    Op1CC = FCmpInst::getSwappedPredicate(Op1CC);
    std::swap(Op1LHS, Op1RHS);
  }
  if (Op0LHS == Op1LHS && Op0RHS == Op1RHS) {
    // Simplify (fcmp cc0 x, y) | (fcmp cc1 x, y).
    if (Op0CC == Op1CC)
      return Builder->CreateFCmp((FCmpInst::Predicate)Op0CC, Op0LHS, Op0RHS);
    if (Op0CC == FCmpInst::FCMP_TRUE || Op1CC == FCmpInst::FCMP_TRUE)
      return ConstantInt::get(CmpInst::makeCmpResultType(LHS->getType()), 1);
    if (Op0CC == FCmpInst::FCMP_FALSE)
      return RHS;
    if (Op1CC == FCmpInst::FCMP_FALSE)
      return LHS;
    bool Op0Ordered;
    bool Op1Ordered;
    unsigned Op0Pred = getFCmpCode(Op0CC, Op0Ordered);
    unsigned Op1Pred = getFCmpCode(Op1CC, Op1Ordered);
    if (Op0Ordered == Op1Ordered) {
      // If both are ordered or unordered, return a new fcmp with
      // or'ed predicates.
      return getFCmpValue(Op0Ordered, Op0Pred|Op1Pred, Op0LHS, Op0RHS, Builder);
    }
  }
  return nullptr;
}

/// FoldOrWithConstants - This helper function folds:
///
///     ((A | B) & C1) | (B & C2)
///
/// into:
///
///     (A & C1) | B
///
/// when the XOR of the two constants is "all ones" (-1).
Instruction *InstCombiner::FoldOrWithConstants(BinaryOperator &I, Value *Op,
                                               Value *A, Value *B, Value *C) {
  ConstantInt *CI1 = dyn_cast<ConstantInt>(C);
  if (!CI1) return nullptr;

  Value *V1 = nullptr;
  ConstantInt *CI2 = nullptr;
  if (!match(Op, m_And(m_Value(V1), m_ConstantInt(CI2)))) return nullptr;

  APInt Xor = CI1->getValue() ^ CI2->getValue();
  if (!Xor.isAllOnesValue()) return nullptr;

  if (V1 == A || V1 == B) {
    Value *NewOp = Builder->CreateAnd((V1 == A) ? B : A, CI1);
    return BinaryOperator::CreateOr(NewOp, V1);
  }

  return nullptr;
}

/// \brief This helper function folds:
///
///     ((A | B) & C1) ^ (B & C2)
///
/// into:
///
///     (A & C1) ^ B
///
/// when the XOR of the two constants is "all ones" (-1).
Instruction *InstCombiner::FoldXorWithConstants(BinaryOperator &I, Value *Op,
                                                Value *A, Value *B, Value *C) {
  ConstantInt *CI1 = dyn_cast<ConstantInt>(C);
  if (!CI1)
    return nullptr;

  Value *V1 = nullptr;
  ConstantInt *CI2 = nullptr;
  if (!match(Op, m_And(m_Value(V1), m_ConstantInt(CI2))))
    return nullptr;

  APInt Xor = CI1->getValue() ^ CI2->getValue();
  if (!Xor.isAllOnesValue())
    return nullptr;

  if (V1 == A || V1 == B) {
    Value *NewOp = Builder->CreateAnd(V1 == A ? B : A, CI1);
    return BinaryOperator::CreateXor(NewOp, V1);
  }

  return nullptr;
}

Instruction *InstCombiner::visitOr(BinaryOperator &I) {
  bool Changed = SimplifyAssociativeOrCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyVectorOp(I))
    return ReplaceInstUsesWith(I, V);

  if (Value *V = SimplifyOrInst(Op0, Op1, DL, TLI, DT, AC))
    return ReplaceInstUsesWith(I, V);

  // (A&B)|(A&C) -> A&(B|C) etc
  if (Value *V = SimplifyUsingDistributiveLaws(I))
    return ReplaceInstUsesWith(I, V);

  // See if we can simplify any instructions used by the instruction whose sole
  // purpose is to compute bits we don't care about.
  if (SimplifyDemandedInstructionBits(I))
    return &I;

  if (Value *V = SimplifyBSwap(I))
    return ReplaceInstUsesWith(I, V);

  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    ConstantInt *C1 = nullptr; Value *X = nullptr;
    // (X & C1) | C2 --> (X | C2) & (C1|C2)
    // iff (C1 & C2) == 0.
    if (match(Op0, m_And(m_Value(X), m_ConstantInt(C1))) &&
        (RHS->getValue() & C1->getValue()) != 0 &&
        Op0->hasOneUse()) {
      Value *Or = Builder->CreateOr(X, RHS);
      Or->takeName(Op0);
      return BinaryOperator::CreateAnd(Or,
                             Builder->getInt(RHS->getValue() | C1->getValue()));
    }

    // (X ^ C1) | C2 --> (X | C2) ^ (C1&~C2)
    if (match(Op0, m_Xor(m_Value(X), m_ConstantInt(C1))) &&
        Op0->hasOneUse()) {
      Value *Or = Builder->CreateOr(X, RHS);
      Or->takeName(Op0);
      return BinaryOperator::CreateXor(Or,
                            Builder->getInt(C1->getValue() & ~RHS->getValue()));
    }

    // Try to fold constant and into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;

    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }

  Value *A = nullptr, *B = nullptr;
  ConstantInt *C1 = nullptr, *C2 = nullptr;

  // (A | B) | C  and  A | (B | C)                  -> bswap if possible.
  // (A >> B) | (C << D)  and  (A << B) | (B >> C)  -> bswap if possible.
  if (match(Op0, m_Or(m_Value(), m_Value())) ||
      match(Op1, m_Or(m_Value(), m_Value())) ||
      (match(Op0, m_LogicalShift(m_Value(), m_Value())) &&
       match(Op1, m_LogicalShift(m_Value(), m_Value())))) {
    if (Instruction *BSwap = MatchBSwap(I))
      return BSwap;
  }

  // (X^C)|Y -> (X|Y)^C iff Y&C == 0
  if (Op0->hasOneUse() &&
      match(Op0, m_Xor(m_Value(A), m_ConstantInt(C1))) &&
      MaskedValueIsZero(Op1, C1->getValue(), 0, &I)) {
    Value *NOr = Builder->CreateOr(A, Op1);
    NOr->takeName(Op0);
    return BinaryOperator::CreateXor(NOr, C1);
  }

  // Y|(X^C) -> (X|Y)^C iff Y&C == 0
  if (Op1->hasOneUse() &&
      match(Op1, m_Xor(m_Value(A), m_ConstantInt(C1))) &&
      MaskedValueIsZero(Op0, C1->getValue(), 0, &I)) {
    Value *NOr = Builder->CreateOr(A, Op0);
    NOr->takeName(Op0);
    return BinaryOperator::CreateXor(NOr, C1);
  }

  // ((~A & B) | A) -> (A | B)
  if (match(Op0, m_And(m_Not(m_Value(A)), m_Value(B))) &&
      match(Op1, m_Specific(A)))
    return BinaryOperator::CreateOr(A, B);

  // ((A & B) | ~A) -> (~A | B)
  if (match(Op0, m_And(m_Value(A), m_Value(B))) &&
      match(Op1, m_Not(m_Specific(A))))
    return BinaryOperator::CreateOr(Builder->CreateNot(A), B);

  // (A & (~B)) | (A ^ B) -> (A ^ B)
  if (match(Op0, m_And(m_Value(A), m_Not(m_Value(B)))) &&
      match(Op1, m_Xor(m_Specific(A), m_Specific(B))))
    return BinaryOperator::CreateXor(A, B);

  // (A ^ B) | ( A & (~B)) -> (A ^ B)
  if (match(Op0, m_Xor(m_Value(A), m_Value(B))) &&
      match(Op1, m_And(m_Specific(A), m_Not(m_Specific(B)))))
    return BinaryOperator::CreateXor(A, B);

  // (A & C)|(B & D)
  Value *C = nullptr, *D = nullptr;
  if (match(Op0, m_And(m_Value(A), m_Value(C))) &&
      match(Op1, m_And(m_Value(B), m_Value(D)))) {
    Value *V1 = nullptr, *V2 = nullptr;
    C1 = dyn_cast<ConstantInt>(C);
    C2 = dyn_cast<ConstantInt>(D);
    if (C1 && C2) {  // (A & C1)|(B & C2)
      if ((C1->getValue() & C2->getValue()) == 0) {
        // ((V | N) & C1) | (V & C2) --> (V|N) & (C1|C2)
        // iff (C1&C2) == 0 and (N&~C1) == 0
        if (match(A, m_Or(m_Value(V1), m_Value(V2))) &&
            ((V1 == B &&
              MaskedValueIsZero(V2, ~C1->getValue(), 0, &I)) || // (V|N)
             (V2 == B &&
              MaskedValueIsZero(V1, ~C1->getValue(), 0, &I))))  // (N|V)
          return BinaryOperator::CreateAnd(A,
                                Builder->getInt(C1->getValue()|C2->getValue()));
        // Or commutes, try both ways.
        if (match(B, m_Or(m_Value(V1), m_Value(V2))) &&
            ((V1 == A &&
              MaskedValueIsZero(V2, ~C2->getValue(), 0, &I)) || // (V|N)
             (V2 == A &&
              MaskedValueIsZero(V1, ~C2->getValue(), 0, &I))))  // (N|V)
          return BinaryOperator::CreateAnd(B,
                                Builder->getInt(C1->getValue()|C2->getValue()));

        // ((V|C3)&C1) | ((V|C4)&C2) --> (V|C3|C4)&(C1|C2)
        // iff (C1&C2) == 0 and (C3&~C1) == 0 and (C4&~C2) == 0.
        ConstantInt *C3 = nullptr, *C4 = nullptr;
        if (match(A, m_Or(m_Value(V1), m_ConstantInt(C3))) &&
            (C3->getValue() & ~C1->getValue()) == 0 &&
            match(B, m_Or(m_Specific(V1), m_ConstantInt(C4))) &&
            (C4->getValue() & ~C2->getValue()) == 0) {
          V2 = Builder->CreateOr(V1, ConstantExpr::getOr(C3, C4), "bitfield");
          return BinaryOperator::CreateAnd(V2,
                                Builder->getInt(C1->getValue()|C2->getValue()));
        }
      }
    }

    // (A & (C0?-1:0)) | (B & ~(C0?-1:0)) ->  C0 ? A : B, and commuted variants.
    // Don't do this for vector select idioms, the code generator doesn't handle
    // them well yet.
    if (!I.getType()->isVectorTy()) {
      if (Instruction *Match = MatchSelectFromAndOr(A, B, C, D))
        return Match;
      if (Instruction *Match = MatchSelectFromAndOr(B, A, D, C))
        return Match;
      if (Instruction *Match = MatchSelectFromAndOr(C, B, A, D))
        return Match;
      if (Instruction *Match = MatchSelectFromAndOr(D, A, B, C))
        return Match;
    }

    // ((A&~B)|(~A&B)) -> A^B
    if ((match(C, m_Not(m_Specific(D))) &&
         match(B, m_Not(m_Specific(A)))))
      return BinaryOperator::CreateXor(A, D);
    // ((~B&A)|(~A&B)) -> A^B
    if ((match(A, m_Not(m_Specific(D))) &&
         match(B, m_Not(m_Specific(C)))))
      return BinaryOperator::CreateXor(C, D);
    // ((A&~B)|(B&~A)) -> A^B
    if ((match(C, m_Not(m_Specific(B))) &&
         match(D, m_Not(m_Specific(A)))))
      return BinaryOperator::CreateXor(A, B);
    // ((~B&A)|(B&~A)) -> A^B
    if ((match(A, m_Not(m_Specific(B))) &&
         match(D, m_Not(m_Specific(C)))))
      return BinaryOperator::CreateXor(C, B);

    // ((A|B)&1)|(B&-2) -> (A&1) | B
    if (match(A, m_Or(m_Value(V1), m_Specific(B))) ||
        match(A, m_Or(m_Specific(B), m_Value(V1)))) {
      Instruction *Ret = FoldOrWithConstants(I, Op1, V1, B, C);
      if (Ret) return Ret;
    }
    // (B&-2)|((A|B)&1) -> (A&1) | B
    if (match(B, m_Or(m_Specific(A), m_Value(V1))) ||
        match(B, m_Or(m_Value(V1), m_Specific(A)))) {
      Instruction *Ret = FoldOrWithConstants(I, Op0, A, V1, D);
      if (Ret) return Ret;
    }
    // ((A^B)&1)|(B&-2) -> (A&1) ^ B
    if (match(A, m_Xor(m_Value(V1), m_Specific(B))) ||
        match(A, m_Xor(m_Specific(B), m_Value(V1)))) {
      Instruction *Ret = FoldXorWithConstants(I, Op1, V1, B, C);
      if (Ret) return Ret;
    }
    // (B&-2)|((A^B)&1) -> (A&1) ^ B
    if (match(B, m_Xor(m_Specific(A), m_Value(V1))) ||
        match(B, m_Xor(m_Value(V1), m_Specific(A)))) {
      Instruction *Ret = FoldXorWithConstants(I, Op0, A, V1, D);
      if (Ret) return Ret;
    }
  }

  // (A ^ B) | ((B ^ C) ^ A) -> (A ^ B) | C
  if (match(Op0, m_Xor(m_Value(A), m_Value(B))))
    if (match(Op1, m_Xor(m_Xor(m_Specific(B), m_Value(C)), m_Specific(A))))
      if (Op1->hasOneUse() || cast<BinaryOperator>(Op1)->hasOneUse())
        return BinaryOperator::CreateOr(Op0, C);

  // ((A ^ C) ^ B) | (B ^ A) -> (B ^ A) | C
  if (match(Op0, m_Xor(m_Xor(m_Value(A), m_Value(C)), m_Value(B))))
    if (match(Op1, m_Xor(m_Specific(B), m_Specific(A))))
      if (Op0->hasOneUse() || cast<BinaryOperator>(Op0)->hasOneUse())
        return BinaryOperator::CreateOr(Op1, C);

  // ((B | C) & A) | B -> B | (A & C)
  if (match(Op0, m_And(m_Or(m_Specific(Op1), m_Value(C)), m_Value(A))))
    return BinaryOperator::CreateOr(Op1, Builder->CreateAnd(A, C));

  // (~A | ~B) == (~(A & B)) - De Morgan's Law
  if (Value *Op0NotVal = dyn_castNotVal(Op0))
    if (Value *Op1NotVal = dyn_castNotVal(Op1))
      if (Op0->hasOneUse() && Op1->hasOneUse()) {
        Value *And = Builder->CreateAnd(Op0NotVal, Op1NotVal,
                                        I.getName()+".demorgan");
        return BinaryOperator::CreateNot(And);
      }

  // Canonicalize xor to the RHS.
  bool SwappedForXor = false;
  if (match(Op0, m_Xor(m_Value(), m_Value()))) {
    std::swap(Op0, Op1);
    SwappedForXor = true;
  }

  // A | ( A ^ B) -> A |  B
  // A | (~A ^ B) -> A | ~B
  // (A & B) | (A ^ B)
  if (match(Op1, m_Xor(m_Value(A), m_Value(B)))) {
    if (Op0 == A || Op0 == B)
      return BinaryOperator::CreateOr(A, B);

    if (match(Op0, m_And(m_Specific(A), m_Specific(B))) ||
        match(Op0, m_And(m_Specific(B), m_Specific(A))))
      return BinaryOperator::CreateOr(A, B);

    if (Op1->hasOneUse() && match(A, m_Not(m_Specific(Op0)))) {
      Value *Not = Builder->CreateNot(B, B->getName()+".not");
      return BinaryOperator::CreateOr(Not, Op0);
    }
    if (Op1->hasOneUse() && match(B, m_Not(m_Specific(Op0)))) {
      Value *Not = Builder->CreateNot(A, A->getName()+".not");
      return BinaryOperator::CreateOr(Not, Op0);
    }
  }

  // A | ~(A | B) -> A | ~B
  // A | ~(A ^ B) -> A | ~B
  if (match(Op1, m_Not(m_Value(A))))
    if (BinaryOperator *B = dyn_cast<BinaryOperator>(A))
      if ((Op0 == B->getOperand(0) || Op0 == B->getOperand(1)) &&
          Op1->hasOneUse() && (B->getOpcode() == Instruction::Or ||
                               B->getOpcode() == Instruction::Xor)) {
        Value *NotOp = Op0 == B->getOperand(0) ? B->getOperand(1) :
                                                 B->getOperand(0);
        Value *Not = Builder->CreateNot(NotOp, NotOp->getName()+".not");
        return BinaryOperator::CreateOr(Not, Op0);
      }

  // (A & B) | ((~A) ^ B) -> (~A ^ B)
  if (match(Op0, m_And(m_Value(A), m_Value(B))) &&
      match(Op1, m_Xor(m_Not(m_Specific(A)), m_Specific(B))))
    return BinaryOperator::CreateXor(Builder->CreateNot(A), B);

  // ((~A) ^ B) | (A & B) -> (~A ^ B)
  if (match(Op0, m_Xor(m_Not(m_Value(A)), m_Value(B))) &&
      match(Op1, m_And(m_Specific(A), m_Specific(B))))
    return BinaryOperator::CreateXor(Builder->CreateNot(A), B);

  if (SwappedForXor)
    std::swap(Op0, Op1);

  {
    ICmpInst *LHS = dyn_cast<ICmpInst>(Op0);
    ICmpInst *RHS = dyn_cast<ICmpInst>(Op1);
    if (LHS && RHS)
      if (Value *Res = FoldOrOfICmps(LHS, RHS, &I))
        return ReplaceInstUsesWith(I, Res);

    // TODO: Make this recursive; it's a little tricky because an arbitrary
    // number of 'or' instructions might have to be created.
    Value *X, *Y;
    if (LHS && match(Op1, m_OneUse(m_Or(m_Value(X), m_Value(Y))))) {
      if (auto *Cmp = dyn_cast<ICmpInst>(X))
        if (Value *Res = FoldOrOfICmps(LHS, Cmp, &I))
          return ReplaceInstUsesWith(I, Builder->CreateOr(Res, Y));
      if (auto *Cmp = dyn_cast<ICmpInst>(Y))
        if (Value *Res = FoldOrOfICmps(LHS, Cmp, &I))
          return ReplaceInstUsesWith(I, Builder->CreateOr(Res, X));
    }
    if (RHS && match(Op0, m_OneUse(m_Or(m_Value(X), m_Value(Y))))) {
      if (auto *Cmp = dyn_cast<ICmpInst>(X))
        if (Value *Res = FoldOrOfICmps(Cmp, RHS, &I))
          return ReplaceInstUsesWith(I, Builder->CreateOr(Res, Y));
      if (auto *Cmp = dyn_cast<ICmpInst>(Y))
        if (Value *Res = FoldOrOfICmps(Cmp, RHS, &I))
          return ReplaceInstUsesWith(I, Builder->CreateOr(Res, X));
    }
  }

  // (fcmp uno x, c) | (fcmp uno y, c)  -> (fcmp uno x, y)
  if (FCmpInst *LHS = dyn_cast<FCmpInst>(I.getOperand(0)))
    if (FCmpInst *RHS = dyn_cast<FCmpInst>(I.getOperand(1)))
      if (Value *Res = FoldOrOfFCmps(LHS, RHS))
        return ReplaceInstUsesWith(I, Res);

  // fold (or (cast A), (cast B)) -> (cast (or A, B))
  if (CastInst *Op0C = dyn_cast<CastInst>(Op0)) {
    CastInst *Op1C = dyn_cast<CastInst>(Op1);
    if (Op1C && Op0C->getOpcode() == Op1C->getOpcode()) {// same cast kind ?
      Type *SrcTy = Op0C->getOperand(0)->getType();
      if (SrcTy == Op1C->getOperand(0)->getType() &&
          SrcTy->isIntOrIntVectorTy()) {
        Value *Op0COp = Op0C->getOperand(0), *Op1COp = Op1C->getOperand(0);

        if ((!isa<ICmpInst>(Op0COp) || !isa<ICmpInst>(Op1COp)) &&
            // Only do this if the casts both really cause code to be
            // generated.
            ShouldOptimizeCast(Op0C->getOpcode(), Op0COp, I.getType()) &&
            ShouldOptimizeCast(Op1C->getOpcode(), Op1COp, I.getType())) {
          Value *NewOp = Builder->CreateOr(Op0COp, Op1COp, I.getName());
          return CastInst::Create(Op0C->getOpcode(), NewOp, I.getType());
        }

        // If this is or(cast(icmp), cast(icmp)), try to fold this even if the
        // cast is otherwise not optimizable.  This happens for vector sexts.
        if (ICmpInst *RHS = dyn_cast<ICmpInst>(Op1COp))
          if (ICmpInst *LHS = dyn_cast<ICmpInst>(Op0COp))
            if (Value *Res = FoldOrOfICmps(LHS, RHS, &I))
              return CastInst::Create(Op0C->getOpcode(), Res, I.getType());

        // If this is or(cast(fcmp), cast(fcmp)), try to fold this even if the
        // cast is otherwise not optimizable.  This happens for vector sexts.
        if (FCmpInst *RHS = dyn_cast<FCmpInst>(Op1COp))
          if (FCmpInst *LHS = dyn_cast<FCmpInst>(Op0COp))
            if (Value *Res = FoldOrOfFCmps(LHS, RHS))
              return CastInst::Create(Op0C->getOpcode(), Res, I.getType());
      }
    }
  }

  // or(sext(A), B) -> A ? -1 : B where A is an i1
  // or(A, sext(B)) -> B ? -1 : A where B is an i1
  if (match(Op0, m_SExt(m_Value(A))) && A->getType()->isIntegerTy(1))
    return SelectInst::Create(A, ConstantInt::getSigned(I.getType(), -1), Op1);
  if (match(Op1, m_SExt(m_Value(A))) && A->getType()->isIntegerTy(1))
    return SelectInst::Create(A, ConstantInt::getSigned(I.getType(), -1), Op0);

  // Note: If we've gotten to the point of visiting the outer OR, then the
  // inner one couldn't be simplified.  If it was a constant, then it won't
  // be simplified by a later pass either, so we try swapping the inner/outer
  // ORs in the hopes that we'll be able to simplify it this way.
  // (X|C) | V --> (X|V) | C
  if (Op0->hasOneUse() && !isa<ConstantInt>(Op1) &&
      match(Op0, m_Or(m_Value(A), m_ConstantInt(C1)))) {
    Value *Inner = Builder->CreateOr(A, Op1);
    Inner->takeName(Op0);
    return BinaryOperator::CreateOr(Inner, C1);
  }

  // Change (or (bool?A:B),(bool?C:D)) --> (bool?(or A,C):(or B,D))
  // Since this OR statement hasn't been optimized further yet, we hope
  // that this transformation will allow the new ORs to be optimized.
  {
    Value *X = nullptr, *Y = nullptr;
    if (Op0->hasOneUse() && Op1->hasOneUse() &&
        match(Op0, m_Select(m_Value(X), m_Value(A), m_Value(B))) &&
        match(Op1, m_Select(m_Value(Y), m_Value(C), m_Value(D))) && X == Y) {
      Value *orTrue = Builder->CreateOr(A, C);
      Value *orFalse = Builder->CreateOr(B, D);
      return SelectInst::Create(X, orTrue, orFalse);
    }
  }

  return Changed ? &I : nullptr;
}

Instruction *InstCombiner::visitXor(BinaryOperator &I) {
  bool Changed = SimplifyAssociativeOrCommutative(I);
  Value *Op0 = I.getOperand(0), *Op1 = I.getOperand(1);

  if (Value *V = SimplifyVectorOp(I))
    return ReplaceInstUsesWith(I, V);

  if (Value *V = SimplifyXorInst(Op0, Op1, DL, TLI, DT, AC))
    return ReplaceInstUsesWith(I, V);

  // (A&B)^(A&C) -> A&(B^C) etc
  if (Value *V = SimplifyUsingDistributiveLaws(I))
    return ReplaceInstUsesWith(I, V);

  // See if we can simplify any instructions used by the instruction whose sole
  // purpose is to compute bits we don't care about.
  if (SimplifyDemandedInstructionBits(I))
    return &I;

  if (Value *V = SimplifyBSwap(I))
    return ReplaceInstUsesWith(I, V);

  // Is this a ~ operation?
  if (Value *NotOp = dyn_castNotVal(&I)) {
    if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(NotOp)) {
      if (Op0I->getOpcode() == Instruction::And ||
          Op0I->getOpcode() == Instruction::Or) {
        // ~(~X & Y) --> (X | ~Y) - De Morgan's Law
        // ~(~X | Y) === (X & ~Y) - De Morgan's Law
        if (dyn_castNotVal(Op0I->getOperand(1)))
          Op0I->swapOperands();
        if (Value *Op0NotVal = dyn_castNotVal(Op0I->getOperand(0))) {
          Value *NotY =
            Builder->CreateNot(Op0I->getOperand(1),
                               Op0I->getOperand(1)->getName()+".not");
          if (Op0I->getOpcode() == Instruction::And)
            return BinaryOperator::CreateOr(Op0NotVal, NotY);
          return BinaryOperator::CreateAnd(Op0NotVal, NotY);
        }

        // ~(X & Y) --> (~X | ~Y) - De Morgan's Law
        // ~(X | Y) === (~X & ~Y) - De Morgan's Law
        if (IsFreeToInvert(Op0I->getOperand(0),
                           Op0I->getOperand(0)->hasOneUse()) &&
            IsFreeToInvert(Op0I->getOperand(1),
                           Op0I->getOperand(1)->hasOneUse())) {
          Value *NotX =
            Builder->CreateNot(Op0I->getOperand(0), "notlhs");
          Value *NotY =
            Builder->CreateNot(Op0I->getOperand(1), "notrhs");
          if (Op0I->getOpcode() == Instruction::And)
            return BinaryOperator::CreateOr(NotX, NotY);
          return BinaryOperator::CreateAnd(NotX, NotY);
        }

      } else if (Op0I->getOpcode() == Instruction::AShr) {
        // ~(~X >>s Y) --> (X >>s Y)
        if (Value *Op0NotVal = dyn_castNotVal(Op0I->getOperand(0)))
          return BinaryOperator::CreateAShr(Op0NotVal, Op0I->getOperand(1));
      }
    }
  }

  if (Constant *RHS = dyn_cast<Constant>(Op1)) {
    if (RHS->isAllOnesValue() && Op0->hasOneUse())
      // xor (cmp A, B), true = not (cmp A, B) = !cmp A, B
      if (CmpInst *CI = dyn_cast<CmpInst>(Op0))
        return CmpInst::Create(CI->getOpcode(),
                               CI->getInversePredicate(),
                               CI->getOperand(0), CI->getOperand(1));
  }

  if (ConstantInt *RHS = dyn_cast<ConstantInt>(Op1)) {
    // fold (xor(zext(cmp)), 1) and (xor(sext(cmp)), -1) to ext(!cmp).
    if (CastInst *Op0C = dyn_cast<CastInst>(Op0)) {
      if (CmpInst *CI = dyn_cast<CmpInst>(Op0C->getOperand(0))) {
        if (CI->hasOneUse() && Op0C->hasOneUse()) {
          Instruction::CastOps Opcode = Op0C->getOpcode();
          if ((Opcode == Instruction::ZExt || Opcode == Instruction::SExt) &&
              (RHS == ConstantExpr::getCast(Opcode, Builder->getTrue(),
                                            Op0C->getDestTy()))) {
            CI->setPredicate(CI->getInversePredicate());
            return CastInst::Create(Opcode, CI, Op0C->getType());
          }
        }
      }
    }

    if (BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0)) {
      // ~(c-X) == X-c-1 == X+(-c-1)
      if (Op0I->getOpcode() == Instruction::Sub && RHS->isAllOnesValue())
        if (Constant *Op0I0C = dyn_cast<Constant>(Op0I->getOperand(0))) {
          Constant *NegOp0I0C = ConstantExpr::getNeg(Op0I0C);
          Constant *ConstantRHS = ConstantExpr::getSub(NegOp0I0C,
                                      ConstantInt::get(I.getType(), 1));
          return BinaryOperator::CreateAdd(Op0I->getOperand(1), ConstantRHS);
        }

      if (ConstantInt *Op0CI = dyn_cast<ConstantInt>(Op0I->getOperand(1))) {
        if (Op0I->getOpcode() == Instruction::Add) {
          // ~(X-c) --> (-c-1)-X
          if (RHS->isAllOnesValue()) {
            Constant *NegOp0CI = ConstantExpr::getNeg(Op0CI);
            return BinaryOperator::CreateSub(
                           ConstantExpr::getSub(NegOp0CI,
                                      ConstantInt::get(I.getType(), 1)),
                                      Op0I->getOperand(0));
          } else if (RHS->getValue().isSignBit()) {
            // (X + C) ^ signbit -> (X + C + signbit)
            Constant *C = Builder->getInt(RHS->getValue() + Op0CI->getValue());
            return BinaryOperator::CreateAdd(Op0I->getOperand(0), C);

          }
        } else if (Op0I->getOpcode() == Instruction::Or) {
          // (X|C1)^C2 -> X^(C1|C2) iff X&~C1 == 0
          if (MaskedValueIsZero(Op0I->getOperand(0), Op0CI->getValue(),
                                0, &I)) {
            Constant *NewRHS = ConstantExpr::getOr(Op0CI, RHS);
            // Anything in both C1 and C2 is known to be zero, remove it from
            // NewRHS.
            Constant *CommonBits = ConstantExpr::getAnd(Op0CI, RHS);
            NewRHS = ConstantExpr::getAnd(NewRHS,
                                       ConstantExpr::getNot(CommonBits));
            Worklist.Add(Op0I);
            I.setOperand(0, Op0I->getOperand(0));
            I.setOperand(1, NewRHS);
            return &I;
          }
        } else if (Op0I->getOpcode() == Instruction::LShr) {
          // ((X^C1) >> C2) ^ C3 -> (X>>C2) ^ ((C1>>C2)^C3)
          // E1 = "X ^ C1"
          BinaryOperator *E1;
          ConstantInt *C1;
          if (Op0I->hasOneUse() &&
              (E1 = dyn_cast<BinaryOperator>(Op0I->getOperand(0))) &&
              E1->getOpcode() == Instruction::Xor &&
              (C1 = dyn_cast<ConstantInt>(E1->getOperand(1)))) {
            // fold (C1 >> C2) ^ C3
            ConstantInt *C2 = Op0CI, *C3 = RHS;
            APInt FoldConst = C1->getValue().lshr(C2->getValue());
            FoldConst ^= C3->getValue();
            // Prepare the two operands.
            Value *Opnd0 = Builder->CreateLShr(E1->getOperand(0), C2);
            Opnd0->takeName(Op0I);
            cast<Instruction>(Opnd0)->setDebugLoc(I.getDebugLoc());
            Value *FoldVal = ConstantInt::get(Opnd0->getType(), FoldConst);

            return BinaryOperator::CreateXor(Opnd0, FoldVal);
          }
        }
      }
    }

    // Try to fold constant and into select arguments.
    if (SelectInst *SI = dyn_cast<SelectInst>(Op0))
      if (Instruction *R = FoldOpIntoSelect(I, SI))
        return R;
    if (isa<PHINode>(Op0))
      if (Instruction *NV = FoldOpIntoPhi(I))
        return NV;
  }

  BinaryOperator *Op1I = dyn_cast<BinaryOperator>(Op1);
  if (Op1I) {
    Value *A, *B;
    if (match(Op1I, m_Or(m_Value(A), m_Value(B)))) {
      if (A == Op0) {              // B^(B|A) == (A|B)^B
        Op1I->swapOperands();
        I.swapOperands();
        std::swap(Op0, Op1);
      } else if (B == Op0) {       // B^(A|B) == (A|B)^B
        I.swapOperands();     // Simplified below.
        std::swap(Op0, Op1);
      }
    } else if (match(Op1I, m_And(m_Value(A), m_Value(B))) &&
               Op1I->hasOneUse()){
      if (A == Op0) {                                      // A^(A&B) -> A^(B&A)
        Op1I->swapOperands();
        std::swap(A, B);
      }
      if (B == Op0) {                                      // A^(B&A) -> (B&A)^A
        I.swapOperands();     // Simplified below.
        std::swap(Op0, Op1);
      }
    }
  }

  BinaryOperator *Op0I = dyn_cast<BinaryOperator>(Op0);
  if (Op0I) {
    Value *A, *B;
    if (match(Op0I, m_Or(m_Value(A), m_Value(B))) &&
        Op0I->hasOneUse()) {
      if (A == Op1)                                  // (B|A)^B == (A|B)^B
        std::swap(A, B);
      if (B == Op1)                                  // (A|B)^B == A & ~B
        return BinaryOperator::CreateAnd(A, Builder->CreateNot(Op1));
    } else if (match(Op0I, m_And(m_Value(A), m_Value(B))) &&
               Op0I->hasOneUse()){
      if (A == Op1)                                        // (A&B)^A -> (B&A)^A
        std::swap(A, B);
      if (B == Op1 &&                                      // (B&A)^A == ~B & A
          !isa<ConstantInt>(Op1)) {  // Canonical form is (B&C)^C
        return BinaryOperator::CreateAnd(Builder->CreateNot(A), Op1);
      }
    }
  }

  if (Op0I && Op1I) {
    Value *A, *B, *C, *D;
    // (A & B)^(A | B) -> A ^ B
    if (match(Op0I, m_And(m_Value(A), m_Value(B))) &&
        match(Op1I, m_Or(m_Value(C), m_Value(D)))) {
      if ((A == C && B == D) || (A == D && B == C))
        return BinaryOperator::CreateXor(A, B);
    }
    // (A | B)^(A & B) -> A ^ B
    if (match(Op0I, m_Or(m_Value(A), m_Value(B))) &&
        match(Op1I, m_And(m_Value(C), m_Value(D)))) {
      if ((A == C && B == D) || (A == D && B == C))
        return BinaryOperator::CreateXor(A, B);
    }
    // (A | ~B) ^ (~A | B) -> A ^ B
    if (match(Op0I, m_Or(m_Value(A), m_Not(m_Value(B)))) &&
        match(Op1I, m_Or(m_Not(m_Specific(A)), m_Specific(B)))) {
      return BinaryOperator::CreateXor(A, B);
    }
    // (~A | B) ^ (A | ~B) -> A ^ B
    if (match(Op0I, m_Or(m_Not(m_Value(A)), m_Value(B))) &&
        match(Op1I, m_Or(m_Specific(A), m_Not(m_Specific(B))))) {
      return BinaryOperator::CreateXor(A, B);
    }
    // (A & ~B) ^ (~A & B) -> A ^ B
    if (match(Op0I, m_And(m_Value(A), m_Not(m_Value(B)))) &&
        match(Op1I, m_And(m_Not(m_Specific(A)), m_Specific(B)))) {
      return BinaryOperator::CreateXor(A, B);
    }
    // (~A & B) ^ (A & ~B) -> A ^ B
    if (match(Op0I, m_And(m_Not(m_Value(A)), m_Value(B))) &&
        match(Op1I, m_And(m_Specific(A), m_Not(m_Specific(B))))) {
      return BinaryOperator::CreateXor(A, B);
    }
    // (A ^ C)^(A | B) -> ((~A) & B) ^ C
    if (match(Op0I, m_Xor(m_Value(D), m_Value(C))) &&
        match(Op1I, m_Or(m_Value(A), m_Value(B)))) {
      if (D == A)
        return BinaryOperator::CreateXor(
            Builder->CreateAnd(Builder->CreateNot(A), B), C);
      if (D == B)
        return BinaryOperator::CreateXor(
            Builder->CreateAnd(Builder->CreateNot(B), A), C);
    }
    // (A | B)^(A ^ C) -> ((~A) & B) ^ C
    if (match(Op0I, m_Or(m_Value(A), m_Value(B))) &&
        match(Op1I, m_Xor(m_Value(D), m_Value(C)))) {
      if (D == A)
        return BinaryOperator::CreateXor(
            Builder->CreateAnd(Builder->CreateNot(A), B), C);
      if (D == B)
        return BinaryOperator::CreateXor(
            Builder->CreateAnd(Builder->CreateNot(B), A), C);
    }
    // (A & B) ^ (A ^ B) -> (A | B)
    if (match(Op0I, m_And(m_Value(A), m_Value(B))) &&
        match(Op1I, m_Xor(m_Specific(A), m_Specific(B))))
      return BinaryOperator::CreateOr(A, B);
    // (A ^ B) ^ (A & B) -> (A | B)
    if (match(Op0I, m_Xor(m_Value(A), m_Value(B))) &&
        match(Op1I, m_And(m_Specific(A), m_Specific(B))))
      return BinaryOperator::CreateOr(A, B);
  }

  Value *A = nullptr, *B = nullptr;
  // (A & ~B) ^ (~A) -> ~(A & B)
  if (match(Op0, m_And(m_Value(A), m_Not(m_Value(B)))) &&
      match(Op1, m_Not(m_Specific(A))))
    return BinaryOperator::CreateNot(Builder->CreateAnd(A, B));

  // (icmp1 A, B) ^ (icmp2 A, B) --> (icmp3 A, B)
  if (ICmpInst *RHS = dyn_cast<ICmpInst>(I.getOperand(1)))
    if (ICmpInst *LHS = dyn_cast<ICmpInst>(I.getOperand(0)))
      if (PredicatesFoldable(LHS->getPredicate(), RHS->getPredicate())) {
        if (LHS->getOperand(0) == RHS->getOperand(1) &&
            LHS->getOperand(1) == RHS->getOperand(0))
          LHS->swapOperands();
        if (LHS->getOperand(0) == RHS->getOperand(0) &&
            LHS->getOperand(1) == RHS->getOperand(1)) {
          Value *Op0 = LHS->getOperand(0), *Op1 = LHS->getOperand(1);
          unsigned Code = getICmpCode(LHS) ^ getICmpCode(RHS);
          bool isSigned = LHS->isSigned() || RHS->isSigned();
          return ReplaceInstUsesWith(I,
                               getNewICmpValue(isSigned, Code, Op0, Op1,
                                               Builder));
        }
      }

  // fold (xor (cast A), (cast B)) -> (cast (xor A, B))
  if (CastInst *Op0C = dyn_cast<CastInst>(Op0)) {
    if (CastInst *Op1C = dyn_cast<CastInst>(Op1))
      if (Op0C->getOpcode() == Op1C->getOpcode()) { // same cast kind?
        Type *SrcTy = Op0C->getOperand(0)->getType();
        if (SrcTy == Op1C->getOperand(0)->getType() && SrcTy->isIntegerTy() &&
            // Only do this if the casts both really cause code to be generated.
            ShouldOptimizeCast(Op0C->getOpcode(), Op0C->getOperand(0),
                               I.getType()) &&
            ShouldOptimizeCast(Op1C->getOpcode(), Op1C->getOperand(0),
                               I.getType())) {
          Value *NewOp = Builder->CreateXor(Op0C->getOperand(0),
                                            Op1C->getOperand(0), I.getName());
          return CastInst::Create(Op0C->getOpcode(), NewOp, I.getType());
        }
      }
  }

  return Changed ? &I : nullptr;
}
