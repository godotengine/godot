//===- MergeFunctions.cpp - Merge identical functions ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass looks for equivalent functions that are mergable and folds them.
//
// Order relation is defined on set of functions. It was made through
// special function comparison procedure that returns
// 0 when functions are equal,
// -1 when Left function is less than right function, and
// 1 for opposite case. We need total-ordering, so we need to maintain
// four properties on the functions set:
// a <= a (reflexivity)
// if a <= b and b <= a then a = b (antisymmetry)
// if a <= b and b <= c then a <= c (transitivity).
// for all a and b: a <= b or b <= a (totality).
//
// Comparison iterates through each instruction in each basic block.
// Functions are kept on binary tree. For each new function F we perform
// lookup in binary tree.
// In practice it works the following way:
// -- We define Function* container class with custom "operator<" (FunctionPtr).
// -- "FunctionPtr" instances are stored in std::set collection, so every
//    std::set::insert operation will give you result in log(N) time.
//
// When a match is found the functions are folded. If both functions are
// overridable, we move the functionality into a new internal function and
// leave two overridable thunks to it.
//
//===----------------------------------------------------------------------===//
//
// Future work:
//
// * virtual functions.
//
// Many functions have their address taken by the virtual function table for
// the object they belong to. However, as long as it's only used for a lookup
// and call, this is irrelevant, and we'd like to fold such functions.
//
// * be smarter about bitcasts.
//
// In order to fold functions, we will sometimes add either bitcast instructions
// or bitcast constant expressions. Unfortunately, this can confound further
// analysis since the two functions differ where one has a bitcast and the
// other doesn't. We should learn to look through bitcasts.
//
// * Compare complex types with pointer types inside.
// * Compare cross-reference cases.
// * Compare complex expressions.
//
// All the three issues above could be described as ability to prove that
// fA == fB == fC == fE == fF == fG in example below:
//
//  void fA() {
//    fB();
//  }
//  void fB() {
//    fA();
//  }
//
//  void fE() {
//    fF();
//  }
//  void fF() {
//    fG();
//  }
//  void fG() {
//    fE();
//  }
//
// Simplest cross-reference case (fA <--> fB) was implemented in previous
// versions of MergeFunctions, though it presented only in two function pairs
// in test-suite (that counts >50k functions)
// Though possibility to detect complex cross-referencing (e.g.: A->B->C->D->A)
// could cover much more cases.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/IPO.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"
#include <vector>
using namespace llvm;

#define DEBUG_TYPE "mergefunc"

STATISTIC(NumFunctionsMerged, "Number of functions merged");
STATISTIC(NumThunksWritten, "Number of thunks generated");
STATISTIC(NumAliasesWritten, "Number of aliases generated");
STATISTIC(NumDoubleWeak, "Number of new functions created");

#if 0 // HLSL Change
static cl::opt<unsigned> NumFunctionsForSanityCheck(
    "mergefunc-sanity",
    cl::desc("How many functions in module could be used for "
             "MergeFunctions pass sanity check. "
             "'0' disables this check. Works only with '-debug' key."),
    cl::init(0), cl::Hidden);
#endif

namespace {

/// FunctionComparator - Compares two functions to determine whether or not
/// they will generate machine code with the same behaviour. DataLayout is
/// used if available. The comparator always fails conservatively (erring on the
/// side of claiming that two functions are different).
class FunctionComparator {
public:
  FunctionComparator(const Function *F1, const Function *F2)
      : FnL(F1), FnR(F2) {}

  /// Test whether the two functions have equivalent behaviour.
  int compare();

private:
  /// Test whether two basic blocks have equivalent behaviour.
  int compare(const BasicBlock *BBL, const BasicBlock *BBR);

  /// Constants comparison.
  /// Its analog to lexicographical comparison between hypothetical numbers
  /// of next format:
  /// <bitcastability-trait><raw-bit-contents>
  ///
  /// 1. Bitcastability.
  /// Check whether L's type could be losslessly bitcasted to R's type.
  /// On this stage method, in case when lossless bitcast is not possible
  /// method returns -1 or 1, thus also defining which type is greater in
  /// context of bitcastability.
  /// Stage 0: If types are equal in terms of cmpTypes, then we can go straight
  ///          to the contents comparison.
  ///          If types differ, remember types comparison result and check
  ///          whether we still can bitcast types.
  /// Stage 1: Types that satisfies isFirstClassType conditions are always
  ///          greater then others.
  /// Stage 2: Vector is greater then non-vector.
  ///          If both types are vectors, then vector with greater bitwidth is
  ///          greater.
  ///          If both types are vectors with the same bitwidth, then types
  ///          are bitcastable, and we can skip other stages, and go to contents
  ///          comparison.
  /// Stage 3: Pointer types are greater than non-pointers. If both types are
  ///          pointers of the same address space - go to contents comparison.
  ///          Different address spaces: pointer with greater address space is
  ///          greater.
  /// Stage 4: Types are neither vectors, nor pointers. And they differ.
  ///          We don't know how to bitcast them. So, we better don't do it,
  ///          and return types comparison result (so it determines the
  ///          relationship among constants we don't know how to bitcast).
  ///
  /// Just for clearance, let's see how the set of constants could look
  /// on single dimension axis:
  ///
  /// [NFCT], [FCT, "others"], [FCT, pointers], [FCT, vectors]
  /// Where: NFCT - Not a FirstClassType
  ///        FCT - FirstClassTyp:
  ///
  /// 2. Compare raw contents.
  /// It ignores types on this stage and only compares bits from L and R.
  /// Returns 0, if L and R has equivalent contents.
  /// -1 or 1 if values are different.
  /// Pretty trivial:
  /// 2.1. If contents are numbers, compare numbers.
  ///    Ints with greater bitwidth are greater. Ints with same bitwidths
  ///    compared by their contents.
  /// 2.2. "And so on". Just to avoid discrepancies with comments
  /// perhaps it would be better to read the implementation itself.
  /// 3. And again about overall picture. Let's look back at how the ordered set
  /// of constants will look like:
  /// [NFCT], [FCT, "others"], [FCT, pointers], [FCT, vectors]
  ///
  /// Now look, what could be inside [FCT, "others"], for example:
  /// [FCT, "others"] =
  /// [
  ///   [double 0.1], [double 1.23],
  ///   [i32 1], [i32 2],
  ///   { double 1.0 },       ; StructTyID, NumElements = 1
  ///   { i32 1 },            ; StructTyID, NumElements = 1
  ///   { double 1, i32 1 },  ; StructTyID, NumElements = 2
  ///   { i32 1, double 1 }   ; StructTyID, NumElements = 2
  /// ]
  ///
  /// Let's explain the order. Float numbers will be less than integers, just
  /// because of cmpType terms: FloatTyID < IntegerTyID.
  /// Floats (with same fltSemantics) are sorted according to their value.
  /// Then you can see integers, and they are, like a floats,
  /// could be easy sorted among each others.
  /// The structures. Structures are grouped at the tail, again because of their
  /// TypeID: StructTyID > IntegerTyID > FloatTyID.
  /// Structures with greater number of elements are greater. Structures with
  /// greater elements going first are greater.
  /// The same logic with vectors, arrays and other possible complex types.
  ///
  /// Bitcastable constants.
  /// Let's assume, that some constant, belongs to some group of
  /// "so-called-equal" values with different types, and at the same time
  /// belongs to another group of constants with equal types
  /// and "really" equal values.
  ///
  /// Now, prove that this is impossible:
  ///
  /// If constant A with type TyA is bitcastable to B with type TyB, then:
  /// 1. All constants with equal types to TyA, are bitcastable to B. Since
  ///    those should be vectors (if TyA is vector), pointers
  ///    (if TyA is pointer), or else (if TyA equal to TyB), those types should
  ///    be equal to TyB.
  /// 2. All constants with non-equal, but bitcastable types to TyA, are
  ///    bitcastable to B.
  ///    Once again, just because we allow it to vectors and pointers only.
  ///    This statement could be expanded as below:
  /// 2.1. All vectors with equal bitwidth to vector A, has equal bitwidth to
  ///      vector B, and thus bitcastable to B as well.
  /// 2.2. All pointers of the same address space, no matter what they point to,
  ///      bitcastable. So if C is pointer, it could be bitcasted to A and to B.
  /// So any constant equal or bitcastable to A is equal or bitcastable to B.
  /// QED.
  ///
  /// In another words, for pointers and vectors, we ignore top-level type and
  /// look at their particular properties (bit-width for vectors, and
  /// address space for pointers).
  /// If these properties are equal - compare their contents.
  int cmpConstants(const Constant *L, const Constant *R);

  /// Assign or look up previously assigned numbers for the two values, and
  /// return whether the numbers are equal. Numbers are assigned in the order
  /// visited.
  /// Comparison order:
  /// Stage 0: Value that is function itself is always greater then others.
  ///          If left and right values are references to their functions, then
  ///          they are equal.
  /// Stage 1: Constants are greater than non-constants.
  ///          If both left and right are constants, then the result of
  ///          cmpConstants is used as cmpValues result.
  /// Stage 2: InlineAsm instances are greater than others. If both left and
  ///          right are InlineAsm instances, InlineAsm* pointers casted to
  ///          integers and compared as numbers.
  /// Stage 3: For all other cases we compare order we meet these values in
  ///          their functions. If right value was met first during scanning,
  ///          then left value is greater.
  ///          In another words, we compare serial numbers, for more details
  ///          see comments for sn_mapL and sn_mapR.
  int cmpValues(const Value *L, const Value *R);

  /// Compare two Instructions for equivalence, similar to
  /// Instruction::isSameOperationAs but with modifications to the type
  /// comparison.
  /// Stages are listed in "most significant stage first" order:
  /// On each stage below, we do comparison between some left and right
  /// operation parts. If parts are non-equal, we assign parts comparison
  /// result to the operation comparison result and exit from method.
  /// Otherwise we proceed to the next stage.
  /// Stages:
  /// 1. Operations opcodes. Compared as numbers.
  /// 2. Number of operands.
  /// 3. Operation types. Compared with cmpType method.
  /// 4. Compare operation subclass optional data as stream of bytes:
  /// just convert it to integers and call cmpNumbers.
  /// 5. Compare in operation operand types with cmpType in
  /// most significant operand first order.
  /// 6. Last stage. Check operations for some specific attributes.
  /// For example, for Load it would be:
  /// 6.1.Load: volatile (as boolean flag)
  /// 6.2.Load: alignment (as integer numbers)
  /// 6.3.Load: synch-scope (as integer numbers)
  /// 6.4.Load: range metadata (as integer numbers)
  /// On this stage its better to see the code, since its not more than 10-15
  /// strings for particular instruction, and could change sometimes.
  int cmpOperations(const Instruction *L, const Instruction *R) const;

  /// Compare two GEPs for equivalent pointer arithmetic.
  /// Parts to be compared for each comparison stage,
  /// most significant stage first:
  /// 1. Address space. As numbers.
  /// 2. Constant offset, (using GEPOperator::accumulateConstantOffset method).
  /// 3. Pointer operand type (using cmpType method).
  /// 4. Number of operands.
  /// 5. Compare operands, using cmpValues method.
  int cmpGEPs(const GEPOperator *GEPL, const GEPOperator *GEPR);
  int cmpGEPs(const GetElementPtrInst *GEPL, const GetElementPtrInst *GEPR) {
    return cmpGEPs(cast<GEPOperator>(GEPL), cast<GEPOperator>(GEPR));
  }

  /// cmpType - compares two types,
  /// defines total ordering among the types set.
  ///
  /// Return values:
  /// 0 if types are equal,
  /// -1 if Left is less than Right,
  /// +1 if Left is greater than Right.
  ///
  /// Description:
  /// Comparison is broken onto stages. Like in lexicographical comparison
  /// stage coming first has higher priority.
  /// On each explanation stage keep in mind total ordering properties.
  ///
  /// 0. Before comparison we coerce pointer types of 0 address space to
  /// integer.
  /// We also don't bother with same type at left and right, so
  /// just return 0 in this case.
  ///
  /// 1. If types are of different kind (different type IDs).
  ///    Return result of type IDs comparison, treating them as numbers.
  /// 2. If types are vectors or integers, compare Type* values as numbers.
  /// 3. Types has same ID, so check whether they belongs to the next group:
  /// * Void
  /// * Float
  /// * Double
  /// * X86_FP80
  /// * FP128
  /// * PPC_FP128
  /// * Label
  /// * Metadata
  /// If so - return 0, yes - we can treat these types as equal only because
  /// their IDs are same.
  /// 4. If Left and Right are pointers, return result of address space
  /// comparison (numbers comparison). We can treat pointer types of same
  /// address space as equal.
  /// 5. If types are complex.
  /// Then both Left and Right are to be expanded and their element types will
  /// be checked with the same way. If we get Res != 0 on some stage, return it.
  /// Otherwise return 0.
  /// 6. For all other cases put llvm_unreachable.
  int cmpTypes(Type *TyL, Type *TyR) const;

  int cmpNumbers(uint64_t L, uint64_t R) const;

  int cmpAPInts(const APInt &L, const APInt &R) const;
  int cmpAPFloats(const APFloat &L, const APFloat &R) const;
  int cmpStrings(StringRef L, StringRef R) const;
  int cmpAttrs(const AttributeSet L, const AttributeSet R) const;

  // The two functions undergoing comparison.
  const Function *FnL, *FnR;

  /// Assign serial numbers to values from left function, and values from
  /// right function.
  /// Explanation:
  /// Being comparing functions we need to compare values we meet at left and
  /// right sides.
  /// Its easy to sort things out for external values. It just should be
  /// the same value at left and right.
  /// But for local values (those were introduced inside function body)
  /// we have to ensure they were introduced at exactly the same place,
  /// and plays the same role.
  /// Let's assign serial number to each value when we meet it first time.
  /// Values that were met at same place will be with same serial numbers.
  /// In this case it would be good to explain few points about values assigned
  /// to BBs and other ways of implementation (see below).
  ///
  /// 1. Safety of BB reordering.
  /// It's safe to change the order of BasicBlocks in function.
  /// Relationship with other functions and serial numbering will not be
  /// changed in this case.
  /// As follows from FunctionComparator::compare(), we do CFG walk: we start
  /// from the entry, and then take each terminator. So it doesn't matter how in
  /// fact BBs are ordered in function. And since cmpValues are called during
  /// this walk, the numbering depends only on how BBs located inside the CFG.
  /// So the answer is - yes. We will get the same numbering.
  ///
  /// 2. Impossibility to use dominance properties of values.
  /// If we compare two instruction operands: first is usage of local
  /// variable AL from function FL, and second is usage of local variable AR
  /// from FR, we could compare their origins and check whether they are
  /// defined at the same place.
  /// But, we are still not able to compare operands of PHI nodes, since those
  /// could be operands from further BBs we didn't scan yet.
  /// So it's impossible to use dominance properties in general.
  DenseMap<const Value*, int> sn_mapL, sn_mapR;
};

class FunctionNode {
  mutable AssertingVH<Function> F;

public:
  FunctionNode(Function *F) : F(F) {}
  Function *getFunc() const { return F; }

  /// Replace the reference to the function F by the function G, assuming their
  /// implementations are equal.
  void replaceBy(Function *G) const {
    assert(!(*this < FunctionNode(G)) && !(FunctionNode(G) < *this) &&
           "The two functions must be equal");

    F = G;
  }

  void release() { F = 0; }
  bool operator<(const FunctionNode &RHS) const {
    return (FunctionComparator(F, RHS.getFunc()).compare()) == -1;
  }
};
}

int FunctionComparator::cmpNumbers(uint64_t L, uint64_t R) const {
  if (L < R) return -1;
  if (L > R) return 1;
  return 0;
}

int FunctionComparator::cmpAPInts(const APInt &L, const APInt &R) const {
  if (int Res = cmpNumbers(L.getBitWidth(), R.getBitWidth()))
    return Res;
  if (L.ugt(R)) return 1;
  if (R.ugt(L)) return -1;
  return 0;
}

int FunctionComparator::cmpAPFloats(const APFloat &L, const APFloat &R) const {
  if (int Res = cmpNumbers((uint64_t)&L.getSemantics(),
                           (uint64_t)&R.getSemantics()))
    return Res;
  return cmpAPInts(L.bitcastToAPInt(), R.bitcastToAPInt());
}

int FunctionComparator::cmpStrings(StringRef L, StringRef R) const {
  // Prevent heavy comparison, compare sizes first.
  if (int Res = cmpNumbers(L.size(), R.size()))
    return Res;

  // Compare strings lexicographically only when it is necessary: only when
  // strings are equal in size.
  return L.compare(R);
}

int FunctionComparator::cmpAttrs(const AttributeSet L,
                                 const AttributeSet R) const {
  if (int Res = cmpNumbers(L.getNumSlots(), R.getNumSlots()))
    return Res;

  for (unsigned i = 0, e = L.getNumSlots(); i != e; ++i) {
    AttributeSet::iterator LI = L.begin(i), LE = L.end(i), RI = R.begin(i),
                           RE = R.end(i);
    for (; LI != LE && RI != RE; ++LI, ++RI) {
      Attribute LA = *LI;
      Attribute RA = *RI;
      if (LA < RA)
        return -1;
      if (RA < LA)
        return 1;
    }
    if (LI != LE)
      return 1;
    if (RI != RE)
      return -1;
  }
  return 0;
}

/// Constants comparison:
/// 1. Check whether type of L constant could be losslessly bitcasted to R
/// type.
/// 2. Compare constant contents.
/// For more details see declaration comments.
int FunctionComparator::cmpConstants(const Constant *L, const Constant *R) {

  Type *TyL = L->getType();
  Type *TyR = R->getType();

  // Check whether types are bitcastable. This part is just re-factored
  // Type::canLosslesslyBitCastTo method, but instead of returning true/false,
  // we also pack into result which type is "less" for us.
  int TypesRes = cmpTypes(TyL, TyR);
  if (TypesRes != 0) {
    // Types are different, but check whether we can bitcast them.
    if (!TyL->isFirstClassType()) {
      if (TyR->isFirstClassType())
        return -1;
      // Neither TyL nor TyR are values of first class type. Return the result
      // of comparing the types
      return TypesRes;
    }
    if (!TyR->isFirstClassType()) {
      if (TyL->isFirstClassType())
        return 1;
      return TypesRes;
    }

    // Vector -> Vector conversions are always lossless if the two vector types
    // have the same size, otherwise not.
    unsigned TyLWidth = 0;
    unsigned TyRWidth = 0;

    if (const VectorType *VecTyL = dyn_cast<VectorType>(TyL))
      TyLWidth = VecTyL->getBitWidth();
    if (const VectorType *VecTyR = dyn_cast<VectorType>(TyR))
      TyRWidth = VecTyR->getBitWidth();

    if (TyLWidth != TyRWidth)
      return cmpNumbers(TyLWidth, TyRWidth);

    // Zero bit-width means neither TyL nor TyR are vectors.
    if (!TyLWidth) {
      PointerType *PTyL = dyn_cast<PointerType>(TyL);
      PointerType *PTyR = dyn_cast<PointerType>(TyR);
      if (PTyL && PTyR) {
        unsigned AddrSpaceL = PTyL->getAddressSpace();
        unsigned AddrSpaceR = PTyR->getAddressSpace();
        if (int Res = cmpNumbers(AddrSpaceL, AddrSpaceR))
          return Res;
      }
      if (PTyL)
        return 1;
      if (PTyR)
        return -1;

      // TyL and TyR aren't vectors, nor pointers. We don't know how to
      // bitcast them.
      return TypesRes;
    }
  }

  // OK, types are bitcastable, now check constant contents.

  if (L->isNullValue() && R->isNullValue())
    return TypesRes;
  if (L->isNullValue() && !R->isNullValue())
    return 1;
  if (!L->isNullValue() && R->isNullValue())
    return -1;

  if (int Res = cmpNumbers(L->getValueID(), R->getValueID()))
    return Res;

  switch (L->getValueID()) {
  case Value::UndefValueVal: return TypesRes;
  case Value::ConstantIntVal: {
    const APInt &LInt = cast<ConstantInt>(L)->getValue();
    const APInt &RInt = cast<ConstantInt>(R)->getValue();
    return cmpAPInts(LInt, RInt);
  }
  case Value::ConstantFPVal: {
    const APFloat &LAPF = cast<ConstantFP>(L)->getValueAPF();
    const APFloat &RAPF = cast<ConstantFP>(R)->getValueAPF();
    return cmpAPFloats(LAPF, RAPF);
  }
  case Value::ConstantArrayVal: {
    const ConstantArray *LA = cast<ConstantArray>(L);
    const ConstantArray *RA = cast<ConstantArray>(R);
    uint64_t NumElementsL = cast<ArrayType>(TyL)->getNumElements();
    uint64_t NumElementsR = cast<ArrayType>(TyR)->getNumElements();
    if (int Res = cmpNumbers(NumElementsL, NumElementsR))
      return Res;
    for (uint64_t i = 0; i < NumElementsL; ++i) {
      if (int Res = cmpConstants(cast<Constant>(LA->getOperand(i)),
                                 cast<Constant>(RA->getOperand(i))))
        return Res;
    }
    return 0;
  }
  case Value::ConstantStructVal: {
    const ConstantStruct *LS = cast<ConstantStruct>(L);
    const ConstantStruct *RS = cast<ConstantStruct>(R);
    unsigned NumElementsL = cast<StructType>(TyL)->getNumElements();
    unsigned NumElementsR = cast<StructType>(TyR)->getNumElements();
    if (int Res = cmpNumbers(NumElementsL, NumElementsR))
      return Res;
    for (unsigned i = 0; i != NumElementsL; ++i) {
      if (int Res = cmpConstants(cast<Constant>(LS->getOperand(i)),
                                 cast<Constant>(RS->getOperand(i))))
        return Res;
    }
    return 0;
  }
  case Value::ConstantVectorVal: {
    const ConstantVector *LV = cast<ConstantVector>(L);
    const ConstantVector *RV = cast<ConstantVector>(R);
    unsigned NumElementsL = cast<VectorType>(TyL)->getNumElements();
    unsigned NumElementsR = cast<VectorType>(TyR)->getNumElements();
    if (int Res = cmpNumbers(NumElementsL, NumElementsR))
      return Res;
    for (uint64_t i = 0; i < NumElementsL; ++i) {
      if (int Res = cmpConstants(cast<Constant>(LV->getOperand(i)),
                                 cast<Constant>(RV->getOperand(i))))
        return Res;
    }
    return 0;
  }
  case Value::ConstantExprVal: {
    const ConstantExpr *LE = cast<ConstantExpr>(L);
    const ConstantExpr *RE = cast<ConstantExpr>(R);
    unsigned NumOperandsL = LE->getNumOperands();
    unsigned NumOperandsR = RE->getNumOperands();
    if (int Res = cmpNumbers(NumOperandsL, NumOperandsR))
      return Res;
    for (unsigned i = 0; i < NumOperandsL; ++i) {
      if (int Res = cmpConstants(cast<Constant>(LE->getOperand(i)),
                                 cast<Constant>(RE->getOperand(i))))
        return Res;
    }
    return 0;
  }
  case Value::FunctionVal:
  case Value::GlobalVariableVal:
  case Value::GlobalAliasVal:
  default: // Unknown constant, cast L and R pointers to numbers and compare.
    return cmpNumbers((uint64_t)L, (uint64_t)R);
  }
}

/// cmpType - compares two types,
/// defines total ordering among the types set.
/// See method declaration comments for more details.
int FunctionComparator::cmpTypes(Type *TyL, Type *TyR) const {

  PointerType *PTyL = dyn_cast<PointerType>(TyL);
  PointerType *PTyR = dyn_cast<PointerType>(TyR);

  const DataLayout &DL = FnL->getParent()->getDataLayout();
  if (PTyL && PTyL->getAddressSpace() == 0)
    TyL = DL.getIntPtrType(TyL);
  if (PTyR && PTyR->getAddressSpace() == 0)
    TyR = DL.getIntPtrType(TyR);

  if (TyL == TyR)
    return 0;

  if (int Res = cmpNumbers(TyL->getTypeID(), TyR->getTypeID()))
    return Res;

  switch (TyL->getTypeID()) {
  default:
    llvm_unreachable("Unknown type!");
    // Fall through in Release mode.
  case Type::IntegerTyID:
  case Type::VectorTyID:
    // TyL == TyR would have returned true earlier.
    return cmpNumbers((uint64_t)TyL, (uint64_t)TyR);

  case Type::VoidTyID:
  case Type::FloatTyID:
  case Type::DoubleTyID:
  case Type::X86_FP80TyID:
  case Type::FP128TyID:
  case Type::PPC_FP128TyID:
  case Type::LabelTyID:
  case Type::MetadataTyID:
    return 0;

  case Type::PointerTyID: {
    assert(PTyL && PTyR && "Both types must be pointers here.");
    return cmpNumbers(PTyL->getAddressSpace(), PTyR->getAddressSpace());
  }

  case Type::StructTyID: {
    StructType *STyL = cast<StructType>(TyL);
    StructType *STyR = cast<StructType>(TyR);
    if (STyL->getNumElements() != STyR->getNumElements())
      return cmpNumbers(STyL->getNumElements(), STyR->getNumElements());

    if (STyL->isPacked() != STyR->isPacked())
      return cmpNumbers(STyL->isPacked(), STyR->isPacked());

    for (unsigned i = 0, e = STyL->getNumElements(); i != e; ++i) {
      if (int Res = cmpTypes(STyL->getElementType(i), STyR->getElementType(i)))
        return Res;
    }
    return 0;
  }

  case Type::FunctionTyID: {
    FunctionType *FTyL = cast<FunctionType>(TyL);
    FunctionType *FTyR = cast<FunctionType>(TyR);
    if (FTyL->getNumParams() != FTyR->getNumParams())
      return cmpNumbers(FTyL->getNumParams(), FTyR->getNumParams());

    if (FTyL->isVarArg() != FTyR->isVarArg())
      return cmpNumbers(FTyL->isVarArg(), FTyR->isVarArg());

    if (int Res = cmpTypes(FTyL->getReturnType(), FTyR->getReturnType()))
      return Res;

    for (unsigned i = 0, e = FTyL->getNumParams(); i != e; ++i) {
      if (int Res = cmpTypes(FTyL->getParamType(i), FTyR->getParamType(i)))
        return Res;
    }
    return 0;
  }

  case Type::ArrayTyID: {
    ArrayType *ATyL = cast<ArrayType>(TyL);
    ArrayType *ATyR = cast<ArrayType>(TyR);
    if (ATyL->getNumElements() != ATyR->getNumElements())
      return cmpNumbers(ATyL->getNumElements(), ATyR->getNumElements());
    return cmpTypes(ATyL->getElementType(), ATyR->getElementType());
  }
  }
}

// Determine whether the two operations are the same except that pointer-to-A
// and pointer-to-B are equivalent. This should be kept in sync with
// Instruction::isSameOperationAs.
// Read method declaration comments for more details.
int FunctionComparator::cmpOperations(const Instruction *L,
                                      const Instruction *R) const {
  // Differences from Instruction::isSameOperationAs:
  //  * replace type comparison with calls to isEquivalentType.
  //  * we test for I->hasSameSubclassOptionalData (nuw/nsw/tail) at the top
  //  * because of the above, we don't test for the tail bit on calls later on
  if (int Res = cmpNumbers(L->getOpcode(), R->getOpcode()))
    return Res;

  if (int Res = cmpNumbers(L->getNumOperands(), R->getNumOperands()))
    return Res;

  if (int Res = cmpTypes(L->getType(), R->getType()))
    return Res;

  if (int Res = cmpNumbers(L->getRawSubclassOptionalData(),
                           R->getRawSubclassOptionalData()))
    return Res;

  if (const AllocaInst *AI = dyn_cast<AllocaInst>(L)) {
    if (int Res = cmpTypes(AI->getAllocatedType(),
                           cast<AllocaInst>(R)->getAllocatedType()))
      return Res;
    if (int Res =
            cmpNumbers(AI->getAlignment(), cast<AllocaInst>(R)->getAlignment()))
      return Res;
  }

  // We have two instructions of identical opcode and #operands.  Check to see
  // if all operands are the same type
  for (unsigned i = 0, e = L->getNumOperands(); i != e; ++i) {
    if (int Res =
            cmpTypes(L->getOperand(i)->getType(), R->getOperand(i)->getType()))
      return Res;
  }

  // Check special state that is a part of some instructions.
  if (const LoadInst *LI = dyn_cast<LoadInst>(L)) {
    if (int Res = cmpNumbers(LI->isVolatile(), cast<LoadInst>(R)->isVolatile()))
      return Res;
    if (int Res =
            cmpNumbers(LI->getAlignment(), cast<LoadInst>(R)->getAlignment()))
      return Res;
    if (int Res =
            cmpNumbers(LI->getOrdering(), cast<LoadInst>(R)->getOrdering()))
      return Res;
    if (int Res =
            cmpNumbers(LI->getSynchScope(), cast<LoadInst>(R)->getSynchScope()))
      return Res;
    return cmpNumbers((uint64_t)LI->getMetadata(LLVMContext::MD_range),
                      (uint64_t)cast<LoadInst>(R)->getMetadata(LLVMContext::MD_range));
  }
  if (const StoreInst *SI = dyn_cast<StoreInst>(L)) {
    if (int Res =
            cmpNumbers(SI->isVolatile(), cast<StoreInst>(R)->isVolatile()))
      return Res;
    if (int Res =
            cmpNumbers(SI->getAlignment(), cast<StoreInst>(R)->getAlignment()))
      return Res;
    if (int Res =
            cmpNumbers(SI->getOrdering(), cast<StoreInst>(R)->getOrdering()))
      return Res;
    return cmpNumbers(SI->getSynchScope(), cast<StoreInst>(R)->getSynchScope());
  }
  if (const CmpInst *CI = dyn_cast<CmpInst>(L))
    return cmpNumbers(CI->getPredicate(), cast<CmpInst>(R)->getPredicate());
  if (const CallInst *CI = dyn_cast<CallInst>(L)) {
    if (int Res = cmpNumbers(CI->getCallingConv(),
                             cast<CallInst>(R)->getCallingConv()))
      return Res;
    if (int Res =
            cmpAttrs(CI->getAttributes(), cast<CallInst>(R)->getAttributes()))
      return Res;
    return cmpNumbers(
        (uint64_t)CI->getMetadata(LLVMContext::MD_range),
        (uint64_t)cast<CallInst>(R)->getMetadata(LLVMContext::MD_range));
  }
  if (const InvokeInst *CI = dyn_cast<InvokeInst>(L)) {
    if (int Res = cmpNumbers(CI->getCallingConv(),
                             cast<InvokeInst>(R)->getCallingConv()))
      return Res;
    if (int Res =
            cmpAttrs(CI->getAttributes(), cast<InvokeInst>(R)->getAttributes()))
      return Res;
    return cmpNumbers(
        (uint64_t)CI->getMetadata(LLVMContext::MD_range),
        (uint64_t)cast<InvokeInst>(R)->getMetadata(LLVMContext::MD_range));
  }
  if (const InsertValueInst *IVI = dyn_cast<InsertValueInst>(L)) {
    ArrayRef<unsigned> LIndices = IVI->getIndices();
    ArrayRef<unsigned> RIndices = cast<InsertValueInst>(R)->getIndices();
    if (int Res = cmpNumbers(LIndices.size(), RIndices.size()))
      return Res;
    for (size_t i = 0, e = LIndices.size(); i != e; ++i) {
      if (int Res = cmpNumbers(LIndices[i], RIndices[i]))
        return Res;
    }
  }
  if (const ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(L)) {
    ArrayRef<unsigned> LIndices = EVI->getIndices();
    ArrayRef<unsigned> RIndices = cast<ExtractValueInst>(R)->getIndices();
    if (int Res = cmpNumbers(LIndices.size(), RIndices.size()))
      return Res;
    for (size_t i = 0, e = LIndices.size(); i != e; ++i) {
      if (int Res = cmpNumbers(LIndices[i], RIndices[i]))
        return Res;
    }
  }
  if (const FenceInst *FI = dyn_cast<FenceInst>(L)) {
    if (int Res =
            cmpNumbers(FI->getOrdering(), cast<FenceInst>(R)->getOrdering()))
      return Res;
    return cmpNumbers(FI->getSynchScope(), cast<FenceInst>(R)->getSynchScope());
  }

  if (const AtomicCmpXchgInst *CXI = dyn_cast<AtomicCmpXchgInst>(L)) {
    if (int Res = cmpNumbers(CXI->isVolatile(),
                             cast<AtomicCmpXchgInst>(R)->isVolatile()))
      return Res;
    if (int Res = cmpNumbers(CXI->isWeak(),
                             cast<AtomicCmpXchgInst>(R)->isWeak()))
      return Res;
    if (int Res = cmpNumbers(CXI->getSuccessOrdering(),
                             cast<AtomicCmpXchgInst>(R)->getSuccessOrdering()))
      return Res;
    if (int Res = cmpNumbers(CXI->getFailureOrdering(),
                             cast<AtomicCmpXchgInst>(R)->getFailureOrdering()))
      return Res;
    return cmpNumbers(CXI->getSynchScope(),
                      cast<AtomicCmpXchgInst>(R)->getSynchScope());
  }
  if (const AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(L)) {
    if (int Res = cmpNumbers(RMWI->getOperation(),
                             cast<AtomicRMWInst>(R)->getOperation()))
      return Res;
    if (int Res = cmpNumbers(RMWI->isVolatile(),
                             cast<AtomicRMWInst>(R)->isVolatile()))
      return Res;
    if (int Res = cmpNumbers(RMWI->getOrdering(),
                             cast<AtomicRMWInst>(R)->getOrdering()))
      return Res;
    return cmpNumbers(RMWI->getSynchScope(),
                      cast<AtomicRMWInst>(R)->getSynchScope());
  }
  return 0;
}

// Determine whether two GEP operations perform the same underlying arithmetic.
// Read method declaration comments for more details.
int FunctionComparator::cmpGEPs(const GEPOperator *GEPL,
                               const GEPOperator *GEPR) {

  unsigned int ASL = GEPL->getPointerAddressSpace();
  unsigned int ASR = GEPR->getPointerAddressSpace();

  if (int Res = cmpNumbers(ASL, ASR))
    return Res;

  // When we have target data, we can reduce the GEP down to the value in bytes
  // added to the address.
  const DataLayout &DL = FnL->getParent()->getDataLayout();
  unsigned BitWidth = DL.getPointerSizeInBits(ASL);
  APInt OffsetL(BitWidth, 0), OffsetR(BitWidth, 0);
  if (GEPL->accumulateConstantOffset(DL, OffsetL) &&
      GEPR->accumulateConstantOffset(DL, OffsetR))
    return cmpAPInts(OffsetL, OffsetR);

  if (int Res = cmpNumbers((uint64_t)GEPL->getPointerOperand()->getType(),
                           (uint64_t)GEPR->getPointerOperand()->getType()))
    return Res;

  if (int Res = cmpNumbers(GEPL->getNumOperands(), GEPR->getNumOperands()))
    return Res;

  for (unsigned i = 0, e = GEPL->getNumOperands(); i != e; ++i) {
    if (int Res = cmpValues(GEPL->getOperand(i), GEPR->getOperand(i)))
      return Res;
  }

  return 0;
}

/// Compare two values used by the two functions under pair-wise comparison. If
/// this is the first time the values are seen, they're added to the mapping so
/// that we will detect mismatches on next use.
/// See comments in declaration for more details.
int FunctionComparator::cmpValues(const Value *L, const Value *R) {
  // Catch self-reference case.
  if (L == FnL) {
    if (R == FnR)
      return 0;
    return -1;
  }
  if (R == FnR) {
    if (L == FnL)
      return 0;
    return 1;
  }

  const Constant *ConstL = dyn_cast<Constant>(L);
  const Constant *ConstR = dyn_cast<Constant>(R);
  if (ConstL && ConstR) {
    if (L == R)
      return 0;
    return cmpConstants(ConstL, ConstR);
  }

  if (ConstL)
    return 1;
  if (ConstR)
    return -1;

  const InlineAsm *InlineAsmL = dyn_cast<InlineAsm>(L);
  const InlineAsm *InlineAsmR = dyn_cast<InlineAsm>(R);

  if (InlineAsmL && InlineAsmR)
    return cmpNumbers((uint64_t)L, (uint64_t)R);
  if (InlineAsmL)
    return 1;
  if (InlineAsmR)
    return -1;

  auto LeftSN = sn_mapL.insert(std::make_pair(L, sn_mapL.size())),
       RightSN = sn_mapR.insert(std::make_pair(R, sn_mapR.size()));

  return cmpNumbers(LeftSN.first->second, RightSN.first->second);
}
// Test whether two basic blocks have equivalent behaviour.
int FunctionComparator::compare(const BasicBlock *BBL, const BasicBlock *BBR) {
  BasicBlock::const_iterator InstL = BBL->begin(), InstLE = BBL->end();
  BasicBlock::const_iterator InstR = BBR->begin(), InstRE = BBR->end();

  do {
    if (int Res = cmpValues(InstL, InstR))
      return Res;

    const GetElementPtrInst *GEPL = dyn_cast<GetElementPtrInst>(InstL);
    const GetElementPtrInst *GEPR = dyn_cast<GetElementPtrInst>(InstR);

    if (GEPL && !GEPR)
      return 1;
    if (GEPR && !GEPL)
      return -1;

    if (GEPL && GEPR) {
      if (int Res =
              cmpValues(GEPL->getPointerOperand(), GEPR->getPointerOperand()))
        return Res;
      if (int Res = cmpGEPs(GEPL, GEPR))
        return Res;
    } else {
      if (int Res = cmpOperations(InstL, InstR))
        return Res;
      assert(InstL->getNumOperands() == InstR->getNumOperands());

      for (unsigned i = 0, e = InstL->getNumOperands(); i != e; ++i) {
        Value *OpL = InstL->getOperand(i);
        Value *OpR = InstR->getOperand(i);
        if (int Res = cmpValues(OpL, OpR))
          return Res;
        if (int Res = cmpNumbers(OpL->getValueID(), OpR->getValueID()))
          return Res;
        // TODO: Already checked in cmpOperation
        if (int Res = cmpTypes(OpL->getType(), OpR->getType()))
          return Res;
      }
    }

    ++InstL, ++InstR;
  } while (InstL != InstLE && InstR != InstRE);

  if (InstL != InstLE && InstR == InstRE)
    return 1;
  if (InstL == InstLE && InstR != InstRE)
    return -1;
  return 0;
}

// Test whether the two functions have equivalent behaviour.
int FunctionComparator::compare() {

  sn_mapL.clear();
  sn_mapR.clear();

  if (int Res = cmpAttrs(FnL->getAttributes(), FnR->getAttributes()))
    return Res;

  if (int Res = cmpNumbers(FnL->hasGC(), FnR->hasGC()))
    return Res;

  if (FnL->hasGC()) {
    if (int Res = cmpNumbers((uint64_t)FnL->getGC(), (uint64_t)FnR->getGC()))
      return Res;
  }

  if (int Res = cmpNumbers(FnL->hasSection(), FnR->hasSection()))
    return Res;

  if (FnL->hasSection()) {
    if (int Res = cmpStrings(FnL->getSection(), FnR->getSection()))
      return Res;
  }

  if (int Res = cmpNumbers(FnL->isVarArg(), FnR->isVarArg()))
    return Res;

  // TODO: if it's internal and only used in direct calls, we could handle this
  // case too.
  if (int Res = cmpNumbers(FnL->getCallingConv(), FnR->getCallingConv()))
    return Res;

  if (int Res = cmpTypes(FnL->getFunctionType(), FnR->getFunctionType()))
    return Res;

  assert(FnL->arg_size() == FnR->arg_size() &&
         "Identically typed functions have different numbers of args!");

  // Visit the arguments so that they get enumerated in the order they're
  // passed in.
  for (Function::const_arg_iterator ArgLI = FnL->arg_begin(),
                                    ArgRI = FnR->arg_begin(),
                                    ArgLE = FnL->arg_end();
       ArgLI != ArgLE; ++ArgLI, ++ArgRI) {
    if (cmpValues(ArgLI, ArgRI) != 0)
      llvm_unreachable("Arguments repeat!");
  }

  // We do a CFG-ordered walk since the actual ordering of the blocks in the
  // linked list is immaterial. Our walk starts at the entry block for both
  // functions, then takes each block from each terminator in order. As an
  // artifact, this also means that unreachable blocks are ignored.
  SmallVector<const BasicBlock *, 8> FnLBBs, FnRBBs;
  SmallSet<const BasicBlock *, 128> VisitedBBs; // in terms of F1.

  FnLBBs.push_back(&FnL->getEntryBlock());
  FnRBBs.push_back(&FnR->getEntryBlock());

  VisitedBBs.insert(FnLBBs[0]);
  while (!FnLBBs.empty()) {
    const BasicBlock *BBL = FnLBBs.pop_back_val();
    const BasicBlock *BBR = FnRBBs.pop_back_val();

    if (int Res = cmpValues(BBL, BBR))
      return Res;

    if (int Res = compare(BBL, BBR))
      return Res;

    const TerminatorInst *TermL = BBL->getTerminator();
    const TerminatorInst *TermR = BBR->getTerminator();

    assert(TermL->getNumSuccessors() == TermR->getNumSuccessors());
    for (unsigned i = 0, e = TermL->getNumSuccessors(); i != e; ++i) {
      if (!VisitedBBs.insert(TermL->getSuccessor(i)).second)
        continue;

      FnLBBs.push_back(TermL->getSuccessor(i));
      FnRBBs.push_back(TermR->getSuccessor(i));
    }
  }
  return 0;
}

namespace {

/// MergeFunctions finds functions which will generate identical machine code,
/// by considering all pointer types to be equivalent. Once identified,
/// MergeFunctions will fold them by replacing a call to one to a call to a
/// bitcast of the other.
///
class MergeFunctions : public ModulePass {
public:
  static char ID;
  MergeFunctions()
    : ModulePass(ID), HasGlobalAliases(false) {
    initializeMergeFunctionsPass(*PassRegistry::getPassRegistry());
  }

  bool runOnModule(Module &M) override;

private:
  typedef std::set<FunctionNode> FnTreeType;

  /// A work queue of functions that may have been modified and should be
  /// analyzed again.
  std::vector<WeakVH> Deferred;

  /// Checks the rules of order relation introduced among functions set.
  /// Returns true, if sanity check has been passed, and false if failed.
  bool doSanityCheck(std::vector<WeakVH> &Worklist);

  /// Insert a ComparableFunction into the FnTree, or merge it away if it's
  /// equal to one that's already present.
  bool insert(Function *NewFunction);

  /// Remove a Function from the FnTree and queue it up for a second sweep of
  /// analysis.
  void remove(Function *F);

  /// Find the functions that use this Value and remove them from FnTree and
  /// queue the functions.
  void removeUsers(Value *V);

  /// Replace all direct calls of Old with calls of New. Will bitcast New if
  /// necessary to make types match.
  void replaceDirectCallers(Function *Old, Function *New);

  /// Merge two equivalent functions. Upon completion, G may be deleted, or may
  /// be converted into a thunk. In either case, it should never be visited
  /// again.
  void mergeTwoFunctions(Function *F, Function *G);

  /// Replace G with a thunk or an alias to F. Deletes G.
  void writeThunkOrAlias(Function *F, Function *G);

  /// Replace G with a simple tail call to bitcast(F). Also replace direct uses
  /// of G with bitcast(F). Deletes G.
  void writeThunk(Function *F, Function *G);

  /// Replace G with an alias to F. Deletes G.
  void writeAlias(Function *F, Function *G);

  /// Replace function F with function G in the function tree.
  void replaceFunctionInTree(FnTreeType::iterator &IterToF, Function *G);

  /// The set of all distinct functions. Use the insert() and remove() methods
  /// to modify it.
  FnTreeType FnTree;

  /// Whether or not the target supports global aliases.
  bool HasGlobalAliases;
};

}  // end anonymous namespace

char MergeFunctions::ID = 0;
INITIALIZE_PASS(MergeFunctions, "mergefunc", "Merge Functions", false, false)

ModulePass *llvm::createMergeFunctionsPass() {
  return new MergeFunctions();
}

bool MergeFunctions::doSanityCheck(std::vector<WeakVH> &Worklist) {
#if 0 // Begin HLSL Change (NumFunctionsForSanityCheck is always zero)
  if (const unsigned Max = NumFunctionsForSanityCheck) {
    unsigned TripleNumber = 0;
    bool Valid = true;

    dbgs() << "MERGEFUNC-SANITY: Started for first " << Max << " functions.\n";

    unsigned i = 0;
    for (std::vector<WeakVH>::iterator I = Worklist.begin(), E = Worklist.end();
         I != E && i < Max; ++I, ++i) {
      unsigned j = i;
      for (std::vector<WeakVH>::iterator J = I; J != E && j < Max; ++J, ++j) {
        Function *F1 = cast<Function>(*I);
        Function *F2 = cast<Function>(*J);
        int Res1 = FunctionComparator(F1, F2).compare();
        int Res2 = FunctionComparator(F2, F1).compare();

        // If F1 <= F2, then F2 >= F1, otherwise report failure.
        if (Res1 != -Res2) {
          dbgs() << "MERGEFUNC-SANITY: Non-symmetric; triple: " << TripleNumber
                 << "\n";
          F1->dump();
          F2->dump();
          Valid = false;
        }

        if (Res1 == 0)
          continue;

        unsigned k = j;
        for (std::vector<WeakVH>::iterator K = J; K != E && k < Max;
             ++k, ++K, ++TripleNumber) {
          if (K == J)
            continue;

          Function *F3 = cast<Function>(*K);
          int Res3 = FunctionComparator(F1, F3).compare();
          int Res4 = FunctionComparator(F2, F3).compare();

          bool Transitive = true;

          if (Res1 != 0 && Res1 == Res4) {
            // F1 > F2, F2 > F3 => F1 > F3
            Transitive = Res3 == Res1;
          } else if (Res3 != 0 && Res3 == -Res4) {
            // F1 > F3, F3 > F2 => F1 > F2
            Transitive = Res3 == Res1;
          } else if (Res4 != 0 && -Res3 == Res4) {
            // F2 > F3, F3 > F1 => F2 > F1
            Transitive = Res4 == -Res1;
          }

          if (!Transitive) {
            dbgs() << "MERGEFUNC-SANITY: Non-transitive; triple: "
                   << TripleNumber << "\n";
            dbgs() << "Res1, Res3, Res4: " << Res1 << ", " << Res3 << ", "
                   << Res4 << "\n";
            F1->dump();
            F2->dump();
            F3->dump();
            Valid = false;
          }
        }
      }
    }

    dbgs() << "MERGEFUNC-SANITY: " << (Valid ? "Passed." : "Failed.") << "\n";
    return Valid;
  }
#endif // End HLSL Change
  return true;
}

bool MergeFunctions::runOnModule(Module &M) {
  bool Changed = false;

  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    if (!I->isDeclaration() && !I->hasAvailableExternallyLinkage())
      Deferred.push_back(WeakVH(I));
  }

  do {
    std::vector<WeakVH> Worklist;
    Deferred.swap(Worklist);

    DEBUG(doSanityCheck(Worklist));

    DEBUG(dbgs() << "size of module: " << M.size() << '\n');
    DEBUG(dbgs() << "size of worklist: " << Worklist.size() << '\n');

    // Insert only strong functions and merge them. Strong function merging
    // always deletes one of them.
    for (std::vector<WeakVH>::iterator I = Worklist.begin(),
           E = Worklist.end(); I != E; ++I) {
      if (!*I) continue;
      Function *F = cast<Function>(*I);
      if (!F->isDeclaration() && !F->hasAvailableExternallyLinkage() &&
          !F->mayBeOverridden()) {
        Changed |= insert(F);
      }
    }

    // Insert only weak functions and merge them. By doing these second we
    // create thunks to the strong function when possible. When two weak
    // functions are identical, we create a new strong function with two weak
    // weak thunks to it which are identical but not mergable.
    for (std::vector<WeakVH>::iterator I = Worklist.begin(),
           E = Worklist.end(); I != E; ++I) {
      if (!*I) continue;
      Function *F = cast<Function>(*I);
      if (!F->isDeclaration() && !F->hasAvailableExternallyLinkage() &&
          F->mayBeOverridden()) {
        Changed |= insert(F);
      }
    }
    DEBUG(dbgs() << "size of FnTree: " << FnTree.size() << '\n');
  } while (!Deferred.empty());

  FnTree.clear();

  return Changed;
}

// Replace direct callers of Old with New.
void MergeFunctions::replaceDirectCallers(Function *Old, Function *New) {
  Constant *BitcastNew = ConstantExpr::getBitCast(New, Old->getType());
  for (auto UI = Old->use_begin(), UE = Old->use_end(); UI != UE;) {
    Use *U = &*UI;
    ++UI;
    CallSite CS(U->getUser());
    if (CS && CS.isCallee(U)) {
      remove(CS.getInstruction()->getParent()->getParent());
      U->set(BitcastNew);
    }
  }
}

// Replace G with an alias to F if possible, or else a thunk to F. Deletes G.
void MergeFunctions::writeThunkOrAlias(Function *F, Function *G) {
  if (HasGlobalAliases && G->hasUnnamedAddr()) {
    if (G->hasExternalLinkage() || G->hasLocalLinkage() ||
        G->hasWeakLinkage()) {
      writeAlias(F, G);
      return;
    }
  }

  writeThunk(F, G);
}

// Helper for writeThunk,
// Selects proper bitcast operation,
// but a bit simpler then CastInst::getCastOpcode.
static Value *createCast(IRBuilder<false> &Builder, Value *V, Type *DestTy) {
  Type *SrcTy = V->getType();
  if (SrcTy->isStructTy()) {
    assert(DestTy->isStructTy());
    assert(SrcTy->getStructNumElements() == DestTy->getStructNumElements());
    Value *Result = UndefValue::get(DestTy);
    for (unsigned int I = 0, E = SrcTy->getStructNumElements(); I < E; ++I) {
      Value *Element = createCast(
          Builder, Builder.CreateExtractValue(V, makeArrayRef(I)),
          DestTy->getStructElementType(I));

      Result =
          Builder.CreateInsertValue(Result, Element, makeArrayRef(I));
    }
    return Result;
  }
  assert(!DestTy->isStructTy());
  if (SrcTy->isIntegerTy() && DestTy->isPointerTy())
    return Builder.CreateIntToPtr(V, DestTy);
  else if (SrcTy->isPointerTy() && DestTy->isIntegerTy())
    return Builder.CreatePtrToInt(V, DestTy);
  else
    return Builder.CreateBitCast(V, DestTy);
}

// Replace G with a simple tail call to bitcast(F). Also replace direct uses
// of G with bitcast(F). Deletes G.
void MergeFunctions::writeThunk(Function *F, Function *G) {
  if (!G->mayBeOverridden()) {
    // Redirect direct callers of G to F.
    replaceDirectCallers(G, F);
  }

  // If G was internal then we may have replaced all uses of G with F. If so,
  // stop here and delete G. There's no need for a thunk.
  if (G->hasLocalLinkage() && G->use_empty()) {
    G->eraseFromParent();
    return;
  }

  Function *NewG = Function::Create(G->getFunctionType(), G->getLinkage(), "",
                                    G->getParent());
  BasicBlock *BB = BasicBlock::Create(F->getContext(), "", NewG);
  IRBuilder<false> Builder(BB);

  SmallVector<Value *, 16> Args;
  unsigned i = 0;
  FunctionType *FFTy = F->getFunctionType();
  for (Function::arg_iterator AI = NewG->arg_begin(), AE = NewG->arg_end();
       AI != AE; ++AI) {
    Args.push_back(createCast(Builder, (Value*)AI, FFTy->getParamType(i)));
    ++i;
  }

  CallInst *CI = Builder.CreateCall(F, Args);
  CI->setTailCall();
  CI->setCallingConv(F->getCallingConv());
  if (NewG->getReturnType()->isVoidTy()) {
    Builder.CreateRetVoid();
  } else {
    Builder.CreateRet(createCast(Builder, CI, NewG->getReturnType()));
  }

  NewG->copyAttributesFrom(G);
  NewG->takeName(G);
  removeUsers(G);
  G->replaceAllUsesWith(NewG);
  G->eraseFromParent();

  DEBUG(dbgs() << "writeThunk: " << NewG->getName() << '\n');
  ++NumThunksWritten;
}

// Replace G with an alias to F and delete G.
void MergeFunctions::writeAlias(Function *F, Function *G) {
  PointerType *PTy = G->getType();
  auto *GA = GlobalAlias::create(PTy, G->getLinkage(), "", F);
  F->setAlignment(std::max(F->getAlignment(), G->getAlignment()));
  GA->takeName(G);
  GA->setVisibility(G->getVisibility());
  removeUsers(G);
  G->replaceAllUsesWith(GA);
  G->eraseFromParent();

  DEBUG(dbgs() << "writeAlias: " << GA->getName() << '\n');
  ++NumAliasesWritten;
}

// Merge two equivalent functions. Upon completion, Function G is deleted.
void MergeFunctions::mergeTwoFunctions(Function *F, Function *G) {
  if (F->mayBeOverridden()) {
    assert(G->mayBeOverridden());

    // Make them both thunks to the same internal function.
    Function *H = Function::Create(F->getFunctionType(), F->getLinkage(), "",
                                   F->getParent());
    H->copyAttributesFrom(F);
    H->takeName(F);
    removeUsers(F);
    F->replaceAllUsesWith(H);

    unsigned MaxAlignment = std::max(G->getAlignment(), H->getAlignment());

    if (HasGlobalAliases) {
      writeAlias(F, G);
      writeAlias(F, H);
    } else {
      writeThunk(F, G);
      writeThunk(F, H);
    }

    F->setAlignment(MaxAlignment);
    F->setLinkage(GlobalValue::PrivateLinkage);
    ++NumDoubleWeak;
  } else {
    writeThunkOrAlias(F, G);
  }

  ++NumFunctionsMerged;
}

/// Replace function F for function G in the map.
void MergeFunctions::replaceFunctionInTree(FnTreeType::iterator &IterToF,
                                           Function *G) {
  Function *F = IterToF->getFunc();

  // A total order is already guaranteed otherwise because we process strong
  // functions before weak functions.
  assert(((F->mayBeOverridden() && G->mayBeOverridden()) ||
          (!F->mayBeOverridden() && !G->mayBeOverridden())) &&
         "Only change functions if both are strong or both are weak");
  (void)F;

  IterToF->replaceBy(G);
}

// Insert a ComparableFunction into the FnTree, or merge it away if equal to one
// that was already inserted.
bool MergeFunctions::insert(Function *NewFunction) {
  std::pair<FnTreeType::iterator, bool> Result =
      FnTree.insert(FunctionNode(NewFunction));

  if (Result.second) {
    DEBUG(dbgs() << "Inserting as unique: " << NewFunction->getName() << '\n');
    return false;
  }

  const FunctionNode &OldF = *Result.first;

  // Don't merge tiny functions, since it can just end up making the function
  // larger.
  // FIXME: Should still merge them if they are unnamed_addr and produce an
  // alias.
  if (NewFunction->size() == 1) {
    if (NewFunction->front().size() <= 2) {
      DEBUG(dbgs() << NewFunction->getName()
                   << " is to small to bother merging\n");
      return false;
    }
  }

  // Impose a total order (by name) on the replacement of functions. This is
  // important when operating on more than one module independently to prevent
  // cycles of thunks calling each other when the modules are linked together.
  //
  // When one function is weak and the other is strong there is an order imposed
  // already. We process strong functions before weak functions.
  if ((OldF.getFunc()->mayBeOverridden() && NewFunction->mayBeOverridden()) ||
      (!OldF.getFunc()->mayBeOverridden() && !NewFunction->mayBeOverridden()))
    if (OldF.getFunc()->getName() > NewFunction->getName()) {
      // Swap the two functions.
      Function *F = OldF.getFunc();
      replaceFunctionInTree(Result.first, NewFunction);
      NewFunction = F;
      assert(OldF.getFunc() != F && "Must have swapped the functions.");
    }

  // Never thunk a strong function to a weak function.
  assert(!OldF.getFunc()->mayBeOverridden() || NewFunction->mayBeOverridden());

  DEBUG(dbgs() << "  " << OldF.getFunc()->getName()
               << " == " << NewFunction->getName() << '\n');

  Function *DeleteF = NewFunction;
  mergeTwoFunctions(OldF.getFunc(), DeleteF);
  return true;
}

// Remove a function from FnTree. If it was already in FnTree, add
// it to Deferred so that we'll look at it in the next round.
void MergeFunctions::remove(Function *F) {
  // We need to make sure we remove F, not a function "equal" to F per the
  // function equality comparator.
  FnTreeType::iterator found = FnTree.find(FunctionNode(F));
  size_t Erased = 0;
  if (found != FnTree.end() && found->getFunc() == F) {
    Erased = 1;
    FnTree.erase(found);
  }

  if (Erased) {
    DEBUG(dbgs() << "Removed " << F->getName()
                 << " from set and deferred it.\n");
    Deferred.emplace_back(F);
  }
}

// For each instruction used by the value, remove() the function that contains
// the instruction. This should happen right before a call to RAUW.
void MergeFunctions::removeUsers(Value *V) {
  std::vector<Value *> Worklist;
  Worklist.push_back(V);
  while (!Worklist.empty()) {
    Value *V = Worklist.back();
    Worklist.pop_back();

    for (User *U : V->users()) {
      if (Instruction *I = dyn_cast<Instruction>(U)) {
        remove(I->getParent()->getParent());
      } else if (isa<GlobalValue>(U)) {
        // do nothing
      } else if (Constant *C = dyn_cast<Constant>(U)) {
        for (User *UU : C->users())
          Worklist.push_back(UU);
      }
    }
  }
}
