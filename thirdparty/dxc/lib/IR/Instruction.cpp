//===-- Instruction.cpp - Implement the Instruction class -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Instruction class for the IR library.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Instruction.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/Type.h"
using namespace llvm;

Instruction::Instruction(Type *ty, unsigned it, Use *Ops, unsigned NumOps,
                         Instruction *InsertBefore)
  : User(ty, Value::InstructionVal + it, Ops, NumOps), Parent(nullptr) {

  // If requested, insert this instruction into a basic block...
  if (InsertBefore) {
    BasicBlock *BB = InsertBefore->getParent();
    assert(BB && "Instruction to insert before is not in a basic block!");
    BB->getInstList().insert(InsertBefore, this);
  }
}

Instruction::Instruction(Type *ty, unsigned it, Use *Ops, unsigned NumOps,
                         BasicBlock *InsertAtEnd)
  : User(ty, Value::InstructionVal + it, Ops, NumOps), Parent(nullptr) {

  // append this instruction into the basic block
  assert(InsertAtEnd && "Basic block to append to may not be NULL!");
  InsertAtEnd->getInstList().push_back(this);
}


// Out of line virtual method, so the vtable, etc has a home.
Instruction::~Instruction() {
  assert(!Parent && "Instruction still linked in the program!");
  if (hasMetadataHashEntry())
    clearMetadataHashEntries();
}


void Instruction::setParent(BasicBlock *P) {
  Parent = P;
}

const Module *Instruction::getModule() const {
  return getParent()->getModule();
}

Module *Instruction::getModule() {
  return getParent()->getModule();
}


void Instruction::removeFromParent() {
  getParent()->getInstList().remove(this);
}

iplist<Instruction>::iterator Instruction::eraseFromParent() {
  return getParent()->getInstList().erase(this);
}

/// insertBefore - Insert an unlinked instructions into a basic block
/// immediately before the specified instruction.
void Instruction::insertBefore(Instruction *InsertPos) {
  InsertPos->getParent()->getInstList().insert(InsertPos, this);
}

/// insertAfter - Insert an unlinked instructions into a basic block
/// immediately after the specified instruction.
void Instruction::insertAfter(Instruction *InsertPos) {
  InsertPos->getParent()->getInstList().insertAfter(InsertPos, this);
}

/// moveBefore - Unlink this instruction from its current basic block and
/// insert it into the basic block that MovePos lives in, right before
/// MovePos.
void Instruction::moveBefore(Instruction *MovePos) {
  MovePos->getParent()->getInstList().splice(MovePos,getParent()->getInstList(),
                                             this);
}

/// Set or clear the unsafe-algebra flag on this instruction, which must be an
/// operator which supports this flag. See LangRef.html for the meaning of this
/// flag.
void Instruction::setHasUnsafeAlgebra(bool B) {
  assert(isa<FPMathOperator>(this) && "setting fast-math flag on invalid op");
  cast<FPMathOperator>(this)->setHasUnsafeAlgebra(B);
}

/// Set or clear the NoNaNs flag on this instruction, which must be an operator
/// which supports this flag. See LangRef.html for the meaning of this flag.
void Instruction::setHasNoNaNs(bool B) {
  assert(isa<FPMathOperator>(this) && "setting fast-math flag on invalid op");
  cast<FPMathOperator>(this)->setHasNoNaNs(B);
}

/// Set or clear the no-infs flag on this instruction, which must be an operator
/// which supports this flag. See LangRef.html for the meaning of this flag.
void Instruction::setHasNoInfs(bool B) {
  assert(isa<FPMathOperator>(this) && "setting fast-math flag on invalid op");
  cast<FPMathOperator>(this)->setHasNoInfs(B);
}

/// Set or clear the no-signed-zeros flag on this instruction, which must be an
/// operator which supports this flag. See LangRef.html for the meaning of this
/// flag.
void Instruction::setHasNoSignedZeros(bool B) {
  assert(isa<FPMathOperator>(this) && "setting fast-math flag on invalid op");
  cast<FPMathOperator>(this)->setHasNoSignedZeros(B);
}

/// Set or clear the allow-reciprocal flag on this instruction, which must be an
/// operator which supports this flag. See LangRef.html for the meaning of this
/// flag.
void Instruction::setHasAllowReciprocal(bool B) {
  assert(isa<FPMathOperator>(this) && "setting fast-math flag on invalid op");
  cast<FPMathOperator>(this)->setHasAllowReciprocal(B);
}

/// Convenience function for setting all the fast-math flags on this
/// instruction, which must be an operator which supports these flags. See
/// LangRef.html for the meaning of these flats.
void Instruction::setFastMathFlags(FastMathFlags FMF) {
  assert(isa<FPMathOperator>(this) && "setting fast-math flag on invalid op");
  cast<FPMathOperator>(this)->setFastMathFlags(FMF);
}

void Instruction::copyFastMathFlags(FastMathFlags FMF) {
  assert(isa<FPMathOperator>(this) && "copying fast-math flag on invalid op");
  cast<FPMathOperator>(this)->copyFastMathFlags(FMF);
}

/// Determine whether the unsafe-algebra flag is set.
bool Instruction::hasUnsafeAlgebra() const {
  assert(isa<FPMathOperator>(this) && "getting fast-math flag on invalid op");
  return cast<FPMathOperator>(this)->hasUnsafeAlgebra();
}

/// Determine whether the no-NaNs flag is set.
bool Instruction::hasNoNaNs() const {
  assert(isa<FPMathOperator>(this) && "getting fast-math flag on invalid op");
  return cast<FPMathOperator>(this)->hasNoNaNs();
}

/// Determine whether the no-infs flag is set.
bool Instruction::hasNoInfs() const {
  assert(isa<FPMathOperator>(this) && "getting fast-math flag on invalid op");
  return cast<FPMathOperator>(this)->hasNoInfs();
}

/// Determine whether the no-signed-zeros flag is set.
bool Instruction::hasNoSignedZeros() const {
  assert(isa<FPMathOperator>(this) && "getting fast-math flag on invalid op");
  return cast<FPMathOperator>(this)->hasNoSignedZeros();
}

/// Determine whether the allow-reciprocal flag is set.
bool Instruction::hasAllowReciprocal() const {
  assert(isa<FPMathOperator>(this) && "getting fast-math flag on invalid op");
  return cast<FPMathOperator>(this)->hasAllowReciprocal();
}

/// Convenience function for getting all the fast-math flags, which must be an
/// operator which supports these flags. See LangRef.html for the meaning of
/// these flags.
FastMathFlags Instruction::getFastMathFlags() const {
  assert(isa<FPMathOperator>(this) && "getting fast-math flag on invalid op");
  return cast<FPMathOperator>(this)->getFastMathFlags();
}

/// Copy I's fast-math flags
void Instruction::copyFastMathFlags(const Instruction *I) {
  copyFastMathFlags(I->getFastMathFlags());
}


const char *Instruction::getOpcodeName(unsigned OpCode) {
  switch (OpCode) {
  // Terminators
  case Ret:    return "ret";
  case Br:     return "br";
  case Switch: return "switch";
  case IndirectBr: return "indirectbr";
  case Invoke: return "invoke";
  case Resume: return "resume";
  case Unreachable: return "unreachable";

  // Standard binary operators...
  case Add: return "add";
  case FAdd: return "fadd";
  case Sub: return "sub";
  case FSub: return "fsub";
  case Mul: return "mul";
  case FMul: return "fmul";
  case UDiv: return "udiv";
  case SDiv: return "sdiv";
  case FDiv: return "fdiv";
  case URem: return "urem";
  case SRem: return "srem";
  case FRem: return "frem";

  // Logical operators...
  case And: return "and";
  case Or : return "or";
  case Xor: return "xor";

  // Memory instructions...
  case Alloca:        return "alloca";
  case Load:          return "load";
  case Store:         return "store";
  case AtomicCmpXchg: return "cmpxchg";
  case AtomicRMW:     return "atomicrmw";
  case Fence:         return "fence";
  case GetElementPtr: return "getelementptr";

  // Convert instructions...
  case Trunc:         return "trunc";
  case ZExt:          return "zext";
  case SExt:          return "sext";
  case FPTrunc:       return "fptrunc";
  case FPExt:         return "fpext";
  case FPToUI:        return "fptoui";
  case FPToSI:        return "fptosi";
  case UIToFP:        return "uitofp";
  case SIToFP:        return "sitofp";
  case IntToPtr:      return "inttoptr";
  case PtrToInt:      return "ptrtoint";
  case BitCast:       return "bitcast";
  case AddrSpaceCast: return "addrspacecast";

  // Other instructions...
  case ICmp:           return "icmp";
  case FCmp:           return "fcmp";
  case PHI:            return "phi";
  case Select:         return "select";
  case Call:           return "call";
  case Shl:            return "shl";
  case LShr:           return "lshr";
  case AShr:           return "ashr";
  case VAArg:          return "va_arg";
  case ExtractElement: return "extractelement";
  case InsertElement:  return "insertelement";
  case ShuffleVector:  return "shufflevector";
  case ExtractValue:   return "extractvalue";
  case InsertValue:    return "insertvalue";
  case LandingPad:     return "landingpad";

  default: return "<Invalid operator> ";
  }
}

/// Return true if both instructions have the same special state
/// This must be kept in sync with lib/Transforms/IPO/MergeFunctions.cpp.
static bool haveSameSpecialState(const Instruction *I1, const Instruction *I2,
                                 bool IgnoreAlignment = false) {
  assert(I1->getOpcode() == I2->getOpcode() &&
         "Can not compare special state of different instructions");

  if (const LoadInst *LI = dyn_cast<LoadInst>(I1))
    return LI->isVolatile() == cast<LoadInst>(I2)->isVolatile() &&
           (LI->getAlignment() == cast<LoadInst>(I2)->getAlignment() ||
            IgnoreAlignment) &&
           LI->getOrdering() == cast<LoadInst>(I2)->getOrdering() &&
           LI->getSynchScope() == cast<LoadInst>(I2)->getSynchScope();
  if (const StoreInst *SI = dyn_cast<StoreInst>(I1))
    return SI->isVolatile() == cast<StoreInst>(I2)->isVolatile() &&
           (SI->getAlignment() == cast<StoreInst>(I2)->getAlignment() ||
            IgnoreAlignment) &&
           SI->getOrdering() == cast<StoreInst>(I2)->getOrdering() &&
           SI->getSynchScope() == cast<StoreInst>(I2)->getSynchScope();
  if (const CmpInst *CI = dyn_cast<CmpInst>(I1))
    return CI->getPredicate() == cast<CmpInst>(I2)->getPredicate();
  if (const CallInst *CI = dyn_cast<CallInst>(I1))
    return CI->isTailCall() == cast<CallInst>(I2)->isTailCall() &&
           CI->getCallingConv() == cast<CallInst>(I2)->getCallingConv() &&
           CI->getAttributes() == cast<CallInst>(I2)->getAttributes();
  if (const InvokeInst *CI = dyn_cast<InvokeInst>(I1))
    return CI->getCallingConv() == cast<InvokeInst>(I2)->getCallingConv() &&
           CI->getAttributes() ==
             cast<InvokeInst>(I2)->getAttributes();
  if (const InsertValueInst *IVI = dyn_cast<InsertValueInst>(I1))
    return IVI->getIndices() == cast<InsertValueInst>(I2)->getIndices();
  if (const ExtractValueInst *EVI = dyn_cast<ExtractValueInst>(I1))
    return EVI->getIndices() == cast<ExtractValueInst>(I2)->getIndices();
  if (const FenceInst *FI = dyn_cast<FenceInst>(I1))
    return FI->getOrdering() == cast<FenceInst>(I2)->getOrdering() &&
           FI->getSynchScope() == cast<FenceInst>(I2)->getSynchScope();
  if (const AtomicCmpXchgInst *CXI = dyn_cast<AtomicCmpXchgInst>(I1))
    return CXI->isVolatile() == cast<AtomicCmpXchgInst>(I2)->isVolatile() &&
           CXI->isWeak() == cast<AtomicCmpXchgInst>(I2)->isWeak() &&
           CXI->getSuccessOrdering() ==
               cast<AtomicCmpXchgInst>(I2)->getSuccessOrdering() &&
           CXI->getFailureOrdering() ==
               cast<AtomicCmpXchgInst>(I2)->getFailureOrdering() &&
           CXI->getSynchScope() == cast<AtomicCmpXchgInst>(I2)->getSynchScope();
  if (const AtomicRMWInst *RMWI = dyn_cast<AtomicRMWInst>(I1))
    return RMWI->getOperation() == cast<AtomicRMWInst>(I2)->getOperation() &&
           RMWI->isVolatile() == cast<AtomicRMWInst>(I2)->isVolatile() &&
           RMWI->getOrdering() == cast<AtomicRMWInst>(I2)->getOrdering() &&
           RMWI->getSynchScope() == cast<AtomicRMWInst>(I2)->getSynchScope();

  return true;
}

/// isIdenticalTo - Return true if the specified instruction is exactly
/// identical to the current one.  This means that all operands match and any
/// extra information (e.g. load is volatile) agree.
bool Instruction::isIdenticalTo(const Instruction *I) const {
  return isIdenticalToWhenDefined(I) &&
         SubclassOptionalData == I->SubclassOptionalData;
}

/// isIdenticalToWhenDefined - This is like isIdenticalTo, except that it
/// ignores the SubclassOptionalData flags, which specify conditions
/// under which the instruction's result is undefined.
bool Instruction::isIdenticalToWhenDefined(const Instruction *I) const {
  if (getOpcode() != I->getOpcode() ||
      getNumOperands() != I->getNumOperands() ||
      getType() != I->getType())
    return false;

  // If both instructions have no operands, they are identical.
  if (getNumOperands() == 0 && I->getNumOperands() == 0)
    return haveSameSpecialState(this, I);

  // We have two instructions of identical opcode and #operands.  Check to see
  // if all operands are the same.
  if (!std::equal(op_begin(), op_end(), I->op_begin()))
    return false;

  if (const PHINode *thisPHI = dyn_cast<PHINode>(this)) {
    const PHINode *otherPHI = cast<PHINode>(I);
    return std::equal(thisPHI->block_begin(), thisPHI->block_end(),
                      otherPHI->block_begin());
  }

  return haveSameSpecialState(this, I);
}

// isSameOperationAs
// This should be kept in sync with isEquivalentOperation in
// lib/Transforms/IPO/MergeFunctions.cpp.
bool Instruction::isSameOperationAs(const Instruction *I,
                                    unsigned flags) const {
  bool IgnoreAlignment = flags & CompareIgnoringAlignment;
  bool UseScalarTypes  = flags & CompareUsingScalarTypes;

  if (getOpcode() != I->getOpcode() ||
      getNumOperands() != I->getNumOperands() ||
      (UseScalarTypes ?
       getType()->getScalarType() != I->getType()->getScalarType() :
       getType() != I->getType()))
    return false;

  // We have two instructions of identical opcode and #operands.  Check to see
  // if all operands are the same type
  for (unsigned i = 0, e = getNumOperands(); i != e; ++i)
    if (UseScalarTypes ?
        getOperand(i)->getType()->getScalarType() !=
          I->getOperand(i)->getType()->getScalarType() :
        getOperand(i)->getType() != I->getOperand(i)->getType())
      return false;

  return haveSameSpecialState(this, I, IgnoreAlignment);
}

/// isUsedOutsideOfBlock - Return true if there are any uses of I outside of the
/// specified block.  Note that PHI nodes are considered to evaluate their
/// operands in the corresponding predecessor block.
bool Instruction::isUsedOutsideOfBlock(const BasicBlock *BB) const {
  for (const Use &U : uses()) {
    // PHI nodes uses values in the corresponding predecessor block.  For other
    // instructions, just check to see whether the parent of the use matches up.
    const Instruction *I = cast<Instruction>(U.getUser());
    const PHINode *PN = dyn_cast<PHINode>(I);
    if (!PN) {
      if (I->getParent() != BB)
        return true;
      continue;
    }

    if (PN->getIncomingBlock(U) != BB)
      return true;
  }
  return false;
}

/// mayReadFromMemory - Return true if this instruction may read memory.
///
bool Instruction::mayReadFromMemory() const {
  switch (getOpcode()) {
  default: return false;
  case Instruction::VAArg:
  case Instruction::Load:
  case Instruction::Fence: // FIXME: refine definition of mayReadFromMemory
  case Instruction::AtomicCmpXchg:
  case Instruction::AtomicRMW:
    return true;
  case Instruction::Call:
    return !cast<CallInst>(this)->doesNotAccessMemory();
  case Instruction::Invoke:
    return !cast<InvokeInst>(this)->doesNotAccessMemory();
  case Instruction::Store:
    return !cast<StoreInst>(this)->isUnordered();
  }
}

/// mayWriteToMemory - Return true if this instruction may modify memory.
///
bool Instruction::mayWriteToMemory() const {
  switch (getOpcode()) {
  default: return false;
  case Instruction::Fence: // FIXME: refine definition of mayWriteToMemory
  case Instruction::Store:
  case Instruction::VAArg:
  case Instruction::AtomicCmpXchg:
  case Instruction::AtomicRMW:
    return true;
  case Instruction::Call:
    return !cast<CallInst>(this)->onlyReadsMemory();
  case Instruction::Invoke:
    return !cast<InvokeInst>(this)->onlyReadsMemory();
  case Instruction::Load:
    return !cast<LoadInst>(this)->isUnordered();
  }
}

bool Instruction::isAtomic() const {
  switch (getOpcode()) {
  default:
    return false;
  case Instruction::AtomicCmpXchg:
  case Instruction::AtomicRMW:
  case Instruction::Fence:
    return true;
  case Instruction::Load:
    return cast<LoadInst>(this)->getOrdering() != NotAtomic;
  case Instruction::Store:
    return cast<StoreInst>(this)->getOrdering() != NotAtomic;
  }
}

bool Instruction::mayThrow() const {
  if (const CallInst *CI = dyn_cast<CallInst>(this))
    return !CI->doesNotThrow();
  return isa<ResumeInst>(this);
}

bool Instruction::mayReturn() const {
  if (const CallInst *CI = dyn_cast<CallInst>(this))
    return !CI->doesNotReturn();
  return true;
}

/// isAssociative - Return true if the instruction is associative:
///
///   Associative operators satisfy:  x op (y op z) === (x op y) op z
///
/// In LLVM, the Add, Mul, And, Or, and Xor operators are associative.
///
bool Instruction::isAssociative(unsigned Opcode) {
  return Opcode == And || Opcode == Or || Opcode == Xor ||
         Opcode == Add || Opcode == Mul;
}

bool Instruction::isAssociative() const {
  unsigned Opcode = getOpcode();
  if (isAssociative(Opcode))
    return true;

  switch (Opcode) {
  case FMul:
  case FAdd:
    return cast<FPMathOperator>(this)->hasUnsafeAlgebra();
  default:
    return false;
  }
}

/// isCommutative - Return true if the instruction is commutative:
///
///   Commutative operators satisfy: (x op y) === (y op x)
///
/// In LLVM, these are the associative operators, plus SetEQ and SetNE, when
/// applied to any type.
///
bool Instruction::isCommutative(unsigned op) {
  switch (op) {
  case Add:
  case FAdd:
  case Mul:
  case FMul:
  case And:
  case Or:
  case Xor:
    return true;
  default:
    return false;
  }
}

/// isIdempotent - Return true if the instruction is idempotent:
///
///   Idempotent operators satisfy:  x op x === x
///
/// In LLVM, the And and Or operators are idempotent.
///
bool Instruction::isIdempotent(unsigned Opcode) {
  return Opcode == And || Opcode == Or;
}

/// isNilpotent - Return true if the instruction is nilpotent:
///
///   Nilpotent operators satisfy:  x op x === Id,
///
///   where Id is the identity for the operator, i.e. a constant such that
///     x op Id === x and Id op x === x for all x.
///
/// In LLVM, the Xor operator is nilpotent.
///
bool Instruction::isNilpotent(unsigned Opcode) {
  return Opcode == Xor;
}

Instruction *Instruction::cloneImpl() const {
  llvm_unreachable("Subclass of Instruction failed to implement cloneImpl");
}

Instruction *Instruction::clone() const {
  Instruction *New = nullptr;
  switch (getOpcode()) {
  default:
    llvm_unreachable("Unhandled Opcode.");
#define HANDLE_INST(num, opc, clas)                                            \
  case Instruction::opc:                                                       \
    New = cast<clas>(this)->cloneImpl();                                       \
    break;
#include "llvm/IR/Instruction.def"
#undef HANDLE_INST
  }

  New->SubclassOptionalData = SubclassOptionalData;
  if (!hasMetadata())
    return New;

  // Otherwise, enumerate and copy over metadata from the old instruction to the
  // new one.
  SmallVector<std::pair<unsigned, MDNode *>, 4> TheMDs;
  getAllMetadataOtherThanDebugLoc(TheMDs);
  for (const auto &MD : TheMDs)
    New->setMetadata(MD.first, MD.second);

  New->setDebugLoc(getDebugLoc());
  return New;
}
