//===------ MemoryBuiltins.cpp - Identify calls to memory builtins --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This family of functions identifies calls to builtin functions that allocate
// or free memory.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/MemoryBuiltins.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;

#define DEBUG_TYPE "memory-builtins"

enum AllocType {
  OpNewLike          = 1<<0, // allocates; never returns null
  MallocLike         = 1<<1 | OpNewLike, // allocates; may return null
  CallocLike         = 1<<2, // allocates + bzero
  ReallocLike        = 1<<3, // reallocates
  StrDupLike         = 1<<4,
  AllocLike          = MallocLike | CallocLike | StrDupLike,
  AnyAlloc           = AllocLike | ReallocLike
};

struct AllocFnsTy {
  LibFunc::Func Func;
  AllocType AllocTy;
  unsigned char NumParams;
  // First and Second size parameters (or -1 if unused)
  signed char FstParam, SndParam;
};

// FIXME: certain users need more information. E.g., SimplifyLibCalls needs to
// know which functions are nounwind, noalias, nocapture parameters, etc.
static const AllocFnsTy AllocationFnData[] = {
  {LibFunc::malloc,              MallocLike,  1, 0,  -1},
  {LibFunc::valloc,              MallocLike,  1, 0,  -1},
  {LibFunc::Znwj,                OpNewLike,   1, 0,  -1}, // new(unsigned int)
  {LibFunc::ZnwjRKSt9nothrow_t,  MallocLike,  2, 0,  -1}, // new(unsigned int, nothrow)
  {LibFunc::Znwm,                OpNewLike,   1, 0,  -1}, // new(unsigned long)
  {LibFunc::ZnwmRKSt9nothrow_t,  MallocLike,  2, 0,  -1}, // new(unsigned long, nothrow)
  {LibFunc::Znaj,                OpNewLike,   1, 0,  -1}, // new[](unsigned int)
  {LibFunc::ZnajRKSt9nothrow_t,  MallocLike,  2, 0,  -1}, // new[](unsigned int, nothrow)
  {LibFunc::Znam,                OpNewLike,   1, 0,  -1}, // new[](unsigned long)
  {LibFunc::ZnamRKSt9nothrow_t,  MallocLike,  2, 0,  -1}, // new[](unsigned long, nothrow)
  {LibFunc::calloc,              CallocLike,  2, 0,   1},
  {LibFunc::realloc,             ReallocLike, 2, 1,  -1},
  {LibFunc::reallocf,            ReallocLike, 2, 1,  -1},
  {LibFunc::strdup,              StrDupLike,  1, -1, -1},
  {LibFunc::strndup,             StrDupLike,  2, 1,  -1}
  // TODO: Handle "int posix_memalign(void **, size_t, size_t)"
};


static Function *getCalledFunction(const Value *V, bool LookThroughBitCast) {
  if (LookThroughBitCast)
    V = V->stripPointerCasts();

  CallSite CS(const_cast<Value*>(V));
  if (!CS.getInstruction())
    return nullptr;

  if (CS.isNoBuiltin())
    return nullptr;

  Function *Callee = CS.getCalledFunction();
  if (!Callee || !Callee->isDeclaration())
    return nullptr;
  return Callee;
}

/// \brief Returns the allocation data for the given value if it is a call to a
/// known allocation function, and NULL otherwise.
static const AllocFnsTy *getAllocationData(const Value *V, AllocType AllocTy,
                                           const TargetLibraryInfo *TLI,
                                           bool LookThroughBitCast = false) {
  // Skip intrinsics
  if (isa<IntrinsicInst>(V))
    return nullptr;

  Function *Callee = getCalledFunction(V, LookThroughBitCast);
  if (!Callee)
    return nullptr;

  // Make sure that the function is available.
  StringRef FnName = Callee->getName();
  LibFunc::Func TLIFn;
  if (!TLI || !TLI->getLibFunc(FnName, TLIFn) || !TLI->has(TLIFn))
    return nullptr;

  unsigned i = 0;
  bool found = false;
  for ( ; i < array_lengthof(AllocationFnData); ++i) {
    if (AllocationFnData[i].Func == TLIFn) {
      found = true;
      break;
    }
  }
  if (!found)
    return nullptr;

  const AllocFnsTy *FnData = &AllocationFnData[i];
  if ((FnData->AllocTy & AllocTy) != FnData->AllocTy)
    return nullptr;

  // Check function prototype.
  int FstParam = FnData->FstParam;
  int SndParam = FnData->SndParam;
  FunctionType *FTy = Callee->getFunctionType();

  if (FTy->getReturnType() == Type::getInt8PtrTy(FTy->getContext()) &&
      FTy->getNumParams() == FnData->NumParams &&
      (FstParam < 0 ||
       (FTy->getParamType(FstParam)->isIntegerTy(32) ||
        FTy->getParamType(FstParam)->isIntegerTy(64))) &&
      (SndParam < 0 ||
       FTy->getParamType(SndParam)->isIntegerTy(32) ||
       FTy->getParamType(SndParam)->isIntegerTy(64)))
    return FnData;
  return nullptr;
}

static bool hasNoAliasAttr(const Value *V, bool LookThroughBitCast) {
  ImmutableCallSite CS(LookThroughBitCast ? V->stripPointerCasts() : V);
  return CS && CS.hasFnAttr(Attribute::NoAlias);
}


/// \brief Tests if a value is a call or invoke to a library function that
/// allocates or reallocates memory (either malloc, calloc, realloc, or strdup
/// like).
bool llvm::isAllocationFn(const Value *V, const TargetLibraryInfo *TLI,
                          bool LookThroughBitCast) {
  return getAllocationData(V, AnyAlloc, TLI, LookThroughBitCast);
}

/// \brief Tests if a value is a call or invoke to a function that returns a
/// NoAlias pointer (including malloc/calloc/realloc/strdup-like functions).
bool llvm::isNoAliasFn(const Value *V, const TargetLibraryInfo *TLI,
                       bool LookThroughBitCast) {
  // it's safe to consider realloc as noalias since accessing the original
  // pointer is undefined behavior
  return isAllocationFn(V, TLI, LookThroughBitCast) ||
         hasNoAliasAttr(V, LookThroughBitCast);
}

/// \brief Tests if a value is a call or invoke to a library function that
/// allocates uninitialized memory (such as malloc).
bool llvm::isMallocLikeFn(const Value *V, const TargetLibraryInfo *TLI,
                          bool LookThroughBitCast) {
  return getAllocationData(V, MallocLike, TLI, LookThroughBitCast);
}

/// \brief Tests if a value is a call or invoke to a library function that
/// allocates zero-filled memory (such as calloc).
bool llvm::isCallocLikeFn(const Value *V, const TargetLibraryInfo *TLI,
                          bool LookThroughBitCast) {
  return getAllocationData(V, CallocLike, TLI, LookThroughBitCast);
}

/// \brief Tests if a value is a call or invoke to a library function that
/// allocates memory (either malloc, calloc, or strdup like).
bool llvm::isAllocLikeFn(const Value *V, const TargetLibraryInfo *TLI,
                         bool LookThroughBitCast) {
  return getAllocationData(V, AllocLike, TLI, LookThroughBitCast);
}

/// \brief Tests if a value is a call or invoke to a library function that
/// reallocates memory (such as realloc).
bool llvm::isReallocLikeFn(const Value *V, const TargetLibraryInfo *TLI,
                           bool LookThroughBitCast) {
  return getAllocationData(V, ReallocLike, TLI, LookThroughBitCast);
}

/// \brief Tests if a value is a call or invoke to a library function that
/// allocates memory and never returns null (such as operator new).
bool llvm::isOperatorNewLikeFn(const Value *V, const TargetLibraryInfo *TLI,
                               bool LookThroughBitCast) {
  return getAllocationData(V, OpNewLike, TLI, LookThroughBitCast);
}

/// extractMallocCall - Returns the corresponding CallInst if the instruction
/// is a malloc call.  Since CallInst::CreateMalloc() only creates calls, we
/// ignore InvokeInst here.
const CallInst *llvm::extractMallocCall(const Value *I,
                                        const TargetLibraryInfo *TLI) {
  return isMallocLikeFn(I, TLI) ? dyn_cast<CallInst>(I) : nullptr;
}

static Value *computeArraySize(const CallInst *CI, const DataLayout &DL,
                               const TargetLibraryInfo *TLI,
                               bool LookThroughSExt = false) {
  if (!CI)
    return nullptr;

  // The size of the malloc's result type must be known to determine array size.
  Type *T = getMallocAllocatedType(CI, TLI);
  if (!T || !T->isSized())
    return nullptr;

  unsigned ElementSize = DL.getTypeAllocSize(T);
  if (StructType *ST = dyn_cast<StructType>(T))
    ElementSize = DL.getStructLayout(ST)->getSizeInBytes();

  // If malloc call's arg can be determined to be a multiple of ElementSize,
  // return the multiple.  Otherwise, return NULL.
  Value *MallocArg = CI->getArgOperand(0);
  Value *Multiple = nullptr;
  if (ComputeMultiple(MallocArg, ElementSize, Multiple,
                      LookThroughSExt))
    return Multiple;

  return nullptr;
}

/// getMallocType - Returns the PointerType resulting from the malloc call.
/// The PointerType depends on the number of bitcast uses of the malloc call:
///   0: PointerType is the calls' return type.
///   1: PointerType is the bitcast's result type.
///  >1: Unique PointerType cannot be determined, return NULL.
PointerType *llvm::getMallocType(const CallInst *CI,
                                 const TargetLibraryInfo *TLI) {
  assert(isMallocLikeFn(CI, TLI) && "getMallocType and not malloc call");

  PointerType *MallocType = nullptr;
  unsigned NumOfBitCastUses = 0;

  // Determine if CallInst has a bitcast use.
  for (Value::const_user_iterator UI = CI->user_begin(), E = CI->user_end();
       UI != E;)
    if (const BitCastInst *BCI = dyn_cast<BitCastInst>(*UI++)) {
      MallocType = cast<PointerType>(BCI->getDestTy());
      NumOfBitCastUses++;
    }

  // Malloc call has 1 bitcast use, so type is the bitcast's destination type.
  if (NumOfBitCastUses == 1)
    return MallocType;

  // Malloc call was not bitcast, so type is the malloc function's return type.
  if (NumOfBitCastUses == 0)
    return cast<PointerType>(CI->getType());

  // Type could not be determined.
  return nullptr;
}

/// getMallocAllocatedType - Returns the Type allocated by malloc call.
/// The Type depends on the number of bitcast uses of the malloc call:
///   0: PointerType is the malloc calls' return type.
///   1: PointerType is the bitcast's result type.
///  >1: Unique PointerType cannot be determined, return NULL.
Type *llvm::getMallocAllocatedType(const CallInst *CI,
                                   const TargetLibraryInfo *TLI) {
  PointerType *PT = getMallocType(CI, TLI);
  return PT ? PT->getElementType() : nullptr;
}

/// getMallocArraySize - Returns the array size of a malloc call.  If the
/// argument passed to malloc is a multiple of the size of the malloced type,
/// then return that multiple.  For non-array mallocs, the multiple is
/// constant 1.  Otherwise, return NULL for mallocs whose array size cannot be
/// determined.
Value *llvm::getMallocArraySize(CallInst *CI, const DataLayout &DL,
                                const TargetLibraryInfo *TLI,
                                bool LookThroughSExt) {
  assert(isMallocLikeFn(CI, TLI) && "getMallocArraySize and not malloc call");
  return computeArraySize(CI, DL, TLI, LookThroughSExt);
}


/// extractCallocCall - Returns the corresponding CallInst if the instruction
/// is a calloc call.
const CallInst *llvm::extractCallocCall(const Value *I,
                                        const TargetLibraryInfo *TLI) {
  return isCallocLikeFn(I, TLI) ? cast<CallInst>(I) : nullptr;
}


/// isFreeCall - Returns non-null if the value is a call to the builtin free()
const CallInst *llvm::isFreeCall(const Value *I, const TargetLibraryInfo *TLI) {
  const CallInst *CI = dyn_cast<CallInst>(I);
  if (!CI || isa<IntrinsicInst>(CI))
    return nullptr;
  Function *Callee = CI->getCalledFunction();
  if (Callee == nullptr)
    return nullptr;

  StringRef FnName = Callee->getName();
  LibFunc::Func TLIFn;
  if (!TLI || !TLI->getLibFunc(FnName, TLIFn) || !TLI->has(TLIFn))
    return nullptr;

  unsigned ExpectedNumParams;
  if (TLIFn == LibFunc::free ||
      TLIFn == LibFunc::ZdlPv || // operator delete(void*)
      TLIFn == LibFunc::ZdaPv)   // operator delete[](void*)
    ExpectedNumParams = 1;
  else if (TLIFn == LibFunc::ZdlPvj ||              // delete(void*, uint)
           TLIFn == LibFunc::ZdlPvm ||              // delete(void*, ulong)
           TLIFn == LibFunc::ZdlPvRKSt9nothrow_t || // delete(void*, nothrow)
           TLIFn == LibFunc::ZdaPvj ||              // delete[](void*, uint)
           TLIFn == LibFunc::ZdaPvm ||              // delete[](void*, ulong)
           TLIFn == LibFunc::ZdaPvRKSt9nothrow_t)   // delete[](void*, nothrow)
    ExpectedNumParams = 2;
  else
    return nullptr;

  // Check free prototype.
  // FIXME: workaround for PR5130, this will be obsolete when a nobuiltin
  // attribute will exist.
  FunctionType *FTy = Callee->getFunctionType();
  if (!FTy->getReturnType()->isVoidTy())
    return nullptr;
  if (FTy->getNumParams() != ExpectedNumParams)
    return nullptr;
  if (FTy->getParamType(0) != Type::getInt8PtrTy(Callee->getContext()))
    return nullptr;

  return CI;
}



//===----------------------------------------------------------------------===//
//  Utility functions to compute size of objects.
//


/// \brief Compute the size of the object pointed by Ptr. Returns true and the
/// object size in Size if successful, and false otherwise.
/// If RoundToAlign is true, then Size is rounded up to the aligment of allocas,
/// byval arguments, and global variables.
bool llvm::getObjectSize(const Value *Ptr, uint64_t &Size, const DataLayout &DL,
                         const TargetLibraryInfo *TLI, bool RoundToAlign) {
  ObjectSizeOffsetVisitor Visitor(DL, TLI, Ptr->getContext(), RoundToAlign);
  SizeOffsetType Data = Visitor.compute(const_cast<Value*>(Ptr));
  if (!Visitor.bothKnown(Data))
    return false;

  APInt ObjSize = Data.first, Offset = Data.second;
  // check for overflow
  if (Offset.slt(0) || ObjSize.ult(Offset))
    Size = 0;
  else
    Size = (ObjSize - Offset).getZExtValue();
  return true;
}


STATISTIC(ObjectVisitorArgument,
          "Number of arguments with unsolved size and offset");
STATISTIC(ObjectVisitorLoad,
          "Number of load instructions with unsolved size and offset");


APInt ObjectSizeOffsetVisitor::align(APInt Size, uint64_t Align) {
  if (RoundToAlign && Align)
    return APInt(IntTyBits, RoundUpToAlignment(Size.getZExtValue(), Align));
  return Size;
}

ObjectSizeOffsetVisitor::ObjectSizeOffsetVisitor(const DataLayout &DL,
                                                 const TargetLibraryInfo *TLI,
                                                 LLVMContext &Context,
                                                 bool RoundToAlign)
    : DL(DL), TLI(TLI), RoundToAlign(RoundToAlign) {
  // Pointer size must be rechecked for each object visited since it could have
  // a different address space.
}

SizeOffsetType ObjectSizeOffsetVisitor::compute(Value *V) {
  IntTyBits = DL.getPointerTypeSizeInBits(V->getType());
  Zero = APInt::getNullValue(IntTyBits);

  V = V->stripPointerCasts();
  if (Instruction *I = dyn_cast<Instruction>(V)) {
    // If we have already seen this instruction, bail out. Cycles can happen in
    // unreachable code after constant propagation.
    if (!SeenInsts.insert(I).second)
      return unknown();

    if (GEPOperator *GEP = dyn_cast<GEPOperator>(V))
      return visitGEPOperator(*GEP);
    return visit(*I);
  }
  if (Argument *A = dyn_cast<Argument>(V))
    return visitArgument(*A);
  if (ConstantPointerNull *P = dyn_cast<ConstantPointerNull>(V))
    return visitConstantPointerNull(*P);
  if (GlobalAlias *GA = dyn_cast<GlobalAlias>(V))
    return visitGlobalAlias(*GA);
  if (GlobalVariable *GV = dyn_cast<GlobalVariable>(V))
    return visitGlobalVariable(*GV);
  if (UndefValue *UV = dyn_cast<UndefValue>(V))
    return visitUndefValue(*UV);
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->getOpcode() == Instruction::IntToPtr)
      return unknown(); // clueless
    if (CE->getOpcode() == Instruction::GetElementPtr)
      return visitGEPOperator(cast<GEPOperator>(*CE));
  }

  DEBUG(dbgs() << "ObjectSizeOffsetVisitor::compute() unhandled value: " << *V
        << '\n');
  return unknown();
}

SizeOffsetType ObjectSizeOffsetVisitor::visitAllocaInst(AllocaInst &I) {
  if (!I.getAllocatedType()->isSized())
    return unknown();

  APInt Size(IntTyBits, DL.getTypeAllocSize(I.getAllocatedType()));
  if (!I.isArrayAllocation())
    return std::make_pair(align(Size, I.getAlignment()), Zero);

  Value *ArraySize = I.getArraySize();
  if (const ConstantInt *C = dyn_cast<ConstantInt>(ArraySize)) {
    Size *= C->getValue().zextOrSelf(IntTyBits);
    return std::make_pair(align(Size, I.getAlignment()), Zero);
  }
  return unknown();
}

SizeOffsetType ObjectSizeOffsetVisitor::visitArgument(Argument &A) {
  // no interprocedural analysis is done at the moment
  if (!A.hasByValOrInAllocaAttr()) {
    ++ObjectVisitorArgument;
    return unknown();
  }
  PointerType *PT = cast<PointerType>(A.getType());
  APInt Size(IntTyBits, DL.getTypeAllocSize(PT->getElementType()));
  return std::make_pair(align(Size, A.getParamAlignment()), Zero);
}

SizeOffsetType ObjectSizeOffsetVisitor::visitCallSite(CallSite CS) {
  const AllocFnsTy *FnData = getAllocationData(CS.getInstruction(), AnyAlloc,
                                               TLI);
  if (!FnData)
    return unknown();

  // handle strdup-like functions separately
  if (FnData->AllocTy == StrDupLike) {
    APInt Size(IntTyBits, GetStringLength(CS.getArgument(0)));
    if (!Size)
      return unknown();

    // strndup limits strlen
    if (FnData->FstParam > 0) {
      ConstantInt *Arg= dyn_cast<ConstantInt>(CS.getArgument(FnData->FstParam));
      if (!Arg)
        return unknown();

      APInt MaxSize = Arg->getValue().zextOrSelf(IntTyBits);
      if (Size.ugt(MaxSize))
        Size = MaxSize + 1;
    }
    return std::make_pair(Size, Zero);
  }

  ConstantInt *Arg = dyn_cast<ConstantInt>(CS.getArgument(FnData->FstParam));
  if (!Arg)
    return unknown();

  APInt Size = Arg->getValue().zextOrSelf(IntTyBits);
  // size determined by just 1 parameter
  if (FnData->SndParam < 0)
    return std::make_pair(Size, Zero);

  Arg = dyn_cast<ConstantInt>(CS.getArgument(FnData->SndParam));
  if (!Arg)
    return unknown();

  Size *= Arg->getValue().zextOrSelf(IntTyBits);
  return std::make_pair(Size, Zero);

  // TODO: handle more standard functions (+ wchar cousins):
  // - strdup / strndup
  // - strcpy / strncpy
  // - strcat / strncat
  // - memcpy / memmove
  // - strcat / strncat
  // - memset
}

SizeOffsetType
ObjectSizeOffsetVisitor::visitConstantPointerNull(ConstantPointerNull&) {
  return std::make_pair(Zero, Zero);
}

SizeOffsetType
ObjectSizeOffsetVisitor::visitExtractElementInst(ExtractElementInst&) {
  return unknown();
}

SizeOffsetType
ObjectSizeOffsetVisitor::visitExtractValueInst(ExtractValueInst&) {
  // Easy cases were already folded by previous passes.
  return unknown();
}

SizeOffsetType ObjectSizeOffsetVisitor::visitGEPOperator(GEPOperator &GEP) {
  SizeOffsetType PtrData = compute(GEP.getPointerOperand());
  APInt Offset(IntTyBits, 0);
  if (!bothKnown(PtrData) || !GEP.accumulateConstantOffset(DL, Offset))
    return unknown();

  return std::make_pair(PtrData.first, PtrData.second + Offset);
}

SizeOffsetType ObjectSizeOffsetVisitor::visitGlobalAlias(GlobalAlias &GA) {
  if (GA.mayBeOverridden())
    return unknown();
  return compute(GA.getAliasee());
}

SizeOffsetType ObjectSizeOffsetVisitor::visitGlobalVariable(GlobalVariable &GV){
  if (!GV.hasDefinitiveInitializer())
    return unknown();

  APInt Size(IntTyBits, DL.getTypeAllocSize(GV.getType()->getElementType()));
  return std::make_pair(align(Size, GV.getAlignment()), Zero);
}

SizeOffsetType ObjectSizeOffsetVisitor::visitIntToPtrInst(IntToPtrInst&) {
  // clueless
  return unknown();
}

SizeOffsetType ObjectSizeOffsetVisitor::visitLoadInst(LoadInst&) {
  ++ObjectVisitorLoad;
  return unknown();
}

SizeOffsetType ObjectSizeOffsetVisitor::visitPHINode(PHINode&) {
  // too complex to analyze statically.
  return unknown();
}

SizeOffsetType ObjectSizeOffsetVisitor::visitSelectInst(SelectInst &I) {
  SizeOffsetType TrueSide  = compute(I.getTrueValue());
  SizeOffsetType FalseSide = compute(I.getFalseValue());
  if (bothKnown(TrueSide) && bothKnown(FalseSide) && TrueSide == FalseSide)
    return TrueSide;
  return unknown();
}

SizeOffsetType ObjectSizeOffsetVisitor::visitUndefValue(UndefValue&) {
  return std::make_pair(Zero, Zero);
}

SizeOffsetType ObjectSizeOffsetVisitor::visitInstruction(Instruction &I) {
  DEBUG(dbgs() << "ObjectSizeOffsetVisitor unknown instruction:" << I << '\n');
  return unknown();
}

ObjectSizeOffsetEvaluator::ObjectSizeOffsetEvaluator(
    const DataLayout &DL, const TargetLibraryInfo *TLI, LLVMContext &Context,
    bool RoundToAlign)
    : DL(DL), TLI(TLI), Context(Context), Builder(Context, TargetFolder(DL)),
      RoundToAlign(RoundToAlign) {
  // IntTy and Zero must be set for each compute() since the address space may
  // be different for later objects.
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::compute(Value *V) {
  // XXX - Are vectors of pointers possible here?
  IntTy = cast<IntegerType>(DL.getIntPtrType(V->getType()));
  Zero = ConstantInt::get(IntTy, 0);

  SizeOffsetEvalType Result = compute_(V);

  if (!bothKnown(Result)) {
    // erase everything that was computed in this iteration from the cache, so
    // that no dangling references are left behind. We could be a bit smarter if
    // we kept a dependency graph. It's probably not worth the complexity.
    for (PtrSetTy::iterator I=SeenVals.begin(), E=SeenVals.end(); I != E; ++I) {
      CacheMapTy::iterator CacheIt = CacheMap.find(*I);
      // non-computable results can be safely cached
      if (CacheIt != CacheMap.end() && anyKnown(CacheIt->second))
        CacheMap.erase(CacheIt);
    }
  }

  SeenVals.clear();
  return Result;
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::compute_(Value *V) {
  ObjectSizeOffsetVisitor Visitor(DL, TLI, Context, RoundToAlign);
  SizeOffsetType Const = Visitor.compute(V);
  if (Visitor.bothKnown(Const))
    return std::make_pair(ConstantInt::get(Context, Const.first),
                          ConstantInt::get(Context, Const.second));

  V = V->stripPointerCasts();

  // check cache
  CacheMapTy::iterator CacheIt = CacheMap.find(V);
  if (CacheIt != CacheMap.end())
    return CacheIt->second;

  // always generate code immediately before the instruction being
  // processed, so that the generated code dominates the same BBs
  Instruction *PrevInsertPoint = Builder.GetInsertPoint();
  if (Instruction *I = dyn_cast<Instruction>(V))
    Builder.SetInsertPoint(I);

  // now compute the size and offset
  SizeOffsetEvalType Result;

  // Record the pointers that were handled in this run, so that they can be
  // cleaned later if something fails. We also use this set to break cycles that
  // can occur in dead code.
  if (!SeenVals.insert(V).second) {
    Result = unknown();
  } else if (GEPOperator *GEP = dyn_cast<GEPOperator>(V)) {
    Result = visitGEPOperator(*GEP);
  } else if (Instruction *I = dyn_cast<Instruction>(V)) {
    Result = visit(*I);
  } else if (isa<Argument>(V) ||
             (isa<ConstantExpr>(V) &&
              cast<ConstantExpr>(V)->getOpcode() == Instruction::IntToPtr) ||
             isa<GlobalAlias>(V) ||
             isa<GlobalVariable>(V)) {
    // ignore values where we cannot do more than what ObjectSizeVisitor can
    Result = unknown();
  } else {
    DEBUG(dbgs() << "ObjectSizeOffsetEvaluator::compute() unhandled value: "
          << *V << '\n');
    Result = unknown();
  }

  if (PrevInsertPoint)
    Builder.SetInsertPoint(PrevInsertPoint);

  // Don't reuse CacheIt since it may be invalid at this point.
  CacheMap[V] = Result;
  return Result;
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::visitAllocaInst(AllocaInst &I) {
  if (!I.getAllocatedType()->isSized())
    return unknown();

  // must be a VLA
  assert(I.isArrayAllocation());
  Value *ArraySize = I.getArraySize();
  Value *Size = ConstantInt::get(ArraySize->getType(),
                                 DL.getTypeAllocSize(I.getAllocatedType()));
  Size = Builder.CreateMul(Size, ArraySize);
  return std::make_pair(Size, Zero);
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::visitCallSite(CallSite CS) {
  const AllocFnsTy *FnData = getAllocationData(CS.getInstruction(), AnyAlloc,
                                               TLI);
  if (!FnData)
    return unknown();

  // handle strdup-like functions separately
  if (FnData->AllocTy == StrDupLike) {
    // TODO
    return unknown();
  }

  Value *FirstArg = CS.getArgument(FnData->FstParam);
  FirstArg = Builder.CreateZExt(FirstArg, IntTy);
  if (FnData->SndParam < 0)
    return std::make_pair(FirstArg, Zero);

  Value *SecondArg = CS.getArgument(FnData->SndParam);
  SecondArg = Builder.CreateZExt(SecondArg, IntTy);
  Value *Size = Builder.CreateMul(FirstArg, SecondArg);
  return std::make_pair(Size, Zero);

  // TODO: handle more standard functions (+ wchar cousins):
  // - strdup / strndup
  // - strcpy / strncpy
  // - strcat / strncat
  // - memcpy / memmove
  // - strcat / strncat
  // - memset
}

SizeOffsetEvalType
ObjectSizeOffsetEvaluator::visitExtractElementInst(ExtractElementInst&) {
  return unknown();
}

SizeOffsetEvalType
ObjectSizeOffsetEvaluator::visitExtractValueInst(ExtractValueInst&) {
  return unknown();
}

SizeOffsetEvalType
ObjectSizeOffsetEvaluator::visitGEPOperator(GEPOperator &GEP) {
  SizeOffsetEvalType PtrData = compute_(GEP.getPointerOperand());
  if (!bothKnown(PtrData))
    return unknown();

  Value *Offset = EmitGEPOffset(&Builder, DL, &GEP, /*NoAssumptions=*/true);
  Offset = Builder.CreateAdd(PtrData.second, Offset);
  return std::make_pair(PtrData.first, Offset);
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::visitIntToPtrInst(IntToPtrInst&) {
  // clueless
  return unknown();
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::visitLoadInst(LoadInst&) {
  return unknown();
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::visitPHINode(PHINode &PHI) {
  // create 2 PHIs: one for size and another for offset
  PHINode *SizePHI   = Builder.CreatePHI(IntTy, PHI.getNumIncomingValues());
  PHINode *OffsetPHI = Builder.CreatePHI(IntTy, PHI.getNumIncomingValues());

  // insert right away in the cache to handle recursive PHIs
  CacheMap[&PHI] = std::make_pair(SizePHI, OffsetPHI);

  // compute offset/size for each PHI incoming pointer
  for (unsigned i = 0, e = PHI.getNumIncomingValues(); i != e; ++i) {
    Builder.SetInsertPoint(PHI.getIncomingBlock(i)->getFirstInsertionPt());
    SizeOffsetEvalType EdgeData = compute_(PHI.getIncomingValue(i));

    if (!bothKnown(EdgeData)) {
      OffsetPHI->replaceAllUsesWith(UndefValue::get(IntTy));
      OffsetPHI->eraseFromParent();
      SizePHI->replaceAllUsesWith(UndefValue::get(IntTy));
      SizePHI->eraseFromParent();
      return unknown();
    }
    SizePHI->addIncoming(EdgeData.first, PHI.getIncomingBlock(i));
    OffsetPHI->addIncoming(EdgeData.second, PHI.getIncomingBlock(i));
  }

  Value *Size = SizePHI, *Offset = OffsetPHI, *Tmp;
  if ((Tmp = SizePHI->hasConstantValue())) {
    Size = Tmp;
    SizePHI->replaceAllUsesWith(Size);
    SizePHI->eraseFromParent();
  }
  if ((Tmp = OffsetPHI->hasConstantValue())) {
    Offset = Tmp;
    OffsetPHI->replaceAllUsesWith(Offset);
    OffsetPHI->eraseFromParent();
  }
  return std::make_pair(Size, Offset);
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::visitSelectInst(SelectInst &I) {
  SizeOffsetEvalType TrueSide  = compute_(I.getTrueValue());
  SizeOffsetEvalType FalseSide = compute_(I.getFalseValue());

  if (!bothKnown(TrueSide) || !bothKnown(FalseSide))
    return unknown();
  if (TrueSide == FalseSide)
    return TrueSide;

  Value *Size = Builder.CreateSelect(I.getCondition(), TrueSide.first,
                                     FalseSide.first);
  Value *Offset = Builder.CreateSelect(I.getCondition(), TrueSide.second,
                                       FalseSide.second);
  return std::make_pair(Size, Offset);
}

SizeOffsetEvalType ObjectSizeOffsetEvaluator::visitInstruction(Instruction &I) {
  DEBUG(dbgs() << "ObjectSizeOffsetEvaluator unknown instruction:" << I <<'\n');
  return unknown();
}
