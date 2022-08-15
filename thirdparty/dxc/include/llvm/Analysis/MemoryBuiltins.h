//===- llvm/Analysis/MemoryBuiltins.h- Calls to memory builtins -*- C++ -*-===//
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

#ifndef LLVM_ANALYSIS_MEMORYBUILTINS_H
#define LLVM_ANALYSIS_MEMORYBUILTINS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/TargetFolder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Operator.h"
#include "llvm/IR/ValueHandle.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
class CallInst;
class PointerType;
class DataLayout;
class TargetLibraryInfo;
class Type;
class Value;


/// \brief Tests if a value is a call or invoke to a library function that
/// allocates or reallocates memory (either malloc, calloc, realloc, or strdup
/// like).
bool isAllocationFn(const Value *V, const TargetLibraryInfo *TLI,
                    bool LookThroughBitCast = false);

/// \brief Tests if a value is a call or invoke to a function that returns a
/// NoAlias pointer (including malloc/calloc/realloc/strdup-like functions).
bool isNoAliasFn(const Value *V, const TargetLibraryInfo *TLI,
                 bool LookThroughBitCast = false);

/// \brief Tests if a value is a call or invoke to a library function that
/// allocates uninitialized memory (such as malloc).
bool isMallocLikeFn(const Value *V, const TargetLibraryInfo *TLI,
                    bool LookThroughBitCast = false);

/// \brief Tests if a value is a call or invoke to a library function that
/// allocates zero-filled memory (such as calloc).
bool isCallocLikeFn(const Value *V, const TargetLibraryInfo *TLI,
                    bool LookThroughBitCast = false);

/// \brief Tests if a value is a call or invoke to a library function that
/// allocates memory (either malloc, calloc, or strdup like).
bool isAllocLikeFn(const Value *V, const TargetLibraryInfo *TLI,
                   bool LookThroughBitCast = false);

/// \brief Tests if a value is a call or invoke to a library function that
/// reallocates memory (such as realloc).
bool isReallocLikeFn(const Value *V, const TargetLibraryInfo *TLI,
                     bool LookThroughBitCast = false);

/// \brief Tests if a value is a call or invoke to a library function that
/// allocates memory and never returns null (such as operator new).
bool isOperatorNewLikeFn(const Value *V, const TargetLibraryInfo *TLI,
                         bool LookThroughBitCast = false);

//===----------------------------------------------------------------------===//
//  malloc Call Utility Functions.
//

/// extractMallocCall - Returns the corresponding CallInst if the instruction
/// is a malloc call.  Since CallInst::CreateMalloc() only creates calls, we
/// ignore InvokeInst here.
const CallInst *extractMallocCall(const Value *I, const TargetLibraryInfo *TLI);
static inline CallInst *extractMallocCall(Value *I,
                                          const TargetLibraryInfo *TLI) {
  return const_cast<CallInst*>(extractMallocCall((const Value*)I, TLI));
}

/// getMallocType - Returns the PointerType resulting from the malloc call.
/// The PointerType depends on the number of bitcast uses of the malloc call:
///   0: PointerType is the malloc calls' return type.
///   1: PointerType is the bitcast's result type.
///  >1: Unique PointerType cannot be determined, return NULL.
PointerType *getMallocType(const CallInst *CI, const TargetLibraryInfo *TLI);

/// getMallocAllocatedType - Returns the Type allocated by malloc call.
/// The Type depends on the number of bitcast uses of the malloc call:
///   0: PointerType is the malloc calls' return type.
///   1: PointerType is the bitcast's result type.
///  >1: Unique PointerType cannot be determined, return NULL.
Type *getMallocAllocatedType(const CallInst *CI, const TargetLibraryInfo *TLI);

/// getMallocArraySize - Returns the array size of a malloc call.  If the
/// argument passed to malloc is a multiple of the size of the malloced type,
/// then return that multiple.  For non-array mallocs, the multiple is
/// constant 1.  Otherwise, return NULL for mallocs whose array size cannot be
/// determined.
Value *getMallocArraySize(CallInst *CI, const DataLayout &DL,
                          const TargetLibraryInfo *TLI,
                          bool LookThroughSExt = false);

//===----------------------------------------------------------------------===//
//  calloc Call Utility Functions.
//

/// extractCallocCall - Returns the corresponding CallInst if the instruction
/// is a calloc call.
const CallInst *extractCallocCall(const Value *I, const TargetLibraryInfo *TLI);
static inline CallInst *extractCallocCall(Value *I,
                                          const TargetLibraryInfo *TLI) {
  return const_cast<CallInst*>(extractCallocCall((const Value*)I, TLI));
}


//===----------------------------------------------------------------------===//
//  free Call Utility Functions.
//

/// isFreeCall - Returns non-null if the value is a call to the builtin free()
const CallInst *isFreeCall(const Value *I, const TargetLibraryInfo *TLI);

static inline CallInst *isFreeCall(Value *I, const TargetLibraryInfo *TLI) {
  return const_cast<CallInst*>(isFreeCall((const Value*)I, TLI));
}
//                                                                           //
///////////////////////////////////////////////////////////////////////////////
//  Utility functions to compute size of objects.
//

/// \brief Compute the size of the object pointed by Ptr. Returns true and the
/// object size in Size if successful, and false otherwise. In this context, by
/// object we mean the region of memory starting at Ptr to the end of the
/// underlying object pointed to by Ptr.
/// If RoundToAlign is true, then Size is rounded up to the aligment of allocas,
/// byval arguments, and global variables.
bool getObjectSize(const Value *Ptr, uint64_t &Size, const DataLayout &DL,
                   const TargetLibraryInfo *TLI, bool RoundToAlign = false);

typedef std::pair<APInt, APInt> SizeOffsetType;

/// \brief Evaluate the size and offset of an object pointed to by a Value*
/// statically. Fails if size or offset are not known at compile time.
class ObjectSizeOffsetVisitor
  : public InstVisitor<ObjectSizeOffsetVisitor, SizeOffsetType> {

  const DataLayout &DL;
  const TargetLibraryInfo *TLI;
  bool RoundToAlign;
  unsigned IntTyBits;
  APInt Zero;
  SmallPtrSet<Instruction *, 8> SeenInsts;

  APInt align(APInt Size, uint64_t Align);

  SizeOffsetType unknown() {
    return std::make_pair(APInt(), APInt());
  }

public:
  ObjectSizeOffsetVisitor(const DataLayout &DL, const TargetLibraryInfo *TLI,
                          LLVMContext &Context, bool RoundToAlign = false);

  SizeOffsetType compute(Value *V);

  bool knownSize(SizeOffsetType &SizeOffset) {
    return SizeOffset.first.getBitWidth() > 1;
  }

  bool knownOffset(SizeOffsetType &SizeOffset) {
    return SizeOffset.second.getBitWidth() > 1;
  }

  bool bothKnown(SizeOffsetType &SizeOffset) {
    return knownSize(SizeOffset) && knownOffset(SizeOffset);
  }

  // These are "private", except they can't actually be made private. Only
  // compute() should be used by external users.
  SizeOffsetType visitAllocaInst(AllocaInst &I);
  SizeOffsetType visitArgument(Argument &A);
  SizeOffsetType visitCallSite(CallSite CS);
  SizeOffsetType visitConstantPointerNull(ConstantPointerNull&);
  SizeOffsetType visitExtractElementInst(ExtractElementInst &I);
  SizeOffsetType visitExtractValueInst(ExtractValueInst &I);
  SizeOffsetType visitGEPOperator(GEPOperator &GEP);
  SizeOffsetType visitGlobalAlias(GlobalAlias &GA);
  SizeOffsetType visitGlobalVariable(GlobalVariable &GV);
  SizeOffsetType visitIntToPtrInst(IntToPtrInst&);
  SizeOffsetType visitLoadInst(LoadInst &I);
  SizeOffsetType visitPHINode(PHINode&);
  SizeOffsetType visitSelectInst(SelectInst &I);
  SizeOffsetType visitUndefValue(UndefValue&);
  SizeOffsetType visitInstruction(Instruction &I);
};

typedef std::pair<Value*, Value*> SizeOffsetEvalType;


/// \brief Evaluate the size and offset of an object pointed to by a Value*.
/// May create code to compute the result at run-time.
class ObjectSizeOffsetEvaluator
  : public InstVisitor<ObjectSizeOffsetEvaluator, SizeOffsetEvalType> {

  typedef IRBuilder<true, TargetFolder> BuilderTy;
  typedef std::pair<WeakVH, WeakVH> WeakEvalType;
  typedef DenseMap<const Value*, WeakEvalType> CacheMapTy;
  typedef SmallPtrSet<const Value*, 8> PtrSetTy;

  const DataLayout &DL;
  const TargetLibraryInfo *TLI;
  LLVMContext &Context;
  BuilderTy Builder;
  IntegerType *IntTy;
  Value *Zero;
  CacheMapTy CacheMap;
  PtrSetTy SeenVals;
  bool RoundToAlign;

  SizeOffsetEvalType unknown() {
    return std::make_pair(nullptr, nullptr);
  }
  SizeOffsetEvalType compute_(Value *V);

public:
  ObjectSizeOffsetEvaluator(const DataLayout &DL, const TargetLibraryInfo *TLI,
                            LLVMContext &Context, bool RoundToAlign = false);
  SizeOffsetEvalType compute(Value *V);

  bool knownSize(SizeOffsetEvalType SizeOffset) {
    return SizeOffset.first;
  }

  bool knownOffset(SizeOffsetEvalType SizeOffset) {
    return SizeOffset.second;
  }

  bool anyKnown(SizeOffsetEvalType SizeOffset) {
    return knownSize(SizeOffset) || knownOffset(SizeOffset);
  }

  bool bothKnown(SizeOffsetEvalType SizeOffset) {
    return knownSize(SizeOffset) && knownOffset(SizeOffset);
  }

  // The individual instruction visitors should be treated as private.
  SizeOffsetEvalType visitAllocaInst(AllocaInst &I);
  SizeOffsetEvalType visitCallSite(CallSite CS);
  SizeOffsetEvalType visitExtractElementInst(ExtractElementInst &I);
  SizeOffsetEvalType visitExtractValueInst(ExtractValueInst &I);
  SizeOffsetEvalType visitGEPOperator(GEPOperator &GEP);
  SizeOffsetEvalType visitIntToPtrInst(IntToPtrInst&);
  SizeOffsetEvalType visitLoadInst(LoadInst &I);
  SizeOffsetEvalType visitPHINode(PHINode &PHI);
  SizeOffsetEvalType visitSelectInst(SelectInst &I);
  SizeOffsetEvalType visitInstruction(Instruction &I);
};

} // End llvm namespace

#endif
