//===---- IRBuilder.cpp - Builder for LLVM Instrs -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the IRBuilder class, which is used as a convenient way
// to create LLVM instructions with a consistent and simplified interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Statepoint.h"
using namespace llvm;

/// CreateGlobalString - Make a new global variable with an initializer that
/// has array of i8 type filled in with the nul terminated string value
/// specified.  If Name is specified, it is the name of the global variable
/// created.
GlobalVariable *IRBuilderBase::CreateGlobalString(StringRef Str,
                                                  const Twine &Name,
                                                  unsigned AddressSpace) {
  Constant *StrConstant = ConstantDataArray::getString(Context, Str);
  Module &M = *BB->getParent()->getParent();
  GlobalVariable *GV = new GlobalVariable(M, StrConstant->getType(),
                                          true, GlobalValue::PrivateLinkage,
                                          StrConstant, Name, nullptr,
                                          GlobalVariable::NotThreadLocal,
                                          AddressSpace);
  GV->setUnnamedAddr(true);
  return GV;
}

Type *IRBuilderBase::getCurrentFunctionReturnType() const {
  assert(BB && BB->getParent() && "No current function!");
  return BB->getParent()->getReturnType();
}

Value *IRBuilderBase::getCastedInt8PtrValue(Value *Ptr) {
  PointerType *PT = cast<PointerType>(Ptr->getType());
  if (PT->getElementType()->isIntegerTy(8))
    return Ptr;
  
  // Otherwise, we need to insert a bitcast.
  PT = getInt8PtrTy(PT->getAddressSpace());
  BitCastInst *BCI = new BitCastInst(Ptr, PT, "");
  BB->getInstList().insert(InsertPt, BCI);
  SetInstDebugLocation(BCI);
  return BCI;
}

static CallInst *createCallHelper(Value *Callee, ArrayRef<Value *> Ops,
                                  IRBuilderBase *Builder,
                                  const Twine& Name="") {
  CallInst *CI = CallInst::Create(Callee, Ops, Name);
  Builder->GetInsertBlock()->getInstList().insert(Builder->GetInsertPoint(),CI);
  Builder->SetInstDebugLocation(CI);
  return CI;  
}

static InvokeInst *createInvokeHelper(Value *Invokee, BasicBlock *NormalDest,
                                      BasicBlock *UnwindDest,
                                      ArrayRef<Value *> Ops,
                                      IRBuilderBase *Builder,
                                      const Twine &Name = "") {
  InvokeInst *II =
      InvokeInst::Create(Invokee, NormalDest, UnwindDest, Ops, Name);
  Builder->GetInsertBlock()->getInstList().insert(Builder->GetInsertPoint(),
                                                  II);
  Builder->SetInstDebugLocation(II);
  return II;
}

CallInst *IRBuilderBase::
CreateMemSet(Value *Ptr, Value *Val, Value *Size, unsigned Align,
             bool isVolatile, MDNode *TBAATag, MDNode *ScopeTag,
             MDNode *NoAliasTag) {
  Ptr = getCastedInt8PtrValue(Ptr);
  Value *Ops[] = { Ptr, Val, Size, getInt32(Align), getInt1(isVolatile) };
  Type *Tys[] = { Ptr->getType(), Size->getType() };
  Module *M = BB->getParent()->getParent();
  Value *TheFn = Intrinsic::getDeclaration(M, Intrinsic::memset, Tys);
  
  CallInst *CI = createCallHelper(TheFn, Ops, this);
  
  // Set the TBAA info if present.
  if (TBAATag)
    CI->setMetadata(LLVMContext::MD_tbaa, TBAATag);

  if (ScopeTag)
    CI->setMetadata(LLVMContext::MD_alias_scope, ScopeTag);
 
  if (NoAliasTag)
    CI->setMetadata(LLVMContext::MD_noalias, NoAliasTag);
 
  return CI;
}

CallInst *IRBuilderBase::
CreateMemCpy(Value *Dst, Value *Src, Value *Size, unsigned Align,
             bool isVolatile, MDNode *TBAATag, MDNode *TBAAStructTag,
             MDNode *ScopeTag, MDNode *NoAliasTag) {
  Dst = getCastedInt8PtrValue(Dst);
  Src = getCastedInt8PtrValue(Src);

  Value *Ops[] = { Dst, Src, Size, getInt32(Align), getInt1(isVolatile) };
  Type *Tys[] = { Dst->getType(), Src->getType(), Size->getType() };
  Module *M = BB->getParent()->getParent();
  Value *TheFn = Intrinsic::getDeclaration(M, Intrinsic::memcpy, Tys);
  
  CallInst *CI = createCallHelper(TheFn, Ops, this);
  
  // Set the TBAA info if present.
  if (TBAATag)
    CI->setMetadata(LLVMContext::MD_tbaa, TBAATag);

  // Set the TBAA Struct info if present.
  if (TBAAStructTag)
    CI->setMetadata(LLVMContext::MD_tbaa_struct, TBAAStructTag);
 
  if (ScopeTag)
    CI->setMetadata(LLVMContext::MD_alias_scope, ScopeTag);
 
  if (NoAliasTag)
    CI->setMetadata(LLVMContext::MD_noalias, NoAliasTag);
 
  return CI;  
}

CallInst *IRBuilderBase::
CreateMemMove(Value *Dst, Value *Src, Value *Size, unsigned Align,
              bool isVolatile, MDNode *TBAATag, MDNode *ScopeTag,
              MDNode *NoAliasTag) {
  Dst = getCastedInt8PtrValue(Dst);
  Src = getCastedInt8PtrValue(Src);
  
  Value *Ops[] = { Dst, Src, Size, getInt32(Align), getInt1(isVolatile) };
  Type *Tys[] = { Dst->getType(), Src->getType(), Size->getType() };
  Module *M = BB->getParent()->getParent();
  Value *TheFn = Intrinsic::getDeclaration(M, Intrinsic::memmove, Tys);
  
  CallInst *CI = createCallHelper(TheFn, Ops, this);
  
  // Set the TBAA info if present.
  if (TBAATag)
    CI->setMetadata(LLVMContext::MD_tbaa, TBAATag);
 
  if (ScopeTag)
    CI->setMetadata(LLVMContext::MD_alias_scope, ScopeTag);
 
  if (NoAliasTag)
    CI->setMetadata(LLVMContext::MD_noalias, NoAliasTag);
 
  return CI;  
}

CallInst *IRBuilderBase::CreateLifetimeStart(Value *Ptr, ConstantInt *Size) {
  assert(isa<PointerType>(Ptr->getType()) &&
         "lifetime.start only applies to pointers.");
  Ptr = getCastedInt8PtrValue(Ptr);
  if (!Size)
    Size = getInt64(-1);
  else
    assert(Size->getType() == getInt64Ty() &&
           "lifetime.start requires the size to be an i64");
  Value *Ops[] = { Size, Ptr };
  Module *M = BB->getParent()->getParent();
  Value *TheFn = Intrinsic::getDeclaration(M, Intrinsic::lifetime_start);
  return createCallHelper(TheFn, Ops, this);
}

CallInst *IRBuilderBase::CreateLifetimeEnd(Value *Ptr, ConstantInt *Size) {
  assert(isa<PointerType>(Ptr->getType()) &&
         "lifetime.end only applies to pointers.");
  Ptr = getCastedInt8PtrValue(Ptr);
  if (!Size)
    Size = getInt64(-1);
  else
    assert(Size->getType() == getInt64Ty() &&
           "lifetime.end requires the size to be an i64");
  Value *Ops[] = { Size, Ptr };
  Module *M = BB->getParent()->getParent();
  Value *TheFn = Intrinsic::getDeclaration(M, Intrinsic::lifetime_end);
  return createCallHelper(TheFn, Ops, this);
}

CallInst *IRBuilderBase::CreateAssumption(Value *Cond) {
  assert(Cond->getType() == getInt1Ty() &&
         "an assumption condition must be of type i1");

  Value *Ops[] = { Cond };
  Module *M = BB->getParent()->getParent();
  Value *FnAssume = Intrinsic::getDeclaration(M, Intrinsic::assume);
  return createCallHelper(FnAssume, Ops, this);
}

/// Create a call to a Masked Load intrinsic.
/// Ptr      - the base pointer for the load
/// Align    - alignment of the source location
/// Mask     - an vector of booleans which indicates what vector lanes should
///            be accessed in memory
/// PassThru - a pass-through value that is used to fill the masked-off lanes
///            of the result
/// Name     - name of the result variable
CallInst *IRBuilderBase::CreateMaskedLoad(Value *Ptr, unsigned Align,
                                          Value *Mask, Value *PassThru,
                                          const Twine &Name) {
  assert(Ptr->getType()->isPointerTy() && "Ptr must be of pointer type");
  // DataTy is the overloaded type
  Type *DataTy = cast<PointerType>(Ptr->getType())->getElementType();
  assert(DataTy->isVectorTy() && "Ptr should point to a vector");
  if (!PassThru)
    PassThru = UndefValue::get(DataTy);
  Value *Ops[] = { Ptr, getInt32(Align), Mask,  PassThru};
  return CreateMaskedIntrinsic(Intrinsic::masked_load, Ops, DataTy, Name);
}

/// Create a call to a Masked Store intrinsic.
/// Val   - the data to be stored,
/// Ptr   - the base pointer for the store
/// Align - alignment of the destination location
/// Mask  - an vector of booleans which indicates what vector lanes should
///         be accessed in memory
CallInst *IRBuilderBase::CreateMaskedStore(Value *Val, Value *Ptr,
                                           unsigned Align, Value *Mask) {
  Value *Ops[] = { Val, Ptr, getInt32(Align), Mask };
  // Type of the data to be stored - the only one overloaded type
  return CreateMaskedIntrinsic(Intrinsic::masked_store, Ops, Val->getType());
}

/// Create a call to a Masked intrinsic, with given intrinsic Id,
/// an array of operands - Ops, and one overloaded type - DataTy
CallInst *IRBuilderBase::CreateMaskedIntrinsic(Intrinsic::ID Id,
                                               ArrayRef<Value *> Ops,
                                               Type *DataTy,
                                               const Twine &Name) {
  Module *M = BB->getParent()->getParent();
  Type *OverloadedTypes[] = { DataTy };
  Value *TheFn = Intrinsic::getDeclaration(M, Id, OverloadedTypes);
  return createCallHelper(TheFn, Ops, this, Name);
}

static std::vector<Value *>
getStatepointArgs(IRBuilderBase &B, uint64_t ID, uint32_t NumPatchBytes,
                  Value *ActualCallee, ArrayRef<Value *> CallArgs,
                  ArrayRef<Value *> DeoptArgs, ArrayRef<Value *> GCArgs) {
  std::vector<Value *> Args;
  Args.push_back(B.getInt64(ID));
  Args.push_back(B.getInt32(NumPatchBytes));
  Args.push_back(ActualCallee);
  Args.push_back(B.getInt32(CallArgs.size()));
  Args.push_back(B.getInt32((unsigned)StatepointFlags::None));
  Args.insert(Args.end(), CallArgs.begin(), CallArgs.end());
  Args.push_back(B.getInt32(0 /* no transition args */));
  Args.push_back(B.getInt32(DeoptArgs.size()));
  Args.insert(Args.end(), DeoptArgs.begin(), DeoptArgs.end());
  Args.insert(Args.end(), GCArgs.begin(), GCArgs.end());

  return Args;
}

CallInst *IRBuilderBase::CreateGCStatepointCall(
    uint64_t ID, uint32_t NumPatchBytes, Value *ActualCallee,
    ArrayRef<Value *> CallArgs, ArrayRef<Value *> DeoptArgs,
    ArrayRef<Value *> GCArgs, const Twine &Name) {
  // Extract out the type of the callee.
  PointerType *FuncPtrType = cast<PointerType>(ActualCallee->getType());
  assert(isa<FunctionType>(FuncPtrType->getElementType()) &&
         "actual callee must be a callable value");

  Module *M = BB->getParent()->getParent();
  // Fill in the one generic type'd argument (the function is also vararg)
  Type *ArgTypes[] = { FuncPtrType };
  Function *FnStatepoint =
    Intrinsic::getDeclaration(M, Intrinsic::experimental_gc_statepoint,
                              ArgTypes);

  std::vector<llvm::Value *> Args = getStatepointArgs(
      *this, ID, NumPatchBytes, ActualCallee, CallArgs, DeoptArgs, GCArgs);
  return createCallHelper(FnStatepoint, Args, this, Name);
}

CallInst *IRBuilderBase::CreateGCStatepointCall(
    uint64_t ID, uint32_t NumPatchBytes, Value *ActualCallee,
    ArrayRef<Use> CallArgs, ArrayRef<Value *> DeoptArgs,
    ArrayRef<Value *> GCArgs, const Twine &Name) {
  std::vector<Value *> VCallArgs;
  for (auto &U : CallArgs)
    VCallArgs.push_back(U.get());
  return CreateGCStatepointCall(ID, NumPatchBytes, ActualCallee, VCallArgs,
                                DeoptArgs, GCArgs, Name);
}

InvokeInst *IRBuilderBase::CreateGCStatepointInvoke(
    uint64_t ID, uint32_t NumPatchBytes, Value *ActualInvokee,
    BasicBlock *NormalDest, BasicBlock *UnwindDest,
    ArrayRef<Value *> InvokeArgs, ArrayRef<Value *> DeoptArgs,
    ArrayRef<Value *> GCArgs, const Twine &Name) {
  // Extract out the type of the callee.
  PointerType *FuncPtrType = cast<PointerType>(ActualInvokee->getType());
  assert(isa<FunctionType>(FuncPtrType->getElementType()) &&
         "actual callee must be a callable value");

  Module *M = BB->getParent()->getParent();
  // Fill in the one generic type'd argument (the function is also vararg)
  Function *FnStatepoint = Intrinsic::getDeclaration(
      M, Intrinsic::experimental_gc_statepoint, {FuncPtrType});

  std::vector<llvm::Value *> Args = getStatepointArgs(
      *this, ID, NumPatchBytes, ActualInvokee, InvokeArgs, DeoptArgs, GCArgs);
  return createInvokeHelper(FnStatepoint, NormalDest, UnwindDest, Args, this,
                            Name);
}

InvokeInst *IRBuilderBase::CreateGCStatepointInvoke(
    uint64_t ID, uint32_t NumPatchBytes, Value *ActualInvokee,
    BasicBlock *NormalDest, BasicBlock *UnwindDest, ArrayRef<Use> InvokeArgs,
    ArrayRef<Value *> DeoptArgs, ArrayRef<Value *> GCArgs, const Twine &Name) {
  std::vector<Value *> VCallArgs;
  for (auto &U : InvokeArgs)
    VCallArgs.push_back(U.get());
  return CreateGCStatepointInvoke(ID, NumPatchBytes, ActualInvokee, NormalDest,
                                  UnwindDest, VCallArgs, DeoptArgs, GCArgs,
                                  Name);
}

CallInst *IRBuilderBase::CreateGCResult(Instruction *Statepoint,
                                       Type *ResultType,
                                       const Twine &Name) {
 Intrinsic::ID ID = Intrinsic::experimental_gc_result;
 Module *M = BB->getParent()->getParent();
 Type *Types[] = {ResultType};
 Value *FnGCResult = Intrinsic::getDeclaration(M, ID, Types);

 Value *Args[] = {Statepoint};
 return createCallHelper(FnGCResult, Args, this, Name);
}

CallInst *IRBuilderBase::CreateGCRelocate(Instruction *Statepoint,
                                         int BaseOffset,
                                         int DerivedOffset,
                                         Type *ResultType,
                                         const Twine &Name) {
 Module *M = BB->getParent()->getParent();
 Type *Types[] = {ResultType};
 Value *FnGCRelocate =
   Intrinsic::getDeclaration(M, Intrinsic::experimental_gc_relocate, Types);

 Value *Args[] = {Statepoint,
                  getInt32(BaseOffset),
                  getInt32(DerivedOffset)};
 return createCallHelper(FnGCRelocate, Args, this, Name);
}
