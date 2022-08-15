//===-- Function.cpp - Implement the Global object classes ----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Function class for the IR library.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Function.h"
#include "LLVMContextImpl.h"
#include "SymbolTableListTraitsImpl.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/CodeGen/ValueTypes.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/RWMutex.h"
#include "llvm/Support/StringPool.h"
#include "llvm/Support/Threading.h"
using namespace llvm;

// Explicit instantiations of SymbolTableListTraits since some of the methods
// are not in the public header file...
template class llvm::SymbolTableListTraits<Argument, Function>;
template class llvm::SymbolTableListTraits<BasicBlock, Function>;

//===----------------------------------------------------------------------===//
// Argument Implementation
//===----------------------------------------------------------------------===//

void Argument::anchor() { }

Argument::Argument(Type *Ty, const Twine &Name, Function *Par)
  : Value(Ty, Value::ArgumentVal) {
  Parent = nullptr;

  if (Par)
    Par->getArgumentList().push_back(this);
  setName(Name);
}

void Argument::setParent(Function *parent) {
  Parent = parent;
}

/// getArgNo - Return the index of this formal argument in its containing
/// function.  For example in "void foo(int a, float b)" a is 0 and b is 1.
unsigned Argument::getArgNo() const {
  const Function *F = getParent();
  assert(F && "Argument is not in a function");

  Function::const_arg_iterator AI = F->arg_begin();
  unsigned ArgIdx = 0;
  for (; &*AI != this; ++AI)
    ++ArgIdx;

  return ArgIdx;
}

/// hasNonNullAttr - Return true if this argument has the nonnull attribute on
/// it in its containing function. Also returns true if at least one byte is
/// known to be dereferenceable and the pointer is in addrspace(0).
bool Argument::hasNonNullAttr() const {
  if (!getType()->isPointerTy()) return false;
  if (getParent()->getAttributes().
        hasAttribute(getArgNo()+1, Attribute::NonNull))
    return true;
  else if (getDereferenceableBytes() > 0 &&
           getType()->getPointerAddressSpace() == 0)
    return true;
  return false;
}

/// hasByValAttr - Return true if this argument has the byval attribute on it
/// in its containing function.
bool Argument::hasByValAttr() const {
  if (!getType()->isPointerTy()) return false;
  return getParent()->getAttributes().
    hasAttribute(getArgNo()+1, Attribute::ByVal);
}

/// \brief Return true if this argument has the inalloca attribute on it in
/// its containing function.
bool Argument::hasInAllocaAttr() const {
  if (!getType()->isPointerTy()) return false;
  return getParent()->getAttributes().
    hasAttribute(getArgNo()+1, Attribute::InAlloca);
}

bool Argument::hasByValOrInAllocaAttr() const {
  if (!getType()->isPointerTy()) return false;
  AttributeSet Attrs = getParent()->getAttributes();
  return Attrs.hasAttribute(getArgNo() + 1, Attribute::ByVal) ||
         Attrs.hasAttribute(getArgNo() + 1, Attribute::InAlloca);
}

unsigned Argument::getParamAlignment() const {
  assert(getType()->isPointerTy() && "Only pointers have alignments");
  return getParent()->getParamAlignment(getArgNo()+1);

}

uint64_t Argument::getDereferenceableBytes() const {
  assert(getType()->isPointerTy() &&
         "Only pointers have dereferenceable bytes");
  return getParent()->getDereferenceableBytes(getArgNo()+1);
}

uint64_t Argument::getDereferenceableOrNullBytes() const {
  assert(getType()->isPointerTy() &&
         "Only pointers have dereferenceable bytes");
  return getParent()->getDereferenceableOrNullBytes(getArgNo()+1);
}

/// hasNestAttr - Return true if this argument has the nest attribute on
/// it in its containing function.
bool Argument::hasNestAttr() const {
  if (!getType()->isPointerTy()) return false;
  return getParent()->getAttributes().
    hasAttribute(getArgNo()+1, Attribute::Nest);
}

/// hasNoAliasAttr - Return true if this argument has the noalias attribute on
/// it in its containing function.
bool Argument::hasNoAliasAttr() const {
  if (!getType()->isPointerTy()) return false;
  return getParent()->getAttributes().
    hasAttribute(getArgNo()+1, Attribute::NoAlias);
}

/// hasNoCaptureAttr - Return true if this argument has the nocapture attribute
/// on it in its containing function.
bool Argument::hasNoCaptureAttr() const {
  if (!getType()->isPointerTy()) return false;
  return getParent()->getAttributes().
    hasAttribute(getArgNo()+1, Attribute::NoCapture);
}

/// hasSRetAttr - Return true if this argument has the sret attribute on
/// it in its containing function.
bool Argument::hasStructRetAttr() const {
  if (!getType()->isPointerTy()) return false;
  return getParent()->getAttributes().
    hasAttribute(getArgNo()+1, Attribute::StructRet);
}

/// hasReturnedAttr - Return true if this argument has the returned attribute on
/// it in its containing function.
bool Argument::hasReturnedAttr() const {
  return getParent()->getAttributes().
    hasAttribute(getArgNo()+1, Attribute::Returned);
}

/// hasZExtAttr - Return true if this argument has the zext attribute on it in
/// its containing function.
bool Argument::hasZExtAttr() const {
  return getParent()->getAttributes().
    hasAttribute(getArgNo()+1, Attribute::ZExt);
}

/// hasSExtAttr Return true if this argument has the sext attribute on it in its
/// containing function.
bool Argument::hasSExtAttr() const {
  return getParent()->getAttributes().
    hasAttribute(getArgNo()+1, Attribute::SExt);
}

/// Return true if this argument has the readonly or readnone attribute on it
/// in its containing function.
bool Argument::onlyReadsMemory() const {
  return getParent()->getAttributes().
      hasAttribute(getArgNo()+1, Attribute::ReadOnly) ||
      getParent()->getAttributes().
      hasAttribute(getArgNo()+1, Attribute::ReadNone);
}

/// addAttr - Add attributes to an argument.
void Argument::addAttr(AttributeSet AS) {
  assert(AS.getNumSlots() <= 1 &&
         "Trying to add more than one attribute set to an argument!");
  AttrBuilder B(AS, AS.getSlotIndex(0));
  getParent()->addAttributes(getArgNo() + 1,
                             AttributeSet::get(Parent->getContext(),
                                               getArgNo() + 1, B));
}

/// removeAttr - Remove attributes from an argument.
void Argument::removeAttr(AttributeSet AS) {
  assert(AS.getNumSlots() <= 1 &&
         "Trying to remove more than one attribute set from an argument!");
  AttrBuilder B(AS, AS.getSlotIndex(0));
  getParent()->removeAttributes(getArgNo() + 1,
                                AttributeSet::get(Parent->getContext(),
                                                  getArgNo() + 1, B));
}

//===----------------------------------------------------------------------===//
// Helper Methods in Function
//===----------------------------------------------------------------------===//

bool Function::isMaterializable() const {
  return getGlobalObjectSubClassData() & IsMaterializableBit;
}

void Function::setIsMaterializable(bool V) {
  setGlobalObjectBit(IsMaterializableBit, V);
}

LLVMContext &Function::getContext() const {
  return getType()->getContext();
}

FunctionType *Function::getFunctionType() const { return Ty; }

bool Function::isVarArg() const {
  return getFunctionType()->isVarArg();
}

Type *Function::getReturnType() const {
  return getFunctionType()->getReturnType();
}

void Function::removeFromParent() {
  getParent()->CallRemoveGlobalHook(this); // HLSL Change
  getParent()->getFunctionList().remove(this);
}

void Function::eraseFromParent() {
  getParent()->CallRemoveGlobalHook(this); // HLSL Change
  getParent()->getFunctionList().erase(this);
}

//===----------------------------------------------------------------------===//
// Function Implementation
//===----------------------------------------------------------------------===//

Function::Function(FunctionType *Ty, LinkageTypes Linkage, const Twine &name,
                   Module *ParentModule)
    : GlobalObject(PointerType::getUnqual(Ty), Value::FunctionVal,
                   OperandTraits<Function>::op_begin(this), 0, Linkage, name),
      Ty(Ty) {
  assert(FunctionType::isValidReturnType(getReturnType()) &&
         "invalid return type");
  setGlobalObjectSubClassData(0);
  SymTab.reset(new ValueSymbolTable()); // HLSL Change: use unique_ptr

  // If the function has arguments, mark them as lazily built.
  if (Ty->getNumParams())
    setValueSubclassData(1);   // Set the "has lazy arguments" bit.

  if (ParentModule)
    ParentModule->getFunctionList().push_back(this);

  // Ensure intrinsics have the right parameter attributes.
  // Note, the IntID field will have been set in Value::setName if this function
  // name is a valid intrinsic ID.
  if (IntID)
    setAttributes(Intrinsic::getAttributes(getContext(), IntID));
}

Function::~Function() {
  dropAllReferences();    // After this it is safe to delete instructions.

  // Delete all of the method arguments and unlink from symbol table...
  ArgumentList.clear();
  SymTab.reset(); // HLSL Change: use unique_ptr

  // Remove the function from the on-the-side GC table.
  clearGC();

  // FIXME: needed by operator delete
  setFunctionNumOperands(1);
}

void Function::BuildLazyArguments() const {
  // Create the arguments vector, all arguments start out unnamed.
  FunctionType *FT = getFunctionType();
  for (unsigned i = 0, e = FT->getNumParams(); i != e; ++i) {
    assert(!FT->getParamType(i)->isVoidTy() &&
           "Cannot have void typed arguments!");
    ArgumentList.push_back(new Argument(FT->getParamType(i)));
  }

  // Clear the lazy arguments bit.
  unsigned SDC = getSubclassDataFromValue();
  const_cast<Function*>(this)->setValueSubclassData(SDC &= ~(1<<0));
}

size_t Function::arg_size() const {
  return getFunctionType()->getNumParams();
}
bool Function::arg_empty() const {
  return getFunctionType()->getNumParams() == 0;
}

void Function::setParent(Module *parent) {
  Parent = parent;
}

// dropAllReferences() - This function causes all the subinstructions to "let
// go" of all references that they are maintaining.  This allows one to
// 'delete' a whole class at a time, even though there may be circular
// references... first all references are dropped, and all use counts go to
// zero.  Then everything is deleted for real.  Note that no operations are
// valid on an object that has "dropped all references", except operator
// delete.
//
void Function::dropAllReferences() {
  setIsMaterializable(false);

  for (iterator I = begin(), E = end(); I != E; ++I)
    I->dropAllReferences();

  // Delete all basic blocks. They are now unused, except possibly by
  // blockaddresses, but BasicBlock's destructor takes care of those.
  while (!BasicBlocks.empty())
    BasicBlocks.begin()->eraseFromParent();

  // Prefix and prologue data are stored in a side table.
  setPrefixData(nullptr);
  setPrologueData(nullptr);

  // Metadata is stored in a side-table.
  clearMetadata();

  setPersonalityFn(nullptr);
}

void Function::addAttribute(unsigned i, Attribute::AttrKind attr) {
  AttributeSet PAL = getAttributes();
  PAL = PAL.addAttribute(getContext(), i, attr);
  setAttributes(PAL);
}

void Function::addAttributes(unsigned i, AttributeSet attrs) {
  AttributeSet PAL = getAttributes();
  PAL = PAL.addAttributes(getContext(), i, attrs);
  setAttributes(PAL);
}

void Function::removeAttributes(unsigned i, AttributeSet attrs) {
  AttributeSet PAL = getAttributes();
  PAL = PAL.removeAttributes(getContext(), i, attrs);
  setAttributes(PAL);
}

void Function::addDereferenceableAttr(unsigned i, uint64_t Bytes) {
  AttributeSet PAL = getAttributes();
  PAL = PAL.addDereferenceableAttr(getContext(), i, Bytes);
  setAttributes(PAL);
}

void Function::addDereferenceableOrNullAttr(unsigned i, uint64_t Bytes) {
  AttributeSet PAL = getAttributes();
  PAL = PAL.addDereferenceableOrNullAttr(getContext(), i, Bytes);
  setAttributes(PAL);
}

// Maintain the GC name for each function in an on-the-side table. This saves
// allocating an additional word in Function for programs which do not use GC
// (i.e., most programs) at the cost of increased overhead for clients which do
// use GC.
#if 0 // HLSL Change
static DenseMap<const Function*,PooledStringPtr> *GCNames;
static StringPool *GCNamePool;
static ManagedStatic<sys::SmartRWMutex<true> > GCLock;
#endif // HLSL Change

bool Function::hasGC() const {
#if 0 // HLSL Change
  sys::SmartScopedReader<true> Reader(*GCLock);
  return GCNames && GCNames->count(this);
#else
  return false;
#endif // HLSL Change Ends
}

const char *Function::getGC() const {
#if 0 // HLSL Change
  assert(hasGC() && "Function has no collector");
  sys::SmartScopedReader<true> Reader(*GCLock);
  return *(*GCNames)[this];
#else
  return nullptr;
#endif // HLSL Change Ends
}

void Function::setGC(const char *Str) {
#if 0 // HLSL Change Starts
  sys::SmartScopedWriter<true> Writer(*GCLock);
  if (!GCNamePool)
    GCNamePool = new StringPool();
  if (!GCNames)
    GCNames = new DenseMap<const Function*,PooledStringPtr>();
  (*GCNames)[this] = GCNamePool->intern(Str);
#else
  assert(false && "GC not supported");
#endif // HLSL Change Ends
}

void Function::clearGC() {
#if 0 // HLSL Change Starts
  sys::SmartScopedWriter<true> Writer(*GCLock);
  if (GCNames) {
    GCNames->erase(this);
    if (GCNames->empty()) {
      delete GCNames;
      GCNames = nullptr;
      if (GCNamePool->empty()) {
        delete GCNamePool;
        GCNamePool = nullptr;
      }
    }
  }
#endif // HLSL Change Ends
}

/// copyAttributesFrom - copy all additional attributes (those not needed to
/// create a Function) from the Function Src to this one.
void Function::copyAttributesFrom(const GlobalValue *Src) {
  assert(isa<Function>(Src) && "Expected a Function!");
  GlobalObject::copyAttributesFrom(Src);
  const Function *SrcF = cast<Function>(Src);
  setCallingConv(SrcF->getCallingConv());
  setAttributes(SrcF->getAttributes());
  if (SrcF->hasGC())
    setGC(SrcF->getGC());
  else
    clearGC();
  if (SrcF->hasPrefixData())
    setPrefixData(SrcF->getPrefixData());
  else
    setPrefixData(nullptr);
  if (SrcF->hasPrologueData())
    setPrologueData(SrcF->getPrologueData());
  else
    setPrologueData(nullptr);
  if (SrcF->hasPersonalityFn())
    setPersonalityFn(SrcF->getPersonalityFn());
  else
    setPersonalityFn(nullptr);
}

/// \brief This does the actual lookup of an intrinsic ID which
/// matches the given function name.
static Intrinsic::ID lookupIntrinsicID(const ValueName *ValName) {
  unsigned Len = ValName->getKeyLength();
  const char *Name = ValName->getKeyData();

#define GET_FUNCTION_RECOGNIZER
#include "llvm/IR/Intrinsics.gen"
#undef GET_FUNCTION_RECOGNIZER

  return Intrinsic::not_intrinsic;
}

void Function::recalculateIntrinsicID() {
  const ValueName *ValName = this->getValueName();
  if (!ValName || !isIntrinsic()) {
    IntID = Intrinsic::not_intrinsic;
    return;
  }
  IntID = lookupIntrinsicID(ValName);
}

/// Returns a stable mangling for the type specified for use in the name
/// mangling scheme used by 'any' types in intrinsic signatures.  The mangling
/// of named types is simply their name.  Manglings for unnamed types consist
/// of a prefix ('p' for pointers, 'a' for arrays, 'f_' for functions)
/// combined with the mangling of their component types.  A vararg function
/// type will have a suffix of 'vararg'.  Since function types can contain
/// other function types, we close a function type mangling with suffix 'f'
/// which can't be confused with it's prefix.  This ensures we don't have
/// collisions between two unrelated function types. Otherwise, you might
/// parse ffXX as f(fXX) or f(fX)X.  (X is a placeholder for any other type.)
/// Manglings of integers, floats, and vectors ('i', 'f', and 'v' prefix in most
/// cases) fall back to the MVT codepath, where they could be mangled to
/// 'x86mmx', for example; matching on derived types is not sufficient to mangle
/// everything.
static std::string getMangledTypeStr(Type* Ty) {
  std::string Result;
  if (PointerType* PTyp = dyn_cast<PointerType>(Ty)) {
    Result += "p" + llvm::utostr(PTyp->getAddressSpace()) +
      getMangledTypeStr(PTyp->getElementType());
  } else if (ArrayType* ATyp = dyn_cast<ArrayType>(Ty)) {
    Result += "a" + llvm::utostr(ATyp->getNumElements()) +
      getMangledTypeStr(ATyp->getElementType());
  } else if (StructType* STyp = dyn_cast<StructType>(Ty)) {
    assert(!STyp->isLiteral() && "TODO: implement literal types");
    Result += STyp->getName();
  } else if (FunctionType* FT = dyn_cast<FunctionType>(Ty)) {
    Result += "f_" + getMangledTypeStr(FT->getReturnType());
    for (size_t i = 0; i < FT->getNumParams(); i++)
      Result += getMangledTypeStr(FT->getParamType(i));
    if (FT->isVarArg())
      Result += "vararg";
    // Ensure nested function types are distinguishable.
    Result += "f"; 
  } else if (Ty)
    Result += EVT::getEVT(Ty).getEVTString();
  return Result;
}

std::string Intrinsic::getName(_In_range_(0, num_intrinsics-1) ID id, ArrayRef<Type*> Tys) {
  assert(id < num_intrinsics && "Invalid intrinsic ID!");
  static const char * const Table[] = {
    "not_intrinsic",
#define GET_INTRINSIC_NAME_TABLE
#include "llvm/IR/Intrinsics.gen"
#undef GET_INTRINSIC_NAME_TABLE
  };
  if (Tys.empty())
    return Table[id];
  std::string Result(Table[id]);
  for (unsigned i = 0; i < Tys.size(); ++i) {
    Result += "." + getMangledTypeStr(Tys[i]);
  }
  return Result;
}

/// IIT_Info - These are enumerators that describe the entries returned by the
/// getIntrinsicInfoTableEntries function.
///
/// NOTE: This must be kept in synch with the copy in TblGen/IntrinsicEmitter!
enum IIT_Info {
  // Common values should be encoded with 0-15.
  IIT_Done = 0,
  IIT_I1   = 1,
  IIT_I8   = 2,
  IIT_I16  = 3,
  IIT_I32  = 4,
  IIT_I64  = 5,
  IIT_F16  = 6,
  IIT_F32  = 7,
  IIT_F64  = 8,
  IIT_V2   = 9,
  IIT_V4   = 10,
  IIT_V8   = 11,
  IIT_V16  = 12,
  IIT_V32  = 13,
  IIT_PTR  = 14,
  IIT_ARG  = 15,

  // Values from 16+ are only encodable with the inefficient encoding.
  IIT_V64  = 16,
  IIT_MMX  = 17,
  IIT_METADATA = 18,
  IIT_EMPTYSTRUCT = 19,
  IIT_STRUCT2 = 20,
  IIT_STRUCT3 = 21,
  IIT_STRUCT4 = 22,
  IIT_STRUCT5 = 23,
  IIT_EXTEND_ARG = 24,
  IIT_TRUNC_ARG = 25,
  IIT_ANYPTR = 26,
  IIT_V1   = 27,
  IIT_VARARG = 28,
  IIT_HALF_VEC_ARG = 29,
  IIT_SAME_VEC_WIDTH_ARG = 30,
  IIT_PTR_TO_ARG = 31,
  IIT_VEC_OF_PTRS_TO_ELT = 32,
  IIT_I128 = 33
};


static void DecodeIITType(unsigned &NextElt, ArrayRef<unsigned char> Infos,
                      SmallVectorImpl<Intrinsic::IITDescriptor> &OutputTable) {
  IIT_Info Info = IIT_Info(Infos[NextElt++]);
  unsigned StructElts = 2;
  using namespace Intrinsic;

  switch (Info) {
  case IIT_Done:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Void, 0));
    return;
  case IIT_VARARG:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::VarArg, 0));
    return;
  case IIT_MMX:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::MMX, 0));
    return;
  case IIT_METADATA:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Metadata, 0));
    return;
  case IIT_F16:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Half, 0));
    return;
  case IIT_F32:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Float, 0));
    return;
  case IIT_F64:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Double, 0));
    return;
  case IIT_I1:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Integer, 1));
    return;
  case IIT_I8:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Integer, 8));
    return;
  case IIT_I16:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Integer,16));
    return;
  case IIT_I32:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Integer, 32));
    return;
  case IIT_I64:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Integer, 64));
    return;
  case IIT_I128:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Integer, 128));
    return;
  case IIT_V1:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Vector, 1));
    DecodeIITType(NextElt, Infos, OutputTable);
    return;
  case IIT_V2:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Vector, 2));
    DecodeIITType(NextElt, Infos, OutputTable);
    return;
  case IIT_V4:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Vector, 4));
    DecodeIITType(NextElt, Infos, OutputTable);
    return;
  case IIT_V8:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Vector, 8));
    DecodeIITType(NextElt, Infos, OutputTable);
    return;
  case IIT_V16:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Vector, 16));
    DecodeIITType(NextElt, Infos, OutputTable);
    return;
  case IIT_V32:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Vector, 32));
    DecodeIITType(NextElt, Infos, OutputTable);
    return;
  case IIT_V64:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Vector, 64));
    DecodeIITType(NextElt, Infos, OutputTable);
    return;
  case IIT_PTR:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Pointer, 0));
    DecodeIITType(NextElt, Infos, OutputTable);
    return;
  case IIT_ANYPTR: {  // [ANYPTR addrspace, subtype]
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Pointer,
                                             Infos[NextElt++]));
    DecodeIITType(NextElt, Infos, OutputTable);
    return;
  }
  case IIT_ARG: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Argument, ArgInfo));
    return;
  }
  case IIT_EXTEND_ARG: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::ExtendArgument,
                                             ArgInfo));
    return;
  }
  case IIT_TRUNC_ARG: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::TruncArgument,
                                             ArgInfo));
    return;
  }
  case IIT_HALF_VEC_ARG: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::HalfVecArgument,
                                             ArgInfo));
    return;
  }
  case IIT_SAME_VEC_WIDTH_ARG: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::SameVecWidthArgument,
                                             ArgInfo));
    return;
  }
  case IIT_PTR_TO_ARG: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::PtrToArgument,
                                             ArgInfo));
    return;
  }
  case IIT_VEC_OF_PTRS_TO_ELT: {
    unsigned ArgInfo = (NextElt == Infos.size() ? 0 : Infos[NextElt++]);
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::VecOfPtrsToElt,
                                             ArgInfo));
    return;
  }
  case IIT_EMPTYSTRUCT:
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Struct, 0));
    return;
  case IIT_STRUCT5: ++StructElts; // FALL THROUGH.
  case IIT_STRUCT4: ++StructElts; // FALL THROUGH.
  case IIT_STRUCT3: ++StructElts; // FALL THROUGH.
  case IIT_STRUCT2: {
    OutputTable.push_back(IITDescriptor::get(IITDescriptor::Struct,StructElts));

    for (unsigned i = 0; i != StructElts; ++i)
      DecodeIITType(NextElt, Infos, OutputTable);
    return;
  }
  }
  llvm_unreachable("unhandled");
}


#define GET_INTRINSIC_GENERATOR_GLOBAL
#include "llvm/IR/Intrinsics.gen"
#undef GET_INTRINSIC_GENERATOR_GLOBAL

void Intrinsic::getIntrinsicInfoTableEntries(ID id,
                                             SmallVectorImpl<IITDescriptor> &T){
  // Check to see if the intrinsic's type was expressible by the table.
  unsigned TableVal = IIT_Table[id-1];

  // Decode the TableVal into an array of IITValues.
  SmallVector<unsigned char, 8> IITValues;
  ArrayRef<unsigned char> IITEntries;
  unsigned NextElt = 0;
  if ((TableVal >> 31) != 0) {
    // This is an offset into the IIT_LongEncodingTable.
    IITEntries = IIT_LongEncodingTable;

    // Strip sentinel bit.
    NextElt = (TableVal << 1) >> 1;
  } else {
    // Decode the TableVal into an array of IITValues.  If the entry was encoded
    // into a single word in the table itself, decode it now.
    do {
      IITValues.push_back(TableVal & 0xF);
      TableVal >>= 4;
    } while (TableVal);

    IITEntries = IITValues;
    NextElt = 0;
  }

  // Okay, decode the table into the output vector of IITDescriptors.
  DecodeIITType(NextElt, IITEntries, T);
  while (NextElt != IITEntries.size() && IITEntries[NextElt] != 0)
    DecodeIITType(NextElt, IITEntries, T);
}


static Type *DecodeFixedType(ArrayRef<Intrinsic::IITDescriptor> &Infos,
                             ArrayRef<Type*> Tys, LLVMContext &Context) {
  using namespace Intrinsic;
  IITDescriptor D = Infos.front();
  Infos = Infos.slice(1);

  switch (D.Kind) {
  case IITDescriptor::Void: return Type::getVoidTy(Context);
  case IITDescriptor::VarArg: return Type::getVoidTy(Context);
  case IITDescriptor::MMX: return Type::getX86_MMXTy(Context);
  case IITDescriptor::Metadata: return Type::getMetadataTy(Context);
  case IITDescriptor::Half: return Type::getHalfTy(Context);
  case IITDescriptor::Float: return Type::getFloatTy(Context);
  case IITDescriptor::Double: return Type::getDoubleTy(Context);

  case IITDescriptor::Integer:
    return IntegerType::get(Context, D.Integer_Width);
  case IITDescriptor::Vector:
    return VectorType::get(DecodeFixedType(Infos, Tys, Context),D.Vector_Width);
  case IITDescriptor::Pointer:
    return PointerType::get(DecodeFixedType(Infos, Tys, Context),
                            D.Pointer_AddressSpace);
  case IITDescriptor::Struct: {
    Type *Elts[5];
    assert(D.Struct_NumElements <= 5 && "Can't handle this yet");
    for (unsigned i = 0, e = D.Struct_NumElements; i != e && i < _countof(Elts); ++i) // HLSL Change - add extra check to help SAL
      Elts[i] = DecodeFixedType(Infos, Tys, Context);
    return StructType::get(Context, makeArrayRef(Elts,D.Struct_NumElements));
  }

  case IITDescriptor::Argument:
    return Tys[D.getArgumentNumber()];
  case IITDescriptor::ExtendArgument: {
    Type *Ty = Tys[D.getArgumentNumber()];
    if (VectorType *VTy = dyn_cast<VectorType>(Ty))
      return VectorType::getExtendedElementVectorType(VTy);

    return IntegerType::get(Context, 2 * cast<IntegerType>(Ty)->getBitWidth());
  }
  case IITDescriptor::TruncArgument: {
    Type *Ty = Tys[D.getArgumentNumber()];
    if (VectorType *VTy = dyn_cast<VectorType>(Ty))
      return VectorType::getTruncatedElementVectorType(VTy);

    IntegerType *ITy = cast<IntegerType>(Ty);
    assert(ITy->getBitWidth() % 2 == 0);
    return IntegerType::get(Context, ITy->getBitWidth() / 2);
  }
  case IITDescriptor::HalfVecArgument:
    return VectorType::getHalfElementsVectorType(cast<VectorType>(
                                                  Tys[D.getArgumentNumber()]));
  case IITDescriptor::SameVecWidthArgument: {
    Type *EltTy = DecodeFixedType(Infos, Tys, Context);
    Type *Ty = Tys[D.getArgumentNumber()];
    if (VectorType *VTy = dyn_cast<VectorType>(Ty)) {
      return VectorType::get(EltTy, VTy->getNumElements());
    }
    llvm_unreachable("unhandled");
  }
  case IITDescriptor::PtrToArgument: {
    Type *Ty = Tys[D.getArgumentNumber()];
    return PointerType::getUnqual(Ty);
  }
  case IITDescriptor::VecOfPtrsToElt: {
    Type *Ty = Tys[D.getArgumentNumber()];
    VectorType *VTy = dyn_cast<VectorType>(Ty);
    if (!VTy)
      llvm_unreachable("Expected an argument of Vector Type");
    Type *EltTy = VTy->getVectorElementType();
    return VectorType::get(PointerType::getUnqual(EltTy),
                           VTy->getNumElements());
  }
  }
  llvm_unreachable("unhandled");
}



FunctionType *Intrinsic::getType(LLVMContext &Context,
                                 ID id, ArrayRef<Type*> Tys) {
  SmallVector<IITDescriptor, 8> Table;
  getIntrinsicInfoTableEntries(id, Table);

  ArrayRef<IITDescriptor> TableRef = Table;
  Type *ResultTy = DecodeFixedType(TableRef, Tys, Context);

  SmallVector<Type*, 8> ArgTys;
  while (!TableRef.empty())
    ArgTys.push_back(DecodeFixedType(TableRef, Tys, Context));

  // DecodeFixedType returns Void for IITDescriptor::Void and IITDescriptor::VarArg
  // If we see void type as the type of the last argument, it is vararg intrinsic
  if (!ArgTys.empty() && ArgTys.back()->isVoidTy()) {
    ArgTys.pop_back();
    return FunctionType::get(ResultTy, ArgTys, true);
  }
  return FunctionType::get(ResultTy, ArgTys, false);
}

bool Intrinsic::isOverloaded(ID id) {
#define GET_INTRINSIC_OVERLOAD_TABLE
#include "llvm/IR/Intrinsics.gen"
#undef GET_INTRINSIC_OVERLOAD_TABLE
}

bool Intrinsic::isLeaf(ID id) {
  switch (id) {
  default:
    return true;

  case Intrinsic::experimental_gc_statepoint:
  case Intrinsic::experimental_patchpoint_void:
  case Intrinsic::experimental_patchpoint_i64:
    return false;
  }
}

/// This defines the "Intrinsic::getAttributes(ID id)" method.
#define GET_INTRINSIC_ATTRIBUTES
#include "llvm/IR/Intrinsics.gen"
#undef GET_INTRINSIC_ATTRIBUTES

Function *Intrinsic::getDeclaration(Module *M, ID id, ArrayRef<Type*> Tys) {
  // There can never be multiple globals with the same name of different types,
  // because intrinsics must be a specific type.
  return
    cast<Function>(M->getOrInsertFunction(getName(id, Tys),
                                          getType(M->getContext(), id, Tys)));
}

// This defines the "Intrinsic::getIntrinsicForGCCBuiltin()" method.
//#pragma optimize("", off) // HLSL Change - comment out pragma optimize directive
#define GET_LLVM_INTRINSIC_FOR_GCC_BUILTIN
#include "llvm/IR/Intrinsics.gen"
#undef GET_LLVM_INTRINSIC_FOR_GCC_BUILTIN
//#pragma optimize("", on)  // HLSL Change - comment out pragma optimize directive

// This defines the "Intrinsic::getIntrinsicForMSBuiltin()" method.
#define GET_LLVM_INTRINSIC_FOR_MS_BUILTIN
#include "llvm/IR/Intrinsics.gen"
#undef GET_LLVM_INTRINSIC_FOR_MS_BUILTIN

/// hasAddressTaken - returns true if there are any uses of this function
/// other than direct calls or invokes to it.
bool Function::hasAddressTaken(const User* *PutOffender) const {
  for (const Use &U : uses()) {
    const User *FU = U.getUser();
    if (isa<BlockAddress>(FU))
      continue;
    if (!isa<CallInst>(FU) && !isa<InvokeInst>(FU))
      return PutOffender ? (*PutOffender = FU, true) : true;
    ImmutableCallSite CS(cast<Instruction>(FU));
    if (!CS.isCallee(&U))
      return PutOffender ? (*PutOffender = FU, true) : true;
  }
  return false;
}

bool Function::isDefTriviallyDead() const {
  // Check the linkage
  if (!hasLinkOnceLinkage() && !hasLocalLinkage() &&
      !hasAvailableExternallyLinkage())
    return false;

  // Check if the function is used by anything other than a blockaddress.
  for (const User *U : users())
    if (!isa<BlockAddress>(U))
      return false;

  return true;
}

/// callsFunctionThatReturnsTwice - Return true if the function has a call to
/// setjmp or other function that gcc recognizes as "returning twice".
bool Function::callsFunctionThatReturnsTwice() const {
  for (const_inst_iterator
         I = inst_begin(this), E = inst_end(this); I != E; ++I) {
    ImmutableCallSite CS(&*I);
    if (CS && CS.hasFnAttr(Attribute::ReturnsTwice))
      return true;
  }

  return false;
}

Constant *Function::getPrefixData() const {
  assert(hasPrefixData());
  const LLVMContextImpl::PrefixDataMapTy &PDMap =
      getContext().pImpl->PrefixDataMap;
  assert(PDMap.find(this) != PDMap.end());
  return cast<Constant>(PDMap.find(this)->second->getReturnValue());
}

void Function::setPrefixData(Constant *PrefixData) {
  if (!PrefixData && !hasPrefixData())
    return;

  unsigned SCData = getSubclassDataFromValue();
  LLVMContextImpl::PrefixDataMapTy &PDMap = getContext().pImpl->PrefixDataMap;
  ReturnInst *&PDHolder = PDMap[this];
  if (PrefixData) {
    if (PDHolder)
      PDHolder->setOperand(0, PrefixData);
    else
      PDHolder = ReturnInst::Create(getContext(), PrefixData);
    SCData |= (1<<1);
  } else {
    delete PDHolder;
    PDMap.erase(this);
    SCData &= ~(1<<1);
  }
  setValueSubclassData(SCData);
}

Constant *Function::getPrologueData() const {
  assert(hasPrologueData());
  const LLVMContextImpl::PrologueDataMapTy &SOMap =
      getContext().pImpl->PrologueDataMap;
  assert(SOMap.find(this) != SOMap.end());
  return cast<Constant>(SOMap.find(this)->second->getReturnValue());
}

void Function::setPrologueData(Constant *PrologueData) {
  if (!PrologueData && !hasPrologueData())
    return;

  unsigned PDData = getSubclassDataFromValue();
  LLVMContextImpl::PrologueDataMapTy &PDMap = getContext().pImpl->PrologueDataMap;
  ReturnInst *&PDHolder = PDMap[this];
  if (PrologueData) {
    if (PDHolder)
      PDHolder->setOperand(0, PrologueData);
    else
      PDHolder = ReturnInst::Create(getContext(), PrologueData);
    PDData |= (1<<2);
  } else {
    delete PDHolder;
    PDMap.erase(this);
    PDData &= ~(1<<2);
  }
  setValueSubclassData(PDData);
}

void Function::setEntryCount(uint64_t Count) {
  MDBuilder MDB(getContext());
  setMetadata(LLVMContext::MD_prof, MDB.createFunctionEntryCount(Count));
}

Optional<uint64_t> Function::getEntryCount() const {
  MDNode *MD = getMetadata(LLVMContext::MD_prof);
  if (MD && MD->getOperand(0))
    if (MDString *MDS = dyn_cast<MDString>(MD->getOperand(0)))
      if (MDS->getString().equals("function_entry_count")) {
        ConstantInt *CI = mdconst::extract<ConstantInt>(MD->getOperand(1));
        return CI->getValue().getZExtValue();
      }
  return None;
}

void Function::setPersonalityFn(Constant *C) {
  if (!C) {
    if (hasPersonalityFn()) {
      // Note, the num operands is used to compute the offset of the operand, so
      // the order here matters.  Clearing the operand then clearing the num
      // operands ensures we have the correct offset to the operand.
      Op<0>().set(nullptr);
      setFunctionNumOperands(0);
    }
  } else {
    // Note, the num operands is used to compute the offset of the operand, so
    // the order here matters.  We need to set num operands to 1 first so that
    // we get the correct offset to the first operand when we set it.
    if (!hasPersonalityFn())
      setFunctionNumOperands(1);
    Op<0>().set(C);
  }
}
