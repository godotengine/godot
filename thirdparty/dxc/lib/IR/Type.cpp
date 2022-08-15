//===-- Type.cpp - Implement the Type class -------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Type class for the IR library.
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/Type.h"
#include "LLVMContextImpl.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/IR/Module.h"
#include <algorithm>
#include <cstdarg>
using namespace llvm;

//===----------------------------------------------------------------------===//
//                         Type Class Implementation
//===----------------------------------------------------------------------===//

Type *Type::getPrimitiveType(LLVMContext &C, TypeID IDNumber) {
  switch (IDNumber) {
  case VoidTyID      : return getVoidTy(C);
  case HalfTyID      : return getHalfTy(C);
  case FloatTyID     : return getFloatTy(C);
  case DoubleTyID    : return getDoubleTy(C);
  case X86_FP80TyID  : return getX86_FP80Ty(C);
  case FP128TyID     : return getFP128Ty(C);
  case PPC_FP128TyID : return getPPC_FP128Ty(C);
  case LabelTyID     : return getLabelTy(C);
  case MetadataTyID  : return getMetadataTy(C);
  case X86_MMXTyID   : return getX86_MMXTy(C);
  default:
    return nullptr;
  }
}

/// getScalarType - If this is a vector type, return the element type,
/// otherwise return this.
Type *Type::getScalarType() {
  if (VectorType *VTy = dyn_cast<VectorType>(this))
    return VTy->getElementType();
  return this;
}

const Type *Type::getScalarType() const {
  if (const VectorType *VTy = dyn_cast<VectorType>(this))
    return VTy->getElementType();
  return this;
}

/// isIntegerTy - Return true if this is an IntegerType of the specified width.
bool Type::isIntegerTy(unsigned Bitwidth) const {
  return isIntegerTy() && cast<IntegerType>(this)->getBitWidth() == Bitwidth;
}

// canLosslesslyBitCastTo - Return true if this type can be converted to
// 'Ty' without any reinterpretation of bits.  For example, i8* to i32*.
//
bool Type::canLosslesslyBitCastTo(_In_ Type *Ty) const {
  // Identity cast means no change so return true
  if (this == Ty) 
    return true;
  
  // They are not convertible unless they are at least first class types
  if (!this->isFirstClassType() || !Ty->isFirstClassType())
    return false;

  // Vector -> Vector conversions are always lossless if the two vector types
  // have the same size, otherwise not.  Also, 64-bit vector types can be
  // converted to x86mmx.
  if (const VectorType *thisPTy = dyn_cast<VectorType>(this)) {
    if (const VectorType *thatPTy = dyn_cast<VectorType>(Ty))
      return thisPTy->getBitWidth() == thatPTy->getBitWidth();
    if (Ty->getTypeID() == Type::X86_MMXTyID &&
        thisPTy->getBitWidth() == 64)
      return true;
  }

  if (this->getTypeID() == Type::X86_MMXTyID)
    if (const VectorType *thatPTy = dyn_cast<VectorType>(Ty))
      if (thatPTy->getBitWidth() == 64)
        return true;

  // At this point we have only various mismatches of the first class types
  // remaining and ptr->ptr. Just select the lossless conversions. Everything
  // else is not lossless. Conservatively assume we can't losslessly convert
  // between pointers with different address spaces.
  if (const PointerType *PTy = dyn_cast<PointerType>(this)) {
    if (const PointerType *OtherPTy = dyn_cast<PointerType>(Ty))
      return PTy->getAddressSpace() == OtherPTy->getAddressSpace();
    return false;
  }
  return false;  // Other types have no identity values
}

bool Type::isEmptyTy() const {
  const ArrayType *ATy = dyn_cast<ArrayType>(this);
  if (ATy) {
    unsigned NumElements = ATy->getNumElements();
    return NumElements == 0 || ATy->getElementType()->isEmptyTy();
  }

  const StructType *STy = dyn_cast<StructType>(this);
  if (STy) {
    unsigned NumElements = STy->getNumElements();
    for (unsigned i = 0; i < NumElements; ++i)
      if (!STy->getElementType(i)->isEmptyTy())
        return false;
    return true;
  }

  return false;
}

unsigned Type::getPrimitiveSizeInBits() const {
  switch (getTypeID()) {
  case Type::HalfTyID: return 16;
  case Type::FloatTyID: return 32;
  case Type::DoubleTyID: return 64;
  case Type::X86_FP80TyID: return 80;
  case Type::FP128TyID: return 128;
  case Type::PPC_FP128TyID: return 128;
  case Type::X86_MMXTyID: return 64;
  case Type::IntegerTyID: return cast<IntegerType>(this)->getBitWidth();
  case Type::VectorTyID:  return cast<VectorType>(this)->getBitWidth();
  default: return 0;
  }
}

/// getScalarSizeInBits - If this is a vector type, return the
/// getPrimitiveSizeInBits value for the element type. Otherwise return the
/// getPrimitiveSizeInBits value for this type.
unsigned Type::getScalarSizeInBits() const {
  return getScalarType()->getPrimitiveSizeInBits();
}

/// getFPMantissaWidth - Return the width of the mantissa of this type.  This
/// is only valid on floating point types.  If the FP type does not
/// have a stable mantissa (e.g. ppc long double), this method returns -1.
int Type::getFPMantissaWidth() const {
  if (const VectorType *VTy = dyn_cast<VectorType>(this))
    return VTy->getElementType()->getFPMantissaWidth();
  assert(isFloatingPointTy() && "Not a floating point type!");
  if (getTypeID() == HalfTyID) return 11;
  if (getTypeID() == FloatTyID) return 24;
  if (getTypeID() == DoubleTyID) return 53;
  if (getTypeID() == X86_FP80TyID) return 64;
  if (getTypeID() == FP128TyID) return 113;
  assert(getTypeID() == PPC_FP128TyID && "unknown fp type");
  return -1;
}

/// isSizedDerivedType - Derived types like structures and arrays are sized
/// iff all of the members of the type are sized as well.  Since asking for
/// their size is relatively uncommon, move this operation out of line.
bool Type::isSizedDerivedType(SmallPtrSetImpl<const Type*> *Visited) const {
  if (const ArrayType *ATy = dyn_cast<ArrayType>(this))
    return ATy->getElementType()->isSized(Visited);

  if (const VectorType *VTy = dyn_cast<VectorType>(this))
    return VTy->getElementType()->isSized(Visited);

  return cast<StructType>(this)->isSized(Visited);
}

//===----------------------------------------------------------------------===//
//                         Subclass Helper Methods
//===----------------------------------------------------------------------===//

unsigned Type::getIntegerBitWidth() const {
  return cast<IntegerType>(this)->getBitWidth();
}

bool Type::isFunctionVarArg() const {
  return cast<FunctionType>(this)->isVarArg();
}

Type *Type::getFunctionParamType(unsigned i) const {
  return cast<FunctionType>(this)->getParamType(i);
}

unsigned Type::getFunctionNumParams() const {
  return cast<FunctionType>(this)->getNumParams();
}

StringRef Type::getStructName() const {
  return cast<StructType>(this)->getName();
}

unsigned Type::getStructNumElements() const {
  return cast<StructType>(this)->getNumElements();
}

Type *Type::getStructElementType(unsigned N) const {
  return cast<StructType>(this)->getElementType(N);
}

Type *Type::getSequentialElementType() const {
  return cast<SequentialType>(this)->getElementType();
}

uint64_t Type::getArrayNumElements() const {
  return cast<ArrayType>(this)->getNumElements();
}

unsigned Type::getVectorNumElements() const {
  return cast<VectorType>(this)->getNumElements();
}

unsigned Type::getPointerAddressSpace() const {
  return cast<PointerType>(getScalarType())->getAddressSpace();
}


//===----------------------------------------------------------------------===//
//                          Primitive 'Type' data
//===----------------------------------------------------------------------===//

Type *Type::getVoidTy(LLVMContext &C) { return &C.pImpl->VoidTy; }
Type *Type::getLabelTy(LLVMContext &C) { return &C.pImpl->LabelTy; }
Type *Type::getHalfTy(LLVMContext &C) { return &C.pImpl->HalfTy; }
Type *Type::getFloatTy(LLVMContext &C) { return &C.pImpl->FloatTy; }
Type *Type::getDoubleTy(LLVMContext &C) { return &C.pImpl->DoubleTy; }
Type *Type::getMetadataTy(LLVMContext &C) { return &C.pImpl->MetadataTy; }
Type *Type::getX86_FP80Ty(LLVMContext &C) { return &C.pImpl->X86_FP80Ty; }
Type *Type::getFP128Ty(LLVMContext &C) { return &C.pImpl->FP128Ty; }
Type *Type::getPPC_FP128Ty(LLVMContext &C) { return &C.pImpl->PPC_FP128Ty; }
Type *Type::getX86_MMXTy(LLVMContext &C) { return &C.pImpl->X86_MMXTy; }

IntegerType *Type::getInt1Ty(LLVMContext &C) { return &C.pImpl->Int1Ty; }
IntegerType *Type::getInt8Ty(LLVMContext &C) { return &C.pImpl->Int8Ty; }
IntegerType *Type::getInt16Ty(LLVMContext &C) { return &C.pImpl->Int16Ty; }
IntegerType *Type::getInt32Ty(LLVMContext &C) { return &C.pImpl->Int32Ty; }
IntegerType *Type::getInt64Ty(LLVMContext &C) { return &C.pImpl->Int64Ty; }
IntegerType *Type::getInt128Ty(LLVMContext &C) { return &C.pImpl->Int128Ty; }

IntegerType *Type::getIntNTy(LLVMContext &C, unsigned N) {
  return IntegerType::get(C, N);
}

PointerType *Type::getHalfPtrTy(LLVMContext &C, unsigned AS) {
  return getHalfTy(C)->getPointerTo(AS);
}

PointerType *Type::getFloatPtrTy(LLVMContext &C, unsigned AS) {
  return getFloatTy(C)->getPointerTo(AS);
}

PointerType *Type::getDoublePtrTy(LLVMContext &C, unsigned AS) {
  return getDoubleTy(C)->getPointerTo(AS);
}

PointerType *Type::getX86_FP80PtrTy(LLVMContext &C, unsigned AS) {
  return getX86_FP80Ty(C)->getPointerTo(AS);
}

PointerType *Type::getFP128PtrTy(LLVMContext &C, unsigned AS) {
  return getFP128Ty(C)->getPointerTo(AS);
}

PointerType *Type::getPPC_FP128PtrTy(LLVMContext &C, unsigned AS) {
  return getPPC_FP128Ty(C)->getPointerTo(AS);
}

PointerType *Type::getX86_MMXPtrTy(LLVMContext &C, unsigned AS) {
  return getX86_MMXTy(C)->getPointerTo(AS);
}

PointerType *Type::getIntNPtrTy(LLVMContext &C, unsigned N, unsigned AS) {
  return getIntNTy(C, N)->getPointerTo(AS);
}

PointerType *Type::getInt1PtrTy(LLVMContext &C, unsigned AS) {
  return getInt1Ty(C)->getPointerTo(AS);
}

PointerType *Type::getInt8PtrTy(LLVMContext &C, unsigned AS) {
  return getInt8Ty(C)->getPointerTo(AS);
}

PointerType *Type::getInt16PtrTy(LLVMContext &C, unsigned AS) {
  return getInt16Ty(C)->getPointerTo(AS);
}

PointerType *Type::getInt32PtrTy(LLVMContext &C, unsigned AS) {
  return getInt32Ty(C)->getPointerTo(AS);
}

PointerType *Type::getInt64PtrTy(LLVMContext &C, unsigned AS) {
  return getInt64Ty(C)->getPointerTo(AS);
}


//===----------------------------------------------------------------------===//
//                       IntegerType Implementation
//===----------------------------------------------------------------------===//

IntegerType *IntegerType::get(LLVMContext &C, unsigned NumBits) {
  assert(NumBits >= MIN_INT_BITS && "bitwidth too small");
  assert(NumBits <= MAX_INT_BITS && "bitwidth too large");
  
  // Check for the built-in integer types
  switch (NumBits) {
  case   1: return cast<IntegerType>(Type::getInt1Ty(C));
  case   8: return cast<IntegerType>(Type::getInt8Ty(C));
  case  16: return cast<IntegerType>(Type::getInt16Ty(C));
  case  32: return cast<IntegerType>(Type::getInt32Ty(C));
  case  64: return cast<IntegerType>(Type::getInt64Ty(C));
  case 128: return cast<IntegerType>(Type::getInt128Ty(C));
  default:
    break;
  }
  
  IntegerType *&Entry = C.pImpl->IntegerTypes[NumBits];

  if (!Entry)
    Entry = new (C.pImpl->TypeAllocator) IntegerType(C, NumBits);
  
  return Entry;
}

bool IntegerType::isPowerOf2ByteWidth() const {
  unsigned BitWidth = getBitWidth();
  return (BitWidth > 7) && isPowerOf2_32(BitWidth);
}

APInt IntegerType::getMask() const {
  return APInt::getAllOnesValue(getBitWidth());
}

//===----------------------------------------------------------------------===//
//                       FunctionType Implementation
//===----------------------------------------------------------------------===//

FunctionType::FunctionType(Type *Result, ArrayRef<Type*> Params,
                           bool IsVarArgs)
  : Type(Result->getContext(), FunctionTyID) {
  Type **SubTys = reinterpret_cast<Type**>(this+1);
  assert(isValidReturnType(Result) && "invalid return type for function");
  setSubclassData(IsVarArgs);

  SubTys[0] = const_cast<Type*>(Result);

  for (unsigned i = 0, e = Params.size(); i != e; ++i) {
    assert(isValidArgumentType(Params[i]) &&
           "Not a valid type for function argument!");
    SubTys[i+1] = Params[i];
  }

  ContainedTys = SubTys;
  NumContainedTys = Params.size() + 1; // + 1 for result type
}

// FunctionType::get - The factory function for the FunctionType class.
FunctionType *FunctionType::get(Type *ReturnType,
                                ArrayRef<Type*> Params, bool isVarArg) {
  LLVMContextImpl *pImpl = ReturnType->getContext().pImpl;
  FunctionTypeKeyInfo::KeyTy Key(ReturnType, Params, isVarArg);
  auto I = pImpl->FunctionTypes.find_as(Key);
  FunctionType *FT;

  if (I == pImpl->FunctionTypes.end()) {
    FT = (FunctionType*) pImpl->TypeAllocator.
      Allocate(sizeof(FunctionType) + sizeof(Type*) * (Params.size() + 1),
               AlignOf<FunctionType>::Alignment);
    new (FT) FunctionType(ReturnType, Params, isVarArg);
    pImpl->FunctionTypes.insert(FT);
  } else {
    FT = *I;
  }

  return FT;
}

FunctionType *FunctionType::get(Type *Result, bool isVarArg) {
  return get(Result, None, isVarArg);
}

/// isValidReturnType - Return true if the specified type is valid as a return
/// type.
bool FunctionType::isValidReturnType(Type *RetTy) {
  return !RetTy->isFunctionTy() && !RetTy->isLabelTy() &&
  !RetTy->isMetadataTy();
}

/// isValidArgumentType - Return true if the specified type is valid as an
/// argument type.
bool FunctionType::isValidArgumentType(Type *ArgTy) {
  return ArgTy->isFirstClassType();
}

//===----------------------------------------------------------------------===//
//                       StructType Implementation
//===----------------------------------------------------------------------===//

// Primitive Constructors.

StructType *StructType::get(LLVMContext &Context, ArrayRef<Type*> ETypes, 
                            bool isPacked) {
  LLVMContextImpl *pImpl = Context.pImpl;
  AnonStructTypeKeyInfo::KeyTy Key(ETypes, isPacked);
  auto I = pImpl->AnonStructTypes.find_as(Key);
  StructType *ST;

  if (I == pImpl->AnonStructTypes.end()) {
    // Value not found.  Create a new type!
    ST = new (Context.pImpl->TypeAllocator) StructType(Context);
    ST->setSubclassData(SCDB_IsLiteral);  // Literal struct.
    ST->setBody(ETypes, isPacked);
    Context.pImpl->AnonStructTypes.insert(ST);
  } else {
    ST = *I;
  }

  return ST;
}

void StructType::setBody(ArrayRef<Type*> Elements, bool isPacked) {
  assert(isOpaque() && "Struct body already set!");
  
  setSubclassData(getSubclassData() | SCDB_HasBody);
  if (isPacked)
    setSubclassData(getSubclassData() | SCDB_Packed);

  unsigned NumElements = Elements.size();
  Type **Elts = getContext().pImpl->TypeAllocator.Allocate<Type*>(NumElements);
  memcpy(Elts, Elements.data(), sizeof(Elements[0]) * NumElements);
  
  ContainedTys = Elts;
  NumContainedTys = NumElements;
}

void StructType::setName(StringRef Name) {
  if (Name == getName()) return;

  StringMap<StructType *> &SymbolTable = getContext().pImpl->NamedStructTypes;
  typedef StringMap<StructType *>::MapEntryTy EntryTy;

  // If this struct already had a name, remove its symbol table entry. Don't
  // delete the data yet because it may be part of the new name.
  if (SymbolTableEntry)
    SymbolTable.remove((EntryTy *)SymbolTableEntry);

  // If this is just removing the name, we're done.
  if (Name.empty()) {
    if (SymbolTableEntry) {
      // Delete the old string data.
      ((EntryTy *)SymbolTableEntry)->Destroy(SymbolTable.getAllocator());
      SymbolTableEntry = nullptr;
    }
    return;
  }
  
  // Look up the entry for the name.
  auto IterBool =
      getContext().pImpl->NamedStructTypes.insert(std::make_pair(Name, this));

  // While we have a name collision, try a random rename.
  if (!IterBool.second) {
    SmallString<64> TempStr(Name);
    TempStr.push_back('.');
    raw_svector_ostream TmpStream(TempStr);
    unsigned NameSize = Name.size();
   
    do {
      TempStr.resize(NameSize + 1);
      TmpStream.resync();
      TmpStream << getContext().pImpl->NamedStructTypesUniqueID++;

      IterBool = getContext().pImpl->NamedStructTypes.insert(
          std::make_pair(TmpStream.str(), this));
    } while (!IterBool.second);
  }

  // Delete the old string data.
  if (SymbolTableEntry)
    ((EntryTy *)SymbolTableEntry)->Destroy(SymbolTable.getAllocator());
  SymbolTableEntry = &*IterBool.first;
}

//===----------------------------------------------------------------------===//
// StructType Helper functions.

StructType *StructType::create(LLVMContext &Context, StringRef Name) {
  StructType *ST = new (Context.pImpl->TypeAllocator) StructType(Context);
  if (!Name.empty())
    ST->setName(Name);
  return ST;
}

StructType *StructType::get(LLVMContext &Context, bool isPacked) {
  return get(Context, None, isPacked);
}

StructType *StructType::get(Type *type, ...) {
  assert(type && "Cannot create a struct type with no elements with this");
  LLVMContext &Ctx = type->getContext();
  va_list ap;
  SmallVector<llvm::Type*, 8> StructFields;
  va_start(ap, type);
  while (type) {
    StructFields.push_back(type);
    type = va_arg(ap, llvm::Type*);
  }
  auto *Ret = llvm::StructType::get(Ctx, StructFields);
  va_end(ap);
  return Ret;
}

StructType *StructType::create(LLVMContext &Context, ArrayRef<Type*> Elements,
                               StringRef Name, bool isPacked) {
  StructType *ST = create(Context, Name);
  ST->setBody(Elements, isPacked);
  return ST;
}

StructType *StructType::create(LLVMContext &Context, ArrayRef<Type*> Elements) {
  return create(Context, Elements, StringRef());
}

StructType *StructType::create(LLVMContext &Context) {
  return create(Context, StringRef());
}

StructType *StructType::create(ArrayRef<Type*> Elements, StringRef Name,
                               bool isPacked) {
  assert(!Elements.empty() &&
         "This method may not be invoked with an empty list");
  return create(Elements[0]->getContext(), Elements, Name, isPacked);
}

StructType *StructType::create(ArrayRef<Type*> Elements) {
  assert(!Elements.empty() &&
         "This method may not be invoked with an empty list");
  return create(Elements[0]->getContext(), Elements, StringRef());
}

StructType *StructType::create(StringRef Name, Type *type, ...) {
  assert(type && "Cannot create a struct type with no elements with this");
  LLVMContext &Ctx = type->getContext();
  va_list ap;
  SmallVector<llvm::Type*, 8> StructFields;
  va_start(ap, type);
  while (type) {
    StructFields.push_back(type);
    type = va_arg(ap, llvm::Type*);
  }
  auto *Ret = llvm::StructType::create(Ctx, StructFields, Name);
  va_end(ap);
  return Ret;
}

bool StructType::isSized(SmallPtrSetImpl<const Type*> *Visited) const {
  if ((getSubclassData() & SCDB_IsSized) != 0)
    return true;
  if (isOpaque())
    return false;

  if (Visited && !Visited->insert(this).second)
    return false;

  // Okay, our struct is sized if all of the elements are, but if one of the
  // elements is opaque, the struct isn't sized *yet*, but may become sized in
  // the future, so just bail out without caching.
  for (element_iterator I = element_begin(), E = element_end(); I != E; ++I)
    if (!(*I)->isSized(Visited))
      return false;

  // Here we cheat a bit and cast away const-ness. The goal is to memoize when
  // we find a sized type, as types can only move from opaque to sized, not the
  // other way.
  const_cast<StructType*>(this)->setSubclassData(
    getSubclassData() | SCDB_IsSized);
  return true;
}

StringRef StructType::getName() const {
  assert(!isLiteral() && "Literal structs never have names");
  if (!SymbolTableEntry) return StringRef();

  return ((StringMapEntry<StructType*> *)SymbolTableEntry)->getKey();
}

void StructType::setBody(Type *type, ...) {
  assert(type && "Cannot create a struct type with no elements with this");
  va_list ap;
  SmallVector<llvm::Type*, 8> StructFields;
  va_start(ap, type);
  while (type) {
    StructFields.push_back(type);
    type = va_arg(ap, llvm::Type*);
  }
  setBody(StructFields);
  va_end(ap);
}

bool StructType::isValidElementType(Type *ElemTy) {
  return !ElemTy->isVoidTy() && !ElemTy->isLabelTy() &&
         !ElemTy->isMetadataTy() && !ElemTy->isFunctionTy();
}

/// isLayoutIdentical - Return true if this is layout identical to the
/// specified struct.
bool StructType::isLayoutIdentical(StructType *Other) const {
  if (this == Other) return true;
  
  if (isPacked() != Other->isPacked() ||
      getNumElements() != Other->getNumElements())
    return false;

  if (!getNumElements())
    return true;
  
  return std::equal(element_begin(), element_end(), Other->element_begin());
}

/// getTypeByName - Return the type with the specified name, or null if there
/// is none by that name.
StructType *Module::getTypeByName(StringRef Name) const {
  return getContext().pImpl->NamedStructTypes.lookup(Name);
}


//===----------------------------------------------------------------------===//
//                       CompositeType Implementation
//===----------------------------------------------------------------------===//

Type *CompositeType::getTypeAtIndex(const Value *V) {
  if (StructType *STy = dyn_cast<StructType>(this)) {
    unsigned Idx =
      (unsigned)cast<Constant>(V)->getUniqueInteger().getZExtValue();
    assert(indexValid(Idx) && "Invalid structure index!");
    return STy->getElementType(Idx);
  }

  return cast<SequentialType>(this)->getElementType();
}
Type *CompositeType::getTypeAtIndex(unsigned Idx) {
  if (StructType *STy = dyn_cast<StructType>(this)) {
    assert(indexValid(Idx) && "Invalid structure index!");
    return STy->getElementType(Idx);
  }
  
  return cast<SequentialType>(this)->getElementType();
}
bool CompositeType::indexValid(const Value *V) const {
  if (const StructType *STy = dyn_cast<StructType>(this)) {
    // Structure indexes require (vectors of) 32-bit integer constants.  In the
    // vector case all of the indices must be equal.
    if (!V->getType()->getScalarType()->isIntegerTy(32))
      return false;
    const Constant *C = dyn_cast<Constant>(V);
    if (C && V->getType()->isVectorTy())
      C = C->getSplatValue();
    const ConstantInt *CU = dyn_cast_or_null<ConstantInt>(C);
    return CU && CU->getZExtValue() < STy->getNumElements();
  }

  // Sequential types can be indexed by any integer.
  return V->getType()->isIntOrIntVectorTy();
}

bool CompositeType::indexValid(unsigned Idx) const {
  if (const StructType *STy = dyn_cast<StructType>(this))
    return Idx < STy->getNumElements();
  // Sequential types can be indexed by any integer.
  return true;
}


//===----------------------------------------------------------------------===//
//                           ArrayType Implementation
//===----------------------------------------------------------------------===//

ArrayType::ArrayType(Type *ElType, uint64_t NumEl)
  : SequentialType(ArrayTyID, ElType) {
  NumElements = NumEl;
}

ArrayType *ArrayType::get(Type *elementType, uint64_t NumElements) {
  Type *ElementType = const_cast<Type*>(elementType);
  assert(isValidElementType(ElementType) && "Invalid type for array element!");
    
  LLVMContextImpl *pImpl = ElementType->getContext().pImpl;
  ArrayType *&Entry = 
    pImpl->ArrayTypes[std::make_pair(ElementType, NumElements)];

  if (!Entry)
    Entry = new (pImpl->TypeAllocator) ArrayType(ElementType, NumElements);
  return Entry;
}

bool ArrayType::isValidElementType(Type *ElemTy) {
  return !ElemTy->isVoidTy() && !ElemTy->isLabelTy() &&
         !ElemTy->isMetadataTy() && !ElemTy->isFunctionTy();
}

//===----------------------------------------------------------------------===//
//                          VectorType Implementation
//===----------------------------------------------------------------------===//

VectorType::VectorType(Type *ElType, unsigned NumEl)
  : SequentialType(VectorTyID, ElType) {
  NumElements = NumEl;
}

VectorType *VectorType::get(Type *elementType, unsigned NumElements) {
  Type *ElementType = const_cast<Type*>(elementType);
  assert(NumElements > 0 && "#Elements of a VectorType must be greater than 0");
  assert(isValidElementType(ElementType) && "Element type of a VectorType must "
                                            "be an integer, floating point, or "
                                            "pointer type.");

  LLVMContextImpl *pImpl = ElementType->getContext().pImpl;
  VectorType *&Entry = ElementType->getContext().pImpl
    ->VectorTypes[std::make_pair(ElementType, NumElements)];

  if (!Entry)
    Entry = new (pImpl->TypeAllocator) VectorType(ElementType, NumElements);
  return Entry;
}

bool VectorType::isValidElementType(Type *ElemTy) {
  return ElemTy->isIntegerTy() || ElemTy->isFloatingPointTy() ||
    ElemTy->isPointerTy();
}

//===----------------------------------------------------------------------===//
//                         PointerType Implementation
//===----------------------------------------------------------------------===//

PointerType *PointerType::get(Type *EltTy, unsigned AddressSpace) {
  assert(EltTy && "Can't get a pointer to <null> type!");
  assert(isValidElementType(EltTy) && "Invalid type for pointer element!");
  
  LLVMContextImpl *CImpl = EltTy->getContext().pImpl;
  
  // Since AddressSpace #0 is the common case, we special case it.
  PointerType *&Entry = AddressSpace == 0 ? CImpl->PointerTypes[EltTy]
     : CImpl->ASPointerTypes[std::make_pair(EltTy, AddressSpace)];

  if (!Entry)
    Entry = new (CImpl->TypeAllocator) PointerType(EltTy, AddressSpace);
  return Entry;
}


PointerType::PointerType(Type *E, unsigned AddrSpace)
  : SequentialType(PointerTyID, E) {
#ifndef NDEBUG
  const unsigned oldNCT = NumContainedTys;
#endif
  setSubclassData(AddrSpace);
  // Check for miscompile. PR11652.
  assert(oldNCT == NumContainedTys && "bitfield written out of bounds?");
}

PointerType *Type::getPointerTo(unsigned addrs) {
  return PointerType::get(this, addrs);
}

bool PointerType::isValidElementType(Type *ElemTy) {
  return !ElemTy->isVoidTy() && !ElemTy->isLabelTy() &&
         !ElemTy->isMetadataTy();
}

bool PointerType::isLoadableOrStorableType(Type *ElemTy) {
  return isValidElementType(ElemTy) && !ElemTy->isFunctionTy();
}
