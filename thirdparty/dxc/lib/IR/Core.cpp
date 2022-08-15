//===-- Core.cpp ----------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the common infrastructure (including the C bindings)
// for libLLVMCore.a, which implements the LLVM intermediate representation.
//
//===----------------------------------------------------------------------===//

#include "llvm-c/Core.h"
#include "llvm/Bitcode/ReaderWriter.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallSite.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/GlobalAlias.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/raw_ostream.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <system_error>

using namespace llvm;

#define DEBUG_TYPE "ir"

void llvm::initializeCore(PassRegistry &Registry) {
  initializeDominatorTreeWrapperPassPass(Registry);
  initializePrintModulePassWrapperPass(Registry);
  initializePrintFunctionPassWrapperPass(Registry);
  initializePrintBasicBlockPassPass(Registry);
  initializeVerifierLegacyPassPass(Registry);
}

void LLVMInitializeCore(LLVMPassRegistryRef R) {
  initializeCore(*unwrap(R));
}

void LLVMShutdown() {
  llvm_shutdown();
}

// HLSL Change: use ISO _strdup rather than strdup, which compiled but fails to link
#ifdef _MSC_VER
#define strdup _strdup
#endif

/*===-- Error handling ----------------------------------------------------===*/

char *LLVMCreateMessage(const char *Message) {
  return strdup(Message);
}

void LLVMDisposeMessage(char *Message) {
  free(Message);
}


/*===-- Operations on contexts --------------------------------------------===*/

LLVMContextRef LLVMContextCreate() {
  return wrap(new LLVMContext());
}

LLVMContextRef LLVMGetGlobalContext() {
  return wrap(&getGlobalContext());
}

void LLVMContextSetDiagnosticHandler(LLVMContextRef C,
                                     LLVMDiagnosticHandler Handler,
                                     void *DiagnosticContext) {
  unwrap(C)->setDiagnosticHandler(
      LLVM_EXTENSION reinterpret_cast<LLVMContext::DiagnosticHandlerTy>(Handler),
      DiagnosticContext);
}

void LLVMContextSetYieldCallback(LLVMContextRef C, LLVMYieldCallback Callback,
                                 void *OpaqueHandle) {
  auto YieldCallback =
    LLVM_EXTENSION reinterpret_cast<LLVMContext::YieldCallbackTy>(Callback);
  unwrap(C)->setYieldCallback(YieldCallback, OpaqueHandle);
}

void LLVMContextDispose(LLVMContextRef C) {
  delete unwrap(C);
}

unsigned LLVMGetMDKindIDInContext(LLVMContextRef C, const char* Name,
                                  unsigned SLen) {
  return unwrap(C)->getMDKindID(StringRef(Name, SLen));
}

unsigned LLVMGetMDKindID(const char* Name, unsigned SLen) {
  return LLVMGetMDKindIDInContext(LLVMGetGlobalContext(), Name, SLen);
}

char *LLVMGetDiagInfoDescription(LLVMDiagnosticInfoRef DI) {
  std::string MsgStorage;
  raw_string_ostream Stream(MsgStorage);
  DiagnosticPrinterRawOStream DP(Stream);

  unwrap(DI)->print(DP);
  Stream.flush();

  return LLVMCreateMessage(MsgStorage.c_str());
}

LLVMDiagnosticSeverity LLVMGetDiagInfoSeverity(LLVMDiagnosticInfoRef DI){
    LLVMDiagnosticSeverity severity;

    switch(unwrap(DI)->getSeverity()) {
    default:
      severity = LLVMDSError;
      break;
    case DS_Warning:
      severity = LLVMDSWarning;
      break;
    case DS_Remark:
      severity = LLVMDSRemark;
      break;
    case DS_Note:
      severity = LLVMDSNote;
      break;
    }

    return severity;
}




/*===-- Operations on modules ---------------------------------------------===*/

LLVMModuleRef LLVMModuleCreateWithName(const char *ModuleID) {
  return wrap(new Module(ModuleID, getGlobalContext()));
}

LLVMModuleRef LLVMModuleCreateWithNameInContext(const char *ModuleID,
                                                LLVMContextRef C) {
  return wrap(new Module(ModuleID, *unwrap(C)));
}

void LLVMDisposeModule(LLVMModuleRef M) {
  delete unwrap(M);
}

/*--.. Data layout .........................................................--*/
const char * LLVMGetDataLayout(LLVMModuleRef M) {
  return unwrap(M)->getDataLayoutStr().c_str();
}

void LLVMSetDataLayout(LLVMModuleRef M, const char *Triple) {
  unwrap(M)->setDataLayout(Triple);
}

/*--.. Target triple .......................................................--*/
const char * LLVMGetTarget(LLVMModuleRef M) {
  return unwrap(M)->getTargetTriple().c_str();
}

void LLVMSetTarget(LLVMModuleRef M, const char *Triple) {
  unwrap(M)->setTargetTriple(Triple);
}

void LLVMDumpModule(LLVMModuleRef M) {
  unwrap(M)->dump();
}

LLVMBool LLVMPrintModuleToFile(LLVMModuleRef M, const char *Filename,
                               char **ErrorMessage) {
  std::error_code EC;
  raw_fd_ostream dest(Filename, EC, sys::fs::F_Text);
  if (EC) {
    *ErrorMessage = strdup(EC.message().c_str());
    return true;
  }

  unwrap(M)->print(dest, nullptr);

  dest.close();

  if (dest.has_error()) {
    *ErrorMessage = strdup("Error printing to file");
    return true;
  }

  return false;
}

char *LLVMPrintModuleToString(LLVMModuleRef M) {
  std::string buf;
  raw_string_ostream os(buf);

  unwrap(M)->print(os, nullptr);
  os.flush();

  return strdup(buf.c_str());
}

/*--.. Operations on inline assembler ......................................--*/
void LLVMSetModuleInlineAsm(LLVMModuleRef M, const char *Asm) {
  unwrap(M)->setModuleInlineAsm(StringRef(Asm));
}


/*--.. Operations on module contexts ......................................--*/
LLVMContextRef LLVMGetModuleContext(LLVMModuleRef M) {
  return wrap(&unwrap(M)->getContext());
}


/*===-- Operations on types -----------------------------------------------===*/

/*--.. Operations on all types (mostly) ....................................--*/

LLVMTypeKind LLVMGetTypeKind(LLVMTypeRef Ty) {
  switch (unwrap(Ty)->getTypeID()) {
  case Type::VoidTyID:
    return LLVMVoidTypeKind;
  case Type::HalfTyID:
    return LLVMHalfTypeKind;
  case Type::FloatTyID:
    return LLVMFloatTypeKind;
  case Type::DoubleTyID:
    return LLVMDoubleTypeKind;
  case Type::X86_FP80TyID:
    return LLVMX86_FP80TypeKind;
  case Type::FP128TyID:
    return LLVMFP128TypeKind;
  case Type::PPC_FP128TyID:
    return LLVMPPC_FP128TypeKind;
  case Type::LabelTyID:
    return LLVMLabelTypeKind;
  case Type::MetadataTyID:
    return LLVMMetadataTypeKind;
  case Type::IntegerTyID:
    return LLVMIntegerTypeKind;
  case Type::FunctionTyID:
    return LLVMFunctionTypeKind;
  case Type::StructTyID:
    return LLVMStructTypeKind;
  case Type::ArrayTyID:
    return LLVMArrayTypeKind;
  case Type::PointerTyID:
    return LLVMPointerTypeKind;
  case Type::VectorTyID:
    return LLVMVectorTypeKind;
  case Type::X86_MMXTyID:
    return LLVMX86_MMXTypeKind;
  }
  llvm_unreachable("Unhandled TypeID.");
}

LLVMBool LLVMTypeIsSized(LLVMTypeRef Ty)
{
    return unwrap(Ty)->isSized();
}

LLVMContextRef LLVMGetTypeContext(LLVMTypeRef Ty) {
  return wrap(&unwrap(Ty)->getContext());
}

void LLVMDumpType(LLVMTypeRef Ty) {
  return unwrap(Ty)->dump();
}

char *LLVMPrintTypeToString(LLVMTypeRef Ty) {
  std::string buf;
  raw_string_ostream os(buf);

  if (unwrap(Ty))
    unwrap(Ty)->print(os);
  else
    os << "Printing <null> Type";

  os.flush();

  return strdup(buf.c_str());
}

/*--.. Operations on integer types .........................................--*/

LLVMTypeRef LLVMInt1TypeInContext(LLVMContextRef C)  {
  return (LLVMTypeRef) Type::getInt1Ty(*unwrap(C));
}
LLVMTypeRef LLVMInt8TypeInContext(LLVMContextRef C)  {
  return (LLVMTypeRef) Type::getInt8Ty(*unwrap(C));
}
LLVMTypeRef LLVMInt16TypeInContext(LLVMContextRef C) {
  return (LLVMTypeRef) Type::getInt16Ty(*unwrap(C));
}
LLVMTypeRef LLVMInt32TypeInContext(LLVMContextRef C) {
  return (LLVMTypeRef) Type::getInt32Ty(*unwrap(C));
}
LLVMTypeRef LLVMInt64TypeInContext(LLVMContextRef C) {
  return (LLVMTypeRef) Type::getInt64Ty(*unwrap(C));
}
LLVMTypeRef LLVMInt128TypeInContext(LLVMContextRef C) {
  return (LLVMTypeRef) Type::getInt128Ty(*unwrap(C));
}
LLVMTypeRef LLVMIntTypeInContext(LLVMContextRef C, unsigned NumBits) {
  return wrap(IntegerType::get(*unwrap(C), NumBits));
}

LLVMTypeRef LLVMInt1Type(void)  {
  return LLVMInt1TypeInContext(LLVMGetGlobalContext());
}
LLVMTypeRef LLVMInt8Type(void)  {
  return LLVMInt8TypeInContext(LLVMGetGlobalContext());
}
LLVMTypeRef LLVMInt16Type(void) {
  return LLVMInt16TypeInContext(LLVMGetGlobalContext());
}
LLVMTypeRef LLVMInt32Type(void) {
  return LLVMInt32TypeInContext(LLVMGetGlobalContext());
}
LLVMTypeRef LLVMInt64Type(void) {
  return LLVMInt64TypeInContext(LLVMGetGlobalContext());
}
LLVMTypeRef LLVMInt128Type(void) {
  return LLVMInt128TypeInContext(LLVMGetGlobalContext());
}
LLVMTypeRef LLVMIntType(unsigned NumBits) {
  return LLVMIntTypeInContext(LLVMGetGlobalContext(), NumBits);
}

unsigned LLVMGetIntTypeWidth(LLVMTypeRef IntegerTy) {
  return unwrap<IntegerType>(IntegerTy)->getBitWidth();
}

/*--.. Operations on real types ............................................--*/

LLVMTypeRef LLVMHalfTypeInContext(LLVMContextRef C) {
  return (LLVMTypeRef) Type::getHalfTy(*unwrap(C));
}
LLVMTypeRef LLVMFloatTypeInContext(LLVMContextRef C) {
  return (LLVMTypeRef) Type::getFloatTy(*unwrap(C));
}
LLVMTypeRef LLVMDoubleTypeInContext(LLVMContextRef C) {
  return (LLVMTypeRef) Type::getDoubleTy(*unwrap(C));
}
LLVMTypeRef LLVMX86FP80TypeInContext(LLVMContextRef C) {
  return (LLVMTypeRef) Type::getX86_FP80Ty(*unwrap(C));
}
LLVMTypeRef LLVMFP128TypeInContext(LLVMContextRef C) {
  return (LLVMTypeRef) Type::getFP128Ty(*unwrap(C));
}
LLVMTypeRef LLVMPPCFP128TypeInContext(LLVMContextRef C) {
  return (LLVMTypeRef) Type::getPPC_FP128Ty(*unwrap(C));
}
LLVMTypeRef LLVMX86MMXTypeInContext(LLVMContextRef C) {
  return (LLVMTypeRef) Type::getX86_MMXTy(*unwrap(C));
}

LLVMTypeRef LLVMHalfType(void) {
  return LLVMHalfTypeInContext(LLVMGetGlobalContext());
}
LLVMTypeRef LLVMFloatType(void) {
  return LLVMFloatTypeInContext(LLVMGetGlobalContext());
}
LLVMTypeRef LLVMDoubleType(void) {
  return LLVMDoubleTypeInContext(LLVMGetGlobalContext());
}
LLVMTypeRef LLVMX86FP80Type(void) {
  return LLVMX86FP80TypeInContext(LLVMGetGlobalContext());
}
LLVMTypeRef LLVMFP128Type(void) {
  return LLVMFP128TypeInContext(LLVMGetGlobalContext());
}
LLVMTypeRef LLVMPPCFP128Type(void) {
  return LLVMPPCFP128TypeInContext(LLVMGetGlobalContext());
}
LLVMTypeRef LLVMX86MMXType(void) {
  return LLVMX86MMXTypeInContext(LLVMGetGlobalContext());
}

/*--.. Operations on function types ........................................--*/

LLVMTypeRef LLVMFunctionType(LLVMTypeRef ReturnType,
                             LLVMTypeRef *ParamTypes, unsigned ParamCount,
                             LLVMBool IsVarArg) {
  ArrayRef<Type*> Tys(unwrap(ParamTypes), ParamCount);
  return wrap(FunctionType::get(unwrap(ReturnType), Tys, IsVarArg != 0));
}

LLVMBool LLVMIsFunctionVarArg(LLVMTypeRef FunctionTy) {
  return unwrap<FunctionType>(FunctionTy)->isVarArg();
}

LLVMTypeRef LLVMGetReturnType(LLVMTypeRef FunctionTy) {
  return wrap(unwrap<FunctionType>(FunctionTy)->getReturnType());
}

unsigned LLVMCountParamTypes(LLVMTypeRef FunctionTy) {
  return unwrap<FunctionType>(FunctionTy)->getNumParams();
}

void LLVMGetParamTypes(LLVMTypeRef FunctionTy, LLVMTypeRef *Dest) {
  FunctionType *Ty = unwrap<FunctionType>(FunctionTy);
  for (FunctionType::param_iterator I = Ty->param_begin(),
                                    E = Ty->param_end(); I != E; ++I)
    *Dest++ = wrap(*I);
}

/*--.. Operations on struct types ..........................................--*/

LLVMTypeRef LLVMStructTypeInContext(LLVMContextRef C, LLVMTypeRef *ElementTypes,
                           unsigned ElementCount, LLVMBool Packed) {
  ArrayRef<Type*> Tys(unwrap(ElementTypes), ElementCount);
  return wrap(StructType::get(*unwrap(C), Tys, Packed != 0));
}

LLVMTypeRef LLVMStructType(LLVMTypeRef *ElementTypes,
                           unsigned ElementCount, LLVMBool Packed) {
  return LLVMStructTypeInContext(LLVMGetGlobalContext(), ElementTypes,
                                 ElementCount, Packed);
}

LLVMTypeRef LLVMStructCreateNamed(LLVMContextRef C, const char *Name)
{
  return wrap(StructType::create(*unwrap(C), Name));
}

const char *LLVMGetStructName(LLVMTypeRef Ty)
{
  StructType *Type = unwrap<StructType>(Ty);
  if (!Type->hasName())
    return nullptr;
  return Type->getName().data();
}

void LLVMStructSetBody(LLVMTypeRef StructTy, LLVMTypeRef *ElementTypes,
                       unsigned ElementCount, LLVMBool Packed) {
  ArrayRef<Type*> Tys(unwrap(ElementTypes), ElementCount);
  unwrap<StructType>(StructTy)->setBody(Tys, Packed != 0);
}

unsigned LLVMCountStructElementTypes(LLVMTypeRef StructTy) {
  return unwrap<StructType>(StructTy)->getNumElements();
}

void LLVMGetStructElementTypes(LLVMTypeRef StructTy, LLVMTypeRef *Dest) {
  StructType *Ty = unwrap<StructType>(StructTy);
  for (StructType::element_iterator I = Ty->element_begin(),
                                    E = Ty->element_end(); I != E; ++I)
    *Dest++ = wrap(*I);
}

LLVMTypeRef LLVMStructGetTypeAtIndex(LLVMTypeRef StructTy, unsigned i) {
  StructType *Ty = unwrap<StructType>(StructTy);
  return wrap(Ty->getTypeAtIndex(i));
}

LLVMBool LLVMIsPackedStruct(LLVMTypeRef StructTy) {
  return unwrap<StructType>(StructTy)->isPacked();
}

LLVMBool LLVMIsOpaqueStruct(LLVMTypeRef StructTy) {
  return unwrap<StructType>(StructTy)->isOpaque();
}

LLVMTypeRef LLVMGetTypeByName(LLVMModuleRef M, const char *Name) {
  return wrap(unwrap(M)->getTypeByName(Name));
}

/*--.. Operations on array, pointer, and vector types (sequence types) .....--*/

LLVMTypeRef LLVMArrayType(LLVMTypeRef ElementType, unsigned ElementCount) {
  return wrap(ArrayType::get(unwrap(ElementType), ElementCount));
}

LLVMTypeRef LLVMPointerType(LLVMTypeRef ElementType, unsigned AddressSpace) {
  return wrap(PointerType::get(unwrap(ElementType), AddressSpace));
}

LLVMTypeRef LLVMVectorType(LLVMTypeRef ElementType, unsigned ElementCount) {
  return wrap(VectorType::get(unwrap(ElementType), ElementCount));
}

LLVMTypeRef LLVMGetElementType(LLVMTypeRef Ty) {
  return wrap(unwrap<SequentialType>(Ty)->getElementType());
}

unsigned LLVMGetArrayLength(LLVMTypeRef ArrayTy) {
  return unwrap<ArrayType>(ArrayTy)->getNumElements();
}

unsigned LLVMGetPointerAddressSpace(LLVMTypeRef PointerTy) {
  return unwrap<PointerType>(PointerTy)->getAddressSpace();
}

unsigned LLVMGetVectorSize(LLVMTypeRef VectorTy) {
  return unwrap<VectorType>(VectorTy)->getNumElements();
}

/*--.. Operations on other types ...........................................--*/

LLVMTypeRef LLVMVoidTypeInContext(LLVMContextRef C)  {
  return wrap(Type::getVoidTy(*unwrap(C)));
}
LLVMTypeRef LLVMLabelTypeInContext(LLVMContextRef C) {
  return wrap(Type::getLabelTy(*unwrap(C)));
}

LLVMTypeRef LLVMVoidType(void)  {
  return LLVMVoidTypeInContext(LLVMGetGlobalContext());
}
LLVMTypeRef LLVMLabelType(void) {
  return LLVMLabelTypeInContext(LLVMGetGlobalContext());
}

/*===-- Operations on values ----------------------------------------------===*/

/*--.. Operations on all values ............................................--*/

LLVMTypeRef LLVMTypeOf(LLVMValueRef Val) {
  return wrap(unwrap(Val)->getType());
}

const char *LLVMGetValueName(LLVMValueRef Val) {
  return unwrap(Val)->getName().data();
}

void LLVMSetValueName(LLVMValueRef Val, const char *Name) {
  unwrap(Val)->setName(Name);
}

void LLVMDumpValue(LLVMValueRef Val) {
  unwrap(Val)->dump();
}

char* LLVMPrintValueToString(LLVMValueRef Val) {
  std::string buf;
  raw_string_ostream os(buf);

  if (unwrap(Val))
    unwrap(Val)->print(os);
  else
    os << "Printing <null> Value";

  os.flush();

  return strdup(buf.c_str());
}

void LLVMReplaceAllUsesWith(LLVMValueRef OldVal, LLVMValueRef NewVal) {
  unwrap(OldVal)->replaceAllUsesWith(unwrap(NewVal));
}

int LLVMHasMetadata(LLVMValueRef Inst) {
  return unwrap<Instruction>(Inst)->hasMetadata();
}

LLVMValueRef LLVMGetMetadata(LLVMValueRef Inst, unsigned KindID) {
  auto *I = unwrap<Instruction>(Inst);
  assert(I && "Expected instruction");
  if (auto *MD = I->getMetadata(KindID))
    return wrap(MetadataAsValue::get(I->getContext(), MD));
  return nullptr;
}

// MetadataAsValue uses a canonical format which strips the actual MDNode for
// MDNode with just a single constant value, storing just a ConstantAsMetadata
// This undoes this canonicalization, reconstructing the MDNode.
static MDNode *extractMDNode(MetadataAsValue *MAV) {
  Metadata *MD = MAV->getMetadata();
  assert((isa<MDNode>(MD) || isa<ConstantAsMetadata>(MD)) &&
      "Expected a metadata node or a canonicalized constant");

  if (MDNode *N = dyn_cast<MDNode>(MD))
    return N;

  return MDNode::get(MAV->getContext(), MD);
}

void LLVMSetMetadata(LLVMValueRef Inst, unsigned KindID, LLVMValueRef Val) {
  MDNode *N = Val ? extractMDNode(unwrap<MetadataAsValue>(Val)) : nullptr;

  unwrap<Instruction>(Inst)->setMetadata(KindID, N);
}

/*--.. Conversion functions ................................................--*/

#define LLVM_DEFINE_VALUE_CAST(name)                                       \
  LLVMValueRef LLVMIsA##name(LLVMValueRef Val) {                           \
    return wrap(static_cast<Value*>(dyn_cast_or_null<name>(unwrap(Val)))); \
  }

LLVM_FOR_EACH_VALUE_SUBCLASS(LLVM_DEFINE_VALUE_CAST)

LLVMValueRef LLVMIsAMDNode(LLVMValueRef Val) {
  if (auto *MD = dyn_cast_or_null<MetadataAsValue>(unwrap(Val)))
    if (isa<MDNode>(MD->getMetadata()) ||
        isa<ValueAsMetadata>(MD->getMetadata()))
      return Val;
  return nullptr;
}

LLVMValueRef LLVMIsAMDString(LLVMValueRef Val) {
  if (auto *MD = dyn_cast_or_null<MetadataAsValue>(unwrap(Val)))
    if (isa<MDString>(MD->getMetadata()))
      return Val;
  return nullptr;
}

/*--.. Operations on Uses ..................................................--*/
LLVMUseRef LLVMGetFirstUse(LLVMValueRef Val) {
  Value *V = unwrap(Val);
  Value::use_iterator I = V->use_begin();
  if (I == V->use_end())
    return nullptr;
  return wrap(&*I);
}

LLVMUseRef LLVMGetNextUse(LLVMUseRef U) {
  Use *Next = unwrap(U)->getNext();
  if (Next)
    return wrap(Next);
  return nullptr;
}

LLVMValueRef LLVMGetUser(LLVMUseRef U) {
  return wrap(unwrap(U)->getUser());
}

LLVMValueRef LLVMGetUsedValue(LLVMUseRef U) {
  return wrap(unwrap(U)->get());
}

/*--.. Operations on Users .................................................--*/

static LLVMValueRef getMDNodeOperandImpl(LLVMContext &Context, const MDNode *N,
                                         unsigned Index) {
  Metadata *Op = N->getOperand(Index);
  if (!Op)
    return nullptr;
  if (auto *C = dyn_cast<ConstantAsMetadata>(Op))
    return wrap(C->getValue());
  return wrap(MetadataAsValue::get(Context, Op));
}

LLVMValueRef LLVMGetOperand(LLVMValueRef Val, unsigned Index) {
  Value *V = unwrap(Val);
  if (auto *MD = dyn_cast<MetadataAsValue>(V)) {
    if (auto *L = dyn_cast<ValueAsMetadata>(MD->getMetadata())) {
      assert(Index == 0 && "Function-local metadata can only have one operand");
      return wrap(L->getValue());
    }
    return getMDNodeOperandImpl(V->getContext(),
                                cast<MDNode>(MD->getMetadata()), Index);
  }

  return wrap(cast<User>(V)->getOperand(Index));
}

LLVMUseRef LLVMGetOperandUse(LLVMValueRef Val, unsigned Index) {
  Value *V = unwrap(Val);
  return wrap(&cast<User>(V)->getOperandUse(Index));
}

void LLVMSetOperand(LLVMValueRef Val, unsigned Index, LLVMValueRef Op) {
  unwrap<User>(Val)->setOperand(Index, unwrap(Op));
}

int LLVMGetNumOperands(LLVMValueRef Val) {
  Value *V = unwrap(Val);
  if (isa<MetadataAsValue>(V))
    return LLVMGetMDNodeNumOperands(Val);

  return cast<User>(V)->getNumOperands();
}

/*--.. Operations on constants of any type .................................--*/

LLVMValueRef LLVMConstNull(LLVMTypeRef Ty) {
  return wrap(Constant::getNullValue(unwrap(Ty)));
}

LLVMValueRef LLVMConstAllOnes(LLVMTypeRef Ty) {
  return wrap(Constant::getAllOnesValue(unwrap(Ty)));
}

LLVMValueRef LLVMGetUndef(LLVMTypeRef Ty) {
  return wrap(UndefValue::get(unwrap(Ty)));
}

LLVMBool LLVMIsConstant(LLVMValueRef Ty) {
  return isa<Constant>(unwrap(Ty));
}

LLVMBool LLVMIsNull(LLVMValueRef Val) {
  if (Constant *C = dyn_cast<Constant>(unwrap(Val)))
    return C->isNullValue();
  return false;
}

LLVMBool LLVMIsUndef(LLVMValueRef Val) {
  return isa<UndefValue>(unwrap(Val));
}

LLVMValueRef LLVMConstPointerNull(LLVMTypeRef Ty) {
  return
      wrap(ConstantPointerNull::get(unwrap<PointerType>(Ty)));
}

/*--.. Operations on metadata nodes ........................................--*/

LLVMValueRef LLVMMDStringInContext(LLVMContextRef C, const char *Str,
                                   unsigned SLen) {
  LLVMContext &Context = *unwrap(C);
  return wrap(MetadataAsValue::get(
      Context, MDString::get(Context, StringRef(Str, SLen))));
}

LLVMValueRef LLVMMDString(const char *Str, unsigned SLen) {
  return LLVMMDStringInContext(LLVMGetGlobalContext(), Str, SLen);
}

LLVMValueRef LLVMMDNodeInContext(LLVMContextRef C, LLVMValueRef *Vals,
                                 unsigned Count) {
  LLVMContext &Context = *unwrap(C);
  SmallVector<Metadata *, 8> MDs;
  for (auto *OV : makeArrayRef(Vals, Count)) {
    Value *V = unwrap(OV);
    Metadata *MD;
    if (!V)
      MD = nullptr;
    else if (auto *C = dyn_cast<Constant>(V))
      MD = ConstantAsMetadata::get(C);
    else if (auto *MDV = dyn_cast<MetadataAsValue>(V)) {
      MD = MDV->getMetadata();
      assert(!isa<LocalAsMetadata>(MD) && "Unexpected function-local metadata "
                                          "outside of direct argument to call");
    } else {
      // This is function-local metadata.  Pretend to make an MDNode.
      assert(Count == 1 &&
             "Expected only one operand to function-local metadata");
      return wrap(MetadataAsValue::get(Context, LocalAsMetadata::get(V)));
    }

    MDs.push_back(MD);
  }
  return wrap(MetadataAsValue::get(Context, MDNode::get(Context, MDs)));
}

LLVMValueRef LLVMMDNode(LLVMValueRef *Vals, unsigned Count) {
  return LLVMMDNodeInContext(LLVMGetGlobalContext(), Vals, Count);
}

const char *LLVMGetMDString(LLVMValueRef V, unsigned* Len) {
  if (const auto *MD = dyn_cast<MetadataAsValue>(unwrap(V)))
    if (const MDString *S = dyn_cast<MDString>(MD->getMetadata())) {
      *Len = S->getString().size();
      return S->getString().data();
    }
  *Len = 0;
  return nullptr;
}

unsigned LLVMGetMDNodeNumOperands(LLVMValueRef V)
{
  auto *MD = cast<MetadataAsValue>(unwrap(V));
  if (isa<ValueAsMetadata>(MD->getMetadata()))
    return 1;
  return cast<MDNode>(MD->getMetadata())->getNumOperands();
}

void LLVMGetMDNodeOperands(LLVMValueRef V, LLVMValueRef *Dest)
{
  auto *MD = cast<MetadataAsValue>(unwrap(V));
  if (auto *MDV = dyn_cast<ValueAsMetadata>(MD->getMetadata())) {
    *Dest = wrap(MDV->getValue());
    return;
  }
  const auto *N = cast<MDNode>(MD->getMetadata());
  const unsigned numOperands = N->getNumOperands();
  LLVMContext &Context = unwrap(V)->getContext();
  for (unsigned i = 0; i < numOperands; i++)
    Dest[i] = getMDNodeOperandImpl(Context, N, i);
}

unsigned LLVMGetNamedMetadataNumOperands(LLVMModuleRef M, const char* name)
{
  if (NamedMDNode *N = unwrap(M)->getNamedMetadata(name)) {
    return N->getNumOperands();
  }
  return 0;
}

void LLVMGetNamedMetadataOperands(LLVMModuleRef M, const char* name, LLVMValueRef *Dest)
{
  NamedMDNode *N = unwrap(M)->getNamedMetadata(name);
  if (!N)
    return;
  LLVMContext &Context = unwrap(M)->getContext();
  for (unsigned i=0;i<N->getNumOperands();i++)
    Dest[i] = wrap(MetadataAsValue::get(Context, N->getOperand(i)));
}

void LLVMAddNamedMetadataOperand(LLVMModuleRef M, const char* name,
                                 LLVMValueRef Val)
{
  NamedMDNode *N = unwrap(M)->getOrInsertNamedMetadata(name);
  if (!N)
    return;
  if (!Val)
    return;
  N->addOperand(extractMDNode(unwrap<MetadataAsValue>(Val)));
}

/*--.. Operations on scalar constants ......................................--*/

LLVMValueRef LLVMConstInt(LLVMTypeRef IntTy, unsigned long long N,
                          LLVMBool SignExtend) {
  return wrap(ConstantInt::get(unwrap<IntegerType>(IntTy), N, SignExtend != 0));
}

LLVMValueRef LLVMConstIntOfArbitraryPrecision(LLVMTypeRef IntTy,
                                              unsigned NumWords,
                                              const uint64_t Words[]) {
    IntegerType *Ty = unwrap<IntegerType>(IntTy);
    return wrap(ConstantInt::get(Ty->getContext(),
                                 APInt(Ty->getBitWidth(),
                                       makeArrayRef(Words, NumWords))));
}

LLVMValueRef LLVMConstIntOfString(LLVMTypeRef IntTy, const char Str[],
                                  uint8_t Radix) {
  return wrap(ConstantInt::get(unwrap<IntegerType>(IntTy), StringRef(Str),
                               Radix));
}

LLVMValueRef LLVMConstIntOfStringAndSize(LLVMTypeRef IntTy, const char Str[],
                                         unsigned SLen, uint8_t Radix) {
  return wrap(ConstantInt::get(unwrap<IntegerType>(IntTy), StringRef(Str, SLen),
                               Radix));
}

LLVMValueRef LLVMConstReal(LLVMTypeRef RealTy, double N) {
  return wrap(ConstantFP::get(unwrap(RealTy), N));
}

LLVMValueRef LLVMConstRealOfString(LLVMTypeRef RealTy, const char *Text) {
  return wrap(ConstantFP::get(unwrap(RealTy), StringRef(Text)));
}

LLVMValueRef LLVMConstRealOfStringAndSize(LLVMTypeRef RealTy, const char Str[],
                                          unsigned SLen) {
  return wrap(ConstantFP::get(unwrap(RealTy), StringRef(Str, SLen)));
}

unsigned long long LLVMConstIntGetZExtValue(LLVMValueRef ConstantVal) {
  return unwrap<ConstantInt>(ConstantVal)->getZExtValue();
}

long long LLVMConstIntGetSExtValue(LLVMValueRef ConstantVal) {
  return unwrap<ConstantInt>(ConstantVal)->getSExtValue();
}

double LLVMConstRealGetDouble(LLVMValueRef ConstantVal, LLVMBool *LosesInfo) {
  ConstantFP *cFP = unwrap<ConstantFP>(ConstantVal) ;
  Type *Ty = cFP->getType();

  if (Ty->isFloatTy()) {
    *LosesInfo = false;
    return cFP->getValueAPF().convertToFloat();
  }

  if (Ty->isDoubleTy()) {
    *LosesInfo = false;
    return cFP->getValueAPF().convertToDouble();
  }

  bool APFLosesInfo;
  APFloat APF = cFP->getValueAPF();
  APF.convert(APFloat::IEEEdouble, APFloat::rmNearestTiesToEven, &APFLosesInfo);
  *LosesInfo = APFLosesInfo;
  return APF.convertToDouble();
}

/*--.. Operations on composite constants ...................................--*/

LLVMValueRef LLVMConstStringInContext(LLVMContextRef C, const char *Str,
                                      unsigned Length,
                                      LLVMBool DontNullTerminate) {
  /* Inverted the sense of AddNull because ', 0)' is a
     better mnemonic for null termination than ', 1)'. */
  return wrap(ConstantDataArray::getString(*unwrap(C), StringRef(Str, Length),
                                           DontNullTerminate == 0));
}
LLVMValueRef LLVMConstStructInContext(LLVMContextRef C,
                                      LLVMValueRef *ConstantVals,
                                      unsigned Count, LLVMBool Packed) {
  Constant **Elements = unwrap<Constant>(ConstantVals, Count);
  return wrap(ConstantStruct::getAnon(*unwrap(C), makeArrayRef(Elements, Count),
                                      Packed != 0));
}

LLVMValueRef LLVMConstString(const char *Str, unsigned Length,
                             LLVMBool DontNullTerminate) {
  return LLVMConstStringInContext(LLVMGetGlobalContext(), Str, Length,
                                  DontNullTerminate);
}

LLVMValueRef LLVMGetElementAsConstant(LLVMValueRef c, unsigned idx) {
  return wrap(static_cast<ConstantDataSequential*>(unwrap(c))->getElementAsConstant(idx));
}

LLVMBool LLVMIsConstantString(LLVMValueRef c) {
  return static_cast<ConstantDataSequential*>(unwrap(c))->isString();
}

const char *LLVMGetAsString(LLVMValueRef c, size_t* Length) {
  StringRef str = static_cast<ConstantDataSequential*>(unwrap(c))->getAsString();
  *Length = str.size();
  return str.data();
}

LLVMValueRef LLVMConstArray(LLVMTypeRef ElementTy,
                            LLVMValueRef *ConstantVals, unsigned Length) {
  ArrayRef<Constant*> V(unwrap<Constant>(ConstantVals, Length), Length);
  return wrap(ConstantArray::get(ArrayType::get(unwrap(ElementTy), Length), V));
}

LLVMValueRef LLVMConstStruct(LLVMValueRef *ConstantVals, unsigned Count,
                             LLVMBool Packed) {
  return LLVMConstStructInContext(LLVMGetGlobalContext(), ConstantVals, Count,
                                  Packed);
}

LLVMValueRef LLVMConstNamedStruct(LLVMTypeRef StructTy,
                                  LLVMValueRef *ConstantVals,
                                  unsigned Count) {
  Constant **Elements = unwrap<Constant>(ConstantVals, Count);
  StructType *Ty = cast<StructType>(unwrap(StructTy));

  return wrap(ConstantStruct::get(Ty, makeArrayRef(Elements, Count)));
}

LLVMValueRef LLVMConstVector(LLVMValueRef *ScalarConstantVals, unsigned Size) {
  return wrap(ConstantVector::get(makeArrayRef(
                            unwrap<Constant>(ScalarConstantVals, Size), Size)));
}

/*-- Opcode mapping */

static LLVMOpcode map_to_llvmopcode(int opcode)
{
    switch (opcode) {
      default: llvm_unreachable("Unhandled Opcode.");
#define HANDLE_INST(num, opc, clas) case num: return LLVM##opc;
#include "llvm/IR/Instruction.def"
#undef HANDLE_INST
    }
}

static int map_from_llvmopcode(LLVMOpcode code)
{
    switch (code) {
#define HANDLE_INST(num, opc, clas) case LLVM##opc: return num;
#include "llvm/IR/Instruction.def"
#undef HANDLE_INST
    }
    llvm_unreachable("Unhandled Opcode.");
}

/*--.. Constant expressions ................................................--*/

LLVMOpcode LLVMGetConstOpcode(LLVMValueRef ConstantVal) {
  return map_to_llvmopcode(unwrap<ConstantExpr>(ConstantVal)->getOpcode());
}

LLVMValueRef LLVMAlignOf(LLVMTypeRef Ty) {
  return wrap(ConstantExpr::getAlignOf(unwrap(Ty)));
}

LLVMValueRef LLVMSizeOf(LLVMTypeRef Ty) {
  return wrap(ConstantExpr::getSizeOf(unwrap(Ty)));
}

LLVMValueRef LLVMConstNeg(LLVMValueRef ConstantVal) {
  return wrap(ConstantExpr::getNeg(unwrap<Constant>(ConstantVal)));
}

LLVMValueRef LLVMConstNSWNeg(LLVMValueRef ConstantVal) {
  return wrap(ConstantExpr::getNSWNeg(unwrap<Constant>(ConstantVal)));
}

LLVMValueRef LLVMConstNUWNeg(LLVMValueRef ConstantVal) {
  return wrap(ConstantExpr::getNUWNeg(unwrap<Constant>(ConstantVal)));
}


LLVMValueRef LLVMConstFNeg(LLVMValueRef ConstantVal) {
  return wrap(ConstantExpr::getFNeg(unwrap<Constant>(ConstantVal)));
}

LLVMValueRef LLVMConstNot(LLVMValueRef ConstantVal) {
  return wrap(ConstantExpr::getNot(unwrap<Constant>(ConstantVal)));
}

LLVMValueRef LLVMConstAdd(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getAdd(unwrap<Constant>(LHSConstant),
                                   unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstNSWAdd(LLVMValueRef LHSConstant,
                             LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getNSWAdd(unwrap<Constant>(LHSConstant),
                                      unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstNUWAdd(LLVMValueRef LHSConstant,
                             LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getNUWAdd(unwrap<Constant>(LHSConstant),
                                      unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstFAdd(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getFAdd(unwrap<Constant>(LHSConstant),
                                    unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstSub(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getSub(unwrap<Constant>(LHSConstant),
                                   unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstNSWSub(LLVMValueRef LHSConstant,
                             LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getNSWSub(unwrap<Constant>(LHSConstant),
                                      unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstNUWSub(LLVMValueRef LHSConstant,
                             LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getNUWSub(unwrap<Constant>(LHSConstant),
                                      unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstFSub(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getFSub(unwrap<Constant>(LHSConstant),
                                    unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstMul(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getMul(unwrap<Constant>(LHSConstant),
                                   unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstNSWMul(LLVMValueRef LHSConstant,
                             LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getNSWMul(unwrap<Constant>(LHSConstant),
                                      unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstNUWMul(LLVMValueRef LHSConstant,
                             LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getNUWMul(unwrap<Constant>(LHSConstant),
                                      unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstFMul(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getFMul(unwrap<Constant>(LHSConstant),
                                    unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstUDiv(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getUDiv(unwrap<Constant>(LHSConstant),
                                    unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstSDiv(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getSDiv(unwrap<Constant>(LHSConstant),
                                    unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstExactSDiv(LLVMValueRef LHSConstant,
                                LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getExactSDiv(unwrap<Constant>(LHSConstant),
                                         unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstFDiv(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getFDiv(unwrap<Constant>(LHSConstant),
                                    unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstURem(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getURem(unwrap<Constant>(LHSConstant),
                                    unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstSRem(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getSRem(unwrap<Constant>(LHSConstant),
                                    unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstFRem(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getFRem(unwrap<Constant>(LHSConstant),
                                    unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstAnd(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getAnd(unwrap<Constant>(LHSConstant),
                                   unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstOr(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getOr(unwrap<Constant>(LHSConstant),
                                  unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstXor(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getXor(unwrap<Constant>(LHSConstant),
                                   unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstICmp(LLVMIntPredicate Predicate,
                           LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getICmp(Predicate,
                                    unwrap<Constant>(LHSConstant),
                                    unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstFCmp(LLVMRealPredicate Predicate,
                           LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getFCmp(Predicate,
                                    unwrap<Constant>(LHSConstant),
                                    unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstShl(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getShl(unwrap<Constant>(LHSConstant),
                                   unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstLShr(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getLShr(unwrap<Constant>(LHSConstant),
                                    unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstAShr(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant) {
  return wrap(ConstantExpr::getAShr(unwrap<Constant>(LHSConstant),
                                    unwrap<Constant>(RHSConstant)));
}

LLVMValueRef LLVMConstGEP(LLVMValueRef ConstantVal,
                          LLVMValueRef *ConstantIndices, unsigned NumIndices) {
  ArrayRef<Constant *> IdxList(unwrap<Constant>(ConstantIndices, NumIndices),
                               NumIndices);
  return wrap(ConstantExpr::getGetElementPtr(
      nullptr, unwrap<Constant>(ConstantVal), IdxList));
}

LLVMValueRef LLVMConstInBoundsGEP(LLVMValueRef ConstantVal,
                                  LLVMValueRef *ConstantIndices,
                                  unsigned NumIndices) {
  Constant* Val = unwrap<Constant>(ConstantVal);
  ArrayRef<Constant *> IdxList(unwrap<Constant>(ConstantIndices, NumIndices),
                               NumIndices);
  return wrap(ConstantExpr::getInBoundsGetElementPtr(nullptr, Val, IdxList));
}

LLVMValueRef LLVMConstTrunc(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getTrunc(unwrap<Constant>(ConstantVal),
                                     unwrap(ToType)));
}

LLVMValueRef LLVMConstSExt(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getSExt(unwrap<Constant>(ConstantVal),
                                    unwrap(ToType)));
}

LLVMValueRef LLVMConstZExt(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getZExt(unwrap<Constant>(ConstantVal),
                                    unwrap(ToType)));
}

LLVMValueRef LLVMConstFPTrunc(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getFPTrunc(unwrap<Constant>(ConstantVal),
                                       unwrap(ToType)));
}

LLVMValueRef LLVMConstFPExt(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getFPExtend(unwrap<Constant>(ConstantVal),
                                        unwrap(ToType)));
}

LLVMValueRef LLVMConstUIToFP(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getUIToFP(unwrap<Constant>(ConstantVal),
                                      unwrap(ToType)));
}

LLVMValueRef LLVMConstSIToFP(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getSIToFP(unwrap<Constant>(ConstantVal),
                                      unwrap(ToType)));
}

LLVMValueRef LLVMConstFPToUI(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getFPToUI(unwrap<Constant>(ConstantVal),
                                      unwrap(ToType)));
}

LLVMValueRef LLVMConstFPToSI(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getFPToSI(unwrap<Constant>(ConstantVal),
                                      unwrap(ToType)));
}

LLVMValueRef LLVMConstPtrToInt(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getPtrToInt(unwrap<Constant>(ConstantVal),
                                        unwrap(ToType)));
}

LLVMValueRef LLVMConstIntToPtr(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getIntToPtr(unwrap<Constant>(ConstantVal),
                                        unwrap(ToType)));
}

LLVMValueRef LLVMConstBitCast(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getBitCast(unwrap<Constant>(ConstantVal),
                                       unwrap(ToType)));
}

LLVMValueRef LLVMConstAddrSpaceCast(LLVMValueRef ConstantVal,
                                    LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getAddrSpaceCast(unwrap<Constant>(ConstantVal),
                                             unwrap(ToType)));
}

LLVMValueRef LLVMConstZExtOrBitCast(LLVMValueRef ConstantVal,
                                    LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getZExtOrBitCast(unwrap<Constant>(ConstantVal),
                                             unwrap(ToType)));
}

LLVMValueRef LLVMConstSExtOrBitCast(LLVMValueRef ConstantVal,
                                    LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getSExtOrBitCast(unwrap<Constant>(ConstantVal),
                                             unwrap(ToType)));
}

LLVMValueRef LLVMConstTruncOrBitCast(LLVMValueRef ConstantVal,
                                     LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getTruncOrBitCast(unwrap<Constant>(ConstantVal),
                                              unwrap(ToType)));
}

LLVMValueRef LLVMConstPointerCast(LLVMValueRef ConstantVal,
                                  LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getPointerCast(unwrap<Constant>(ConstantVal),
                                           unwrap(ToType)));
}

LLVMValueRef LLVMConstIntCast(LLVMValueRef ConstantVal, LLVMTypeRef ToType,
                              LLVMBool isSigned) {
  return wrap(ConstantExpr::getIntegerCast(unwrap<Constant>(ConstantVal),
                                           unwrap(ToType), isSigned));
}

LLVMValueRef LLVMConstFPCast(LLVMValueRef ConstantVal, LLVMTypeRef ToType) {
  return wrap(ConstantExpr::getFPCast(unwrap<Constant>(ConstantVal),
                                      unwrap(ToType)));
}

LLVMValueRef LLVMConstSelect(LLVMValueRef ConstantCondition,
                             LLVMValueRef ConstantIfTrue,
                             LLVMValueRef ConstantIfFalse) {
  return wrap(ConstantExpr::getSelect(unwrap<Constant>(ConstantCondition),
                                      unwrap<Constant>(ConstantIfTrue),
                                      unwrap<Constant>(ConstantIfFalse)));
}

LLVMValueRef LLVMConstExtractElement(LLVMValueRef VectorConstant,
                                     LLVMValueRef IndexConstant) {
  return wrap(ConstantExpr::getExtractElement(unwrap<Constant>(VectorConstant),
                                              unwrap<Constant>(IndexConstant)));
}

LLVMValueRef LLVMConstInsertElement(LLVMValueRef VectorConstant,
                                    LLVMValueRef ElementValueConstant,
                                    LLVMValueRef IndexConstant) {
  return wrap(ConstantExpr::getInsertElement(unwrap<Constant>(VectorConstant),
                                         unwrap<Constant>(ElementValueConstant),
                                             unwrap<Constant>(IndexConstant)));
}

LLVMValueRef LLVMConstShuffleVector(LLVMValueRef VectorAConstant,
                                    LLVMValueRef VectorBConstant,
                                    LLVMValueRef MaskConstant) {
  return wrap(ConstantExpr::getShuffleVector(unwrap<Constant>(VectorAConstant),
                                             unwrap<Constant>(VectorBConstant),
                                             unwrap<Constant>(MaskConstant)));
}

LLVMValueRef LLVMConstExtractValue(LLVMValueRef AggConstant, unsigned *IdxList,
                                   unsigned NumIdx) {
  return wrap(ConstantExpr::getExtractValue(unwrap<Constant>(AggConstant),
                                            makeArrayRef(IdxList, NumIdx)));
}

LLVMValueRef LLVMConstInsertValue(LLVMValueRef AggConstant,
                                  LLVMValueRef ElementValueConstant,
                                  unsigned *IdxList, unsigned NumIdx) {
  return wrap(ConstantExpr::getInsertValue(unwrap<Constant>(AggConstant),
                                         unwrap<Constant>(ElementValueConstant),
                                           makeArrayRef(IdxList, NumIdx)));
}

LLVMValueRef LLVMConstInlineAsm(LLVMTypeRef Ty, const char *AsmString,
                                const char *Constraints,
                                LLVMBool HasSideEffects,
                                LLVMBool IsAlignStack) {
  return wrap(InlineAsm::get(dyn_cast<FunctionType>(unwrap(Ty)), AsmString,
                             Constraints, HasSideEffects, IsAlignStack));
}

LLVMValueRef LLVMBlockAddress(LLVMValueRef F, LLVMBasicBlockRef BB) {
  return wrap(BlockAddress::get(unwrap<Function>(F), unwrap(BB)));
}

/*--.. Operations on global variables, functions, and aliases (globals) ....--*/

LLVMModuleRef LLVMGetGlobalParent(LLVMValueRef Global) {
  return wrap(unwrap<GlobalValue>(Global)->getParent());
}

LLVMBool LLVMIsDeclaration(LLVMValueRef Global) {
  return unwrap<GlobalValue>(Global)->isDeclaration();
}

LLVMLinkage LLVMGetLinkage(LLVMValueRef Global) {
  switch (unwrap<GlobalValue>(Global)->getLinkage()) {
  case GlobalValue::ExternalLinkage:
    return LLVMExternalLinkage;
  case GlobalValue::AvailableExternallyLinkage:
    return LLVMAvailableExternallyLinkage;
  case GlobalValue::LinkOnceAnyLinkage:
    return LLVMLinkOnceAnyLinkage;
  case GlobalValue::LinkOnceODRLinkage:
    return LLVMLinkOnceODRLinkage;
  case GlobalValue::WeakAnyLinkage:
    return LLVMWeakAnyLinkage;
  case GlobalValue::WeakODRLinkage:
    return LLVMWeakODRLinkage;
  case GlobalValue::AppendingLinkage:
    return LLVMAppendingLinkage;
  case GlobalValue::InternalLinkage:
    return LLVMInternalLinkage;
  case GlobalValue::PrivateLinkage:
    return LLVMPrivateLinkage;
  case GlobalValue::ExternalWeakLinkage:
    return LLVMExternalWeakLinkage;
  case GlobalValue::CommonLinkage:
    return LLVMCommonLinkage;
  }

  llvm_unreachable("Invalid GlobalValue linkage!");
}

void LLVMSetLinkage(LLVMValueRef Global, LLVMLinkage Linkage) {
  GlobalValue *GV = unwrap<GlobalValue>(Global);

  switch (Linkage) {
  case LLVMExternalLinkage:
    GV->setLinkage(GlobalValue::ExternalLinkage);
    break;
  case LLVMAvailableExternallyLinkage:
    GV->setLinkage(GlobalValue::AvailableExternallyLinkage);
    break;
  case LLVMLinkOnceAnyLinkage:
    GV->setLinkage(GlobalValue::LinkOnceAnyLinkage);
    break;
  case LLVMLinkOnceODRLinkage:
    GV->setLinkage(GlobalValue::LinkOnceODRLinkage);
    break;
  case LLVMLinkOnceODRAutoHideLinkage:
    DEBUG(errs() << "LLVMSetLinkage(): LLVMLinkOnceODRAutoHideLinkage is no "
                    "longer supported.");
    break;
  case LLVMWeakAnyLinkage:
    GV->setLinkage(GlobalValue::WeakAnyLinkage);
    break;
  case LLVMWeakODRLinkage:
    GV->setLinkage(GlobalValue::WeakODRLinkage);
    break;
  case LLVMAppendingLinkage:
    GV->setLinkage(GlobalValue::AppendingLinkage);
    break;
  case LLVMInternalLinkage:
    GV->setLinkage(GlobalValue::InternalLinkage);
    break;
  case LLVMPrivateLinkage:
    GV->setLinkage(GlobalValue::PrivateLinkage);
    break;
  case LLVMLinkerPrivateLinkage:
    GV->setLinkage(GlobalValue::PrivateLinkage);
    break;
  case LLVMLinkerPrivateWeakLinkage:
    GV->setLinkage(GlobalValue::PrivateLinkage);
    break;
  case LLVMDLLImportLinkage:
    DEBUG(errs()
          << "LLVMSetLinkage(): LLVMDLLImportLinkage is no longer supported.");
    break;
  case LLVMDLLExportLinkage:
    DEBUG(errs()
          << "LLVMSetLinkage(): LLVMDLLExportLinkage is no longer supported.");
    break;
  case LLVMExternalWeakLinkage:
    GV->setLinkage(GlobalValue::ExternalWeakLinkage);
    break;
  case LLVMGhostLinkage:
    DEBUG(errs()
          << "LLVMSetLinkage(): LLVMGhostLinkage is no longer supported.");
    break;
  case LLVMCommonLinkage:
    GV->setLinkage(GlobalValue::CommonLinkage);
    break;
  }
}

const char *LLVMGetSection(LLVMValueRef Global) {
  return unwrap<GlobalValue>(Global)->getSection();
}

void LLVMSetSection(LLVMValueRef Global, const char *Section) {
  unwrap<GlobalObject>(Global)->setSection(Section);
}

LLVMVisibility LLVMGetVisibility(LLVMValueRef Global) {
  return static_cast<LLVMVisibility>(
    unwrap<GlobalValue>(Global)->getVisibility());
}

void LLVMSetVisibility(LLVMValueRef Global, LLVMVisibility Viz) {
  unwrap<GlobalValue>(Global)
    ->setVisibility(static_cast<GlobalValue::VisibilityTypes>(Viz));
}

LLVMDLLStorageClass LLVMGetDLLStorageClass(LLVMValueRef Global) {
  return static_cast<LLVMDLLStorageClass>(
      unwrap<GlobalValue>(Global)->getDLLStorageClass());
}

void LLVMSetDLLStorageClass(LLVMValueRef Global, LLVMDLLStorageClass Class) {
  unwrap<GlobalValue>(Global)->setDLLStorageClass(
      static_cast<GlobalValue::DLLStorageClassTypes>(Class));
}

LLVMBool LLVMHasUnnamedAddr(LLVMValueRef Global) {
  return unwrap<GlobalValue>(Global)->hasUnnamedAddr();
}

void LLVMSetUnnamedAddr(LLVMValueRef Global, LLVMBool HasUnnamedAddr) {
  unwrap<GlobalValue>(Global)->setUnnamedAddr(HasUnnamedAddr);
}

/*--.. Operations on global variables, load and store instructions .........--*/

unsigned LLVMGetAlignment(LLVMValueRef V) {
  Value *P = unwrap<Value>(V);
  if (GlobalValue *GV = dyn_cast<GlobalValue>(P))
    return GV->getAlignment();
  if (AllocaInst *AI = dyn_cast<AllocaInst>(P))
    return AI->getAlignment();
  if (LoadInst *LI = dyn_cast<LoadInst>(P))
    return LI->getAlignment();
  if (StoreInst *SI = dyn_cast<StoreInst>(P))
    return SI->getAlignment();

  llvm_unreachable(
      "only GlobalValue, AllocaInst, LoadInst and StoreInst have alignment");
}

void LLVMSetAlignment(LLVMValueRef V, unsigned Bytes) {
  Value *P = unwrap<Value>(V);
  if (GlobalObject *GV = dyn_cast<GlobalObject>(P))
    GV->setAlignment(Bytes);
  else if (AllocaInst *AI = dyn_cast<AllocaInst>(P))
    AI->setAlignment(Bytes);
  else if (LoadInst *LI = dyn_cast<LoadInst>(P))
    LI->setAlignment(Bytes);
  else if (StoreInst *SI = dyn_cast<StoreInst>(P))
    SI->setAlignment(Bytes);
  else
    llvm_unreachable(
        "only GlobalValue, AllocaInst, LoadInst and StoreInst have alignment");
}

/*--.. Operations on global variables ......................................--*/

LLVMValueRef LLVMAddGlobal(LLVMModuleRef M, LLVMTypeRef Ty, const char *Name) {
  return wrap(new GlobalVariable(*unwrap(M), unwrap(Ty), false,
                                 GlobalValue::ExternalLinkage, nullptr, Name));
}

LLVMValueRef LLVMAddGlobalInAddressSpace(LLVMModuleRef M, LLVMTypeRef Ty,
                                         const char *Name,
                                         unsigned AddressSpace) {
  return wrap(new GlobalVariable(*unwrap(M), unwrap(Ty), false,
                                 GlobalValue::ExternalLinkage, nullptr, Name,
                                 nullptr, GlobalVariable::NotThreadLocal,
                                 AddressSpace));
}

LLVMValueRef LLVMGetNamedGlobal(LLVMModuleRef M, const char *Name) {
  return wrap(unwrap(M)->getNamedGlobal(Name));
}

LLVMValueRef LLVMGetFirstGlobal(LLVMModuleRef M) {
  Module *Mod = unwrap(M);
  Module::global_iterator I = Mod->global_begin();
  if (I == Mod->global_end())
    return nullptr;
  return wrap(I);
}

LLVMValueRef LLVMGetLastGlobal(LLVMModuleRef M) {
  Module *Mod = unwrap(M);
  Module::global_iterator I = Mod->global_end();
  if (I == Mod->global_begin())
    return nullptr;
  return wrap(--I);
}

LLVMValueRef LLVMGetNextGlobal(LLVMValueRef GlobalVar) {
  GlobalVariable *GV = unwrap<GlobalVariable>(GlobalVar);
  Module::global_iterator I = GV;
  if (++I == GV->getParent()->global_end())
    return nullptr;
  return wrap(I);
}

LLVMValueRef LLVMGetPreviousGlobal(LLVMValueRef GlobalVar) {
  GlobalVariable *GV = unwrap<GlobalVariable>(GlobalVar);
  Module::global_iterator I = GV;
  if (I == GV->getParent()->global_begin())
    return nullptr;
  return wrap(--I);
}

void LLVMDeleteGlobal(LLVMValueRef GlobalVar) {
  unwrap<GlobalVariable>(GlobalVar)->eraseFromParent();
}

LLVMValueRef LLVMGetInitializer(LLVMValueRef GlobalVar) {
  GlobalVariable* GV = unwrap<GlobalVariable>(GlobalVar);
  if ( !GV->hasInitializer() )
    return nullptr;
  return wrap(GV->getInitializer());
}

void LLVMSetInitializer(LLVMValueRef GlobalVar, LLVMValueRef ConstantVal) {
  unwrap<GlobalVariable>(GlobalVar)
    ->setInitializer(unwrap<Constant>(ConstantVal));
}

LLVMBool LLVMIsThreadLocal(LLVMValueRef GlobalVar) {
  return unwrap<GlobalVariable>(GlobalVar)->isThreadLocal();
}

void LLVMSetThreadLocal(LLVMValueRef GlobalVar, LLVMBool IsThreadLocal) {
  unwrap<GlobalVariable>(GlobalVar)->setThreadLocal(IsThreadLocal != 0);
}

LLVMBool LLVMIsGlobalConstant(LLVMValueRef GlobalVar) {
  return unwrap<GlobalVariable>(GlobalVar)->isConstant();
}

void LLVMSetGlobalConstant(LLVMValueRef GlobalVar, LLVMBool IsConstant) {
  unwrap<GlobalVariable>(GlobalVar)->setConstant(IsConstant != 0);
}

LLVMThreadLocalMode LLVMGetThreadLocalMode(LLVMValueRef GlobalVar) {
  switch (unwrap<GlobalVariable>(GlobalVar)->getThreadLocalMode()) {
  case GlobalVariable::NotThreadLocal:
    return LLVMNotThreadLocal;
  case GlobalVariable::GeneralDynamicTLSModel:
    return LLVMGeneralDynamicTLSModel;
  case GlobalVariable::LocalDynamicTLSModel:
    return LLVMLocalDynamicTLSModel;
  case GlobalVariable::InitialExecTLSModel:
    return LLVMInitialExecTLSModel;
  case GlobalVariable::LocalExecTLSModel:
    return LLVMLocalExecTLSModel;
  }

  llvm_unreachable("Invalid GlobalVariable thread local mode");
}

void LLVMSetThreadLocalMode(LLVMValueRef GlobalVar, LLVMThreadLocalMode Mode) {
  GlobalVariable *GV = unwrap<GlobalVariable>(GlobalVar);

  switch (Mode) {
  case LLVMNotThreadLocal:
    GV->setThreadLocalMode(GlobalVariable::NotThreadLocal);
    break;
  case LLVMGeneralDynamicTLSModel:
    GV->setThreadLocalMode(GlobalVariable::GeneralDynamicTLSModel);
    break;
  case LLVMLocalDynamicTLSModel:
    GV->setThreadLocalMode(GlobalVariable::LocalDynamicTLSModel);
    break;
  case LLVMInitialExecTLSModel:
    GV->setThreadLocalMode(GlobalVariable::InitialExecTLSModel);
    break;
  case LLVMLocalExecTLSModel:
    GV->setThreadLocalMode(GlobalVariable::LocalExecTLSModel);
    break;
  }
}

LLVMBool LLVMIsExternallyInitialized(LLVMValueRef GlobalVar) {
  return unwrap<GlobalVariable>(GlobalVar)->isExternallyInitialized();
}

void LLVMSetExternallyInitialized(LLVMValueRef GlobalVar, LLVMBool IsExtInit) {
  unwrap<GlobalVariable>(GlobalVar)->setExternallyInitialized(IsExtInit);
}

/*--.. Operations on aliases ......................................--*/

LLVMValueRef LLVMAddAlias(LLVMModuleRef M, LLVMTypeRef Ty, LLVMValueRef Aliasee,
                          const char *Name) {
  auto *PTy = cast<PointerType>(unwrap(Ty));
  return wrap(GlobalAlias::create(PTy, GlobalValue::ExternalLinkage, Name,
                                  unwrap<Constant>(Aliasee), unwrap(M)));
}

/*--.. Operations on functions .............................................--*/

LLVMValueRef LLVMAddFunction(LLVMModuleRef M, const char *Name,
                             LLVMTypeRef FunctionTy) {
  return wrap(Function::Create(unwrap<FunctionType>(FunctionTy),
                               GlobalValue::ExternalLinkage, Name, unwrap(M)));
}

LLVMValueRef LLVMGetNamedFunction(LLVMModuleRef M, const char *Name) {
  return wrap(unwrap(M)->getFunction(Name));
}

LLVMValueRef LLVMGetFirstFunction(LLVMModuleRef M) {
  Module *Mod = unwrap(M);
  Module::iterator I = Mod->begin();
  if (I == Mod->end())
    return nullptr;
  return wrap(I);
}

LLVMValueRef LLVMGetLastFunction(LLVMModuleRef M) {
  Module *Mod = unwrap(M);
  Module::iterator I = Mod->end();
  if (I == Mod->begin())
    return nullptr;
  return wrap(--I);
}

LLVMValueRef LLVMGetNextFunction(LLVMValueRef Fn) {
  Function *Func = unwrap<Function>(Fn);
  Module::iterator I = Func;
  if (++I == Func->getParent()->end())
    return nullptr;
  return wrap(I);
}

LLVMValueRef LLVMGetPreviousFunction(LLVMValueRef Fn) {
  Function *Func = unwrap<Function>(Fn);
  Module::iterator I = Func;
  if (I == Func->getParent()->begin())
    return nullptr;
  return wrap(--I);
}

void LLVMDeleteFunction(LLVMValueRef Fn) {
  unwrap<Function>(Fn)->eraseFromParent();
}

LLVMValueRef LLVMGetPersonalityFn(LLVMValueRef Fn) {
  return wrap(unwrap<Function>(Fn)->getPersonalityFn());
}

void LLVMSetPersonalityFn(LLVMValueRef Fn, LLVMValueRef PersonalityFn) {
  unwrap<Function>(Fn)->setPersonalityFn(unwrap<Constant>(PersonalityFn));
}

unsigned LLVMGetIntrinsicID(LLVMValueRef Fn) {
  if (Function *F = dyn_cast<Function>(unwrap(Fn)))
    return F->getIntrinsicID();
  return 0;
}

unsigned LLVMGetFunctionCallConv(LLVMValueRef Fn) {
  return unwrap<Function>(Fn)->getCallingConv();
}

void LLVMSetFunctionCallConv(LLVMValueRef Fn, unsigned CC) {
  return unwrap<Function>(Fn)->setCallingConv(
    static_cast<CallingConv::ID>(CC));
}

const char *LLVMGetGC(LLVMValueRef Fn) {
  Function *F = unwrap<Function>(Fn);
  return F->hasGC()? F->getGC() : nullptr;
}

void LLVMSetGC(LLVMValueRef Fn, const char *GC) {
  Function *F = unwrap<Function>(Fn);
  if (GC)
    F->setGC(GC);
  else
    F->clearGC();
}

void LLVMAddFunctionAttr(LLVMValueRef Fn, LLVMAttribute PA) {
  Function *Func = unwrap<Function>(Fn);
  const AttributeSet PAL = Func->getAttributes();
  AttrBuilder B(PA);
  const AttributeSet PALnew =
    PAL.addAttributes(Func->getContext(), AttributeSet::FunctionIndex,
                      AttributeSet::get(Func->getContext(),
                                        AttributeSet::FunctionIndex, B));
  Func->setAttributes(PALnew);
}

void LLVMAddTargetDependentFunctionAttr(LLVMValueRef Fn, const char *A,
                                        const char *V) {
  Function *Func = unwrap<Function>(Fn);
  AttributeSet::AttrIndex Idx =
    AttributeSet::AttrIndex(AttributeSet::FunctionIndex);
  AttrBuilder B;

  B.addAttribute(A, V);
  AttributeSet Set = AttributeSet::get(Func->getContext(), Idx, B);
  Func->addAttributes(Idx, Set);
}

void LLVMRemoveFunctionAttr(LLVMValueRef Fn, LLVMAttribute PA) {
  Function *Func = unwrap<Function>(Fn);
  const AttributeSet PAL = Func->getAttributes();
  AttrBuilder B(PA);
  const AttributeSet PALnew =
    PAL.removeAttributes(Func->getContext(), AttributeSet::FunctionIndex,
                         AttributeSet::get(Func->getContext(),
                                           AttributeSet::FunctionIndex, B));
  Func->setAttributes(PALnew);
}

LLVMAttribute LLVMGetFunctionAttr(LLVMValueRef Fn) {
  Function *Func = unwrap<Function>(Fn);
  const AttributeSet PAL = Func->getAttributes();
  return (LLVMAttribute)PAL.Raw(AttributeSet::FunctionIndex);
}

/*--.. Operations on parameters ............................................--*/

unsigned LLVMCountParams(LLVMValueRef FnRef) {
  // This function is strictly redundant to
  //   LLVMCountParamTypes(LLVMGetElementType(LLVMTypeOf(FnRef)))
  return unwrap<Function>(FnRef)->arg_size();
}

void LLVMGetParams(LLVMValueRef FnRef, LLVMValueRef *ParamRefs) {
  Function *Fn = unwrap<Function>(FnRef);
  for (Function::arg_iterator I = Fn->arg_begin(),
                              E = Fn->arg_end(); I != E; I++)
    *ParamRefs++ = wrap(I);
}

LLVMValueRef LLVMGetParam(LLVMValueRef FnRef, unsigned index) {
  Function::arg_iterator AI = unwrap<Function>(FnRef)->arg_begin();
  while (index --> 0)
    AI++;
  return wrap(AI);
}

LLVMValueRef LLVMGetParamParent(LLVMValueRef V) {
  return wrap(unwrap<Argument>(V)->getParent());
}

LLVMValueRef LLVMGetFirstParam(LLVMValueRef Fn) {
  Function *Func = unwrap<Function>(Fn);
  Function::arg_iterator I = Func->arg_begin();
  if (I == Func->arg_end())
    return nullptr;
  return wrap(I);
}

LLVMValueRef LLVMGetLastParam(LLVMValueRef Fn) {
  Function *Func = unwrap<Function>(Fn);
  Function::arg_iterator I = Func->arg_end();
  if (I == Func->arg_begin())
    return nullptr;
  return wrap(--I);
}

LLVMValueRef LLVMGetNextParam(LLVMValueRef Arg) {
  Argument *A = unwrap<Argument>(Arg);
  Function::arg_iterator I = A;
  if (++I == A->getParent()->arg_end())
    return nullptr;
  return wrap(I);
}

LLVMValueRef LLVMGetPreviousParam(LLVMValueRef Arg) {
  Argument *A = unwrap<Argument>(Arg);
  Function::arg_iterator I = A;
  if (I == A->getParent()->arg_begin())
    return nullptr;
  return wrap(--I);
}

void LLVMAddAttribute(LLVMValueRef Arg, LLVMAttribute PA) {
  Argument *A = unwrap<Argument>(Arg);
  AttrBuilder B(PA);
  A->addAttr(AttributeSet::get(A->getContext(), A->getArgNo() + 1,  B));
}

void LLVMRemoveAttribute(LLVMValueRef Arg, LLVMAttribute PA) {
  Argument *A = unwrap<Argument>(Arg);
  AttrBuilder B(PA);
  A->removeAttr(AttributeSet::get(A->getContext(), A->getArgNo() + 1,  B));
}

LLVMAttribute LLVMGetAttribute(LLVMValueRef Arg) {
  Argument *A = unwrap<Argument>(Arg);
  return (LLVMAttribute)A->getParent()->getAttributes().
    Raw(A->getArgNo()+1);
}


void LLVMSetParamAlignment(LLVMValueRef Arg, unsigned align) {
  Argument *A = unwrap<Argument>(Arg);
  AttrBuilder B;
  B.addAlignmentAttr(align);
  A->addAttr(AttributeSet::get(A->getContext(),A->getArgNo() + 1, B));
}

/*--.. Operations on basic blocks ..........................................--*/

LLVMValueRef LLVMBasicBlockAsValue(LLVMBasicBlockRef BB) {
  return wrap(static_cast<Value*>(unwrap(BB)));
}

LLVMBool LLVMValueIsBasicBlock(LLVMValueRef Val) {
  return isa<BasicBlock>(unwrap(Val));
}

LLVMBasicBlockRef LLVMValueAsBasicBlock(LLVMValueRef Val) {
  return wrap(unwrap<BasicBlock>(Val));
}

LLVMValueRef LLVMGetBasicBlockParent(LLVMBasicBlockRef BB) {
  return wrap(unwrap(BB)->getParent());
}

LLVMValueRef LLVMGetBasicBlockTerminator(LLVMBasicBlockRef BB) {
  return wrap(unwrap(BB)->getTerminator());
}

unsigned LLVMCountBasicBlocks(LLVMValueRef FnRef) {
  return unwrap<Function>(FnRef)->size();
}

void LLVMGetBasicBlocks(LLVMValueRef FnRef, LLVMBasicBlockRef *BasicBlocksRefs){
  Function *Fn = unwrap<Function>(FnRef);
  for (Function::iterator I = Fn->begin(), E = Fn->end(); I != E; I++)
    *BasicBlocksRefs++ = wrap(I);
}

LLVMBasicBlockRef LLVMGetEntryBasicBlock(LLVMValueRef Fn) {
  return wrap(&unwrap<Function>(Fn)->getEntryBlock());
}

LLVMBasicBlockRef LLVMGetFirstBasicBlock(LLVMValueRef Fn) {
  Function *Func = unwrap<Function>(Fn);
  Function::iterator I = Func->begin();
  if (I == Func->end())
    return nullptr;
  return wrap(I);
}

LLVMBasicBlockRef LLVMGetLastBasicBlock(LLVMValueRef Fn) {
  Function *Func = unwrap<Function>(Fn);
  Function::iterator I = Func->end();
  if (I == Func->begin())
    return nullptr;
  return wrap(--I);
}

LLVMBasicBlockRef LLVMGetNextBasicBlock(LLVMBasicBlockRef BB) {
  BasicBlock *Block = unwrap(BB);
  Function::iterator I = Block;
  if (++I == Block->getParent()->end())
    return nullptr;
  return wrap(I);
}

LLVMBasicBlockRef LLVMGetPreviousBasicBlock(LLVMBasicBlockRef BB) {
  BasicBlock *Block = unwrap(BB);
  Function::iterator I = Block;
  if (I == Block->getParent()->begin())
    return nullptr;
  return wrap(--I);
}

LLVMBasicBlockRef LLVMAppendBasicBlockInContext(LLVMContextRef C,
                                                LLVMValueRef FnRef,
                                                const char *Name) {
  return wrap(BasicBlock::Create(*unwrap(C), Name, unwrap<Function>(FnRef)));
}

LLVMBasicBlockRef LLVMAppendBasicBlock(LLVMValueRef FnRef, const char *Name) {
  return LLVMAppendBasicBlockInContext(LLVMGetGlobalContext(), FnRef, Name);
}

LLVMBasicBlockRef LLVMInsertBasicBlockInContext(LLVMContextRef C,
                                                LLVMBasicBlockRef BBRef,
                                                const char *Name) {
  BasicBlock *BB = unwrap(BBRef);
  return wrap(BasicBlock::Create(*unwrap(C), Name, BB->getParent(), BB));
}

LLVMBasicBlockRef LLVMInsertBasicBlock(LLVMBasicBlockRef BBRef,
                                       const char *Name) {
  return LLVMInsertBasicBlockInContext(LLVMGetGlobalContext(), BBRef, Name);
}

void LLVMDeleteBasicBlock(LLVMBasicBlockRef BBRef) {
  unwrap(BBRef)->eraseFromParent();
}

void LLVMRemoveBasicBlockFromParent(LLVMBasicBlockRef BBRef) {
  unwrap(BBRef)->removeFromParent();
}

void LLVMMoveBasicBlockBefore(LLVMBasicBlockRef BB, LLVMBasicBlockRef MovePos) {
  unwrap(BB)->moveBefore(unwrap(MovePos));
}

void LLVMMoveBasicBlockAfter(LLVMBasicBlockRef BB, LLVMBasicBlockRef MovePos) {
  unwrap(BB)->moveAfter(unwrap(MovePos));
}

/*--.. Operations on instructions ..........................................--*/

LLVMBasicBlockRef LLVMGetInstructionParent(LLVMValueRef Inst) {
  return wrap(unwrap<Instruction>(Inst)->getParent());
}

LLVMValueRef LLVMGetFirstInstruction(LLVMBasicBlockRef BB) {
  BasicBlock *Block = unwrap(BB);
  BasicBlock::iterator I = Block->begin();
  if (I == Block->end())
    return nullptr;
  return wrap(I);
}

LLVMValueRef LLVMGetLastInstruction(LLVMBasicBlockRef BB) {
  BasicBlock *Block = unwrap(BB);
  BasicBlock::iterator I = Block->end();
  if (I == Block->begin())
    return nullptr;
  return wrap(--I);
}

LLVMValueRef LLVMGetNextInstruction(LLVMValueRef Inst) {
  Instruction *Instr = unwrap<Instruction>(Inst);
  BasicBlock::iterator I = Instr;
  if (++I == Instr->getParent()->end())
    return nullptr;
  return wrap(I);
}

LLVMValueRef LLVMGetPreviousInstruction(LLVMValueRef Inst) {
  Instruction *Instr = unwrap<Instruction>(Inst);
  BasicBlock::iterator I = Instr;
  if (I == Instr->getParent()->begin())
    return nullptr;
  return wrap(--I);
}

void LLVMInstructionEraseFromParent(LLVMValueRef Inst) {
  unwrap<Instruction>(Inst)->eraseFromParent();
}

LLVMIntPredicate LLVMGetICmpPredicate(LLVMValueRef Inst) {
  if (ICmpInst *I = dyn_cast<ICmpInst>(unwrap(Inst)))
    return (LLVMIntPredicate)I->getPredicate();
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(unwrap(Inst)))
    if (CE->getOpcode() == Instruction::ICmp)
      return (LLVMIntPredicate)CE->getPredicate();
  return (LLVMIntPredicate)0;
}

LLVMRealPredicate LLVMGetFCmpPredicate(LLVMValueRef Inst) {
  if (FCmpInst *I = dyn_cast<FCmpInst>(unwrap(Inst)))
    return (LLVMRealPredicate)I->getPredicate();
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(unwrap(Inst)))
    if (CE->getOpcode() == Instruction::FCmp)
      return (LLVMRealPredicate)CE->getPredicate();
  return (LLVMRealPredicate)0;
}

LLVMOpcode LLVMGetInstructionOpcode(LLVMValueRef Inst) {
  if (Instruction *C = dyn_cast<Instruction>(unwrap(Inst)))
    return map_to_llvmopcode(C->getOpcode());
  return (LLVMOpcode)0;
}

LLVMValueRef LLVMInstructionClone(LLVMValueRef Inst) {
  if (Instruction *C = dyn_cast<Instruction>(unwrap(Inst)))
    return wrap(C->clone());
  return nullptr;
}

/*--.. Call and invoke instructions ........................................--*/

unsigned LLVMGetInstructionCallConv(LLVMValueRef Instr) {
  Value *V = unwrap(Instr);
  if (CallInst *CI = dyn_cast<CallInst>(V))
    return CI->getCallingConv();
  if (InvokeInst *II = dyn_cast<InvokeInst>(V))
    return II->getCallingConv();
  llvm_unreachable("LLVMGetInstructionCallConv applies only to call and invoke!");
}

void LLVMSetInstructionCallConv(LLVMValueRef Instr, unsigned CC) {
  Value *V = unwrap(Instr);
  if (CallInst *CI = dyn_cast<CallInst>(V))
    return CI->setCallingConv(static_cast<CallingConv::ID>(CC));
  else if (InvokeInst *II = dyn_cast<InvokeInst>(V))
    return II->setCallingConv(static_cast<CallingConv::ID>(CC));
  llvm_unreachable("LLVMSetInstructionCallConv applies only to call and invoke!");
}

void LLVMAddInstrAttribute(LLVMValueRef Instr, unsigned index,
                           LLVMAttribute PA) {
  CallSite Call = CallSite(unwrap<Instruction>(Instr));
  AttrBuilder B(PA);
  Call.setAttributes(
    Call.getAttributes().addAttributes(Call->getContext(), index,
                                       AttributeSet::get(Call->getContext(),
                                                         index, B)));
}

void LLVMRemoveInstrAttribute(LLVMValueRef Instr, unsigned index,
                              LLVMAttribute PA) {
  CallSite Call = CallSite(unwrap<Instruction>(Instr));
  AttrBuilder B(PA);
  Call.setAttributes(Call.getAttributes()
                       .removeAttributes(Call->getContext(), index,
                                         AttributeSet::get(Call->getContext(),
                                                           index, B)));
}

void LLVMSetInstrParamAlignment(LLVMValueRef Instr, unsigned index,
                                unsigned align) {
  CallSite Call = CallSite(unwrap<Instruction>(Instr));
  AttrBuilder B;
  B.addAlignmentAttr(align);
  Call.setAttributes(Call.getAttributes()
                       .addAttributes(Call->getContext(), index,
                                      AttributeSet::get(Call->getContext(),
                                                        index, B)));
}

/*--.. Operations on call instructions (only) ..............................--*/

LLVMBool LLVMIsTailCall(LLVMValueRef Call) {
  return unwrap<CallInst>(Call)->isTailCall();
}

void LLVMSetTailCall(LLVMValueRef Call, LLVMBool isTailCall) {
  unwrap<CallInst>(Call)->setTailCall(isTailCall);
}

/*--.. Operations on terminators ...........................................--*/

unsigned LLVMGetNumSuccessors(LLVMValueRef Term) {
  return unwrap<TerminatorInst>(Term)->getNumSuccessors();
}

LLVMBasicBlockRef LLVMGetSuccessor(LLVMValueRef Term, unsigned i) {
  return wrap(unwrap<TerminatorInst>(Term)->getSuccessor(i));
}

void LLVMSetSuccessor(LLVMValueRef Term, unsigned i, LLVMBasicBlockRef block) {
  return unwrap<TerminatorInst>(Term)->setSuccessor(i,unwrap(block));
}

/*--.. Operations on branch instructions (only) ............................--*/

LLVMBool LLVMIsConditional(LLVMValueRef Branch) {
  return unwrap<BranchInst>(Branch)->isConditional();
}

LLVMValueRef LLVMGetCondition(LLVMValueRef Branch) {
  return wrap(unwrap<BranchInst>(Branch)->getCondition());
}

void LLVMSetCondition(LLVMValueRef Branch, LLVMValueRef Cond) {
  return unwrap<BranchInst>(Branch)->setCondition(unwrap(Cond));
}

/*--.. Operations on switch instructions (only) ............................--*/

LLVMBasicBlockRef LLVMGetSwitchDefaultDest(LLVMValueRef Switch) {
  return wrap(unwrap<SwitchInst>(Switch)->getDefaultDest());
}

/*--.. Operations on phi nodes .............................................--*/

void LLVMAddIncoming(LLVMValueRef PhiNode, LLVMValueRef *IncomingValues,
                     LLVMBasicBlockRef *IncomingBlocks, unsigned Count) {
  PHINode *PhiVal = unwrap<PHINode>(PhiNode);
  for (unsigned I = 0; I != Count; ++I)
    PhiVal->addIncoming(unwrap(IncomingValues[I]), unwrap(IncomingBlocks[I]));
}

unsigned LLVMCountIncoming(LLVMValueRef PhiNode) {
  return unwrap<PHINode>(PhiNode)->getNumIncomingValues();
}

LLVMValueRef LLVMGetIncomingValue(LLVMValueRef PhiNode, unsigned Index) {
  return wrap(unwrap<PHINode>(PhiNode)->getIncomingValue(Index));
}

LLVMBasicBlockRef LLVMGetIncomingBlock(LLVMValueRef PhiNode, unsigned Index) {
  return wrap(unwrap<PHINode>(PhiNode)->getIncomingBlock(Index));
}


/*===-- Instruction builders ----------------------------------------------===*/

LLVMBuilderRef LLVMCreateBuilderInContext(LLVMContextRef C) {
  return wrap(new IRBuilder<>(*unwrap(C)));
}

LLVMBuilderRef LLVMCreateBuilder(void) {
  return LLVMCreateBuilderInContext(LLVMGetGlobalContext());
}

void LLVMPositionBuilder(LLVMBuilderRef Builder, LLVMBasicBlockRef Block,
                         LLVMValueRef Instr) {
  BasicBlock *BB = unwrap(Block);
  Instruction *I = Instr? unwrap<Instruction>(Instr) : (Instruction*) BB->end();
  unwrap(Builder)->SetInsertPoint(BB, I);
}

void LLVMPositionBuilderBefore(LLVMBuilderRef Builder, LLVMValueRef Instr) {
  Instruction *I = unwrap<Instruction>(Instr);
  unwrap(Builder)->SetInsertPoint(I->getParent(), I);
}

void LLVMPositionBuilderAtEnd(LLVMBuilderRef Builder, LLVMBasicBlockRef Block) {
  BasicBlock *BB = unwrap(Block);
  unwrap(Builder)->SetInsertPoint(BB);
}

LLVMBasicBlockRef LLVMGetInsertBlock(LLVMBuilderRef Builder) {
   return wrap(unwrap(Builder)->GetInsertBlock());
}

void LLVMClearInsertionPosition(LLVMBuilderRef Builder) {
  unwrap(Builder)->ClearInsertionPoint();
}

void LLVMInsertIntoBuilder(LLVMBuilderRef Builder, LLVMValueRef Instr) {
  unwrap(Builder)->Insert(unwrap<Instruction>(Instr));
}

void LLVMInsertIntoBuilderWithName(LLVMBuilderRef Builder, LLVMValueRef Instr,
                                   const char *Name) {
  unwrap(Builder)->Insert(unwrap<Instruction>(Instr), Name);
}

void LLVMDisposeBuilder(LLVMBuilderRef Builder) {
  delete unwrap(Builder);
}

/*--.. Metadata builders ...................................................--*/

void LLVMSetCurrentDebugLocation(LLVMBuilderRef Builder, LLVMValueRef L) {
  MDNode *Loc =
      L ? cast<MDNode>(unwrap<MetadataAsValue>(L)->getMetadata()) : nullptr;
  unwrap(Builder)->SetCurrentDebugLocation(DebugLoc(Loc));
}

LLVMValueRef LLVMGetCurrentDebugLocation(LLVMBuilderRef Builder) {
  LLVMContext &Context = unwrap(Builder)->getContext();
  return wrap(MetadataAsValue::get(
      Context, unwrap(Builder)->getCurrentDebugLocation().getAsMDNode()));
}

void LLVMSetInstDebugLocation(LLVMBuilderRef Builder, LLVMValueRef Inst) {
  unwrap(Builder)->SetInstDebugLocation(unwrap<Instruction>(Inst));
}


/*--.. Instruction builders ................................................--*/

LLVMValueRef LLVMBuildRetVoid(LLVMBuilderRef B) {
  return wrap(unwrap(B)->CreateRetVoid());
}

LLVMValueRef LLVMBuildRet(LLVMBuilderRef B, LLVMValueRef V) {
  return wrap(unwrap(B)->CreateRet(unwrap(V)));
}

LLVMValueRef LLVMBuildAggregateRet(LLVMBuilderRef B, LLVMValueRef *RetVals,
                                   unsigned N) {
  return wrap(unwrap(B)->CreateAggregateRet(unwrap(RetVals), N));
}

LLVMValueRef LLVMBuildBr(LLVMBuilderRef B, LLVMBasicBlockRef Dest) {
  return wrap(unwrap(B)->CreateBr(unwrap(Dest)));
}

LLVMValueRef LLVMBuildCondBr(LLVMBuilderRef B, LLVMValueRef If,
                             LLVMBasicBlockRef Then, LLVMBasicBlockRef Else) {
  return wrap(unwrap(B)->CreateCondBr(unwrap(If), unwrap(Then), unwrap(Else)));
}

LLVMValueRef LLVMBuildSwitch(LLVMBuilderRef B, LLVMValueRef V,
                             LLVMBasicBlockRef Else, unsigned NumCases) {
  return wrap(unwrap(B)->CreateSwitch(unwrap(V), unwrap(Else), NumCases));
}

LLVMValueRef LLVMBuildIndirectBr(LLVMBuilderRef B, LLVMValueRef Addr,
                                 unsigned NumDests) {
  return wrap(unwrap(B)->CreateIndirectBr(unwrap(Addr), NumDests));
}

LLVMValueRef LLVMBuildInvoke(LLVMBuilderRef B, LLVMValueRef Fn,
                             LLVMValueRef *Args, unsigned NumArgs,
                             LLVMBasicBlockRef Then, LLVMBasicBlockRef Catch,
                             const char *Name) {
  return wrap(unwrap(B)->CreateInvoke(unwrap(Fn), unwrap(Then), unwrap(Catch),
                                      makeArrayRef(unwrap(Args), NumArgs),
                                      Name));
}

LLVMValueRef LLVMBuildLandingPad(LLVMBuilderRef B, LLVMTypeRef Ty,
                                 unsigned NumClauses, const char *Name) {
  return wrap(unwrap(B)->CreateLandingPad(unwrap(Ty), NumClauses, Name));
}

LLVMValueRef LLVMBuildResume(LLVMBuilderRef B, LLVMValueRef Exn) {
  return wrap(unwrap(B)->CreateResume(unwrap(Exn)));
}

LLVMValueRef LLVMBuildUnreachable(LLVMBuilderRef B) {
  return wrap(unwrap(B)->CreateUnreachable());
}

void LLVMAddCase(LLVMValueRef Switch, LLVMValueRef OnVal,
                 LLVMBasicBlockRef Dest) {
  unwrap<SwitchInst>(Switch)->addCase(unwrap<ConstantInt>(OnVal), unwrap(Dest));
}

void LLVMAddDestination(LLVMValueRef IndirectBr, LLVMBasicBlockRef Dest) {
  unwrap<IndirectBrInst>(IndirectBr)->addDestination(unwrap(Dest));
}

void LLVMAddClause(LLVMValueRef LandingPad, LLVMValueRef ClauseVal) {
  unwrap<LandingPadInst>(LandingPad)->
    addClause(cast<Constant>(unwrap(ClauseVal)));
}

void LLVMSetCleanup(LLVMValueRef LandingPad, LLVMBool Val) {
  unwrap<LandingPadInst>(LandingPad)->setCleanup(Val);
}

/*--.. Arithmetic ..........................................................--*/

LLVMValueRef LLVMBuildAdd(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateAdd(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildNSWAdd(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateNSWAdd(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildNUWAdd(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateNUWAdd(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildFAdd(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateFAdd(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildSub(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateSub(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildNSWSub(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateNSWSub(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildNUWSub(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateNUWSub(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildFSub(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateFSub(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildMul(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateMul(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildNSWMul(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateNSWMul(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildNUWMul(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateNUWMul(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildFMul(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateFMul(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildUDiv(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateUDiv(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildSDiv(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateSDiv(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildExactSDiv(LLVMBuilderRef B, LLVMValueRef LHS,
                                LLVMValueRef RHS, const char *Name) {
  return wrap(unwrap(B)->CreateExactSDiv(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildFDiv(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateFDiv(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildURem(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateURem(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildSRem(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateSRem(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildFRem(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateFRem(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildShl(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateShl(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildLShr(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateLShr(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildAShr(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateAShr(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildAnd(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateAnd(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildOr(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                         const char *Name) {
  return wrap(unwrap(B)->CreateOr(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildXor(LLVMBuilderRef B, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name) {
  return wrap(unwrap(B)->CreateXor(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildBinOp(LLVMBuilderRef B, LLVMOpcode Op,
                            LLVMValueRef LHS, LLVMValueRef RHS,
                            const char *Name) {
  return wrap(unwrap(B)->CreateBinOp(Instruction::BinaryOps(map_from_llvmopcode(Op)), unwrap(LHS),
                                     unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildNeg(LLVMBuilderRef B, LLVMValueRef V, const char *Name) {
  return wrap(unwrap(B)->CreateNeg(unwrap(V), Name));
}

LLVMValueRef LLVMBuildNSWNeg(LLVMBuilderRef B, LLVMValueRef V,
                             const char *Name) {
  return wrap(unwrap(B)->CreateNSWNeg(unwrap(V), Name));
}

LLVMValueRef LLVMBuildNUWNeg(LLVMBuilderRef B, LLVMValueRef V,
                             const char *Name) {
  return wrap(unwrap(B)->CreateNUWNeg(unwrap(V), Name));
}

LLVMValueRef LLVMBuildFNeg(LLVMBuilderRef B, LLVMValueRef V, const char *Name) {
  return wrap(unwrap(B)->CreateFNeg(unwrap(V), Name));
}

LLVMValueRef LLVMBuildNot(LLVMBuilderRef B, LLVMValueRef V, const char *Name) {
  return wrap(unwrap(B)->CreateNot(unwrap(V), Name));
}

/*--.. Memory ..............................................................--*/

LLVMValueRef LLVMBuildMalloc(LLVMBuilderRef B, LLVMTypeRef Ty,
                             const char *Name) {
  Type* ITy = Type::getInt32Ty(unwrap(B)->GetInsertBlock()->getContext());
  Constant* AllocSize = ConstantExpr::getSizeOf(unwrap(Ty));
  AllocSize = ConstantExpr::getTruncOrBitCast(AllocSize, ITy);
  Instruction* Malloc = CallInst::CreateMalloc(unwrap(B)->GetInsertBlock(),
                                               ITy, unwrap(Ty), AllocSize,
                                               nullptr, nullptr, "");
  return wrap(unwrap(B)->Insert(Malloc, Twine(Name)));
}

LLVMValueRef LLVMBuildArrayMalloc(LLVMBuilderRef B, LLVMTypeRef Ty,
                                  LLVMValueRef Val, const char *Name) {
  Type* ITy = Type::getInt32Ty(unwrap(B)->GetInsertBlock()->getContext());
  Constant* AllocSize = ConstantExpr::getSizeOf(unwrap(Ty));
  AllocSize = ConstantExpr::getTruncOrBitCast(AllocSize, ITy);
  Instruction* Malloc = CallInst::CreateMalloc(unwrap(B)->GetInsertBlock(),
                                               ITy, unwrap(Ty), AllocSize,
                                               unwrap(Val), nullptr, "");
  return wrap(unwrap(B)->Insert(Malloc, Twine(Name)));
}

LLVMValueRef LLVMBuildAlloca(LLVMBuilderRef B, LLVMTypeRef Ty,
                             const char *Name) {
  return wrap(unwrap(B)->CreateAlloca(unwrap(Ty), nullptr, Name));
}

LLVMValueRef LLVMBuildArrayAlloca(LLVMBuilderRef B, LLVMTypeRef Ty,
                                  LLVMValueRef Val, const char *Name) {
  return wrap(unwrap(B)->CreateAlloca(unwrap(Ty), unwrap(Val), Name));
}

LLVMValueRef LLVMBuildFree(LLVMBuilderRef B, LLVMValueRef PointerVal) {
  return wrap(unwrap(B)->Insert(
     CallInst::CreateFree(unwrap(PointerVal), unwrap(B)->GetInsertBlock())));
}


LLVMValueRef LLVMBuildLoad(LLVMBuilderRef B, LLVMValueRef PointerVal,
                           const char *Name) {
  return wrap(unwrap(B)->CreateLoad(unwrap(PointerVal), Name));
}

LLVMValueRef LLVMBuildStore(LLVMBuilderRef B, LLVMValueRef Val,
                            LLVMValueRef PointerVal) {
  return wrap(unwrap(B)->CreateStore(unwrap(Val), unwrap(PointerVal)));
}

static AtomicOrdering mapFromLLVMOrdering(LLVMAtomicOrdering Ordering) {
  switch (Ordering) {
    case LLVMAtomicOrderingNotAtomic: return NotAtomic;
    case LLVMAtomicOrderingUnordered: return Unordered;
    case LLVMAtomicOrderingMonotonic: return Monotonic;
    case LLVMAtomicOrderingAcquire: return Acquire;
    case LLVMAtomicOrderingRelease: return Release;
    case LLVMAtomicOrderingAcquireRelease: return AcquireRelease;
    case LLVMAtomicOrderingSequentiallyConsistent:
      return SequentiallyConsistent;
  }

  llvm_unreachable("Invalid LLVMAtomicOrdering value!");
}

LLVMValueRef LLVMBuildFence(LLVMBuilderRef B, LLVMAtomicOrdering Ordering,
                            LLVMBool isSingleThread, const char *Name) {
  return wrap(
    unwrap(B)->CreateFence(mapFromLLVMOrdering(Ordering),
                           isSingleThread ? SingleThread : CrossThread,
                           Name));
}

LLVMValueRef LLVMBuildGEP(LLVMBuilderRef B, LLVMValueRef Pointer,
                          LLVMValueRef *Indices, unsigned NumIndices,
                          const char *Name) {
  ArrayRef<Value *> IdxList(unwrap(Indices), NumIndices);
  return wrap(unwrap(B)->CreateGEP(nullptr, unwrap(Pointer), IdxList, Name));
}

LLVMValueRef LLVMBuildInBoundsGEP(LLVMBuilderRef B, LLVMValueRef Pointer,
                                  LLVMValueRef *Indices, unsigned NumIndices,
                                  const char *Name) {
  ArrayRef<Value *> IdxList(unwrap(Indices), NumIndices);
  return wrap(
      unwrap(B)->CreateInBoundsGEP(nullptr, unwrap(Pointer), IdxList, Name));
}

LLVMValueRef LLVMBuildStructGEP(LLVMBuilderRef B, LLVMValueRef Pointer,
                                unsigned Idx, const char *Name) {
  return wrap(unwrap(B)->CreateStructGEP(nullptr, unwrap(Pointer), Idx, Name));
}

LLVMValueRef LLVMBuildGlobalString(LLVMBuilderRef B, const char *Str,
                                   const char *Name) {
  return wrap(unwrap(B)->CreateGlobalString(Str, Name));
}

LLVMValueRef LLVMBuildGlobalStringPtr(LLVMBuilderRef B, const char *Str,
                                      const char *Name) {
  return wrap(unwrap(B)->CreateGlobalStringPtr(Str, Name));
}

LLVMBool LLVMGetVolatile(LLVMValueRef MemAccessInst) {
  Value *P = unwrap<Value>(MemAccessInst);
  if (LoadInst *LI = dyn_cast<LoadInst>(P))
    return LI->isVolatile();
  return cast<StoreInst>(P)->isVolatile();
}

void LLVMSetVolatile(LLVMValueRef MemAccessInst, LLVMBool isVolatile) {
  Value *P = unwrap<Value>(MemAccessInst);
  if (LoadInst *LI = dyn_cast<LoadInst>(P))
    return LI->setVolatile(isVolatile);
  return cast<StoreInst>(P)->setVolatile(isVolatile);
}

/*--.. Casts ...............................................................--*/

LLVMValueRef LLVMBuildTrunc(LLVMBuilderRef B, LLVMValueRef Val,
                            LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateTrunc(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildZExt(LLVMBuilderRef B, LLVMValueRef Val,
                           LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateZExt(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildSExt(LLVMBuilderRef B, LLVMValueRef Val,
                           LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateSExt(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildFPToUI(LLVMBuilderRef B, LLVMValueRef Val,
                             LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateFPToUI(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildFPToSI(LLVMBuilderRef B, LLVMValueRef Val,
                             LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateFPToSI(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildUIToFP(LLVMBuilderRef B, LLVMValueRef Val,
                             LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateUIToFP(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildSIToFP(LLVMBuilderRef B, LLVMValueRef Val,
                             LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateSIToFP(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildFPTrunc(LLVMBuilderRef B, LLVMValueRef Val,
                              LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateFPTrunc(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildFPExt(LLVMBuilderRef B, LLVMValueRef Val,
                            LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateFPExt(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildPtrToInt(LLVMBuilderRef B, LLVMValueRef Val,
                               LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreatePtrToInt(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildIntToPtr(LLVMBuilderRef B, LLVMValueRef Val,
                               LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateIntToPtr(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildBitCast(LLVMBuilderRef B, LLVMValueRef Val,
                              LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateBitCast(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildAddrSpaceCast(LLVMBuilderRef B, LLVMValueRef Val,
                                    LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateAddrSpaceCast(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildZExtOrBitCast(LLVMBuilderRef B, LLVMValueRef Val,
                                    LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateZExtOrBitCast(unwrap(Val), unwrap(DestTy),
                                             Name));
}

LLVMValueRef LLVMBuildSExtOrBitCast(LLVMBuilderRef B, LLVMValueRef Val,
                                    LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateSExtOrBitCast(unwrap(Val), unwrap(DestTy),
                                             Name));
}

LLVMValueRef LLVMBuildTruncOrBitCast(LLVMBuilderRef B, LLVMValueRef Val,
                                     LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateTruncOrBitCast(unwrap(Val), unwrap(DestTy),
                                              Name));
}

LLVMValueRef LLVMBuildCast(LLVMBuilderRef B, LLVMOpcode Op, LLVMValueRef Val,
                           LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateCast(Instruction::CastOps(map_from_llvmopcode(Op)), unwrap(Val),
                                    unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildPointerCast(LLVMBuilderRef B, LLVMValueRef Val,
                                  LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreatePointerCast(unwrap(Val), unwrap(DestTy), Name));
}

LLVMValueRef LLVMBuildIntCast(LLVMBuilderRef B, LLVMValueRef Val,
                              LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateIntCast(unwrap(Val), unwrap(DestTy),
                                       /*isSigned*/true, Name));
}

LLVMValueRef LLVMBuildFPCast(LLVMBuilderRef B, LLVMValueRef Val,
                             LLVMTypeRef DestTy, const char *Name) {
  return wrap(unwrap(B)->CreateFPCast(unwrap(Val), unwrap(DestTy), Name));
}

/*--.. Comparisons .........................................................--*/

LLVMValueRef LLVMBuildICmp(LLVMBuilderRef B, LLVMIntPredicate Op,
                           LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateICmp(static_cast<ICmpInst::Predicate>(Op),
                                    unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildFCmp(LLVMBuilderRef B, LLVMRealPredicate Op,
                           LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name) {
  return wrap(unwrap(B)->CreateFCmp(static_cast<FCmpInst::Predicate>(Op),
                                    unwrap(LHS), unwrap(RHS), Name));
}

/*--.. Miscellaneous instructions ..........................................--*/

LLVMValueRef LLVMBuildPhi(LLVMBuilderRef B, LLVMTypeRef Ty, const char *Name) {
  return wrap(unwrap(B)->CreatePHI(unwrap(Ty), 0, Name));
}

LLVMValueRef LLVMBuildCall(LLVMBuilderRef B, LLVMValueRef Fn,
                           LLVMValueRef *Args, unsigned NumArgs,
                           const char *Name) {
  return wrap(unwrap(B)->CreateCall(unwrap(Fn),
                                    makeArrayRef(unwrap(Args), NumArgs),
                                    Name));
}

LLVMValueRef LLVMBuildSelect(LLVMBuilderRef B, LLVMValueRef If,
                             LLVMValueRef Then, LLVMValueRef Else,
                             const char *Name) {
  return wrap(unwrap(B)->CreateSelect(unwrap(If), unwrap(Then), unwrap(Else),
                                      Name));
}

LLVMValueRef LLVMBuildVAArg(LLVMBuilderRef B, LLVMValueRef List,
                            LLVMTypeRef Ty, const char *Name) {
  return wrap(unwrap(B)->CreateVAArg(unwrap(List), unwrap(Ty), Name));
}

LLVMValueRef LLVMBuildExtractElement(LLVMBuilderRef B, LLVMValueRef VecVal,
                                      LLVMValueRef Index, const char *Name) {
  return wrap(unwrap(B)->CreateExtractElement(unwrap(VecVal), unwrap(Index),
                                              Name));
}

LLVMValueRef LLVMBuildInsertElement(LLVMBuilderRef B, LLVMValueRef VecVal,
                                    LLVMValueRef EltVal, LLVMValueRef Index,
                                    const char *Name) {
  return wrap(unwrap(B)->CreateInsertElement(unwrap(VecVal), unwrap(EltVal),
                                             unwrap(Index), Name));
}

LLVMValueRef LLVMBuildShuffleVector(LLVMBuilderRef B, LLVMValueRef V1,
                                    LLVMValueRef V2, LLVMValueRef Mask,
                                    const char *Name) {
  return wrap(unwrap(B)->CreateShuffleVector(unwrap(V1), unwrap(V2),
                                             unwrap(Mask), Name));
}

LLVMValueRef LLVMBuildExtractValue(LLVMBuilderRef B, LLVMValueRef AggVal,
                                   unsigned Index, const char *Name) {
  return wrap(unwrap(B)->CreateExtractValue(unwrap(AggVal), Index, Name));
}

LLVMValueRef LLVMBuildInsertValue(LLVMBuilderRef B, LLVMValueRef AggVal,
                                  LLVMValueRef EltVal, unsigned Index,
                                  const char *Name) {
  return wrap(unwrap(B)->CreateInsertValue(unwrap(AggVal), unwrap(EltVal),
                                           Index, Name));
}

LLVMValueRef LLVMBuildIsNull(LLVMBuilderRef B, LLVMValueRef Val,
                             const char *Name) {
  return wrap(unwrap(B)->CreateIsNull(unwrap(Val), Name));
}

LLVMValueRef LLVMBuildIsNotNull(LLVMBuilderRef B, LLVMValueRef Val,
                                const char *Name) {
  return wrap(unwrap(B)->CreateIsNotNull(unwrap(Val), Name));
}

LLVMValueRef LLVMBuildPtrDiff(LLVMBuilderRef B, LLVMValueRef LHS,
                              LLVMValueRef RHS, const char *Name) {
  return wrap(unwrap(B)->CreatePtrDiff(unwrap(LHS), unwrap(RHS), Name));
}

LLVMValueRef LLVMBuildAtomicRMW(LLVMBuilderRef B,LLVMAtomicRMWBinOp op,
                               LLVMValueRef PTR, LLVMValueRef Val,
                               LLVMAtomicOrdering ordering,
                               LLVMBool singleThread) {
  AtomicRMWInst::BinOp intop;
  switch (op) {
    case LLVMAtomicRMWBinOpXchg: intop = AtomicRMWInst::Xchg; break;
    case LLVMAtomicRMWBinOpAdd: intop = AtomicRMWInst::Add; break;
    case LLVMAtomicRMWBinOpSub: intop = AtomicRMWInst::Sub; break;
    case LLVMAtomicRMWBinOpAnd: intop = AtomicRMWInst::And; break;
    case LLVMAtomicRMWBinOpNand: intop = AtomicRMWInst::Nand; break;
    case LLVMAtomicRMWBinOpOr: intop = AtomicRMWInst::Or; break;
    case LLVMAtomicRMWBinOpXor: intop = AtomicRMWInst::Xor; break;
    case LLVMAtomicRMWBinOpMax: intop = AtomicRMWInst::Max; break;
    case LLVMAtomicRMWBinOpMin: intop = AtomicRMWInst::Min; break;
    case LLVMAtomicRMWBinOpUMax: intop = AtomicRMWInst::UMax; break;
    case LLVMAtomicRMWBinOpUMin: intop = AtomicRMWInst::UMin; break;
  }
  return wrap(unwrap(B)->CreateAtomicRMW(intop, unwrap(PTR), unwrap(Val),
    mapFromLLVMOrdering(ordering), singleThread ? SingleThread : CrossThread));
}


/*===-- Module providers --------------------------------------------------===*/

LLVMModuleProviderRef
LLVMCreateModuleProviderForExistingModule(LLVMModuleRef M) {
  return reinterpret_cast<LLVMModuleProviderRef>(M);
}

void LLVMDisposeModuleProvider(LLVMModuleProviderRef MP) {
  delete unwrap(MP);
}


/*===-- Memory buffers ----------------------------------------------------===*/

LLVMBool LLVMCreateMemoryBufferWithContentsOfFile(
    const char *Path,
    LLVMMemoryBufferRef *OutMemBuf,
    char **OutMessage) {

  ErrorOr<std::unique_ptr<MemoryBuffer>> MBOrErr = MemoryBuffer::getFile(Path);
  if (std::error_code EC = MBOrErr.getError()) {
    *OutMessage = strdup(EC.message().c_str());
    return 1;
  }
  *OutMemBuf = wrap(MBOrErr.get().release());
  return 0;
}

LLVMBool LLVMCreateMemoryBufferWithSTDIN(LLVMMemoryBufferRef *OutMemBuf,
                                         char **OutMessage) {
  ErrorOr<std::unique_ptr<MemoryBuffer>> MBOrErr = MemoryBuffer::getSTDIN();
  if (std::error_code EC = MBOrErr.getError()) {
    *OutMessage = strdup(EC.message().c_str());
    return 1;
  }
  *OutMemBuf = wrap(MBOrErr.get().release());
  return 0;
}

LLVMMemoryBufferRef LLVMCreateMemoryBufferWithMemoryRange(
    const char *InputData,
    size_t InputDataLength,
    const char *BufferName,
    LLVMBool RequiresNullTerminator) {

  return wrap(MemoryBuffer::getMemBuffer(StringRef(InputData, InputDataLength),
                                         StringRef(BufferName),
                                         RequiresNullTerminator).release());
}

LLVMMemoryBufferRef LLVMCreateMemoryBufferWithMemoryRangeCopy(
    const char *InputData,
    size_t InputDataLength,
    const char *BufferName) {

  return wrap(
      MemoryBuffer::getMemBufferCopy(StringRef(InputData, InputDataLength),
                                     StringRef(BufferName)).release());
}

const char *LLVMGetBufferStart(LLVMMemoryBufferRef MemBuf) {
  return unwrap(MemBuf)->getBufferStart();
}

size_t LLVMGetBufferSize(LLVMMemoryBufferRef MemBuf) {
  return unwrap(MemBuf)->getBufferSize();
}

void LLVMDisposeMemoryBuffer(LLVMMemoryBufferRef MemBuf) {
  delete unwrap(MemBuf);
}

/*===-- Pass Registry -----------------------------------------------------===*/

LLVMPassRegistryRef LLVMGetGlobalPassRegistry(void) {
  return wrap(PassRegistry::getPassRegistry());
}

/*===-- Pass Manager ------------------------------------------------------===*/

LLVMPassManagerRef LLVMCreatePassManager() {
  return wrap(new legacy::PassManager());
}

LLVMPassManagerRef LLVMCreateFunctionPassManagerForModule(LLVMModuleRef M) {
  return wrap(new legacy::FunctionPassManager(unwrap(M)));
}

LLVMPassManagerRef LLVMCreateFunctionPassManager(LLVMModuleProviderRef P) {
  return LLVMCreateFunctionPassManagerForModule(
                                            reinterpret_cast<LLVMModuleRef>(P));
}

LLVMBool LLVMRunPassManager(LLVMPassManagerRef PM, LLVMModuleRef M) {
  return unwrap<legacy::PassManager>(PM)->run(*unwrap(M));
}

LLVMBool LLVMInitializeFunctionPassManager(LLVMPassManagerRef FPM) {
  return unwrap<legacy::FunctionPassManager>(FPM)->doInitialization();
}

LLVMBool LLVMRunFunctionPassManager(LLVMPassManagerRef FPM, LLVMValueRef F) {
  return unwrap<legacy::FunctionPassManager>(FPM)->run(*unwrap<Function>(F));
}

LLVMBool LLVMFinalizeFunctionPassManager(LLVMPassManagerRef FPM) {
  return unwrap<legacy::FunctionPassManager>(FPM)->doFinalization();
}

void LLVMDisposePassManager(LLVMPassManagerRef PM) {
  delete unwrap(PM);
}

/*===-- Threading ------------------------------------------------------===*/

LLVMBool LLVMStartMultithreaded() {
  return LLVMIsMultithreaded();
}

void LLVMStopMultithreaded() {
}

LLVMBool LLVMIsMultithreaded() {
  return llvm_is_multithreaded();
}
