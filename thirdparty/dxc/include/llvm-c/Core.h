/*===-- llvm-c/Core.h - Core Library C Interface ------------------*- C -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header declares the C interface to libLLVMCore.a, which implements    *|
|* the LLVM intermediate representation.                                      *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_CORE_H
#define LLVM_C_CORE_H

#include "dxc/Support/WinAdapter.h" // HLSL Change
#include "llvm-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @defgroup LLVMC LLVM-C: C interface to LLVM
 *
 * This module exposes parts of the LLVM library as a C API.
 *
 * @{
 */

/**
 * @defgroup LLVMCTransforms Transforms
 */

/**
 * @defgroup LLVMCCore Core
 *
 * This modules provide an interface to libLLVMCore, which implements
 * the LLVM intermediate representation as well as other related types
 * and utilities.
 *
 * LLVM uses a polymorphic type hierarchy which C cannot represent, therefore
 * parameters must be passed as base types. Despite the declared types, most
 * of the functions provided operate only on branches of the type hierarchy.
 * The declared parameter names are descriptive and specify which type is
 * required. Additionally, each type hierarchy is documented along with the
 * functions that operate upon it. For more detail, refer to LLVM's C++ code.
 * If in doubt, refer to Core.cpp, which performs parameter downcasts in the
 * form unwrap<RequiredType>(Param).
 *
 * Many exotic languages can interoperate with C code but have a harder time
 * with C++ due to name mangling. So in addition to C, this interface enables
 * tools written in such languages.
 *
 * @{
 */

/**
 * @defgroup LLVMCCoreTypes Types and Enumerations
 *
 * @{
 */

/* Opaque types. */

/**
 * The top-level container for all LLVM global data. See the LLVMContext class.
 */
typedef struct LLVMOpaqueContext *LLVMContextRef;

/**
 * The top-level container for all other LLVM Intermediate Representation (IR)
 * objects.
 *
 * @see llvm::Module
 */
typedef struct LLVMOpaqueModule *LLVMModuleRef;

/**
 * Each value in the LLVM IR has a type, an LLVMTypeRef.
 *
 * @see llvm::Type
 */
typedef struct LLVMOpaqueType *LLVMTypeRef;

/**
 * Represents an individual value in LLVM IR.
 *
 * This models llvm::Value.
 */
typedef struct LLVMOpaqueValue *LLVMValueRef;

/**
 * Represents a basic block of instructions in LLVM IR.
 *
 * This models llvm::BasicBlock.
 */
typedef struct LLVMOpaqueBasicBlock *LLVMBasicBlockRef;

/**
 * Represents an LLVM basic block builder.
 *
 * This models llvm::IRBuilder.
 */
typedef struct LLVMOpaqueBuilder *LLVMBuilderRef;

/**
 * Interface used to provide a module to JIT or interpreter.
 * This is now just a synonym for llvm::Module, but we have to keep using the
 * different type to keep binary compatibility.
 */
typedef struct LLVMOpaqueModuleProvider *LLVMModuleProviderRef;

/** @see llvm::PassManagerBase */
typedef struct LLVMOpaquePassManager *LLVMPassManagerRef;

/** @see llvm::PassRegistry */
typedef struct LLVMOpaquePassRegistry *LLVMPassRegistryRef;

/**
 * Used to get the users and usees of a Value.
 *
 * @see llvm::Use */
typedef struct LLVMOpaqueUse *LLVMUseRef;


/**
 * @see llvm::DiagnosticInfo
 */
typedef struct LLVMOpaqueDiagnosticInfo *LLVMDiagnosticInfoRef;

typedef enum {
    LLVMZExtAttribute       = 1<<0,
    LLVMSExtAttribute       = 1<<1,
    LLVMNoReturnAttribute   = 1<<2,
    LLVMInRegAttribute      = 1<<3,
    LLVMStructRetAttribute  = 1<<4,
    LLVMNoUnwindAttribute   = 1<<5,
    LLVMNoAliasAttribute    = 1<<6,
    LLVMByValAttribute      = 1<<7,
    LLVMNestAttribute       = 1<<8,
    LLVMReadNoneAttribute   = 1<<9,
    LLVMReadOnlyAttribute   = 1<<10,
    LLVMNoInlineAttribute   = 1<<11,
    LLVMAlwaysInlineAttribute    = 1<<12,
    LLVMOptimizeForSizeAttribute = 1<<13,
    LLVMStackProtectAttribute    = 1<<14,
    LLVMStackProtectReqAttribute = 1<<15,
    LLVMAlignment = 31<<16,
    LLVMNoCaptureAttribute  = 1<<21,
    LLVMNoRedZoneAttribute  = 1<<22,
    LLVMNoImplicitFloatAttribute = 1<<23,
    LLVMNakedAttribute      = 1<<24,
    LLVMInlineHintAttribute = 1<<25,
    LLVMStackAlignment = 7<<26,
    LLVMReturnsTwice = 1 << 29,
    LLVMUWTable = 1 << 30,
    LLVMNonLazyBind = 1 << 31

    /* FIXME: These attributes are currently not included in the C API as
       a temporary measure until the API/ABI impact to the C API is understood
       and the path forward agreed upon.
    LLVMSanitizeAddressAttribute = 1ULL << 32,
    LLVMStackProtectStrongAttribute = 1ULL<<35,
    LLVMColdAttribute = 1ULL << 40,
    LLVMOptimizeNoneAttribute = 1ULL << 42,
    LLVMInAllocaAttribute = 1ULL << 43,
    LLVMNonNullAttribute = 1ULL << 44,
    LLVMJumpTableAttribute = 1ULL << 45,
    LLVMConvergentAttribute = 1ULL << 46,
    LLVMSafeStackAttribute = 1ULL << 47,
    */
} LLVMAttribute;

typedef enum {
  /* Terminator Instructions */
  LLVMRet            = 1,
  LLVMBr             = 2,
  LLVMSwitch         = 3,
  LLVMIndirectBr     = 4,
  LLVMInvoke         = 5,
  /* removed 6 due to API changes */
  LLVMUnreachable    = 7,

  /* Standard Binary Operators */
  LLVMAdd            = 8,
  LLVMFAdd           = 9,
  LLVMSub            = 10,
  LLVMFSub           = 11,
  LLVMMul            = 12,
  LLVMFMul           = 13,
  LLVMUDiv           = 14,
  LLVMSDiv           = 15,
  LLVMFDiv           = 16,
  LLVMURem           = 17,
  LLVMSRem           = 18,
  LLVMFRem           = 19,

  /* Logical Operators */
  LLVMShl            = 20,
  LLVMLShr           = 21,
  LLVMAShr           = 22,
  LLVMAnd            = 23,
  LLVMOr             = 24,
  LLVMXor            = 25,

  /* Memory Operators */
  LLVMAlloca         = 26,
  LLVMLoad           = 27,
  LLVMStore          = 28,
  LLVMGetElementPtr  = 29,

  /* Cast Operators */
  LLVMTrunc          = 30,
  LLVMZExt           = 31,
  LLVMSExt           = 32,
  LLVMFPToUI         = 33,
  LLVMFPToSI         = 34,
  LLVMUIToFP         = 35,
  LLVMSIToFP         = 36,
  LLVMFPTrunc        = 37,
  LLVMFPExt          = 38,
  LLVMPtrToInt       = 39,
  LLVMIntToPtr       = 40,
  LLVMBitCast        = 41,
  LLVMAddrSpaceCast  = 60,

  /* Other Operators */
  LLVMICmp           = 42,
  LLVMFCmp           = 43,
  LLVMPHI            = 44,
  LLVMCall           = 45,
  LLVMSelect         = 46,
  LLVMUserOp1        = 47,
  LLVMUserOp2        = 48,
  LLVMVAArg          = 49,
  LLVMExtractElement = 50,
  LLVMInsertElement  = 51,
  LLVMShuffleVector  = 52,
  LLVMExtractValue   = 53,
  LLVMInsertValue    = 54,

  /* Atomic operators */
  LLVMFence          = 55,
  LLVMAtomicCmpXchg  = 56,
  LLVMAtomicRMW      = 57,

  /* Exception Handling Operators */
  LLVMResume         = 58,
  LLVMLandingPad     = 59

} LLVMOpcode;

typedef enum {
  LLVMVoidTypeKind,        /**< type with no size */
  LLVMHalfTypeKind,        /**< 16 bit floating point type */
  LLVMFloatTypeKind,       /**< 32 bit floating point type */
  LLVMDoubleTypeKind,      /**< 64 bit floating point type */
  LLVMX86_FP80TypeKind,    /**< 80 bit floating point type (X87) */
  LLVMFP128TypeKind,       /**< 128 bit floating point type (112-bit mantissa)*/
  LLVMPPC_FP128TypeKind,   /**< 128 bit floating point type (two 64-bits) */
  LLVMLabelTypeKind,       /**< Labels */
  LLVMIntegerTypeKind,     /**< Arbitrary bit width integers */
  LLVMFunctionTypeKind,    /**< Functions */
  LLVMStructTypeKind,      /**< Structures */
  LLVMArrayTypeKind,       /**< Arrays */
  LLVMPointerTypeKind,     /**< Pointers */
  LLVMVectorTypeKind,      /**< SIMD 'packed' format, or other vector type */
  LLVMMetadataTypeKind,    /**< Metadata */
  LLVMX86_MMXTypeKind      /**< X86 MMX */
} LLVMTypeKind;

typedef enum {
  LLVMExternalLinkage,    /**< Externally visible function */
  LLVMAvailableExternallyLinkage,
  LLVMLinkOnceAnyLinkage, /**< Keep one copy of function when linking (inline)*/
  LLVMLinkOnceODRLinkage, /**< Same, but only replaced by something
                            equivalent. */
  LLVMLinkOnceODRAutoHideLinkage, /**< Obsolete */
  LLVMWeakAnyLinkage,     /**< Keep one copy of function when linking (weak) */
  LLVMWeakODRLinkage,     /**< Same, but only replaced by something
                            equivalent. */
  LLVMAppendingLinkage,   /**< Special purpose, only applies to global arrays */
  LLVMInternalLinkage,    /**< Rename collisions when linking (static
                               functions) */
  LLVMPrivateLinkage,     /**< Like Internal, but omit from symbol table */
  LLVMDLLImportLinkage,   /**< Obsolete */
  LLVMDLLExportLinkage,   /**< Obsolete */
  LLVMExternalWeakLinkage,/**< ExternalWeak linkage description */
  LLVMGhostLinkage,       /**< Obsolete */
  LLVMCommonLinkage,      /**< Tentative definitions */
  LLVMLinkerPrivateLinkage, /**< Like Private, but linker removes. */
  LLVMLinkerPrivateWeakLinkage /**< Like LinkerPrivate, but is weak. */
} LLVMLinkage;

typedef enum {
  LLVMDefaultVisibility,  /**< The GV is visible */
  LLVMHiddenVisibility,   /**< The GV is hidden */
  LLVMProtectedVisibility /**< The GV is protected */
} LLVMVisibility;

typedef enum {
  LLVMDefaultStorageClass   = 0,
  LLVMDLLImportStorageClass = 1, /**< Function to be imported from DLL. */
  LLVMDLLExportStorageClass = 2  /**< Function to be accessible from DLL. */
} LLVMDLLStorageClass;

typedef enum {
  LLVMCCallConv           = 0,
  LLVMFastCallConv        = 8,
  LLVMColdCallConv        = 9,
  LLVMWebKitJSCallConv    = 12,
  LLVMAnyRegCallConv      = 13,
  LLVMX86StdcallCallConv  = 64,
  LLVMX86FastcallCallConv = 65
} LLVMCallConv;

typedef enum {
  LLVMIntEQ = 32, /**< equal */
  LLVMIntNE,      /**< not equal */
  LLVMIntUGT,     /**< unsigned greater than */
  LLVMIntUGE,     /**< unsigned greater or equal */
  LLVMIntULT,     /**< unsigned less than */
  LLVMIntULE,     /**< unsigned less or equal */
  LLVMIntSGT,     /**< signed greater than */
  LLVMIntSGE,     /**< signed greater or equal */
  LLVMIntSLT,     /**< signed less than */
  LLVMIntSLE      /**< signed less or equal */
} LLVMIntPredicate;

typedef enum {
  LLVMRealPredicateFalse, /**< Always false (always folded) */
  LLVMRealOEQ,            /**< True if ordered and equal */
  LLVMRealOGT,            /**< True if ordered and greater than */
  LLVMRealOGE,            /**< True if ordered and greater than or equal */
  LLVMRealOLT,            /**< True if ordered and less than */
  LLVMRealOLE,            /**< True if ordered and less than or equal */
  LLVMRealONE,            /**< True if ordered and operands are unequal */
  LLVMRealORD,            /**< True if ordered (no nans) */
  LLVMRealUNO,            /**< True if unordered: isnan(X) | isnan(Y) */
  LLVMRealUEQ,            /**< True if unordered or equal */
  LLVMRealUGT,            /**< True if unordered or greater than */
  LLVMRealUGE,            /**< True if unordered, greater than, or equal */
  LLVMRealULT,            /**< True if unordered or less than */
  LLVMRealULE,            /**< True if unordered, less than, or equal */
  LLVMRealUNE,            /**< True if unordered or not equal */
  LLVMRealPredicateTrue   /**< Always true (always folded) */
} LLVMRealPredicate;

typedef enum {
  LLVMLandingPadCatch,    /**< A catch clause   */
  LLVMLandingPadFilter    /**< A filter clause  */
} LLVMLandingPadClauseTy;

typedef enum {
  LLVMNotThreadLocal = 0,
  LLVMGeneralDynamicTLSModel,
  LLVMLocalDynamicTLSModel,
  LLVMInitialExecTLSModel,
  LLVMLocalExecTLSModel
} LLVMThreadLocalMode;

typedef enum {
  LLVMAtomicOrderingNotAtomic = 0, /**< A load or store which is not atomic */
  LLVMAtomicOrderingUnordered = 1, /**< Lowest level of atomicity, guarantees
                                     somewhat sane results, lock free. */
  LLVMAtomicOrderingMonotonic = 2, /**< guarantees that if you take all the
                                     operations affecting a specific address,
                                     a consistent ordering exists */
  LLVMAtomicOrderingAcquire = 4, /**< Acquire provides a barrier of the sort
                                   necessary to acquire a lock to access other
                                   memory with normal loads and stores. */
  LLVMAtomicOrderingRelease = 5, /**< Release is similar to Acquire, but with
                                   a barrier of the sort necessary to release
                                   a lock. */
  LLVMAtomicOrderingAcquireRelease = 6, /**< provides both an Acquire and a
                                          Release barrier (for fences and
                                          operations which both read and write
                                           memory). */
  LLVMAtomicOrderingSequentiallyConsistent = 7 /**< provides Acquire semantics
                                                 for loads and Release
                                                 semantics for stores.
                                                 Additionally, it guarantees
                                                 that a total ordering exists
                                                 between all
                                                 SequentiallyConsistent
                                                 operations. */
} LLVMAtomicOrdering;

typedef enum {
    LLVMAtomicRMWBinOpXchg, /**< Set the new value and return the one old */
    LLVMAtomicRMWBinOpAdd, /**< Add a value and return the old one */
    LLVMAtomicRMWBinOpSub, /**< Subtract a value and return the old one */
    LLVMAtomicRMWBinOpAnd, /**< And a value and return the old one */
    LLVMAtomicRMWBinOpNand, /**< Not-And a value and return the old one */
    LLVMAtomicRMWBinOpOr, /**< OR a value and return the old one */
    LLVMAtomicRMWBinOpXor, /**< Xor a value and return the old one */
    LLVMAtomicRMWBinOpMax, /**< Sets the value if it's greater than the
                             original using a signed comparison and return
                             the old one */
    LLVMAtomicRMWBinOpMin, /**< Sets the value if it's Smaller than the
                             original using a signed comparison and return
                             the old one */
    LLVMAtomicRMWBinOpUMax, /**< Sets the value if it's greater than the
                             original using an unsigned comparison and return
                             the old one */
    LLVMAtomicRMWBinOpUMin /**< Sets the value if it's greater than the
                             original using an unsigned comparison  and return
                             the old one */
} LLVMAtomicRMWBinOp;

typedef enum {
    LLVMDSError,
    LLVMDSWarning,
    LLVMDSRemark,
    LLVMDSNote
} LLVMDiagnosticSeverity;

/**
 * @}
 */

void LLVMInitializeCore(LLVMPassRegistryRef R);

/** Deallocate and destroy all ManagedStatic variables.
    @see llvm::llvm_shutdown
    @see ManagedStatic */
void LLVMShutdown(void);


/*===-- Error handling ----------------------------------------------------===*/

char *LLVMCreateMessage(const char *Message);
void LLVMDisposeMessage(_Out_opt_ char *Message);

typedef void (*LLVMFatalErrorHandler)(const char *Reason);

/**
 * Install a fatal error handler. By default, if LLVM detects a fatal error, it
 * will call exit(1). This may not be appropriate in many contexts. For example,
 * doing exit(1) will bypass many crash reporting/tracing system tools. This
 * function allows you to install a callback that will be invoked prior to the
 * call to exit(1).
 */
void LLVMInstallFatalErrorHandler(LLVMFatalErrorHandler Handler);

/**
 * Reset the fatal error handler. This resets LLVM's fatal error handling
 * behavior to the default.
 */
void LLVMResetFatalErrorHandler(void);

/**
 * Enable LLVM's built-in stack trace code. This intercepts the OS's crash
 * signals and prints which component of LLVM you were in at the time if the
 * crash.
 */
void LLVMEnablePrettyStackTrace(void);

/**
 * @defgroup LLVMCCoreContext Contexts
 *
 * Contexts are execution states for the core LLVM IR system.
 *
 * Most types are tied to a context instance. Multiple contexts can
 * exist simultaneously. A single context is not thread safe. However,
 * different contexts can execute on different threads simultaneously.
 *
 * @{
 */

typedef void (*LLVMDiagnosticHandler)(LLVMDiagnosticInfoRef, void *);
typedef void (*LLVMYieldCallback)(LLVMContextRef, void *);

/**
 * Create a new context.
 *
 * Every call to this function should be paired with a call to
 * LLVMContextDispose() or the context will leak memory.
 */
LLVMContextRef LLVMContextCreate(void);

/**
 * Obtain the global context instance.
 */
LLVMContextRef LLVMGetGlobalContext(void);

/**
 * Set the diagnostic handler for this context.
 */
void LLVMContextSetDiagnosticHandler(LLVMContextRef C,
                                     LLVMDiagnosticHandler Handler,
                                     void *DiagnosticContext);

/**
 * Set the yield callback function for this context.
 *
 * @see LLVMContext::setYieldCallback()
 */
void LLVMContextSetYieldCallback(LLVMContextRef C, LLVMYieldCallback Callback,
                                 void *OpaqueHandle);

/**
 * Destroy a context instance.
 *
 * This should be called for every call to LLVMContextCreate() or memory
 * will be leaked.
 */
void LLVMContextDispose(LLVMContextRef C);

/**
 * Return a string representation of the DiagnosticInfo. Use
 * LLVMDisposeMessage to free the string.
 *
 * @see DiagnosticInfo::print()
 */
char *LLVMGetDiagInfoDescription(LLVMDiagnosticInfoRef DI);

/**
 * Return an enum LLVMDiagnosticSeverity.
 *
 * @see DiagnosticInfo::getSeverity()
 */
LLVMDiagnosticSeverity LLVMGetDiagInfoSeverity(LLVMDiagnosticInfoRef DI);

unsigned LLVMGetMDKindIDInContext(LLVMContextRef C, const char* Name,
                                  unsigned SLen);
unsigned LLVMGetMDKindID(const char* Name, unsigned SLen);

/**
 * @}
 */

/**
 * @defgroup LLVMCCoreModule Modules
 *
 * Modules represent the top-level structure in an LLVM program. An LLVM
 * module is effectively a translation unit or a collection of
 * translation units merged together.
 *
 * @{
 */

/**
 * Create a new, empty module in the global context.
 *
 * This is equivalent to calling LLVMModuleCreateWithNameInContext with
 * LLVMGetGlobalContext() as the context parameter.
 *
 * Every invocation should be paired with LLVMDisposeModule() or memory
 * will be leaked.
 */
LLVMModuleRef LLVMModuleCreateWithName(const char *ModuleID);

/**
 * Create a new, empty module in a specific context.
 *
 * Every invocation should be paired with LLVMDisposeModule() or memory
 * will be leaked.
 */
LLVMModuleRef LLVMModuleCreateWithNameInContext(const char *ModuleID,
                                                LLVMContextRef C);
/**
 * Return an exact copy of the specified module.
 */
LLVMModuleRef LLVMCloneModule(LLVMModuleRef M);

/**
 * Destroy a module instance.
 *
 * This must be called for every created module or memory will be
 * leaked.
 */
void LLVMDisposeModule(LLVMModuleRef M);

/**
 * Obtain the data layout for a module.
 *
 * @see Module::getDataLayout()
 */
const char *LLVMGetDataLayout(LLVMModuleRef M);

/**
 * Set the data layout for a module.
 *
 * @see Module::setDataLayout()
 */
void LLVMSetDataLayout(LLVMModuleRef M, const char *Triple);

/**
 * Obtain the target triple for a module.
 *
 * @see Module::getTargetTriple()
 */
const char *LLVMGetTarget(LLVMModuleRef M);

/**
 * Set the target triple for a module.
 *
 * @see Module::setTargetTriple()
 */
void LLVMSetTarget(LLVMModuleRef M, const char *Triple);

/**
 * Dump a representation of a module to stderr.
 *
 * @see Module::dump()
 */
void LLVMDumpModule(LLVMModuleRef M);

/**
 * Print a representation of a module to a file. The ErrorMessage needs to be
 * disposed with LLVMDisposeMessage. Returns 0 on success, 1 otherwise.
 *
 * @see Module::print()
 */
LLVMBool LLVMPrintModuleToFile(LLVMModuleRef M, const char *Filename,
    _Out_opt_ char **ErrorMessage);

/**
 * Return a string representation of the module. Use
 * LLVMDisposeMessage to free the string.
 *
 * @see Module::print()
 */
char *LLVMPrintModuleToString(LLVMModuleRef M);

/**
 * Set inline assembly for a module.
 *
 * @see Module::setModuleInlineAsm()
 */
void LLVMSetModuleInlineAsm(LLVMModuleRef M, const char *Asm);

/**
 * Obtain the context to which this module is associated.
 *
 * @see Module::getContext()
 */
LLVMContextRef LLVMGetModuleContext(LLVMModuleRef M);

/**
 * Obtain a Type from a module by its registered name.
 */
LLVMTypeRef LLVMGetTypeByName(LLVMModuleRef M, const char *Name);

/**
 * Obtain the number of operands for named metadata in a module.
 *
 * @see llvm::Module::getNamedMetadata()
 */
unsigned LLVMGetNamedMetadataNumOperands(LLVMModuleRef M, const char* name);

/**
 * Obtain the named metadata operands for a module.
 *
 * The passed LLVMValueRef pointer should refer to an array of
 * LLVMValueRef at least LLVMGetNamedMetadataNumOperands long. This
 * array will be populated with the LLVMValueRef instances. Each
 * instance corresponds to a llvm::MDNode.
 *
 * @see llvm::Module::getNamedMetadata()
 * @see llvm::MDNode::getOperand()
 */
void LLVMGetNamedMetadataOperands(LLVMModuleRef M, const char* name, LLVMValueRef *Dest);

/**
 * Add an operand to named metadata.
 *
 * @see llvm::Module::getNamedMetadata()
 * @see llvm::MDNode::addOperand()
 */
void LLVMAddNamedMetadataOperand(LLVMModuleRef M, const char* name,
                                 LLVMValueRef Val);

/**
 * Add a function to a module under a specified name.
 *
 * @see llvm::Function::Create()
 */
LLVMValueRef LLVMAddFunction(LLVMModuleRef M, const char *Name,
                             LLVMTypeRef FunctionTy);

/**
 * Obtain a Function value from a Module by its name.
 *
 * The returned value corresponds to a llvm::Function value.
 *
 * @see llvm::Module::getFunction()
 */
LLVMValueRef LLVMGetNamedFunction(LLVMModuleRef M, const char *Name);

/**
 * Obtain an iterator to the first Function in a Module.
 *
 * @see llvm::Module::begin()
 */
LLVMValueRef LLVMGetFirstFunction(LLVMModuleRef M);

/**
 * Obtain an iterator to the last Function in a Module.
 *
 * @see llvm::Module::end()
 */
LLVMValueRef LLVMGetLastFunction(LLVMModuleRef M);

/**
 * Advance a Function iterator to the next Function.
 *
 * Returns NULL if the iterator was already at the end and there are no more
 * functions.
 */
LLVMValueRef LLVMGetNextFunction(LLVMValueRef Fn);

/**
 * Decrement a Function iterator to the previous Function.
 *
 * Returns NULL if the iterator was already at the beginning and there are
 * no previous functions.
 */
LLVMValueRef LLVMGetPreviousFunction(LLVMValueRef Fn);

/**
 * @}
 */

/**
 * @defgroup LLVMCCoreType Types
 *
 * Types represent the type of a value.
 *
 * Types are associated with a context instance. The context internally
 * deduplicates types so there is only 1 instance of a specific type
 * alive at a time. In other words, a unique type is shared among all
 * consumers within a context.
 *
 * A Type in the C API corresponds to llvm::Type.
 *
 * Types have the following hierarchy:
 *
 *   types:
 *     integer type
 *     real type
 *     function type
 *     sequence types:
 *       array type
 *       pointer type
 *       vector type
 *     void type
 *     label type
 *     opaque type
 *
 * @{
 */

/**
 * Obtain the enumerated type of a Type instance.
 *
 * @see llvm::Type:getTypeID()
 */
LLVMTypeKind LLVMGetTypeKind(LLVMTypeRef Ty);

/**
 * Whether the type has a known size.
 *
 * Things that don't have a size are abstract types, labels, and void.a
 *
 * @see llvm::Type::isSized()
 */
LLVMBool LLVMTypeIsSized(LLVMTypeRef Ty);

/**
 * Obtain the context to which this type instance is associated.
 *
 * @see llvm::Type::getContext()
 */
LLVMContextRef LLVMGetTypeContext(LLVMTypeRef Ty);

/**
 * Dump a representation of a type to stderr.
 *
 * @see llvm::Type::dump()
 */
void LLVMDumpType(LLVMTypeRef Val);

/**
 * Return a string representation of the type. Use
 * LLVMDisposeMessage to free the string.
 *
 * @see llvm::Type::print()
 */
char *LLVMPrintTypeToString(LLVMTypeRef Val);

/**
 * @defgroup LLVMCCoreTypeInt Integer Types
 *
 * Functions in this section operate on integer types.
 *
 * @{
 */

/**
 * Obtain an integer type from a context with specified bit width.
 */
LLVMTypeRef LLVMInt1TypeInContext(LLVMContextRef C);
LLVMTypeRef LLVMInt8TypeInContext(LLVMContextRef C);
LLVMTypeRef LLVMInt16TypeInContext(LLVMContextRef C);
LLVMTypeRef LLVMInt32TypeInContext(LLVMContextRef C);
LLVMTypeRef LLVMInt64TypeInContext(LLVMContextRef C);
LLVMTypeRef LLVMIntTypeInContext(LLVMContextRef C, unsigned NumBits);

/**
 * Obtain an integer type from the global context with a specified bit
 * width.
 */
LLVMTypeRef LLVMInt1Type(void);
LLVMTypeRef LLVMInt8Type(void);
LLVMTypeRef LLVMInt16Type(void);
LLVMTypeRef LLVMInt32Type(void);
LLVMTypeRef LLVMInt64Type(void);
LLVMTypeRef LLVMIntType(unsigned NumBits);
unsigned LLVMGetIntTypeWidth(LLVMTypeRef IntegerTy);

/**
 * @}
 */

/**
 * @defgroup LLVMCCoreTypeFloat Floating Point Types
 *
 * @{
 */

/**
 * Obtain a 16-bit floating point type from a context.
 */
LLVMTypeRef LLVMHalfTypeInContext(LLVMContextRef C);

/**
 * Obtain a 32-bit floating point type from a context.
 */
LLVMTypeRef LLVMFloatTypeInContext(LLVMContextRef C);

/**
 * Obtain a 64-bit floating point type from a context.
 */
LLVMTypeRef LLVMDoubleTypeInContext(LLVMContextRef C);

/**
 * Obtain a 80-bit floating point type (X87) from a context.
 */
LLVMTypeRef LLVMX86FP80TypeInContext(LLVMContextRef C);

/**
 * Obtain a 128-bit floating point type (112-bit mantissa) from a
 * context.
 */
LLVMTypeRef LLVMFP128TypeInContext(LLVMContextRef C);

/**
 * Obtain a 128-bit floating point type (two 64-bits) from a context.
 */
LLVMTypeRef LLVMPPCFP128TypeInContext(LLVMContextRef C);

/**
 * Obtain a floating point type from the global context.
 *
 * These map to the functions in this group of the same name.
 */
LLVMTypeRef LLVMHalfType(void);
LLVMTypeRef LLVMFloatType(void);
LLVMTypeRef LLVMDoubleType(void);
LLVMTypeRef LLVMX86FP80Type(void);
LLVMTypeRef LLVMFP128Type(void);
LLVMTypeRef LLVMPPCFP128Type(void);

/**
 * @}
 */

/**
 * @defgroup LLVMCCoreTypeFunction Function Types
 *
 * @{
 */

/**
 * Obtain a function type consisting of a specified signature.
 *
 * The function is defined as a tuple of a return Type, a list of
 * parameter types, and whether the function is variadic.
 */
LLVMTypeRef LLVMFunctionType(LLVMTypeRef ReturnType,
                             LLVMTypeRef *ParamTypes, unsigned ParamCount,
                             LLVMBool IsVarArg);

/**
 * Returns whether a function type is variadic.
 */
LLVMBool LLVMIsFunctionVarArg(LLVMTypeRef FunctionTy);

/**
 * Obtain the Type this function Type returns.
 */
LLVMTypeRef LLVMGetReturnType(LLVMTypeRef FunctionTy);

/**
 * Obtain the number of parameters this function accepts.
 */
unsigned LLVMCountParamTypes(LLVMTypeRef FunctionTy);

/**
 * Obtain the types of a function's parameters.
 *
 * The Dest parameter should point to a pre-allocated array of
 * LLVMTypeRef at least LLVMCountParamTypes() large. On return, the
 * first LLVMCountParamTypes() entries in the array will be populated
 * with LLVMTypeRef instances.
 *
 * @param FunctionTy The function type to operate on.
 * @param Dest Memory address of an array to be filled with result.
 */
void LLVMGetParamTypes(LLVMTypeRef FunctionTy, LLVMTypeRef *Dest);

/**
 * @}
 */

/**
 * @defgroup LLVMCCoreTypeStruct Structure Types
 *
 * These functions relate to LLVMTypeRef instances.
 *
 * @see llvm::StructType
 *
 * @{
 */

/**
 * Create a new structure type in a context.
 *
 * A structure is specified by a list of inner elements/types and
 * whether these can be packed together.
 *
 * @see llvm::StructType::create()
 */
LLVMTypeRef LLVMStructTypeInContext(LLVMContextRef C, LLVMTypeRef *ElementTypes,
                                    unsigned ElementCount, LLVMBool Packed);

/**
 * Create a new structure type in the global context.
 *
 * @see llvm::StructType::create()
 */
LLVMTypeRef LLVMStructType(LLVMTypeRef *ElementTypes, unsigned ElementCount,
                           LLVMBool Packed);

/**
 * Create an empty structure in a context having a specified name.
 *
 * @see llvm::StructType::create()
 */
LLVMTypeRef LLVMStructCreateNamed(LLVMContextRef C, const char *Name);

/**
 * Obtain the name of a structure.
 *
 * @see llvm::StructType::getName()
 */
const char *LLVMGetStructName(LLVMTypeRef Ty);

/**
 * Set the contents of a structure type.
 *
 * @see llvm::StructType::setBody()
 */
void LLVMStructSetBody(LLVMTypeRef StructTy, LLVMTypeRef *ElementTypes,
                       unsigned ElementCount, LLVMBool Packed);

/**
 * Get the number of elements defined inside the structure.
 *
 * @see llvm::StructType::getNumElements()
 */
unsigned LLVMCountStructElementTypes(LLVMTypeRef StructTy);

/**
 * Get the elements within a structure.
 *
 * The function is passed the address of a pre-allocated array of
 * LLVMTypeRef at least LLVMCountStructElementTypes() long. After
 * invocation, this array will be populated with the structure's
 * elements. The objects in the destination array will have a lifetime
 * of the structure type itself, which is the lifetime of the context it
 * is contained in.
 */
void LLVMGetStructElementTypes(LLVMTypeRef StructTy, LLVMTypeRef *Dest);

/**
 * Get the type of the element at a given index in the structure.
 *
 * @see llvm::StructType::getTypeAtIndex()
 */
LLVMTypeRef LLVMStructGetTypeAtIndex(LLVMTypeRef StructTy, unsigned i);

/**
 * Determine whether a structure is packed.
 *
 * @see llvm::StructType::isPacked()
 */
LLVMBool LLVMIsPackedStruct(LLVMTypeRef StructTy);

/**
 * Determine whether a structure is opaque.
 *
 * @see llvm::StructType::isOpaque()
 */
LLVMBool LLVMIsOpaqueStruct(LLVMTypeRef StructTy);

/**
 * @}
 */


/**
 * @defgroup LLVMCCoreTypeSequential Sequential Types
 *
 * Sequential types represents "arrays" of types. This is a super class
 * for array, vector, and pointer types.
 *
 * @{
 */

/**
 * Obtain the type of elements within a sequential type.
 *
 * This works on array, vector, and pointer types.
 *
 * @see llvm::SequentialType::getElementType()
 */
LLVMTypeRef LLVMGetElementType(LLVMTypeRef Ty);

/**
 * Create a fixed size array type that refers to a specific type.
 *
 * The created type will exist in the context that its element type
 * exists in.
 *
 * @see llvm::ArrayType::get()
 */
LLVMTypeRef LLVMArrayType(LLVMTypeRef ElementType, unsigned ElementCount);

/**
 * Obtain the length of an array type.
 *
 * This only works on types that represent arrays.
 *
 * @see llvm::ArrayType::getNumElements()
 */
unsigned LLVMGetArrayLength(LLVMTypeRef ArrayTy);

/**
 * Create a pointer type that points to a defined type.
 *
 * The created type will exist in the context that its pointee type
 * exists in.
 *
 * @see llvm::PointerType::get()
 */
LLVMTypeRef LLVMPointerType(LLVMTypeRef ElementType, unsigned AddressSpace);

/**
 * Obtain the address space of a pointer type.
 *
 * This only works on types that represent pointers.
 *
 * @see llvm::PointerType::getAddressSpace()
 */
unsigned LLVMGetPointerAddressSpace(LLVMTypeRef PointerTy);

/**
 * Create a vector type that contains a defined type and has a specific
 * number of elements.
 *
 * The created type will exist in the context thats its element type
 * exists in.
 *
 * @see llvm::VectorType::get()
 */
LLVMTypeRef LLVMVectorType(LLVMTypeRef ElementType, unsigned ElementCount);

/**
 * Obtain the number of elements in a vector type.
 *
 * This only works on types that represent vectors.
 *
 * @see llvm::VectorType::getNumElements()
 */
unsigned LLVMGetVectorSize(LLVMTypeRef VectorTy);

/**
 * @}
 */

/**
 * @defgroup LLVMCCoreTypeOther Other Types
 *
 * @{
 */

/**
 * Create a void type in a context.
 */
LLVMTypeRef LLVMVoidTypeInContext(LLVMContextRef C);

/**
 * Create a label type in a context.
 */
LLVMTypeRef LLVMLabelTypeInContext(LLVMContextRef C);

/**
 * Create a X86 MMX type in a context.
 */
LLVMTypeRef LLVMX86MMXTypeInContext(LLVMContextRef C);

/**
 * These are similar to the above functions except they operate on the
 * global context.
 */
LLVMTypeRef LLVMVoidType(void);
LLVMTypeRef LLVMLabelType(void);
LLVMTypeRef LLVMX86MMXType(void);

/**
 * @}
 */

/**
 * @}
 */

/**
 * @defgroup LLVMCCoreValues Values
 *
 * The bulk of LLVM's object model consists of values, which comprise a very
 * rich type hierarchy.
 *
 * LLVMValueRef essentially represents llvm::Value. There is a rich
 * hierarchy of classes within this type. Depending on the instance
 * obtained, not all APIs are available.
 *
 * Callers can determine the type of an LLVMValueRef by calling the
 * LLVMIsA* family of functions (e.g. LLVMIsAArgument()). These
 * functions are defined by a macro, so it isn't obvious which are
 * available by looking at the Doxygen source code. Instead, look at the
 * source definition of LLVM_FOR_EACH_VALUE_SUBCLASS and note the list
 * of value names given. These value names also correspond to classes in
 * the llvm::Value hierarchy.
 *
 * @{
 */

#define LLVM_FOR_EACH_VALUE_SUBCLASS(macro) \
  macro(Argument)                           \
  macro(BasicBlock)                         \
  macro(InlineAsm)                          \
  macro(User)                               \
    macro(Constant)                         \
      macro(BlockAddress)                   \
      macro(ConstantAggregateZero)          \
      macro(ConstantArray)                  \
      macro(ConstantDataSequential)         \
        macro(ConstantDataArray)            \
        macro(ConstantDataVector)           \
      macro(ConstantExpr)                   \
      macro(ConstantFP)                     \
      macro(ConstantInt)                    \
      macro(ConstantPointerNull)            \
      macro(ConstantStruct)                 \
      macro(ConstantVector)                 \
      macro(GlobalValue)                    \
        macro(GlobalAlias)                  \
        macro(GlobalObject)                 \
          macro(Function)                   \
          macro(GlobalVariable)             \
      macro(UndefValue)                     \
    macro(Instruction)                      \
      macro(BinaryOperator)                 \
      macro(CallInst)                       \
        macro(IntrinsicInst)                \
          macro(DbgInfoIntrinsic)           \
            macro(DbgDeclareInst)           \
          macro(MemIntrinsic)               \
            macro(MemCpyInst)               \
            macro(MemMoveInst)              \
            macro(MemSetInst)               \
      macro(CmpInst)                        \
        macro(FCmpInst)                     \
        macro(ICmpInst)                     \
      macro(ExtractElementInst)             \
      macro(GetElementPtrInst)              \
      macro(InsertElementInst)              \
      macro(InsertValueInst)                \
      macro(LandingPadInst)                 \
      macro(PHINode)                        \
      macro(SelectInst)                     \
      macro(ShuffleVectorInst)              \
      macro(StoreInst)                      \
      macro(TerminatorInst)                 \
        macro(BranchInst)                   \
        macro(IndirectBrInst)               \
        macro(InvokeInst)                   \
        macro(ReturnInst)                   \
        macro(SwitchInst)                   \
        macro(UnreachableInst)              \
        macro(ResumeInst)                   \
      macro(UnaryInstruction)               \
        macro(AllocaInst)                   \
        macro(CastInst)                     \
          macro(AddrSpaceCastInst)          \
          macro(BitCastInst)                \
          macro(FPExtInst)                  \
          macro(FPToSIInst)                 \
          macro(FPToUIInst)                 \
          macro(FPTruncInst)                \
          macro(IntToPtrInst)               \
          macro(PtrToIntInst)               \
          macro(SExtInst)                   \
          macro(SIToFPInst)                 \
          macro(TruncInst)                  \
          macro(UIToFPInst)                 \
          macro(ZExtInst)                   \
        macro(ExtractValueInst)             \
        macro(LoadInst)                     \
        macro(VAArgInst)

/**
 * @defgroup LLVMCCoreValueGeneral General APIs
 *
 * Functions in this section work on all LLVMValueRef instances,
 * regardless of their sub-type. They correspond to functions available
 * on llvm::Value.
 *
 * @{
 */

/**
 * Obtain the type of a value.
 *
 * @see llvm::Value::getType()
 */
LLVMTypeRef LLVMTypeOf(LLVMValueRef Val);

/**
 * Obtain the string name of a value.
 *
 * @see llvm::Value::getName()
 */
const char *LLVMGetValueName(LLVMValueRef Val);

/**
 * Set the string name of a value.
 *
 * @see llvm::Value::setName()
 */
void LLVMSetValueName(LLVMValueRef Val, const char *Name);

/**
 * Dump a representation of a value to stderr.
 *
 * @see llvm::Value::dump()
 */
void LLVMDumpValue(LLVMValueRef Val);

/**
 * Return a string representation of the value. Use
 * LLVMDisposeMessage to free the string.
 *
 * @see llvm::Value::print()
 */
char *LLVMPrintValueToString(LLVMValueRef Val);

/**
 * Replace all uses of a value with another one.
 *
 * @see llvm::Value::replaceAllUsesWith()
 */
void LLVMReplaceAllUsesWith(LLVMValueRef OldVal, LLVMValueRef NewVal);

/**
 * Determine whether the specified constant instance is constant.
 */
LLVMBool LLVMIsConstant(LLVMValueRef Val);

/**
 * Determine whether a value instance is undefined.
 */
LLVMBool LLVMIsUndef(LLVMValueRef Val);

/**
 * Convert value instances between types.
 *
 * Internally, an LLVMValueRef is "pinned" to a specific type. This
 * series of functions allows you to cast an instance to a specific
 * type.
 *
 * If the cast is not valid for the specified type, NULL is returned.
 *
 * @see llvm::dyn_cast_or_null<>
 */
#define LLVM_DECLARE_VALUE_CAST(name) \
  LLVMValueRef LLVMIsA##name(LLVMValueRef Val);
LLVM_FOR_EACH_VALUE_SUBCLASS(LLVM_DECLARE_VALUE_CAST)

LLVMValueRef LLVMIsAMDNode(LLVMValueRef Val);
LLVMValueRef LLVMIsAMDString(LLVMValueRef Val);

/**
 * @}
 */

/**
 * @defgroup LLVMCCoreValueUses Usage
 *
 * This module defines functions that allow you to inspect the uses of a
 * LLVMValueRef.
 *
 * It is possible to obtain an LLVMUseRef for any LLVMValueRef instance.
 * Each LLVMUseRef (which corresponds to a llvm::Use instance) holds a
 * llvm::User and llvm::Value.
 *
 * @{
 */

/**
 * Obtain the first use of a value.
 *
 * Uses are obtained in an iterator fashion. First, call this function
 * to obtain a reference to the first use. Then, call LLVMGetNextUse()
 * on that instance and all subsequently obtained instances until
 * LLVMGetNextUse() returns NULL.
 *
 * @see llvm::Value::use_begin()
 */
LLVMUseRef LLVMGetFirstUse(LLVMValueRef Val);

/**
 * Obtain the next use of a value.
 *
 * This effectively advances the iterator. It returns NULL if you are on
 * the final use and no more are available.
 */
LLVMUseRef LLVMGetNextUse(LLVMUseRef U);

/**
 * Obtain the user value for a user.
 *
 * The returned value corresponds to a llvm::User type.
 *
 * @see llvm::Use::getUser()
 */
LLVMValueRef LLVMGetUser(LLVMUseRef U);

/**
 * Obtain the value this use corresponds to.
 *
 * @see llvm::Use::get().
 */
LLVMValueRef LLVMGetUsedValue(LLVMUseRef U);

/**
 * @}
 */

/**
 * @defgroup LLVMCCoreValueUser User value
 *
 * Function in this group pertain to LLVMValueRef instances that descent
 * from llvm::User. This includes constants, instructions, and
 * operators.
 *
 * @{
 */

/**
 * Obtain an operand at a specific index in a llvm::User value.
 *
 * @see llvm::User::getOperand()
 */
LLVMValueRef LLVMGetOperand(LLVMValueRef Val, unsigned Index);

/**
 * Obtain the use of an operand at a specific index in a llvm::User value.
 *
 * @see llvm::User::getOperandUse()
 */
LLVMUseRef LLVMGetOperandUse(LLVMValueRef Val, unsigned Index);

/**
 * Set an operand at a specific index in a llvm::User value.
 *
 * @see llvm::User::setOperand()
 */
void LLVMSetOperand(LLVMValueRef User, unsigned Index, LLVMValueRef Val);

/**
 * Obtain the number of operands in a llvm::User value.
 *
 * @see llvm::User::getNumOperands()
 */
int LLVMGetNumOperands(LLVMValueRef Val);

/**
 * @}
 */

/**
 * @defgroup LLVMCCoreValueConstant Constants
 *
 * This section contains APIs for interacting with LLVMValueRef that
 * correspond to llvm::Constant instances.
 *
 * These functions will work for any LLVMValueRef in the llvm::Constant
 * class hierarchy.
 *
 * @{
 */

/**
 * Obtain a constant value referring to the null instance of a type.
 *
 * @see llvm::Constant::getNullValue()
 */
LLVMValueRef LLVMConstNull(LLVMTypeRef Ty); /* all zeroes */

/**
 * Obtain a constant value referring to the instance of a type
 * consisting of all ones.
 *
 * This is only valid for integer types.
 *
 * @see llvm::Constant::getAllOnesValue()
 */
LLVMValueRef LLVMConstAllOnes(LLVMTypeRef Ty);

/**
 * Obtain a constant value referring to an undefined value of a type.
 *
 * @see llvm::UndefValue::get()
 */
LLVMValueRef LLVMGetUndef(LLVMTypeRef Ty);

/**
 * Determine whether a value instance is null.
 *
 * @see llvm::Constant::isNullValue()
 */
LLVMBool LLVMIsNull(LLVMValueRef Val);

/**
 * Obtain a constant that is a constant pointer pointing to NULL for a
 * specified type.
 */
LLVMValueRef LLVMConstPointerNull(LLVMTypeRef Ty);

/**
 * @defgroup LLVMCCoreValueConstantScalar Scalar constants
 *
 * Functions in this group model LLVMValueRef instances that correspond
 * to constants referring to scalar types.
 *
 * For integer types, the LLVMTypeRef parameter should correspond to a
 * llvm::IntegerType instance and the returned LLVMValueRef will
 * correspond to a llvm::ConstantInt.
 *
 * For floating point types, the LLVMTypeRef returned corresponds to a
 * llvm::ConstantFP.
 *
 * @{
 */

/**
 * Obtain a constant value for an integer type.
 *
 * The returned value corresponds to a llvm::ConstantInt.
 *
 * @see llvm::ConstantInt::get()
 *
 * @param IntTy Integer type to obtain value of.
 * @param N The value the returned instance should refer to.
 * @param SignExtend Whether to sign extend the produced value.
 */
LLVMValueRef LLVMConstInt(LLVMTypeRef IntTy, unsigned long long N,
                          LLVMBool SignExtend);

/**
 * Obtain a constant value for an integer of arbitrary precision.
 *
 * @see llvm::ConstantInt::get()
 */
LLVMValueRef LLVMConstIntOfArbitraryPrecision(LLVMTypeRef IntTy,
                                              unsigned NumWords,
                                              const uint64_t Words[]);

/**
 * Obtain a constant value for an integer parsed from a string.
 *
 * A similar API, LLVMConstIntOfStringAndSize is also available. If the
 * string's length is available, it is preferred to call that function
 * instead.
 *
 * @see llvm::ConstantInt::get()
 */
LLVMValueRef LLVMConstIntOfString(LLVMTypeRef IntTy, const char *Text,
                                  uint8_t Radix);

/**
 * Obtain a constant value for an integer parsed from a string with
 * specified length.
 *
 * @see llvm::ConstantInt::get()
 */
LLVMValueRef LLVMConstIntOfStringAndSize(LLVMTypeRef IntTy, const char *Text,
                                         unsigned SLen, uint8_t Radix);

/**
 * Obtain a constant value referring to a double floating point value.
 */
LLVMValueRef LLVMConstReal(LLVMTypeRef RealTy, double N);

/**
 * Obtain a constant for a floating point value parsed from a string.
 *
 * A similar API, LLVMConstRealOfStringAndSize is also available. It
 * should be used if the input string's length is known.
 */
LLVMValueRef LLVMConstRealOfString(LLVMTypeRef RealTy, const char *Text);

/**
 * Obtain a constant for a floating point value parsed from a string.
 */
LLVMValueRef LLVMConstRealOfStringAndSize(LLVMTypeRef RealTy, const char *Text,
                                          unsigned SLen);

/**
 * Obtain the zero extended value for an integer constant value.
 *
 * @see llvm::ConstantInt::getZExtValue()
 */
unsigned long long LLVMConstIntGetZExtValue(LLVMValueRef ConstantVal);

/**
 * Obtain the sign extended value for an integer constant value.
 *
 * @see llvm::ConstantInt::getSExtValue()
 */
long long LLVMConstIntGetSExtValue(LLVMValueRef ConstantVal);

/**
 * Obtain the double value for an floating point constant value.
 * losesInfo indicates if some precision was lost in the conversion.
 *
 * @see llvm::ConstantFP::getDoubleValue
 */
double LLVMConstRealGetDouble(LLVMValueRef ConstantVal, LLVMBool *losesInfo);

/**
 * @}
 */

/**
 * @defgroup LLVMCCoreValueConstantComposite Composite Constants
 *
 * Functions in this group operate on composite constants.
 *
 * @{
 */

/**
 * Create a ConstantDataSequential and initialize it with a string.
 *
 * @see llvm::ConstantDataArray::getString()
 */
LLVMValueRef LLVMConstStringInContext(LLVMContextRef C, const char *Str,
                                      unsigned Length, LLVMBool DontNullTerminate);

/**
 * Create a ConstantDataSequential with string content in the global context.
 *
 * This is the same as LLVMConstStringInContext except it operates on the
 * global context.
 *
 * @see LLVMConstStringInContext()
 * @see llvm::ConstantDataArray::getString()
 */
LLVMValueRef LLVMConstString(const char *Str, unsigned Length,
                             LLVMBool DontNullTerminate);

/**
 * Returns true if the specified constant is an array of i8.
 *
 * @see ConstantDataSequential::getAsString()
 */
LLVMBool LLVMIsConstantString(LLVMValueRef c);

/**
 * Get the given constant data sequential as a string.
 *
 * @see ConstantDataSequential::getAsString()
 */
const char *LLVMGetAsString(LLVMValueRef c, size_t* out);

/**
 * Create an anonymous ConstantStruct with the specified values.
 *
 * @see llvm::ConstantStruct::getAnon()
 */
LLVMValueRef LLVMConstStructInContext(LLVMContextRef C,
                                      LLVMValueRef *ConstantVals,
                                      unsigned Count, LLVMBool Packed);

/**
 * Create a ConstantStruct in the global Context.
 *
 * This is the same as LLVMConstStructInContext except it operates on the
 * global Context.
 *
 * @see LLVMConstStructInContext()
 */
LLVMValueRef LLVMConstStruct(LLVMValueRef *ConstantVals, unsigned Count,
                             LLVMBool Packed);

/**
 * Create a ConstantArray from values.
 *
 * @see llvm::ConstantArray::get()
 */
LLVMValueRef LLVMConstArray(LLVMTypeRef ElementTy,
                            LLVMValueRef *ConstantVals, unsigned Length);

/**
 * Create a non-anonymous ConstantStruct from values.
 *
 * @see llvm::ConstantStruct::get()
 */
LLVMValueRef LLVMConstNamedStruct(LLVMTypeRef StructTy,
                                  LLVMValueRef *ConstantVals,
                                  unsigned Count);

/**
 * Get an element at specified index as a constant.
 *
 * @see ConstantDataSequential::getElementAsConstant()
 */
LLVMValueRef LLVMGetElementAsConstant(LLVMValueRef c, unsigned idx);

/**
 * Create a ConstantVector from values.
 *
 * @see llvm::ConstantVector::get()
 */
LLVMValueRef LLVMConstVector(LLVMValueRef *ScalarConstantVals, unsigned Size);

/**
 * @}
 */

/**
 * @defgroup LLVMCCoreValueConstantExpressions Constant Expressions
 *
 * Functions in this group correspond to APIs on llvm::ConstantExpr.
 *
 * @see llvm::ConstantExpr.
 *
 * @{
 */
LLVMOpcode LLVMGetConstOpcode(LLVMValueRef ConstantVal);
LLVMValueRef LLVMAlignOf(LLVMTypeRef Ty);
LLVMValueRef LLVMSizeOf(LLVMTypeRef Ty);
LLVMValueRef LLVMConstNeg(LLVMValueRef ConstantVal);
LLVMValueRef LLVMConstNSWNeg(LLVMValueRef ConstantVal);
LLVMValueRef LLVMConstNUWNeg(LLVMValueRef ConstantVal);
LLVMValueRef LLVMConstFNeg(LLVMValueRef ConstantVal);
LLVMValueRef LLVMConstNot(LLVMValueRef ConstantVal);
LLVMValueRef LLVMConstAdd(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstNSWAdd(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstNUWAdd(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstFAdd(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstSub(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstNSWSub(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstNUWSub(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstFSub(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstMul(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstNSWMul(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstNUWMul(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstFMul(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstUDiv(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstSDiv(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstExactSDiv(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstFDiv(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstURem(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstSRem(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstFRem(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstAnd(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstOr(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstXor(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstICmp(LLVMIntPredicate Predicate,
                           LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstFCmp(LLVMRealPredicate Predicate,
                           LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstShl(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstLShr(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstAShr(LLVMValueRef LHSConstant, LLVMValueRef RHSConstant);
LLVMValueRef LLVMConstGEP(LLVMValueRef ConstantVal,
                          LLVMValueRef *ConstantIndices, unsigned NumIndices);
LLVMValueRef LLVMConstInBoundsGEP(LLVMValueRef ConstantVal,
                                  LLVMValueRef *ConstantIndices,
                                  unsigned NumIndices);
LLVMValueRef LLVMConstTrunc(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstSExt(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstZExt(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstFPTrunc(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstFPExt(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstUIToFP(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstSIToFP(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstFPToUI(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstFPToSI(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstPtrToInt(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstIntToPtr(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstBitCast(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstAddrSpaceCast(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstZExtOrBitCast(LLVMValueRef ConstantVal,
                                    LLVMTypeRef ToType);
LLVMValueRef LLVMConstSExtOrBitCast(LLVMValueRef ConstantVal,
                                    LLVMTypeRef ToType);
LLVMValueRef LLVMConstTruncOrBitCast(LLVMValueRef ConstantVal,
                                     LLVMTypeRef ToType);
LLVMValueRef LLVMConstPointerCast(LLVMValueRef ConstantVal,
                                  LLVMTypeRef ToType);
LLVMValueRef LLVMConstIntCast(LLVMValueRef ConstantVal, LLVMTypeRef ToType,
                              LLVMBool isSigned);
LLVMValueRef LLVMConstFPCast(LLVMValueRef ConstantVal, LLVMTypeRef ToType);
LLVMValueRef LLVMConstSelect(LLVMValueRef ConstantCondition,
                             LLVMValueRef ConstantIfTrue,
                             LLVMValueRef ConstantIfFalse);
LLVMValueRef LLVMConstExtractElement(LLVMValueRef VectorConstant,
                                     LLVMValueRef IndexConstant);
LLVMValueRef LLVMConstInsertElement(LLVMValueRef VectorConstant,
                                    LLVMValueRef ElementValueConstant,
                                    LLVMValueRef IndexConstant);
LLVMValueRef LLVMConstShuffleVector(LLVMValueRef VectorAConstant,
                                    LLVMValueRef VectorBConstant,
                                    LLVMValueRef MaskConstant);
LLVMValueRef LLVMConstExtractValue(LLVMValueRef AggConstant, unsigned *IdxList,
                                   unsigned NumIdx);
LLVMValueRef LLVMConstInsertValue(LLVMValueRef AggConstant,
                                  LLVMValueRef ElementValueConstant,
                                  unsigned *IdxList, unsigned NumIdx);
LLVMValueRef LLVMConstInlineAsm(LLVMTypeRef Ty,
                                const char *AsmString, const char *Constraints,
                                LLVMBool HasSideEffects, LLVMBool IsAlignStack);
LLVMValueRef LLVMBlockAddress(LLVMValueRef F, LLVMBasicBlockRef BB);

/**
 * @}
 */

/**
 * @defgroup LLVMCCoreValueConstantGlobals Global Values
 *
 * This group contains functions that operate on global values. Functions in
 * this group relate to functions in the llvm::GlobalValue class tree.
 *
 * @see llvm::GlobalValue
 *
 * @{
 */

LLVMModuleRef LLVMGetGlobalParent(LLVMValueRef Global);
LLVMBool LLVMIsDeclaration(LLVMValueRef Global);
LLVMLinkage LLVMGetLinkage(LLVMValueRef Global);
void LLVMSetLinkage(LLVMValueRef Global, LLVMLinkage Linkage);
const char *LLVMGetSection(LLVMValueRef Global);
void LLVMSetSection(LLVMValueRef Global, const char *Section);
LLVMVisibility LLVMGetVisibility(LLVMValueRef Global);
void LLVMSetVisibility(LLVMValueRef Global, LLVMVisibility Viz);
LLVMDLLStorageClass LLVMGetDLLStorageClass(LLVMValueRef Global);
void LLVMSetDLLStorageClass(LLVMValueRef Global, LLVMDLLStorageClass Class);
LLVMBool LLVMHasUnnamedAddr(LLVMValueRef Global);
void LLVMSetUnnamedAddr(LLVMValueRef Global, LLVMBool HasUnnamedAddr);

/**
 * @defgroup LLVMCCoreValueWithAlignment Values with alignment
 *
 * Functions in this group only apply to values with alignment, i.e.
 * global variables, load and store instructions.
 */

/**
 * Obtain the preferred alignment of the value.
 * @see llvm::AllocaInst::getAlignment()
 * @see llvm::LoadInst::getAlignment()
 * @see llvm::StoreInst::getAlignment()
 * @see llvm::GlobalValue::getAlignment()
 */
unsigned LLVMGetAlignment(LLVMValueRef V);

/**
 * Set the preferred alignment of the value.
 * @see llvm::AllocaInst::setAlignment()
 * @see llvm::LoadInst::setAlignment()
 * @see llvm::StoreInst::setAlignment()
 * @see llvm::GlobalValue::setAlignment()
 */
void LLVMSetAlignment(LLVMValueRef V, unsigned Bytes);

/**
  * @}
  */

/**
 * @defgroup LLVMCoreValueConstantGlobalVariable Global Variables
 *
 * This group contains functions that operate on global variable values.
 *
 * @see llvm::GlobalVariable
 *
 * @{
 */
LLVMValueRef LLVMAddGlobal(LLVMModuleRef M, LLVMTypeRef Ty, const char *Name);
LLVMValueRef LLVMAddGlobalInAddressSpace(LLVMModuleRef M, LLVMTypeRef Ty,
                                         const char *Name,
                                         unsigned AddressSpace);
LLVMValueRef LLVMGetNamedGlobal(LLVMModuleRef M, const char *Name);
LLVMValueRef LLVMGetFirstGlobal(LLVMModuleRef M);
LLVMValueRef LLVMGetLastGlobal(LLVMModuleRef M);
LLVMValueRef LLVMGetNextGlobal(LLVMValueRef GlobalVar);
LLVMValueRef LLVMGetPreviousGlobal(LLVMValueRef GlobalVar);
void LLVMDeleteGlobal(LLVMValueRef GlobalVar);
LLVMValueRef LLVMGetInitializer(LLVMValueRef GlobalVar);
void LLVMSetInitializer(LLVMValueRef GlobalVar, LLVMValueRef ConstantVal);
LLVMBool LLVMIsThreadLocal(LLVMValueRef GlobalVar);
void LLVMSetThreadLocal(LLVMValueRef GlobalVar, LLVMBool IsThreadLocal);
LLVMBool LLVMIsGlobalConstant(LLVMValueRef GlobalVar);
void LLVMSetGlobalConstant(LLVMValueRef GlobalVar, LLVMBool IsConstant);
LLVMThreadLocalMode LLVMGetThreadLocalMode(LLVMValueRef GlobalVar);
void LLVMSetThreadLocalMode(LLVMValueRef GlobalVar, LLVMThreadLocalMode Mode);
LLVMBool LLVMIsExternallyInitialized(LLVMValueRef GlobalVar);
void LLVMSetExternallyInitialized(LLVMValueRef GlobalVar, LLVMBool IsExtInit);

/**
 * @}
 */

/**
 * @defgroup LLVMCoreValueConstantGlobalAlias Global Aliases
 *
 * This group contains function that operate on global alias values.
 *
 * @see llvm::GlobalAlias
 *
 * @{
 */
LLVMValueRef LLVMAddAlias(LLVMModuleRef M, LLVMTypeRef Ty, LLVMValueRef Aliasee,
                          const char *Name);

/**
 * @}
 */

/**
 * @defgroup LLVMCCoreValueFunction Function values
 *
 * Functions in this group operate on LLVMValueRef instances that
 * correspond to llvm::Function instances.
 *
 * @see llvm::Function
 *
 * @{
 */

/**
 * Remove a function from its containing module and deletes it.
 *
 * @see llvm::Function::eraseFromParent()
 */
void LLVMDeleteFunction(LLVMValueRef Fn);

/**
 * Obtain the personality function attached to the function.
 *
 * @see llvm::Function::getPersonalityFn()
 */
LLVMValueRef LLVMGetPersonalityFn(LLVMValueRef Fn);

/**
 * Set the personality function attached to the function.
 *
 * @see llvm::Function::setPersonalityFn()
 */
void LLVMSetPersonalityFn(LLVMValueRef Fn, LLVMValueRef PersonalityFn);

/**
 * Obtain the ID number from a function instance.
 *
 * @see llvm::Function::getIntrinsicID()
 */
unsigned LLVMGetIntrinsicID(LLVMValueRef Fn);

/**
 * Obtain the calling function of a function.
 *
 * The returned value corresponds to the LLVMCallConv enumeration.
 *
 * @see llvm::Function::getCallingConv()
 */
unsigned LLVMGetFunctionCallConv(LLVMValueRef Fn);

/**
 * Set the calling convention of a function.
 *
 * @see llvm::Function::setCallingConv()
 *
 * @param Fn Function to operate on
 * @param CC LLVMCallConv to set calling convention to
 */
void LLVMSetFunctionCallConv(LLVMValueRef Fn, unsigned CC);

/**
 * Obtain the name of the garbage collector to use during code
 * generation.
 *
 * @see llvm::Function::getGC()
 */
const char *LLVMGetGC(LLVMValueRef Fn);

/**
 * Define the garbage collector to use during code generation.
 *
 * @see llvm::Function::setGC()
 */
void LLVMSetGC(LLVMValueRef Fn, const char *Name);

/**
 * Add an attribute to a function.
 *
 * @see llvm::Function::addAttribute()
 */
void LLVMAddFunctionAttr(LLVMValueRef Fn, LLVMAttribute PA);

/**
 * Add a target-dependent attribute to a fuction
 * @see llvm::AttrBuilder::addAttribute()
 */
void LLVMAddTargetDependentFunctionAttr(LLVMValueRef Fn, const char *A,
                                        const char *V);

/**
 * Obtain an attribute from a function.
 *
 * @see llvm::Function::getAttributes()
 */
LLVMAttribute LLVMGetFunctionAttr(LLVMValueRef Fn);

/**
 * Remove an attribute from a function.
 */
void LLVMRemoveFunctionAttr(LLVMValueRef Fn, LLVMAttribute PA);

/**
 * @defgroup LLVMCCoreValueFunctionParameters Function Parameters
 *
 * Functions in this group relate to arguments/parameters on functions.
 *
 * Functions in this group expect LLVMValueRef instances that correspond
 * to llvm::Function instances.
 *
 * @{
 */

/**
 * Obtain the number of parameters in a function.
 *
 * @see llvm::Function::arg_size()
 */
unsigned LLVMCountParams(LLVMValueRef Fn);

/**
 * Obtain the parameters in a function.
 *
 * The takes a pointer to a pre-allocated array of LLVMValueRef that is
 * at least LLVMCountParams() long. This array will be filled with
 * LLVMValueRef instances which correspond to the parameters the
 * function receives. Each LLVMValueRef corresponds to a llvm::Argument
 * instance.
 *
 * @see llvm::Function::arg_begin()
 */
void LLVMGetParams(LLVMValueRef Fn, LLVMValueRef *Params);

/**
 * Obtain the parameter at the specified index.
 *
 * Parameters are indexed from 0.
 *
 * @see llvm::Function::arg_begin()
 */
LLVMValueRef LLVMGetParam(LLVMValueRef Fn, unsigned Index);

/**
 * Obtain the function to which this argument belongs.
 *
 * Unlike other functions in this group, this one takes an LLVMValueRef
 * that corresponds to a llvm::Attribute.
 *
 * The returned LLVMValueRef is the llvm::Function to which this
 * argument belongs.
 */
LLVMValueRef LLVMGetParamParent(LLVMValueRef Inst);

/**
 * Obtain the first parameter to a function.
 *
 * @see llvm::Function::arg_begin()
 */
LLVMValueRef LLVMGetFirstParam(LLVMValueRef Fn);

/**
 * Obtain the last parameter to a function.
 *
 * @see llvm::Function::arg_end()
 */
LLVMValueRef LLVMGetLastParam(LLVMValueRef Fn);

/**
 * Obtain the next parameter to a function.
 *
 * This takes an LLVMValueRef obtained from LLVMGetFirstParam() (which is
 * actually a wrapped iterator) and obtains the next parameter from the
 * underlying iterator.
 */
LLVMValueRef LLVMGetNextParam(LLVMValueRef Arg);

/**
 * Obtain the previous parameter to a function.
 *
 * This is the opposite of LLVMGetNextParam().
 */
LLVMValueRef LLVMGetPreviousParam(LLVMValueRef Arg);

/**
 * Add an attribute to a function argument.
 *
 * @see llvm::Argument::addAttr()
 */
void LLVMAddAttribute(LLVMValueRef Arg, LLVMAttribute PA);

/**
 * Remove an attribute from a function argument.
 *
 * @see llvm::Argument::removeAttr()
 */
void LLVMRemoveAttribute(LLVMValueRef Arg, LLVMAttribute PA);

/**
 * Get an attribute from a function argument.
 */
LLVMAttribute LLVMGetAttribute(LLVMValueRef Arg);

/**
 * Set the alignment for a function parameter.
 *
 * @see llvm::Argument::addAttr()
 * @see llvm::AttrBuilder::addAlignmentAttr()
 */
void LLVMSetParamAlignment(LLVMValueRef Arg, unsigned align);

/**
 * @}
 */

/**
 * @}
 */

/**
 * @}
 */

/**
 * @}
 */

/**
 * @defgroup LLVMCCoreValueMetadata Metadata
 *
 * @{
 */

/**
 * Obtain a MDString value from a context.
 *
 * The returned instance corresponds to the llvm::MDString class.
 *
 * The instance is specified by string data of a specified length. The
 * string content is copied, so the backing memory can be freed after
 * this function returns.
 */
LLVMValueRef LLVMMDStringInContext(LLVMContextRef C, const char *Str,
                                   unsigned SLen);

/**
 * Obtain a MDString value from the global context.
 */
LLVMValueRef LLVMMDString(const char *Str, unsigned SLen);

/**
 * Obtain a MDNode value from a context.
 *
 * The returned value corresponds to the llvm::MDNode class.
 */
LLVMValueRef LLVMMDNodeInContext(LLVMContextRef C, LLVMValueRef *Vals,
                                 unsigned Count);

/**
 * Obtain a MDNode value from the global context.
 */
LLVMValueRef LLVMMDNode(LLVMValueRef *Vals, unsigned Count);

/**
 * Obtain the underlying string from a MDString value.
 *
 * @param V Instance to obtain string from.
 * @param Len Memory address which will hold length of returned string.
 * @return String data in MDString.
 */
const char  *LLVMGetMDString(LLVMValueRef V, unsigned* Len);

/**
 * Obtain the number of operands from an MDNode value.
 *
 * @param V MDNode to get number of operands from.
 * @return Number of operands of the MDNode.
 */
unsigned LLVMGetMDNodeNumOperands(LLVMValueRef V);

/**
 * Obtain the given MDNode's operands.
 *
 * The passed LLVMValueRef pointer should point to enough memory to hold all of
 * the operands of the given MDNode (see LLVMGetMDNodeNumOperands) as
 * LLVMValueRefs. This memory will be populated with the LLVMValueRefs of the
 * MDNode's operands.
 *
 * @param V MDNode to get the operands from.
 * @param Dest Destination array for operands.
 */
void LLVMGetMDNodeOperands(LLVMValueRef V, LLVMValueRef *Dest);

/**
 * @}
 */

/**
 * @defgroup LLVMCCoreValueBasicBlock Basic Block
 *
 * A basic block represents a single entry single exit section of code.
 * Basic blocks contain a list of instructions which form the body of
 * the block.
 *
 * Basic blocks belong to functions. They have the type of label.
 *
 * Basic blocks are themselves values. However, the C API models them as
 * LLVMBasicBlockRef.
 *
 * @see llvm::BasicBlock
 *
 * @{
 */

/**
 * Convert a basic block instance to a value type.
 */
LLVMValueRef LLVMBasicBlockAsValue(LLVMBasicBlockRef BB);

/**
 * Determine whether an LLVMValueRef is itself a basic block.
 */
LLVMBool LLVMValueIsBasicBlock(LLVMValueRef Val);

/**
 * Convert an LLVMValueRef to an LLVMBasicBlockRef instance.
 */
LLVMBasicBlockRef LLVMValueAsBasicBlock(LLVMValueRef Val);

/**
 * Obtain the function to which a basic block belongs.
 *
 * @see llvm::BasicBlock::getParent()
 */
LLVMValueRef LLVMGetBasicBlockParent(LLVMBasicBlockRef BB);

/**
 * Obtain the terminator instruction for a basic block.
 *
 * If the basic block does not have a terminator (it is not well-formed
 * if it doesn't), then NULL is returned.
 *
 * The returned LLVMValueRef corresponds to a llvm::TerminatorInst.
 *
 * @see llvm::BasicBlock::getTerminator()
 */
LLVMValueRef LLVMGetBasicBlockTerminator(LLVMBasicBlockRef BB);

/**
 * Obtain the number of basic blocks in a function.
 *
 * @param Fn Function value to operate on.
 */
unsigned LLVMCountBasicBlocks(LLVMValueRef Fn);

/**
 * Obtain all of the basic blocks in a function.
 *
 * This operates on a function value. The BasicBlocks parameter is a
 * pointer to a pre-allocated array of LLVMBasicBlockRef of at least
 * LLVMCountBasicBlocks() in length. This array is populated with
 * LLVMBasicBlockRef instances.
 */
void LLVMGetBasicBlocks(LLVMValueRef Fn, LLVMBasicBlockRef *BasicBlocks);

/**
 * Obtain the first basic block in a function.
 *
 * The returned basic block can be used as an iterator. You will likely
 * eventually call into LLVMGetNextBasicBlock() with it.
 *
 * @see llvm::Function::begin()
 */
LLVMBasicBlockRef LLVMGetFirstBasicBlock(LLVMValueRef Fn);

/**
 * Obtain the last basic block in a function.
 *
 * @see llvm::Function::end()
 */
LLVMBasicBlockRef LLVMGetLastBasicBlock(LLVMValueRef Fn);

/**
 * Advance a basic block iterator.
 */
LLVMBasicBlockRef LLVMGetNextBasicBlock(LLVMBasicBlockRef BB);

/**
 * Go backwards in a basic block iterator.
 */
LLVMBasicBlockRef LLVMGetPreviousBasicBlock(LLVMBasicBlockRef BB);

/**
 * Obtain the basic block that corresponds to the entry point of a
 * function.
 *
 * @see llvm::Function::getEntryBlock()
 */
LLVMBasicBlockRef LLVMGetEntryBasicBlock(LLVMValueRef Fn);

/**
 * Append a basic block to the end of a function.
 *
 * @see llvm::BasicBlock::Create()
 */
LLVMBasicBlockRef LLVMAppendBasicBlockInContext(LLVMContextRef C,
                                                LLVMValueRef Fn,
                                                const char *Name);

/**
 * Append a basic block to the end of a function using the global
 * context.
 *
 * @see llvm::BasicBlock::Create()
 */
LLVMBasicBlockRef LLVMAppendBasicBlock(LLVMValueRef Fn, const char *Name);

/**
 * Insert a basic block in a function before another basic block.
 *
 * The function to add to is determined by the function of the
 * passed basic block.
 *
 * @see llvm::BasicBlock::Create()
 */
LLVMBasicBlockRef LLVMInsertBasicBlockInContext(LLVMContextRef C,
                                                LLVMBasicBlockRef BB,
                                                const char *Name);

/**
 * Insert a basic block in a function using the global context.
 *
 * @see llvm::BasicBlock::Create()
 */
LLVMBasicBlockRef LLVMInsertBasicBlock(LLVMBasicBlockRef InsertBeforeBB,
                                       const char *Name);

/**
 * Remove a basic block from a function and delete it.
 *
 * This deletes the basic block from its containing function and deletes
 * the basic block itself.
 *
 * @see llvm::BasicBlock::eraseFromParent()
 */
void LLVMDeleteBasicBlock(LLVMBasicBlockRef BB);

/**
 * Remove a basic block from a function.
 *
 * This deletes the basic block from its containing function but keep
 * the basic block alive.
 *
 * @see llvm::BasicBlock::removeFromParent()
 */
void LLVMRemoveBasicBlockFromParent(LLVMBasicBlockRef BB);

/**
 * Move a basic block to before another one.
 *
 * @see llvm::BasicBlock::moveBefore()
 */
void LLVMMoveBasicBlockBefore(LLVMBasicBlockRef BB, LLVMBasicBlockRef MovePos);

/**
 * Move a basic block to after another one.
 *
 * @see llvm::BasicBlock::moveAfter()
 */
void LLVMMoveBasicBlockAfter(LLVMBasicBlockRef BB, LLVMBasicBlockRef MovePos);

/**
 * Obtain the first instruction in a basic block.
 *
 * The returned LLVMValueRef corresponds to a llvm::Instruction
 * instance.
 */
LLVMValueRef LLVMGetFirstInstruction(LLVMBasicBlockRef BB);

/**
 * Obtain the last instruction in a basic block.
 *
 * The returned LLVMValueRef corresponds to an LLVM:Instruction.
 */
LLVMValueRef LLVMGetLastInstruction(LLVMBasicBlockRef BB);

/**
 * @}
 */

/**
 * @defgroup LLVMCCoreValueInstruction Instructions
 *
 * Functions in this group relate to the inspection and manipulation of
 * individual instructions.
 *
 * In the C++ API, an instruction is modeled by llvm::Instruction. This
 * class has a large number of descendents. llvm::Instruction is a
 * llvm::Value and in the C API, instructions are modeled by
 * LLVMValueRef.
 *
 * This group also contains sub-groups which operate on specific
 * llvm::Instruction types, e.g. llvm::CallInst.
 *
 * @{
 */

/**
 * Determine whether an instruction has any metadata attached.
 */
int LLVMHasMetadata(LLVMValueRef Val);

/**
 * Return metadata associated with an instruction value.
 */
LLVMValueRef LLVMGetMetadata(LLVMValueRef Val, unsigned KindID);

/**
 * Set metadata associated with an instruction value.
 */
void LLVMSetMetadata(LLVMValueRef Val, unsigned KindID, LLVMValueRef Node);

/**
 * Obtain the basic block to which an instruction belongs.
 *
 * @see llvm::Instruction::getParent()
 */
LLVMBasicBlockRef LLVMGetInstructionParent(LLVMValueRef Inst);

/**
 * Obtain the instruction that occurs after the one specified.
 *
 * The next instruction will be from the same basic block.
 *
 * If this is the last instruction in a basic block, NULL will be
 * returned.
 */
LLVMValueRef LLVMGetNextInstruction(LLVMValueRef Inst);

/**
 * Obtain the instruction that occurred before this one.
 *
 * If the instruction is the first instruction in a basic block, NULL
 * will be returned.
 */
LLVMValueRef LLVMGetPreviousInstruction(LLVMValueRef Inst);

/**
 * Remove and delete an instruction.
 *
 * The instruction specified is removed from its containing building
 * block and then deleted.
 *
 * @see llvm::Instruction::eraseFromParent()
 */
void LLVMInstructionEraseFromParent(LLVMValueRef Inst);

/**
 * Obtain the code opcode for an individual instruction.
 *
 * @see llvm::Instruction::getOpCode()
 */
LLVMOpcode   LLVMGetInstructionOpcode(LLVMValueRef Inst);

/**
 * Obtain the predicate of an instruction.
 *
 * This is only valid for instructions that correspond to llvm::ICmpInst
 * or llvm::ConstantExpr whose opcode is llvm::Instruction::ICmp.
 *
 * @see llvm::ICmpInst::getPredicate()
 */
LLVMIntPredicate LLVMGetICmpPredicate(LLVMValueRef Inst);

/**
 * Obtain the float predicate of an instruction.
 *
 * This is only valid for instructions that correspond to llvm::FCmpInst
 * or llvm::ConstantExpr whose opcode is llvm::Instruction::FCmp.
 *
 * @see llvm::FCmpInst::getPredicate()
 */
LLVMRealPredicate LLVMGetFCmpPredicate(LLVMValueRef Inst);

/**
 * Create a copy of 'this' instruction that is identical in all ways
 * except the following:
 *   * The instruction has no parent
 *   * The instruction has no name
 *
 * @see llvm::Instruction::clone()
 */
LLVMValueRef LLVMInstructionClone(LLVMValueRef Inst);

/**
 * @defgroup LLVMCCoreValueInstructionCall Call Sites and Invocations
 *
 * Functions in this group apply to instructions that refer to call
 * sites and invocations. These correspond to C++ types in the
 * llvm::CallInst class tree.
 *
 * @{
 */

/**
 * Set the calling convention for a call instruction.
 *
 * This expects an LLVMValueRef that corresponds to a llvm::CallInst or
 * llvm::InvokeInst.
 *
 * @see llvm::CallInst::setCallingConv()
 * @see llvm::InvokeInst::setCallingConv()
 */
void LLVMSetInstructionCallConv(LLVMValueRef Instr, unsigned CC);

/**
 * Obtain the calling convention for a call instruction.
 *
 * This is the opposite of LLVMSetInstructionCallConv(). Reads its
 * usage.
 *
 * @see LLVMSetInstructionCallConv()
 */
unsigned LLVMGetInstructionCallConv(LLVMValueRef Instr);


void LLVMAddInstrAttribute(LLVMValueRef Instr, unsigned index, LLVMAttribute);
void LLVMRemoveInstrAttribute(LLVMValueRef Instr, unsigned index,
                              LLVMAttribute);
void LLVMSetInstrParamAlignment(LLVMValueRef Instr, unsigned index,
                                unsigned align);

/**
 * Obtain whether a call instruction is a tail call.
 *
 * This only works on llvm::CallInst instructions.
 *
 * @see llvm::CallInst::isTailCall()
 */
LLVMBool LLVMIsTailCall(LLVMValueRef CallInst);

/**
 * Set whether a call instruction is a tail call.
 *
 * This only works on llvm::CallInst instructions.
 *
 * @see llvm::CallInst::setTailCall()
 */
void LLVMSetTailCall(LLVMValueRef CallInst, LLVMBool IsTailCall);

/**
 * @}
 */

/**
 * @defgroup LLVMCCoreValueInstructionTerminator Terminators
 *
 * Functions in this group only apply to instructions that map to
 * llvm::TerminatorInst instances.
 *
 * @{
 */

/**
 * Return the number of successors that this terminator has.
 *
 * @see llvm::TerminatorInst::getNumSuccessors
 */
unsigned LLVMGetNumSuccessors(LLVMValueRef Term);

/**
 * Return the specified successor.
 *
 * @see llvm::TerminatorInst::getSuccessor
 */
LLVMBasicBlockRef LLVMGetSuccessor(LLVMValueRef Term, unsigned i);

/**
 * Update the specified successor to point at the provided block.
 *
 * @see llvm::TerminatorInst::setSuccessor
 */
void LLVMSetSuccessor(LLVMValueRef Term, unsigned i, LLVMBasicBlockRef block);

/**
 * Return if a branch is conditional.
 *
 * This only works on llvm::BranchInst instructions.
 *
 * @see llvm::BranchInst::isConditional
 */
LLVMBool LLVMIsConditional(LLVMValueRef Branch);

/**
 * Return the condition of a branch instruction.
 *
 * This only works on llvm::BranchInst instructions.
 *
 * @see llvm::BranchInst::getCondition
 */
LLVMValueRef LLVMGetCondition(LLVMValueRef Branch);

/**
 * Set the condition of a branch instruction.
 *
 * This only works on llvm::BranchInst instructions.
 *
 * @see llvm::BranchInst::setCondition
 */
void LLVMSetCondition(LLVMValueRef Branch, LLVMValueRef Cond);

/**
 * Obtain the default destination basic block of a switch instruction.
 *
 * This only works on llvm::SwitchInst instructions.
 *
 * @see llvm::SwitchInst::getDefaultDest()
 */
LLVMBasicBlockRef LLVMGetSwitchDefaultDest(LLVMValueRef SwitchInstr);

/**
 * @}
 */

/**
 * @defgroup LLVMCCoreValueInstructionPHINode PHI Nodes
 *
 * Functions in this group only apply to instructions that map to
 * llvm::PHINode instances.
 *
 * @{
 */

/**
 * Add an incoming value to the end of a PHI list.
 */
void LLVMAddIncoming(LLVMValueRef PhiNode, LLVMValueRef *IncomingValues,
                     LLVMBasicBlockRef *IncomingBlocks, unsigned Count);

/**
 * Obtain the number of incoming basic blocks to a PHI node.
 */
unsigned LLVMCountIncoming(LLVMValueRef PhiNode);

/**
 * Obtain an incoming value to a PHI node as an LLVMValueRef.
 */
LLVMValueRef LLVMGetIncomingValue(LLVMValueRef PhiNode, unsigned Index);

/**
 * Obtain an incoming value to a PHI node as an LLVMBasicBlockRef.
 */
LLVMBasicBlockRef LLVMGetIncomingBlock(LLVMValueRef PhiNode, unsigned Index);

/**
 * @}
 */

/**
 * @}
 */

/**
 * @}
 */

/**
 * @defgroup LLVMCCoreInstructionBuilder Instruction Builders
 *
 * An instruction builder represents a point within a basic block and is
 * the exclusive means of building instructions using the C interface.
 *
 * @{
 */

LLVMBuilderRef LLVMCreateBuilderInContext(LLVMContextRef C);
LLVMBuilderRef LLVMCreateBuilder(void);
void LLVMPositionBuilder(LLVMBuilderRef Builder, LLVMBasicBlockRef Block,
                         LLVMValueRef Instr);
void LLVMPositionBuilderBefore(LLVMBuilderRef Builder, LLVMValueRef Instr);
void LLVMPositionBuilderAtEnd(LLVMBuilderRef Builder, LLVMBasicBlockRef Block);
LLVMBasicBlockRef LLVMGetInsertBlock(LLVMBuilderRef Builder);
void LLVMClearInsertionPosition(LLVMBuilderRef Builder);
void LLVMInsertIntoBuilder(LLVMBuilderRef Builder, LLVMValueRef Instr);
void LLVMInsertIntoBuilderWithName(LLVMBuilderRef Builder, LLVMValueRef Instr,
                                   const char *Name);
void LLVMDisposeBuilder(LLVMBuilderRef Builder);

/* Metadata */
void LLVMSetCurrentDebugLocation(LLVMBuilderRef Builder, LLVMValueRef L);
LLVMValueRef LLVMGetCurrentDebugLocation(LLVMBuilderRef Builder);
void LLVMSetInstDebugLocation(LLVMBuilderRef Builder, LLVMValueRef Inst);

/* Terminators */
LLVMValueRef LLVMBuildRetVoid(LLVMBuilderRef);
LLVMValueRef LLVMBuildRet(LLVMBuilderRef, LLVMValueRef V);
LLVMValueRef LLVMBuildAggregateRet(LLVMBuilderRef, LLVMValueRef *RetVals,
                                   unsigned N);
LLVMValueRef LLVMBuildBr(LLVMBuilderRef, LLVMBasicBlockRef Dest);
LLVMValueRef LLVMBuildCondBr(LLVMBuilderRef, LLVMValueRef If,
                             LLVMBasicBlockRef Then, LLVMBasicBlockRef Else);
LLVMValueRef LLVMBuildSwitch(LLVMBuilderRef, LLVMValueRef V,
                             LLVMBasicBlockRef Else, unsigned NumCases);
LLVMValueRef LLVMBuildIndirectBr(LLVMBuilderRef B, LLVMValueRef Addr,
                                 unsigned NumDests);
LLVMValueRef LLVMBuildInvoke(LLVMBuilderRef, LLVMValueRef Fn,
                             LLVMValueRef *Args, unsigned NumArgs,
                             LLVMBasicBlockRef Then, LLVMBasicBlockRef Catch,
                             const char *Name);
LLVMValueRef LLVMBuildLandingPad(LLVMBuilderRef B, LLVMTypeRef Ty,
                                 unsigned NumClauses, const char *Name);
LLVMValueRef LLVMBuildResume(LLVMBuilderRef B, LLVMValueRef Exn);
LLVMValueRef LLVMBuildUnreachable(LLVMBuilderRef);

/* Add a case to the switch instruction */
void LLVMAddCase(LLVMValueRef Switch, LLVMValueRef OnVal,
                 LLVMBasicBlockRef Dest);

/* Add a destination to the indirectbr instruction */
void LLVMAddDestination(LLVMValueRef IndirectBr, LLVMBasicBlockRef Dest);

/* Add a catch or filter clause to the landingpad instruction */
void LLVMAddClause(LLVMValueRef LandingPad, LLVMValueRef ClauseVal);

/* Set the 'cleanup' flag in the landingpad instruction */
void LLVMSetCleanup(LLVMValueRef LandingPad, LLVMBool Val);

/* Arithmetic */
LLVMValueRef LLVMBuildAdd(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name);
LLVMValueRef LLVMBuildNSWAdd(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                             const char *Name);
LLVMValueRef LLVMBuildNUWAdd(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                             const char *Name);
LLVMValueRef LLVMBuildFAdd(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name);
LLVMValueRef LLVMBuildSub(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name);
LLVMValueRef LLVMBuildNSWSub(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                             const char *Name);
LLVMValueRef LLVMBuildNUWSub(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                             const char *Name);
LLVMValueRef LLVMBuildFSub(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name);
LLVMValueRef LLVMBuildMul(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name);
LLVMValueRef LLVMBuildNSWMul(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                             const char *Name);
LLVMValueRef LLVMBuildNUWMul(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                             const char *Name);
LLVMValueRef LLVMBuildFMul(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name);
LLVMValueRef LLVMBuildUDiv(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name);
LLVMValueRef LLVMBuildSDiv(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name);
LLVMValueRef LLVMBuildExactSDiv(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                                const char *Name);
LLVMValueRef LLVMBuildFDiv(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name);
LLVMValueRef LLVMBuildURem(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name);
LLVMValueRef LLVMBuildSRem(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name);
LLVMValueRef LLVMBuildFRem(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name);
LLVMValueRef LLVMBuildShl(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name);
LLVMValueRef LLVMBuildLShr(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name);
LLVMValueRef LLVMBuildAShr(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name);
LLVMValueRef LLVMBuildAnd(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name);
LLVMValueRef LLVMBuildOr(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name);
LLVMValueRef LLVMBuildXor(LLVMBuilderRef, LLVMValueRef LHS, LLVMValueRef RHS,
                          const char *Name);
LLVMValueRef LLVMBuildBinOp(LLVMBuilderRef B, LLVMOpcode Op,
                            LLVMValueRef LHS, LLVMValueRef RHS,
                            const char *Name);
LLVMValueRef LLVMBuildNeg(LLVMBuilderRef, LLVMValueRef V, const char *Name);
LLVMValueRef LLVMBuildNSWNeg(LLVMBuilderRef B, LLVMValueRef V,
                             const char *Name);
LLVMValueRef LLVMBuildNUWNeg(LLVMBuilderRef B, LLVMValueRef V,
                             const char *Name);
LLVMValueRef LLVMBuildFNeg(LLVMBuilderRef, LLVMValueRef V, const char *Name);
LLVMValueRef LLVMBuildNot(LLVMBuilderRef, LLVMValueRef V, const char *Name);

/* Memory */
LLVMValueRef LLVMBuildMalloc(LLVMBuilderRef, LLVMTypeRef Ty, const char *Name);
LLVMValueRef LLVMBuildArrayMalloc(LLVMBuilderRef, LLVMTypeRef Ty,
                                  LLVMValueRef Val, const char *Name);
LLVMValueRef LLVMBuildAlloca(LLVMBuilderRef, LLVMTypeRef Ty, const char *Name);
LLVMValueRef LLVMBuildArrayAlloca(LLVMBuilderRef, LLVMTypeRef Ty,
                                  LLVMValueRef Val, const char *Name);
LLVMValueRef LLVMBuildFree(LLVMBuilderRef, LLVMValueRef PointerVal);
LLVMValueRef LLVMBuildLoad(LLVMBuilderRef, LLVMValueRef PointerVal,
                           const char *Name);
LLVMValueRef LLVMBuildStore(LLVMBuilderRef, LLVMValueRef Val, LLVMValueRef Ptr);
LLVMValueRef LLVMBuildGEP(LLVMBuilderRef B, LLVMValueRef Pointer,
                          LLVMValueRef *Indices, unsigned NumIndices,
                          const char *Name);
LLVMValueRef LLVMBuildInBoundsGEP(LLVMBuilderRef B, LLVMValueRef Pointer,
                                  LLVMValueRef *Indices, unsigned NumIndices,
                                  const char *Name);
LLVMValueRef LLVMBuildStructGEP(LLVMBuilderRef B, LLVMValueRef Pointer,
                                unsigned Idx, const char *Name);
LLVMValueRef LLVMBuildGlobalString(LLVMBuilderRef B, const char *Str,
                                   const char *Name);
LLVMValueRef LLVMBuildGlobalStringPtr(LLVMBuilderRef B, const char *Str,
                                      const char *Name);
LLVMBool LLVMGetVolatile(LLVMValueRef MemoryAccessInst);
void LLVMSetVolatile(LLVMValueRef MemoryAccessInst, LLVMBool IsVolatile);

/* Casts */
LLVMValueRef LLVMBuildTrunc(LLVMBuilderRef, LLVMValueRef Val,
                            LLVMTypeRef DestTy, const char *Name);
LLVMValueRef LLVMBuildZExt(LLVMBuilderRef, LLVMValueRef Val,
                           LLVMTypeRef DestTy, const char *Name);
LLVMValueRef LLVMBuildSExt(LLVMBuilderRef, LLVMValueRef Val,
                           LLVMTypeRef DestTy, const char *Name);
LLVMValueRef LLVMBuildFPToUI(LLVMBuilderRef, LLVMValueRef Val,
                             LLVMTypeRef DestTy, const char *Name);
LLVMValueRef LLVMBuildFPToSI(LLVMBuilderRef, LLVMValueRef Val,
                             LLVMTypeRef DestTy, const char *Name);
LLVMValueRef LLVMBuildUIToFP(LLVMBuilderRef, LLVMValueRef Val,
                             LLVMTypeRef DestTy, const char *Name);
LLVMValueRef LLVMBuildSIToFP(LLVMBuilderRef, LLVMValueRef Val,
                             LLVMTypeRef DestTy, const char *Name);
LLVMValueRef LLVMBuildFPTrunc(LLVMBuilderRef, LLVMValueRef Val,
                              LLVMTypeRef DestTy, const char *Name);
LLVMValueRef LLVMBuildFPExt(LLVMBuilderRef, LLVMValueRef Val,
                            LLVMTypeRef DestTy, const char *Name);
LLVMValueRef LLVMBuildPtrToInt(LLVMBuilderRef, LLVMValueRef Val,
                               LLVMTypeRef DestTy, const char *Name);
LLVMValueRef LLVMBuildIntToPtr(LLVMBuilderRef, LLVMValueRef Val,
                               LLVMTypeRef DestTy, const char *Name);
LLVMValueRef LLVMBuildBitCast(LLVMBuilderRef, LLVMValueRef Val,
                              LLVMTypeRef DestTy, const char *Name);
LLVMValueRef LLVMBuildAddrSpaceCast(LLVMBuilderRef, LLVMValueRef Val,
                                    LLVMTypeRef DestTy, const char *Name);
LLVMValueRef LLVMBuildZExtOrBitCast(LLVMBuilderRef, LLVMValueRef Val,
                                    LLVMTypeRef DestTy, const char *Name);
LLVMValueRef LLVMBuildSExtOrBitCast(LLVMBuilderRef, LLVMValueRef Val,
                                    LLVMTypeRef DestTy, const char *Name);
LLVMValueRef LLVMBuildTruncOrBitCast(LLVMBuilderRef, LLVMValueRef Val,
                                     LLVMTypeRef DestTy, const char *Name);
LLVMValueRef LLVMBuildCast(LLVMBuilderRef B, LLVMOpcode Op, LLVMValueRef Val,
                           LLVMTypeRef DestTy, const char *Name);
LLVMValueRef LLVMBuildPointerCast(LLVMBuilderRef, LLVMValueRef Val,
                                  LLVMTypeRef DestTy, const char *Name);
LLVMValueRef LLVMBuildIntCast(LLVMBuilderRef, LLVMValueRef Val, /*Signed cast!*/
                              LLVMTypeRef DestTy, const char *Name);
LLVMValueRef LLVMBuildFPCast(LLVMBuilderRef, LLVMValueRef Val,
                             LLVMTypeRef DestTy, const char *Name);

/* Comparisons */
LLVMValueRef LLVMBuildICmp(LLVMBuilderRef, LLVMIntPredicate Op,
                           LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name);
LLVMValueRef LLVMBuildFCmp(LLVMBuilderRef, LLVMRealPredicate Op,
                           LLVMValueRef LHS, LLVMValueRef RHS,
                           const char *Name);

/* Miscellaneous instructions */
LLVMValueRef LLVMBuildPhi(LLVMBuilderRef, LLVMTypeRef Ty, const char *Name);
LLVMValueRef LLVMBuildCall(LLVMBuilderRef, LLVMValueRef Fn,
                           LLVMValueRef *Args, unsigned NumArgs,
                           const char *Name);
LLVMValueRef LLVMBuildSelect(LLVMBuilderRef, LLVMValueRef If,
                             LLVMValueRef Then, LLVMValueRef Else,
                             const char *Name);
LLVMValueRef LLVMBuildVAArg(LLVMBuilderRef, LLVMValueRef List, LLVMTypeRef Ty,
                            const char *Name);
LLVMValueRef LLVMBuildExtractElement(LLVMBuilderRef, LLVMValueRef VecVal,
                                     LLVMValueRef Index, const char *Name);
LLVMValueRef LLVMBuildInsertElement(LLVMBuilderRef, LLVMValueRef VecVal,
                                    LLVMValueRef EltVal, LLVMValueRef Index,
                                    const char *Name);
LLVMValueRef LLVMBuildShuffleVector(LLVMBuilderRef, LLVMValueRef V1,
                                    LLVMValueRef V2, LLVMValueRef Mask,
                                    const char *Name);
LLVMValueRef LLVMBuildExtractValue(LLVMBuilderRef, LLVMValueRef AggVal,
                                   unsigned Index, const char *Name);
LLVMValueRef LLVMBuildInsertValue(LLVMBuilderRef, LLVMValueRef AggVal,
                                  LLVMValueRef EltVal, unsigned Index,
                                  const char *Name);

LLVMValueRef LLVMBuildIsNull(LLVMBuilderRef, LLVMValueRef Val,
                             const char *Name);
LLVMValueRef LLVMBuildIsNotNull(LLVMBuilderRef, LLVMValueRef Val,
                                const char *Name);
LLVMValueRef LLVMBuildPtrDiff(LLVMBuilderRef, LLVMValueRef LHS,
                              LLVMValueRef RHS, const char *Name);
LLVMValueRef LLVMBuildFence(LLVMBuilderRef B, LLVMAtomicOrdering ordering,
                            LLVMBool singleThread, const char *Name);
LLVMValueRef LLVMBuildAtomicRMW(LLVMBuilderRef B, LLVMAtomicRMWBinOp op,
                                LLVMValueRef PTR, LLVMValueRef Val,
                                LLVMAtomicOrdering ordering,
                                LLVMBool singleThread);

/**
 * @}
 */

/**
 * @defgroup LLVMCCoreModuleProvider Module Providers
 *
 * @{
 */

/**
 * Changes the type of M so it can be passed to FunctionPassManagers and the
 * JIT.  They take ModuleProviders for historical reasons.
 */
LLVMModuleProviderRef
LLVMCreateModuleProviderForExistingModule(LLVMModuleRef M);

/**
 * Destroys the module M.
 */
void LLVMDisposeModuleProvider(LLVMModuleProviderRef M);

/**
 * @}
 */

/**
 * @defgroup LLVMCCoreMemoryBuffers Memory Buffers
 *
 * @{
 */

LLVMBool LLVMCreateMemoryBufferWithContentsOfFile(const char *Path,
                                                  LLVMMemoryBufferRef *OutMemBuf,
                                                  _Out_opt_ char **OutMessage);
LLVMBool LLVMCreateMemoryBufferWithSTDIN(LLVMMemoryBufferRef *OutMemBuf,
    _Out_opt_ char **OutMessage);
LLVMMemoryBufferRef LLVMCreateMemoryBufferWithMemoryRange(const char *InputData,
                                                          size_t InputDataLength,
                                                          const char *BufferName,
                                                          LLVMBool RequiresNullTerminator);
LLVMMemoryBufferRef LLVMCreateMemoryBufferWithMemoryRangeCopy(const char *InputData,
                                                              size_t InputDataLength,
                                                              const char *BufferName);
const char *LLVMGetBufferStart(LLVMMemoryBufferRef MemBuf);
size_t LLVMGetBufferSize(LLVMMemoryBufferRef MemBuf);
void LLVMDisposeMemoryBuffer(LLVMMemoryBufferRef MemBuf);

/**
 * @}
 */

/**
 * @defgroup LLVMCCorePassRegistry Pass Registry
 *
 * @{
 */

/** Return the global pass registry, for use with initialization functions.
    @see llvm::PassRegistry::getPassRegistry */
LLVMPassRegistryRef LLVMGetGlobalPassRegistry(void);

/**
 * @}
 */

/**
 * @defgroup LLVMCCorePassManagers Pass Managers
 *
 * @{
 */

/** Constructs a new whole-module pass pipeline. This type of pipeline is
    suitable for link-time optimization and whole-module transformations.
    @see llvm::PassManager::PassManager */
LLVMPassManagerRef LLVMCreatePassManager(void);

/** Constructs a new function-by-function pass pipeline over the module
    provider. It does not take ownership of the module provider. This type of
    pipeline is suitable for code generation and JIT compilation tasks.
    @see llvm::FunctionPassManager::FunctionPassManager */
LLVMPassManagerRef LLVMCreateFunctionPassManagerForModule(LLVMModuleRef M);

/** Deprecated: Use LLVMCreateFunctionPassManagerForModule instead. */
LLVMPassManagerRef LLVMCreateFunctionPassManager(LLVMModuleProviderRef MP);

/** Initializes, executes on the provided module, and finalizes all of the
    passes scheduled in the pass manager. Returns 1 if any of the passes
    modified the module, 0 otherwise.
    @see llvm::PassManager::run(Module&) */
LLVMBool LLVMRunPassManager(LLVMPassManagerRef PM, LLVMModuleRef M);

/** Initializes all of the function passes scheduled in the function pass
    manager. Returns 1 if any of the passes modified the module, 0 otherwise.
    @see llvm::FunctionPassManager::doInitialization */
LLVMBool LLVMInitializeFunctionPassManager(LLVMPassManagerRef FPM);

/** Executes all of the function passes scheduled in the function pass manager
    on the provided function. Returns 1 if any of the passes modified the
    function, false otherwise.
    @see llvm::FunctionPassManager::run(Function&) */
LLVMBool LLVMRunFunctionPassManager(LLVMPassManagerRef FPM, LLVMValueRef F);

/** Finalizes all of the function passes scheduled in in the function pass
    manager. Returns 1 if any of the passes modified the module, 0 otherwise.
    @see llvm::FunctionPassManager::doFinalization */
LLVMBool LLVMFinalizeFunctionPassManager(LLVMPassManagerRef FPM);

/** Frees the memory of a pass pipeline. For function pipelines, does not free
    the module provider.
    @see llvm::PassManagerBase::~PassManagerBase. */
void LLVMDisposePassManager(LLVMPassManagerRef PM);

/**
 * @}
 */

/**
 * @defgroup LLVMCCoreThreading Threading
 *
 * Handle the structures needed to make LLVM safe for multithreading.
 *
 * @{
 */

/** Deprecated: Multi-threading can only be enabled/disabled with the compile
    time define LLVM_ENABLE_THREADS.  This function always returns
    LLVMIsMultithreaded(). */
LLVMBool LLVMStartMultithreaded(void);

/** Deprecated: Multi-threading can only be enabled/disabled with the compile
    time define LLVM_ENABLE_THREADS. */
void LLVMStopMultithreaded(void);

/** Check whether LLVM is executing in thread-safe mode or not.
    @see llvm::llvm_is_multithreaded */
LLVMBool LLVMIsMultithreaded(void);

/**
 * @}
 */

/**
 * @}
 */

/**
 * @}
 */

#ifdef __cplusplus
}
#endif /* !defined(__cplusplus) */

#endif /* !defined(LLVM_C_CORE_H) */
