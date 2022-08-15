/*===-- llvm-c/Disassembler.h - Disassembler Public C Interface ---*- C -*-===*\
|*                                                                            *|
|*                     The LLVM Compiler Infrastructure                       *|
|*                                                                            *|
|* This file is distributed under the University of Illinois Open Source      *|
|* License. See LICENSE.TXT for details.                                      *|
|*                                                                            *|
|*===----------------------------------------------------------------------===*|
|*                                                                            *|
|* This header provides a public interface to a disassembler library.         *|
|* LLVM provides an implementation of this interface.                         *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef LLVM_C_DISASSEMBLER_H
#define LLVM_C_DISASSEMBLER_H

#include "dxc/Support/WinAdapter.h" // HLSL Change
#include "llvm/Support/DataTypes.h"
#include <stddef.h>

/**
 * @defgroup LLVMCDisassembler Disassembler
 * @ingroup LLVMC
 *
 * @{
 */

/**
 * An opaque reference to a disassembler context.
 */
typedef void *LLVMDisasmContextRef;

/**
 * The type for the operand information call back function.  This is called to
 * get the symbolic information for an operand of an instruction.  Typically
 * this is from the relocation information, symbol table, etc.  That block of
 * information is saved when the disassembler context is created and passed to
 * the call back in the DisInfo parameter.  The instruction containing operand
 * is at the PC parameter.  For some instruction sets, there can be more than
 * one operand with symbolic information.  To determine the symbolic operand
 * information for each operand, the bytes for the specific operand in the
 * instruction are specified by the Offset parameter and its byte widith is the
 * size parameter.  For instructions sets with fixed widths and one symbolic
 * operand per instruction, the Offset parameter will be zero and Size parameter
 * will be the instruction width.  The information is returned in TagBuf and is
 * Triple specific with its specific information defined by the value of
 * TagType for that Triple.  If symbolic information is returned the function
 * returns 1, otherwise it returns 0.
 */
typedef int (*LLVMOpInfoCallback)(void *DisInfo, uint64_t PC,
                                  uint64_t Offset, uint64_t Size,
                                  int TagType, void *TagBuf);

/**
 * The initial support in LLVM MC for the most general form of a relocatable
 * expression is "AddSymbol - SubtractSymbol + Offset".  For some Darwin targets
 * this full form is encoded in the relocation information so that AddSymbol and
 * SubtractSymbol can be link edited independent of each other.  Many other
 * platforms only allow a relocatable expression of the form AddSymbol + Offset
 * to be encoded.
 *
 * The LLVMOpInfoCallback() for the TagType value of 1 uses the struct
 * LLVMOpInfo1.  The value of the relocatable expression for the operand,
 * including any PC adjustment, is passed in to the call back in the Value
 * field.  The symbolic information about the operand is returned using all
 * the fields of the structure with the Offset of the relocatable expression
 * returned in the Value field.  It is possible that some symbols in the
 * relocatable expression were assembly temporary symbols, for example
 * "Ldata - LpicBase + constant", and only the Values of the symbols without
 * symbol names are present in the relocation information.  The VariantKind
 * type is one of the Target specific #defines below and is used to print
 * operands like "_foo@GOT", ":lower16:_foo", etc.
 */
struct LLVMOpInfoSymbol1 {
  uint64_t Present;  /* 1 if this symbol is present */
  const char *Name;  /* symbol name if not NULL */
  uint64_t Value;    /* symbol value if name is NULL */
};

struct LLVMOpInfo1 {
  struct LLVMOpInfoSymbol1 AddSymbol;
  struct LLVMOpInfoSymbol1 SubtractSymbol;
  uint64_t Value;
  uint64_t VariantKind;
};

/**
 * The operand VariantKinds for symbolic disassembly.
 */
#define LLVMDisassembler_VariantKind_None 0 /* all targets */

/**
 * The ARM target VariantKinds.
 */
#define LLVMDisassembler_VariantKind_ARM_HI16 1 /* :upper16: */
#define LLVMDisassembler_VariantKind_ARM_LO16 2 /* :lower16: */

/**
 * The ARM64 target VariantKinds.
 */
#define LLVMDisassembler_VariantKind_ARM64_PAGE       1 /* @page */
#define LLVMDisassembler_VariantKind_ARM64_PAGEOFF    2 /* @pageoff */
#define LLVMDisassembler_VariantKind_ARM64_GOTPAGE    3 /* @gotpage */
#define LLVMDisassembler_VariantKind_ARM64_GOTPAGEOFF 4 /* @gotpageoff */
#define LLVMDisassembler_VariantKind_ARM64_TLVP       5 /* @tvlppage */
#define LLVMDisassembler_VariantKind_ARM64_TLVOFF     6 /* @tvlppageoff */

/**
 * The type for the symbol lookup function.  This may be called by the
 * disassembler for things like adding a comment for a PC plus a constant
 * offset load instruction to use a symbol name instead of a load address value.
 * It is passed the block information is saved when the disassembler context is
 * created and the ReferenceValue to look up as a symbol.  If no symbol is found
 * for the ReferenceValue NULL is returned.  The ReferenceType of the
 * instruction is passed indirectly as is the PC of the instruction in
 * ReferencePC.  If the output reference can be determined its type is returned
 * indirectly in ReferenceType along with ReferenceName if any, or that is set
 * to NULL.
 */
typedef const char *(*LLVMSymbolLookupCallback)(void *DisInfo,
                                                uint64_t ReferenceValue,
                                                uint64_t *ReferenceType,
                                                uint64_t ReferencePC,
                                                const char **ReferenceName);
/**
 * The reference types on input and output.
 */
/* No input reference type or no output reference type. */
#define LLVMDisassembler_ReferenceType_InOut_None 0

/* The input reference is from a branch instruction. */
#define LLVMDisassembler_ReferenceType_In_Branch 1
/* The input reference is from a PC relative load instruction. */
#define LLVMDisassembler_ReferenceType_In_PCrel_Load 2

/* The input reference is from an ARM64::ADRP instruction. */
#define LLVMDisassembler_ReferenceType_In_ARM64_ADRP 0x100000001
/* The input reference is from an ARM64::ADDXri instruction. */
#define LLVMDisassembler_ReferenceType_In_ARM64_ADDXri 0x100000002
/* The input reference is from an ARM64::LDRXui instruction. */
#define LLVMDisassembler_ReferenceType_In_ARM64_LDRXui 0x100000003
/* The input reference is from an ARM64::LDRXl instruction. */
#define LLVMDisassembler_ReferenceType_In_ARM64_LDRXl 0x100000004
/* The input reference is from an ARM64::ADR instruction. */
#define LLVMDisassembler_ReferenceType_In_ARM64_ADR 0x100000005

/* The output reference is to as symbol stub. */
#define LLVMDisassembler_ReferenceType_Out_SymbolStub 1
/* The output reference is to a symbol address in a literal pool. */
#define LLVMDisassembler_ReferenceType_Out_LitPool_SymAddr 2
/* The output reference is to a cstring address in a literal pool. */
#define LLVMDisassembler_ReferenceType_Out_LitPool_CstrAddr 3

/* The output reference is to a Objective-C CoreFoundation string. */
#define LLVMDisassembler_ReferenceType_Out_Objc_CFString_Ref 4
/* The output reference is to a Objective-C message. */
#define LLVMDisassembler_ReferenceType_Out_Objc_Message 5
/* The output reference is to a Objective-C message ref. */
#define LLVMDisassembler_ReferenceType_Out_Objc_Message_Ref 6
/* The output reference is to a Objective-C selector ref. */
#define LLVMDisassembler_ReferenceType_Out_Objc_Selector_Ref 7
/* The output reference is to a Objective-C class ref. */
#define LLVMDisassembler_ReferenceType_Out_Objc_Class_Ref 8

/* The output reference is to a C++ symbol name. */
#define LLVMDisassembler_ReferenceType_DeMangled_Name 9

#ifdef __cplusplus
extern "C" {
#endif /* !defined(__cplusplus) */

/**
 * Create a disassembler for the TripleName.  Symbolic disassembly is supported
 * by passing a block of information in the DisInfo parameter and specifying the
 * TagType and callback functions as described above.  These can all be passed
 * as NULL.  If successful, this returns a disassembler context.  If not, it
 * returns NULL. This function is equivalent to calling
 * LLVMCreateDisasmCPUFeatures() with an empty CPU name and feature set.
 */
LLVMDisasmContextRef LLVMCreateDisasm(const char *TripleName, void *DisInfo,
                                      int TagType, LLVMOpInfoCallback GetOpInfo,
                                      LLVMSymbolLookupCallback SymbolLookUp);

/**
 * Create a disassembler for the TripleName and a specific CPU.  Symbolic
 * disassembly is supported by passing a block of information in the DisInfo
 * parameter and specifying the TagType and callback functions as described
 * above.  These can all be passed * as NULL.  If successful, this returns a
 * disassembler context.  If not, it returns NULL. This function is equivalent
 * to calling LLVMCreateDisasmCPUFeatures() with an empty feature set.
 */
LLVMDisasmContextRef LLVMCreateDisasmCPU(const char *Triple, const char *CPU,
                                         void *DisInfo, int TagType,
                                         LLVMOpInfoCallback GetOpInfo,
                                         LLVMSymbolLookupCallback SymbolLookUp);

/**
 * Create a disassembler for the TripleName, a specific CPU and specific feature
 * string.  Symbolic disassembly is supported by passing a block of information
 * in the DisInfo parameter and specifying the TagType and callback functions as
 * described above.  These can all be passed * as NULL.  If successful, this
 * returns a disassembler context.  If not, it returns NULL.
 */
LLVMDisasmContextRef
LLVMCreateDisasmCPUFeatures(const char *Triple, const char *CPU,
                            const char *Features, void *DisInfo, int TagType,
                            LLVMOpInfoCallback GetOpInfo,
                            LLVMSymbolLookupCallback SymbolLookUp);

/**
 * Set the disassembler's options.  Returns 1 if it can set the Options and 0
 * otherwise.
 */
int LLVMSetDisasmOptions(LLVMDisasmContextRef DC, uint64_t Options);

/* The option to produce marked up assembly. */
#define LLVMDisassembler_Option_UseMarkup 1
/* The option to print immediates as hex. */
#define LLVMDisassembler_Option_PrintImmHex 2
/* The option use the other assembler printer variant */
#define LLVMDisassembler_Option_AsmPrinterVariant 4
/* The option to set comment on instructions */
#define LLVMDisassembler_Option_SetInstrComments 8
  /* The option to print latency information alongside instructions */
#define LLVMDisassembler_Option_PrintLatency 16

/**
 * Dispose of a disassembler context.
 */
void LLVMDisasmDispose(LLVMDisasmContextRef DC);

/**
 * Disassemble a single instruction using the disassembler context specified in
 * the parameter DC.  The bytes of the instruction are specified in the
 * parameter Bytes, and contains at least BytesSize number of bytes.  The
 * instruction is at the address specified by the PC parameter.  If a valid
 * instruction can be disassembled, its string is returned indirectly in
 * OutString whose size is specified in the parameter OutStringSize.  This
 * function returns the number of bytes in the instruction or zero if there was
 * no valid instruction.
 */
size_t LLVMDisasmInstruction(LLVMDisasmContextRef DC, uint8_t *Bytes,
                             uint64_t BytesSize, uint64_t PC,
                             _Out_writes_z_(OutStringSize) char *OutString, size_t OutStringSize); // HLSL Change: annotation

/**
 * @}
 */

#ifdef __cplusplus
}
#endif /* !defined(__cplusplus) */

#endif /* !defined(LLVM_C_DISASSEMBLER_H) */
