#ifndef GOOGLE_BREAKPAD_COMMON_ANDROID_INCLUDE_ASM_MIPS_ASM_H
#define GOOGLE_BREAKPAD_COMMON_ANDROID_INCLUDE_ASM_MIPS_ASM_H

#if defined(__has_include_next) && __has_include_next(<asm/asm.h>)
#include_next <asm/asm.h>
#else

/****************************************************************************
 ****************************************************************************
 ***
 ***   This header was automatically generated from a Linux kernel header
 ***   of the same name, to make information necessary for userspace to
 ***   call into the kernel available to libc.  It contains only constants,
 ***   structures, and macros generated from the original header, and thus,
 ***   contains no copyrightable information.
 ***
 ***   To edit the content of this header, modify the corresponding
 ***   source file (e.g. under external/kernel-headers/original/) then
 ***   run bionic/libc/kernel/tools/update_all.py
 ***
 ***   Any manual change here will be lost the next time this script will
 ***   be run. You've been warned!
 ***
 ****************************************************************************
 ****************************************************************************/

#include <asm/sgidefs.h>
#ifndef CAT
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#ifdef __STDC__
#define __CAT(str1, str2) str1##str2
#else
#define __CAT(str1, str2) str1 str2
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#endif
#define CAT(str1, str2) __CAT(str1, str2)
#endif
#ifdef __PIC__
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define CPRESTORE(register)   .cprestore register
#define CPADD(register)   .cpadd register
#define CPLOAD(register)   .cpload register
#else
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define CPRESTORE(register)
#define CPADD(register)
#define CPLOAD(register)
#endif
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define LEAF(symbol)   .globl symbol;   .align 2;   .type symbol, @function;   .ent symbol, 0;  symbol: .frame sp, 0, ra
#define NESTED(symbol, framesize, rpc)   .globl symbol;   .align 2;   .type symbol, @function;   .ent symbol, 0;  symbol: .frame sp, framesize, rpc
#define END(function)   .end function;   .size function, .-function
#define EXPORT(symbol)   .globl symbol;  symbol:
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define FEXPORT(symbol)   .globl symbol;   .type symbol, @function;  symbol:
#define ABS(symbol,value)   .globl symbol;  symbol = value
#define PANIC(msg)   .set push;   .set reorder;   PTR_LA a0, 8f;   jal panic;  9: b 9b;   .set pop;   TEXT(msg)
#define PRINT(string)
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define TEXT(msg)   .pushsection .data;  8: .asciiz msg;   .popsection;
#define TTABLE(string)   .pushsection .text;   .word 1f;   .popsection   .pushsection .data;  1: .asciiz string;   .popsection
#define PREF(hint, addr)
#define PREFX(hint, addr)
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#if _MIPS_ISA == _MIPS_ISA_MIPS1
#define MOVN(rd, rs, rt)   .set push;   .set reorder;   beqz rt, 9f;   move rd, rs;   .set pop;  9:
#define MOVZ(rd, rs, rt)   .set push;   .set reorder;   bnez rt, 9f;   move rd, rs;   .set pop;  9:
#endif
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#if _MIPS_ISA == _MIPS_ISA_MIPS2 || _MIPS_ISA == _MIPS_ISA_MIPS3
#define MOVN(rd, rs, rt)   .set push;   .set noreorder;   bnezl rt, 9f;   move rd, rs;   .set pop;  9:
#define MOVZ(rd, rs, rt)   .set push;   .set noreorder;   beqzl rt, 9f;   move rd, rs;   .set pop;  9:
#endif
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#if _MIPS_ISA == _MIPS_ISA_MIPS4 || _MIPS_ISA == _MIPS_ISA_MIPS5 || _MIPS_ISA == _MIPS_ISA_MIPS32 || _MIPS_ISA == _MIPS_ISA_MIPS64
#define MOVN(rd, rs, rt)   movn rd, rs, rt
#define MOVZ(rd, rs, rt)   movz rd, rs, rt
#endif
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#if _MIPS_SIM == _MIPS_SIM_ABI32
#define ALSZ 7
#define ALMASK ~7
#endif
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#if _MIPS_SIM == _MIPS_SIM_NABI32 || _MIPS_SIM == _MIPS_SIM_ABI64
#define ALSZ 15
#define ALMASK ~15
#endif
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#ifdef __mips64
#define SZREG 8
#else
#define SZREG 4
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#endif
#if _MIPS_SIM == _MIPS_SIM_ABI32
#define REG_S sw
#define REG_L lw
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define REG_SUBU subu
#define REG_ADDU addu
#endif
#if _MIPS_SIM == _MIPS_SIM_NABI32 || _MIPS_SIM == _MIPS_SIM_ABI64
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define REG_S sd
#define REG_L ld
#define REG_SUBU dsubu
#define REG_ADDU daddu
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#endif
#if _MIPS_SZINT == 32
#define INT_ADD add
#define INT_ADDU addu
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define INT_ADDI addi
#define INT_ADDIU addiu
#define INT_SUB sub
#define INT_SUBU subu
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define INT_L lw
#define INT_S sw
#define INT_SLL sll
#define INT_SLLV sllv
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define INT_SRL srl
#define INT_SRLV srlv
#define INT_SRA sra
#define INT_SRAV srav
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#endif
#if _MIPS_SZINT == 64
#define INT_ADD dadd
#define INT_ADDU daddu
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define INT_ADDI daddi
#define INT_ADDIU daddiu
#define INT_SUB dsub
#define INT_SUBU dsubu
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define INT_L ld
#define INT_S sd
#define INT_SLL dsll
#define INT_SLLV dsllv
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define INT_SRL dsrl
#define INT_SRLV dsrlv
#define INT_SRA dsra
#define INT_SRAV dsrav
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#endif
#if _MIPS_SZLONG == 32
#define LONG_ADD add
#define LONG_ADDU addu
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define LONG_ADDI addi
#define LONG_ADDIU addiu
#define LONG_SUB sub
#define LONG_SUBU subu
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define LONG_L lw
#define LONG_S sw
#define LONG_SLL sll
#define LONG_SLLV sllv
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define LONG_SRL srl
#define LONG_SRLV srlv
#define LONG_SRA sra
#define LONG_SRAV srav
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define LONG .word
#define LONGSIZE 4
#define LONGMASK 3
#define LONGLOG 2
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#endif
#if _MIPS_SZLONG == 64
#define LONG_ADD dadd
#define LONG_ADDU daddu
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define LONG_ADDI daddi
#define LONG_ADDIU daddiu
#define LONG_SUB dsub
#define LONG_SUBU dsubu
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define LONG_L ld
#define LONG_S sd
#define LONG_SLL dsll
#define LONG_SLLV dsllv
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define LONG_SRL dsrl
#define LONG_SRLV dsrlv
#define LONG_SRA dsra
#define LONG_SRAV dsrav
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define LONG .dword
#define LONGSIZE 8
#define LONGMASK 7
#define LONGLOG 3
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#endif
#if _MIPS_SZPTR == 32
#define PTR_ADD add
#define PTR_ADDU addu
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define PTR_ADDI addi
#define PTR_ADDIU addiu
#define PTR_SUB sub
#define PTR_SUBU subu
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define PTR_L lw
#define PTR_S sw
#define PTR_LA la
#define PTR_LI li
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define PTR_SLL sll
#define PTR_SLLV sllv
#define PTR_SRL srl
#define PTR_SRLV srlv
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define PTR_SRA sra
#define PTR_SRAV srav
#define PTR_SCALESHIFT 2
#define PTR .word
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define PTRSIZE 4
#define PTRLOG 2
#endif
#if _MIPS_SZPTR == 64
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define PTR_ADD dadd
#define PTR_ADDU daddu
#define PTR_ADDI daddi
#define PTR_ADDIU daddiu
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define PTR_SUB dsub
#define PTR_SUBU dsubu
#define PTR_L ld
#define PTR_S sd
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define PTR_LA dla
#define PTR_LI dli
#define PTR_SLL dsll
#define PTR_SLLV dsllv
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define PTR_SRL dsrl
#define PTR_SRLV dsrlv
#define PTR_SRA dsra
#define PTR_SRAV dsrav
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define PTR_SCALESHIFT 3
#define PTR .dword
#define PTRSIZE 8
#define PTRLOG 3
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#endif
#if _MIPS_SIM == _MIPS_SIM_ABI32
#define MFC0 mfc0
#define MTC0 mtc0
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#endif
#if _MIPS_SIM == _MIPS_SIM_NABI32 || _MIPS_SIM == _MIPS_SIM_ABI64
#define MFC0 dmfc0
#define MTC0 dmtc0
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#endif
#define SSNOP sll zero, zero, 1
#define R10KCBARRIER(addr)
#endif  // defined(__has_include_next) && __has_include_next(<asm/asm.h>)
#endif  // GOOGLE_BREAKPAD_COMMON_ANDROID_INCLUDE_ASM_MIPS_ASM_H
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
