#ifndef GOOGLE_BREAKPAD_COMMON_ANDROID_INCLUDE_ASM_MIPS_REGDEF_H
#define GOOGLE_BREAKPAD_COMMON_ANDROID_INCLUDE_ASM_MIPS_REGDEF_H

#if defined(__has_include_next) && __has_include_next(<asm/regdef.h>)
#include_next <asm/regdef.h>
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
#if _MIPS_SIM == _MIPS_SIM_ABI32
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define zero $0  
#define AT $1  
#define v0 $2  
#define v1 $3
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define a0 $4  
#define a1 $5
#define a2 $6
#define a3 $7
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define t0 $8  
#define t1 $9
#define t2 $10
#define t3 $11
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define t4 $12
#define t5 $13
#define t6 $14
#define t7 $15
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define s0 $16  
#define s1 $17
#define s2 $18
#define s3 $19
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define s4 $20
#define s5 $21
#define s6 $22
#define s7 $23
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define t8 $24  
#define t9 $25
#define jp $25  
#define k0 $26  
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define k1 $27
#define gp $28  
#define sp $29  
#define fp $30  
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define s8 $30  
#define ra $31  
#endif
#if _MIPS_SIM == _MIPS_SIM_ABI64 || _MIPS_SIM == _MIPS_SIM_NABI32
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define zero $0  
#define AT $at  
#define v0 $2  
#define v1 $3
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define a0 $4  
#define a1 $5
#define a2 $6
#define a3 $7
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define a4 $8  
#define ta0 $8
#define a5 $9
#define ta1 $9
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define a6 $10
#define ta2 $10
#define a7 $11
#define ta3 $11
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define t0 $12  
#define t1 $13
#define t2 $14
#define t3 $15
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define s0 $16  
#define s1 $17
#define s2 $18
#define s3 $19
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define s4 $20
#define s5 $21
#define s6 $22
#define s7 $23
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define t8 $24  
#define t9 $25  
#define jp $25  
#define k0 $26  
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define k1 $27
#define gp $28  
#define sp $29  
#define fp $30  
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define s8 $30  
#define ra $31  
#endif
#endif  // defined(__has_include_next) && __has_include_next(<asm/regdef.h>)
#endif  // GOOGLE_BREAKPAD_COMMON_ANDROID_INCLUDE_ASM_MIPS_REGDEF_H
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
