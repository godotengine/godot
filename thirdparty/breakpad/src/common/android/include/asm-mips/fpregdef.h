#ifndef GOOGLE_BREAKPAD_COMMON_ANDROID_INCLUDE_ASM_MIPS_FPREGDEF_H
#define GOOGLE_BREAKPAD_COMMON_ANDROID_INCLUDE_ASM_MIPS_FPREGDEF_H

#if defined(__has_include_next) && __has_include_next(<asm/fpregdef.h>)
#include_next <asm/fpregdef.h>
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
#define fv0 $f0  
#define fv0f $f1
#define fv1 $f2
#define fv1f $f3
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define fa0 $f12  
#define fa0f $f13
#define fa1 $f14
#define fa1f $f15
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define ft0 $f4  
#define ft0f $f5
#define ft1 $f6
#define ft1f $f7
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define ft2 $f8
#define ft2f $f9
#define ft3 $f10
#define ft3f $f11
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define ft4 $f16
#define ft4f $f17
#define ft5 $f18
#define ft5f $f19
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define fs0 $f20  
#define fs0f $f21
#define fs1 $f22
#define fs1f $f23
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define fs2 $f24
#define fs2f $f25
#define fs3 $f26
#define fs3f $f27
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define fs4 $f28
#define fs4f $f29
#define fs5 $f30
#define fs5f $f31
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define fcr31 $31  
#endif
#if _MIPS_SIM == _MIPS_SIM_ABI64 || _MIPS_SIM == _MIPS_SIM_NABI32
#define fv0 $f0  
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define fv1 $f2
#define fa0 $f12  
#define fa1 $f13
#define fa2 $f14
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define fa3 $f15
#define fa4 $f16
#define fa5 $f17
#define fa6 $f18
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define fa7 $f19
#define ft0 $f4  
#define ft1 $f5
#define ft2 $f6
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define ft3 $f7
#define ft4 $f8
#define ft5 $f9
#define ft6 $f10
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define ft7 $f11
#define ft8 $f20
#define ft9 $f21
#define ft10 $f22
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define ft11 $f23
#define ft12 $f1
#define ft13 $f3
#define fs0 $f24  
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define fs1 $f25
#define fs2 $f26
#define fs3 $f27
#define fs4 $f28
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#define fs5 $f29
#define fs6 $f30
#define fs7 $f31
#define fcr31 $31
/* WARNING: DO NOT EDIT, AUTO-GENERATED CODE - SEE TOP FOR INSTRUCTIONS */
#endif
#endif  // defined(__has_include_next) && __has_include_next(<asm/fpregdef.h>)
#endif  // GOOGLE_BREAKPAD_COMMON_ANDROID_INCLUDE_ASM_MIPS_FPREGDEF_H
