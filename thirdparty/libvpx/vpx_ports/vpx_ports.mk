##
##  Copyright (c) 2012 The WebM project authors. All Rights Reserved.
##
##  Use of this source code is governed by a BSD-style license
##  that can be found in the LICENSE file in the root of the source
##  tree. An additional intellectual property rights grant can be found
##  in the file PATENTS.  All contributing project authors may
##  be found in the AUTHORS file in the root of the source tree.
##


PORTS_SRCS-yes += vpx_ports.mk

PORTS_SRCS-yes += bitops.h
PORTS_SRCS-yes += compiler_attributes.h
PORTS_SRCS-yes += mem.h
PORTS_SRCS-yes += static_assert.h
PORTS_SRCS-yes += system_state.h
PORTS_SRCS-yes += vpx_timer.h

ifeq ($(VPX_ARCH_X86),yes)
PORTS_SRCS-$(HAVE_MMX) += emms_mmx.c
endif
ifeq ($(VPX_ARCH_X86_64),yes)
# Visual Studio x64 does not support the _mm_empty() intrinsic.
PORTS_SRCS-$(HAVE_MMX) += emms_mmx.asm
endif

ifeq ($(VPX_ARCH_X86_64),yes)
PORTS_SRCS-$(CONFIG_MSVS) += float_control_word.asm
endif

ifeq ($(VPX_ARCH_X86)$(VPX_ARCH_X86_64),yes)
PORTS_SRCS-yes += x86.h
PORTS_SRCS-yes += x86_abi_support.asm
endif

ifeq ($(VPX_ARCH_AARCH64),yes)
PORTS_SRCS-yes += aarch64_cpudetect.c
else
PORTS_SRCS-$(VPX_ARCH_ARM) += aarch32_cpudetect.c
endif
PORTS_SRCS-$(VPX_ARCH_ARM) += arm_cpudetect.h
PORTS_SRCS-$(VPX_ARCH_ARM) += arm.h

PORTS_SRCS-$(VPX_ARCH_PPC) += ppc_cpudetect.c
PORTS_SRCS-$(VPX_ARCH_PPC) += ppc.h

PORTS_SRCS-$(VPX_ARCH_MIPS) += mips_cpudetect.c
PORTS_SRCS-$(VPX_ARCH_MIPS) += mips.h

PORTS_SRCS-$(VPX_ARCH_LOONGARCH) += loongarch_cpudetect.c
PORTS_SRCS-$(VPX_ARCH_LOONGARCH) += loongarch.h

ifeq ($(VPX_ARCH_MIPS), yes)
PORTS_SRCS-yes += asmdefs_mmi.h
endif
