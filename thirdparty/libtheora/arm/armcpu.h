/********************************************************************
 *                                                                  *
 * THIS FILE IS PART OF THE OggTheora SOFTWARE CODEC SOURCE CODE.   *
 * USE, DISTRIBUTION AND REPRODUCTION OF THIS LIBRARY SOURCE IS     *
 * GOVERNED BY A BSD-STYLE SOURCE LICENSE INCLUDED WITH THIS SOURCE *
 * IN 'COPYING'. PLEASE READ THESE TERMS BEFORE DISTRIBUTING.       *
 *                                                                  *
 * THE Theora SOURCE CODE IS COPYRIGHT (C) 2002-2010                *
 * by the Xiph.Org Foundation and contributors http://www.xiph.org/ *
 *                                                                  *
 ********************************************************************
 function:
    last mod: $Id: cpu.h 17344 2010-07-21 01:42:18Z tterribe $

 ********************************************************************/

#if !defined(_arm_armcpu_H)
# define _arm_armcpu_H (1)
#include "../internal.h"

/*"Parallel instructions" from ARM v6 and above.*/
#define OC_CPU_ARM_MEDIA    (1<<24)
/*Flags chosen to match arch/arm/include/asm/hwcap.h in the Linux kernel.*/
#define OC_CPU_ARM_EDSP     (1<<7)
#define OC_CPU_ARM_NEON     (1<<12)

ogg_uint32_t oc_cpu_flags_get(void);

#endif
