#include "vpx_config.h"

#if defined(WEBM_X86ASM) && (ARCH_X86 || ARCH_X86_64)
	#include "rtcd/vpx_dsp_rtcd_x86.h"
#elif defined(WEBM_ARMASM) && ARCH_ARM
	#include "rtcd/vpx_dsp_rtcd_arm.h"
#else
	#include "rtcd/vpx_dsp_rtcd_c.h"
#endif
