/*
 * mptCPU.h
 * --------
 * Purpose: CPU feature detection.
 * Notes  : (currently none)
 * Authors: OpenMPT Devs
 * The OpenMPT source code is released under the BSD license. Read LICENSE for more details.
 */


#pragma once


OPENMPT_NAMESPACE_BEGIN


#ifdef ENABLE_ASM

#define PROCSUPPORT_CPUID        0x00001 // Processor supports CPUID instruction (i586)
#define PROCSUPPORT_TSC          0x00002 // Processor supports RDTSC instruction (i586)
#define PROCSUPPORT_CMOV         0x00004 // Processor supports conditional move instructions (i686)
#define PROCSUPPORT_FPU          0x00008 // Processor supports x87 instructions
#define PROCSUPPORT_MMX          0x00010 // Processor supports MMX instructions
#define PROCSUPPORT_AMD_MMXEXT   0x00020 // Processor supports AMD MMX extensions
#define PROCSUPPORT_AMD_3DNOW    0x00040 // Processor supports AMD 3DNow! instructions
#define PROCSUPPORT_AMD_3DNOWEXT 0x00080 // Processor supports AMD 3DNow!2 instructions
#define PROCSUPPORT_SSE          0x00100 // Processor supports SSE instructions
#define PROCSUPPORT_SSE2         0x00200 // Processor supports SSE2 instructions
#define PROCSUPPORT_SSE3         0x00400 // Processor supports SSE3 instructions
#define PROCSUPPORT_SSSE3        0x00800 // Processor supports SSSE3 instructions
#define PROCSUPPORT_SSE4_1       0x01000 // Processor supports SSE4.1 instructions
#define PROCSUPPORT_SSE4_2       0x02000 // Processor supports SSE4.2 instructions

static const uint32 PROCSUPPORT_i486     = 0u                                        | PROCSUPPORT_FPU                                     ;
static const uint32 PROCSUPPORT_i586     = 0u | PROCSUPPORT_CPUID                    | PROCSUPPORT_FPU                                     ;
static const uint32 PROCSUPPORT_i686     = 0u | PROCSUPPORT_CPUID | PROCSUPPORT_CMOV | PROCSUPPORT_FPU                                     ;
static const uint32 PROCSUPPORT_x86_SSE  = 0u | PROCSUPPORT_CPUID | PROCSUPPORT_CMOV | PROCSUPPORT_FPU | PROCSUPPORT_SSE                   ;
static const uint32 PROCSUPPORT_x86_SSE2 = 0u | PROCSUPPORT_CPUID | PROCSUPPORT_CMOV | PROCSUPPORT_FPU | PROCSUPPORT_SSE | PROCSUPPORT_SSE2;
static const uint32 PROCSUPPORT_AMD64    = 0u | PROCSUPPORT_CPUID | PROCSUPPORT_CMOV                   | PROCSUPPORT_SSE | PROCSUPPORT_SSE2;

extern uint32 RealProcSupport;
extern uint32 ProcSupport;
extern char ProcVendorID[16+1];
extern uint16 ProcFamily;
extern uint8 ProcModel;
extern uint8 ProcStepping;

void InitProcSupport();

// enabled processor features for inline asm and intrinsics
static inline uint32 GetProcSupport()
{
	return ProcSupport;
}

// available processor features
static inline uint32 GetRealProcSupport()
{
	return RealProcSupport;
}

#endif // ENABLE_ASM


#ifdef MODPLUG_TRACKER
uint32 GetMinimumProcSupportFlags();
int GetMinimumSSEVersion();
int GetMinimumAVXVersion();
#endif


OPENMPT_NAMESPACE_END
