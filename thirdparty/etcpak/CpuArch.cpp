#include "CpuArch.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)

#if defined(__INTEL_COMPILER) && (__INTEL_COMPILER >= 1300)

#include <immintrin.h>

int check_4th_gen_intel_core_features()
{
    const int the_4th_gen_features = 
        (_FEATURE_AVX2 | _FEATURE_FMA | _FEATURE_BMI | _FEATURE_LZCNT | _FEATURE_MOVBE);
    return _may_i_use_cpu_feature( the_4th_gen_features );
}

#else /* non-Intel compiler */

#include <stdint.h>
#if defined(_MSC_VER)
# include <intrin.h>
#endif

void run_cpuid(uint32_t eax, uint32_t ecx, uint32_t* abcd)
{
#if defined(_MSC_VER)
    __cpuidex((int*)abcd, eax, ecx);
#else
    uint32_t ebx, edx;
# if defined( __i386__ ) && defined ( __PIC__ )
    /* in case of PIC under 32-bit EBX cannot be clobbered */
    __asm__ ( "movl %%ebx, %%edi \n\t cpuid \n\t xchgl %%ebx, %%edi" : "=D" (ebx),
# else
    __asm__ ( "cpuid" : "+b" (ebx),
# endif
        "+a" (eax), "+c" (ecx), "=d" (edx) );
    abcd[0] = eax; abcd[1] = ebx; abcd[2] = ecx; abcd[3] = edx;
#endif
}     

int check_xcr0_ymm() 
{
    uint32_t xcr0;
#if defined(_MSC_VER)
    xcr0 = (uint32_t)_xgetbv(0);  /* min VS2010 SP1 compiler is required */
#else
    __asm__ ("xgetbv" : "=a" (xcr0) : "c" (0) : "%edx" );
#endif
    return ((xcr0 & 6) == 6); /* checking if xmm and ymm state are enabled in XCR0 */
}


int check_4th_gen_intel_core_features()
{
    uint32_t abcd[4];
    uint32_t fma_movbe_osxsave_mask = ((1 << 12) | (1 << 22) | (1 << 27));
    uint32_t avx2_bmi12_mask = (1 << 5) | (1 << 3) | (1 << 8);

    /* CPUID.(EAX=01H, ECX=0H):ECX.FMA[bit 12]==1   && 
    CPUID.(EAX=01H, ECX=0H):ECX.MOVBE[bit 22]==1 && 
    CPUID.(EAX=01H, ECX=0H):ECX.OSXSAVE[bit 27]==1 */
    run_cpuid( 1, 0, abcd );
    if ( (abcd[2] & fma_movbe_osxsave_mask) != fma_movbe_osxsave_mask ) 
        return 0;

    if ( ! check_xcr0_ymm() )
        return 0;

    /*  CPUID.(EAX=07H, ECX=0H):EBX.AVX2[bit 5]==1  &&
    CPUID.(EAX=07H, ECX=0H):EBX.BMI1[bit 3]==1  &&
    CPUID.(EAX=07H, ECX=0H):EBX.BMI2[bit 8]==1  */
    run_cpuid( 7, 0, abcd );
    if ( (abcd[1] & avx2_bmi12_mask) != avx2_bmi12_mask ) 
        return 0;

    /* CPUID.(EAX=80000001H):ECX.LZCNT[bit 5]==1 */
    run_cpuid( 0x80000001, 0, abcd );
    if ( (abcd[2] & (1 << 5)) == 0)
        return 0;

    return 1;
}

#endif /* non-Intel compiler */


bool can_use_intel_core_4th_gen_features()
{
    static int the_4th_gen_features_available = -1;
    /* test is performed once */
    if (the_4th_gen_features_available < 0 )
        the_4th_gen_features_available = check_4th_gen_intel_core_features();

    return the_4th_gen_features_available == 1;
}

#else

bool can_use_intel_core_4th_gen_features()
{
    return false;
}

#endif
