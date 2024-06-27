#pragma once
#if defined(__arm__)
# ifndef (__ARM_NEON)
#   define __ARM_NEON
# endif
#else
# ifndef __AVX__
#   define __AVX__
# endif
# ifndef __SSE__
#   define __SSE__
# endif
#endif