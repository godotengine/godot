#ifndef __PROCESSALPHA_AVX2_HPP__
#define __PROCESSALPHA_AVX2_HPP__

#ifdef __SSE4_1__

#include <stdint.h>

uint64_t ProcessAlpha_AVX2( const uint8_t* src );

#endif

#endif
