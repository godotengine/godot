#ifndef __PROCESSRGB_AVX2_HPP__
#define __PROCESSRGB_AVX2_HPP__

#ifdef __SSE4_1__

#include <stdint.h>

uint64_t ProcessRGB_AVX2( const uint8_t* src );
uint64_t ProcessRGB_4x2_AVX2( const uint8_t* src );
uint64_t ProcessRGB_2x4_AVX2( const uint8_t* src );
uint64_t ProcessRGB_ETC2_AVX2( const uint8_t* src );

#endif

#endif
