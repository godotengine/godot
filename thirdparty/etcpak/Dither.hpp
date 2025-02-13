#ifndef __DITHER_HPP__
#define __DITHER_HPP__

#include <stddef.h>
#include <stdint.h>

#ifdef __AVX2__
#  ifdef _MSC_VER
#    include <intrin.h>
#  else
#    include <x86intrin.h>
#  endif
#endif

void Dither( uint8_t* data );

#ifdef __AVX2__
void DitherAvx2( uint8_t* data, __m128i px0, __m128i px1, __m128i px2, __m128i px3 );
#endif

#endif
