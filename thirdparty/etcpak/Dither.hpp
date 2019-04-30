#ifndef __DITHER_HPP__
#define __DITHER_HPP__

#include <stddef.h>
#include <stdint.h>

void InitDither();
void Dither( uint8_t* data );

void Swizzle(const uint8_t* data, const ptrdiff_t pitch, uint8_t* output);

#ifdef __SSE4_1__
void Dither_SSE41(const uint8_t* data0, const uint8_t* data1, uint8_t* output0, uint8_t* output1);
void Swizzle_SSE41(const uint8_t* data, const ptrdiff_t pitch, uint8_t* output0, uint8_t* output1);
void Dither_Swizzle_SSE41(const uint8_t* data, const ptrdiff_t pitch, uint8_t* output0, uint8_t* output1);
#endif

#endif
