#pragma once
#include <stdint.h>

typedef uint64_t dword_t;
typedef uint32_t word_t;
typedef uint16_t hword_t;
typedef uint8_t byte_t;
typedef int64_t dlong_t;
typedef int32_t long_t;
typedef int16_t short_t;
typedef int8_t char_t;
typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;

#define BIT(n) (1U << (n))

static inline uint16_t __local_bswap16(uint16_t x) {
	return ((x << 8) & 0xff00) | ((x >> 8) & 0x00ff);
}


static inline uint32_t __local_bswap32(uint32_t x) {
	return	((x << 24) & 0xff000000 ) |
			((x <<  8) & 0x00ff0000 ) |
			((x >>  8) & 0x0000ff00 ) |
			((x >> 24) & 0x000000ff );
}

static inline uint64_t __local_bswap64(uint64_t x)
{
	return (uint64_t)__local_bswap32(x>>32) |
	      ((uint64_t)__local_bswap32(x&0xFFFFFFFF) << 32);
}

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#define be_dword(a) __local_bswap64(a)
#define be_word(a)  __local_bswap32(a)
#define be_hword(a) __local_bswap16(a)
#define le_dword(a) (a)
#define le_word(a)  (a)
#define le_hword(a) (a)
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define be_dword(a) (a)
#define be_word(a)  (a)
#define be_hword(a) (a)
#define le_dword(a) __local_bswap64(a)
#define le_word(a)  __local_bswap32(a)
#define le_hword(a) __local_bswap16(a)
#else
#error "What's the endianness of the platform you're targeting?"
#endif

