#pragma once

#include <stdint.h>

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;

#define BLZ_NORMAL    0          // normal mode
#define BLZ_BEST      1          // best mode

u8 *BLZ_Code(u8 *raw_buffer, int raw_len, u32 *new_len, int best);