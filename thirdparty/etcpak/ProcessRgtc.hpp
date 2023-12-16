// -- GODOT start --

#ifndef __PROCESSRGTC_HPP__
#define __PROCESSRGTC_HPP__

#include <stddef.h>
#include <stdint.h>

void CompressRgtcR(const uint32_t *src, uint64_t *dst, uint32_t blocks, size_t width);
void CompressRgtcRG(const uint32_t *src, uint64_t *dst, uint32_t blocks, size_t width);

#endif

// -- GODOT end --
