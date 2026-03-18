#ifndef GS_SORT_KEY_GLSL
#define GS_SORT_KEY_GLSL

// 64-bit key layout: hi = sortable depth, lo = tie-break.
uint gs_float_to_sortable_uint(float value) {
    uint bits = floatBitsToUint(value);
    uint mask = ((bits & 0x80000000u) != 0u) ? 0xffffffffu : 0x80000000u;
    return bits ^ mask;
}

uvec2 gs_pack_sort_key64(float depth, uint tie_break) {
    return uvec2(tie_break, gs_float_to_sortable_uint(depth));
}

#endif // GS_SORT_KEY_GLSL
