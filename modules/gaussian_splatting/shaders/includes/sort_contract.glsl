#ifndef GS_SORT_CONTRACT_GLSL
#define GS_SORT_CONTRACT_GLSL

// Sort padding contract shared by CPU and GPU paths.
#ifndef GS_SORT_PAD_DEPTH_VALUE
#error "GS_SORT_PAD_DEPTH_VALUE must be defined by the host shader preamble."
#endif
const float GS_SORT_PAD_DEPTH = GS_SORT_PAD_DEPTH_VALUE;

#endif // GS_SORT_CONTRACT_GLSL
