#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

// -- BLAZIUM start --
#include <zlib.h> // Should come before including tinyexr.
// -- BLAZIUM end --

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"
