#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

// -- GODOT start --
#include <zlib.h> // Should come before including tinyexr.
// -- GODOT end --

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"
