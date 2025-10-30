#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif

#include <zlib.h> // Should come before including tinyexr.

#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"
