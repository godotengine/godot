#define VMA_IMPLEMENTATION
#ifdef DEBUG_ENABLED
#ifndef _MSC_VER
#define _DEBUG
#endif
#endif

// AMD integrated GPUs report a low VkPhysicalDeviceLimits::maxMemoryAllocationCount
// (typically 4096). VMA 3.3.0 changed VMA_DEBUG_DONT_EXCEED_MAX_MEMORY_ALLOCATION_COUNT
// to default to 1, which makes VMA itself fail allocations with VK_ERROR_TOO_MANY_OBJECTS
// once that count is reached, instead of forwarding the request to the driver. This
// regressed content-heavy projects that ran fine on 4.6 (VMA 3.1.0, where the default was
// 0) on those GPUs. Restore the previous behavior and let the driver decide.
// See https://github.com/godotengine/godot/issues/120534
#define VMA_DEBUG_DONT_EXCEED_MAX_MEMORY_ALLOCATION_COUNT 0

#include "vk_mem_alloc.h"
