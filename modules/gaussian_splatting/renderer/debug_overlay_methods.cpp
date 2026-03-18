#include "gaussian_splat_renderer.h"
#include "../interfaces/debug_overlay_system.h"
#include "../interfaces/debug_overlay_macros.h"

#include "core/math/math_funcs.h"
#include <algorithm>

// Debug overlay controls for projection verification (Issue #125)
// Phase 8 migration: All state now managed by DebugOverlaySystem
//
// Macro-based implementation reduces boilerplate for repetitive setter/getter pairs.
// Note: set_debug_overlay_opacity is implemented in render_debug_state_orchestrator.cpp
// with additional DebugOverlaySystem invalidation logic.

// Simple delegation setters and getters
GS_RENDERER_DEBUG_ACCESSOR_IMPL(show_tile_bounds)
GS_RENDERER_DEBUG_ACCESSOR_IMPL(show_splat_coverage)
GS_RENDERER_DEBUG_ACCESSOR_IMPL(show_overflow_tiles)
GS_RENDERER_DEBUG_ACCESSOR_IMPL(show_projection_issues)
GS_RENDERER_DEBUG_ACCESSOR_IMPL(show_white_albedo)
GS_RENDERER_DEBUG_ACCESSOR_IMPL(show_shadow_opacity)
GS_RENDERER_DEBUG_ACCESSOR_IMPL(show_resolve_input)
GS_RENDERER_DEBUG_ACCESSOR_IMPL(show_resolve_output)
