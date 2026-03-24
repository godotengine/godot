#ifndef GS_DEBUG_OVERLAY_MACROS_H
#define GS_DEBUG_OVERLAY_MACROS_H

/**
 * @file debug_overlay_macros.h
 * @brief Helper macros to reduce boilerplate in debug overlay setter implementations.
 *
 * These macros consolidate repetitive patterns found across:
 * - DebugOverlaySystem (12 nearly-identical setters)
 * - RenderDebugStateOrchestrator (7 nearly-identical setters)
 * - debug_overlay_methods.cpp (delegation methods)
 *
 * Usage Example:
 *   GS_DEBUG_OVERLAY_SETTER_IMPL(show_tile_grid)
 *   expands to:
 *   void DebugOverlaySystem::set_show_tile_grid(bool p_enabled) {
 *       if (options.show_tile_grid != p_enabled) {
 *           options.show_tile_grid = p_enabled;
 *           _mark_dirty();
 *       }
 *   }
 */

// ============================================================================
// DebugOverlaySystem setter macros
// ============================================================================

/**
 * Standard boolean setter for DebugOverlayOptions fields.
 * Marks system dirty on change.
 */
#define GS_DEBUG_OVERLAY_SETTER_IMPL(field_name)                     \
    void DebugOverlaySystem::set_##field_name(bool p_enabled) {      \
        if (options.field_name != p_enabled) {                       \
            options.field_name = p_enabled;                          \
            _mark_dirty();                                           \
        }                                                            \
    }

/**
 * Boolean setter with mutual exclusivity.
 * When enabled, disables the other_field.
 */
#define GS_DEBUG_OVERLAY_SETTER_EXCLUSIVE_IMPL(field_name, other_field) \
    void DebugOverlaySystem::set_##field_name(bool p_enabled) {         \
        if (options.field_name != p_enabled) {                          \
            options.field_name = p_enabled;                             \
            if (p_enabled) {                                            \
                options.other_field = false;                            \
            }                                                           \
            _mark_dirty();                                              \
        }                                                               \
    }

// ============================================================================
// DebugOverlaySystem renderer-sync setter macros
// ============================================================================

/**
 * Command-sink setter that updates both DebugOverlaySystem and debug state,
 * then invalidates the overlay.
 */
#define GS_DEBUG_OVERLAY_RENDERER_SETTER_OVERLAY_IMPL(field_name)                          \
    void DebugOverlaySystem::set_renderer_##field_name(                                    \
            const DebugOverlayCommandSink &p_sink, bool p_enabled) {                       \
        if (!p_sink.debug_state) {                                                         \
            return;                                                                        \
        }                                                                                  \
        auto &debug_state = *p_sink.debug_state;                                           \
        if (debug_state.field_name == p_enabled) {                                         \
            return;                                                                        \
        }                                                                                  \
        debug_state.field_name = p_enabled;                                                \
        set_##field_name(p_enabled);                                                       \
        invalidate_renderer_overlay(p_sink, true);                                         \
    }

/**
 * Command-sink setter that updates both DebugOverlaySystem and debug state,
 * then invalidates the HUD.
 */
#define GS_DEBUG_OVERLAY_RENDERER_SETTER_HUD_IMPL(field_name)                              \
    void DebugOverlaySystem::set_renderer_##field_name(                                    \
            const DebugOverlayCommandSink &p_sink, bool p_enabled) {                       \
        if (!p_sink.debug_state) {                                                         \
            return;                                                                        \
        }                                                                                  \
        auto &debug_state = *p_sink.debug_state;                                           \
        if (debug_state.field_name == p_enabled) {                                         \
            return;                                                                        \
        }                                                                                  \
        debug_state.field_name = p_enabled;                                                \
        set_##field_name(p_enabled);                                                       \
        invalidate_renderer_hud(p_sink, true);                                             \
    }

// ============================================================================
// RenderDebugStateOrchestrator setter macros
// ============================================================================

/**
 * Orchestrator setter that delegates to DebugOverlaySystem if available,
 * otherwise updates the orchestrator-owned debug state directly.
 * Uses the explicit command sink delegation method.
 */
#define GS_DEBUG_ORCHESTRATOR_SETTER_IMPL(field_name)                                       \
    void RenderDebugStateOrchestrator::set_debug_##field_name(bool p_enabled) {             \
        if (renderer->subsystem_state.debug_overlay_system.is_valid()) {                    \
            renderer->subsystem_state.debug_overlay_system->build_command_sink(renderer)     \
                    .set_##field_name(p_enabled);                                            \
            return;                                                                         \
        }                                                                                   \
        if (debug_state.field_name == p_enabled) {                                \
            return;                                                                         \
        }                                                                                   \
        debug_state.field_name = p_enabled;                                       \
    }

// ============================================================================
// GaussianSplatRenderer delegation macros (debug_overlay_methods.cpp)
// ============================================================================

/**
 * Simple setter delegation to DebugOverlaySystem.
 */
#define GS_RENDERER_DEBUG_SETTER_IMPL(field_name)                                           \
    void GaussianSplatRenderer::set_debug_##field_name(bool p_enabled) {                    \
        if (subsystem_state.debug_overlay_system.is_valid()) {                              \
            subsystem_state.debug_overlay_system->set_##field_name(p_enabled);              \
        }                                                                                   \
    }

/**
 * Simple getter delegation to DebugOverlaySystem.
 */
#define GS_RENDERER_DEBUG_GETTER_IMPL(field_name)                                           \
    bool GaussianSplatRenderer::get_debug_##field_name() const {                            \
        return subsystem_state.debug_overlay_system.is_valid()                              \
                ? subsystem_state.debug_overlay_system->get_##field_name()                  \
                : false;                                                                    \
    }

/**
 * Combined setter and getter delegation.
 */
#define GS_RENDERER_DEBUG_ACCESSOR_IMPL(field_name)                                         \
    GS_RENDERER_DEBUG_SETTER_IMPL(field_name)                                               \
    GS_RENDERER_DEBUG_GETTER_IMPL(field_name)

#endif // GS_DEBUG_OVERLAY_MACROS_H
