#ifndef GAUSSIAN_RENDERING_DIAGNOSTICS_H
#define GAUSSIAN_RENDERING_DIAGNOSTICS_H

#include "core/config/engine.h"
#include "core/object/object.h"
#include "core/os/os.h"
#include "core/templates/hash_map.h"
#include "core/templates/vector.h"
#include "core/variant/dictionary.h"

class GaussianSplatRenderer;
class RenderingError;

class GaussianRenderingDiagnostics {
public:
    static GaussianRenderingDiagnostics *get_singleton();
    static void ensure_singleton();

    void process_command_line_requests();
    void register_renderer(GaussianSplatRenderer *p_renderer);
    void unregister_renderer(GaussianSplatRenderer *p_renderer);
    void notify_error(GaussianSplatRenderer *p_renderer, const RenderingError &p_error);
    void notify_recovery(GaussianSplatRenderer *p_renderer, const RenderingError &p_error);
    void notify_frame_completed(GaussianSplatRenderer *p_renderer);
    void request_runtime_report();

private:
    // Global coordinator instance; created on demand when diagnostics are first requested.
    static GaussianRenderingDiagnostics *singleton;

    // Active renderers we query when assembling runtime diagnostics.
    Vector<GaussianSplatRenderer *> registered_renderers;
    // Ring buffer of recent error dictionaries we attach to the next report.
    Vector<Dictionary> recent_errors;
    // Tracks command-line driven requests and throttle state so we avoid duplicate parsing/emission.
    bool cli_diagnostic_requested = false;
    bool cli_diagnostic_emitted = false;
    // Guards repeated validation-layer toggles once diagnostics are armed.
    bool validation_layers_forced = false;
    bool command_line_processed = false;
    // Last time we printed a report; used to enforce the 5s throttle in _emit_report_if_ready().
    uint64_t last_report_time_usec = 0;

    // Ensure GPU validation layers stay enabled while diagnostics are active.
    void _ensure_validation_layers();
    // Print a report if diagnostics are armed, renderers exist, and the throttle window has elapsed.
    void _emit_report_if_ready();
    // Collect renderer snapshots and recent errors into a JSON payload for stdout.
    void _emit_report();
    // Append an error entry, trimming the sliding window to the maximum history length.
    void _record_error_dictionary(const Dictionary &p_dict);
};

#endif // GAUSSIAN_RENDERING_DIAGNOSTICS_H
#include "../logger/logging_config.h"
