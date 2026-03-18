#include "rendering_diagnostics.h"

#include "rendering_error.h"

#include "core/io/json.h"
#include "core/variant/array.h"
#include "core/os/os.h"
#include "gaussian_splat_renderer.h"
#include "../logger/gs_logger.h"

GaussianRenderingDiagnostics *GaussianRenderingDiagnostics::singleton = nullptr;

GaussianRenderingDiagnostics *GaussianRenderingDiagnostics::get_singleton() {
    return singleton;
}

void GaussianRenderingDiagnostics::ensure_singleton() {
    if (!singleton) {
        singleton = memnew(GaussianRenderingDiagnostics);
    }
}

void GaussianRenderingDiagnostics::process_command_line_requests() {
    if (command_line_processed) {
        return;
    }
    command_line_processed = true;

    List<String> args = OS::get_singleton()->get_cmdline_args();
    for (List<String>::Element *E = args.front(); E; E = E->next()) {
        const String &arg = E->get();
        if (arg == "--diagnose-gaussian-rendering") {
            cli_diagnostic_requested = true;
            _ensure_validation_layers();
            GS_LOG_INFO_DEFAULT("[GaussianDiagnostics] Command line diagnostics requested (--diagnose-gaussian-rendering)");
            break;
        }
    }
}

void GaussianRenderingDiagnostics::register_renderer(GaussianSplatRenderer *p_renderer) {
    if (!p_renderer) {
        return;
    }
    if (registered_renderers.find(p_renderer) == -1) {
        registered_renderers.push_back(p_renderer);
    }
    if (cli_diagnostic_requested) {
        _emit_report_if_ready();
    }
}

void GaussianRenderingDiagnostics::unregister_renderer(GaussianSplatRenderer *p_renderer) {
    if (!p_renderer) {
        return;
    }
    int index = registered_renderers.find(p_renderer);
    if (index != -1) {
        registered_renderers.remove_at(index);
    }
}

void GaussianRenderingDiagnostics::notify_error(GaussianSplatRenderer *p_renderer, const RenderingError &p_error) {
    if (!p_renderer) {
        return;
    }
    Dictionary dict = p_error.to_dictionary();
    dict["renderer_id"] = reinterpret_cast<uint64_t>(p_renderer);
    _record_error_dictionary(dict);
    // PERFORMANCE FIX: Don't automatically enable diagnostics on error (causes CPU spikes)
    // cli_diagnostic_requested = true;
    // _ensure_validation_layers();
    // _emit_report_if_ready();
}

void GaussianRenderingDiagnostics::notify_recovery(GaussianSplatRenderer *p_renderer, const RenderingError &p_error) {
    if (!p_renderer) {
        return;
    }
    Dictionary dict = p_error.to_dictionary();
    dict["renderer_id"] = reinterpret_cast<uint64_t>(p_renderer);
    dict["recovered"] = true;
    _record_error_dictionary(dict);
    _emit_report_if_ready();
}

void GaussianRenderingDiagnostics::notify_frame_completed(GaussianSplatRenderer *p_renderer) {
    if (!p_renderer) {
        return;
    }
    if (cli_diagnostic_requested) {
        _emit_report_if_ready();
    }
}

void GaussianRenderingDiagnostics::request_runtime_report() {
    cli_diagnostic_requested = true;
    _emit_report_if_ready();
}

void GaussianRenderingDiagnostics::_ensure_validation_layers() {
    if (validation_layers_forced) {
        return;
    }
    validation_layers_forced = true;
    if (Engine *engine = Engine::get_singleton()) {
        engine->set_validation_layers_enabled(true);
    }
}

void GaussianRenderingDiagnostics::_emit_report_if_ready() {
	// PERFORMANCE FIX: Disable automatic reports (causes multi-hundred ms CPU spikes)
	// Only emit reports when explicitly requested via --diagnose-gaussian-rendering
	if (!cli_diagnostic_requested) {
		return;
	}
    if (cli_diagnostic_emitted) {
        uint64_t now = OS::get_singleton()->get_ticks_usec();
        if (now - last_report_time_usec < 5000000) { // 5 seconds throttle
            return;
        }
    }

    if (registered_renderers.is_empty()) {
        return;
    }

    _emit_report();
}

void GaussianRenderingDiagnostics::_emit_report() {
    // Throttle/disable reports based on logging config.
    // LOG_EVERY_FRAMES_VERBOSE == 0 means never emit automatically.
    if (LOG_EVERY_FRAMES_VERBOSE == 0 && !cli_diagnostic_requested) {
        return;
    }
    Array renderer_reports;
    for (GaussianSplatRenderer *renderer : registered_renderers) {
        if (!renderer) {
            continue;
        }
        renderer_reports.push_back(renderer->get_runtime_diagnostic_snapshot());
    }

    Dictionary report;
    report["timestamp_usec"] = static_cast<int64_t>(OS::get_singleton()->get_ticks_usec());
    report["renderer_reports"] = renderer_reports;
    Array recent_error_array;
    recent_error_array.resize(recent_errors.size());
    for (int i = 0; i < recent_errors.size(); i++) {
        recent_error_array[i] = recent_errors[i];
    }
    report["recent_errors"] = recent_error_array;

    const String report_json = JSON::stringify(report);
    if (!report_json.is_empty()) {
        GS_LOG_INFO_DEFAULT(vformat("[GaussianDiagnostics][CI] %s", report_json));
    }

    cli_diagnostic_emitted = true;
    last_report_time_usec = OS::get_singleton()->get_ticks_usec();
}

void GaussianRenderingDiagnostics::_record_error_dictionary(const Dictionary &p_dict) {
    recent_errors.push_back(p_dict);
    const int MAX_ERRORS = 32;
    while (recent_errors.size() > MAX_ERRORS) {
        recent_errors.remove_at(0);
    }
}
