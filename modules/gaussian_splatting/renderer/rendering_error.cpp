#include "rendering_error.h"

#include "core/io/json.h"
#include "core/templates/hash_map.h"

RenderingError::RenderingError(Code p_code, Severity p_severity, const String &p_message) {
    code = p_code;
    severity = p_severity;
    message = p_message;
    timestamp_usec = OS::get_singleton() ? OS::get_singleton()->get_ticks_usec() : 0;
}

void RenderingError::add_context(const StringName &p_key, const Variant &p_value) {
    context[p_key] = p_value;
}

void RenderingError::add_recovery_step(const String &p_description) {
    recovery_steps.push_back(p_description);
}

String RenderingError::category_to_string(Category p_category) {
    switch (p_category) {
        case Category::DEVICE:
            return "DEVICE";
        case Category::TEXTURE:
            return "TEXTURE";
        case Category::COMMAND_BUFFER:
            return "COMMAND_BUFFER";
        case Category::PIPELINE:
            return "PIPELINE";
        case Category::SYNCHRONIZATION:
            return "SYNCHRONIZATION";
        case Category::STREAMING:
            return "STREAMING";
        case Category::SHADER:
            return "SHADER";
        case Category::PERFORMANCE:
            return "PERFORMANCE";
        case Category::UNKNOWN:
        default:
            return "UNKNOWN";
    }
}

String RenderingError::severity_to_string(Severity p_severity) {
    switch (p_severity) {
        case Severity::INFO:
            return "INFO";
        case Severity::WARNING:
            return "WARNING";
        case Severity::RECOVERABLE:
            return "RECOVERABLE";
        case Severity::FATAL:
            return "FATAL";
        default:
            return "UNKNOWN";
    }
}

RenderingError::Code RenderingError::make_code(Category p_category, int p_id, const char *p_label) {
    RenderingError::Code result;
    result.category = p_category;
    result.id = p_id;
    result.label = StringName(p_label ? p_label : "UNKNOWN");
    return result;
}

String RenderingError::to_string() const {
    Dictionary payload = to_dictionary();
    // Avoid costly JSON stringify in hot paths; return concise string by default.
    return vformat("%s:%d [%s] %s", category_to_string(code.category), code.id, severity_to_string(severity), message);
}

Dictionary RenderingError::to_dictionary() const {
    Dictionary dict;
    dict["code"] = code.id;
    dict["label"] = code.label;
    dict["category"] = category_to_string(code.category);
    dict["severity"] = severity_to_string(severity);
    dict["message"] = message;
    dict["timestamp_usec"] = static_cast<int64_t>(timestamp_usec);
    dict["context"] = context;
    Array recovery_array;
    for (const String &step : recovery_steps) {
        recovery_array.push_back(step);
    }
    dict["recovery_steps"] = recovery_array;
    return dict;
}

namespace RenderingErrorCodes {

static RenderingError::Code _make_cached(RenderingError::Category p_category, int p_id, const char *p_label) {
    static HashMap<uint64_t, RenderingError::Code> cache;
    uint64_t key = (static_cast<uint64_t>(p_category) << 32) | static_cast<uint64_t>(p_id);
    if (RenderingError::Code *existing = cache.getptr(key)) {
        return *existing;
    }
    RenderingError::Code code = RenderingError::make_code(p_category, p_id, p_label);
    cache.insert(key, code);
    return code;
}

RenderingError::Code device_unavailable() {
    return _make_cached(RenderingError::Category::DEVICE, 100, "DEVICE_UNAVAILABLE");
}

RenderingError::Code submission_device_unavailable() {
    return _make_cached(RenderingError::Category::DEVICE, 101, "SUBMISSION_DEVICE_UNAVAILABLE");
}

RenderingError::Code compute_pipeline_initialization_failed() {
    return _make_cached(RenderingError::Category::PIPELINE, 200, "COMPUTE_PIPELINE_INIT_FAILED");
}

RenderingError::Code compute_buffer_prepare_failed() {
    return _make_cached(RenderingError::Category::PIPELINE, 201, "COMPUTE_BUFFER_PREP_FAILED");
}

RenderingError::Code command_buffer_synchronization_failed() {
    return _make_cached(RenderingError::Category::SYNCHRONIZATION, 300, "COMMAND_BUFFER_SYNC_FAILED");
}

RenderingError::Code texture_tracking_lost() {
    return _make_cached(RenderingError::Category::TEXTURE, 400, "TEXTURE_TRACKING_LOST");
}

RenderingError::Code framebuffer_validation_failed() {
    return _make_cached(RenderingError::Category::PIPELINE, 500, "FRAMEBUFFER_VALIDATION_FAILED");
}

RenderingError::Code gpu_sort_unavailable() {
    return _make_cached(RenderingError::Category::PIPELINE, 210, "GPU_SORT_UNAVAILABLE");
}

RenderingError::Code runtime_diagnostics_forced() {
    return _make_cached(RenderingError::Category::PERFORMANCE, 900, "RUNTIME_DIAGNOSTICS_FORCED");
}

} // namespace RenderingErrorCodes
