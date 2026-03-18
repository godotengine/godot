#ifndef GAUSSIAN_RENDERING_ERROR_H
#define GAUSSIAN_RENDERING_ERROR_H

#include "core/object/object.h"
#include "core/os/os.h"
#include "core/string/string_name.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"

#include <utility>

class RenderingError {
public:
    enum class Category {
        DEVICE,
        TEXTURE,
        COMMAND_BUFFER,
        PIPELINE,
        SYNCHRONIZATION,
        STREAMING,
        SHADER,
        PERFORMANCE,
        UNKNOWN
    };

    enum class Severity {
        INFO,
        WARNING,
        RECOVERABLE,
        FATAL
    };

    struct Code {
        Category category = Category::UNKNOWN;
        int id = 0;
        StringName label = StringName("UNKNOWN");
    };

    RenderingError() = default;
    RenderingError(Code p_code, Severity p_severity, const String &p_message);

    const Code &get_code() const { return code; }
    Category get_category() const { return code.category; }
    Severity get_severity() const { return severity; }
    const String &get_message() const { return message; }
    uint64_t get_timestamp_usec() const { return timestamp_usec; }
    const Dictionary &get_context() const { return context; }
    const Vector<String> &get_recovery_steps() const { return recovery_steps; }

    void add_context(const StringName &p_key, const Variant &p_value);
    void add_recovery_step(const String &p_description);
    String to_string() const;
    Dictionary to_dictionary() const;

    static String category_to_string(Category p_category);
    static String severity_to_string(Severity p_severity);
    static Code make_code(Category p_category, int p_id, const char *p_label);

private:
    Code code;
    Severity severity = Severity::WARNING;
    String message;
    Dictionary context;
    Vector<String> recovery_steps;
    uint64_t timestamp_usec = 0;
};

namespace RenderingErrorCodes {
RenderingError::Code device_unavailable();
RenderingError::Code submission_device_unavailable();
RenderingError::Code compute_pipeline_initialization_failed();
RenderingError::Code compute_buffer_prepare_failed();
RenderingError::Code command_buffer_synchronization_failed();
RenderingError::Code texture_tracking_lost();
RenderingError::Code framebuffer_validation_failed();
RenderingError::Code gpu_sort_unavailable();
RenderingError::Code runtime_diagnostics_forced();
} // namespace RenderingErrorCodes

#endif // GAUSSIAN_RENDERING_ERROR_H
