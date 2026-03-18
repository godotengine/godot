#ifndef STREAMING_TELEMETRY_ADAPTER_H
#define STREAMING_TELEMETRY_ADAPTER_H

#include "core/variant/dictionary.h"
#include "core/string/ustring.h"
#include <cstdint>

class StreamingTelemetryAdapter {
public:
    struct QueuePressureSnapshot {
        bool active = false;
        String source = "none";
        String reason = "none";
        uint32_t backlog_depth = 0;
        bool pack_source_active = false;
        bool upload_source_active = false;
        bool sync_source_active = false;
    };

    static void apply_queue_pressure_analytics(Dictionary &r_analytics_snapshot, const QueuePressureSnapshot &p_snapshot);
    static void apply_queue_pressure_diagnostics(Dictionary &r_diagnostics_snapshot, const QueuePressureSnapshot &p_snapshot);
};

#endif // STREAMING_TELEMETRY_ADAPTER_H
