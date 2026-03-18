#include "streaming_telemetry_adapter.h"
#include "core/variant/variant.h"

namespace {

static void _apply_queue_pressure_common(Dictionary &r_snapshot,
        const StreamingTelemetryAdapter::QueuePressureSnapshot &p_snapshot) {
    r_snapshot[Variant("queue_pressure_active")] = p_snapshot.active;
    r_snapshot[Variant("queue_pressure_source")] = p_snapshot.source;
    r_snapshot[Variant("queue_pressure_reason")] = p_snapshot.reason;
    r_snapshot[Variant("queue_pressure_backlog_depth")] = static_cast<int64_t>(p_snapshot.backlog_depth);
    r_snapshot[Variant("queue_pressure_pack_source_active")] = p_snapshot.pack_source_active;
    r_snapshot[Variant("queue_pressure_upload_source_active")] = p_snapshot.upload_source_active;
    r_snapshot[Variant("queue_pressure_sync_source_active")] = p_snapshot.sync_source_active;
}

} // namespace

void StreamingTelemetryAdapter::apply_queue_pressure_analytics(
        Dictionary &r_analytics_snapshot, const QueuePressureSnapshot &p_snapshot) {
    _apply_queue_pressure_common(r_analytics_snapshot, p_snapshot);
}

void StreamingTelemetryAdapter::apply_queue_pressure_diagnostics(
        Dictionary &r_diagnostics_snapshot, const QueuePressureSnapshot &p_snapshot) {
    _apply_queue_pressure_common(r_diagnostics_snapshot, p_snapshot);
}
