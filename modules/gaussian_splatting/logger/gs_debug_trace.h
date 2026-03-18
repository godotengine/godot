#ifndef GS_DEBUG_TRACE_H
#define GS_DEBUG_TRACE_H

#include "core/math/color.h"
#include "core/typedefs.h"
#include "core/string/ustring.h"
#include "core/variant/variant.h"

namespace GaussianSplatting {

bool debug_trace_is_enabled();
void debug_trace_begin_frame(uint64_t p_frame_id);

void debug_trace_record_event(const String &p_category, const String &p_message, bool p_is_error = false);

void debug_trace_record_pack_sh(const Color &p_sh_dc, float p_opacity);
void debug_trace_record_pack_range(uint32_t p_count, uint32_t p_start, uint32_t p_src_size,
		const Color &p_sh_dc, float p_opacity);
void debug_trace_record_gaussians_check(uint32_t p_asset_id, uint32_t p_total, uint32_t p_chunk_start, uint32_t p_chunk_count,
		const Color &p_sh_dc, float p_opacity);
void debug_trace_record_buffer_mgr(uint32_t p_total, const Color &p_sh_dc, float p_opacity);
void debug_trace_record_chunk_diagnostics(uint32_t p_chunk_idx, uint32_t p_zero_scale, uint32_t p_zero_opacity,
		uint32_t p_nan_pos, uint32_t p_total);
void debug_trace_record_instance_buffer(uint32_t p_out_count, uint32_t p_world_count, uint32_t p_skip_count);
void debug_trace_record_instance_counts(uint32_t p_visible_clamped, uint32_t p_visible_raw, uint32_t p_overflow_flag);
void debug_trace_record_instance_flags(uint32_t p_total, uint32_t p_rotation_identity, uint32_t p_scale_identity,
		uint32_t p_translation_zero, uint32_t p_fully_identity);

Dictionary debug_trace_get_data_flow_snapshot();
Array debug_trace_get_recent_events();

} // namespace GaussianSplatting

#endif // GS_DEBUG_TRACE_H
