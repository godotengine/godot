#include "render_route_labels.h"

#include "render_debug_state_orchestrator.h"

namespace {

static String _normalize_route_uid(const String &p_route_uid) {
	return p_route_uid.is_empty() ? String(RenderRouteUID::COMMON_UNKNOWN_ROUTE) : p_route_uid;
}

static String _normalize_sort_route_uid(const String &p_sort_route_uid) {
	return p_sort_route_uid.is_empty() ? String(RenderRouteUID::COMMON_UNKNOWN_SORT_ROUTE) : p_sort_route_uid;
}

static String _normalize_cull_route_uid(const String &p_cull_route_uid) {
	return p_cull_route_uid.is_empty() ? String(RenderRouteUID::COMMON_UNKNOWN_ROUTE) : p_cull_route_uid;
}

static String _describe_streaming_not_ready_route(const String &p_route_uid) {
	static const String prefix = "COMMON.SKIP.STREAMING_NOT_READY.";
	if (!p_route_uid.begins_with(prefix)) {
		return String();
	}

	String suffix = p_route_uid.substr(prefix.length()).to_lower().replace("_", " ");
	if (suffix.is_empty()) {
		return "Skipped because streaming data is not ready";
	}
	return vformat("Skipped because streaming data is not ready (%s)", suffix);
}

static String _describe_common_route_uid(const String &p_route_uid) {
	const String streaming_not_ready = _describe_streaming_not_ready_route(p_route_uid);
	if (!streaming_not_ready.is_empty()) {
		return streaming_not_ready;
	}

	if (p_route_uid == String(RenderRouteUID::COMMON_UNSET_ROUTE)) {
		return "Route has not been assigned yet";
	}
	if (p_route_uid == String(RenderRouteUID::COMMON_UNKNOWN_ROUTE)) {
		return "Route is unknown";
	}
	if (p_route_uid == String(RenderRouteUID::COMMON_UNSET_SORT_ROUTE)) {
		return "Sort route has not been assigned yet";
	}
	if (p_route_uid == String(RenderRouteUID::COMMON_UNKNOWN_SORT_ROUTE)) {
		return "Sort route is unknown";
	}
	if (p_route_uid == String(RenderRouteUID::COMMON_SKIP_NO_DATA)) {
		return "Skipped because no Gaussian data is available";
	}
	if (p_route_uid == String(RenderRouteUID::COMMON_SKIP_NO_VISIBLE)) {
		return "Skipped because nothing is visible";
	}
	if (p_route_uid == String(RenderRouteUID::COMMON_SKIP_CAMERA_STABLE)) {
		return "Skipped because the camera and visibility stayed stable";
	}
	if (p_route_uid == String(RenderRouteUID::COMMON_SKIP_GPU_CULLER_UNAVAILABLE)) {
		return "Skipped because the GPU culler is unavailable";
	}
	if (p_route_uid == String(RenderRouteUID::COMMON_SKIP_LEGACY_GPU_DISABLED)) {
		return "Skipped because legacy GPU culling is disabled";
	}
	if (p_route_uid == String(RenderRouteUID::COMMON_SKIP_LEGACY_GPU_STREAMING_ACTIVE)) {
		return "Skipped because legacy GPU culling is incompatible with active streaming";
	}
	if (p_route_uid == String(RenderRouteUID::COMMON_SKIP_LEGACY_GPU_BUFFER_MISSING)) {
		return "Skipped because legacy GPU buffers are missing";
	}
	if (p_route_uid == String(RenderRouteUID::COMMON_FAIL_NO_DEVICE)) {
		return "Failed because no rendering device is available";
	}
	if (p_route_uid == String(RenderRouteUID::COMMON_FAIL_SORT_FAILED)) {
		return "Failed because sorting did not produce valid output";
	}
	if (p_route_uid == String(RenderRouteUID::COMMON_FAIL_NO_OUTPUT)) {
		return "Failed because rasterization produced no output";
	}

	return String();
}

static String _format_uid_with_description(const String &p_uid, const String &p_description) {
	if (p_description.is_empty()) {
		return p_uid;
	}
	return vformat("%s [%s]", p_description, p_uid);
}

static String _format_reason_with_description(const String &p_reason, const String &p_description) {
	if (p_reason.is_empty()) {
		return String();
	}
	if (p_description.is_empty()) {
		return p_reason;
	}
	return p_description;
}

static String _lowercase_first(const String &p_text) {
	if (p_text.is_empty()) {
		return p_text;
	}
	return p_text.substr(0, 1).to_lower() + p_text.substr(1);
}

static String _describe_submission_hint_source(const String &p_source) {
	if (p_source == "world_submission") {
		return "the world submission";
	}
	if (p_source == "instance_submission") {
		return "the instance submissions";
	}
	if (p_source == "mixed_instance_submissions") {
		return "mixed instance submissions";
	}
	if (p_source == "director_unavailable") {
		return "the scene director";
	}
	if (p_source == "none" || p_source.is_empty()) {
		return "a submission hint";
	}
	return p_source.replace("_", " ");
}

static String _describe_backend_reason_segment(const String &p_reason) {
	if (p_reason.is_empty()) {
		return String();
	}

	static const String not_feasible_separator = "_not_feasible:";
	const int not_feasible_at = p_reason.find(not_feasible_separator);
	if (not_feasible_at != -1) {
		const String base_reason = p_reason.substr(0, not_feasible_at);
		const String failure_reason = p_reason.substr(not_feasible_at + not_feasible_separator.length());
		const String base_description = _describe_backend_reason_segment(base_reason);
		const String failure_description = _describe_backend_reason_segment(failure_reason);
		if (!base_description.is_empty() && !failure_description.is_empty()) {
			return vformat("%s was not feasible because %s",
					base_description,
					_lowercase_first(failure_description));
		}
	}

	if (p_reason == "requested_resident_policy") {
		return "Resident was requested by the route policy";
	}
	if (p_reason == "requested_streaming_policy") {
		return "Streaming was requested by the route policy";
	}
	if (p_reason == "resident_contract_published") {
		return "Published the resident instance contract";
	}
	if (p_reason == "streaming_contract_published") {
		return "Published the streaming instance contract";
	}
	if (p_reason == "streaming_frame_not_ready_fallback") {
		return "Fell back to the resident path because the streaming frame was not ready";
	}
	if (p_reason == "streaming_unavailable_fallback") {
		return "Fell back to the resident path because streaming was unavailable";
	}
	if (p_reason == "resident_quantization_unsupported") {
		return "Quantized resident data cannot publish the resident instance contract";
	}
	if (p_reason == "no_render_data") {
		return "No Gaussian render data was available";
	}
	if (p_reason == "not_evaluated") {
		return "Backend selection has not been evaluated yet";
	}

	static const String resident_hint_prefix = "submission_hint_resident";
	if (p_reason.begins_with(resident_hint_prefix)) {
		const String source = p_reason.length() > resident_hint_prefix.length() + 1
				? p_reason.substr(resident_hint_prefix.length() + 1)
				: String();
		return vformat("Resident was requested by %s", _describe_submission_hint_source(source));
	}

	static const String streaming_hint_prefix = "submission_hint_streaming";
	if (p_reason.begins_with(streaming_hint_prefix)) {
		const String source = p_reason.length() > streaming_hint_prefix.length() + 1
				? p_reason.substr(streaming_hint_prefix.length() + 1)
				: String();
		return vformat("Streaming was requested by %s", _describe_submission_hint_source(source));
	}

	return String();
}

static String _describe_cull_reason_token(const String &p_reason) {
	if (p_reason.is_empty()) {
		return String();
	}

	static const String streaming_not_ready_prefix = "streaming_not_ready_";
	if (p_reason.begins_with(streaming_not_ready_prefix)) {
		const String suffix = p_reason.substr(streaming_not_ready_prefix.length()).to_lower().replace("_", " ");
		if (suffix.is_empty()) {
			return "Streaming data was not ready";
		}
		return vformat("Streaming data was not ready (%s)", suffix);
	}

	if (p_reason == "gpu_culler_unavailable") {
		return "GPU culling was unavailable";
	}
	if (p_reason == "missing_source_data") {
		return "Source Gaussian data was unavailable";
	}
	if (p_reason == "instance_pipeline_active") {
		return "Instance pipeline culling handled the frame";
	}
	if (p_reason == "instance_pipeline_failed") {
		return "Instance pipeline culling failed and CPU fallback ran";
	}
	if (p_reason == "instance_pipeline_failed_no_fallback") {
		return "Instance pipeline culling failed and no fallback source data was available";
	}
	if (p_reason == "global_cpu_path") {
		return "Global CPU culling handled the frame";
	}

	return String();
}

} // namespace

namespace GaussianRenderRouteLabels {

String describe_route_uid(const String &p_route_uid) {
	const String route_uid = _normalize_route_uid(p_route_uid);
	const String common = _describe_common_route_uid(route_uid);
	if (!common.is_empty()) {
		return common;
	}
	if (route_uid == String(RenderRouteUID::RESIDENT_SELECTED)) {
		return "Selected the resident backend";
	}
	if (route_uid == String(RenderRouteUID::INSTANCE_ENTRY_INSTANCED_FAST)) {
		return "Entered the shared instance pipeline";
	}
	if (route_uid == String(RenderRouteUID::INSTANCE_RESIDENT)) {
		return "Resident instanced path";
	}
	if (route_uid == String(RenderRouteUID::INSTANCE_STREAMING)) {
		return "Streaming instanced path";
	}
	if (route_uid == String(RenderRouteUID::INSTANCE_RASTER_COMPUTE)) {
		return "Rasterized through the compute path";
	}
	if (route_uid == String(RenderRouteUID::INSTANCE_RASTER_FRAGMENT)) {
		return "Rasterized through the fragment path";
	}
	if (route_uid == String(RenderRouteUID::INSTANCE_RASTER_CACHED)) {
		return "Reused cached raster output";
	}
	if (route_uid == String(RenderRouteUID::INSTANCE_RASTER_PAINTERLY)) {
		return "Rasterized through the painterly path";
	}
	return "Using an unrecognized render route";
}

String describe_sort_route_uid(const String &p_sort_route_uid) {
	const String sort_route_uid = _normalize_sort_route_uid(p_sort_route_uid);
	const String common = _describe_common_route_uid(sort_route_uid);
	if (!common.is_empty()) {
		return common;
	}
	if (sort_route_uid == String(RenderRouteUID::INSTANCE_SORT_GPU)) {
		return "Sorted visible splats on the GPU";
	}
	if (sort_route_uid == String(RenderRouteUID::INSTANCE_SORT_CPU_FALLBACK)) {
		return "Sorted visible splats on the CPU fallback path";
	}
	if (sort_route_uid == String(RenderRouteUID::INSTANCE_SORT_CACHED)) {
		return "Reused cached sort order";
	}
	if (sort_route_uid == String(RenderRouteUID::INSTANCE_SORT_IDENTITY_FALLBACK)) {
		return "Used the incoming cull order without a fresh sort";
	}
	return "Using an unrecognized sort route";
}

String describe_cull_route_uid(const String &p_cull_route_uid) {
	const String cull_route_uid = _normalize_cull_route_uid(p_cull_route_uid);
	const String common = _describe_common_route_uid(cull_route_uid);
	if (!common.is_empty()) {
		return common;
	}
	if (cull_route_uid == String(RenderRouteUID::INSTANCE_CULL_GPU)) {
		return "Culled splats with the GPU instance culler";
	}
	if (cull_route_uid == String(RenderRouteUID::INSTANCE_CULL_CPU_FALLBACK)) {
		return "Fell back to CPU culling for the instance pipeline";
	}
	if (cull_route_uid == String(RenderRouteUID::GLOBAL_CULL_CPU)) {
		return "Used the global CPU culling path";
	}
	return "Using an unrecognized cull route";
}

String describe_backend_selection_reason(const String &p_reason) {
	if (p_reason.is_empty()) {
		return String();
	}

	PackedStringArray segments = p_reason.split(" -> ");
	String description;
	for (int i = 0; i < segments.size(); i++) {
		const String segment_description = _describe_backend_reason_segment(segments[i]);
		const String normalized_segment = segment_description.is_empty() ? segments[i] : segment_description;
		if (description.is_empty()) {
			description = normalized_segment;
		} else {
			description += "; then " + _lowercase_first(normalized_segment);
		}
	}
	return description;
}

String describe_cull_route_reason(const String &p_reason) {
	const String description = _describe_cull_reason_token(p_reason);
	return description.is_empty() ? p_reason : description;
}

String format_route_uid(const String &p_route_uid) {
	const String route_uid = _normalize_route_uid(p_route_uid);
	return _format_uid_with_description(route_uid, describe_route_uid(route_uid));
}

String format_sort_route_uid(const String &p_sort_route_uid) {
	const String sort_route_uid = _normalize_sort_route_uid(p_sort_route_uid);
	return _format_uid_with_description(sort_route_uid, describe_sort_route_uid(sort_route_uid));
}

String format_cull_route_uid(const String &p_cull_route_uid) {
	const String cull_route_uid = _normalize_cull_route_uid(p_cull_route_uid);
	return _format_uid_with_description(cull_route_uid, describe_cull_route_uid(cull_route_uid));
}

String format_backend_selection_reason(const String &p_reason) {
	return _format_reason_with_description(p_reason, describe_backend_selection_reason(p_reason));
}

String format_cull_route_reason(const String &p_reason) {
	return _format_reason_with_description(p_reason, describe_cull_route_reason(p_reason));
}

} // namespace GaussianRenderRouteLabels
