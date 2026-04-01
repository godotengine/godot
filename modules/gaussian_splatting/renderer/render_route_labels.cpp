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
	if (route_uid == String(RenderRouteUID::INSTANCE_STREAMING)) {
		return "Selected the streaming backend";
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

} // namespace GaussianRenderRouteLabels
