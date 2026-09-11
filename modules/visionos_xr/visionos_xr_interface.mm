/**************************************************************************/
/*  visionos_xr_interface.mm                                              */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifdef VISIONOS_ENABLED

#include "visionos_xr_interface.h"

#include "visionos_simd_helpers.h"

#include "core/config/project_settings.h"
#include "core/error/error_macros.h"
#include "core/input/input.h"
#include "core/math/transform_3d.h"
#include "core/object/callable_mp.h"
#include "core/object/class_db.h"
#include "core/os/os.h"
#include "core/os/thread.h"
#include "core/string/print_string.h"
#include "drivers/metal/metal3_objects.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering/rendering_server.h" // ERR_NOT_ON_RENDER_THREAD_V
#include "servers/rendering/rendering_server_globals.h"
#include "servers/rendering/rendering_server_types.h"
#include "servers/xr/xr_server.h"

#include "platform/visionos/godot_app_delegate_service_visionos.h"

const String VisionOSXRInterface::name = "visionOS";

StringName VisionOSXRInterface::get_signal_name(SignalEnum p_signal) {
	switch (p_signal) {
		case VISIONOS_XR_SIGNAL_SESSION_STARTED:
			return SNAME("session_started");
			break;
		case VISIONOS_XR_SIGNAL_SESSION_PAUSED:
			return SNAME("session_paused");
			break;
		case VISIONOS_XR_SIGNAL_SESSION_RESUMED:
			return SNAME("session_resumed");
			break;
		case VISIONOS_XR_SIGNAL_SESSION_INVALIDATED:
			return SNAME("session_invalidated");
			break;
		case VISIONOS_XR_SIGNAL_POSE_RECENTERED:
			return SNAME("pose_recentered");
			break;
		default:
			return "";
			break;
	}
}

void VisionOSXRInterface::emit_signal_enum(SignalEnum p_signal) {
	emit_signal(get_signal_name(p_signal));
}

void VisionOSXRInterface::_bind_methods() {
	// Signals
	for (int i = 0; i < VISIONOS_XR_SIGNAL_MAX; i++) {
		ADD_SIGNAL(MethodInfo(get_signal_name((SignalEnum)i)));
	}

	ClassDB::bind_method(D_METHOD("get_current_render_quality"), &VisionOSXRInterface::get_current_render_quality);
	ClassDB::bind_method(D_METHOD("set_current_render_quality", "render_quality"), &VisionOSXRInterface::set_current_render_quality);
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "current_render_quality"), "set_current_render_quality", "get_current_render_quality");

	BIND_ENUM_CONSTANT(IMMERSION_STYLE_FULL);
	BIND_ENUM_CONSTANT(IMMERSION_STYLE_MIXED);
	BIND_ENUM_CONSTANT(IMMERSION_STYLE_PROGRESSIVE);
	ClassDB::bind_method(D_METHOD("get_immersion_style"), &VisionOSXRInterface::get_immersion_style);
	ClassDB::bind_method(D_METHOD("set_immersion_style", "immersion_style"), &VisionOSXRInterface::set_immersion_style);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "immersion_style", PROPERTY_HINT_ENUM, "Full,Mixed,Progressive"), "set_immersion_style", "get_immersion_style");

	BIND_ENUM_CONSTANT(VISIBILITY_AUTOMATIC);
	BIND_ENUM_CONSTANT(VISIBILITY_VISIBLE);
	BIND_ENUM_CONSTANT(VISIBILITY_HIDDEN);
	ClassDB::bind_method(D_METHOD("get_upper_limb_visibility"), &VisionOSXRInterface::get_upper_limb_visibility);
	ClassDB::bind_method(D_METHOD("set_upper_limb_visibility", "upper_limb_visibility"), &VisionOSXRInterface::set_upper_limb_visibility);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "upper_limb_visibility", PROPERTY_HINT_ENUM, "Automatic,Visible,Hidden"), "set_upper_limb_visibility", "get_upper_limb_visibility");

	ClassDB::bind_method(D_METHOD("get_persistent_system_overlays"), &VisionOSXRInterface::get_persistent_system_overlays);
	ClassDB::bind_method(D_METHOD("set_persistent_system_overlays", "persistent_system_overlays"), &VisionOSXRInterface::set_persistent_system_overlays);
	ADD_PROPERTY(PropertyInfo(Variant::INT, "persistent_system_overlays", PROPERTY_HINT_ENUM, "Automatic,Visible,Hidden"), "set_persistent_system_overlays", "get_persistent_system_overlays");
}

VisionOSXRInterface::VisionOSXRInterface() {}

VisionOSXRInterface::~VisionOSXRInterface() {
	if (is_initialized()) {
		uninitialize();
	}
}

StringName VisionOSXRInterface::get_name() const {
	return VisionOSXRInterface::name;
}

uint32_t VisionOSXRInterface::get_capabilities() const {
	return XRInterface::XR_VR + XRInterface::XR_AR + XRInterface::XR_STEREO;
}

XRInterface::TrackingStatus VisionOSXRInterface::get_tracking_status() const {
	return tracking_state;
}

bool VisionOSXRInterface::is_initialized() const {
	return initialized;
}

void VisionOSXRInterface::RenderThread::set_world_tracking_provider(uint64_t p_world_tracking_provider) {
	this->world_tracking_provider = (__bridge ar_world_tracking_provider_t)(void *)p_world_tracking_provider;
}

bool VisionOSXRInterface::initialize() {
	ERR_FAIL_COND_V_MSG(initialized, true, "VisionOSXRInterface already initialized.");

	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, false);

	// Checking features
	GDTRenderMode app_delegate_render_mode = GDTAppDelegateServiceVisionOS.renderMode;
	cs.enabled = (app_delegate_render_mode == GDTRenderModeCompositorServices);
	hands.enabled = GLOBAL_GET("xr/visionos/enable_hand_tracking");
	controllers.enabled = GLOBAL_GET("xr/visionos/enable_controller_tracking");

	// ARKit session
	ar_session = ar_session_create();

	// CompositorServices
	if (cs.enabled) {
		cs.initialize(xr_server);

		// RenderThread
		rendering_server = RenderingServer::get_singleton();
		ERR_FAIL_NULL_V(rendering_server, false);
		rendering_server->call_on_render_thread(callable_mp(&rt, &RenderThread::initialize));
		rendering_server->call_on_render_thread(callable_mp(&rt, &RenderThread::set_world_tracking_provider).bind((uint64_t)(__bridge void *)cs.world_tracking_provider));

		float minimum_supported_near_plane = cp_layer_renderer_capabilities_supported_minimum_near_plane_distance(cs.layer_renderer_capabilities);
		rendering_server->call_on_render_thread(callable_mp(&rt, &RenderThread::set_minimum_supported_near_plane).bind(minimum_supported_near_plane));

		rendering_server->call_on_render_thread(callable_mp(&rt, &RenderThread::prepare_screen));

		// Make this our primary interface, since it's used for rendering
		xr_server->set_primary_interface(this);
	}

	// Hand tracking
	if (hands.enabled) {
		hands.initialize(xr_server);
	}

	// Controllers
	if (controllers.enabled) {
		controllers.initialize(xr_server, this);
	}

	// Running the ARKit session for head tracking, at first
	run_ar_session();

	// Check the authorizations asynchronously
	if (hands.enabled || controllers.enabled) {
		update_authorizations_async();
	}

	initialized = true;

	print_verbose(String("VisionOSXRInterface initialized with:") + " compositorservices=" + (cs.enabled ? "yes" : "no") + " hands=" + (hands.enabled ? "yes" : "no") + " controllers=" + (controllers.enabled ? "yes" : "no"));

	return initialized;
}

bool VisionOSXRInterface::CompositorServicesData::initialize(XRServer *p_xr_server) {
	String driver_name = OS::get_singleton()->get_current_rendering_driver_name().to_lower();
	ERR_FAIL_COND_V_MSG(driver_name != "metal", false, "The visionOS XR interface requires the Metal rendering driver.");

	layer_renderer = GDTAppDelegateServiceVisionOS.layerRenderer;
	layer_renderer_capabilities = GDTAppDelegateServiceVisionOS.layerRendererCapabilities;

	ERR_FAIL_NULL_V_MSG(layer_renderer, false, "GDTAppDelegateServiceVisionOS.layerRenderer not set.");
	ERR_FAIL_NULL_V_MSG(layer_renderer_capabilities, false, "GDTAppDelegateServiceVisionOS.layerRendererCapabilities not set.");

	ar_world_tracking_configuration_t world_tracking_configuration = ar_world_tracking_configuration_create();
	world_tracking_provider = ar_world_tracking_provider_create(world_tracking_configuration);
	current_device_anchor = ar_device_anchor_create();

	// Head tracker initialization
	head_tracker.instantiate();
	head_tracker->set_tracker_type(XRServer::TRACKER_HEAD);
	head_tracker->set_tracker_name("head");
	head_tracker->set_tracker_desc("Device head pose");
	p_xr_server->add_tracker(head_tracker);

	return true;
}

void VisionOSXRInterface::uninitialize() {
	if (!initialized) {
		return;
	}

	if (cs.enabled && rendering_server) {
		rendering_server->call_on_render_thread(callable_mp(&rt, &RenderThread::uninitialize));
	}

	XRServer *xr_server = XRServer::get_singleton();
	if (xr_server != nullptr) {
		if (controllers.enabled) {
			controllers.uninitialize(xr_server);
		}

		if (hands.enabled) {
			if (hands.left_hand_tracker.is_valid()) {
				xr_server->remove_tracker(hands.left_hand_tracker);
				hands.left_hand_tracker.unref();
			}
			if (hands.right_hand_tracker.is_valid()) {
				xr_server->remove_tracker(hands.right_hand_tracker);
				hands.right_hand_tracker.unref();
			}
		}

		if (cs.enabled) {
			if (cs.head_tracker.is_valid()) {
				xr_server->remove_tracker(cs.head_tracker);
				cs.head_tracker.unref();
			}

			if (xr_server->get_primary_interface() == this) {
				// no longer our primary interface
				xr_server->set_primary_interface(nullptr);
			}
		}

		initialized = false;
	}

	// equivalent to "ar_release(ar_session)" since automatic reference counting is enabled
	ar_session = nullptr;
}

void VisionOSXRInterface::RenderThread::initialize() {
	ERR_NOT_ON_RENDER_THREAD;
	rendering_device = RenderingDevice::get_singleton();
	RenderingDeviceDriverMetal *rendering_device_driver_metal = (RenderingDeviceDriverMetal *)rendering_device->get_device_driver();
	pixel_formats = &rendering_device_driver_metal->get_pixel_formats();

	current_device_anchor = ar_device_anchor_create();

	initialized = true;
}

void VisionOSXRInterface::RenderThread::prepare_screen() {
	ERR_NOT_ON_RENDER_THREAD;

	// Trigger the swap-chain resize so the format is initialized; must happen outside any submission.
	rendering_device->screen_prepare_for_drawing(DisplayServerEnums::MAIN_WINDOW_ID);
}

void VisionOSXRInterface::RenderThread::uninitialize() {
	ERR_NOT_ON_RENDER_THREAD;
	if (current_color_texture_id != RID()) {
		rendering_device->free_rid(current_color_texture_id);
	}
	if (current_depth_texture_id != RID()) {
		rendering_device->free_rid(current_depth_texture_id);
	}
	if (current_rasterization_rate_map_id != RID()) {
		rendering_device->free_rid(current_rasterization_rate_map_id);
	}
	initialized = false;
}

void VisionOSXRInterface::update_layer_renderer(cp_layer_renderer_t p_layer_renderer, cp_layer_renderer_capabilities_t p_layer_renderer_capabilities) {
	cs.layer_renderer = p_layer_renderer;
	cs.layer_renderer_capabilities = p_layer_renderer_capabilities;

	float minimum_supported_near_plane = cp_layer_renderer_capabilities_supported_minimum_near_plane_distance(cs.layer_renderer_capabilities);
	rendering_server->call_on_render_thread(callable_mp(&rt, &RenderThread::set_minimum_supported_near_plane).bind(minimum_supported_near_plane));
}

Dictionary VisionOSXRInterface::get_system_info() {
	Dictionary dict;

	dict[SNAME("XRRuntimeName")] = String("Godot visionOS XR interface");
	dict[SNAME("XRRuntimeVersion")] = String("1.0");

	return dict;
}

VisionOSXRInterface::VRSTextureFormat VisionOSXRInterface::get_vrs_texture_format() {
	return XR_VRS_TEXTURE_FORMAT_RASTERIZATION_RATE_MAP;
}

bool VisionOSXRInterface::supports_play_area_mode(XRInterface::PlayAreaMode p_mode) {
	return p_mode == XR_PLAY_AREA_ROOMSCALE;
}

XRInterface::PlayAreaMode VisionOSXRInterface::get_play_area_mode() const {
	return XR_PLAY_AREA_ROOMSCALE;
}

bool VisionOSXRInterface::set_play_area_mode(XRInterface::PlayAreaMode p_mode) {
	return p_mode == XR_PLAY_AREA_ROOMSCALE;
}

float VisionOSXRInterface::get_current_render_quality() {
	return cp_layer_renderer_get_render_quality(cs.layer_renderer);
}

void VisionOSXRInterface::set_current_render_quality(float p_render_quality) {
	ERR_FAIL_COND_MSG(!GDTAppDelegateServiceVisionOS.isDynamicRenderQualityEnabled, "Attempting to set current render quality but Dynamic Render Quality has not been enabled in Project Settings.");
	float maxRenderQuality = GDTAppDelegateServiceVisionOS.maxRenderQuality;
	ERR_FAIL_COND_MSG(p_render_quality > GDTAppDelegateServiceVisionOS.maxRenderQuality, vformat("Attempting to set a current render quality higher than the Max Render Quality configured in Project Settings (%f).", maxRenderQuality));
	cp_layer_renderer_set_render_quality(cs.layer_renderer, p_render_quality);
}

VisionOSXRInterface::ImmersionStyle VisionOSXRInterface::get_immersion_style() {
	switch (GDTAppDelegateServiceVisionOS.immersionStyle) {
		case GDTImmersionStyleFull:
			return IMMERSION_STYLE_FULL;
		case GDTImmersionStyleMixed:
			return IMMERSION_STYLE_MIXED;
		case GDTImmersionStyleProgressive:
			return IMMERSION_STYLE_PROGRESSIVE;
		default:
			return IMMERSION_STYLE_FULL;
	}
}

void VisionOSXRInterface::set_immersion_style(ImmersionStyle p_immersion_style) {
	switch (p_immersion_style) {
		case IMMERSION_STYLE_FULL:
			GDTAppDelegateServiceVisionOS.immersionStyle = GDTImmersionStyleFull;
			break;
		case IMMERSION_STYLE_MIXED:
			GDTAppDelegateServiceVisionOS.immersionStyle = GDTImmersionStyleMixed;
			break;
		case IMMERSION_STYLE_PROGRESSIVE:
			GDTAppDelegateServiceVisionOS.immersionStyle = GDTImmersionStyleProgressive;
			break;
	}
}

VisionOSXRInterface::Visibility VisionOSXRInterface::get_upper_limb_visibility() {
	switch (GDTAppDelegateServiceVisionOS.upperLimbVisibility) {
		case GDTVisibilityAutomatic:
			return VISIBILITY_AUTOMATIC;
		case GDTVisibilityVisible:
			return VISIBILITY_VISIBLE;
		case GDTVisibilityHidden:
			return VISIBILITY_HIDDEN;
		default:
			return VISIBILITY_AUTOMATIC;
	}
}

void VisionOSXRInterface::set_upper_limb_visibility(Visibility p_upper_limb_visibility) {
	switch (p_upper_limb_visibility) {
		case VISIBILITY_AUTOMATIC:
			GDTAppDelegateServiceVisionOS.upperLimbVisibility = GDTVisibilityAutomatic;
			break;
		case VISIBILITY_VISIBLE:
			GDTAppDelegateServiceVisionOS.upperLimbVisibility = GDTVisibilityVisible;
			break;
		case VISIBILITY_HIDDEN:
			GDTAppDelegateServiceVisionOS.upperLimbVisibility = GDTVisibilityHidden;
			break;
	}
}

VisionOSXRInterface::Visibility VisionOSXRInterface::get_persistent_system_overlays() {
	switch (GDTAppDelegateServiceVisionOS.persistentSystemOverlays) {
		case GDTVisibilityAutomatic:
			return VISIBILITY_AUTOMATIC;
		case GDTVisibilityVisible:
			return VISIBILITY_VISIBLE;
		case GDTVisibilityHidden:
			return VISIBILITY_HIDDEN;
		default:
			return VISIBILITY_AUTOMATIC;
	}
}

void VisionOSXRInterface::set_persistent_system_overlays(Visibility p_persistent_system_overlays) {
	switch (p_persistent_system_overlays) {
		case VISIBILITY_AUTOMATIC:
			GDTAppDelegateServiceVisionOS.persistentSystemOverlays = GDTVisibilityAutomatic;
			break;
		case VISIBILITY_VISIBLE:
			GDTAppDelegateServiceVisionOS.persistentSystemOverlays = GDTVisibilityVisible;
			break;
		case VISIBILITY_HIDDEN:
			GDTAppDelegateServiceVisionOS.persistentSystemOverlays = GDTVisibilityHidden;
			break;
	}
}

void VisionOSXRInterface::set_head_pose_from_arkit() {
	ERR_FAIL_NULL_MSG(cs.current_frame, "Current frame is nil, process() has probably not been called, using identity transform.");

	cs.current_timing = cp_frame_predict_timing(cs.current_frame);

	CFTimeInterval presentation_time = cp_time_to_cf_time_interval(cp_frame_timing_get_presentation_time(cs.current_timing));
	ar_device_anchor_query_status_t query_anchor_result = ar_world_tracking_provider_query_device_anchor_at_timestamp(cs.world_tracking_provider, presentation_time, cs.current_device_anchor);

	if (query_anchor_result != ar_device_anchor_query_status_success) {
		tracking_state = XRInterface::XR_NOT_TRACKING;
		ERR_FAIL_MSG("Cannot query device anchor, result: " + itos(query_anchor_result) + ".");
	}

	simd_float4x4 origin_from_head_simd = ar_anchor_get_origin_from_anchor_transform(cs.current_device_anchor);
	tracking_state = XRInterface::XR_NORMAL_TRACKING;

	if (cs.head_tracker.is_valid()) {
		// Set our head position (in real space, reference frame and world scale is applied later)
		cs.head_tracker->set_pose("default", MTL::simd_to_transform3D(origin_from_head_simd), Vector3(), Vector3(), XRPose::XR_TRACKING_CONFIDENCE_HIGH);
	}
}

namespace {
VisionOSAuthorizationStatus convert(ar_authorization_status_t p_status) {
	switch (p_status) {
		case ar_authorization_status_not_determined:
			return VisionOSAuthorizationStatus::NOT_DETERMINED;
		case ar_authorization_status_allowed:
			return VisionOSAuthorizationStatus::ALLOWED;
		case ar_authorization_status_denied:
			return VisionOSAuthorizationStatus::DENIED;
	}
	// Unknown/future cases
	return VisionOSAuthorizationStatus::NOT_DETERMINED;
}
} // namespace

void VisionOSXRInterface::update_authorizations_async() {
	uintptr_t types = ar_authorization_type_none;

	if (hands.enabled) {
		types |= ar_authorization_type_hand_tracking;
	}

	if (controllers.enabled) {
		types |= ar_authorization_type_accessory_tracking;
	}

	if (types == ar_authorization_type_none) {
		// Nothing to request
		return;
	}

	Ref<VisionOSXRInterface> ref_this = this;
	ar_session_request_authorization(ar_session, static_cast<ar_authorization_type_t>(types), ^(ar_authorization_results_t authorization_results, ar_error_t _Nullable error) {
		dispatch_async(dispatch_get_main_queue(), ^(void) {
			ERR_FAIL_COND_MSG(error != nullptr, "Could not query ARKit authorizations.");

			if (ref_this->is_initialized()) {
				ref_this->update_from_authorizations(authorization_results);
			}
		});
	});
}

void VisionOSXRInterface::update_from_authorizations(ar_authorization_results_t p_authorization_results) {
	VisionOSAuthorizationStatus previous_hands = hands.authorization;
	VisionOSAuthorizationStatus previous_controllers = controllers.authorization;

	ar_authorization_results_enumerate_results(p_authorization_results, ^bool(ar_authorization_result_t authorization_result) {
		ar_authorization_type_t type = ar_authorization_result_get_authorization_type(authorization_result);
		ar_authorization_status_t status = ar_authorization_result_get_status(authorization_result);

		switch (type) {
			case ar_authorization_type_hand_tracking:
				hands.authorization = convert(status);
				if (status == ar_authorization_status_denied) {
					ERR_PRINT("Hand tracking not authorized. Enable it in `Settings > Privacy` and restart the app.");
				}
				break;
			case ar_authorization_type_accessory_tracking:
				controllers.authorization = convert(status);
				if (status == ar_authorization_status_denied) {
					ERR_PRINT("Controller tracking not authorized. Enable it in `Settings > Privacy` and restart the app.");
				}
				break;
			default:
				break;
		}

		return true; // continue with the enumeration
	});

	// If something changed, re-run the ARKit session with updated authorizations
	if (previous_hands != hands.authorization || previous_controllers != controllers.authorization) {
		run_ar_session();
	}
}

void VisionOSXRInterface::run_ar_session() {
	ar_data_providers_t ar_data_providers = ar_data_providers_create();

	if (cs.enabled) {
		ar_data_providers_add_data_provider(ar_data_providers, cs.world_tracking_provider);
	}

	if (hands.active()) {
		ar_data_providers_add_data_provider(ar_data_providers, hands.hand_tracking_provider);
	}

	if (controllers.active()) {
		ar_data_providers_add_data_provider(ar_data_providers, controllers.accessory_tracking_provider);
	}

	// Running the ARSession with the given providers, after it has been configured
	ar_session_run(ar_session, ar_data_providers);
}

CFTimeInterval VisionOSXRInterface::get_trackable_anchor_time() {
	// Computing the time to use for pose prediction
	CFTimeInterval trackable_anchor_time = 0;

	if (cs.enabled) {
		// If CompositorServices is enabled, use its presentation time for pose prediction
		trackable_anchor_time = cp_time_to_cf_time_interval(cp_frame_timing_get_trackable_anchor_time(cs.current_timing));
	} else {
		// If not using CompositorServices, we obtain the estimatedPresentationTime from the active UIScene
		UIWindowScene *window_scene = nil;
		for (UIScene *scene in UIApplication.sharedApplication.connectedScenes) {
			if ([scene isKindOfClass:[UIWindowScene class]]) {
				UIWindowScene *window_scene_candidate = (UIWindowScene *)scene;
				if (window_scene_candidate.activationState == UISceneActivationStateForegroundActive) {
					window_scene = window_scene_candidate;
					break;
				}
			}
		}
		if (window_scene != nil) {
			UIUpdateInfo *ui_update_info = [UIUpdateInfo currentUpdateInfoForWindowScene:window_scene];
			trackable_anchor_time = ui_update_info.estimatedPresentationTime;
		}
	}

	return trackable_anchor_time;
}

void VisionOSXRInterface::process() {
	if (!initialized) {
		return;
	}

	if (cs.enabled) {
		cs.current_frame = cp_layer_renderer_query_next_frame(cs.layer_renderer);

		ERR_FAIL_NULL_MSG(cs.current_frame, "Layer renderer unexpectedly returned a nil frame, the layer renderer has probably been invalidated and it hasn't been updated to a new one.");

		// Set head pose before engine update, so scripts can access fresh head tracker data
		set_head_pose_from_arkit();

		rendering_server->call_on_render_thread(callable_mp(&rt, &RenderThread::set_current_frame).bind((uint64_t)cs.current_frame));
		rendering_server->call_on_render_thread(callable_mp(&rt, &RenderThread::start_frame_update));
	}

	if (hands.active() || controllers.active()) {
		CFTimeInterval trackable_anchor_time = get_trackable_anchor_time();

		if (hands.active()) {
			hands.update_hand_trackers_from_arkit(trackable_anchor_time);
		}

		if (controllers.active()) {
			controllers.update_controller_trackers_from_arkit(trackable_anchor_time);
		}
	}
}

Size2 VisionOSXRInterface::get_render_target_size() {
	return rt.get_render_target_size();
}

void VisionOSXRInterface::RenderThread::set_minimum_supported_near_plane(float p_minimum_supported_near_plane) {
	ERR_NOT_ON_RENDER_THREAD;
	minimum_supported_near_plane = p_minimum_supported_near_plane;
}

void VisionOSXRInterface::RenderThread::set_current_frame(uint64_t p_current_frame) {
	ERR_NOT_ON_RENDER_THREAD;
	current_frame = (cp_frame_t)p_current_frame;

	// Query anchor again from the render thread
	cp_frame_timing_t current_timing = cp_frame_predict_timing(current_frame);
	CFTimeInterval presentation_time = cp_time_to_cf_time_interval(cp_frame_timing_get_presentation_time(current_timing));
	ar_device_anchor_query_status_t query_anchor_result = ar_world_tracking_provider_query_device_anchor_at_timestamp(world_tracking_provider, presentation_time, current_device_anchor);

	if (query_anchor_result != ar_device_anchor_query_status_success) {
		ERR_FAIL_MSG("Cannot query device anchor, result: " + itos(query_anchor_result) + ".");
	}

	simd_float4x4 origin_from_head_simd = ar_anchor_get_origin_from_anchor_transform(current_device_anchor);
	origin_from_head = MTL::simd_to_transform3D(origin_from_head_simd);
}

uint32_t VisionOSXRInterface::RenderThread::get_view_count() {
	// No need for ERR_NOT_ON_RENDER_THREAD
	return 2;
}

Transform3D VisionOSXRInterface::RenderThread::get_camera_transform() {
	Transform3D camera_transform;
	ERR_NOT_ON_RENDER_THREAD_V(camera_transform);

	if (!initialized) {
		return camera_transform;
	}

	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, camera_transform);
	// scale our origin point of our transform
	float world_scale = xr_server->get_world_scale();
	camera_transform = origin_from_head;
	camera_transform.origin *= world_scale;
	return camera_transform;
}

Transform3D VisionOSXRInterface::RenderThread::get_transform_for_view(uint32_t p_view, const Transform3D &p_cam_transform) {
	Transform3D origin_from_eye;
	ERR_NOT_ON_RENDER_THREAD_V(origin_from_eye);

	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, origin_from_eye);
	if (initialized) {
		ERR_FAIL_COND_V(p_view > get_view_count(), origin_from_eye);
		ERR_FAIL_NULL_V_MSG(current_drawable, origin_from_eye, "Current drawable is nil, pre_render() has probably not been called, using identity transform.");

		cp_view_t view = cp_drawable_get_view(current_drawable, p_view);
		simd_float4x4 head_from_eye_simd = cp_view_get_transform(view);
		Transform3D head_from_eye = MTL::simd_to_transform3D(head_from_eye_simd);

		origin_from_eye = origin_from_head * head_from_eye;

		// Scale origin point by XROrigin3D's World Scale attribute
		float world_scale = xr_server->get_world_scale();
		origin_from_eye.origin *= world_scale;
	} else {
		ERR_PRINT("vision_vr_interface not initialized, returning received camera transform.");
		origin_from_eye = Transform3D();
	}
	Transform3D reference_frame = xr_server->get_reference_frame();
	return p_cam_transform * reference_frame * origin_from_eye;
}

Projection VisionOSXRInterface::RenderThread::get_projection_for_view(uint32_t p_view, double p_aspect, double p_z_near, double p_z_far) {
	Projection eye_projection;
	ERR_NOT_ON_RENDER_THREAD_V(eye_projection);

	if (!initialized) {
		return eye_projection;
	}

	ERR_FAIL_COND_V(p_view > get_view_count(), eye_projection);
	ERR_FAIL_NULL_V_MSG(current_drawable, eye_projection, "Current drawable is nil, pre_render() has probably not been called.");

	XRServer *xr_server = XRServer::get_singleton();
	float world_scale = xr_server->get_world_scale();

	double scaled_z_far = p_z_far / world_scale;
	double scaled_z_near = p_z_near / world_scale;

	ERR_FAIL_COND_V_MSG(scaled_z_near < minimum_supported_near_plane, eye_projection, "Your XRCamera3D Near value is lower than the minimum value supported by the visionOS platform. Make sure that Near divided by XROrigin's World Scale is higher than or equal to the value returned by LayerRender.Capabilities.supportedMinimumNearPlaneDistance. This value is 0.1 for Apple Vision Pro.");

	simd_float2 depth_range = simd_make_float2(scaled_z_far, scaled_z_near);
	cp_drawable_set_depth_range(current_drawable, depth_range);
	simd_float4x4 eye_simd_projection = cp_drawable_compute_projection(current_drawable, cp_axis_direction_convention_right_up_forward, p_view);
	eye_projection = MTL::simd_to_projection(eye_simd_projection);

	// Godot renderers work in the normalized [-1, 1] depth space, and they do a final z remap of the projection matrixes to the [0, 1] depth space in RenderSceneDataRD::update_ubo().
	// Compositor Services projection matrices are already in the [0, 1] depth space, so we need to apply the inverse z remap before passing them to the renderer.
	Projection normalized_depth_correction;
	normalized_depth_correction.set_depth_correction(false, false, true);

	// Correct depth by world_scale
	Projection reverse_z;
	real_t *m = &reverse_z.columns[0][0];
	m[10] = -1.0;
	m[14] = 1.0;

	Projection world_scale_correction;
	world_scale_correction.make_scale(Vector3(1, 1, world_scale));

	eye_projection = normalized_depth_correction.inverse() * reverse_z.inverse() * world_scale_correction * reverse_z * eye_projection;
	return eye_projection;
}

// The render region is the logical texture size. With foveated rendering, it's bigger than the
// physical texture size. This value is equivalent to rasterizationRateMap.screenSize.
Rect2i VisionOSXRInterface::RenderThread::get_render_region() {
	Rect2 viewport_rect;

	ERR_NOT_ON_RENDER_THREAD_V(viewport_rect);

	if (!initialized) {
		return viewport_rect;
	}

	ERR_FAIL_NULL_V_MSG(current_drawable, viewport_rect, "Current drawable is nil, pre_render() has probably not been called.");

	// The viewport should be the same for both eyes, so only get it from the first view
	cp_view_t view = cp_drawable_get_view(current_drawable, 0);
	cp_view_texture_map_t view_texture_map = cp_view_get_view_texture_map(view);
	MTLViewport viewport = cp_view_texture_map_get_viewport(view_texture_map);
	viewport_rect = MTL::rect_from_mtl_viewport(viewport);
	return viewport_rect;
}

Size2 VisionOSXRInterface::RenderThread::get_render_target_size() {
	// Read atomic values cached by pre_render().
	return Size2(cached_render_target_width.get(), cached_render_target_height.get());
}

void VisionOSXRInterface::RenderThread::start_frame_update() {
	ERR_NOT_ON_RENDER_THREAD;

	if (!initialized) {
		return;
	}

	ERR_FAIL_NULL_MSG(current_frame, "Current frame is nil, process() has probably not been called.");
	cp_frame_start_update(current_frame);
}

void VisionOSXRInterface::RenderThread::end_frame_update() {
	ERR_NOT_ON_RENDER_THREAD;

	if (!initialized) {
		return;
	}

	ERR_FAIL_NULL_MSG(current_frame, "Current frame is nil, process() has probably not been called.");
	cp_frame_end_update(current_frame);
}

void VisionOSXRInterface::RenderThread::pre_render() {
	ERR_NOT_ON_RENDER_THREAD;

	if (!initialized) {
		return;
	}

	end_frame_update();

	cp_frame_timing_t timing = cp_frame_predict_timing(current_frame);
	cp_time_wait_until(cp_frame_timing_get_optimal_input_time(timing));

	cp_frame_start_submission(current_frame);
	cp_drawable_array_t drawables = cp_frame_query_drawables(current_frame);
	size_t drawable_count = cp_drawable_array_get_count(drawables);

	for (size_t i = 0; i < drawable_count; i++) {
		cp_drawable_t drawable = cp_drawable_array_get_drawable(drawables, i);
		// Find screen drawable (target = cp_drawable_target_built_in).
		// High quality recording (target = cp_drawable_target_capture) not supported yet,
		// to support this feature, we'd need Godot to perform an additional render pass on the extra drawable
		if (cp_drawable_get_target(drawable) == cp_drawable_target_built_in) {
			current_drawable = drawable;
		}
	}
	ERR_FAIL_NULL_MSG(current_drawable, "Built-in drawable not found, aborting.");

	// Cache the render target size so it can be read from the game thread.
	id<MTLTexture> color_texture = cp_drawable_get_color_texture(current_drawable, 0);
	cached_render_target_width.set(color_texture.width);
	cached_render_target_height.set(color_texture.height);

	if (current_device_anchor != nil) {
		cp_drawable_set_device_anchor(current_drawable, current_device_anchor);
	} else {
		ERR_PRINT("Current device anchor is nil, will present drawable without a device anchor.");
	}
}

Vector<RenderingServerTypes::BlitToScreen> VisionOSXRInterface::RenderThread::post_draw_viewport(RID p_render_target, const Rect2 &p_screen_rect) {
	ERR_NOT_ON_RENDER_THREAD_V(Vector<RenderingServerTypes::BlitToScreen>());

	if (!initialized) {
		return Vector<RenderingServerTypes::BlitToScreen>();
	}

	// We're overriding the color and depth textures, no need for screen blits, return empty BlitToScreen vector
	// However, we need to acquire the dummy frame buffer
	RD::get_singleton()->screen_prepare_for_drawing(DisplayServerEnums::MAIN_WINDOW_ID);
	return Vector<RenderingServerTypes::BlitToScreen>();
}

// Wraps cp_drawable_encode_present in a drawable render context with a no-op load/store pass,
// which Compositor Services requires whenever the layer supports progressive immersion.
void VisionOSXRInterface::RenderThread::encode_drawable_no_op_and_present(cp_drawable_t p_drawable, cp_frame_t p_frame, void *p_command_buffer) {
	id<MTLCommandBuffer> command_buffer = (__bridge id<MTLCommandBuffer>)p_command_buffer;
	// A nil command buffer makes the Compositor Services API fail.
	ERR_FAIL_NULL_MSG(command_buffer, "Command buffer is nil, cannot add a drawable render context.");
	cp_drawable_render_context_t drawable_render_context = cp_drawable_add_render_context(p_drawable, command_buffer);

	id<MTLTexture> color_texture = cp_drawable_get_color_texture(p_drawable, 0);
	id<MTLTexture> depth_texture = cp_drawable_get_depth_texture(p_drawable, 0);

	MTLRenderPassDescriptor *render_pass_descriptor = [MTLRenderPassDescriptor renderPassDescriptor];
	render_pass_descriptor.colorAttachments[0].texture = color_texture;
	render_pass_descriptor.colorAttachments[0].loadAction = MTLLoadActionLoad;
	render_pass_descriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
	// Compositor Services' compositing pipeline has no stencil attachment.
	render_pass_descriptor.depthAttachment.texture = depth_texture;
	render_pass_descriptor.depthAttachment.loadAction = MTLLoadActionLoad;
	render_pass_descriptor.depthAttachment.storeAction = MTLStoreActionStore;
	render_pass_descriptor.renderTargetArrayLength = cp_frame_get_drawable_target_view_count(p_frame, cp_drawable_get_target(p_drawable));
	size_t count = cp_drawable_get_rasterization_rate_map_count(p_drawable);
	if (count > 0) {
		id<MTLRasterizationRateMap> rasterization_rate_map = cp_drawable_get_rasterization_rate_map(p_drawable, 0);
		MTLSize logical_size = rasterization_rate_map.screenSize;
		render_pass_descriptor.rasterizationRateMap = rasterization_rate_map;
		render_pass_descriptor.renderTargetWidth = logical_size.width;
		render_pass_descriptor.renderTargetHeight = logical_size.height;
	}

	id<MTLRenderCommandEncoder> command_encoder = [command_buffer renderCommandEncoderWithDescriptor:render_pass_descriptor];

	cp_drawable_render_context_end_encoding(drawable_render_context, command_encoder);

	cp_drawable_encode_present(p_drawable, command_buffer);
}

void VisionOSXRInterface::RenderThread::encode_present(MTL3::MDCommandBuffer *p_cmd_buffer) {
	ERR_NOT_ON_RENDER_THREAD;

	if (!initialized) {
		return;
	}

	ERR_FAIL_NULL_MSG(current_drawable, "Current drawable is nil, process() has probably not been called.");
	encode_drawable_no_op_and_present(current_drawable, current_frame, p_cmd_buffer->get_command_buffer());
	current_drawable = nullptr;
}

void VisionOSXRInterface::RenderThread::end_frame() {
	ERR_NOT_ON_RENDER_THREAD;

	if (!initialized) {
		return;
	}

	ERR_FAIL_NULL_MSG(current_frame, "Current frame is nil, process() has probably not been called.");
	cp_frame_end_submission(current_frame);
	current_frame = nullptr;
}

RID VisionOSXRInterface::RenderThread::get_color_texture() {
	ERR_NOT_ON_RENDER_THREAD_V(RID());

	if (!initialized) {
		return RID();
	}

	if (current_color_texture_id != RID()) {
		rendering_device->free_rid(current_color_texture_id);
	}

	ERR_FAIL_NULL_V_MSG(current_drawable, RID(), "Current drawable is nil, pre_render() has probably not been called.");

	id<MTLTexture> color_texture = cp_drawable_get_color_texture(current_drawable, 0);
	current_color_texture_id = rendering_device->texture_create_from_extension(
			MTL::texture_type_from_metal(color_texture.textureType),
			pixel_formats->getDataFormat((MTL::PixelFormat)color_texture.pixelFormat),
			MTL::texture_samples_from_metal(color_texture.sampleCount),
			RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT,
			(uint64_t)color_texture,
			color_texture.width,
			color_texture.height,
			color_texture.depth,
			color_texture.arrayLength,
			color_texture.mipmapLevelCount);

	return current_color_texture_id;
}

RID VisionOSXRInterface::RenderThread::get_depth_texture() {
	ERR_NOT_ON_RENDER_THREAD_V(RID());

	if (!initialized) {
		return RID();
	}

	if (current_depth_texture_id != RID()) {
		rendering_device->free_rid(current_depth_texture_id);
	}

	ERR_FAIL_NULL_V_MSG(current_drawable, RID(), "Current drawable is nil, pre_render() has probably not been called.");
	id<MTLTexture> depth_texture = cp_drawable_get_depth_texture(current_drawable, 0);

	current_depth_texture_id = rendering_device->texture_create_from_extension(
			MTL::texture_type_from_metal(depth_texture.textureType),
			pixel_formats->getDataFormat((MTL::PixelFormat)depth_texture.pixelFormat),
			MTL::texture_samples_from_metal(depth_texture.sampleCount),
			RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_DEPTH_RESOLVE_ATTACHMENT_BIT,
			(uint64_t)depth_texture,
			depth_texture.width,
			depth_texture.height,
			depth_texture.depth,
			depth_texture.arrayLength,
			depth_texture.mipmapLevelCount);

	return current_depth_texture_id;
}

RID VisionOSXRInterface::RenderThread::get_vrs_texture() {
	ERR_NOT_ON_RENDER_THREAD_V(RID());

	if (!initialized) {
		return RID();
	}

	if (current_rasterization_rate_map_id != RID()) {
		rendering_device->free_rid(current_rasterization_rate_map_id);
	}

	ERR_FAIL_NULL_V_MSG(current_drawable, RID(), "Current drawable is nil, pre_render() has probably not been called.");
	size_t count = cp_drawable_get_rasterization_rate_map_count(current_drawable);
	ERR_FAIL_COND_V_MSG(count == 0, RID(), "No rasterizationRateMaps found.");
	id<MTLRasterizationRateMap> rasterization_rate_map = cp_drawable_get_rasterization_rate_map(current_drawable, 0);
	MTLSize logical_size = rasterization_rate_map.screenSize;

	// The type, format and sample count are spoofed. They satisfy
	// RenderingDevice::_render_pass_create() validation and have no other use.
	current_rasterization_rate_map_id = rendering_device->texture_create_from_extension(
			RD::TEXTURE_TYPE_2D_ARRAY,
			RD::DATA_FORMAT_R8_UINT,
			RD::TEXTURE_SAMPLES_1,
			RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_VRS_ATTACHMENT_BIT,
			(uint64_t)(__bridge void *)rasterization_rate_map,
			logical_size.width,
			logical_size.height,
			1,
			rasterization_rate_map.layerCount,
			1);

	return current_rasterization_rate_map_id;
}

void VisionOSXRInterface::trigger_haptic_pulse(const String &p_action_name, const StringName &p_tracker_name, double p_frequency, double p_amplitude, double p_duration_sec, double p_delay_sec) {
	if (controllers.enabled) {
		controllers.trigger_haptic_pulse(p_action_name, p_tracker_name, p_frequency, p_amplitude, p_duration_sec, p_delay_sec);
	}
}

#endif // VISIONOS_ENABLED
