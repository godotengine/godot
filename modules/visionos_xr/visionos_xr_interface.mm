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

#include "core/input/input.h"
#include "core/os/os.h"
#include "drivers/metal/metal_objects.h"
#include "platform/visionos/godot_app_delegate_service_visionos.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering/rendering_server_globals.h"

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
}

VisionOSXRInterface::VisionOSXRInterface() {}

VisionOSXRInterface::~VisionOSXRInterface() {
	// and make sure we cleanup if we haven't already
	if (is_initialized()) {
		uninitialize();
	};
}

StringName VisionOSXRInterface::get_name() const {
	return VisionOSXRInterface::name;
}

uint32_t VisionOSXRInterface::get_capabilities() const {
	return XRInterface::XR_VR + XRInterface::XR_AR + XRInterface::XR_STEREO;
}

uint32_t VisionOSXRInterface::get_view_count() {
	return 2;
}

XRInterface::TrackingStatus VisionOSXRInterface::get_tracking_status() const {
	return tracking_state;
}

bool VisionOSXRInterface::is_initialized() const {
	return (initialized);
}

bool VisionOSXRInterface::initialize() {
	print_verbose("VisionOSXRInterface.initialize()");

	if (initialized) {
		ERR_PRINT("VisionOSXRInterface already initialized");
		return true;
	}

	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, false);

	String driver_name = OS::get_singleton()->get_current_rendering_driver_name().to_lower();
	ERR_FAIL_COND_V_MSG(driver_name != "metal", false, "The visionOS XR interface requires the Metal rendering driver.");

	GDTRenderMode app_delegate_render_mode = GDTAppDelegateServiceVisionOS.renderMode;
	ERR_FAIL_COND_V_MSG(app_delegate_render_mode != GDTRenderModeCompositorServices, false, "The visionOS XR interface requires GDTRenderModeCompositorServices render mode.");

	layer_renderer = GDTAppDelegateServiceVisionOS.layerRenderer;
	layer_renderer_capabilities = GDTAppDelegateServiceVisionOS.layerRendererCapabilities;

	ERR_FAIL_NULL_V_MSG(layer_renderer, false, "GDTAppDelegateServiceVisionOS.layerRenderer not set");
	ERR_FAIL_NULL_V_MSG(layer_renderer_capabilities, false, "GDTAppDelegateServiceVisionOS.layerRendererCapabilities not set");

	rendering_device = RenderingDevice::get_singleton();
	rendering_device_driver_metal = (RenderingDeviceDriverMetal *)rendering_device->get_device_driver();
	pixel_formats = &rendering_device_driver_metal->get_pixel_formats();

	// ARKit session initialization
	ar_session = ar_session_create();
	ar_world_tracking_configuration_t world_tracking_configuration = ar_world_tracking_configuration_create();
	world_tracking_provider = ar_world_tracking_provider_create(world_tracking_configuration);
	current_device_anchor = ar_device_anchor_create();
	ar_data_providers_t data_providers = ar_data_providers_create();
	ar_data_providers_add_data_provider(data_providers, world_tracking_provider);
	ar_session_run(ar_session, data_providers);

	current_drawable = nullptr;

	// Head tracker initialization
	origin_from_head_simd = matrix_identity_float4x4;

	head_tracker.instantiate();
	head_tracker->set_tracker_type(XRServer::TRACKER_HEAD);
	head_tracker->set_tracker_name("head");
	head_tracker->set_tracker_desc("Device head pose");
	xr_server->add_tracker(head_tracker);

	// Make this our primary interface
	xr_server->set_primary_interface(this);

	initialized = true;
	return initialized;
}

void VisionOSXRInterface::uninitialize() {
	if (!initialized) {
		return;
	}

	if (current_color_texture_id != RID()) {
		rendering_device->texture_owner.free(current_color_texture_id);
	}
	if (current_depth_texture_id != RID()) {
		rendering_device->texture_owner.free(current_depth_texture_id);
	}
	if (current_rasterization_rate_map_id != RID()) {
		rendering_device->texture_owner.free(current_rasterization_rate_map_id);
	}

	XRServer *xr_server = XRServer::get_singleton();
	if (xr_server != nullptr) {
		if (head_tracker.is_valid()) {
			xr_server->remove_tracker(head_tracker);
			head_tracker.unref();
		}

		if (xr_server->get_primary_interface() == this) {
			// no longer our primary interface
			xr_server->set_primary_interface(nullptr);
		}

		initialized = false;
	}
}

Dictionary VisionOSXRInterface::get_system_info() {
	Dictionary dict;

	dict[SNAME("XRRuntimeName")] = String("Godot visionOS XR interface");
	dict[SNAME("XRRuntimeVersion")] = String("1.0");

	return dict;
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

Transform3D VisionOSXRInterface::get_camera_transform() {
	_THREAD_SAFE_METHOD_

	Transform3D camera_transform;
	if (!initialized) {
		return camera_transform;
	}
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, camera_transform);
	// scale our origin point of our transform
	float world_scale = xr_server->get_world_scale();
	Transform3D origin_from_head = MTL::simd_to_transform3D(origin_from_head_simd);
	origin_from_head.origin *= world_scale;
	camera_transform = origin_from_head;
	return camera_transform;
}

Transform3D VisionOSXRInterface::get_transform_for_view(uint32_t p_view, const Transform3D &p_cam_transform) {
	_THREAD_SAFE_METHOD_

	Transform3D origin_from_eye;
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, origin_from_eye);
	if (initialized) {
		ERR_FAIL_COND_V(p_view > get_view_count(), origin_from_eye);
		ERR_FAIL_NULL_V_MSG(current_drawable, origin_from_eye, "Current drawable is nil, probably pre_render() has not been called, using identity transform");

		Transform3D origin_from_head = MTL::simd_to_transform3D(origin_from_head_simd);

		cp_view_t view = cp_drawable_get_view(current_drawable, p_view);
		simd_float4x4 head_from_eye_simd = cp_view_get_transform(view);
		Transform3D head_from_eye = MTL::simd_to_transform3D(head_from_eye_simd);

		origin_from_eye = origin_from_head * head_from_eye;

		// Scale origin point by XROrigin3D's World Scale attribute
		float world_scale = xr_server->get_world_scale();
		origin_from_eye.origin *= world_scale;
	} else {
		ERR_PRINT("vision_vr_interface not initialized, returning received camera transform");
		origin_from_eye = Transform3D();
	};
	Transform3D reference_frame = xr_server->get_reference_frame();
	return p_cam_transform * reference_frame * origin_from_eye;
}

Projection VisionOSXRInterface::get_projection_for_view(uint32_t p_view, double p_aspect, double p_z_near, double p_z_far) {
	_THREAD_SAFE_METHOD_

	Projection eye_projection;
	if (!initialized) {
		return eye_projection;
	}

	ERR_FAIL_COND_V(p_view > get_view_count(), eye_projection);
	ERR_FAIL_NULL_V_MSG(current_drawable, eye_projection, "Current drawable is nil, probably pre_render() has not been called");

	float minimum_supported_near_plane = cp_layer_renderer_capabilities_supported_minimum_near_plane_distance(layer_renderer_capabilities);

	XRServer *xr_server = XRServer::get_singleton();
	float world_scale = xr_server->get_world_scale();

	double scaled_z_far = p_z_far / world_scale;
	double scaled_z_near = p_z_near / world_scale;

	ERR_FAIL_COND_V_MSG(scaled_z_near < minimum_supported_near_plane, eye_projection, "Your XRCamera3D Near value is lower than the minimum value supported by the visionOS platform. Make sure that Near divided by XROrigin's World Scale is higher or equal than the value returned by LayerRender.Capabilities.supportedMinimumNearPlaneDistance. This value is 0.1 for Apple Vision Pro.");

	simd_float2 depth_range = simd_make_float2(scaled_z_far, scaled_z_near);
	cp_drawable_set_depth_range(current_drawable, depth_range);
	simd_float4x4 eye_simd_projection = cp_drawable_compute_projection(current_drawable, cp_axis_direction_convention_right_up_forward, p_view);
	eye_projection = MTL::simd_to_projection(eye_simd_projection);

	// Godot renderers work in the normalized [-1, 1] depth space, and they do a final z remap of the projection matrixes to the [0, 1] depth space in RenderSceneDataRD::update_ubo().
	// Compositor Services projection matrices are already in the [0, 1] depth space, so we need to apply the inverse z remap before passing them to the renderer.
	Projection correction;
	correction.set_depth_correction(false, false, true);
	eye_projection = correction.inverse() * eye_projection;

	return eye_projection;
}

// The render region is the logical texture size. With foveated rendering, it's bigger than the
// physical texture size. This value is equivalent to rasterizationRateMap.screenSize.
Rect2i VisionOSXRInterface::get_render_region() {
	_THREAD_SAFE_METHOD_

	Rect2 viewport_rect;
	if (!initialized) {
		return viewport_rect;
	}

	ERR_FAIL_NULL_V_MSG(current_drawable, viewport_rect, "Current drawable is nil, probably pre_render() has not been called");

	// The viewport should be the same for both eyes, so only get it from the first view
	cp_view_t view = cp_drawable_get_view(current_drawable, 0);
	cp_view_texture_map_t view_texture_map = cp_view_get_view_texture_map(view);
	MTLViewport viewport = cp_view_texture_map_get_viewport(view_texture_map);
	viewport_rect = MTL::rect_from_mtl_viewport(viewport);
	return viewport_rect;
}

Size2 VisionOSXRInterface::get_render_target_size() {
	Size2 target_size;
	if (!initialized) {
		return target_size;
	}
	ERR_FAIL_NULL_V_MSG(current_drawable, target_size, "Current drawable is nil, probably pre_render() has not been called");
	id<MTLTexture> color_texture = cp_drawable_get_color_texture(current_drawable, 0);
	target_size = Size2(color_texture.width, color_texture.height);
	return target_size;
}

void VisionOSXRInterface::set_head_pose_from_arkit(bool p_use_drawable) {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_NULL_MSG(current_frame, "Current frame is nil, probably process() has not been called, using identity transform");
	if (p_use_drawable) {
		ERR_FAIL_NULL_MSG(current_drawable, "Current drawable is nil, probably process() has not been called, using identity transform");
	}

	cp_frame_timing_t frame_timing;
	if (p_use_drawable) {
		frame_timing = cp_drawable_get_frame_timing(current_drawable);
	} else {
		frame_timing = cp_frame_predict_timing(current_frame);
	}
	CFTimeInterval presentation_time = cp_time_to_cf_time_interval(cp_frame_timing_get_presentation_time(frame_timing));
	ar_device_anchor_query_status_t query_anchor_result = ar_world_tracking_provider_query_device_anchor_at_timestamp(world_tracking_provider, presentation_time, current_device_anchor);

	if (query_anchor_result != ar_device_anchor_query_status_success) {
		tracking_state = XRInterface::XR_NOT_TRACKING;
		tracking_confidence = XRPose::XR_TRACKING_CONFIDENCE_NONE;
		ERR_FAIL_MSG("cannot query device anchor, result: " + itos(query_anchor_result));
	}

	origin_from_head_simd = ar_anchor_get_origin_from_anchor_transform(current_device_anchor);
	tracking_state = XRInterface::XR_NORMAL_TRACKING;
	tracking_confidence = XRPose::XR_TRACKING_CONFIDENCE_HIGH;

	if (head_tracker.is_valid()) {
		// Set our head position (in real space, reference frame and world scale is applied later)
		head_tracker->set_pose("default", MTL::simd_to_transform3D(origin_from_head_simd), Vector3(), Vector3(), tracking_confidence);
	}
}

void VisionOSXRInterface::process() {
	if (!initialized) {
		return;
	}
	current_frame = cp_layer_renderer_query_next_frame(layer_renderer);

	// Set head pose before engine update, so scripts can access fresh head tracker data
	set_head_pose_from_arkit(false);

	cp_frame_start_update(current_frame);
}

void VisionOSXRInterface::pre_render() {
	_THREAD_SAFE_METHOD_

	if (!initialized) {
		return;
	}
	ERR_FAIL_NULL_MSG(current_frame, "Current frame is nil, probably process() has not been called");
	cp_frame_end_update(current_frame);

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
	ERR_FAIL_NULL_MSG(current_drawable, "Built-in drawable not found, aborting");

	// Set head pose again to get closer presentation time prediction
	set_head_pose_from_arkit(true);

	if (current_device_anchor != nil) {
		cp_drawable_set_device_anchor(current_drawable, current_device_anchor);
	} else {
		ERR_PRINT("Current device anchor is nil, will present drawable without a device anchor");
	}
}

Vector<BlitToScreen> VisionOSXRInterface::post_draw_viewport(RID p_render_target, const Rect2 &p_screen_rect) {
	_THREAD_SAFE_METHOD_
	// We're overriding the color and depth textures, no need for screen blits, return empty BlitToScreen vector
	// However, we need to acquire the dummy frame buffer
	RD::get_singleton()->screen_prepare_for_drawing(DisplayServer::MAIN_WINDOW_ID);
	return Vector<BlitToScreen>();
}

void VisionOSXRInterface::encode_present(MDCommandBuffer *p_cmd_buffer) {
	_THREAD_SAFE_METHOD_

	if (!initialized) {
		return;
	}
	ERR_FAIL_NULL_MSG(current_drawable, "Current drawable is nil, probably process() has not been called");
	cp_drawable_encode_present(current_drawable, p_cmd_buffer->get_command_buffer());
}

void VisionOSXRInterface::end_frame() {
	_THREAD_SAFE_METHOD_

	if (!initialized) {
		return;
	}
	ERR_FAIL_NULL_MSG(current_frame, "Current frame is nil, probably process() has not been called");
	cp_frame_end_submission(current_frame);
}

RID VisionOSXRInterface::get_color_texture() {
	_THREAD_SAFE_METHOD_

	if (!initialized) {
		return RID();
	}

	if (current_color_texture_id != RID()) {
		rendering_device->texture_owner.free(current_color_texture_id);
	}

	ERR_FAIL_NULL_V_MSG(current_drawable, RID(), "Current drawable is nil, probably pre_render() has not been called");

	id<MTLTexture> color_texture = cp_drawable_get_color_texture(current_drawable, 0);
	current_color_texture_id = rendering_device->texture_create_from_extension(
			MTL::texture_type_from_metal(color_texture.textureType),
			pixel_formats->getDataFormat(color_texture.pixelFormat),
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

RID VisionOSXRInterface::get_depth_texture() {
	_THREAD_SAFE_METHOD_

	if (!initialized) {
		return RID();
	}

	if (current_depth_texture_id != RID()) {
		rendering_device->texture_owner.free(current_depth_texture_id);
	}

	ERR_FAIL_NULL_V_MSG(current_drawable, RID(), "Current drawable is nil, probably pre_render() has not been called");
	id<MTLTexture> depth_texture = cp_drawable_get_depth_texture(current_drawable, 0);

	current_depth_texture_id = rendering_device->texture_create_from_extension(
			MTL::texture_type_from_metal(depth_texture.textureType),
			pixel_formats->getDataFormat(depth_texture.pixelFormat),
			MTL::texture_samples_from_metal(depth_texture.sampleCount),
			RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT,
			(uint64_t)depth_texture,
			depth_texture.width,
			depth_texture.height,
			depth_texture.depth,
			depth_texture.arrayLength,
			depth_texture.mipmapLevelCount);

	return current_depth_texture_id;
}

RID VisionOSXRInterface::get_vrs_texture() {
	_THREAD_SAFE_METHOD_

	if (!initialized) {
		return RID();
	}

	if (current_rasterization_rate_map_id != RID()) {
		rendering_device->texture_owner.free(current_rasterization_rate_map_id);
	}

	ERR_FAIL_NULL_V_MSG(current_drawable, RID(), "Current drawable is nil, probably pre_render() has not been called");
	size_t count = cp_drawable_get_rasterization_rate_map_count(current_drawable);
	ERR_FAIL_COND_V_MSG(count == 0, RID(), "No rasterizationRateMaps found");
	id<MTLRasterizationRateMap> rasterization_rate_map = cp_drawable_get_rasterization_rate_map(current_drawable, 0);
	MTLSize logical_size = rasterization_rate_map.screenSize;

	RD::Texture texture;
	texture.driver_id = RDD::TextureID((__bridge void *)rasterization_rate_map);
	texture.usage_flags = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT | RD::TEXTURE_USAGE_STORAGE_BIT | RD::TEXTURE_USAGE_VRS_ATTACHMENT_BIT;
	texture.width = logical_size.width;
	texture.height = logical_size.height;
	texture.layers = rasterization_rate_map.layerCount;
	// The following spoofed values are unused, but they are required
	// to pass RenderingDevice::_render_pass_create() validation
	texture.type = RDD::TEXTURE_TYPE_2D_ARRAY;
	texture.format = RDD::DATA_FORMAT_R8_UINT;
	texture.samples = RDD::TEXTURE_SAMPLES_1;
	texture.depth = 1;
	texture.mipmaps = 1;
	ERR_FAIL_COND_V(!texture.driver_id, RID());

	current_rasterization_rate_map = texture;
	current_rasterization_rate_map_id = rendering_device->texture_owner.make_rid(current_rasterization_rate_map);

	return current_rasterization_rate_map_id;
}

VisionOSXRInterface::VRSTextureFormat VisionOSXRInterface::get_vrs_texture_format() {
	return XR_VRS_TEXTURE_FORMAT_RASTERIZATION_RATE_MAP;
}

#endif // VISIONOS_ENABLED
