/**************************************************************************/
/*  visionos_vr_interface.mm                                              */
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

#include "visionos_vr_interface.h"

#include "core/input/input.h"
#include "core/os/os.h"
#include "drivers/metal/metal_objects.h"
#include "platform/visionos/godot_app_delegate_service_visionos.h"
#include "servers/display_server.h"
#include "servers/rendering/rendering_device.h"
#include "servers/rendering/rendering_server_globals.h"

VisionOSVRInterface::VisionOSVRInterface() {}

VisionOSVRInterface::~VisionOSVRInterface() {
	// and make sure we cleanup if we haven't already
	if (is_initialized()) {
		uninitialize();
	};
}

StringName VisionOSVRInterface::get_name() const {
	return VisionOSVRInterface::name();
}

uint32_t VisionOSVRInterface::get_capabilities() const {
	return XRInterface::XR_VR + XRInterface::XR_STEREO;
}

uint32_t VisionOSVRInterface::get_view_count() {
	return 2;
}

bool VisionOSVRInterface::get_viewports_are_hdr() {
	return true;
}

XRInterface::TrackingStatus VisionOSVRInterface::get_tracking_status() const {
	return tracking_state;
}

bool VisionOSVRInterface::is_initialized() const {
	return (initialized);
}

bool VisionOSVRInterface::initialize() {
	print_verbose("VisionOSVRInterface.initialize()");

	if (initialized) {
		ERR_PRINT("vision_vr_interface already initialized");
		return true;
	}

	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, false);

	String driver_name = OS::get_singleton()->get_current_rendering_driver_name().to_lower();
	ERR_FAIL_COND_V_MSG(driver_name != "metal", false, "The visionOS VR interface requires the Metal rendering driver.");

	GDTRenderMode app_delegate_render_mode = GDTAppDelegateServiceVisionOS.renderMode;
	ERR_FAIL_COND_V_MSG(app_delegate_render_mode != GDTRenderModeCompositorServices, false, "The visionOS VR interface requires GDTRenderModeCompositorServices render mode.");

	layer_renderer = GDTAppDelegateServiceVisionOS.layerRenderer;
	ERR_FAIL_NULL_V_MSG(layer_renderer, false, "GDTAppDelegateServiceVisionOS.layerRenderer not set");

	rendering_device = RenderingDevice::get_singleton();
	rendering_device_driver_metal = (RenderingDeviceDriverMetal *)rendering_device->get_device_driver();

	// Initialize ARKit session
	ar_session = ar_session_create();
	ar_world_tracking_configuration_t world_tracking_configuration = ar_world_tracking_configuration_create();
	world_tracking_provider = ar_world_tracking_provider_create(world_tracking_configuration);
	current_device_anchor = ar_device_anchor_create();
	ar_data_providers_t data_providers = ar_data_providers_create();
	ar_data_providers_add_data_provider(data_providers, world_tracking_provider);
	ar_session_run(ar_session, data_providers);

	current_drawable = nullptr;

	// reset our sensor data
	head_pose = matrix_identity_float4x4;
	head_transform.basis = Basis();
	head_transform.origin = Vector3(0.0, 0.0, 0.0);

	// we must create a tracker for our head
	head.instantiate();
	head->set_tracker_type(XRServer::TRACKER_HEAD);
	head->set_tracker_name("head");
	head->set_tracker_desc("Device head pose");
	xr_server->add_tracker(head);

	// make this our primary interface
	xr_server->set_primary_interface(this);

	initialized = true;
	return initialized;
}

void VisionOSVRInterface::uninitialize() {
	if (!initialized) {
		return;
	}

	XRServer *xr_server = XRServer::get_singleton();
	if (xr_server != nullptr) {
		if (head.is_valid()) {
			xr_server->remove_tracker(head);
			head.unref();
		}

		if (xr_server->get_primary_interface() == this) {
			// no longer our primary interface
			xr_server->set_primary_interface(nullptr);
		}

		initialized = false;
	}
}

Dictionary VisionOSVRInterface::get_system_info() {
	Dictionary dict;

	dict[SNAME("XRRuntimeName")] = String("Godot visionOS VR interface");
	dict[SNAME("XRRuntimeVersion")] = String("1.0");

	return dict;
}

bool VisionOSVRInterface::supports_play_area_mode(XRInterface::PlayAreaMode p_mode) {
	return p_mode == XR_PLAY_AREA_ROOMSCALE;
}

XRInterface::PlayAreaMode VisionOSVRInterface::get_play_area_mode() const {
	return XR_PLAY_AREA_ROOMSCALE;
}

bool VisionOSVRInterface::set_play_area_mode(XRInterface::PlayAreaMode p_mode) {
	return p_mode == XR_PLAY_AREA_ROOMSCALE;
}

Transform3D VisionOSVRInterface::get_camera_transform() {
	_THREAD_SAFE_METHOD_

	Transform3D camera_transform;
	if (!initialized) {
		return camera_transform;
	}
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, camera_transform);
	// scale our origin point of our transform
	float world_scale = xr_server->get_world_scale();
	Transform3D _head_transform = head_transform;
	_head_transform.origin *= world_scale;
	camera_transform = _head_transform;
	return camera_transform;
}

Transform3D VisionOSVRInterface::get_transform_for_view(uint32_t p_view, const Transform3D &p_cam_transform) {
	_THREAD_SAFE_METHOD_

	Transform3D eye_transform;
	XRServer *xr_server = XRServer::get_singleton();
	ERR_FAIL_NULL_V(xr_server, eye_transform);
	if (initialized) {
		ERR_FAIL_COND_V(p_view > get_view_count(), eye_transform);
		ERR_FAIL_NULL_V_MSG(current_drawable, eye_transform, "Current drawable is nil, probably pre_render() has not been called, using identity transform");

		float world_scale = xr_server->get_world_scale();

		// scale our origin point of our transform
		Transform3D _head_transform = head_transform;
		_head_transform.origin *= world_scale;

		// get eye transform
		cp_view_t view = cp_drawable_get_view(current_drawable, p_view);
		simd_float4x4 eye_offset = cp_view_get_transform(view);
		simd_float4x4 eye_pose = simd_mul(head_pose, eye_offset);
		eye_transform = MTL::simd_to_transform3D(eye_pose);
	} else {
		ERR_PRINT("vision_vr_interface not initialized, returning received camera transform");
		eye_transform = Transform3D();
	};
	Transform3D reference_frame = xr_server->get_reference_frame();
	return p_cam_transform * reference_frame * eye_transform;
}

Projection VisionOSVRInterface::get_projection_for_view(uint32_t p_view, double p_aspect, double p_z_near, double p_z_far) {
	_THREAD_SAFE_METHOD_

	Projection eye_projection;
	if (!initialized) {
		return eye_projection;
	}

	ERR_FAIL_COND_V(p_view > get_view_count(), eye_projection);
	ERR_FAIL_NULL_V_MSG(current_drawable, eye_projection, "Current drawable is nil, probably pre_render() has not been called");

	// Clamp depth range to expected minimum value by cp_drawable_set_depth_range()
	simd_float2 depth_range = simd_make_float2(p_z_far, MAX(p_z_near, 0.100000));
	cp_drawable_set_depth_range(current_drawable, depth_range);
	simd_float4x4 eye_simd_projection = cp_drawable_compute_projection(current_drawable, cp_axis_direction_convention_right_up_forward, p_view);
	eye_projection = MTL::simd_to_projection(eye_simd_projection);
	return eye_projection;
}

Rect2i VisionOSVRInterface::get_viewport_for_view(uint32_t p_view) {
	_THREAD_SAFE_METHOD_

	Rect2 viewport_rect;
	if (!initialized) {
		return viewport_rect;
	}

	ERR_FAIL_COND_V(p_view > get_view_count(), viewport_rect);
	ERR_FAIL_NULL_V_MSG(current_drawable, viewport_rect, "Current drawable is nil, probably pre_render() has not been called");

	cp_view_t view = cp_drawable_get_view(current_drawable, p_view);
	cp_view_texture_map_t view_texture_map = cp_view_get_view_texture_map(view);
	MTLViewport viewport = cp_view_texture_map_get_viewport(view_texture_map);
	viewport_rect = MTL::rect_from_mtl_viewport(viewport);
	return viewport_rect;
}

Size2 VisionOSVRInterface::get_render_target_size() {
	Size2 target_size;
	if (!initialized) {
		return target_size;
	}
	ERR_FAIL_NULL_V_MSG(current_drawable, target_size, "Current drawable is nil, probably pre_render() has not been called");
	id<MTLTexture> color_texture = cp_drawable_get_color_texture(current_drawable, 0);
	target_size = Size2(color_texture.width, color_texture.height);
	return target_size;
}

void VisionOSVRInterface::set_head_pose_from_arkit() {
	_THREAD_SAFE_METHOD_

	ERR_FAIL_NULL_MSG(current_frame, "Current frame is nil, probably process() has not been called, using identity transform");
	ERR_FAIL_NULL_MSG(current_drawable, "Current drawable is nil, probably process() has not been called, using identity transform");

	cp_frame_timing_t frame_timing = cp_drawable_get_frame_timing(current_drawable);
	CFTimeInterval presentation_time = cp_time_to_cf_time_interval(cp_frame_timing_get_presentation_time(frame_timing));
	ar_device_anchor_query_status_t query_anchor_result = ar_world_tracking_provider_query_device_anchor_at_timestamp(world_tracking_provider, presentation_time, current_device_anchor);

	if (query_anchor_result != ar_device_anchor_query_status_success) {
		tracking_state = XRInterface::XR_NOT_TRACKING;
		tracking_confidence = XRPose::XR_TRACKING_CONFIDENCE_NONE;
		ERR_FAIL_MSG("cannot query device anchor, result: " + itos(query_anchor_result));
	}

	head_pose = ar_anchor_get_origin_from_anchor_transform(current_device_anchor);
	head_transform = MTL::simd_to_transform3D(head_pose);
	tracking_state = XRInterface::XR_NORMAL_TRACKING;
	tracking_confidence = XRPose::XR_TRACKING_CONFIDENCE_HIGH;
}

void VisionOSVRInterface::process() {
	if (!initialized) {
		return;
	}
	current_frame = cp_layer_renderer_query_next_frame(layer_renderer);
	cp_frame_start_update(current_frame);
}

void VisionOSVRInterface::pre_render() {
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

	set_head_pose_from_arkit();

	if (head.is_valid()) {
		// Set our head position (in real space, reference frame and world scale is applied later)
		head->set_pose("default", head_transform, Vector3(), Vector3(), tracking_confidence);
	}

	if (current_device_anchor != nil) {
		cp_drawable_set_device_anchor(current_drawable, current_device_anchor);
	} else {
		ERR_PRINT("Current device anchor is nil, will present drawable without a device anchor");
	}
}

Vector<BlitToScreen> VisionOSVRInterface::post_draw_viewport(RID p_render_target, const Rect2 &p_screen_rect) {
	_THREAD_SAFE_METHOD_
	// We're overriding the color and depth textures, no need for screen blits
	return Vector<BlitToScreen>();
}

void VisionOSVRInterface::encode_present(MDCommandBuffer *p_cmd_buffer) {
	_THREAD_SAFE_METHOD_

	if (!initialized) {
		return;
	}
	ERR_FAIL_NULL_MSG(current_drawable, "Current drawable is nil, probably process() has not been called");
	cp_drawable_encode_present(current_drawable, p_cmd_buffer->get_command_buffer());
}

void VisionOSVRInterface::end_frame() {
	_THREAD_SAFE_METHOD_

	if (!initialized) {
		return;
	}
	ERR_FAIL_NULL_MSG(current_frame, "Current frame is nil, probably process() has not been called");
	cp_frame_end_submission(current_frame);
}

RID VisionOSVRInterface::get_color_texture() {
	_THREAD_SAFE_METHOD_

	if (!initialized) {
		return RID();
	}

	if (current_color_texture_id != RID()) {
		rendering_device->texture_owner.free(current_color_texture_id);
	}

	ERR_FAIL_NULL_V_MSG(current_drawable, RID(), "Current drawable is nil, probably pre_render() has not been called");
	id<MTLTexture> color_texture = cp_drawable_get_color_texture(current_drawable, 0);

	PixelFormats pixel_formats = rendering_device_driver_metal->get_pixel_formats();

	RD::Texture texture;
	texture.driver_id = rid::make(color_texture);
	ERR_FAIL_COND_V(!texture.driver_id, RID());
	texture.type = MTL::texture_type_from_metal(color_texture.textureType);
	texture.format = pixel_formats.getDataFormat(color_texture.pixelFormat);
	texture.width = color_texture.width;
	texture.height = color_texture.height;
	texture.depth = color_texture.depth;
	texture.layers = color_texture.arrayLength;
	texture.mipmaps = color_texture.mipmapLevelCount;
	texture.base_mipmap = 0;
	texture.base_layer = 0;
	texture.usage_flags = RD::TEXTURE_USAGE_COLOR_ATTACHMENT_BIT | RD::TEXTURE_USAGE_SAMPLING_BIT;
	texture.samples = MTL::texture_samples_from_metal(color_texture.sampleCount);
	texture.is_resolve_buffer = false;
	texture.has_initial_data = false;

	texture.draw_tracker = RDG::resource_tracker_create();
	texture.draw_tracker->texture_driver_id = texture.driver_id;
	texture.draw_tracker->texture_subresources = texture.barrier_range();
	texture.draw_tracker->texture_usage = texture.usage_flags;
	texture.draw_tracker->reference_count = 1;

	current_color_texture = texture;
	current_color_texture_id = rendering_device->texture_owner.make_rid(current_color_texture);

	return current_color_texture_id;
}

RID VisionOSVRInterface::get_depth_texture() {
	_THREAD_SAFE_METHOD_

	if (!initialized) {
		return RID();
	}

	if (current_depth_texture_id != RID()) {
		rendering_device->texture_owner.free(current_depth_texture_id);
	}

	ERR_FAIL_NULL_V_MSG(current_drawable, RID(), "Current drawable is nil, probably pre_render() has not been called");
	id<MTLTexture> depth_texture = cp_drawable_get_depth_texture(current_drawable, 0);

	PixelFormats pixel_formats = rendering_device_driver_metal->get_pixel_formats();

	RD::Texture texture;
	texture.driver_id = rid::make(depth_texture);
	ERR_FAIL_COND_V(!texture.driver_id, RID());
	texture.type = MTL::texture_type_from_metal(depth_texture.textureType);
	texture.format = pixel_formats.getDataFormat(depth_texture.pixelFormat);
	texture.width = depth_texture.width;
	texture.height = depth_texture.height;
	texture.depth = depth_texture.depth;
	texture.layers = depth_texture.arrayLength;
	texture.mipmaps = depth_texture.mipmapLevelCount;
	texture.base_mipmap = 0;
	texture.base_layer = 0;
	texture.usage_flags = RD::TEXTURE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
	texture.samples = MTL::texture_samples_from_metal(depth_texture.sampleCount);
	texture.is_resolve_buffer = false;
	texture.has_initial_data = false;

	texture.draw_tracker = RDG::resource_tracker_create();
	texture.draw_tracker->texture_driver_id = texture.driver_id;
	texture.draw_tracker->texture_subresources = texture.barrier_range();
	texture.draw_tracker->texture_usage = texture.usage_flags;
	texture.draw_tracker->reference_count = 1;

	current_depth_texture = texture;
	current_depth_texture_id = rendering_device->texture_owner.make_rid(current_depth_texture);

	return current_depth_texture_id;
}

RID VisionOSVRInterface::get_vrs_texture() {
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

	RD::Texture texture;
	texture.driver_id = RDD::TextureID((__bridge void *)rasterization_rate_map);
	ERR_FAIL_COND_V(!texture.driver_id, RID());

	current_rasterization_rate_map = texture;
	current_rasterization_rate_map_id = rendering_device->texture_owner.make_rid(current_rasterization_rate_map);

	return current_rasterization_rate_map_id;
}

VisionOSVRInterface::VRSTextureFormat VisionOSVRInterface::get_vrs_texture_format() {
	return XR_VRS_TEXTURE_FORMAT_RASTERIZATION_RATE_MAP;
}

#endif // VISIONOS_ENABLED
