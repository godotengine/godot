/*************************************************************************/
/*  openxr_api.h                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#ifndef OPENXR_DRIVER_H
#define OPENXR_DRIVER_H

#include "core/error/error_macros.h"
#include "core/math/camera_matrix.h"
#include "core/math/transform_3d.h"
#include "core/math/vector2.h"
#include "core/os/memory.h"
#include "core/string/ustring.h"
#include "core/templates/map.h"
#include "core/templates/rid_owner.h"
#include "core/templates/vector.h"
#include "servers/xr/xr_pose.h"

#include "thirdparty/openxr/src/common/xr_linear.h"
#include <openxr/openxr.h>

#include "action_map/openxr_action.h"

#include "extensions/openxr_composition_layer_provider.h"
#include "extensions/openxr_extension_wrapper.h"

// Note, OpenXR code that we wrote for our plugin makes use of C++20 notation for initialising structs which ensures zeroing out unspecified members.
// Godot is currently restricted to C++17 which doesn't allow this notation. Make sure critical fields are set.

// forward declarations, we don't want to include these fully
class OpenXRVulkanExtension;

class OpenXRAPI {
private:
	// our singleton
	static OpenXRAPI *singleton;

	// layers
	uint32_t num_layer_properties = 0;
	XrApiLayerProperties *layer_properties = nullptr;

	// extensions
	uint32_t num_supported_extensions = 0;
	XrExtensionProperties *supported_extensions = nullptr;
	Vector<OpenXRExtensionWrapper *> registered_extension_wrappers;
	Vector<const char *> enabled_extensions;

	// composition layer providers
	Vector<OpenXRCompositionLayerProvider *> composition_layer_providers;

	// view configuration
	uint32_t num_view_configuration_types = 0;
	XrViewConfigurationType *supported_view_configuration_types = nullptr;

	// reference spaces
	uint32_t num_reference_spaces = 0;
	XrReferenceSpaceType *supported_reference_spaces = nullptr;

	// swapchains (note these are platform dependent)
	uint32_t num_swapchain_formats = 0;
	int64_t *supported_swapchain_formats = nullptr;

	// configuration
	XrFormFactor form_factor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
	XrViewConfigurationType view_configuration = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
	XrReferenceSpaceType reference_space = XR_REFERENCE_SPACE_TYPE_STAGE;
	XrEnvironmentBlendMode environment_blend_mode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;

	// state
	XrInstance instance = XR_NULL_HANDLE;
	XrSystemId system_id;
	String system_name;
	uint32_t vendor_id;
	XrSystemTrackingProperties tracking_properties;
	XrSession session = XR_NULL_HANDLE;
	XrSessionState session_state = XR_SESSION_STATE_UNKNOWN;
	bool running = false;
	XrFrameState frame_state = { XR_TYPE_FRAME_STATE, NULL, 0, 0, false };

	OpenXRGraphicsExtensionWrapper *graphics_extension = nullptr;
	XrSystemGraphicsProperties graphics_properties;
	void *swapchain_graphics_data = nullptr;
	uint32_t image_index = 0;
	bool image_acquired = false;

	uint32_t view_count = 0;
	XrViewConfigurationView *view_configuration_views = nullptr;
	XrView *views = nullptr;
	XrCompositionLayerProjectionView *projection_views = nullptr;
	XrSwapchain swapchain = XR_NULL_HANDLE;

	XrSpace play_space = XR_NULL_HANDLE;
	XrSpace view_space = XR_NULL_HANDLE;
	bool view_pose_valid = false;
	XRPose::TrackingConfidence head_pose_confidence = XRPose::XR_TRACKING_CONFIDENCE_NONE;

	bool load_layer_properties();
	bool load_supported_extensions();
	bool is_extension_supported(const char *p_extension) const;

	// instance
	bool create_instance();
	bool get_system_info();
	bool load_supported_view_configuration_types();
	bool is_view_configuration_supported(XrViewConfigurationType p_configuration_type) const;
	bool load_supported_view_configuration_views(XrViewConfigurationType p_configuration_type);
	void destroy_instance();

	// session
	bool create_session();
	bool load_supported_reference_spaces();
	bool is_reference_space_supported(XrReferenceSpaceType p_reference_space);
	bool setup_spaces();
	bool load_supported_swapchain_formats();
	bool is_swapchain_format_supported(int64_t p_swapchain_format);
	bool create_main_swapchain();
	void destroy_session();

	// swapchains
	bool create_swapchain(int64_t p_swapchain_format, uint32_t p_width, uint32_t p_height, uint32_t p_sample_count, uint32_t p_array_size, XrSwapchain &r_swapchain, void **r_swapchain_graphics_data);
	bool acquire_image(XrSwapchain p_swapchain, uint32_t &r_image_index);
	bool release_image(XrSwapchain p_swapchain);

	// action map
	struct Path {
		XrPath path;
	};
	RID_Owner<Path, true> xr_path_owner;

	struct ActionSet {
		bool is_attached;
		XrActionSet handle;
	};
	RID_Owner<ActionSet, true> action_set_owner;

	struct PathWithSpace {
		XrPath toplevel_path;
		XrSpace space;
		bool was_location_valid;
	};

	struct Action {
		XrActionType action_type;
		Vector<PathWithSpace> toplevel_paths;
		XrAction handle;
	};
	RID_Owner<Action, true> action_owner;

	// state changes
	bool poll_events();
	bool on_state_idle();
	bool on_state_ready();
	bool on_state_synchronized();
	bool on_state_visible();
	bool on_state_focused();
	bool on_state_stopping();
	bool on_state_loss_pending();
	bool on_state_exiting();

	// convencience
	void copy_string_to_char_buffer(const String p_string, char *p_buffer, int p_buffer_len);

protected:
	friend class OpenXRVulkanExtension;

	XrInstance get_instance() const { return instance; };
	XrSystemId get_system_id() const { return system_id; };
	XrSession get_session() const { return session; };

	// helper method to convert an XrPosef to a Transform3D
	Transform3D transform_from_pose(const XrPosef &p_pose);

	// helper method to get a valid Transform3D from an openxr space location
	XRPose::TrackingConfidence transform_from_location(const XrSpaceLocation &p_location, Transform3D &r_transform);
	XRPose::TrackingConfidence transform_from_location(const XrHandJointLocationEXT &p_location, Transform3D &r_transform);
	void parse_velocities(const XrSpaceVelocity &p_velocity, Vector3 &r_linear_velocity, Vector3 r_angular_velocity);

public:
	static void setup_global_defs();
	static bool openxr_is_enabled();
	static OpenXRAPI *get_singleton();

	String get_error_string(XrResult result);
	String get_swapchain_format_name(int64_t p_swapchain_format) const;

	void register_extension_wrapper(OpenXRExtensionWrapper *p_extension_wrapper);

	bool is_initialized();
	bool is_running();
	bool initialise(const String &p_rendering_driver);
	bool initialise_session();
	void finish();

	XrTime get_next_frame_time() { return frame_state.predictedDisplayTime + frame_state.predictedDisplayPeriod; };
	bool can_render() { return instance != XR_NULL_HANDLE && session != XR_NULL_HANDLE && running && view_pose_valid && frame_state.shouldRender; };

	Size2 get_recommended_target_size();
	XRPose::TrackingConfidence get_head_center(Transform3D &r_transform, Vector3 &r_linear_velocity, Vector3 &r_angular_velocity);
	bool get_view_transform(uint32_t p_view, Transform3D &r_transform);
	bool get_view_projection(uint32_t p_view, double p_z_near, double p_z_far, CameraMatrix &p_camera_matrix);
	bool process();

	void pre_render();
	bool pre_draw_viewport(RID p_render_target);
	void post_draw_viewport(RID p_render_target);
	void end_frame();

	// action map
	String get_default_action_map_resource_name();
	RID path_create(const String p_name);
	void path_free(RID p_path);
	RID action_set_create(const String p_name, const String p_localized_name, const int p_priority);
	bool action_set_attach(RID p_action_set);
	void action_set_free(RID p_action_set);
	RID action_create(RID p_action_set, const String p_name, const String p_localized_name, OpenXRAction::ActionType p_action_type, const Vector<RID> &p_toplevel_paths);
	void action_free(RID p_action);

	struct Binding {
		RID action;
		String path;
	};
	bool suggest_bindings(const String p_interaction_profile, const Vector<Binding> p_bindings);

	bool sync_action_sets(const Vector<RID> p_active_sets);
	bool get_action_bool(RID p_action, RID p_path);
	float get_action_float(RID p_action, RID p_path);
	Vector2 get_action_vector2(RID p_action, RID p_path);
	XRPose::TrackingConfidence get_action_pose(RID p_action, RID p_path, Transform3D &r_transform, Vector3 &r_linear_velocity, Vector3 &r_angular_velocity);
	bool trigger_haptic_pulse(RID p_action, RID p_path, float p_frequency, float p_amplitude, XrDuration p_duration_ns);

	OpenXRAPI();
	~OpenXRAPI();
};

#endif // !OPENXR_DRIVER_H
