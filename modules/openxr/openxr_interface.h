/*************************************************************************/
/*  openxr_interface.h                                                   */
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

#ifndef OPENXR_INTERFACE_H
#define OPENXR_INTERFACE_H

#include "servers/xr/xr_interface.h"
#include "servers/xr/xr_positional_tracker.h"

#include "action_map/openxr_action_map.h"
#include "openxr_api.h"

class OpenXRInterface : public XRInterface {
	GDCLASS(OpenXRInterface, XRInterface);

private:
	OpenXRAPI *openxr_api = nullptr;
	bool initialized = false;
	XRInterface::TrackingStatus tracking_state;

	// At a minimum we need a tracker for our head
	Ref<XRPositionalTracker> head;
	Transform3D head_transform;
	Vector3 head_linear_velocity;
	Vector3 head_angular_velocity;
	Transform3D transform_for_view[2]; // We currently assume 2, but could be 4 for VARJO which we do not support yet

	void _load_action_map();

	struct Action {
		String action_name;
		OpenXRAction::ActionType action_type;
		RID action_rid;
	};
	struct ActionSet {
		String action_set_name;
		bool is_active;
		RID action_set_rid;
		Vector<Action *> actions;
	};
	struct Tracker {
		String path_name;
		RID path_rid;
		Ref<XRPositionalTracker> positional_tracker;
		Vector<Action *> actions;
	};

	Vector<ActionSet *> action_sets;
	Vector<Tracker *> trackers;

	ActionSet *create_action_set(const String &p_action_set_name, const String &p_localized_name, const int p_priority);
	void free_action_sets();

	Action *create_action(ActionSet *p_action_set, const String &p_action_name, const String &p_localized_name, OpenXRAction::ActionType p_action_type, const Vector<RID> p_toplevel_paths);
	Action *find_action(const String &p_action_name);
	void free_actions(ActionSet *p_action_set);

	Tracker *get_tracker(const String &p_path_name);
	Tracker *find_tracker(const String &p_positional_tracker_name);
	void link_action_to_tracker(Tracker *p_tracker, Action *p_action);
	void handle_tracker(Tracker *p_tracker);
	void free_trackers();

	void _set_default_pos(Transform3D &p_transform, double p_world_scale, uint64_t p_eye);

protected:
	static void _bind_methods();

public:
	virtual StringName get_name() const override;
	virtual uint32_t get_capabilities() const override;

	virtual TrackingStatus get_tracking_status() const override;

	bool initialise_on_startup() const;
	virtual bool is_initialized() const override;
	virtual bool initialize() override;
	virtual void uninitialize() override;

	virtual void trigger_haptic_pulse(const String &p_action_name, const StringName &p_tracker_name, double p_frequency, double p_amplitude, double p_duration_sec, double p_delay_sec = 0) override;

	virtual bool supports_play_area_mode(XRInterface::PlayAreaMode p_mode) override;
	virtual XRInterface::PlayAreaMode get_play_area_mode() const override;
	virtual bool set_play_area_mode(XRInterface::PlayAreaMode p_mode) override;

	virtual Size2 get_render_target_size() override;
	virtual uint32_t get_view_count() override;
	virtual Transform3D get_camera_transform() override;
	virtual Transform3D get_transform_for_view(uint32_t p_view, const Transform3D &p_cam_transform) override;
	virtual CameraMatrix get_projection_for_view(uint32_t p_view, double p_aspect, double p_z_near, double p_z_far) override;

	virtual void process() override;
	virtual void pre_render() override;
	bool pre_draw_viewport(RID p_render_target) override;
	virtual Vector<BlitToScreen> post_draw_viewport(RID p_render_target, const Rect2 &p_screen_rect) override;
	virtual void end_frame() override;

	OpenXRInterface();
	~OpenXRInterface();
};

#endif // !OPENXR_INTERFACE_H
