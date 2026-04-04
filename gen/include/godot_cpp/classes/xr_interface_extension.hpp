/**************************************************************************/
/*  xr_interface_extension.hpp                                            */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/xr_interface.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_float64_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/rid.hpp>
#include <godot_cpp/variant/string_name.hpp>
#include <godot_cpp/variant/transform3d.hpp>
#include <godot_cpp/variant/vector2.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

struct Rect2;
struct Rect2i;
class String;

class XRInterfaceExtension : public XRInterface {
	GDEXTENSION_CLASS(XRInterfaceExtension, XRInterface)

public:
	RID get_color_texture();
	RID get_depth_texture();
	RID get_velocity_texture();
	void add_blit(const RID &p_render_target, const Rect2 &p_src_rect, const Rect2i &p_dst_rect, bool p_use_layer, uint32_t p_layer, bool p_apply_lens_distortion, const Vector2 &p_eye_center, double p_k1, double p_k2, double p_upscale, double p_aspect_ratio);
	RID get_render_target_texture(const RID &p_render_target);
	virtual StringName _get_name() const;
	virtual uint32_t _get_capabilities() const;
	virtual bool _is_initialized() const;
	virtual bool _initialize();
	virtual void _uninitialize();
	virtual Dictionary _get_system_info() const;
	virtual bool _supports_play_area_mode(XRInterface::PlayAreaMode p_mode) const;
	virtual XRInterface::PlayAreaMode _get_play_area_mode() const;
	virtual bool _set_play_area_mode(XRInterface::PlayAreaMode p_mode) const;
	virtual PackedVector3Array _get_play_area() const;
	virtual Vector2 _get_render_target_size();
	virtual uint32_t _get_view_count();
	virtual Transform3D _get_camera_transform();
	virtual Transform3D _get_transform_for_view(uint32_t p_view, const Transform3D &p_cam_transform);
	virtual PackedFloat64Array _get_projection_for_view(uint32_t p_view, double p_aspect, double p_z_near, double p_z_far);
	virtual RID _get_vrs_texture();
	virtual XRInterface::VRSTextureFormat _get_vrs_texture_format();
	virtual void _process();
	virtual void _pre_render();
	virtual bool _pre_draw_viewport(const RID &p_render_target);
	virtual void _post_draw_viewport(const RID &p_render_target, const Rect2 &p_screen_rect);
	virtual void _end_frame();
	virtual PackedStringArray _get_suggested_tracker_names() const;
	virtual PackedStringArray _get_suggested_pose_names(const StringName &p_tracker_name) const;
	virtual XRInterface::TrackingStatus _get_tracking_status() const;
	virtual void _trigger_haptic_pulse(const String &p_action_name, const StringName &p_tracker_name, double p_frequency, double p_amplitude, double p_duration_sec, double p_delay_sec);
	virtual bool _get_anchor_detection_is_enabled() const;
	virtual void _set_anchor_detection_is_enabled(bool p_enabled);
	virtual int32_t _get_camera_feed_id() const;
	virtual RID _get_color_texture();
	virtual RID _get_depth_texture();
	virtual RID _get_velocity_texture();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		XRInterface::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_get_name), decltype(&T::_get_name)>) {
			BIND_VIRTUAL_METHOD(T, _get_name, 2002593661);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_capabilities), decltype(&T::_get_capabilities)>) {
			BIND_VIRTUAL_METHOD(T, _get_capabilities, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_is_initialized), decltype(&T::_is_initialized)>) {
			BIND_VIRTUAL_METHOD(T, _is_initialized, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_initialize), decltype(&T::_initialize)>) {
			BIND_VIRTUAL_METHOD(T, _initialize, 2240911060);
		}
		if constexpr (!std::is_same_v<decltype(&B::_uninitialize), decltype(&T::_uninitialize)>) {
			BIND_VIRTUAL_METHOD(T, _uninitialize, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_system_info), decltype(&T::_get_system_info)>) {
			BIND_VIRTUAL_METHOD(T, _get_system_info, 3102165223);
		}
		if constexpr (!std::is_same_v<decltype(&B::_supports_play_area_mode), decltype(&T::_supports_play_area_mode)>) {
			BIND_VIRTUAL_METHOD(T, _supports_play_area_mode, 2693703033);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_play_area_mode), decltype(&T::_get_play_area_mode)>) {
			BIND_VIRTUAL_METHOD(T, _get_play_area_mode, 1615132885);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_play_area_mode), decltype(&T::_set_play_area_mode)>) {
			BIND_VIRTUAL_METHOD(T, _set_play_area_mode, 2693703033);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_play_area), decltype(&T::_get_play_area)>) {
			BIND_VIRTUAL_METHOD(T, _get_play_area, 497664490);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_render_target_size), decltype(&T::_get_render_target_size)>) {
			BIND_VIRTUAL_METHOD(T, _get_render_target_size, 1497962370);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_view_count), decltype(&T::_get_view_count)>) {
			BIND_VIRTUAL_METHOD(T, _get_view_count, 2455072627);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_camera_transform), decltype(&T::_get_camera_transform)>) {
			BIND_VIRTUAL_METHOD(T, _get_camera_transform, 4183770049);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_transform_for_view), decltype(&T::_get_transform_for_view)>) {
			BIND_VIRTUAL_METHOD(T, _get_transform_for_view, 518934792);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_projection_for_view), decltype(&T::_get_projection_for_view)>) {
			BIND_VIRTUAL_METHOD(T, _get_projection_for_view, 4067457445);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_vrs_texture), decltype(&T::_get_vrs_texture)>) {
			BIND_VIRTUAL_METHOD(T, _get_vrs_texture, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_vrs_texture_format), decltype(&T::_get_vrs_texture_format)>) {
			BIND_VIRTUAL_METHOD(T, _get_vrs_texture_format, 1500923256);
		}
		if constexpr (!std::is_same_v<decltype(&B::_process), decltype(&T::_process)>) {
			BIND_VIRTUAL_METHOD(T, _process, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_pre_render), decltype(&T::_pre_render)>) {
			BIND_VIRTUAL_METHOD(T, _pre_render, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_pre_draw_viewport), decltype(&T::_pre_draw_viewport)>) {
			BIND_VIRTUAL_METHOD(T, _pre_draw_viewport, 3521089500);
		}
		if constexpr (!std::is_same_v<decltype(&B::_post_draw_viewport), decltype(&T::_post_draw_viewport)>) {
			BIND_VIRTUAL_METHOD(T, _post_draw_viewport, 1378122625);
		}
		if constexpr (!std::is_same_v<decltype(&B::_end_frame), decltype(&T::_end_frame)>) {
			BIND_VIRTUAL_METHOD(T, _end_frame, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_suggested_tracker_names), decltype(&T::_get_suggested_tracker_names)>) {
			BIND_VIRTUAL_METHOD(T, _get_suggested_tracker_names, 1139954409);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_suggested_pose_names), decltype(&T::_get_suggested_pose_names)>) {
			BIND_VIRTUAL_METHOD(T, _get_suggested_pose_names, 1761182771);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_tracking_status), decltype(&T::_get_tracking_status)>) {
			BIND_VIRTUAL_METHOD(T, _get_tracking_status, 167423259);
		}
		if constexpr (!std::is_same_v<decltype(&B::_trigger_haptic_pulse), decltype(&T::_trigger_haptic_pulse)>) {
			BIND_VIRTUAL_METHOD(T, _trigger_haptic_pulse, 3752640163);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_anchor_detection_is_enabled), decltype(&T::_get_anchor_detection_is_enabled)>) {
			BIND_VIRTUAL_METHOD(T, _get_anchor_detection_is_enabled, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_anchor_detection_is_enabled), decltype(&T::_set_anchor_detection_is_enabled)>) {
			BIND_VIRTUAL_METHOD(T, _set_anchor_detection_is_enabled, 2586408642);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_camera_feed_id), decltype(&T::_get_camera_feed_id)>) {
			BIND_VIRTUAL_METHOD(T, _get_camera_feed_id, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_color_texture), decltype(&T::_get_color_texture)>) {
			BIND_VIRTUAL_METHOD(T, _get_color_texture, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_depth_texture), decltype(&T::_get_depth_texture)>) {
			BIND_VIRTUAL_METHOD(T, _get_depth_texture, 529393457);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_velocity_texture), decltype(&T::_get_velocity_texture)>) {
			BIND_VIRTUAL_METHOD(T, _get_velocity_texture, 529393457);
		}
	}

public:
};

} // namespace godot

