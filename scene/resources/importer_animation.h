/*************************************************************************/
/*  importer_animation.h                                                 */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef IMPORTERANIMATION_H
#define IMPORTERANIMATION_H

#include "core/templates/local_vector.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/main/node.h"
#include "scene/resources/animation.h"

class ImporterAnimation : public Resource {
	GDCLASS(ImporterAnimation, Resource)
public:
	enum TrackType {
		TRACK_TYPE_TRANSLATION,
		TRACK_TYPE_ROTATION,
		TRACK_TYPE_SCALE,
		TRACK_TYPE_BLEND_SHAPE_KEY_0,
		TRACK_TYPE_MAX
	};

	enum RotationMode {
		ROTATION_MODE_EULER_XYZ,
		ROTATION_MODE_EULER_XZY,
		ROTATION_MODE_EULER_YXZ,
		ROTATION_MODE_EULER_YZX,
		ROTATION_MODE_EULER_ZXY,
		ROTATION_MODE_EULER_ZYX,
		ROTATION_MODE_QUATERNION,
	};

	enum InterpolationMode {
		INTERPOLATION_MODE_NEAREST,
		INTERPOLATION_MODE_LINEAR,
		INTERPOLATION_MODE_CATMULL_ROM_SPLINE,
		INTERPOLATION_MODE_CUBIC_SPLINE,
		INTERPOLATION_MODE_GLTF2_SPLINE,
	};

	enum LoopMode {
		LOOP_MODE_DISABLED,
		LOOP_MODE_FORWARD,
		LOOP_MODE_PINGPONG,
	};

private:
	struct Point {
		real_t time = 0;
		real_t value = 0;
	};
	struct Key {
		Point point;
		Point in_handle;
		Point out_handle;
	};

	struct QuaternionKey {
		real_t time = 0;
		Quaternion value;
		Quaternion in_handle;
		Quaternion out_handle;
	};

	struct Channel {
		InterpolationMode interpolation_mode = INTERPOLATION_MODE_LINEAR;
		LocalVector<Key> keys;
	};

	struct QuaternionChannel {
		InterpolationMode interpolation_mode = INTERPOLATION_MODE_LINEAR;
		LocalVector<QuaternionKey> keys;
	};

	struct Track {
		Channel channels[3];
		QuaternionChannel quaternion_channel;
	};

	struct NodeData {
		NodePath path;
		RotationMode rotation_mode = ROTATION_MODE_EULER_XYZ;
		LocalVector<Track> tracks;
		LocalVector<StringName> blend_shape_tracks;
		NodeData() {
			tracks.resize(TRACK_TYPE_MAX - 1);
		}
	};

	LocalVector<NodeData> nodes;
	LoopMode loop_mode = LOOP_MODE_DISABLED;
	real_t length = 1.0;

	void _set_nodes(const Array &p_nodes);
	Array _get_nodes() const;

protected:
	static void _bind_methods();

public:
	int get_node_count() const;
	void add_node(const NodePath &p_path);
	NodePath node_get_path(uint32_t p_node) const;
	void node_set_rotation_mode(uint32_t p_node, RotationMode p_mode);
	RotationMode node_get_rotation_mode(uint32_t p_node) const;
	void node_set_blend_shape_track_count(uint32_t p_node, uint32_t p_count);
	uint32_t node_get_blend_shape_track_count(uint32_t p_node) const;
	void node_set_blend_shape_track_name(uint32_t p_node, uint32_t p_track, const StringName &p_name);
	StringName node_get_blend_shape_track_name(uint32_t p_node, uint32_t p_track) const;
	void node_track_set_axis_channel_interpolation_mode(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel, InterpolationMode p_mode);
	void node_track_set_quaternion_channel_interpolation_mode(uint32_t p_node, uint32_t p_track_type, InterpolationMode p_mode);
	InterpolationMode node_track_get_axis_channel_interpolation_mode(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel) const;
	InterpolationMode node_track_get_quaternion_channel_interpolation_mode(uint32_t p_node, uint32_t p_track_type) const;
	void node_track_add_axis_key(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel, real_t p_time, real_t p_value, real_t p_in_handle_delta_time = 0, real_t p_in_handle_value = 0, real_t p_out_handle_delta_time = 0, real_t p_out_handle_value = 0);
	void node_track_add_quaternion_key(uint32_t p_node, uint32_t p_track_type, real_t p_time, const Quaternion &p_quaternion, const Quaternion &p_in = Quaternion(), const Quaternion &p_out = Quaternion());
	bool node_has_track(uint32_t p_node, uint32_t p_track_type) const;
	bool node_track_has_axis_channel(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel) const;
	bool node_track_has_quaternion_channel(uint32_t p_node, uint32_t p_track_type) const;
	bool node_has_transform_tracks(uint32_t p_node) const;
	int node_track_channel_get_axis_key_count(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel) const;
	int node_track_channel_get_quaternion_key_count(uint32_t p_node, uint32_t p_track_type) const;
	real_t node_track_channel_get_axis_key_time(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel, uint32_t p_key) const;
	real_t node_track_channel_get_axis_key_value(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel, uint32_t p_key) const;
	real_t node_track_channel_get_axis_key_in_handle_time(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel, uint32_t p_key) const;
	real_t node_track_channel_get_axis_key_in_handle_value(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel, uint32_t p_key) const;
	real_t node_track_channel_get_axis_key_out_handle_time(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel, uint32_t p_key) const;
	real_t node_track_channel_get_axis_key_out_handle_value(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel, uint32_t p_key) const;
	real_t node_track_channel_get_quaternion_key_time(uint32_t p_node, uint32_t p_track_type, uint32_t p_key) const;
	Quaternion node_track_channel_get_quaternion_key_value(uint32_t p_node, uint32_t p_track_type, uint32_t p_key) const;
	Quaternion node_track_channel_get_quaternion_key_in_handle(uint32_t p_node, uint32_t p_track_type, uint32_t p_key) const;
	Quaternion node_track_channel_get_quaternion_key_out_handle(uint32_t p_node, uint32_t p_track_type, uint32_t p_key) const;
	void node_track_channel_remove_axis_key(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel, uint32_t p_key);
	void node_track_channel_remove_quaternion_key(uint32_t p_node, uint32_t p_track_type, uint32_t p_key);

	void node_track_remove_axis_channel(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel);
	void node_track_remove_quaternion_channel(uint32_t p_node, uint32_t p_track_type);
	void node_remove_track(uint32_t p_node, uint32_t p_track_type);
	void remove_node(uint32_t p_node);

	real_t node_track_axis_interpolate(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel, real_t p_time);
	Vector3 node_track_interpolate_translation(uint32_t p_node, real_t p_time);
	Quaternion node_track_interpolate_rotation(uint32_t p_node, real_t p_time);
	Vector3 node_track_interpolate_scale(uint32_t p_node, real_t p_time);
	float node_track_interpolate_blend_shape(uint32_t p_node, uint32_t p_blend_shape, real_t p_time);

	void clear();

	void set_loop_mode(LoopMode p_mode);
	LoopMode get_loop_mode() const;

	void set_length(real_t p_length);
	real_t get_length() const;

	Ref<Animation> bake(int p_bake_fps, Node *p_base_path, bool p_relative_transform);

	ImporterAnimation();
};

VARIANT_ENUM_CAST(ImporterAnimation::TrackType)
VARIANT_ENUM_CAST(ImporterAnimation::RotationMode)
VARIANT_ENUM_CAST(ImporterAnimation::InterpolationMode)
VARIANT_ENUM_CAST(ImporterAnimation::LoopMode)

#endif // IMPORTERANIMATION_H
