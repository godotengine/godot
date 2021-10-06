/*************************************************************************/
/*  importer_animation.cpp                                               */
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

#include "importer_animation.h"

int ImporterAnimation::get_node_count() const {
	return nodes.size();
}
void ImporterAnimation::add_node(const NodePath &p_path) {
	ERR_FAIL_COND(p_path.is_empty());

	for (uint32_t i = 0; i < nodes.size(); i++) {
		ERR_FAIL_COND_MSG(nodes[i].path == p_path, "Node path already exist in animation: " + String(p_path));
	}
	NodeData n;
	n.path = p_path;
	nodes.push_back(n);
}
NodePath ImporterAnimation::node_get_path(uint32_t p_node) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), NodePath());
	return nodes[p_node].path;
}
void ImporterAnimation::node_set_rotation_mode(uint32_t p_node, RotationMode p_mode) {
	ERR_FAIL_UNSIGNED_INDEX(p_node, nodes.size());
	NodeData *n = &nodes[p_node];
	n->rotation_mode = p_mode;
}

ImporterAnimation::RotationMode ImporterAnimation::node_get_rotation_mode(uint32_t p_node) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), ROTATION_MODE_EULER_XYZ);
	const NodeData *n = &nodes[p_node];
	return n->rotation_mode;
}

void ImporterAnimation::node_set_blend_shape_track_count(uint32_t p_node, uint32_t p_count) {
	ERR_FAIL_UNSIGNED_INDEX(p_node, nodes.size());
	NodeData *n = &nodes[p_node];
	n->tracks.resize(TRACK_TYPE_BLEND_SHAPE_KEY_0 + p_count);
	n->blend_shape_tracks.resize(p_count);
}
uint32_t ImporterAnimation::node_get_blend_shape_track_count(uint32_t p_node) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), 0);
	const NodeData *n = &nodes[p_node];
	return n->blend_shape_tracks.size();
}
void ImporterAnimation::node_set_blend_shape_track_name(uint32_t p_node, uint32_t p_track, const StringName &p_name) {
	ERR_FAIL_UNSIGNED_INDEX(p_node, nodes.size());
	NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX(p_track, n->blend_shape_tracks.size());
	n->blend_shape_tracks[p_track] = p_name;
}
StringName ImporterAnimation::node_get_blend_shape_track_name(uint32_t p_node, uint32_t p_track) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), StringName());
	const NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX_V(p_track, n->blend_shape_tracks.size(), StringName());
	return n->blend_shape_tracks[p_track];
}

void ImporterAnimation::node_track_set_axis_channel_interpolation_mode(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel, InterpolationMode p_mode) {
	ERR_FAIL_UNSIGNED_INDEX(p_node, nodes.size());
	NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX(p_track_type, n->tracks.size());
	ERR_FAIL_UNSIGNED_INDEX(p_channel, 3);
	n->tracks[p_track_type].channels[p_channel].interpolation_mode = p_mode;
}

void ImporterAnimation::node_track_set_quaternion_channel_interpolation_mode(uint32_t p_node, uint32_t p_track_type, InterpolationMode p_mode) {
	ERR_FAIL_UNSIGNED_INDEX(p_node, nodes.size());
	NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX(p_track_type, n->tracks.size());
	n->tracks[p_track_type].quaternion_channel.interpolation_mode = p_mode;
}

ImporterAnimation::InterpolationMode ImporterAnimation::node_track_get_axis_channel_interpolation_mode(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), INTERPOLATION_MODE_NEAREST);
	const NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX_V(p_track_type, n->tracks.size(), INTERPOLATION_MODE_NEAREST);
	ERR_FAIL_UNSIGNED_INDEX_V(p_channel, 3, INTERPOLATION_MODE_NEAREST);
	return n->tracks[p_track_type].channels[p_channel].interpolation_mode;
}

ImporterAnimation::InterpolationMode ImporterAnimation::node_track_get_quaternion_channel_interpolation_mode(uint32_t p_node, uint32_t p_track_type) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), INTERPOLATION_MODE_NEAREST);
	const NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX_V(p_track_type, n->tracks.size(), INTERPOLATION_MODE_NEAREST);

	return n->tracks[p_track_type].quaternion_channel.interpolation_mode;
}

void ImporterAnimation::node_track_add_axis_key(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel, real_t p_time, real_t p_value, real_t p_in_handle_delta_time, real_t p_in_handle_value, real_t p_out_handle_delta_time, real_t p_out_handle_value) {
	ERR_FAIL_UNSIGNED_INDEX(p_node, nodes.size());
	NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX(p_track_type, n->tracks.size());
	ERR_FAIL_UNSIGNED_INDEX(p_channel, 3);

	Key k;
	k.point.time = p_time;
	k.point.value = p_value;
	k.in_handle.time = p_in_handle_delta_time;
	k.in_handle.value = p_in_handle_value;
	k.out_handle.time = p_out_handle_delta_time;
	k.out_handle.value = p_out_handle_value;

	int total_keys = n->tracks[p_track_type].channels[p_channel].keys.size();
	int idx = total_keys - 1;
	while (idx >= 0) {
		real_t t = n->tracks[p_track_type].channels[p_channel].keys[idx].point.time;
		if (p_time > t) {
			break;
		} else if (p_time == t) {
			//same time, overwrite
			n->tracks[p_track_type].channels[p_channel].keys[idx] = k;
			return;
		}
		idx--;
	}
	idx++;
	if (idx == total_keys) {
		n->tracks[p_track_type].channels[p_channel].keys.push_back(k);
	} else {
		n->tracks[p_track_type].channels[p_channel].keys.insert(idx, k);
	}
}

void ImporterAnimation::node_track_add_quaternion_key(uint32_t p_node, uint32_t p_track_type, real_t p_time, const Quaternion &p_quaternion, const Quaternion &p_in, const Quaternion &p_out) {
	ERR_FAIL_UNSIGNED_INDEX(p_node, nodes.size());
	NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX(p_track_type, n->tracks.size());

	QuaternionKey k;
	k.time = p_time;
	k.value = p_quaternion;
	k.in_handle = p_in;
	k.out_handle = p_out;

	int total_keys = n->tracks[p_track_type].quaternion_channel.keys.size();
	int idx = total_keys - 1;
	while (idx >= 0) {
		real_t t = n->tracks[p_track_type].quaternion_channel.keys[idx].time;
		if (p_time > t) {
			break;
		} else if (p_time == t) {
			//same time, overwrite
			n->tracks[p_track_type].quaternion_channel.keys[idx] = k;
			return;
		}
		idx--;
	}
	idx++;
	if (idx == total_keys) {
		n->tracks[p_track_type].quaternion_channel.keys.push_back(k);
	} else {
		n->tracks[p_track_type].quaternion_channel.keys.insert(idx, k);
	}
}

bool ImporterAnimation::node_has_track(uint32_t p_node, uint32_t p_track_type) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), false);
	const NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX_V(p_track_type, n->tracks.size(), false);
	for (int i = 0; i < 3; i++) {
		if (n->tracks[p_track_type].channels[i].keys.size() > 0) {
			return true;
		}
	}
	return n->tracks[p_track_type].quaternion_channel.keys.size();
}

bool ImporterAnimation::node_track_has_axis_channel(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), false);
	const NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX_V(p_track_type, n->tracks.size(), false);
	ERR_FAIL_UNSIGNED_INDEX_V(p_channel, 3, false);
	return n->tracks[p_track_type].channels[p_channel].keys.size() > 0;
}

bool ImporterAnimation::node_track_has_quaternion_channel(uint32_t p_node, uint32_t p_track_type) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), false);
	const NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX_V(p_track_type, n->tracks.size(), false);
	return n->tracks[p_track_type].quaternion_channel.keys.size() > 0;
}

bool ImporterAnimation::node_has_transform_tracks(uint32_t p_node) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), false);
	const NodeData *n = &nodes[p_node];
	for (int j = 0; j < TRACK_TYPE_BLEND_SHAPE_KEY_0; j++) {
		for (int i = 0; i < 3; i++) {
			if (n->tracks[j].channels[i].keys.size() > 0) {
				return true;
			}
		}

		if (n->tracks[j].quaternion_channel.keys.size()) {
			return true;
		}
	}

	return false;
}

int ImporterAnimation::node_track_channel_get_axis_key_count(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), -1);
	const NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX_V(p_track_type, n->tracks.size(), -1);
	ERR_FAIL_UNSIGNED_INDEX_V(p_channel, 3, -1);
	return n->tracks[p_track_type].channels[p_channel].keys.size();
}

int ImporterAnimation::node_track_channel_get_quaternion_key_count(uint32_t p_node, uint32_t p_track_type) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), -1);
	const NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX_V(p_track_type, n->tracks.size(), -1);
	return n->tracks[p_track_type].quaternion_channel.keys.size();
}

real_t ImporterAnimation::node_track_channel_get_axis_key_time(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel, uint32_t p_key) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), 0.0);
	const NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX_V(p_track_type, n->tracks.size(), 0.0);
	ERR_FAIL_UNSIGNED_INDEX_V(p_channel, 3, 0.0);
	ERR_FAIL_UNSIGNED_INDEX_V(p_key, n->tracks[p_track_type].channels[p_channel].keys.size(), 0.0);
	return n->tracks[p_track_type].channels[p_channel].keys[p_key].point.time;
}

real_t ImporterAnimation::node_track_channel_get_quaternion_key_time(uint32_t p_node, uint32_t p_track_type, uint32_t p_key) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), 0.0);
	const NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX_V(p_track_type, n->tracks.size(), 0.0);
	ERR_FAIL_UNSIGNED_INDEX_V(p_key, n->tracks[p_track_type].quaternion_channel.keys.size(), 0.0);
	return n->tracks[p_track_type].quaternion_channel.keys[p_key].time;
}

real_t ImporterAnimation::node_track_channel_get_axis_key_value(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel, uint32_t p_key) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), 0.0);
	const NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX_V(p_track_type, n->tracks.size(), 0.0);
	ERR_FAIL_UNSIGNED_INDEX_V(p_channel, 3, 0.0);
	ERR_FAIL_UNSIGNED_INDEX_V(p_key, n->tracks[p_track_type].channels[p_channel].keys.size(), 0.0);
	return n->tracks[p_track_type].channels[p_channel].keys[p_key].point.value;
}

real_t ImporterAnimation::node_track_channel_get_axis_key_in_handle_time(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel, uint32_t p_key) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), 0.0);
	const NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX_V(p_track_type, n->tracks.size(), 0.0);
	ERR_FAIL_UNSIGNED_INDEX_V(p_channel, 3, 0.0);
	ERR_FAIL_UNSIGNED_INDEX_V(p_key, n->tracks[p_track_type].channels[p_channel].keys.size(), 0.0);
	return n->tracks[p_track_type].channels[p_channel].keys[p_key].in_handle.time;
}

real_t ImporterAnimation::node_track_channel_get_axis_key_in_handle_value(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel, uint32_t p_key) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), 0.0);
	const NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX_V(p_track_type, n->tracks.size(), 0.0);
	ERR_FAIL_UNSIGNED_INDEX_V(p_channel, 3, 0.0);
	ERR_FAIL_UNSIGNED_INDEX_V(p_key, n->tracks[p_track_type].channels[p_channel].keys.size(), 0.0);
	return n->tracks[p_track_type].channels[p_channel].keys[p_key].in_handle.value;
}

real_t ImporterAnimation::node_track_channel_get_axis_key_out_handle_time(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel, uint32_t p_key) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), 0.0);
	const NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX_V(p_track_type, n->tracks.size(), 0.0);
	ERR_FAIL_UNSIGNED_INDEX_V(p_channel, 3, 0.0);
	ERR_FAIL_UNSIGNED_INDEX_V(p_key, n->tracks[p_track_type].channels[p_channel].keys.size(), 0.0);
	return n->tracks[p_track_type].channels[p_channel].keys[p_key].out_handle.time;
}

real_t ImporterAnimation::node_track_channel_get_axis_key_out_handle_value(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel, uint32_t p_key) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), 0.0);
	const NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX_V(p_track_type, n->tracks.size(), 0.0);
	ERR_FAIL_UNSIGNED_INDEX_V(p_channel, 3, 0.0);
	ERR_FAIL_UNSIGNED_INDEX_V(p_key, n->tracks[p_track_type].channels[p_channel].keys.size(), 0.0);
	return n->tracks[p_track_type].channels[p_channel].keys[p_key].out_handle.value;
}

Quaternion ImporterAnimation::node_track_channel_get_quaternion_key_value(uint32_t p_node, uint32_t p_track_type, uint32_t p_key) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), Quaternion());
	const NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX_V(p_track_type, n->tracks.size(), Quaternion());
	ERR_FAIL_UNSIGNED_INDEX_V(p_key, n->tracks[p_track_type].quaternion_channel.keys.size(), Quaternion());
	return n->tracks[p_track_type].quaternion_channel.keys[p_key].value;
}

Quaternion ImporterAnimation::node_track_channel_get_quaternion_key_in_handle(uint32_t p_node, uint32_t p_track_type, uint32_t p_key) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), Quaternion());
	const NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX_V(p_track_type, n->tracks.size(), Quaternion());
	ERR_FAIL_UNSIGNED_INDEX_V(p_key, n->tracks[p_track_type].quaternion_channel.keys.size(), Quaternion());
	return n->tracks[p_track_type].quaternion_channel.keys[p_key].in_handle;
}

Quaternion ImporterAnimation::node_track_channel_get_quaternion_key_out_handle(uint32_t p_node, uint32_t p_track_type, uint32_t p_key) const {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), Quaternion());
	const NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX_V(p_track_type, n->tracks.size(), Quaternion());
	ERR_FAIL_UNSIGNED_INDEX_V(p_key, n->tracks[p_track_type].quaternion_channel.keys.size(), Quaternion());
	return n->tracks[p_track_type].quaternion_channel.keys[p_key].out_handle;
}

void ImporterAnimation::node_track_channel_remove_axis_key(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel, uint32_t p_key) {
	ERR_FAIL_UNSIGNED_INDEX(p_node, nodes.size());
	NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX(p_track_type, n->tracks.size());
	ERR_FAIL_UNSIGNED_INDEX(p_channel, 3);
	ERR_FAIL_UNSIGNED_INDEX(p_key, n->tracks[p_track_type].channels[p_channel].keys.size());
	n->tracks[p_track_type].channels[p_channel].keys.remove(p_key);
}

void ImporterAnimation::node_track_channel_remove_quaternion_key(uint32_t p_node, uint32_t p_track_type, uint32_t p_key) {
	ERR_FAIL_UNSIGNED_INDEX(p_node, nodes.size());
	NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX(p_track_type, n->tracks.size());
	ERR_FAIL_UNSIGNED_INDEX(p_key, n->tracks[p_track_type].quaternion_channel.keys.size());
	n->tracks[p_track_type].quaternion_channel.keys.remove(p_key);
}

void ImporterAnimation::node_track_remove_axis_channel(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel) {
	ERR_FAIL_UNSIGNED_INDEX(p_node, nodes.size());
	NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX(p_track_type, n->tracks.size());
	ERR_FAIL_UNSIGNED_INDEX(p_channel, 3);
	n->tracks[p_track_type].channels[p_channel].keys.clear();
}
void ImporterAnimation::node_track_remove_quaternion_channel(uint32_t p_node, uint32_t p_track_type) {
	ERR_FAIL_UNSIGNED_INDEX(p_node, nodes.size());
	NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX(p_track_type, n->tracks.size());
	n->tracks[p_track_type].quaternion_channel.keys.clear();
}

void ImporterAnimation::node_remove_track(uint32_t p_node, uint32_t p_track_type) {
	ERR_FAIL_UNSIGNED_INDEX(p_node, nodes.size());
	NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX(p_track_type, n->tracks.size());
	for (int i = 0; i < 3; i++) {
		n->tracks[p_track_type].channels[i].keys.clear();
	}
	n->tracks[p_track_type].quaternion_channel.keys.clear();
}

void ImporterAnimation::remove_node(uint32_t p_node) {
	ERR_FAIL_UNSIGNED_INDEX(p_node, nodes.size());
	nodes.remove(p_node);
}

static _FORCE_INLINE_ real_t interp_linear(const real_t &a, const real_t &b, real_t c) {
	return a + (b - a) * c;
}

static _FORCE_INLINE_ real_t interp_crom(const real_t &p0, const real_t &p1, const real_t &p2, const real_t &p3, real_t t) {
	const real_t t2 = t * t;
	const real_t t3 = t2 * t;

	return 0.5f * ((2.0f * p1) + (-p0 + p2) * t + (2.0f * p0 - 5.0f * p1 + 4.0f * p2 - p3) * t2 + (-p0 + 3.0f * p1 - 3.0f * p2 + p3) * t3);
}

template <class T>
static _FORCE_INLINE_ T interp_bezier(real_t t, const T &start, const T &control_1, const T &control_2, const T &end) {
	/* Formula from Wikipedia article on Bezier curves. */
	real_t omt = (1.0 - t);
	real_t omt2 = omt * omt;
	real_t omt3 = omt2 * omt;
	real_t t2 = t * t;
	real_t t3 = t2 * t;

	return start * omt3 + control_1 * omt2 * t * 3.0 + control_2 * omt * t2 * 3.0 + end * t3;
}

real_t ImporterAnimation::node_track_axis_interpolate(uint32_t p_node, uint32_t p_track_type, Vector3::Axis p_channel, real_t p_time) {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), 0.0);
	const NodeData *n = &nodes[p_node];
	ERR_FAIL_UNSIGNED_INDEX_V(p_track_type, n->tracks.size(), 0.0);
	ERR_FAIL_UNSIGNED_INDEX_V(p_channel, 3, 0.0);

	const Channel *c = &n->tracks[p_track_type].channels[p_channel];

	p_time = CLAMP(p_time, 0, length);

	if (c->keys.size() == 0) {
		return 0.0;
	}

	if (c->keys.size() == 1) {
		return c->keys[0].point.value;
	}

	Key prev;
	Key next;

	//could use binary search, worth it?
	int idx = -1;
	for (uint32_t i = 0; i < c->keys.size(); i++) {
		if (p_time > c->keys[i].point.time) {
			break;
		}
		idx++;
	}

	if (idx == -1) {
		//before first
		if (loop_mode == LOOP_MODE_FORWARD) {
			prev = c->keys[c->keys.size() - 1];
			prev.point.time -= length;
			next = c->keys[0];
		} else {
			return c->keys[0].point.value;
		}
	} else if (idx == (int)c->keys.size() - 1 || c->keys[idx + 1].point.time > length) {
		//after last
		if (loop_mode == LOOP_MODE_FORWARD) {
			prev = c->keys[idx];
			next = c->keys[0];
			next.point.time += length;
		} else if (idx == (int)c->keys.size() - 1) {
			return c->keys[idx].point.value;
		} else {
			prev = c->keys[idx];
			next = c->keys[idx + 1];
		}
	} else {
		prev = c->keys[idx];
		next = c->keys[idx + 1];
	}

	switch (c->interpolation_mode) {
		case INTERPOLATION_MODE_NEAREST: {
			return prev.point.value;
		} break;
		case INTERPOLATION_MODE_LINEAR: {
			const real_t cf = (p_time - prev.point.time) / (next.point.time - prev.point.time);

			return interp_linear(prev.point.value, next.point.value, cf);
		} break;
		case INTERPOLATION_MODE_CATMULL_ROM_SPLINE: {
			const real_t cf = (p_time - prev.point.time) / (next.point.time - prev.point.time);

			real_t pre_prev = prev.in_handle.value;
			real_t post_next = next.out_handle.value;
			return interp_crom(pre_prev, prev.point.value, next.point.value, post_next, cf);
		} break;
		case INTERPOLATION_MODE_CUBIC_SPLINE: {
			double t = p_time - prev.point.time;

			int iterations = 10;

			real_t duration = next.point.time - prev.point.time; // time duration between our two keyframes
			real_t low = 0.0; // 0% of the current animation segment
			real_t high = 1.0; // 100% of the current animation segment
			real_t middle;

			Vector2 start(0, prev.point.value);
			Vector2 start_out = start + Vector2(prev.out_handle.time, prev.out_handle.value);
			Vector2 end(duration, next.point.value);
			Vector2 end_in = end + Vector2(next.in_handle.time, next.in_handle.value);

			//narrow high and low as much as possible
			for (int i = 0; i < iterations; i++) {
				middle = (low + high) / 2;

				Vector2 interp = interp_bezier(middle, start, start_out, end_in, end);

				if (interp.x < t) {
					low = middle;
				} else {
					high = middle;
				}
			}

			//interpolate the result:
			Vector2 low_pos = interp_bezier(low, start, start_out, end_in, end);
			Vector2 high_pos = interp_bezier(high, start, start_out, end_in, end);
			real_t cf = (t - low_pos.x) / (high_pos.x - low_pos.x);

			return low_pos.lerp(high_pos, cf).y;
		} break;
		case INTERPOLATION_MODE_GLTF2_SPLINE: {
			const real_t cf = (p_time - prev.point.time) / (next.point.time - prev.point.time);

			real_t c1 = prev.out_handle.value;
			real_t c2 = next.in_handle.value;
			return interp_bezier(cf, prev.point.value, c1, c2, next.point.value);
		} break;
		default: {
		}
	}

	return 0.0;
}

Vector3 ImporterAnimation::node_track_interpolate_translation(uint32_t p_node, real_t p_time) {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), Vector3());

	Vector3 translation;
	for (int i = 0; i < 3; i++) {
		if (node_track_has_axis_channel(p_node, TRACK_TYPE_TRANSLATION, Vector3::Axis(i))) {
			translation[i] = node_track_axis_interpolate(p_node, TRACK_TYPE_TRANSLATION, Vector3::Axis(i), p_time);
		}
	}

	return translation;
}

Quaternion ImporterAnimation::node_track_interpolate_rotation(uint32_t p_node, real_t p_time) {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), Quaternion());
	const NodeData *n = &nodes[p_node];

	if (!node_has_track(p_node, TRACK_TYPE_ROTATION)) {
		return Quaternion();
	}

	if (n->rotation_mode == ROTATION_MODE_QUATERNION) {
		//quaternion interpolation
		if (!node_track_has_quaternion_channel(p_node, TRACK_TYPE_ROTATION)) {
			return Quaternion();
		}

		const QuaternionChannel *c = &n->tracks[TRACK_TYPE_ROTATION].quaternion_channel;

		p_time = CLAMP(p_time, 0, length);

		if (c->keys.size() == 0) {
			return Quaternion();
		}

		if (c->keys.size() == 1) {
			return c->keys[0].value;
		}

		QuaternionKey prev;
		QuaternionKey next;

		//could use binary search, worth it?
		int idx = -1;
		for (uint32_t i = 0; i < c->keys.size(); i++) {
			if (p_time > c->keys[i].time) {
				break;
			}
			idx++;
		}

		if (idx == -1) {
			//before first
			if (loop_mode == LOOP_MODE_FORWARD) {
				prev = c->keys[c->keys.size() - 1];
				prev.time -= length;
				next = c->keys[0];
			} else {
				return c->keys[0].value;
			}
		} else if (idx == (int)c->keys.size() - 1 || c->keys[idx + 1].time > length) {
			//after last
			if (loop_mode == LOOP_MODE_FORWARD) {
				prev = c->keys[idx];
				next = c->keys[0];
				next.time += length;
			} else if (idx == (int)c->keys.size() - 1) {
				return c->keys[idx].value;
			} else {
				prev = c->keys[idx];
				next = c->keys[idx + 1];
			}
		} else {
			prev = c->keys[idx];
			next = c->keys[idx + 1];
		}

		switch (c->interpolation_mode) {
			case INTERPOLATION_MODE_NEAREST: {
				return prev.value;
			} break;
			default: {
				//the other modes, no idea how to do this with a quaternion.
				const real_t cf = (p_time - prev.time) / (next.time - prev.time);
				return prev.value.slerp(next.value, cf);
			} break;
		}

	} else {
		Vector3 euler;
		for (int i = 0; i < 3; i++) {
			if (node_track_has_axis_channel(p_node, TRACK_TYPE_ROTATION, Vector3::Axis(i))) {
				euler[i] = node_track_axis_interpolate(p_node, TRACK_TYPE_ROTATION, Vector3::Axis(i), p_time);
			}
		}
		Basis b;
		switch (n->rotation_mode) {
			case ROTATION_MODE_EULER_XYZ: {
				b.set_euler_xyz(euler);
			} break;
			case ROTATION_MODE_EULER_XZY: {
				b.set_euler_xzy(euler);
			} break;
			case ROTATION_MODE_EULER_YXZ: {
				b.set_euler_yxz(euler);
			} break;
			case ROTATION_MODE_EULER_YZX: {
				b.set_euler_yzx(euler);
			} break;
			case ROTATION_MODE_EULER_ZXY: {
				b.set_euler_zxy(euler);
			} break;
			case ROTATION_MODE_EULER_ZYX: {
				b.set_euler_zyx(euler);
			} break;
			default: {
			}
		}
		return b.operator Quaternion();
	}
	return Quaternion();
}

Vector3 ImporterAnimation::node_track_interpolate_scale(uint32_t p_node, real_t p_time) {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), Vector3());

	Vector3 scale(1, 1, 1); //scale is 1 by default
	for (int i = 0; i < 3; i++) {
		if (node_track_has_axis_channel(p_node, TRACK_TYPE_SCALE, Vector3::Axis(i))) {
			scale[i] = node_track_axis_interpolate(p_node, TRACK_TYPE_SCALE, Vector3::Axis(i), p_time);
		}
	}

	return scale;
}

float ImporterAnimation::node_track_interpolate_blend_shape(uint32_t p_node, uint32_t p_blend_shape, real_t p_time) {
	ERR_FAIL_UNSIGNED_INDEX_V(p_node, nodes.size(), 0);

	if (node_track_has_axis_channel(p_node, TRACK_TYPE_BLEND_SHAPE_KEY_0 + p_blend_shape, Vector3::AXIS_X)) {
		return node_track_axis_interpolate(p_node, TRACK_TYPE_BLEND_SHAPE_KEY_0 + p_blend_shape, Vector3::AXIS_X, p_time);
	}
	return 0;
}

void ImporterAnimation::clear() {
	nodes.clear();
}

void ImporterAnimation::set_loop_mode(LoopMode p_mode) {
	ERR_FAIL_UNSIGNED_INDEX(p_mode, 3);
	loop_mode = p_mode;
}

ImporterAnimation::LoopMode ImporterAnimation::get_loop_mode() const {
	return loop_mode;
}

void ImporterAnimation::set_length(real_t p_length) {
	length = p_length;
}

real_t ImporterAnimation::get_length() const {
	return length;
}

static String _get_track_name(uint32_t p_track) {
	static const char *track_names[ImporterAnimation::TRACK_TYPE_MAX] = {
		"position_track", "rotation_track", "scale_track", "blend_shape_key_track"
	};

	String name = track_names[MIN(p_track, ImporterAnimation::TRACK_TYPE_MAX - 1)];
	if (p_track >= ImporterAnimation::TRACK_TYPE_BLEND_SHAPE_KEY_0) {
		name += itos(ImporterAnimation::TRACK_TYPE_BLEND_SHAPE_KEY_0 - p_track);
	}
	return name;
}

void ImporterAnimation::_set_nodes(const Array &p_nodes) {
	nodes.clear();
	for (int i = 0; i < p_nodes.size(); i++) {
		Dictionary d;
		ERR_FAIL_COND(!d.has("path"));
		ERR_FAIL_COND(!d.has("rotation_mode"));
		add_node(d["path"]);
		node_set_rotation_mode(i, RotationMode(int(d["rotation_mode"])));
		Vector<String> blend_shapes;
		if (d.has("blend_shapes")) {
			blend_shapes = d["blend_shapes"];
			node_set_blend_shape_track_count(i, blend_shapes.size());
			for (int j = 0; j < blend_shapes.size(); j++) {
				node_set_blend_shape_track_name(i, j, blend_shapes[j]);
			}
		}

		for (int j = 0; j < TRACK_TYPE_BLEND_SHAPE_KEY_0 + blend_shapes.size(); j++) {
			if (!d.has(_get_track_name(j))) {
				continue;
			}

			Variant td = d[_get_track_name(j)];
			if (td.get_type() == Variant::ARRAY) { //axis keys
				Array cs = td;
				ERR_FAIL_COND(cs.size() != 3);
				for (int k = 0; k < 3; k++) {
					Dictionary c = cs[k];
					if (c.size() == 0) {
						continue;
					}
					ERR_CONTINUE(!c.has("interpolation_mode"));
					ERR_CONTINUE(!c.has("keys"));
					node_track_set_axis_channel_interpolation_mode(i, j, Vector3::Axis(k), InterpolationMode(int(c["interpolation_mode"])));
					Vector<real_t> keys = c["keys"];
					ERR_FAIL_COND(keys.size() % 6 != 0);

					for (int l = 0; l < keys.size(); l++) {
						node_track_add_axis_key(i, j, Vector3::Axis(k), keys[l * 6 + 0], keys[l * 6 + 1], keys[l * 6 + 2], keys[l * 6 + 3], keys[l * 6 + 4], keys[l * 6 + 5]);
					}
				}
			} else if (td.get_type() == Variant::DICTIONARY) {
				//quaternion keys
				Dictionary c = td;
				if (c.size() == 0) {
					continue;
				}
				ERR_CONTINUE(!c.has("interpolation_mode"));
				ERR_CONTINUE(!c.has("keys"));
				node_track_set_quaternion_channel_interpolation_mode(i, j, InterpolationMode(int(c["interpolation_mode"])));
				Array keys = c["keys"];
				ERR_FAIL_COND(keys.size() % 4 != 0);
				for (int l = 0; l < keys.size(); l++) {
					real_t time = keys[l * 4 + 0];
					Quaternion value = keys[l * 4 + 1];
					Quaternion in_handle = keys[l * 4 + 2];
					Quaternion out_handle = keys[l * 4 + 3];
					node_track_add_quaternion_key(i, j, time, value, in_handle, out_handle);
				}
			}
		}
	}
}

Array ImporterAnimation::_get_nodes() const {
	Array ret;

	for (uint32_t i = 0; i < nodes.size(); i++) {
		const NodeData *n = &nodes[i];
		Dictionary d;
		d["path"] = n->path;
		d["rotation_mode"] = n->rotation_mode;

		Vector<String> blend_shapes;
		for (uint32_t j = 0; j < node_get_blend_shape_track_count(i); j++) {
			blend_shapes.push_back(node_get_blend_shape_track_name(i, j));
		}

		if (blend_shapes.size()) {
			d["blend_shapes"] = blend_shapes;
		}

		for (uint32_t j = 0; j < n->tracks.size(); j++) {
			if (!node_has_track(i, j)) {
				continue;
			}

			if (node_track_has_quaternion_channel(i, j)) {
				Dictionary c;
				c["interpolation_mode"] = n->tracks[j].quaternion_channel.interpolation_mode;
				Array keys;
				keys.resize(n->tracks[j].quaternion_channel.keys.size() * 4);
				for (uint32_t l = 0; l < n->tracks[j].quaternion_channel.keys.size(); l++) {
					keys[l * 4 + 0] = n->tracks[j].quaternion_channel.keys[l].time;
					keys[l * 4 + 1] = n->tracks[j].quaternion_channel.keys[l].value;
					keys[l * 4 + 2] = n->tracks[j].quaternion_channel.keys[l].in_handle;
					keys[l * 4 + 3] = n->tracks[j].quaternion_channel.keys[l].out_handle;
				}

				c["keys"] = keys;

				d[_get_track_name(j)] = c;
			} else {
				Array cs;
				cs.resize(3);
				for (int k = 0; k < 3; k++) {
					if (!node_track_has_axis_channel(i, j, Vector3::Axis(k))) {
						continue;
					}
					Dictionary c;
					c["interpolation_mode"] = n->tracks[j].channels[k].interpolation_mode;
					Vector<real_t> keys;
					keys.resize(n->tracks[j].channels[k].keys.size() * 6);
					for (uint32_t l = 0; l < n->tracks[j].channels[k].keys.size(); l++) {
						keys.write[l * 6 + 0] = n->tracks[j].channels[k].keys[l].point.time;
						keys.write[l * 6 + 1] = n->tracks[j].channels[k].keys[l].point.value;
						keys.write[l * 6 + 2] = n->tracks[j].channels[k].keys[l].in_handle.time;
						keys.write[l * 6 + 3] = n->tracks[j].channels[k].keys[l].in_handle.value;
						keys.write[l * 6 + 4] = n->tracks[j].channels[k].keys[l].out_handle.time;
						keys.write[l * 6 + 5] = n->tracks[j].channels[k].keys[l].out_handle.value;
					}

					c["keys"] = keys;
					cs[k] = c;
				}

				d[_get_track_name(j)] = cs;
			}
		}
		ret.push_back(d);
	}
	return ret;
}

Ref<Animation> ImporterAnimation::bake(int p_bake_fps, Node *p_base_path, bool p_relative_to_rest) {
	Ref<Animation> anim;
	anim.instantiate();

	anim->set_length(length);

	Vector<real_t> base_snapshots;
	{
		float f = 0;
		float snapshot_interval = 1.0 / p_bake_fps; //should be customizable somewhere...

		while (f < length) {
			base_snapshots.push_back(f);

			f += snapshot_interval;

			if (f >= length) {
				base_snapshots.push_back(length);
			}
		}
	}

	for (int i = 0; i < get_node_count(); i++) {
		NodePath path = node_get_path(i);
		Node *n = p_base_path->get_node(path);
		ERR_CONTINUE(n);

		Node3D *n3d = Object::cast_to<Node3D>(n);
		if (!n3d) {
			continue; //nothing to do with this track
		}
		Transform3D rest;
		bool has_rest = false;
		if (p_relative_to_rest) {
			Skeleton3D *sn = Object::cast_to<Skeleton3D>(n3d);
			if (sn && path.get_subnames().size() == 1) {
				StringName bone = path.get_subname(0);
				int idx = sn->find_bone(bone);
				if (idx >= 0) {
					rest = sn->get_bone_rest(idx).affine_inverse();
					has_rest = true;
				}
			}
		}

		if (node_has_transform_tracks(i)) {
			int track_idx = anim->get_track_count();

			anim->add_track(Animation::TYPE_TRANSFORM3D);
			anim->track_set_path(track_idx, path);

			for (int j = 0; j < base_snapshots.size(); j++) {
				float t = base_snapshots[j];
				Quaternion rot = node_track_interpolate_rotation(i, j);
				Vector3 loc = node_track_interpolate_translation(i, j);
				Vector3 scale = node_track_interpolate_translation(i, j);
				if (has_rest) {
					Transform3D xform;
					xform.origin = loc;
					xform.basis.set_quaternion_scale(rot, scale);
					xform = rest * t;

					scale = xform.basis.get_scale();
					bool singular_matrix = Math::is_zero_approx(scale.x) || Math::is_zero_approx(scale.y) || Math::is_zero_approx(scale.z);
					rot = singular_matrix ? Quaternion() : xform.basis.get_rotation_quaternion();
					loc = xform.origin;
				}
				anim->transform_track_insert_key(track_idx, t, loc, rot, scale);
			}
		}

		int bscount = node_get_blend_shape_track_count(i);
		if (bscount) {
			for (int k = 0; k < bscount; k++) {
				int track_idx = anim->get_track_count();

				anim->add_track(Animation::TYPE_VALUE);
				NodePath bspath = String(NodePath(path.get_names(), false)) + ":blend_shapes/" + String(node_get_blend_shape_track_name(i, k));
				anim->track_set_path(track_idx, bspath);

				for (int j = 0; j < base_snapshots.size(); j++) {
					float t = base_snapshots[j];
					float s = node_track_interpolate_blend_shape(i, j, t);
					anim->track_insert_key(track_idx, t, s);
				}
			}
		}
	}

	return anim;
}

void ImporterAnimation::_bind_methods() {
	ClassDB::bind_method(D_METHOD("get_node_count"), &ImporterAnimation::get_node_count);
	ClassDB::bind_method(D_METHOD("add_node", "path"), &ImporterAnimation::add_node);
	ClassDB::bind_method(D_METHOD("node_get_path", "node_idx"), &ImporterAnimation::node_get_path);
	ClassDB::bind_method(D_METHOD("node_set_rotation_mode", "node_idx", "rotation_mode"), &ImporterAnimation::node_set_rotation_mode);
	ClassDB::bind_method(D_METHOD("node_get_rotation_mode", "node_idx"), &ImporterAnimation::node_get_rotation_mode);

	ClassDB::bind_method(D_METHOD("node_set_blend_shape_track_count", "node_idx", "count"), &ImporterAnimation::node_set_blend_shape_track_count);
	ClassDB::bind_method(D_METHOD("node_get_blend_shape_track_count", "node_idx"), &ImporterAnimation::node_get_blend_shape_track_count);

	ClassDB::bind_method(D_METHOD("node_set_blend_shape_track_name", "node_idx", "blend_shape_track_idx", "name"), &ImporterAnimation::node_set_blend_shape_track_name);
	ClassDB::bind_method(D_METHOD("node_get_blend_shape_track_name", "node_idx", "blend_shape_track_idx"), &ImporterAnimation::node_get_blend_shape_track_name);

	ClassDB::bind_method(D_METHOD("node_track_set_axis_channel_interpolation_mode", "node_idx", "track_type", "axis_channel", "mode"), &ImporterAnimation::node_track_set_axis_channel_interpolation_mode);
	ClassDB::bind_method(D_METHOD("node_track_get_axis_channel_interpolation_mode", "node_idx", "track_type", "axis_channel"), &ImporterAnimation::node_track_get_axis_channel_interpolation_mode);

	ClassDB::bind_method(D_METHOD("node_track_set_quaternion_channel_interpolation_mode", "node_idx", "track_type", "mode"), &ImporterAnimation::node_track_set_quaternion_channel_interpolation_mode);
	ClassDB::bind_method(D_METHOD("node_track_get_quaternion_channel_interpolation_mode", "node_idx", "track_type"), &ImporterAnimation::node_track_get_quaternion_channel_interpolation_mode);

	ClassDB::bind_method(D_METHOD("node_track_add_axis_key", "node_idx", "track_type", "axis", "time", "value", "in_handle_dt", "in_handle_value", "out_handle_dt", "out_handle_value"), &ImporterAnimation::node_track_add_axis_key, DEFVAL(0.0), DEFVAL(0.0), DEFVAL(0.0), DEFVAL(0.0));
	ClassDB::bind_method(D_METHOD("node_track_add_quaternion_key", "node_idx", "track_type", "axis", "time", "value", "in_handle", "out_handle"), &ImporterAnimation::node_track_add_quaternion_key, DEFVAL(Quaternion()), DEFVAL(Quaternion()));

	ClassDB::bind_method(D_METHOD("node_has_track", "node_idx", "track_type"), &ImporterAnimation::node_has_track);
	ClassDB::bind_method(D_METHOD("node_track_has_axis_channel", "node_idx", "track_type", "axis"), &ImporterAnimation::node_track_has_axis_channel);
	ClassDB::bind_method(D_METHOD("node_track_has_quaternion_channel", "node_idx", "track_type"), &ImporterAnimation::node_track_has_quaternion_channel);

	ClassDB::bind_method(D_METHOD("node_track_channel_get_axis_key_count", "node_idx", "track_type", "axis"), &ImporterAnimation::node_track_channel_get_axis_key_count);
	ClassDB::bind_method(D_METHOD("node_track_channel_get_axis_key_time", "node_idx", "track_type", "axis", "key_idx"), &ImporterAnimation::node_track_channel_get_axis_key_time);
	ClassDB::bind_method(D_METHOD("node_track_channel_get_axis_key_value", "node_idx", "track_type", "axis", "key_idx"), &ImporterAnimation::node_track_channel_get_axis_key_value);

	ClassDB::bind_method(D_METHOD("node_track_channel_get_axis_key_in_handle_time", "node_idx", "track_type", "axis", "key_idx"), &ImporterAnimation::node_track_channel_get_axis_key_in_handle_time);
	ClassDB::bind_method(D_METHOD("node_track_channel_get_axis_key_in_handle_value", "node_idx", "track_type", "axis", "key_idx"), &ImporterAnimation::node_track_channel_get_axis_key_in_handle_value);
	ClassDB::bind_method(D_METHOD("node_track_channel_get_axis_key_out_handle_time", "node_idx", "track_type", "axis", "key_idx"), &ImporterAnimation::node_track_channel_get_axis_key_out_handle_time);
	ClassDB::bind_method(D_METHOD("node_track_channel_get_axis_key_out_handle_value", "node_idx", "track_type", "axis", "key_idx"), &ImporterAnimation::node_track_channel_get_axis_key_out_handle_value);

	ClassDB::bind_method(D_METHOD("node_track_channel_remove_axis_key", "node_idx", "track_type", "axis", "key_idx"), &ImporterAnimation::node_track_channel_remove_axis_key);
	ClassDB::bind_method(D_METHOD("node_track_channel_remove_quaternion_key", "node_idx", "track_type", "axis", "key_idx"), &ImporterAnimation::node_track_channel_remove_quaternion_key);

	ClassDB::bind_method(D_METHOD("node_track_remove_axis_channel", "node_idx", "track_type", "axis"), &ImporterAnimation::node_track_remove_axis_channel);
	ClassDB::bind_method(D_METHOD("node_track_remove_quaternion_channel", "node_idx", "track_type"), &ImporterAnimation::node_track_remove_quaternion_channel);
	ClassDB::bind_method(D_METHOD("node_remove_track", "node_idx", "track_type"), &ImporterAnimation::node_remove_track);
	ClassDB::bind_method(D_METHOD("remove_node", "node_idx"), &ImporterAnimation::remove_node);

	ClassDB::bind_method(D_METHOD("node_track_axis_interpolate", "node_idx", "track_type", "axis", "time"), &ImporterAnimation::node_track_axis_interpolate);

	ClassDB::bind_method(D_METHOD("node_track_interpolate_translation", "node_idx", "time"), &ImporterAnimation::node_track_interpolate_translation);
	ClassDB::bind_method(D_METHOD("node_track_interpolate_rotation", "node_idx", "time"), &ImporterAnimation::node_track_interpolate_rotation);
	ClassDB::bind_method(D_METHOD("node_track_interpolate_scale", "node_idx", "time"), &ImporterAnimation::node_track_interpolate_scale);
	ClassDB::bind_method(D_METHOD("node_track_interpolate_blend_shape", "node_idx", "blend_shape_index", "time"), &ImporterAnimation::node_track_interpolate_blend_shape);

	ClassDB::bind_method(D_METHOD("clear"), &ImporterAnimation::clear);

	ClassDB::bind_method(D_METHOD("set_loop_mode", "loop_mode"), &ImporterAnimation::set_loop_mode);
	ClassDB::bind_method(D_METHOD("get_loop_mode"), &ImporterAnimation::get_loop_mode);

	ClassDB::bind_method(D_METHOD("set_length", "length"), &ImporterAnimation::set_length);
	ClassDB::bind_method(D_METHOD("get_length"), &ImporterAnimation::get_length);

	ClassDB::bind_method(D_METHOD("_set_nodes", "nodes"), &ImporterAnimation::_set_nodes);
	ClassDB::bind_method(D_METHOD("_get_nodes"), &ImporterAnimation::_get_nodes);

	ADD_PROPERTY(PropertyInfo(Variant::INT, "loop_mode", PROPERTY_HINT_ENUM, "Disabled,Forward,PingPong"), "set_loop_mode", "get_loop_mode");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "length", PROPERTY_HINT_RANGE, "0.001,99999.0,0.001"), "set_length", "get_length");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "_nodes", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "_set_nodes", "_get_nodes");

	BIND_ENUM_CONSTANT(TRACK_TYPE_TRANSLATION);
	BIND_ENUM_CONSTANT(TRACK_TYPE_ROTATION);
	BIND_ENUM_CONSTANT(TRACK_TYPE_SCALE);
	BIND_ENUM_CONSTANT(TRACK_TYPE_BLEND_SHAPE_KEY_0);
	BIND_ENUM_CONSTANT(TRACK_TYPE_MAX);

	BIND_ENUM_CONSTANT(ROTATION_MODE_EULER_XYZ);
	BIND_ENUM_CONSTANT(ROTATION_MODE_EULER_XZY);
	BIND_ENUM_CONSTANT(ROTATION_MODE_EULER_YXZ);
	BIND_ENUM_CONSTANT(ROTATION_MODE_EULER_YZX);
	BIND_ENUM_CONSTANT(ROTATION_MODE_EULER_ZXY);
	BIND_ENUM_CONSTANT(ROTATION_MODE_EULER_ZYX);
	BIND_ENUM_CONSTANT(ROTATION_MODE_QUATERNION);

	BIND_ENUM_CONSTANT(INTERPOLATION_MODE_NEAREST);
	BIND_ENUM_CONSTANT(INTERPOLATION_MODE_LINEAR);
	BIND_ENUM_CONSTANT(INTERPOLATION_MODE_CUBIC_SPLINE);

	BIND_ENUM_CONSTANT(LOOP_MODE_DISABLED);
	BIND_ENUM_CONSTANT(LOOP_MODE_FORWARD);
	BIND_ENUM_CONSTANT(LOOP_MODE_PINGPONG);
}

ImporterAnimation::ImporterAnimation() {
}
