/**************************************************************************/
/*  animation_blend_space_1d.cpp                                          */
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

#include "animation_blend_space_1d.h"
#include "animation_blend_space_1d.compat.inc"

#include "core/object/class_db.h"
#include "scene/animation/animation_blend_tree.h"

void AnimationNodeBlendSpace1D::get_parameter_list(LocalVector<PropertyInfo> *r_list) const {
	AnimationNode::get_parameter_list(r_list);
	r_list->push_back(PropertyInfo(Variant::FLOAT, blend_position));
	r_list->push_back(PropertyInfo(Variant::INT, closest, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE));
}

Variant AnimationNodeBlendSpace1D::get_parameter_default_value(const StringName &p_parameter) const {
	Variant ret = AnimationNode::get_parameter_default_value(p_parameter);
	if (ret != Variant()) {
		return ret;
	}

	if (p_parameter == closest) {
		return (int)-1;
	} else {
		return 0.0;
	}
}

Ref<AnimationNode> AnimationNodeBlendSpace1D::get_child_by_name(const StringName &p_name) const {
	int point_index = find_blend_point_by_name(p_name);
	if (point_index != -1) {
		return get_blend_point_node(point_index);
	}
	return Ref<AnimationRootNode>();
}

void AnimationNodeBlendSpace1D::_tree_changed() {
	AnimationRootNode::_tree_changed();
}

void AnimationNodeBlendSpace1D::_animation_node_renamed(const ObjectID &p_oid, const String &p_old_name, const String &p_new_name) {
	AnimationRootNode::_animation_node_renamed(p_oid, p_old_name, p_new_name);
}

void AnimationNodeBlendSpace1D::_animation_node_removed(const ObjectID &p_oid, const StringName &p_node) {
	AnimationRootNode::_animation_node_removed(p_oid, p_node);
}

void AnimationNodeBlendSpace1D::validate_node(const AnimationTree *p_tree, const StringName &p_path) const {
	AnimationRootNode::validate_node(p_tree, p_path);

	if (get_blend_point_count() == 0) {
		add_validation_error(p_tree, p_path, RTR("No blend points exist, so blending cannot take place."));
	}
}

void AnimationNodeBlendSpace1D::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_blend_point", "node", "pos", "at_index", "name"), &AnimationNodeBlendSpace1D::add_blend_point, DEFVAL(-1), DEFVAL(StringName()));
	ClassDB::bind_method(D_METHOD("set_blend_point_position", "point", "pos"), &AnimationNodeBlendSpace1D::set_blend_point_position);
	ClassDB::bind_method(D_METHOD("get_blend_point_position", "point"), &AnimationNodeBlendSpace1D::get_blend_point_position);
	ClassDB::bind_method(D_METHOD("set_blend_point_node", "point", "node"), &AnimationNodeBlendSpace1D::set_blend_point_node);
	ClassDB::bind_method(D_METHOD("get_blend_point_node", "point"), &AnimationNodeBlendSpace1D::get_blend_point_node);
	ClassDB::bind_method(D_METHOD("set_blend_point_name", "point", "name"), &AnimationNodeBlendSpace1D::set_blend_point_name);
	ClassDB::bind_method(D_METHOD("get_blend_point_name", "point"), &AnimationNodeBlendSpace1D::get_blend_point_name);
	ClassDB::bind_method(D_METHOD("find_blend_point_by_name", "name"), &AnimationNodeBlendSpace1D::find_blend_point_by_name);
	ClassDB::bind_method(D_METHOD("remove_blend_point", "point"), &AnimationNodeBlendSpace1D::remove_blend_point);
	ClassDB::bind_method(D_METHOD("get_blend_point_count"), &AnimationNodeBlendSpace1D::get_blend_point_count);
	ClassDB::bind_method(D_METHOD("reorder_blend_point", "from_index", "to_index"), &AnimationNodeBlendSpace1D::reorder_blend_point);

	ClassDB::bind_method(D_METHOD("set_min_space", "min_space"), &AnimationNodeBlendSpace1D::set_min_space);
	ClassDB::bind_method(D_METHOD("get_min_space"), &AnimationNodeBlendSpace1D::get_min_space);

	ClassDB::bind_method(D_METHOD("set_max_space", "max_space"), &AnimationNodeBlendSpace1D::set_max_space);
	ClassDB::bind_method(D_METHOD("get_max_space"), &AnimationNodeBlendSpace1D::get_max_space);

	ClassDB::bind_method(D_METHOD("set_snap", "snap"), &AnimationNodeBlendSpace1D::set_snap);
	ClassDB::bind_method(D_METHOD("get_snap"), &AnimationNodeBlendSpace1D::get_snap);

	ClassDB::bind_method(D_METHOD("set_value_label", "text"), &AnimationNodeBlendSpace1D::set_value_label);
	ClassDB::bind_method(D_METHOD("get_value_label"), &AnimationNodeBlendSpace1D::get_value_label);

	ClassDB::bind_method(D_METHOD("set_blend_mode", "mode"), &AnimationNodeBlendSpace1D::set_blend_mode);
	ClassDB::bind_method(D_METHOD("get_blend_mode"), &AnimationNodeBlendSpace1D::get_blend_mode);

	ClassDB::bind_method(D_METHOD("set_use_sync", "enable"), &AnimationNodeBlendSpace1D::set_use_sync);
	ClassDB::bind_method(D_METHOD("is_using_sync"), &AnimationNodeBlendSpace1D::is_using_sync);

	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "min_space", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_min_space", "get_min_space");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "max_space", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_max_space", "get_max_space");
	ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "snap", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_snap", "get_snap");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "value_label", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_value_label", "get_value_label");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "blend_mode", PROPERTY_HINT_ENUM, "Interpolated,Discrete,Carry", PROPERTY_USAGE_NO_EDITOR), "set_blend_mode", "get_blend_mode");
	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "sync", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_use_sync", "is_using_sync");

	BIND_ENUM_CONSTANT(BLEND_MODE_INTERPOLATED);
	BIND_ENUM_CONSTANT(BLEND_MODE_DISCRETE);
	BIND_ENUM_CONSTANT(BLEND_MODE_DISCRETE_CARRY);
}

void AnimationNodeBlendSpace1D::get_child_nodes(LocalVector<ChildNode> *r_child_nodes) {
	for (int i = 0; i < blend_points_used; i++) {
		ChildNode cn;
		cn.name = blend_points[i].name;
		cn.node = blend_points[i].node;
		r_child_nodes->push_back(cn);
	}
}

void AnimationNodeBlendSpace1D::add_blend_point(const Ref<AnimationRootNode> &p_node, float p_position, int p_at_index, const StringName &p_name) {
	ERR_FAIL_COND(blend_points_used >= MAX_BLEND_POINTS);
	ERR_FAIL_COND(p_node.is_null());
	ERR_FAIL_COND(p_at_index < -1 || p_at_index > blend_points_used);
#ifndef DISABLE_DEPRECATED
	if (p_name == StringName()) {
		_add_blend_point_bind_compat_110369(p_node, p_position, p_at_index);
		WARN_PRINT_ED("AnimationNodeBlendSpace1D::add_blend_point: No name provided, using safe index as reference. In the future, empty names will be deprecated, so explicitly passing a name is recommended.");
		return;
	}
#else
	ERR_FAIL_COND(p_name == StringName());
#endif
	if (p_at_index == -1 || p_at_index == blend_points_used) {
		p_at_index = blend_points_used;
	} else {
		for (int i = blend_points_used - 1; i > p_at_index; i--) {
			blend_points[i] = blend_points[i - 1];
		}
	}

	blend_points[p_at_index].node = p_node;
	blend_points[p_at_index].position = p_position;
	blend_points[p_at_index].name = p_name;

	_add_node(blend_points[p_at_index].node);

	blend_points_used++;
	_tree_changed();
}

void AnimationNodeBlendSpace1D::set_blend_point_position(int p_point, float p_position) {
	ERR_FAIL_INDEX(p_point, blend_points_used);

	blend_points[p_point].position = p_position;
}

void AnimationNodeBlendSpace1D::set_blend_point_node(int p_point, const Ref<AnimationRootNode> &p_node) {
	ERR_FAIL_INDEX(p_point, blend_points_used);
	ERR_FAIL_COND(p_node.is_null());

	if (blend_points[p_point].node.is_valid()) {
		_remove_node(blend_points[p_point].node);
	}

	blend_points[p_point].node = p_node;
	_add_node(blend_points[p_point].node);

	_tree_changed();
}

float AnimationNodeBlendSpace1D::get_blend_point_position(int p_point) const {
	ERR_FAIL_INDEX_V(p_point, MAX_BLEND_POINTS, 0);
	return blend_points[p_point].position;
}

Ref<AnimationRootNode> AnimationNodeBlendSpace1D::get_blend_point_node(int p_point) const {
	ERR_FAIL_INDEX_V(p_point, MAX_BLEND_POINTS, Ref<AnimationRootNode>());
	return blend_points[p_point].node;
}

void AnimationNodeBlendSpace1D::set_blend_point_name(int p_point, const StringName &p_name) {
	ERR_FAIL_INDEX(p_point, blend_points_used);
	String new_name = p_name;
	ERR_FAIL_COND(new_name.is_empty() || new_name.contains_char('.') || new_name.contains_char('/'));

	String old_name = blend_points[p_point].name;
	if (new_name != old_name) {
		blend_points[p_point].name = p_name;
		emit_signal(SNAME("animation_node_renamed"), get_instance_id(), old_name, new_name);
	}
}

const StringName &AnimationNodeBlendSpace1D::get_blend_point_name(int p_point) const {
	const static StringName empty = StringName();
	ERR_FAIL_INDEX_V(p_point, blend_points_used, empty);
	return blend_points[p_point].name;
}

int AnimationNodeBlendSpace1D::find_blend_point_by_name(const StringName &p_name) const {
	for (int i = 0; i < blend_points_used; i++) {
		if (blend_points[i].name == p_name) {
			return i;
		}
	}
	return -1;
}

void AnimationNodeBlendSpace1D::remove_blend_point(int p_point) {
	ERR_FAIL_INDEX(p_point, blend_points_used);

	ERR_FAIL_COND(blend_points[p_point].node.is_null());
	_remove_node(blend_points[p_point].node);

	for (int i = p_point; i < blend_points_used - 1; i++) {
		blend_points[i] = blend_points[i + 1];
	}

	blend_points_used--;

	blend_points[blend_points_used].name = StringName();

	emit_signal(SNAME("animation_node_removed"), get_instance_id(), itos(p_point));
	_tree_changed();
}

int AnimationNodeBlendSpace1D::get_blend_point_count() const {
	return blend_points_used;
}

void AnimationNodeBlendSpace1D::reorder_blend_point(int p_from_index, int p_to_index) {
	ERR_FAIL_INDEX(p_from_index, blend_points_used);
	ERR_FAIL_INDEX(p_to_index, blend_points_used);

	if (p_from_index == p_to_index) {
		return;
	}

	BlendPoint temp = blend_points[p_from_index];

	blend_points[p_from_index] = blend_points[p_to_index];
	blend_points[p_to_index] = temp;

	emit_signal(SNAME("tree_changed"));
}

void AnimationNodeBlendSpace1D::set_min_space(float p_min) {
	min_space = p_min;

	if (min_space >= max_space) {
		min_space = max_space - 1;
	}
}

float AnimationNodeBlendSpace1D::get_min_space() const {
	return min_space;
}

void AnimationNodeBlendSpace1D::set_max_space(float p_max) {
	max_space = p_max;

	if (max_space <= min_space) {
		max_space = min_space + 1;
	}
}

float AnimationNodeBlendSpace1D::get_max_space() const {
	return max_space;
}

void AnimationNodeBlendSpace1D::set_snap(float p_snap) {
	snap = p_snap;
}

float AnimationNodeBlendSpace1D::get_snap() const {
	return snap;
}

void AnimationNodeBlendSpace1D::set_value_label(const String &p_label) {
	value_label = p_label;
}

String AnimationNodeBlendSpace1D::get_value_label() const {
	return value_label;
}

void AnimationNodeBlendSpace1D::set_blend_mode(BlendMode p_blend_mode) {
	blend_mode = p_blend_mode;
}

AnimationNodeBlendSpace1D::BlendMode AnimationNodeBlendSpace1D::get_blend_mode() const {
	return blend_mode;
}

void AnimationNodeBlendSpace1D::set_use_sync(bool p_sync) {
	sync = p_sync;
}

bool AnimationNodeBlendSpace1D::is_using_sync() const {
	return sync;
}

bool AnimationNodeBlendSpace1D::_set(const StringName &p_name, const Variant &p_value) {
	String prop = p_name;
	if (prop.begins_with("blend_point_")) {
		int idx = prop.get_slicec('_', 2).to_int();
		String what = prop.get_slicec('/', 1);
		if (what == "node") {
			Ref<AnimationRootNode> node = p_value;
#ifndef DISABLE_DEPRECATED
			if (idx == blend_points_used) {
				add_blend_point(node, 0, -1, blend_points[idx].name.is_empty() ? StringName(itos(idx)) : blend_points[idx].name);
			} else {
				set_blend_point_node(idx, node);
			}
#else
			if (idx == blend_points_used) {
				add_blend_point(node, 0, -1, blend_points[idx].name);
			} else {
				set_blend_point_node(idx, node);
			}
#endif // DISABLE_DEPRECATED
		} else if (what == "pos") {
			set_blend_point_position(idx, p_value);
		} else if (what == "name") {
			set_blend_point_name(idx, p_value);
		} else {
			return false;
		}
		return true;
	}
	return false;
}

bool AnimationNodeBlendSpace1D::_get(const StringName &p_name, Variant &r_ret) const {
	String prop = p_name;
	if (prop.begins_with("blend_point_")) {
		int idx = prop.get_slicec('_', 2).to_int();
		String what = prop.get_slicec('/', 1);
		if (what == "node") {
			r_ret = get_blend_point_node(idx);
		} else if (what == "pos") {
			r_ret = get_blend_point_position(idx);
		} else if (what == "name") {
			r_ret = get_blend_point_name(idx);
		} else {
			return false;
		}
		return true;
	}
	return false;
}

void AnimationNodeBlendSpace1D::_get_property_list(List<PropertyInfo> *p_list) const {
	for (int i = 0; i < blend_points_used; i++) {
		p_list->push_back(PropertyInfo(Variant::OBJECT, "blend_point_" + itos(i) + "/node", PROPERTY_HINT_RESOURCE_TYPE, AnimationRootNode::get_class_static(), PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::FLOAT, "blend_point_" + itos(i) + "/pos", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
		p_list->push_back(PropertyInfo(Variant::STRING, "blend_point_" + itos(i) + "/name", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL));
	}
}

AnimationNode::NodeTimeInfo AnimationNodeBlendSpace1D::_process(ProcessState &p_process_state, AnimationNodeInstance &p_instance, const AnimationMixer::PlaybackInfo &p_playback_info, bool p_test_only) {
	if (!blend_points_used) {
		return NodeTimeInfo();
	}

	AnimationMixer::PlaybackInfo pi = p_playback_info;

	if (blend_points_used == 1) {
		// only one point available, just play that animation
		pi.weight = 1.0;
		AnimationNodeInstance &other_instance = p_instance.get_child_instance_by_path(get_blend_point_name(0));
		return blend_node(p_process_state, p_instance, &other_instance, pi, FILTER_IGNORE, true, p_test_only);
	}

	double blend_pos = p_instance.get_parameter(blend_position);
	int cur_closest = p_instance.get_parameter_closest();
	NodeTimeInfo mind;

	if (blend_mode == BLEND_MODE_INTERPOLATED) {
		int point_lower = -1;
		float pos_lower = 0.0;
		int point_higher = -1;
		float pos_higher = 0.0;

		// find the closest two points to blend between
		for (int i = 0; i < blend_points_used; i++) {
			float pos = blend_points[i].position;

			if (pos <= blend_pos) {
				if (point_lower == -1 || pos > pos_lower) {
					point_lower = i;
					pos_lower = pos;
				}
			} else if (point_higher == -1 || pos < pos_higher) {
				point_higher = i;
				pos_higher = pos;
			}
		}

		// fill in weights
		float weights[MAX_BLEND_POINTS] = {};
		if (point_lower == -1 && point_higher != -1) {
			// we are on the left side, no other point to the left
			// we just play the next point.

			weights[point_higher] = 1.0;
		} else if (point_higher == -1) {
			// we are on the right side, no other point to the right
			// we just play the previous point

			weights[point_lower] = 1.0;
		} else {
			// we are between two points.
			// figure out weights, then blend the animations

			float distance_between_points = pos_higher - pos_lower;

			float current_pos_inbetween = blend_pos - pos_lower;

			float blend_percentage = current_pos_inbetween / distance_between_points;

			float blend_lower = 1.0 - blend_percentage;
			float blend_higher = blend_percentage;

			weights[point_lower] = blend_lower;
			weights[point_higher] = blend_higher;
		}

		// actually blend the animations now
		bool first = true;
		double max_weight = 0.0;
		for (int i = 0; i < blend_points_used; i++) {
			if (i == point_lower || i == point_higher) {
				pi.weight = weights[i];
				AnimationNodeInstance &other_instance = p_instance.get_child_instance_by_path(get_blend_point_name(i));
				NodeTimeInfo t = blend_node(p_process_state, p_instance, &other_instance, pi, FILTER_IGNORE, true, p_test_only);
				if (first || pi.weight > max_weight) {
					max_weight = pi.weight;
					mind = t;
					first = false;
				}
			} else if (sync) {
				pi.weight = 0;
				AnimationNodeInstance &other_instance = p_instance.get_child_instance_by_path(get_blend_point_name(i));
				blend_node(p_process_state, p_instance, &other_instance, pi, FILTER_IGNORE, true, p_test_only);
			}
		}
	} else {
		int new_closest = -1;
		double new_closest_dist = 1e20;

		for (int i = 0; i < blend_points_used; i++) {
			double d = std::abs(blend_points[i].position - blend_pos);
			if (d < new_closest_dist) {
				new_closest = i;
				new_closest_dist = d;
			}
		}

		AnimationNodeInstance *instance_current_closest = p_instance.get_child_instance_by_path_or_null(get_blend_point_name(cur_closest));
		if (new_closest != cur_closest && new_closest != -1) {
			AnimationNodeInstance *instance_new_closest = p_instance.get_child_instance_by_path_or_null(get_blend_point_name(new_closest));

			if (blend_mode == BLEND_MODE_DISCRETE_CARRY && cur_closest != -1) {
				NodeTimeInfo from;
				// For ping-pong loop.
				Ref<AnimationNodeAnimation> na_c = static_cast<Ref<AnimationNodeAnimation>>(blend_points[cur_closest].node);
				Ref<AnimationNodeAnimation> na_n = static_cast<Ref<AnimationNodeAnimation>>(blend_points[new_closest].node);
				if (na_c.is_valid() && na_n.is_valid() && instance_current_closest && instance_new_closest) {
					na_n->set_backward(*instance_new_closest, p_process_state, na_c->is_backward(*instance_current_closest, p_process_state));
					na_n = nullptr;
					na_c = nullptr;
				}
				// See how much animation remains.
				pi.seeked = false;
				pi.weight = 0;
				from = blend_node(p_process_state, p_instance, instance_current_closest, pi, FILTER_IGNORE, true, true);
				pi.time = from.position;
			}
			pi.seeked = true;
			pi.weight = 1.0;
			mind = blend_node(p_process_state, p_instance, instance_new_closest, pi, FILTER_IGNORE, true, p_test_only);
			cur_closest = new_closest;
		} else {
			pi.weight = 1.0;
			mind = blend_node(p_process_state, p_instance, instance_current_closest, pi, FILTER_IGNORE, true, p_test_only);
		}

		if (sync) {
			pi = p_playback_info;
			pi.weight = 0;
			for (int i = 0; i < blend_points_used; i++) {
				if (i != cur_closest) {
					AnimationNodeInstance &other_instance = p_instance.get_child_instance_by_path(get_blend_point_name(i));
					blend_node(p_process_state, p_instance, &other_instance, pi, FILTER_IGNORE, true, p_test_only);
				}
			}
		}
	}

	p_instance.set_parameter_closest(cur_closest, p_process_state.is_testing);
	return mind;
}

String AnimationNodeBlendSpace1D::get_caption() const {
	return "BlendSpace1D";
}

AnimationNodeBlendSpace1D::AnimationNodeBlendSpace1D() {
}

AnimationNodeBlendSpace1D::~AnimationNodeBlendSpace1D() {
}
