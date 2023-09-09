/**************************************************************************/
/*  animation_tree.cpp                                                    */
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

#include "animation_tree.h"

#include "animation_blend_tree.h"
#include "core/config/engine.h"
#include "scene/resources/animation.h"
#include "scene/scene_string_names.h"
#include "servers/audio/audio_stream.h"

void AnimationNode::get_parameter_list(List<PropertyInfo> *r_list) const {
	Array parameters;

	if (GDVIRTUAL_CALL(_get_parameter_list, parameters)) {
		for (int i = 0; i < parameters.size(); i++) {
			Dictionary d = parameters[i];
			ERR_CONTINUE(d.is_empty());
			r_list->push_back(PropertyInfo::from_dict(d));
		}
	}
}

Variant AnimationNode::get_parameter_default_value(const StringName &p_parameter) const {
	Variant ret;
	GDVIRTUAL_CALL(_get_parameter_default_value, p_parameter, ret);
	return ret;
}

bool AnimationNode::is_parameter_read_only(const StringName &p_parameter) const {
	bool ret = false;
	GDVIRTUAL_CALL(_is_parameter_read_only, p_parameter, ret);
	return ret;
}

void AnimationNode::set_parameter(const StringName &p_name, const Variant &p_value) {
	if (is_testing) {
		return;
	}
	ERR_FAIL_NULL(state);
	ERR_FAIL_COND(!state->tree->property_parent_map.has(base_path));
	ERR_FAIL_COND(!state->tree->property_parent_map[base_path].has(p_name));
	StringName path = state->tree->property_parent_map[base_path][p_name];

	state->tree->property_map[path].first = p_value;
}

Variant AnimationNode::get_parameter(const StringName &p_name) const {
	ERR_FAIL_NULL_V(state, Variant());
	ERR_FAIL_COND_V(!state->tree->property_parent_map.has(base_path), Variant());
	ERR_FAIL_COND_V(!state->tree->property_parent_map[base_path].has(p_name), Variant());

	StringName path = state->tree->property_parent_map[base_path][p_name];
	return state->tree->property_map[path].first;
}

void AnimationNode::get_child_nodes(List<ChildNode> *r_child_nodes) {
	Dictionary cn;
	if (GDVIRTUAL_CALL(_get_child_nodes, cn)) {
		List<Variant> keys;
		cn.get_key_list(&keys);
		for (const Variant &E : keys) {
			ChildNode child;
			child.name = E;
			child.node = cn[E];
			r_child_nodes->push_back(child);
		}
	}
}

void AnimationNode::blend_animation(const StringName &p_animation, double p_time, double p_delta, bool p_seeked, bool p_is_external_seeking, real_t p_blend, Animation::LoopedFlag p_looped_flag) {
	ERR_FAIL_NULL(state);
	ERR_FAIL_COND(!state->player->has_animation(p_animation));

	Ref<Animation> animation = state->player->get_animation(p_animation);

	if (animation.is_null()) {
		AnimationNodeBlendTree *btree = Object::cast_to<AnimationNodeBlendTree>(parent);
		if (btree) {
			String node_name = btree->get_node_name(Ref<AnimationNodeAnimation>(this));
			make_invalid(vformat(RTR("In node '%s', invalid animation: '%s'."), node_name, p_animation));
		} else {
			make_invalid(vformat(RTR("Invalid animation: '%s'."), p_animation));
		}
		return;
	}

	ERR_FAIL_COND(!animation.is_valid());

	AnimationState anim_state;
	anim_state.blend = p_blend;
	anim_state.track_blends = blends;
	anim_state.delta = p_delta;
	anim_state.time = p_time;
	anim_state.animation = animation;
	anim_state.seeked = p_seeked;
	anim_state.looped_flag = p_looped_flag;
	anim_state.is_external_seeking = p_is_external_seeking;

	state->animation_states.push_back(anim_state);
}

double AnimationNode::_pre_process(const StringName &p_base_path, AnimationNode *p_parent, State *p_state, double p_time, bool p_seek, bool p_is_external_seeking, const Vector<StringName> &p_connections, bool p_test_only) {
	base_path = p_base_path;
	parent = p_parent;
	connections = p_connections;
	state = p_state;

	double t = process(p_time, p_seek, p_is_external_seeking, p_test_only);

	state = nullptr;
	parent = nullptr;
	base_path = StringName();
	connections.clear();

	return t;
}

AnimationTree *AnimationNode::get_animation_tree() const {
	ERR_FAIL_NULL_V(state, nullptr);
	return state->tree;
}

void AnimationNode::make_invalid(const String &p_reason) {
	ERR_FAIL_NULL(state);
	state->valid = false;
	if (!state->invalid_reasons.is_empty()) {
		state->invalid_reasons += "\n";
	}
	state->invalid_reasons += String::utf8("•  ") + p_reason;
}

double AnimationNode::blend_input(int p_input, double p_time, bool p_seek, bool p_is_external_seeking, real_t p_blend, FilterAction p_filter, bool p_sync, bool p_test_only) {
	ERR_FAIL_INDEX_V(p_input, inputs.size(), 0);
	ERR_FAIL_NULL_V(state, 0);

	AnimationNodeBlendTree *blend_tree = Object::cast_to<AnimationNodeBlendTree>(parent);
	ERR_FAIL_NULL_V(blend_tree, 0);

	StringName node_name = connections[p_input];

	if (!blend_tree->has_node(node_name)) {
		String node_name2 = blend_tree->get_node_name(Ref<AnimationNode>(this));
		make_invalid(vformat(RTR("Nothing connected to input '%s' of node '%s'."), get_input_name(p_input), node_name2));
		return 0;
	}

	Ref<AnimationNode> node = blend_tree->get_node(node_name);

	//inputs.write[p_input].last_pass = state->last_pass;
	real_t activity = 0.0;
	double ret = _blend_node(node_name, blend_tree->get_node_connection_array(node_name), nullptr, node, p_time, p_seek, p_is_external_seeking, p_blend, p_filter, p_sync, &activity, p_test_only);

	Vector<AnimationTree::Activity> *activity_ptr = state->tree->input_activity_map.getptr(base_path);

	if (activity_ptr && p_input < activity_ptr->size()) {
		activity_ptr->write[p_input].last_pass = state->last_pass;
		activity_ptr->write[p_input].activity = activity;
	}
	return ret;
}

double AnimationNode::blend_node(const StringName &p_sub_path, Ref<AnimationNode> p_node, double p_time, bool p_seek, bool p_is_external_seeking, real_t p_blend, FilterAction p_filter, bool p_sync, bool p_test_only) {
	return _blend_node(p_sub_path, Vector<StringName>(), this, p_node, p_time, p_seek, p_is_external_seeking, p_blend, p_filter, p_sync, nullptr, p_test_only);
}

double AnimationNode::_blend_node(const StringName &p_subpath, const Vector<StringName> &p_connections, AnimationNode *p_new_parent, Ref<AnimationNode> p_node, double p_time, bool p_seek, bool p_is_external_seeking, real_t p_blend, FilterAction p_filter, bool p_sync, real_t *r_max, bool p_test_only) {
	ERR_FAIL_COND_V(!p_node.is_valid(), 0);
	ERR_FAIL_NULL_V(state, 0);

	int blend_count = blends.size();

	if (p_node->blends.size() != blend_count) {
		p_node->blends.resize(blend_count);
	}

	real_t *blendw = p_node->blends.ptrw();
	const real_t *blendr = blends.ptr();

	bool any_valid = false;

	if (has_filter() && is_filter_enabled() && p_filter != FILTER_IGNORE) {
		for (int i = 0; i < blend_count; i++) {
			blendw[i] = 0.0; //all to zero by default
		}

		for (const KeyValue<NodePath, bool> &E : filter) {
			if (!state->track_map.has(E.key)) {
				continue;
			}
			int idx = state->track_map[E.key];
			blendw[idx] = 1.0; //filtered goes to one
		}

		switch (p_filter) {
			case FILTER_IGNORE:
				break; //will not happen anyway
			case FILTER_PASS: {
				//values filtered pass, the rest don't
				for (int i = 0; i < blend_count; i++) {
					if (blendw[i] == 0) { //not filtered, does not pass
						continue;
					}

					blendw[i] = blendr[i] * p_blend;
					if (!Math::is_zero_approx(blendw[i])) {
						any_valid = true;
					}
				}

			} break;
			case FILTER_STOP: {
				//values filtered don't pass, the rest are blended

				for (int i = 0; i < blend_count; i++) {
					if (blendw[i] > 0) { //filtered, does not pass
						continue;
					}

					blendw[i] = blendr[i] * p_blend;
					if (!Math::is_zero_approx(blendw[i])) {
						any_valid = true;
					}
				}

			} break;
			case FILTER_BLEND: {
				//filtered values are blended, the rest are passed without blending

				for (int i = 0; i < blend_count; i++) {
					if (blendw[i] == 1.0) {
						blendw[i] = blendr[i] * p_blend; //filtered, blend
					} else {
						blendw[i] = blendr[i]; //not filtered, do not blend
					}

					if (!Math::is_zero_approx(blendw[i])) {
						any_valid = true;
					}
				}

			} break;
		}
	} else {
		for (int i = 0; i < blend_count; i++) {
			//regular blend
			blendw[i] = blendr[i] * p_blend;
			if (!Math::is_zero_approx(blendw[i])) {
				any_valid = true;
			}
		}
	}

	if (r_max) {
		*r_max = 0;
		for (int i = 0; i < blend_count; i++) {
			*r_max = MAX(*r_max, Math::abs(blendw[i]));
		}
	}

	String new_path;
	AnimationNode *new_parent;

	// This is the slowest part of processing, but as strings process in powers of 2, and the paths always exist, it will not result in that many allocations.
	if (p_new_parent) {
		new_parent = p_new_parent;
		new_path = String(base_path) + String(p_subpath) + "/";
	} else {
		ERR_FAIL_NULL_V(parent, 0);
		new_parent = parent;
		new_path = String(parent->base_path) + String(p_subpath) + "/";
	}

	// This process, which depends on p_sync is needed to process sync correctly in the case of
	// that a synced AnimationNodeSync exists under the un-synced AnimationNodeSync.
	if (!p_seek && !p_sync && !any_valid) {
		return p_node->_pre_process(new_path, new_parent, state, 0, p_seek, p_is_external_seeking, p_connections, p_test_only);
	}
	return p_node->_pre_process(new_path, new_parent, state, p_time, p_seek, p_is_external_seeking, p_connections, p_test_only);
}

String AnimationNode::get_caption() const {
	String ret = "Node";
	GDVIRTUAL_CALL(_get_caption, ret);
	return ret;
}

bool AnimationNode::add_input(const String &p_name) {
	//root nodes can't add inputs
	ERR_FAIL_COND_V(Object::cast_to<AnimationRootNode>(this) != nullptr, false);
	Input input;
	ERR_FAIL_COND_V(p_name.contains(".") || p_name.contains("/"), false);
	input.name = p_name;
	inputs.push_back(input);
	emit_changed();
	return true;
}

void AnimationNode::remove_input(int p_index) {
	ERR_FAIL_INDEX(p_index, inputs.size());
	inputs.remove_at(p_index);
	emit_changed();
}

bool AnimationNode::set_input_name(int p_input, const String &p_name) {
	ERR_FAIL_INDEX_V(p_input, inputs.size(), false);
	ERR_FAIL_COND_V(p_name.contains(".") || p_name.contains("/"), false);
	inputs.write[p_input].name = p_name;
	emit_changed();
	return true;
}

String AnimationNode::get_input_name(int p_input) const {
	ERR_FAIL_INDEX_V(p_input, inputs.size(), String());
	return inputs[p_input].name;
}

int AnimationNode::get_input_count() const {
	return inputs.size();
}

int AnimationNode::find_input(const String &p_name) const {
	int idx = -1;
	for (int i = 0; i < inputs.size(); i++) {
		if (inputs[i].name == p_name) {
			idx = i;
			break;
		}
	}
	return idx;
}

double AnimationNode::process(double p_time, bool p_seek, bool p_is_external_seeking, bool p_test_only) {
	is_testing = p_test_only;
	return _process(p_time, p_seek, p_is_external_seeking, p_test_only);
}

double AnimationNode::_process(double p_time, bool p_seek, bool p_is_external_seeking, bool p_test_only) {
	double ret = 0;
	GDVIRTUAL_CALL(_process, p_time, p_seek, p_is_external_seeking, p_test_only, ret);
	return ret;
}

void AnimationNode::set_filter_path(const NodePath &p_path, bool p_enable) {
	if (p_enable) {
		filter[p_path] = true;
	} else {
		filter.erase(p_path);
	}
}

void AnimationNode::set_filter_enabled(bool p_enable) {
	filter_enabled = p_enable;
}

bool AnimationNode::is_filter_enabled() const {
	return filter_enabled;
}

void AnimationNode::set_closable(bool p_closable) {
	closable = p_closable;
}

bool AnimationNode::is_closable() const {
	return closable;
}

bool AnimationNode::is_path_filtered(const NodePath &p_path) const {
	return filter.has(p_path);
}

bool AnimationNode::has_filter() const {
	bool ret = false;
	GDVIRTUAL_CALL(_has_filter, ret);
	return ret;
}

Array AnimationNode::_get_filters() const {
	Array paths;

	for (const KeyValue<NodePath, bool> &E : filter) {
		paths.push_back(String(E.key)); //use strings, so sorting is possible
	}
	paths.sort(); //done so every time the scene is saved, it does not change

	return paths;
}

void AnimationNode::_set_filters(const Array &p_filters) {
	filter.clear();
	for (int i = 0; i < p_filters.size(); i++) {
		set_filter_path(p_filters[i], true);
	}
}

void AnimationNode::_validate_property(PropertyInfo &p_property) const {
	if (!has_filter() && (p_property.name == "filter_enabled" || p_property.name == "filters")) {
		p_property.usage = PROPERTY_USAGE_NONE;
	}
}

Ref<AnimationNode> AnimationNode::get_child_by_name(const StringName &p_name) const {
	Ref<AnimationNode> ret;
	GDVIRTUAL_CALL(_get_child_by_name, p_name, ret);
	return ret;
}

Ref<AnimationNode> AnimationNode::find_node_by_path(const String &p_name) const {
	Vector<String> split = p_name.split("/");
	Ref<AnimationNode> ret = const_cast<AnimationNode *>(this);
	for (int i = 0; i < split.size(); i++) {
		ret = ret->get_child_by_name(split[i]);
		if (!ret.is_valid()) {
			break;
		}
	}
	return ret;
}

void AnimationNode::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_input", "name"), &AnimationNode::add_input);
	ClassDB::bind_method(D_METHOD("remove_input", "index"), &AnimationNode::remove_input);
	ClassDB::bind_method(D_METHOD("set_input_name", "input", "name"), &AnimationNode::set_input_name);
	ClassDB::bind_method(D_METHOD("get_input_name", "input"), &AnimationNode::get_input_name);
	ClassDB::bind_method(D_METHOD("get_input_count"), &AnimationNode::get_input_count);
	ClassDB::bind_method(D_METHOD("find_input", "name"), &AnimationNode::find_input);

	ClassDB::bind_method(D_METHOD("set_filter_path", "path", "enable"), &AnimationNode::set_filter_path);
	ClassDB::bind_method(D_METHOD("is_path_filtered", "path"), &AnimationNode::is_path_filtered);

	ClassDB::bind_method(D_METHOD("set_filter_enabled", "enable"), &AnimationNode::set_filter_enabled);
	ClassDB::bind_method(D_METHOD("is_filter_enabled"), &AnimationNode::is_filter_enabled);

	ClassDB::bind_method(D_METHOD("_set_filters", "filters"), &AnimationNode::_set_filters);
	ClassDB::bind_method(D_METHOD("_get_filters"), &AnimationNode::_get_filters);

	ClassDB::bind_method(D_METHOD("blend_animation", "animation", "time", "delta", "seeked", "is_external_seeking", "blend", "looped_flag"), &AnimationNode::blend_animation, DEFVAL(Animation::LOOPED_FLAG_NONE));
	ClassDB::bind_method(D_METHOD("blend_node", "name", "node", "time", "seek", "is_external_seeking", "blend", "filter", "sync", "test_only"), &AnimationNode::blend_node, DEFVAL(FILTER_IGNORE), DEFVAL(true), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("blend_input", "input_index", "time", "seek", "is_external_seeking", "blend", "filter", "sync", "test_only"), &AnimationNode::blend_input, DEFVAL(FILTER_IGNORE), DEFVAL(true), DEFVAL(false));

	ClassDB::bind_method(D_METHOD("set_parameter", "name", "value"), &AnimationNode::set_parameter);
	ClassDB::bind_method(D_METHOD("get_parameter", "name"), &AnimationNode::get_parameter);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "filter_enabled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_filter_enabled", "is_filter_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "filters", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "_set_filters", "_get_filters");

	GDVIRTUAL_BIND(_get_child_nodes);
	GDVIRTUAL_BIND(_get_parameter_list);
	GDVIRTUAL_BIND(_get_child_by_name, "name");
	GDVIRTUAL_BIND(_get_parameter_default_value, "parameter");
	GDVIRTUAL_BIND(_is_parameter_read_only, "parameter");
	GDVIRTUAL_BIND(_process, "time", "seek", "is_external_seeking", "test_only");
	GDVIRTUAL_BIND(_get_caption);
	GDVIRTUAL_BIND(_has_filter);

	ADD_SIGNAL(MethodInfo("tree_changed"));
	ADD_SIGNAL(MethodInfo("animation_node_renamed", PropertyInfo(Variant::INT, "object_id"), PropertyInfo(Variant::STRING, "old_name"), PropertyInfo(Variant::STRING, "new_name")));
	ADD_SIGNAL(MethodInfo("animation_node_removed", PropertyInfo(Variant::INT, "object_id"), PropertyInfo(Variant::STRING, "name")));

	BIND_ENUM_CONSTANT(FILTER_IGNORE);
	BIND_ENUM_CONSTANT(FILTER_PASS);
	BIND_ENUM_CONSTANT(FILTER_STOP);
	BIND_ENUM_CONSTANT(FILTER_BLEND);
}

AnimationNode::AnimationNode() {
}

////////////////////

void AnimationRootNode::_tree_changed() {
	emit_signal(SNAME("tree_changed"));
}

void AnimationRootNode::_animation_node_renamed(const ObjectID &p_oid, const String &p_old_name, const String &p_new_name) {
	emit_signal(SNAME("animation_node_renamed"), p_oid, p_old_name, p_new_name);
}

void AnimationRootNode::_animation_node_removed(const ObjectID &p_oid, const StringName &p_node) {
	emit_signal(SNAME("animation_node_removed"), p_oid, p_node);
}

////////////////////

void AnimationTree::set_tree_root(const Ref<AnimationNode> &p_root) {
	if (root.is_valid()) {
		root->disconnect("tree_changed", callable_mp(this, &AnimationTree::_tree_changed));
		root->disconnect("animation_node_renamed", callable_mp(this, &AnimationTree::_animation_node_renamed));
		root->disconnect("animation_node_removed", callable_mp(this, &AnimationTree::_animation_node_removed));
	}

	root = p_root;

	if (root.is_valid()) {
		root->connect("tree_changed", callable_mp(this, &AnimationTree::_tree_changed));
		root->connect("animation_node_renamed", callable_mp(this, &AnimationTree::_animation_node_renamed));
		root->connect("animation_node_removed", callable_mp(this, &AnimationTree::_animation_node_removed));
	}

	properties_dirty = true;

	update_configuration_warnings();
}

Ref<AnimationNode> AnimationTree::get_tree_root() const {
	return root;
}

void AnimationTree::set_active(bool p_active) {
	if (active == p_active) {
		return;
	}

	active = p_active;
	started = active;

	if (process_callback == ANIMATION_PROCESS_IDLE) {
		set_process_internal(active);
	} else {
		set_physics_process_internal(active);
	}

	if (!active && is_inside_tree()) {
		_clear_caches();
	}
}

bool AnimationTree::is_active() const {
	return active;
}

void AnimationTree::set_process_callback(AnimationProcessCallback p_mode) {
	if (process_callback == p_mode) {
		return;
	}

	bool was_active = is_active();
	if (was_active) {
		set_active(false);
	}

	process_callback = p_mode;

	if (was_active) {
		set_active(true);
	}
}

AnimationTree::AnimationProcessCallback AnimationTree::get_process_callback() const {
	return process_callback;
}

void AnimationTree::_node_removed(Node *p_node) {
	cache_valid = false;
}

bool AnimationTree::_update_caches(AnimationPlayer *player) {
	setup_pass++;

	if (!player->has_node(player->get_root())) {
		ERR_PRINT("AnimationTree: AnimationPlayer root is invalid.");
		set_active(false);
		_clear_caches();
		return false;
	}
	Node *parent = player->get_node(player->get_root());

	List<StringName> sname;
	player->get_animation_list(&sname);

	root_motion_cache.loc = Vector3(0, 0, 0);
	root_motion_cache.rot = Quaternion(0, 0, 0, 1);
	root_motion_cache.scale = Vector3(1, 1, 1);

	Ref<Animation> reset_anim;
	bool has_reset_anim = player->has_animation(SceneStringNames::get_singleton()->RESET);
	if (has_reset_anim) {
		reset_anim = player->get_animation(SceneStringNames::get_singleton()->RESET);
	}
	for (const StringName &E : sname) {
		Ref<Animation> anim = player->get_animation(E);
		for (int i = 0; i < anim->get_track_count(); i++) {
			NodePath path = anim->track_get_path(i);
			Animation::TrackType track_type = anim->track_get_type(i);

			Animation::TrackType track_cache_type = track_type;
			if (track_cache_type == Animation::TYPE_POSITION_3D || track_cache_type == Animation::TYPE_ROTATION_3D || track_cache_type == Animation::TYPE_SCALE_3D) {
				track_cache_type = Animation::TYPE_POSITION_3D; //reference them as position3D tracks, even if they modify rotation or scale
			}

			TrackCache *track = nullptr;
			if (track_cache.has(path)) {
				track = track_cache.get(path);
			}

			//if not valid, delete track
			if (track && (track->type != track_cache_type || ObjectDB::get_instance(track->object_id) == nullptr)) {
				playing_caches.erase(track);
				memdelete(track);
				track_cache.erase(path);
				track = nullptr;
			}

			if (!track) {
				Ref<Resource> resource;
				Vector<StringName> leftover_path;
				Node *child = parent->get_node_and_resource(path, resource, leftover_path);

				if (!child) {
					ERR_PRINT("AnimationTree: '" + String(E) + "', couldn't resolve track:  '" + String(path) + "'");
					continue;
				}

				if (!child->is_connected("tree_exited", callable_mp(this, &AnimationTree::_node_removed))) {
					child->connect("tree_exited", callable_mp(this, &AnimationTree::_node_removed).bind(child));
				}

				switch (track_type) {
					case Animation::TYPE_VALUE: {
						TrackCacheValue *track_value = memnew(TrackCacheValue);

						if (resource.is_valid()) {
							track_value->object = resource.ptr();
						} else {
							track_value->object = child;
						}

						track_value->is_discrete = anim->value_track_get_update_mode(i) == Animation::UPDATE_DISCRETE;
						track_value->is_using_angle = anim->track_get_interpolation_type(i) == Animation::INTERPOLATION_LINEAR_ANGLE || anim->track_get_interpolation_type(i) == Animation::INTERPOLATION_CUBIC_ANGLE;

						track_value->subpath = leftover_path;
						track_value->object_id = track_value->object->get_instance_id();

						track = track_value;

						// If a value track without a key is cached first, the initial value cannot be determined.
						// It is a corner case, but which may cause problems with blending.
						ERR_CONTINUE_MSG(anim->track_get_key_count(i) == 0, "AnimationTree: '" + String(E) + "', Value Track:  '" + String(path) + "' must have at least one key to cache for blending.");
						track_value->init_value = anim->track_get_key_value(i, 0);
						track_value->init_value.zero();

						// If there is a Reset Animation, it takes precedence by overwriting.
						if (has_reset_anim) {
							int rt = reset_anim->find_track(path, track_type);
							if (rt >= 0 && reset_anim->track_get_key_count(rt) > 0) {
								track_value->init_value = reset_anim->track_get_key_value(rt, 0);
							}
						}
					} break;
					case Animation::TYPE_POSITION_3D:
					case Animation::TYPE_ROTATION_3D:
					case Animation::TYPE_SCALE_3D: {
#ifndef _3D_DISABLED
						Node3D *node_3d = Object::cast_to<Node3D>(child);

						if (!node_3d) {
							ERR_PRINT("AnimationTree: '" + String(E) + "', transform track does not point to Node3D:  '" + String(path) + "'");
							continue;
						}

						TrackCacheTransform *track_xform = memnew(TrackCacheTransform);
						track_xform->type = Animation::TYPE_POSITION_3D;

						track_xform->node_3d = node_3d;
						track_xform->skeleton = nullptr;
						track_xform->bone_idx = -1;

						bool has_rest = false;
						if (path.get_subname_count() == 1 && Object::cast_to<Skeleton3D>(node_3d)) {
							Skeleton3D *sk = Object::cast_to<Skeleton3D>(node_3d);
							track_xform->skeleton = sk;
							int bone_idx = sk->find_bone(path.get_subname(0));
							if (bone_idx != -1) {
								has_rest = true;
								track_xform->bone_idx = bone_idx;
								Transform3D rest = sk->get_bone_rest(bone_idx);
								track_xform->init_loc = rest.origin;
								track_xform->init_rot = rest.basis.get_rotation_quaternion();
								track_xform->init_scale = rest.basis.get_scale();
							}
						}

						track_xform->object = node_3d;
						track_xform->object_id = track_xform->object->get_instance_id();

						track = track_xform;

						switch (track_type) {
							case Animation::TYPE_POSITION_3D: {
								track_xform->loc_used = true;
							} break;
							case Animation::TYPE_ROTATION_3D: {
								track_xform->rot_used = true;
							} break;
							case Animation::TYPE_SCALE_3D: {
								track_xform->scale_used = true;
							} break;
							default: {
							}
						}

						// For non Skeleton3D bone animation.
						if (has_reset_anim && !has_rest) {
							int rt = reset_anim->find_track(path, track_type);
							if (rt >= 0 && reset_anim->track_get_key_count(rt) > 0) {
								switch (track_type) {
									case Animation::TYPE_POSITION_3D: {
										track_xform->init_loc = reset_anim->track_get_key_value(rt, 0);
									} break;
									case Animation::TYPE_ROTATION_3D: {
										track_xform->init_rot = reset_anim->track_get_key_value(rt, 0);
									} break;
									case Animation::TYPE_SCALE_3D: {
										track_xform->init_scale = reset_anim->track_get_key_value(rt, 0);
									} break;
									default: {
									}
								}
							}
						}
#endif // _3D_DISABLED
					} break;
					case Animation::TYPE_BLEND_SHAPE: {
#ifndef _3D_DISABLED
						if (path.get_subname_count() != 1) {
							ERR_PRINT("AnimationTree: '" + String(E) + "', blend shape track does not contain a blend shape subname:  '" + String(path) + "'");
							continue;
						}
						MeshInstance3D *mesh_3d = Object::cast_to<MeshInstance3D>(child);

						if (!mesh_3d) {
							ERR_PRINT("AnimationTree: '" + String(E) + "', blend shape track does not point to MeshInstance3D:  '" + String(path) + "'");
							continue;
						}

						StringName blend_shape_name = path.get_subname(0);
						int blend_shape_idx = mesh_3d->find_blend_shape_by_name(blend_shape_name);
						if (blend_shape_idx == -1) {
							ERR_PRINT("AnimationTree: '" + String(E) + "', blend shape track points to a non-existing name:  '" + String(blend_shape_name) + "'");
							continue;
						}

						TrackCacheBlendShape *track_bshape = memnew(TrackCacheBlendShape);

						track_bshape->mesh_3d = mesh_3d;
						track_bshape->shape_index = blend_shape_idx;

						track_bshape->object = mesh_3d;
						track_bshape->object_id = mesh_3d->get_instance_id();
						track = track_bshape;

						if (has_reset_anim) {
							int rt = reset_anim->find_track(path, track_type);
							if (rt >= 0 && reset_anim->track_get_key_count(rt) > 0) {
								track_bshape->init_value = reset_anim->track_get_key_value(rt, 0);
							}
						}
#endif
					} break;
					case Animation::TYPE_METHOD: {
						TrackCacheMethod *track_method = memnew(TrackCacheMethod);

						if (resource.is_valid()) {
							track_method->object = resource.ptr();
						} else {
							track_method->object = child;
						}

						track_method->object_id = track_method->object->get_instance_id();

						track = track_method;

					} break;
					case Animation::TYPE_BEZIER: {
						TrackCacheBezier *track_bezier = memnew(TrackCacheBezier);

						if (resource.is_valid()) {
							track_bezier->object = resource.ptr();
						} else {
							track_bezier->object = child;
						}

						track_bezier->subpath = leftover_path;
						track_bezier->object_id = track_bezier->object->get_instance_id();

						track = track_bezier;

						if (has_reset_anim) {
							int rt = reset_anim->find_track(path, track_type);
							if (rt >= 0 && reset_anim->track_get_key_count(rt) > 0) {
								track_bezier->init_value = (reset_anim->track_get_key_value(rt, 0).operator Array())[0];
							}
						}
					} break;
					case Animation::TYPE_AUDIO: {
						TrackCacheAudio *track_audio = memnew(TrackCacheAudio);

						track_audio->object = child;
						track_audio->object_id = track_audio->object->get_instance_id();
						track_audio->audio_stream.instantiate();
						track_audio->audio_stream->set_polyphony(audio_max_polyphony);

						track = track_audio;

					} break;
					case Animation::TYPE_ANIMATION: {
						TrackCacheAnimation *track_animation = memnew(TrackCacheAnimation);

						track_animation->object = child;
						track_animation->object_id = track_animation->object->get_instance_id();

						track = track_animation;

					} break;
					default: {
						ERR_PRINT("Animation corrupted (invalid track type)");
						continue;
					}
				}

				track_cache[path] = track;
			} else if (track_cache_type == Animation::TYPE_POSITION_3D) {
				TrackCacheTransform *track_xform = static_cast<TrackCacheTransform *>(track);
				if (track->setup_pass != setup_pass) {
					track_xform->loc_used = false;
					track_xform->rot_used = false;
					track_xform->scale_used = false;
				}
				switch (track_type) {
					case Animation::TYPE_POSITION_3D: {
						track_xform->loc_used = true;
					} break;
					case Animation::TYPE_ROTATION_3D: {
						track_xform->rot_used = true;
					} break;
					case Animation::TYPE_SCALE_3D: {
						track_xform->scale_used = true;
					} break;
					default: {
					}
				}
			} else if (track_cache_type == Animation::TYPE_VALUE) {
				// If it has at least one angle interpolation, it also uses angle interpolation for blending.
				TrackCacheValue *track_value = static_cast<TrackCacheValue *>(track);
				bool was_discrete = track_value->is_discrete;
				bool was_using_angle = track_value->is_using_angle;
				track_value->is_discrete |= anim->value_track_get_update_mode(i) == Animation::UPDATE_DISCRETE;
				track_value->is_using_angle |= anim->track_get_interpolation_type(i) == Animation::INTERPOLATION_LINEAR_ANGLE || anim->track_get_interpolation_type(i) == Animation::INTERPOLATION_CUBIC_ANGLE;

				if (was_discrete != track_value->is_discrete) {
					ERR_PRINT_ED("Value Track: " + String(path) + " with different update modes are blended. Blending prioritizes Discrete mode, so other update mode tracks will not be blended.");
				}
				if (was_using_angle != track_value->is_using_angle) {
					WARN_PRINT_ED("Value Track: " + String(path) + " with different interpolation types for rotation are blended. Blending prioritizes angle interpolation, so the blending result uses the shortest path referenced to the initial (RESET animation) value.");
				}
			}

			track->setup_pass = setup_pass;
		}
	}

	List<NodePath> to_delete;

	for (const KeyValue<NodePath, TrackCache *> &K : track_cache) {
		TrackCache *tc = track_cache[K.key];
		if (tc->setup_pass != setup_pass) {
			to_delete.push_back(K.key);
		}
	}

	while (to_delete.front()) {
		NodePath np = to_delete.front()->get();
		memdelete(track_cache[np]);
		track_cache.erase(np);
		to_delete.pop_front();
	}

	state.track_map.clear();

	int idx = 0;
	for (const KeyValue<NodePath, TrackCache *> &K : track_cache) {
		state.track_map[K.key] = idx;
		idx++;
	}

	state.track_count = idx;

	cache_valid = true;

	return true;
}

void AnimationTree::_animation_player_changed() {
	emit_signal(SNAME("animation_player_changed"));
	_clear_caches();
}

void AnimationTree::_clear_caches() {
	_clear_audio_streams();
	_clear_playing_caches();
	for (KeyValue<NodePath, TrackCache *> &K : track_cache) {
		memdelete(K.value);
	}
	track_cache.clear();
	cache_valid = false;
}

void AnimationTree::_clear_audio_streams() {
	for (int i = 0; i < playing_audio_stream_players.size(); i++) {
		playing_audio_stream_players[i]->call(SNAME("stop"));
		playing_audio_stream_players[i]->call(SNAME("set_stream"), Ref<AudioStream>());
	}
	playing_audio_stream_players.clear();
}

void AnimationTree::_clear_playing_caches() {
	for (const TrackCache *E : playing_caches) {
		if (ObjectDB::get_instance(E->object_id)) {
			E->object->call(SNAME("stop"));
		}
	}
	playing_caches.clear();
}

void AnimationTree::_call_object(Object *p_object, const StringName &p_method, const Vector<Variant> &p_params, bool p_deferred) {
	// Separate function to use alloca() more efficiently
	const Variant **argptrs = (const Variant **)alloca(sizeof(const Variant **) * p_params.size());
	const Variant *args = p_params.ptr();
	uint32_t argcount = p_params.size();
	for (uint32_t i = 0; i < argcount; i++) {
		argptrs[i] = &args[i];
	}
	if (p_deferred) {
		MessageQueue::get_singleton()->push_callp(p_object, p_method, argptrs, argcount);
	} else {
		Callable::CallError ce;
		p_object->callp(p_method, argptrs, argcount, ce);
	}
}
void AnimationTree::_process_graph(double p_delta) {
	_update_properties(); //if properties need updating, update them

	//check all tracks, see if they need modification
	root_motion_position = Vector3(0, 0, 0);
	root_motion_rotation = Quaternion(0, 0, 0, 1);
	root_motion_scale = Vector3(0, 0, 0);

	if (!root.is_valid()) {
		ERR_PRINT("AnimationTree: root AnimationNode is not set, disabling playback.");
		set_active(false);
		cache_valid = false;
		return;
	}

	if (!has_node(animation_player)) {
		ERR_PRINT("AnimationTree: no valid AnimationPlayer path set, disabling playback");
		set_active(false);
		cache_valid = false;
		return;
	}

	AnimationPlayer *player = Object::cast_to<AnimationPlayer>(get_node(animation_player));

	ObjectID current_animation_player;

	if (player) {
		current_animation_player = player->get_instance_id();
	}

	if (last_animation_player != current_animation_player) {
		if (last_animation_player.is_valid()) {
			Object *old_player = ObjectDB::get_instance(last_animation_player);
			if (old_player) {
				old_player->disconnect("caches_cleared", callable_mp(this, &AnimationTree::_clear_caches));
			}
		}

		if (player) {
			player->connect("caches_cleared", callable_mp(this, &AnimationTree::_clear_caches));
		}

		last_animation_player = current_animation_player;
	}

	if (!player) {
		ERR_PRINT("AnimationTree: path points to a node not an AnimationPlayer, disabling playback");
		set_active(false);
		cache_valid = false;
		return;
	}

	if (!cache_valid) {
		if (!_update_caches(player)) {
			return;
		}
	}

	{ //setup

		process_pass++;

		state.valid = true;
		state.invalid_reasons = "";
		state.animation_states.clear(); //will need to be re-created
		state.player = player;
		state.last_pass = process_pass;
		state.tree = this;

		// root source blends

		root->blends.resize(state.track_count);
		real_t *src_blendsw = root->blends.ptrw();
		for (int i = 0; i < state.track_count; i++) {
			src_blendsw[i] = 1.0; //by default all go to 1 for the root input
		}
	}

	//process

	{
		if (started) {
			//if started, seek
			root->_pre_process(SceneStringNames::get_singleton()->parameters_base_path, nullptr, &state, 0, true, false, Vector<StringName>());
			started = false;
		}

		root->_pre_process(SceneStringNames::get_singleton()->parameters_base_path, nullptr, &state, p_delta, false, false, Vector<StringName>());
	}

	if (!state.valid) {
		return; //state is not valid. do nothing.
	}

	// Init all value/transform/blend/bezier tracks that track_cache has.
	{
		for (const KeyValue<NodePath, TrackCache *> &K : track_cache) {
			TrackCache *track = K.value;

			switch (track->type) {
				case Animation::TYPE_POSITION_3D: {
					TrackCacheTransform *t = static_cast<TrackCacheTransform *>(track);
					if (track->root_motion) {
						root_motion_cache.loc = Vector3(0, 0, 0);
						root_motion_cache.rot = Quaternion(0, 0, 0, 1);
						root_motion_cache.scale = Vector3(1, 1, 1);
					}
					t->loc = t->init_loc;
					t->rot = t->init_rot;
					t->scale = t->init_scale;
				} break;
				case Animation::TYPE_BLEND_SHAPE: {
					TrackCacheBlendShape *t = static_cast<TrackCacheBlendShape *>(track);
					t->value = t->init_value;
				} break;
				case Animation::TYPE_VALUE: {
					TrackCacheValue *t = static_cast<TrackCacheValue *>(track);
					t->value = t->init_value;
				} break;
				case Animation::TYPE_BEZIER: {
					TrackCacheBezier *t = static_cast<TrackCacheBezier *>(track);
					t->value = t->init_value;
				} break;
				case Animation::TYPE_AUDIO: {
					TrackCacheAudio *t = static_cast<TrackCacheAudio *>(track);
					for (KeyValue<ObjectID, PlayingAudioTrackInfo> &L : t->playing_streams) {
						PlayingAudioTrackInfo &track_info = L.value;
						track_info.volume = 0.0;
					}
				} break;
				default: {
				} break;
			}
		}
	}

	// Apply value/transform/blend/bezier blends to track caches and execute method/audio/animation tracks.
	{
#ifdef TOOLS_ENABLED
		bool can_call = is_inside_tree() && !Engine::get_singleton()->is_editor_hint();
#endif // TOOLS_ENABLED
		for (const AnimationNode::AnimationState &as : state.animation_states) {
			Ref<Animation> a = as.animation;
			double time = as.time;
			double delta = as.delta;
			real_t weight = as.blend;
			bool seeked = as.seeked;
			Animation::LoopedFlag looped_flag = as.looped_flag;
			bool is_external_seeking = as.is_external_seeking;
			bool backward = signbit(delta); // This flag is used by the root motion calculates or detecting the end of audio stream.
#ifndef _3D_DISABLED
			bool calc_root = !seeked || is_external_seeking;
#endif // _3D_DISABLED

			for (int i = 0; i < a->get_track_count(); i++) {
				if (!a->track_is_enabled(i)) {
					continue;
				}

				NodePath path = a->track_get_path(i);
				if (!track_cache.has(path)) {
					continue; // No path, but avoid error spamming.
				}
				TrackCache *track = track_cache[path];

				ERR_CONTINUE(!state.track_map.has(path));
				int blend_idx = state.track_map[path];
				ERR_CONTINUE(blend_idx < 0 || blend_idx >= state.track_count);
				real_t blend = (as.track_blends)[blend_idx] * weight;

				Animation::TrackType ttype = a->track_get_type(i);
				if (ttype != Animation::TYPE_POSITION_3D && ttype != Animation::TYPE_ROTATION_3D && ttype != Animation::TYPE_SCALE_3D && track->type != ttype) {
					//broken animation, but avoid error spamming
					continue;
				}
				track->root_motion = root_motion_track == path;

				switch (ttype) {
					case Animation::TYPE_POSITION_3D: {
#ifndef _3D_DISABLED
						if (Math::is_zero_approx(blend)) {
							continue; // Nothing to blend.
						}
						TrackCacheTransform *t = static_cast<TrackCacheTransform *>(track);

						if (track->root_motion && calc_root) {
							double prev_time = time - delta;
							if (!backward) {
								if (prev_time < 0) {
									switch (a->get_loop_mode()) {
										case Animation::LOOP_NONE: {
											prev_time = 0;
										} break;
										case Animation::LOOP_LINEAR: {
											prev_time = Math::fposmod(prev_time, (double)a->get_length());
										} break;
										case Animation::LOOP_PINGPONG: {
											prev_time = Math::pingpong(prev_time, (double)a->get_length());
										} break;
										default:
											break;
									}
								}
							} else {
								if (prev_time > a->get_length()) {
									switch (a->get_loop_mode()) {
										case Animation::LOOP_NONE: {
											prev_time = (double)a->get_length();
										} break;
										case Animation::LOOP_LINEAR: {
											prev_time = Math::fposmod(prev_time, (double)a->get_length());
										} break;
										case Animation::LOOP_PINGPONG: {
											prev_time = Math::pingpong(prev_time, (double)a->get_length());
										} break;
										default:
											break;
									}
								}
							}

							Vector3 loc[2];

							if (!backward) {
								if (prev_time > time) {
									Error err = a->try_position_track_interpolate(i, prev_time, &loc[0]);
									if (err != OK) {
										continue;
									}
									loc[0] = post_process_key_value(a, i, loc[0], t->object, t->bone_idx);
									a->try_position_track_interpolate(i, (double)a->get_length(), &loc[1]);
									loc[1] = post_process_key_value(a, i, loc[1], t->object, t->bone_idx);
									root_motion_cache.loc += (loc[1] - loc[0]) * blend;
									prev_time = 0;
								}
							} else {
								if (prev_time < time) {
									Error err = a->try_position_track_interpolate(i, prev_time, &loc[0]);
									if (err != OK) {
										continue;
									}
									loc[0] = post_process_key_value(a, i, loc[0], t->object, t->bone_idx);
									a->try_position_track_interpolate(i, 0, &loc[1]);
									loc[1] = post_process_key_value(a, i, loc[1], t->object, t->bone_idx);
									root_motion_cache.loc += (loc[1] - loc[0]) * blend;
									prev_time = (double)a->get_length();
								}
							}

							Error err = a->try_position_track_interpolate(i, prev_time, &loc[0]);
							if (err != OK) {
								continue;
							}
							loc[0] = post_process_key_value(a, i, loc[0], t->object, t->bone_idx);
							a->try_position_track_interpolate(i, time, &loc[1]);
							loc[1] = post_process_key_value(a, i, loc[1], t->object, t->bone_idx);
							root_motion_cache.loc += (loc[1] - loc[0]) * blend;
							prev_time = !backward ? 0 : (double)a->get_length();
						}

						{
							Vector3 loc;

							Error err = a->try_position_track_interpolate(i, time, &loc);
							if (err != OK) {
								continue;
							}
							loc = post_process_key_value(a, i, loc, t->object, t->bone_idx);

							t->loc += (loc - t->init_loc) * blend;
						}
#endif // _3D_DISABLED
					} break;
					case Animation::TYPE_ROTATION_3D: {
#ifndef _3D_DISABLED
						if (Math::is_zero_approx(blend)) {
							continue; // Nothing to blend.
						}
						TrackCacheTransform *t = static_cast<TrackCacheTransform *>(track);

						if (track->root_motion && calc_root) {
							double prev_time = time - delta;
							if (!backward) {
								if (prev_time < 0) {
									switch (a->get_loop_mode()) {
										case Animation::LOOP_NONE: {
											prev_time = 0;
										} break;
										case Animation::LOOP_LINEAR: {
											prev_time = Math::fposmod(prev_time, (double)a->get_length());
										} break;
										case Animation::LOOP_PINGPONG: {
											prev_time = Math::pingpong(prev_time, (double)a->get_length());
										} break;
										default:
											break;
									}
								}
							} else {
								if (prev_time > a->get_length()) {
									switch (a->get_loop_mode()) {
										case Animation::LOOP_NONE: {
											prev_time = (double)a->get_length();
										} break;
										case Animation::LOOP_LINEAR: {
											prev_time = Math::fposmod(prev_time, (double)a->get_length());
										} break;
										case Animation::LOOP_PINGPONG: {
											prev_time = Math::pingpong(prev_time, (double)a->get_length());
										} break;
										default:
											break;
									}
								}
							}

							Quaternion rot[2];

							if (!backward) {
								if (prev_time > time) {
									Error err = a->try_rotation_track_interpolate(i, prev_time, &rot[0]);
									if (err != OK) {
										continue;
									}
									rot[0] = post_process_key_value(a, i, rot[0], t->object, t->bone_idx);
									a->try_rotation_track_interpolate(i, (double)a->get_length(), &rot[1]);
									rot[1] = post_process_key_value(a, i, rot[1], t->object, t->bone_idx);
									root_motion_cache.rot = (root_motion_cache.rot * Quaternion().slerp(rot[0].inverse() * rot[1], blend)).normalized();
									prev_time = 0;
								}
							} else {
								if (prev_time < time) {
									Error err = a->try_rotation_track_interpolate(i, prev_time, &rot[0]);
									if (err != OK) {
										continue;
									}
									rot[0] = post_process_key_value(a, i, rot[0], t->object, t->bone_idx);
									a->try_rotation_track_interpolate(i, 0, &rot[1]);
									root_motion_cache.rot = (root_motion_cache.rot * Quaternion().slerp(rot[0].inverse() * rot[1], blend)).normalized();
									prev_time = (double)a->get_length();
								}
							}

							Error err = a->try_rotation_track_interpolate(i, prev_time, &rot[0]);
							if (err != OK) {
								continue;
							}
							rot[0] = post_process_key_value(a, i, rot[0], t->object, t->bone_idx);

							a->try_rotation_track_interpolate(i, time, &rot[1]);
							rot[1] = post_process_key_value(a, i, rot[1], t->object, t->bone_idx);
							root_motion_cache.rot = (root_motion_cache.rot * Quaternion().slerp(rot[0].inverse() * rot[1], blend)).normalized();
							prev_time = !backward ? 0 : (double)a->get_length();
						}

						{
							Quaternion rot;

							Error err = a->try_rotation_track_interpolate(i, time, &rot);
							if (err != OK) {
								continue;
							}
							rot = post_process_key_value(a, i, rot, t->object, t->bone_idx);

							t->rot = (t->rot * Quaternion().slerp(t->init_rot.inverse() * rot, blend)).normalized();
						}
#endif // _3D_DISABLED
					} break;
					case Animation::TYPE_SCALE_3D: {
#ifndef _3D_DISABLED
						if (Math::is_zero_approx(blend)) {
							continue; // Nothing to blend.
						}
						TrackCacheTransform *t = static_cast<TrackCacheTransform *>(track);

						if (track->root_motion && calc_root) {
							double prev_time = time - delta;
							if (!backward) {
								if (prev_time < 0) {
									switch (a->get_loop_mode()) {
										case Animation::LOOP_NONE: {
											prev_time = 0;
										} break;
										case Animation::LOOP_LINEAR: {
											prev_time = Math::fposmod(prev_time, (double)a->get_length());
										} break;
										case Animation::LOOP_PINGPONG: {
											prev_time = Math::pingpong(prev_time, (double)a->get_length());
										} break;
										default:
											break;
									}
								}
							} else {
								if (prev_time > a->get_length()) {
									switch (a->get_loop_mode()) {
										case Animation::LOOP_NONE: {
											prev_time = (double)a->get_length();
										} break;
										case Animation::LOOP_LINEAR: {
											prev_time = Math::fposmod(prev_time, (double)a->get_length());
										} break;
										case Animation::LOOP_PINGPONG: {
											prev_time = Math::pingpong(prev_time, (double)a->get_length());
										} break;
										default:
											break;
									}
								}
							}

							Vector3 scale[2];

							if (!backward) {
								if (prev_time > time) {
									Error err = a->try_scale_track_interpolate(i, prev_time, &scale[0]);
									if (err != OK) {
										continue;
									}
									scale[0] = post_process_key_value(a, i, scale[0], t->object, t->bone_idx);
									a->try_scale_track_interpolate(i, (double)a->get_length(), &scale[1]);
									root_motion_cache.scale += (scale[1] - scale[0]) * blend;
									scale[1] = post_process_key_value(a, i, scale[1], t->object, t->bone_idx);
									prev_time = 0;
								}
							} else {
								if (prev_time < time) {
									Error err = a->try_scale_track_interpolate(i, prev_time, &scale[0]);
									if (err != OK) {
										continue;
									}
									scale[0] = post_process_key_value(a, i, scale[0], t->object, t->bone_idx);
									a->try_scale_track_interpolate(i, 0, &scale[1]);
									scale[1] = post_process_key_value(a, i, scale[1], t->object, t->bone_idx);
									root_motion_cache.scale += (scale[1] - scale[0]) * blend;
									prev_time = (double)a->get_length();
								}
							}

							Error err = a->try_scale_track_interpolate(i, prev_time, &scale[0]);
							if (err != OK) {
								continue;
							}
							scale[0] = post_process_key_value(a, i, scale[0], t->object, t->bone_idx);

							a->try_scale_track_interpolate(i, time, &scale[1]);
							scale[1] = post_process_key_value(a, i, scale[1], t->object, t->bone_idx);
							root_motion_cache.scale += (scale[1] - scale[0]) * blend;
							prev_time = !backward ? 0 : (double)a->get_length();
						}

						{
							Vector3 scale;

							Error err = a->try_scale_track_interpolate(i, time, &scale);
							if (err != OK) {
								continue;
							}
							scale = post_process_key_value(a, i, scale, t->object, t->bone_idx);

							t->scale += (scale - t->init_scale) * blend;
						}
#endif // _3D_DISABLED
					} break;
					case Animation::TYPE_BLEND_SHAPE: {
#ifndef _3D_DISABLED
						if (Math::is_zero_approx(blend)) {
							continue; // Nothing to blend.
						}
						TrackCacheBlendShape *t = static_cast<TrackCacheBlendShape *>(track);

						float value;

						Error err = a->try_blend_shape_track_interpolate(i, time, &value);
						//ERR_CONTINUE(err!=OK); //used for testing, should be removed

						if (err != OK) {
							continue;
						}
						value = post_process_key_value(a, i, value, t->object, t->shape_index);

						t->value += (value - t->init_value) * blend;
#endif // _3D_DISABLED
					} break;
					case Animation::TYPE_VALUE: {
						if (Math::is_zero_approx(blend)) {
							continue; // Nothing to blend.
						}
						TrackCacheValue *t = static_cast<TrackCacheValue *>(track);

						Animation::UpdateMode update_mode = a->value_track_get_update_mode(i);

						if (update_mode == Animation::UPDATE_CONTINUOUS || update_mode == Animation::UPDATE_CAPTURE) {
							Variant value = a->value_track_interpolate(i, time);
							value = post_process_key_value(a, i, value, t->object);

							if (value == Variant()) {
								continue;
							}

							// Special case for angle interpolation.
							if (t->is_using_angle) {
								// For blending consistency, it prevents rotation of more than 180 degrees from init_value.
								// This is the same as for Quaternion blends.
								float rot_a = t->value;
								float rot_b = value;
								float rot_init = t->init_value;
								rot_a = Math::fposmod(rot_a, (float)Math_TAU);
								rot_b = Math::fposmod(rot_b, (float)Math_TAU);
								rot_init = Math::fposmod(rot_init, (float)Math_TAU);
								if (rot_init < Math_PI) {
									rot_a = rot_a > rot_init + Math_PI ? rot_a - Math_TAU : rot_a;
									rot_b = rot_b > rot_init + Math_PI ? rot_b - Math_TAU : rot_b;
								} else {
									rot_a = rot_a < rot_init - Math_PI ? rot_a + Math_TAU : rot_a;
									rot_b = rot_b < rot_init - Math_PI ? rot_b + Math_TAU : rot_b;
								}
								t->value = Math::fposmod(rot_a + (rot_b - rot_init) * (float)blend, (float)Math_TAU);
							} else {
								if (t->init_value.get_type() == Variant::BOOL) {
									value = Animation::subtract_variant(value.operator real_t(), t->init_value.operator real_t());
									t->value = Animation::blend_variant(t->value.operator real_t(), value.operator real_t(), blend);
								} else {
									value = Animation::subtract_variant(value, t->init_value);
									t->value = Animation::blend_variant(t->value, value, blend);
								}
							}
						} else {
							if (seeked) {
								int idx = a->track_find_key(i, time, is_external_seeking ? Animation::FIND_MODE_NEAREST : Animation::FIND_MODE_EXACT);
								if (idx < 0) {
									continue;
								}
								Variant value = a->track_get_key_value(i, idx);
								value = post_process_key_value(a, i, value, t->object);
								t->object->set_indexed(t->subpath, value);
							} else {
								List<int> indices;
								a->track_get_key_indices_in_range(i, time, delta, &indices, looped_flag);
								for (int &F : indices) {
									Variant value = a->track_get_key_value(i, F);
									value = post_process_key_value(a, i, value, t->object);
									t->object->set_indexed(t->subpath, value);
								}
							}
						}

					} break;
					case Animation::TYPE_METHOD: {
#ifdef TOOLS_ENABLED
						if (!can_call) {
							continue;
						}
#endif // TOOLS_ENABLED
						if (Math::is_zero_approx(blend)) {
							continue; // Nothing to blend.
						}
						TrackCacheMethod *t = static_cast<TrackCacheMethod *>(track);

						if (seeked) {
							int idx = a->track_find_key(i, time, is_external_seeking ? Animation::FIND_MODE_NEAREST : Animation::FIND_MODE_EXACT);
							if (idx < 0) {
								continue;
							}
							StringName method = a->method_track_get_name(i, idx);
							Vector<Variant> params = a->method_track_get_params(i, idx);
							_call_object(t->object, method, params, false);
						} else {
							List<int> indices;
							a->track_get_key_indices_in_range(i, time, delta, &indices, looped_flag);
							for (int &F : indices) {
								StringName method = a->method_track_get_name(i, F);
								Vector<Variant> params = a->method_track_get_params(i, F);
								_call_object(t->object, method, params, true);
							}
						}
					} break;
					case Animation::TYPE_BEZIER: {
						if (Math::is_zero_approx(blend)) {
							continue; // Nothing to blend.
						}
						TrackCacheBezier *t = static_cast<TrackCacheBezier *>(track);

						real_t bezier = a->bezier_track_interpolate(i, time);
						bezier = post_process_key_value(a, i, bezier, t->object);

						t->value += (bezier - t->init_value) * blend;
					} break;
					case Animation::TYPE_AUDIO: {
						TrackCacheAudio *t = static_cast<TrackCacheAudio *>(track);

						Node *asp = Object::cast_to<Node>(t->object);
						if (!asp) {
							t->playing_streams.clear();
							continue;
						}

						ObjectID oid = a->get_instance_id();
						if (!t->playing_streams.has(oid)) {
							t->playing_streams[oid] = PlayingAudioTrackInfo();
						}
						// The end of audio should be observed even if the blend value is 0, build up the information and store to the cache for that.
						PlayingAudioTrackInfo &track_info = t->playing_streams[oid];
						track_info.length = a->get_length();
						track_info.time = time;
						track_info.volume += blend;
						track_info.loop = a->get_loop_mode() != Animation::LOOP_NONE;
						track_info.backward = backward;
						track_info.use_blend = a->audio_track_is_use_blend(i);

						HashMap<int, PlayingAudioStreamInfo> &map = track_info.stream_info;
						// Find stream.
						int idx = -1;
						if (seeked) {
							idx = a->track_find_key(i, time, is_external_seeking ? Animation::FIND_MODE_NEAREST : Animation::FIND_MODE_EXACT);
							// Discard previous stream when seeking.
							if (map.has(idx)) {
								t->audio_stream_playback->stop_stream(map[idx].index);
								map.erase(idx);
							}
						} else {
							List<int> to_play;
							a->track_get_key_indices_in_range(i, time, delta, &to_play, looped_flag);
							if (to_play.size()) {
								idx = to_play.back()->get();
							}
						}
						if (idx < 0) {
							continue;
						}

						// Play stream.
						Ref<AudioStream> stream = a->audio_track_get_key_stream(i, idx);
						if (stream.is_valid()) {
							double start_ofs = a->audio_track_get_key_start_offset(i, idx);
							double end_ofs = a->audio_track_get_key_end_offset(i, idx);
							double len = stream->get_length();

							if (seeked) {
								start_ofs += time - a->track_get_key_time(i, idx);
							}

							if (t->object->call(SNAME("get_stream")) != t->audio_stream) {
								t->object->call(SNAME("set_stream"), t->audio_stream);
								t->audio_stream_playback.unref();
								if (!playing_audio_stream_players.has(asp)) {
									playing_audio_stream_players.push_back(asp);
								}
							}
							if (!t->object->call(SNAME("is_playing"))) {
								t->object->call(SNAME("play"));
							}
							if (!t->object->call(SNAME("has_stream_playback"))) {
								t->audio_stream_playback.unref();
								continue;
							}
							if (t->audio_stream_playback.is_null()) {
								t->audio_stream_playback = t->object->call(SNAME("get_stream_playback"));
							}

							PlayingAudioStreamInfo pasi;
							pasi.index = t->audio_stream_playback->play_stream(stream, start_ofs);
							pasi.start = time;
							if (len && end_ofs > 0) { // Force an end at a time.
								pasi.len = len - start_ofs - end_ofs;
							} else {
								pasi.len = 0;
							}
							map[idx] = pasi;
						}

					} break;
					case Animation::TYPE_ANIMATION: {
						if (Math::is_zero_approx(blend)) {
							continue; // Nothing to blend.
						}
						TrackCacheAnimation *t = static_cast<TrackCacheAnimation *>(track);

						AnimationPlayer *player2 = Object::cast_to<AnimationPlayer>(t->object);

						if (!player2) {
							continue;
						}

						if (seeked) {
							//seek
							int idx = a->track_find_key(i, time, is_external_seeking ? Animation::FIND_MODE_NEAREST : Animation::FIND_MODE_EXACT);
							if (idx < 0) {
								continue;
							}

							double pos = a->track_get_key_time(i, idx);

							StringName anim_name = a->animation_track_get_key_animation(i, idx);
							if (String(anim_name) == "[stop]" || !player2->has_animation(anim_name)) {
								continue;
							}

							Ref<Animation> anim = player2->get_animation(anim_name);

							double at_anim_pos = 0.0;

							switch (anim->get_loop_mode()) {
								case Animation::LOOP_NONE: {
									at_anim_pos = MAX((double)anim->get_length(), time - pos); //seek to end
								} break;
								case Animation::LOOP_LINEAR: {
									at_anim_pos = Math::fposmod(time - pos, (double)anim->get_length()); //seek to loop
								} break;
								case Animation::LOOP_PINGPONG: {
									at_anim_pos = Math::pingpong(time - pos, (double)a->get_length());
								} break;
								default:
									break;
							}

							if (player2->is_playing() || seeked) {
								player2->seek(at_anim_pos);
								player2->play(anim_name);
								t->playing = true;
								playing_caches.insert(t);
							} else {
								player2->set_assigned_animation(anim_name);
								player2->seek(at_anim_pos, true);
							}
						} else {
							//find stuff to play
							List<int> to_play;
							a->track_get_key_indices_in_range(i, time, delta, &to_play, looped_flag);
							if (to_play.size()) {
								int idx = to_play.back()->get();

								StringName anim_name = a->animation_track_get_key_animation(i, idx);
								if (String(anim_name) == "[stop]" || !player2->has_animation(anim_name)) {
									if (playing_caches.has(t)) {
										playing_caches.erase(t);
										player2->stop();
										t->playing = false;
									}
								} else {
									player2->play(anim_name);
									t->playing = true;
									playing_caches.insert(t);
								}
							}
						}

					} break;
				}
			}
		}
	}

	{
		// finally, set the tracks
		for (const KeyValue<NodePath, TrackCache *> &K : track_cache) {
			TrackCache *track = K.value;

			switch (track->type) {
				case Animation::TYPE_POSITION_3D: {
#ifndef _3D_DISABLED
					TrackCacheTransform *t = static_cast<TrackCacheTransform *>(track);

					if (t->root_motion) {
						root_motion_position = root_motion_cache.loc;
						root_motion_rotation = root_motion_cache.rot;
						root_motion_scale = root_motion_cache.scale - Vector3(1, 1, 1);
						root_motion_position_accumulator = t->loc;
						root_motion_rotation_accumulator = t->rot;
						root_motion_scale_accumulator = t->scale;
					} else if (t->skeleton && t->bone_idx >= 0) {
						if (t->loc_used) {
							t->skeleton->set_bone_pose_position(t->bone_idx, t->loc);
						}
						if (t->rot_used) {
							t->skeleton->set_bone_pose_rotation(t->bone_idx, t->rot);
						}
						if (t->scale_used) {
							t->skeleton->set_bone_pose_scale(t->bone_idx, t->scale);
						}

					} else if (!t->skeleton) {
						if (t->loc_used) {
							t->node_3d->set_position(t->loc);
						}
						if (t->rot_used) {
							t->node_3d->set_rotation(t->rot.get_euler());
						}
						if (t->scale_used) {
							t->node_3d->set_scale(t->scale);
						}
					}
#endif // _3D_DISABLED
				} break;
				case Animation::TYPE_BLEND_SHAPE: {
#ifndef _3D_DISABLED
					TrackCacheBlendShape *t = static_cast<TrackCacheBlendShape *>(track);

					if (t->mesh_3d) {
						t->mesh_3d->set_blend_shape_value(t->shape_index, t->value);
					}
#endif // _3D_DISABLED
				} break;
				case Animation::TYPE_VALUE: {
					TrackCacheValue *t = static_cast<TrackCacheValue *>(track);

					if (t->is_discrete) {
						break; // Don't overwrite the value set by UPDATE_DISCRETE.
					}

					if (t->init_value.get_type() == Variant::BOOL) {
						t->object->set_indexed(t->subpath, t->value.operator real_t() >= 0.5);
					} else {
						t->object->set_indexed(t->subpath, t->value);
					}

				} break;
				case Animation::TYPE_BEZIER: {
					TrackCacheBezier *t = static_cast<TrackCacheBezier *>(track);

					t->object->set_indexed(t->subpath, t->value);

				} break;
				case Animation::TYPE_AUDIO: {
					TrackCacheAudio *t = static_cast<TrackCacheAudio *>(track);

					// Audio ending process.
					LocalVector<ObjectID> erase_maps;
					for (KeyValue<ObjectID, PlayingAudioTrackInfo> &L : t->playing_streams) {
						PlayingAudioTrackInfo &track_info = L.value;
						float db = Math::linear_to_db(track_info.use_blend ? track_info.volume : 1.0);
						LocalVector<int> erase_streams;
						HashMap<int, PlayingAudioStreamInfo> &map = track_info.stream_info;
						for (const KeyValue<int, PlayingAudioStreamInfo> &M : map) {
							PlayingAudioStreamInfo pasi = M.value;

							bool stop = false;
							if (!t->audio_stream_playback->is_stream_playing(pasi.index)) {
								stop = true;
							}
							if (!track_info.loop) {
								if (!track_info.backward) {
									if (track_info.time < pasi.start) {
										stop = true;
									}
								} else if (track_info.backward) {
									if (track_info.time > pasi.start) {
										stop = true;
									}
								}
							}
							if (pasi.len > 0) {
								double len = 0.0;
								if (!track_info.backward) {
									len = pasi.start > track_info.time ? (track_info.length - pasi.start) + track_info.time : track_info.time - pasi.start;
								} else {
									len = pasi.start < track_info.time ? (track_info.length - track_info.time) + pasi.start : pasi.start - track_info.time;
								}
								if (len > pasi.len) {
									stop = true;
								}
							}
							if (stop) {
								// Time to stop.
								t->audio_stream_playback->stop_stream(pasi.index);
								erase_streams.push_back(M.key);
							} else {
								t->audio_stream_playback->set_stream_volume(pasi.index, db);
							}
						}
						for (uint32_t erase_idx = 0; erase_idx < erase_streams.size(); erase_idx++) {
							map.erase(erase_streams[erase_idx]);
						}
						if (map.size() == 0) {
							erase_maps.push_back(L.key);
						}
					}
					for (uint32_t erase_idx = 0; erase_idx < erase_maps.size(); erase_idx++) {
						t->playing_streams.erase(erase_maps[erase_idx]);
					}
				} break;
				default: {
				} //the rest don't matter
			}
		}
	}
}

Variant AnimationTree::post_process_key_value(const Ref<Animation> &p_anim, int p_track, Variant p_value, const Object *p_object, int p_object_idx) {
	Variant res;
	if (GDVIRTUAL_CALL(_post_process_key_value, p_anim, p_track, p_value, const_cast<Object *>(p_object), p_object_idx, res)) {
		return res;
	}

	return _post_process_key_value(p_anim, p_track, p_value, p_object, p_object_idx);
}

Variant AnimationTree::_post_process_key_value(const Ref<Animation> &p_anim, int p_track, Variant p_value, const Object *p_object, int p_object_idx) {
	switch (p_anim->track_get_type(p_track)) {
#ifndef _3D_DISABLED
		case Animation::TYPE_POSITION_3D: {
			if (p_object_idx >= 0) {
				const Skeleton3D *skel = Object::cast_to<Skeleton3D>(p_object);
				return Vector3(p_value) * skel->get_motion_scale();
			}
			return p_value;
		} break;
#endif // _3D_DISABLED
		default: {
		} break;
	}
	return p_value;
}

void AnimationTree::advance(double p_time) {
	_process_graph(p_time);
}

void AnimationTree::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_setup_animation_player();
			if (last_animation_player.is_valid()) {
				Object *player = ObjectDB::get_instance(last_animation_player);
				if (player) {
					player->connect("caches_cleared", callable_mp(this, &AnimationTree::_clear_caches));
				}
			}
		} break;

		case NOTIFICATION_EXIT_TREE: {
			_clear_caches();
			if (last_animation_player.is_valid()) {
				Object *player = ObjectDB::get_instance(last_animation_player);
				if (player) {
					player->disconnect("caches_cleared", callable_mp(this, &AnimationTree::_clear_caches));
				}
			}
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			if (active && process_callback == ANIMATION_PROCESS_IDLE) {
				_process_graph(get_process_delta_time());
			}
		} break;

		case NOTIFICATION_INTERNAL_PHYSICS_PROCESS: {
			if (active && process_callback == ANIMATION_PROCESS_PHYSICS) {
				_process_graph(get_physics_process_delta_time());
			}
		} break;
	}
}

void AnimationTree::_setup_animation_player() {
	if (!is_inside_tree()) {
		return;
	}

	cache_valid = false;

	AnimationPlayer *new_player = nullptr;
	if (!animation_player.is_empty()) {
		new_player = Object::cast_to<AnimationPlayer>(get_node_or_null(animation_player));
		if (new_player && !new_player->is_connected("animation_list_changed", callable_mp(this, &AnimationTree::_animation_player_changed))) {
			new_player->connect("animation_list_changed", callable_mp(this, &AnimationTree::_animation_player_changed));
		}
	}

	if (new_player) {
		if (!last_animation_player.is_valid()) {
			// Animation player set newly.
			emit_signal(SNAME("animation_player_changed"));
			return;
		} else if (last_animation_player == new_player->get_instance_id()) {
			// Animation player isn't changed.
			return;
		}
	} else if (!last_animation_player.is_valid()) {
		// Animation player is being empty.
		return;
	}

	AnimationPlayer *old_player = Object::cast_to<AnimationPlayer>(ObjectDB::get_instance(last_animation_player));
	if (old_player && old_player->is_connected("animation_list_changed", callable_mp(this, &AnimationTree::_animation_player_changed))) {
		old_player->disconnect("animation_list_changed", callable_mp(this, &AnimationTree::_animation_player_changed));
	}
	emit_signal(SNAME("animation_player_changed"));
}

void AnimationTree::set_animation_player(const NodePath &p_player) {
	animation_player = p_player;
	_setup_animation_player();
	update_configuration_warnings();
}

NodePath AnimationTree::get_animation_player() const {
	return animation_player;
}

void AnimationTree::set_advance_expression_base_node(const NodePath &p_advance_expression_base_node) {
	advance_expression_base_node = p_advance_expression_base_node;
}

NodePath AnimationTree::get_advance_expression_base_node() const {
	return advance_expression_base_node;
}

void AnimationTree::set_audio_max_polyphony(int p_audio_max_polyphony) {
	ERR_FAIL_COND(p_audio_max_polyphony < 0 || p_audio_max_polyphony > 128);
	audio_max_polyphony = p_audio_max_polyphony;
}

int AnimationTree::get_audio_max_polyphony() const {
	return audio_max_polyphony;
}

bool AnimationTree::is_state_invalid() const {
	return !state.valid;
}

String AnimationTree::get_invalid_state_reason() const {
	return state.invalid_reasons;
}

uint64_t AnimationTree::get_last_process_pass() const {
	return process_pass;
}

PackedStringArray AnimationTree::get_configuration_warnings() const {
	PackedStringArray warnings = Node::get_configuration_warnings();

	if (!root.is_valid()) {
		warnings.push_back(RTR("No root AnimationNode for the graph is set."));
	}

	if (!has_node(animation_player)) {
		warnings.push_back(RTR("Path to an AnimationPlayer node containing animations is not set."));
	} else {
		AnimationPlayer *player = Object::cast_to<AnimationPlayer>(get_node(animation_player));

		if (!player) {
			warnings.push_back(RTR("Path set for AnimationPlayer does not lead to an AnimationPlayer node."));
		} else if (!player->has_node(player->get_root())) {
			warnings.push_back(RTR("The AnimationPlayer root node is not a valid node."));
		}
	}

	return warnings;
}

void AnimationTree::set_root_motion_track(const NodePath &p_track) {
	root_motion_track = p_track;
}

NodePath AnimationTree::get_root_motion_track() const {
	return root_motion_track;
}

Vector3 AnimationTree::get_root_motion_position() const {
	return root_motion_position;
}

Quaternion AnimationTree::get_root_motion_rotation() const {
	return root_motion_rotation;
}

Vector3 AnimationTree::get_root_motion_scale() const {
	return root_motion_scale;
}

Vector3 AnimationTree::get_root_motion_position_accumulator() const {
	return root_motion_position_accumulator;
}

Quaternion AnimationTree::get_root_motion_rotation_accumulator() const {
	return root_motion_rotation_accumulator;
}

Vector3 AnimationTree::get_root_motion_scale_accumulator() const {
	return root_motion_scale_accumulator;
}

void AnimationTree::_tree_changed() {
	if (properties_dirty) {
		return;
	}

	call_deferred(SNAME("_update_properties"));
	properties_dirty = true;
}

void AnimationTree::_animation_node_renamed(const ObjectID &p_oid, const String &p_old_name, const String &p_new_name) {
	ERR_FAIL_COND(!property_reference_map.has(p_oid));
	String base_path = property_reference_map[p_oid];
	String old_base = base_path + p_old_name;
	String new_base = base_path + p_new_name;
	for (const PropertyInfo &E : properties) {
		if (E.name.begins_with(old_base)) {
			String new_name = E.name.replace_first(old_base, new_base);
			property_map[new_name] = property_map[E.name];
			property_map.erase(E.name);
		}
	}

	//update tree second
	properties_dirty = true;
	_update_properties();
}

void AnimationTree::_animation_node_removed(const ObjectID &p_oid, const StringName &p_node) {
	ERR_FAIL_COND(!property_reference_map.has(p_oid));
	String base_path = String(property_reference_map[p_oid]) + String(p_node);
	for (const PropertyInfo &E : properties) {
		if (E.name.begins_with(base_path)) {
			property_map.erase(E.name);
		}
	}

	//update tree second
	properties_dirty = true;
	_update_properties();
}

void AnimationTree::_update_properties_for_node(const String &p_base_path, Ref<AnimationNode> node) {
	ERR_FAIL_COND(node.is_null());
	if (!property_parent_map.has(p_base_path)) {
		property_parent_map[p_base_path] = HashMap<StringName, StringName>();
	}
	if (!property_reference_map.has(node->get_instance_id())) {
		property_reference_map[node->get_instance_id()] = p_base_path;
	}

	if (node->get_input_count() && !input_activity_map.has(p_base_path)) {
		Vector<Activity> activity;
		for (int i = 0; i < node->get_input_count(); i++) {
			Activity a;
			a.activity = 0;
			a.last_pass = 0;
			activity.push_back(a);
		}
		input_activity_map[p_base_path] = activity;
		input_activity_map_get[String(p_base_path).substr(0, String(p_base_path).length() - 1)] = &input_activity_map[p_base_path];
	}

	List<PropertyInfo> plist;
	node->get_parameter_list(&plist);
	for (PropertyInfo &pinfo : plist) {
		StringName key = pinfo.name;

		if (!property_map.has(p_base_path + key)) {
			Pair<Variant, bool> param;
			param.first = node->get_parameter_default_value(key);
			param.second = node->is_parameter_read_only(key);
			property_map[p_base_path + key] = param;
		}

		property_parent_map[p_base_path][key] = p_base_path + key;

		pinfo.name = p_base_path + key;
		properties.push_back(pinfo);
	}

	List<AnimationNode::ChildNode> children;
	node->get_child_nodes(&children);

	for (const AnimationNode::ChildNode &E : children) {
		_update_properties_for_node(p_base_path + E.name + "/", E.node);
	}
}

void AnimationTree::_update_properties() {
	if (!properties_dirty) {
		return;
	}

	properties.clear();
	property_reference_map.clear();
	property_parent_map.clear();
	input_activity_map.clear();
	input_activity_map_get.clear();

	if (root.is_valid()) {
		_update_properties_for_node(SceneStringNames::get_singleton()->parameters_base_path, root);
	}

	properties_dirty = false;

	notify_property_list_changed();
}

bool AnimationTree::_set(const StringName &p_name, const Variant &p_value) {
	if (properties_dirty) {
		_update_properties();
	}

	if (property_map.has(p_name)) {
		if (is_inside_tree() && property_map[p_name].second) {
			return false; // Prevent to set property by user.
		}
		property_map[p_name].first = p_value;
		return true;
	}

	return false;
}

bool AnimationTree::_get(const StringName &p_name, Variant &r_ret) const {
	if (properties_dirty) {
		const_cast<AnimationTree *>(this)->_update_properties();
	}

	if (property_map.has(p_name)) {
		r_ret = property_map[p_name].first;
		return true;
	}

	return false;
}

void AnimationTree::_get_property_list(List<PropertyInfo> *p_list) const {
	if (properties_dirty) {
		const_cast<AnimationTree *>(this)->_update_properties();
	}

	for (const PropertyInfo &E : properties) {
		p_list->push_back(E);
	}
}

real_t AnimationTree::get_connection_activity(const StringName &p_path, int p_connection) const {
	if (!input_activity_map_get.has(p_path)) {
		return 0;
	}
	const Vector<Activity> *activity = input_activity_map_get[p_path];

	if (!activity || p_connection < 0 || p_connection >= activity->size()) {
		return 0;
	}

	if ((*activity)[p_connection].last_pass != process_pass) {
		return 0;
	}

	return (*activity)[p_connection].activity;
}

void AnimationTree::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_active", "active"), &AnimationTree::set_active);
	ClassDB::bind_method(D_METHOD("is_active"), &AnimationTree::is_active);

	ClassDB::bind_method(D_METHOD("set_tree_root", "root"), &AnimationTree::set_tree_root);
	ClassDB::bind_method(D_METHOD("get_tree_root"), &AnimationTree::get_tree_root);

	ClassDB::bind_method(D_METHOD("set_process_callback", "mode"), &AnimationTree::set_process_callback);
	ClassDB::bind_method(D_METHOD("get_process_callback"), &AnimationTree::get_process_callback);

	ClassDB::bind_method(D_METHOD("set_animation_player", "root"), &AnimationTree::set_animation_player);
	ClassDB::bind_method(D_METHOD("get_animation_player"), &AnimationTree::get_animation_player);

	ClassDB::bind_method(D_METHOD("set_advance_expression_base_node", "node"), &AnimationTree::set_advance_expression_base_node);
	ClassDB::bind_method(D_METHOD("get_advance_expression_base_node"), &AnimationTree::get_advance_expression_base_node);

	ClassDB::bind_method(D_METHOD("set_root_motion_track", "path"), &AnimationTree::set_root_motion_track);
	ClassDB::bind_method(D_METHOD("get_root_motion_track"), &AnimationTree::get_root_motion_track);

	ClassDB::bind_method(D_METHOD("set_audio_max_polyphony", "max_polyphony"), &AnimationTree::set_audio_max_polyphony);
	ClassDB::bind_method(D_METHOD("get_audio_max_polyphony"), &AnimationTree::get_audio_max_polyphony);

	ClassDB::bind_method(D_METHOD("get_root_motion_position"), &AnimationTree::get_root_motion_position);
	ClassDB::bind_method(D_METHOD("get_root_motion_rotation"), &AnimationTree::get_root_motion_rotation);
	ClassDB::bind_method(D_METHOD("get_root_motion_scale"), &AnimationTree::get_root_motion_scale);
	ClassDB::bind_method(D_METHOD("get_root_motion_position_accumulator"), &AnimationTree::get_root_motion_position_accumulator);
	ClassDB::bind_method(D_METHOD("get_root_motion_rotation_accumulator"), &AnimationTree::get_root_motion_rotation_accumulator);
	ClassDB::bind_method(D_METHOD("get_root_motion_scale_accumulator"), &AnimationTree::get_root_motion_scale_accumulator);

	ClassDB::bind_method(D_METHOD("_update_properties"), &AnimationTree::_update_properties);

	ClassDB::bind_method(D_METHOD("advance", "delta"), &AnimationTree::advance);

	GDVIRTUAL_BIND(_post_process_key_value, "animation", "track", "value", "object", "object_idx");

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "tree_root", PROPERTY_HINT_RESOURCE_TYPE, "AnimationRootNode"), "set_tree_root", "get_tree_root");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "anim_player", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "AnimationPlayer"), "set_animation_player", "get_animation_player");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "advance_expression_base_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node"), "set_advance_expression_base_node", "get_advance_expression_base_node");

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "active"), "set_active", "is_active");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "process_callback", PROPERTY_HINT_ENUM, "Physics,Idle,Manual"), "set_process_callback", "get_process_callback");
	ADD_GROUP("Audio", "audio_");
	ADD_PROPERTY(PropertyInfo(Variant::INT, "audio_max_polyphony", PROPERTY_HINT_RANGE, "1,127,1"), "set_audio_max_polyphony", "get_audio_max_polyphony");
	ADD_GROUP("Root Motion", "root_motion_");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "root_motion_track"), "set_root_motion_track", "get_root_motion_track");

	BIND_ENUM_CONSTANT(ANIMATION_PROCESS_PHYSICS);
	BIND_ENUM_CONSTANT(ANIMATION_PROCESS_IDLE);
	BIND_ENUM_CONSTANT(ANIMATION_PROCESS_MANUAL);

	ADD_SIGNAL(MethodInfo("animation_player_changed"));

	// Signals from AnimationNodes.
	ADD_SIGNAL(MethodInfo("animation_started", PropertyInfo(Variant::STRING_NAME, "anim_name")));
	ADD_SIGNAL(MethodInfo("animation_finished", PropertyInfo(Variant::STRING_NAME, "anim_name")));
}

AnimationTree::AnimationTree() {
}

AnimationTree::~AnimationTree() {
}
