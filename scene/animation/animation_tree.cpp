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
#include "animation_tree.compat.inc"

#include "animation_blend_tree.h"
#include "scene/animation/animation_player.h"

thread_local AnimationNode::ProcessState *AnimationNode::tls_process_state = nullptr;

void AnimationNode::get_parameter_list(LocalVector<PropertyInfo> *r_list) const {
	Array parameters;

	if (GDVIRTUAL_CALL(_get_parameter_list, parameters)) {
		for (int i = 0; i < parameters.size(); i++) {
			Dictionary d = parameters[i];
			ERR_CONTINUE(d.is_empty());
			r_list->push_back(PropertyInfo::from_dict(d));
		}
	}

	r_list->push_back(PropertyInfo(Variant::FLOAT, current_length, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY));
	r_list->push_back(PropertyInfo(Variant::FLOAT, current_position, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY));
	r_list->push_back(PropertyInfo(Variant::FLOAT, current_delta, PROPERTY_HINT_NONE, "", PROPERTY_USAGE_READ_ONLY));
}

Variant AnimationNode::get_parameter_default_value(const StringName &p_parameter) const {
	Variant ret;
	if (p_parameter == current_length || p_parameter == current_position || p_parameter == current_delta) {
		return 0.0;
	}
	GDVIRTUAL_CALL(_get_parameter_default_value, p_parameter, ret);
	return ret;
}

bool AnimationNode::is_parameter_read_only(const StringName &p_parameter) const {
	bool ret = false;
	if (GDVIRTUAL_CALL(_is_parameter_read_only, p_parameter, ret) && ret) {
		return true;
	}

	if (p_parameter == current_length || p_parameter == current_position || p_parameter == current_delta) {
		return true;
	}

	return false;
}

void AnimationNode::set_parameter_ex(const StringName &p_name, const Variant &p_value) {
	ERR_FAIL_NULL(tls_process_state);
	AnimationNodeInstance &instance = tls_process_state->tree->get_node_instance(get_instance_id());
	instance.set_parameter(p_name, p_value, tls_process_state->is_testing);
}

Variant AnimationNode::get_parameter_ex(const StringName &p_name) const {
	ERR_FAIL_NULL_V(tls_process_state, Variant());
	AnimationNodeInstance &instance = tls_process_state->tree->get_node_instance(get_instance_id());
	return instance.get_parameter(p_name);
}

void AnimationNode::set_node_time_info(AnimationNodeInstance &instance, ProcessState &p_process_state, const NodeTimeInfo &p_node_time_info) {
	if (p_process_state.is_testing) {
		return;
	}
	instance.set_parameter(current_length, p_node_time_info.length, p_process_state.is_testing);
	instance.set_parameter(current_position, p_node_time_info.position, p_process_state.is_testing);
	instance.set_parameter(current_delta, p_node_time_info.delta, p_process_state.is_testing);
}

AnimationNode::NodeTimeInfo AnimationNode::get_node_time_info(AnimationNodeInstance &instance, ProcessState &p_process_state) const {
	NodeTimeInfo nti;
	nti.length = instance.get_parameter(current_length);
	nti.position = instance.get_parameter(current_position);
	nti.delta = instance.get_parameter(current_delta);
	return nti;
}

void AnimationNode::get_child_nodes(LocalVector<ChildNode> *r_child_nodes) {
	Dictionary cn;
	if (GDVIRTUAL_CALL(_get_child_nodes, cn)) {
		for (const KeyValue<Variant, Variant> &kv : cn) {
			ChildNode child;
			child.name = kv.key;
			child.node = kv.value;
			r_child_nodes->push_back(child);
		}
	}
}

void AnimationNode::blend_animation(ProcessState &p_process_state, AnimationNodeInstance &p_instance, const StringName &p_animation, AnimationMixer::PlaybackInfo &p_playback_info) {
	p_playback_info.track_weights = &p_instance.track_weights;
	p_process_state.tree->make_animation_instance(p_animation, p_playback_info);
}

AnimationNode::NodeTimeInfo AnimationNode::_pre_process(ProcessState &p_process_state, AnimationNodeInstance &p_instance, const AnimationMixer::PlaybackInfo &p_playback_info, bool p_test_only) {
	tls_process_state = &p_process_state;
	NodeTimeInfo nti = process(p_process_state, p_instance, p_playback_info, p_test_only);
	tls_process_state = nullptr;
	return nti;
}

void AnimationNode::make_invalid(ProcessState &p_process_state, const String &p_reason) {
	p_process_state.valid = false;
	if (!p_process_state.invalid_reasons.is_empty()) {
		p_process_state.invalid_reasons += "\n";
	}
	p_process_state.invalid_reasons += String::utf8("•  ") + p_reason;
}

AnimationNode::NodeTimeInfo AnimationNode::blend_input(ProcessState &p_process_state, AnimationNodeInstance &p_instance, int p_input, const AnimationMixer::PlaybackInfo &p_playback_info, FilterAction p_filter, bool p_sync, bool p_test_only) {
	ERR_FAIL_INDEX_V(p_input, (int64_t)inputs.size(), NodeTimeInfo());

	AnimationNodeBlendTree *blend_tree = Object::cast_to<AnimationNodeBlendTree>(p_instance.parent);
	ERR_FAIL_NULL_V(blend_tree, NodeTimeInfo());

	// Update connections.
	const StringName &current_name = blend_tree->get_node_name(*this);
	p_instance.connections = blend_tree->get_node_connection_array(current_name);

	// Get node which is connected input port.
	const StringName &node_name = p_instance.connections[p_input];
	const Ref<AnimationNode> &node = blend_tree->get_node_or_null(node_name);

	if (node.is_null()) {
		make_invalid(p_process_state, vformat(RTR("Nothing connected to input '%s' of node '%s'."), get_input_name(p_input), current_name));
		return NodeTimeInfo();
	}

	real_t activity = 0.0;
	LocalVector<AnimationTree::Activity> *activity_ptr = p_process_state.tree->input_activity_map.getptr(p_instance.base_path);
	AnimationNodeInstance &node_instance = p_process_state.tree->get_node_instance(node->get_instance_id());
	NodeTimeInfo nti = _blend_node(p_process_state, p_instance, node_instance, node, node_name, nullptr, p_playback_info, p_filter, p_sync, p_test_only, &activity);

	if (activity_ptr && p_input < (int64_t)activity_ptr->size()) {
		(*activity_ptr)[p_input].last_pass = p_process_state.last_pass;
		(*activity_ptr)[p_input].activity = activity;
	}
	return nti;
}

AnimationNode::NodeTimeInfo AnimationNode::blend_node(ProcessState &p_process_state, AnimationNodeInstance &p_instance, const Ref<AnimationNode> &p_node, const StringName &p_subpath, const AnimationMixer::PlaybackInfo &p_playback_info, FilterAction p_filter, bool p_sync, bool p_test_only) {
	ERR_FAIL_COND_V(p_node.is_null(), NodeTimeInfo());
	p_instance.connections.clear();
	AnimationNodeInstance &node_instance = p_process_state.tree->get_node_instance(p_node->get_instance_id());
	return _blend_node(p_process_state, p_instance, node_instance, p_node, p_subpath, this, p_playback_info, p_filter, p_sync, p_test_only, nullptr);
}

AnimationNode::NodeTimeInfo AnimationNode::_blend_node(ProcessState &p_process_state, AnimationNodeInstance &p_instance, AnimationNodeInstance &p_node_instance, const Ref<AnimationNode> &p_node, const StringName &p_subpath, AnimationNode *p_new_parent, AnimationMixer::PlaybackInfo p_playback_info, FilterAction p_filter, bool p_sync, bool p_test_only, real_t *r_activity) {
	int blend_count = p_instance.track_weights.size();

	if ((int64_t)p_node_instance.track_weights.size() != blend_count) {
		p_node_instance.track_weights.resize(blend_count);
	}

	real_t *blendw = p_node_instance.track_weights.ptr();
	const real_t *blendr = p_instance.track_weights.ptr();

	bool any_valid = false;

	if (has_filter() && is_filter_enabled() && p_filter != FILTER_IGNORE) {
		for (int i = 0; i < blend_count; i++) {
			blendw[i] = 0.0; // All to zero by default.
		}

		for (const KeyValue<NodePath, bool> &E : filter) {
			const AHashMap<NodePath, int> &map = *p_process_state.track_map;
			const int *p = map.getptr(E.key);
			if (!p) {
				continue;
			}
			const int idx = *p;
			blendw[idx] = 1.0; // Filtered goes to one.
		}

		switch (p_filter) {
			case FILTER_IGNORE:
				break; // Will not happen anyway.
			case FILTER_PASS: {
				// Values filtered pass, the rest don't.
				for (int i = 0; i < blend_count; i++) {
					if (blendw[i] == 0) { // Not filtered, does not pass.
						continue;
					}

					blendw[i] = blendr[i] * p_playback_info.weight;
					if (!Math::is_zero_approx(blendw[i])) {
						any_valid = true;
					}
				}

			} break;
			case FILTER_STOP: {
				// Values filtered don't pass, the rest are blended.

				for (int i = 0; i < blend_count; i++) {
					if (blendw[i] > 0) { // Filtered, does not pass.
						continue;
					}

					blendw[i] = blendr[i] * p_playback_info.weight;
					if (!Math::is_zero_approx(blendw[i])) {
						any_valid = true;
					}
				}

			} break;
			case FILTER_BLEND: {
				// Filtered values are blended, the rest are passed without blending.

				for (int i = 0; i < blend_count; i++) {
					if (blendw[i] == 1.0) {
						blendw[i] = blendr[i] * p_playback_info.weight; // Filtered, blend.
					} else {
						blendw[i] = blendr[i]; // Not filtered, do not blend.
					}

					if (!Math::is_zero_approx(blendw[i])) {
						any_valid = true;
					}
				}

			} break;
		}
	} else {
		for (int i = 0; i < blend_count; i++) {
			// Regular blend.
			blendw[i] = blendr[i] * p_playback_info.weight;
			if (!Math::is_zero_approx(blendw[i])) {
				any_valid = true;
			}
		}
	}

	if (r_activity) {
		*r_activity = 0;
		for (int i = 0; i < blend_count; i++) {
			*r_activity = MAX(*r_activity, Math::abs(blendw[i]));
		}
	}

	StringName *new_path;
	AnimationNode *new_parent;

	AHashMap<StringName, StringName> *childmap = nullptr;
	if (p_new_parent) {
		new_parent = p_new_parent;
		childmap = &p_instance.child_base_cache;
	} else {
		ERR_FAIL_NULL_V(p_instance.parent, NodeTimeInfo());
		AnimationNodeInstance &parent_instance = p_process_state.tree->get_node_instance(p_instance.parent->get_instance_id());
		childmap = &parent_instance.child_base_cache;
		new_parent = p_instance.parent;
	}

	ERR_FAIL_NULL_V(childmap, NodeTimeInfo());
	if (StringName *found = childmap->getptr(p_subpath)) {
		new_path = found;
	} else {
		ERR_FAIL_V(NodeTimeInfo());
	}

	// This process, which depends on p_sync is needed to process sync correctly in the case of
	// that a synced AnimationNodeSync exists under the un-synced AnimationNodeSync.
	p_node_instance.base_path = *new_path;
	p_node_instance.parent = new_parent;

	if (!p_playback_info.seeked && !p_sync && !any_valid) {
		p_playback_info.delta = 0.0;
	}
	return p_node->_pre_process(p_process_state, p_node_instance, p_playback_info, p_test_only);
}

String AnimationNode::get_caption() const {
	String ret = "Node";
	GDVIRTUAL_CALL(_get_caption, ret);
	return ret;
}

bool AnimationNode::add_input(const String &p_name) {
	// Root nodes can't add inputs.
	ERR_FAIL_COND_V(Object::cast_to<AnimationRootNode>(this) != nullptr, false);
	Input input;
	ERR_FAIL_COND_V(p_name.contains_char('.') || p_name.contains_char('/'), false);
	input.name = p_name;
	inputs.push_back(input);
	emit_changed();
	return true;
}

void AnimationNode::remove_input(int p_index) {
	ERR_FAIL_INDEX(p_index, (int64_t)inputs.size());
	inputs.remove_at(p_index);
	emit_changed();
}

bool AnimationNode::set_input_name(int p_input, const String &p_name) {
	ERR_FAIL_INDEX_V(p_input, (int64_t)inputs.size(), false);
	ERR_FAIL_COND_V(p_name.contains_char('.') || p_name.contains_char('/'), false);
	inputs[p_input].name = p_name;
	emit_changed();
	return true;
}

String AnimationNode::get_input_name(int p_input) const {
	ERR_FAIL_INDEX_V(p_input, (int64_t)inputs.size(), String());
	return inputs[p_input].name;
}

int AnimationNode::get_input_count() const {
	return inputs.size();
}

int AnimationNode::find_input(const String &p_name) const {
	int idx = -1;
	for (int i = 0; i < (int64_t)inputs.size(); i++) {
		if (inputs[i].name == p_name) {
			idx = i;
			break;
		}
	}
	return idx;
}

AnimationNode::NodeTimeInfo AnimationNode::process(ProcessState &p_process_state, AnimationNodeInstance &p_instance, const AnimationMixer::PlaybackInfo &p_playback_info, bool p_test_only) {
	p_process_state.is_testing = p_test_only;

	AnimationMixer::PlaybackInfo pi = p_playback_info;
	if (p_playback_info.seeked) {
		if (p_playback_info.is_external_seeking) {
			pi.delta = get_node_time_info(p_instance, p_process_state).position - p_playback_info.time;
		}
	} else {
		pi.time = get_node_time_info(p_instance, p_process_state).position + p_playback_info.delta;
	}

	NodeTimeInfo nti = _process(p_process_state, p_instance, pi, p_test_only);

	if (!p_test_only) {
		set_node_time_info(p_instance, p_process_state, nti);
	}

	return nti;
}

AnimationNode::NodeTimeInfo AnimationNode::_process(ProcessState &p_process_state, AnimationNodeInstance &p_instance, const AnimationMixer::PlaybackInfo &p_playback_info, bool p_test_only) {
	double r_ret = 0.0;
	GDVIRTUAL_CALL(_process, p_playback_info.time, p_playback_info.seeked, p_playback_info.is_external_seeking, p_test_only, r_ret);
	NodeTimeInfo nti;
	nti.delta = r_ret;
	return nti;
}

void AnimationNode::set_filter_path(const NodePath &p_path, bool p_enable) {
	if (p_enable) {
		(void)p_path.hash(); // Make sure the cache is valid.
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

void AnimationNode::set_deletable(bool p_closable) {
	closable = p_closable;
}

bool AnimationNode::is_deletable() const {
	return closable;
}

ObjectID AnimationNode::get_processing_animation_tree_instance_id() const {
	ERR_FAIL_NULL_V(tls_process_state, ObjectID());
	return tls_process_state->tree->get_instance_id();
}

bool AnimationNode::is_process_testing() const {
	ERR_FAIL_NULL_V(tls_process_state, false);
	return tls_process_state->is_testing;
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
		paths.push_back(String(E.key)); // Use strings, so sorting is possible.
	}
	paths.sort(); // Done so every time the scene is saved, it does not change.

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
		if (ret.is_null()) {
			break;
		}
	}
	return ret;
}

void AnimationNode::blend_animation_ex(const StringName &p_animation, double p_time, double p_delta, bool p_seeked, bool p_is_external_seeking, real_t p_blend, Animation::LoopedFlag p_looped_flag) {
	ERR_FAIL_NULL(tls_process_state);
	AnimationMixer::PlaybackInfo info;
	info.time = p_time;
	info.delta = p_delta;
	info.seeked = p_seeked;
	info.is_external_seeking = p_is_external_seeking;
	info.weight = p_blend;
	info.looped_flag = p_looped_flag;
	AnimationNodeInstance &instance = tls_process_state->tree->get_node_instance(get_instance_id());
	blend_animation(*tls_process_state, instance, p_animation, info);
}

double AnimationNode::blend_node_ex(const StringName &p_sub_path, const Ref<AnimationNode> &p_node, double p_time, bool p_seek, bool p_is_external_seeking, real_t p_blend, FilterAction p_filter, bool p_sync, bool p_test_only) {
	AnimationMixer::PlaybackInfo info;
	info.time = p_time;
	info.seeked = p_seek;
	info.is_external_seeking = p_is_external_seeking;
	info.weight = p_blend;
	AnimationNodeInstance &instance = tls_process_state->tree->get_node_instance(get_instance_id());
	NodeTimeInfo nti = blend_node(*tls_process_state, instance, p_node, p_sub_path, info, p_filter, p_sync, p_test_only);
	return nti.length - nti.position;
}

double AnimationNode::blend_input_ex(int p_input, double p_time, bool p_seek, bool p_is_external_seeking, real_t p_blend, FilterAction p_filter, bool p_sync, bool p_test_only) {
	AnimationMixer::PlaybackInfo info;
	info.time = p_time;
	info.seeked = p_seek;
	info.is_external_seeking = p_is_external_seeking;
	info.weight = p_blend;
	AnimationNodeInstance &instance = tls_process_state->tree->get_node_instance(get_instance_id());
	NodeTimeInfo nti = blend_input(*tls_process_state, instance, p_input, info, p_filter, p_sync, p_test_only);
	return nti.length - nti.position;
}

#ifdef TOOLS_ENABLED
void AnimationNode::get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const {
	const String pf = p_function;
	if (p_idx == 0) {
		if (pf == "find_input") {
			for (const AnimationNode::Input &E : inputs) {
				r_options->push_back(E.name.quote());
			}
		} else if (pf == "get_parameter" || pf == "set_parameter") {
			bool is_setter = pf == "set_parameter";
			LocalVector<PropertyInfo> parameters;
			get_parameter_list(&parameters);
			for (const PropertyInfo &E : parameters) {
				if (is_setter && is_parameter_read_only(E.name)) {
					continue;
				}
				r_options->push_back(E.name.quote());
			}
		} else if (pf == "set_filter_path" || pf == "is_path_filtered") {
			for (const KeyValue<NodePath, bool> &E : filter) {
				r_options->push_back(String(E.key).quote());
			}
		}
	}
	Resource::get_argument_options(p_function, p_idx, r_options);
}
#endif

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

	ClassDB::bind_method(D_METHOD("get_processing_animation_tree_instance_id"), &AnimationNode::get_processing_animation_tree_instance_id);

	ClassDB::bind_method(D_METHOD("is_process_testing"), &AnimationNode::is_process_testing);

	ClassDB::bind_method(D_METHOD("_set_filters", "filters"), &AnimationNode::_set_filters);
	ClassDB::bind_method(D_METHOD("_get_filters"), &AnimationNode::_get_filters);

	ClassDB::bind_method(D_METHOD("blend_animation", "animation", "time", "delta", "seeked", "is_external_seeking", "blend", "looped_flag"), &AnimationNode::blend_animation_ex, DEFVAL(Animation::LOOPED_FLAG_NONE));
	ClassDB::bind_method(D_METHOD("blend_node", "name", "node", "time", "seek", "is_external_seeking", "blend", "filter", "sync", "test_only"), &AnimationNode::blend_node_ex, DEFVAL(FILTER_IGNORE), DEFVAL(true), DEFVAL(false));
	ClassDB::bind_method(D_METHOD("blend_input", "input_index", "time", "seek", "is_external_seeking", "blend", "filter", "sync", "test_only"), &AnimationNode::blend_input_ex, DEFVAL(FILTER_IGNORE), DEFVAL(true), DEFVAL(false));

	ClassDB::bind_method(D_METHOD("set_parameter", "name", "value"), &AnimationNode::set_parameter_ex);
	ClassDB::bind_method(D_METHOD("get_parameter", "name"), &AnimationNode::get_parameter_ex);

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

void AnimationTree::set_root_animation_node(const Ref<AnimationRootNode> &p_animation_node) {
	if (root_animation_node.is_valid()) {
		root_animation_node->disconnect(SNAME("tree_changed"), callable_mp(this, &AnimationTree::_tree_changed));
		root_animation_node->disconnect(SNAME("animation_node_renamed"), callable_mp(this, &AnimationTree::_animation_node_renamed));
		root_animation_node->disconnect(SNAME("animation_node_removed"), callable_mp(this, &AnimationTree::_animation_node_removed));
	}

	root_animation_node = p_animation_node;

	if (root_animation_node.is_valid()) {
		root_animation_node->connect(SNAME("tree_changed"), callable_mp(this, &AnimationTree::_tree_changed));
		root_animation_node->connect(SNAME("animation_node_renamed"), callable_mp(this, &AnimationTree::_animation_node_renamed));
		root_animation_node->connect(SNAME("animation_node_removed"), callable_mp(this, &AnimationTree::_animation_node_removed));
	}

	properties_dirty = true;

	update_configuration_warnings();
}

Ref<AnimationRootNode> AnimationTree::get_root_animation_node() const {
	return root_animation_node;
}

bool AnimationTree::_blend_pre_process(double p_delta, int p_track_count, const AHashMap<NodePath, int> &p_track_map) {
	_update_properties(); // If properties need updating, update them.

	if (root_animation_node.is_null()) {
		return false;
	}
	AnimationNodeInstance &instance = get_node_instance(root_animation_node->get_instance_id());

	{ // Setup.
		process_pass++;

		// Init process state.
		process_state = AnimationNode::ProcessState();
		process_state.tree = this;
		process_state.valid = true;
		process_state.invalid_reasons = "";
		process_state.last_pass = process_pass;
		process_state.track_map = &p_track_map;
		process_state.is_testing = false;

		// Init node state for root AnimationNode.
		instance.track_weights.resize(p_track_count);
		real_t *src_blendsw = instance.track_weights.ptr();
		for (int i = 0; i < p_track_count; i++) {
			src_blendsw[i] = 1.0; // By default all go to 1 for the root input.
		}
		instance.base_path = SNAME(Animation::PARAMETERS_BASE_PATH.ascii().get_data());
		instance.parent = nullptr;
	}

	// Process.
	{
		PlaybackInfo pi;
		pi.delta = p_delta;

		if (started) {
			started = false;
			// If started, seek.
			pi.seeked = true;
		} else {
			pi.seeked = false;
		}

		root_animation_node->_pre_process(process_state, instance, pi, false);
	}

	if (!process_state.valid) {
		return false; // State is not valid, abort process.
	}

	return true;
}

void AnimationTree::_set_active(bool p_active) {
	_set_process(p_active);
	started = p_active;
}

void AnimationTree::set_advance_expression_base_node(const NodePath &p_path) {
	advance_expression_base_node = p_path;
}

NodePath AnimationTree::get_advance_expression_base_node() const {
	return advance_expression_base_node;
}

bool AnimationTree::is_state_invalid() const {
	return !process_state.valid;
}

String AnimationTree::get_invalid_state_reason() const {
	return process_state.invalid_reasons;
}

uint64_t AnimationTree::get_last_process_pass() const {
	return process_pass;
}

PackedStringArray AnimationTree::get_configuration_warnings() const {
	PackedStringArray warnings = AnimationMixer::get_configuration_warnings();
	if (root_animation_node.is_null()) {
		warnings.push_back(RTR("No root AnimationNode for the graph is set."));
	}
	return warnings;
}

void AnimationTree::_tree_changed() {
	if (properties_dirty) {
		return;
	}

	callable_mp(this, &AnimationTree::_update_properties).call_deferred();
	properties_dirty = true;
}

void AnimationTree::_animation_node_renamed(const ObjectID &p_oid, const String &p_old_name, const String &p_new_name) {
	const AnimationNodeInstance *instance = instance_map.getptr(p_oid);
	ERR_FAIL_COND(!instance);
	String base_path = instance->base_path;
	String old_base = base_path + p_old_name;
	String new_base = base_path + p_new_name;
	for (const PropertyInfo &E : properties) {
		if (E.name.begins_with(old_base)) {
			StringName name = E.name;
			StringName new_name = E.name.replace_first(old_base, new_base);
			property_map[new_name] = property_map[name];
			property_map.erase(name);
		}
	}

	// Update tree second.
	properties_dirty = true;
	_update_properties();
}

void AnimationTree::_animation_node_removed(const ObjectID &p_oid, const StringName &p_node) {
	const AnimationNodeInstance *instance = instance_map.getptr(p_oid);
	ERR_FAIL_COND(!instance);
	String base_path = String(instance->base_path) + String(p_node);
	for (const PropertyInfo &E : properties) {
		if (E.name.begins_with(base_path)) {
			property_map.erase(E.name);
		}
	}

	// Update tree second.
	properties_dirty = true;
	_update_properties();
}

void AnimationTree::_update_properties_for_node(const StringName &p_base_path, const Ref<AnimationNode> &p_node) const {
	ERR_FAIL_COND(p_node.is_null());
	const String base_path_str = String(p_base_path);

	AnimationNodeInstance &instance = instance_map[p_node->get_instance_id()];
	instance.base_path = p_base_path;

	if (p_node->get_input_count() && !input_activity_map.has(p_base_path)) {
		LocalVector<Activity> activity;
		for (int i = 0; i < p_node->get_input_count(); i++) {
			Activity a;
			a.activity = 0;
			a.last_pass = 0;
			activity.push_back(a);
		}
		input_activity_map[p_base_path] = activity;
		input_activity_map_get[base_path_str.substr(0, base_path_str.length() - 1)] = input_activity_map.get_index(p_base_path);
	}

	LocalVector<PropertyInfo> plist;
	p_node->get_parameter_list(&plist);
	for (PropertyInfo &pinfo : plist) {
		StringName pname = pinfo.name;
		StringName key = base_path_str + pname;

		bool property_was_added = false;
		Pair<Variant, bool> &param = property_map.get_value_ref_or_add_default(key, property_was_added);
		if (property_was_added) {
			param.first = p_node->get_parameter_default_value(pname);
			param.second = p_node->is_parameter_read_only(pname);
		}

		instance.property_parent_map[pname] = key;

		pinfo.name = key;
		properties.push_back(pinfo);
	}

	LocalVector<AnimationNode::ChildNode> children;
	p_node->get_child_nodes(&children);

	for (const AnimationNode::ChildNode &E : children) {
		const String child_base = base_path_str + E.name + "/";
		instance.child_base_cache[E.name] = child_base;
	}

	for (const AnimationNode::ChildNode &E : children) {
		_update_properties_for_node(base_path_str + E.name + "/", E.node);
	}
}

void AnimationTree::_update_properties() const {
	if (!properties_dirty) {
		return;
	}

	properties.clear();
	instance_map.clear();
	input_activity_map.clear();
	input_activity_map_get.clear();

	if (root_animation_node.is_valid()) {
		_update_properties_for_node(Animation::PARAMETERS_BASE_PATH, root_animation_node);

		// Now that the properties are stable, we can update each instance.
		for (const KeyValue<ObjectID, AnimationNodeInstance> &E : instance_map) {
			AnimationNodeInstance *instance = instance_map.getptr(E.key);
			ERR_FAIL_NULL(instance);

			const AnimationNode *node = Object::cast_to<AnimationNode>(ObjectDB::get_instance(E.key));
			ERR_FAIL_NULL(node);

			LocalVector<PropertyInfo> plist;
			instance->parameter_ptrs_by_slot.reserve(plist.size());
			node->get_parameter_list(&plist);
			for (const PropertyInfo &pinfo : plist) {
				StringName pname = pinfo.name;

				const StringName *k = instance->property_parent_map.getptr(pname);
				ERR_CONTINUE(!k);
				const StringName &key = *k;

				Pair<Variant, bool> *p = property_map.getptr(key);
				ERR_CONTINUE(!p);
				Pair<Variant, bool> *pair = p;

				instance->property_ptrs[pname] = pair;
				const unsigned slot_index = instance->parameter_ptrs_by_slot.size();
				instance->parameter_ptrs_by_slot.push_back(pair);

				if (pname == node->current_length) {
					instance->slot_current_length = slot_index;
				} else if (pname == node->current_position) {
					instance->slot_current_position = slot_index;
				} else if (pname == node->current_delta) {
					instance->slot_current_delta = slot_index;
				}
			}
		}
	}

	properties_dirty = false;

	const_cast<AnimationTree *>(this)->notify_property_list_changed();
}

void AnimationTree::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			_setup_animation_player();
			if (active) {
				_set_process(true);
			}
		} break;
	}
}

void AnimationTree::set_animation_player(const NodePath &p_path) {
	animation_player = p_path;
	if (p_path.is_empty()) {
		set_root_node(SceneStringName(path_pp));
		while (animation_libraries.size()) {
			remove_animation_library(animation_libraries[0].name);
		}
	}
	emit_signal(SNAME("animation_player_changed")); // Needs to unpin AnimationPlayerEditor.
	_setup_animation_player();
	notify_property_list_changed();
}

NodePath AnimationTree::get_animation_player() const {
	return animation_player;
}

void AnimationTree::_setup_animation_player() {
	if (!is_inside_tree()) {
		return;
	}

	cache_valid = false;

	if (animation_player.is_empty()) {
		clear_caches();
		return;
	}

	// Using AnimationPlayer here is for compatibility. Changing to AnimationMixer needs extra work like error handling.
	AnimationPlayer *player = Object::cast_to<AnimationPlayer>(get_node_or_null(animation_player));
	if (player) {
		if (!player->is_connected(SNAME("caches_cleared"), callable_mp(this, &AnimationTree::_setup_animation_player))) {
			player->connect(SNAME("caches_cleared"), callable_mp(this, &AnimationTree::_setup_animation_player), CONNECT_DEFERRED);
		}
		if (!player->is_connected(SNAME("animation_list_changed"), callable_mp(this, &AnimationTree::_setup_animation_player))) {
			player->connect(SNAME("animation_list_changed"), callable_mp(this, &AnimationTree::_setup_animation_player), CONNECT_DEFERRED);
		}
		Node *root = player->get_node_or_null(player->get_root_node());
		if (root) {
			set_root_node(get_path_to(root, true));
		}
		while (animation_libraries.size()) {
			remove_animation_library(animation_libraries[0].name);
		}
		LocalVector<StringName> list;
		player->get_animation_library_list(&list);
		for (const StringName &E : list) {
			Ref<AnimationLibrary> lib = player->get_animation_library(E);
			if (lib.is_valid()) {
				add_animation_library(E, lib);
			}
		}
	}

	clear_caches();
}

// `libraries` is a dynamic property, so we can't use `_validate_property` to change it.
uint32_t AnimationTree::_get_libraries_property_usage() const {
	if (!animation_player.is_empty()) {
		return PROPERTY_USAGE_EDITOR | PROPERTY_USAGE_READ_ONLY;
	}
	return PROPERTY_USAGE_DEFAULT;
}

void AnimationTree::_validate_property(PropertyInfo &p_property) const {
	if (!Engine::get_singleton()->is_editor_hint()) {
		return;
	}

	if (!animation_player.is_empty()) {
		if (p_property.name == "root_node") {
			p_property.usage |= PROPERTY_USAGE_READ_ONLY;
		}
	}
}

bool AnimationTree::_set(const StringName &p_name, const Variant &p_value) {
#ifndef DISABLE_DEPRECATED
	String name = p_name;
	if (name == "process_callback") {
		set_callback_mode_process(static_cast<AnimationCallbackModeProcess>((int)p_value));
		return true;
	}
#endif // DISABLE_DEPRECATED
	if (properties_dirty) {
		_update_properties();
	}

	if (property_map.has(p_name)) {
		if (is_inside_tree() && property_map[p_name].second) {
			return false; // Prevent to set property by user.
		}
		Pair<Variant, bool> &prop = property_map[p_name];
		Variant value = p_value;
		if (Animation::validate_type_match(prop.first, value)) {
			prop.first = value;
		}
		return true;
	}

	return false;
}

bool AnimationTree::_get(const StringName &p_name, Variant &r_ret) const {
#ifndef DISABLE_DEPRECATED
	if (p_name == "process_callback") {
		r_ret = get_callback_mode_process();
		return true;
	}
#endif // DISABLE_DEPRECATED
	if (properties_dirty) {
		_update_properties();
	}

	if (const Pair<Variant, bool> *p = property_map.getptr(p_name)) {
		r_ret = p->first;
		return true;
	}

	return false;
}

void AnimationTree::_get_property_list(List<PropertyInfo> *p_list) const {
	if (properties_dirty) {
		_update_properties();
	}

	for (const PropertyInfo &E : properties) {
		p_list->push_back(E);
	}
}

real_t AnimationTree::get_connection_activity(const StringName &p_path, int p_connection) const {
	if (!input_activity_map_get.has(p_path)) {
		return 0;
	}

	int index = input_activity_map_get[p_path];
	const LocalVector<Activity> &activity = input_activity_map.get_by_index(index).value;

	if (p_connection < 0 || p_connection >= (int64_t)activity.size() || activity[p_connection].last_pass != process_pass) {
		return 0;
	}

	return activity[p_connection].activity;
}

#ifdef TOOLS_ENABLED
String AnimationTree::get_editor_error_message() const {
	if (!is_active()) {
		return TTR("The AnimationTree is inactive.\nActivate it in the inspector to enable playback; check node warnings if activation fails.");
	} else if (!is_enabled()) {
		return TTR("The AnimationTree node (or one of its parents) has its process mode set to Disabled.\nChange the process mode in the inspector to allow playback.");
	} else if (is_state_invalid()) {
		return get_invalid_state_reason();
	}

	return "";
}
#endif

void AnimationTree::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_tree_root", "animation_node"), &AnimationTree::set_root_animation_node);
	ClassDB::bind_method(D_METHOD("get_tree_root"), &AnimationTree::get_root_animation_node);

	ClassDB::bind_method(D_METHOD("set_advance_expression_base_node", "path"), &AnimationTree::set_advance_expression_base_node);
	ClassDB::bind_method(D_METHOD("get_advance_expression_base_node"), &AnimationTree::get_advance_expression_base_node);

	ClassDB::bind_method(D_METHOD("set_animation_player", "path"), &AnimationTree::set_animation_player);
	ClassDB::bind_method(D_METHOD("get_animation_player"), &AnimationTree::get_animation_player);

	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "tree_root", PROPERTY_HINT_RESOURCE_TYPE, "AnimationRootNode"), "set_tree_root", "get_tree_root");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "advance_expression_base_node", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "Node"), "set_advance_expression_base_node", "get_advance_expression_base_node");
	ADD_PROPERTY(PropertyInfo(Variant::NODE_PATH, "anim_player", PROPERTY_HINT_NODE_PATH_VALID_TYPES, "AnimationPlayer"), "set_animation_player", "get_animation_player");

	ADD_SIGNAL(MethodInfo(SNAME("animation_player_changed")));
}

AnimationTree::AnimationTree() {
	deterministic = true;
	callback_mode_discrete = ANIMATION_CALLBACK_MODE_DISCRETE_FORCE_CONTINUOUS;
}

AnimationTree::~AnimationTree() {
}
