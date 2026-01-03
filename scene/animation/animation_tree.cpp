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
thread_local AnimationNodeInstance *AnimationNode::current_instance = nullptr;

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
	ERR_FAIL_NULL(current_instance);
	current_instance->set_parameter(p_name, p_value, tls_process_state->is_testing);
}

Variant AnimationNode::get_parameter_ex(const StringName &p_name) const {
	ERR_FAIL_NULL_V(tls_process_state, Variant());
	ERR_FAIL_NULL_V(current_instance, Variant());
	return current_instance->get_parameter(p_name);
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
	ERR_FAIL_NULL_V(tls_process_state, NodeTimeInfo()); // Should not ever happen.
	ERR_FAIL_COND_V_MSG(tls_process_state != &p_process_state, NodeTimeInfo(), "AnimationNodes can only be processed from within their own AnimationTree.");

	AnimationNodeInstance *prev_instance = current_instance;

	current_instance = &p_instance;
	NodeTimeInfo nti = process(p_process_state, p_instance, p_playback_info, p_test_only);
	current_instance = prev_instance;

	return nti;
}

void AnimationNode::add_validation_error(const AnimationTree *p_tree, const StringName &p_path, const String &p_error, int p_input_index) const {
	p_tree->_add_validation_error(p_path, p_error, p_input_index);
}

void AnimationNode::make_invalid(ProcessState &p_process_state, AnimationNodeInstance &p_instance, const String &p_reason) {
	p_process_state.valid = false;

	InvalidInstance &invalid_instance = p_process_state.invalid_instances[p_instance.path];
	invalid_instance.errors.push_back(p_reason);
}

AnimationNode::NodeTimeInfo AnimationNode::blend_input(ProcessState &p_process_state, AnimationNodeInstance &p_instance, int p_input, const AnimationMixer::PlaybackInfo &p_playback_info, FilterAction p_filter, bool p_sync, bool p_test_only) {
	ERR_FAIL_INDEX_V(p_input, (int64_t)inputs.size(), NodeTimeInfo());

	AnimationNodeInstance *node_instance = nullptr;
	if (likely(p_instance.connection_instances.size() > 0)) {
		node_instance = p_instance.connection_instances[p_input];
	}
	// Should not ever happen due to validation earlier.
	ERR_FAIL_NULL_V(node_instance, NodeTimeInfo());

	real_t activity = 0.0;

	NodeTimeInfo nti = _blend_node(p_process_state, p_instance, *node_instance, p_playback_info, p_filter, p_sync, p_test_only, &activity);

#ifdef ENABLE_ACTIVITY_TRACKING
	LocalVector<AnimationNodeInstance::Activity> &input_activity = p_instance.input_activity;
	input_activity[p_input].last_pass = p_process_state.last_pass;
	input_activity[p_input].activity = activity;
#endif

	return nti;
}

AnimationNode::NodeTimeInfo AnimationNode::blend_node(ProcessState &p_process_state, AnimationNodeInstance &p_instance, AnimationNodeInstance *p_other, const AnimationMixer::PlaybackInfo &p_playback_info, FilterAction p_filter, bool p_sync, bool p_test_only) {
	ERR_FAIL_NULL_V(p_other, NodeTimeInfo());
	return _blend_node(p_process_state, p_instance, *p_other, p_playback_info, p_filter, p_sync, p_test_only, nullptr);
}

AnimationNode::NodeTimeInfo AnimationNode::_blend_node(ProcessState &p_process_state, AnimationNodeInstance &p_instance, AnimationNodeInstance &p_other, AnimationMixer::PlaybackInfo p_playback_info, FilterAction p_filter, bool p_sync, bool p_test_only, real_t *r_activity) {
	int blend_count = p_instance.track_weights.size();

	if ((int64_t)p_other.track_weights.size() != blend_count) {
		p_other.track_weights.resize(blend_count);
	}

	real_t *blendw = p_other.track_weights.ptr();
	const real_t *blendr = p_instance.track_weights.ptr();

	bool any_valid = false;

	if (has_filter() && is_filter_enabled() && p_filter != FILTER_IGNORE) {
		_update_filter_cache(p_process_state, p_instance);
		// All to zero by default.
		memset(blendw, 0, sizeof(real_t) * blend_count);

		for (const int idx : p_instance.filtered_track_indices_cache) {
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
#ifdef ENABLE_ACTIVITY_TRACKING
		for (int i = 0; i < blend_count; i++) {
			*r_activity = MAX(*r_activity, Math::abs(blendw[i]));
		}
#endif
	}

	// This process, which depends on p_sync is needed to process sync correctly in the case of
	// that a synced AnimationNodeSync exists under the un-synced AnimationNodeSync.
	if (!p_playback_info.seeked && !p_sync && !any_valid) {
		p_playback_info.delta = 0.0;
	}
	return p_other.resource->_pre_process(p_process_state, p_other, p_playback_info, p_test_only);
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

	double &length = *VariantInternal::get_float(p_instance.parameter_ptrs_by_slot[AnimationNodeInstance::SLOT_CURRENT_LENGTH]);
	double &position = *VariantInternal::get_float(p_instance.parameter_ptrs_by_slot[AnimationNodeInstance::SLOT_CURRENT_POSITION]);
	double &delta = *VariantInternal::get_float(p_instance.parameter_ptrs_by_slot[AnimationNodeInstance::SLOT_CURRENT_DELTA]);

	AnimationMixer::PlaybackInfo pi = p_playback_info;
	if (p_playback_info.seeked) {
		if (p_playback_info.is_external_seeking) {
			pi.delta = position - p_playback_info.time;
		}
	} else {
		pi.time = position + get_process_delta(p_instance, p_playback_info);
	}

	NodeTimeInfo nti = _process(p_process_state, p_instance, pi, p_test_only);

	if (!p_test_only) {
		length = nti.length;
		position = nti.position;
		delta = nti.delta;
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
		filter.insert(p_path);
	} else {
		filter.erase(p_path);
	}
	filters_dirty = true;
}

void AnimationNode::set_filter_enabled(bool p_enable) {
	filter_enabled = p_enable;
	filters_dirty = true;
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

	for (const NodePath &E : filter) {
		paths.push_back(String(E)); // Use strings, so sorting is possible.
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

void AnimationNode::_update_filter_cache(const ProcessState &p_process_state, const AnimationNodeInstance &p_instance) {
	if (!p_process_state.track_map_updated && !filters_dirty) {
		return; // Cache is valid.
	}

	p_instance.filtered_track_indices_cache.clear();
	if (p_instance.filtered_track_indices_cache.size() < filter.size()) {
		p_instance.filtered_track_indices_cache.reserve(filter.size());
	}

	for (const NodePath &path : filter) {
		if (const int *p = p_process_state.track_map->getptr(path)) {
			p_instance.filtered_track_indices_cache.push_back(*p);
		}
	}
	filters_dirty = false;
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
	ERR_FAIL_NULL(current_instance);

	AnimationMixer::PlaybackInfo info;
	info.time = p_time;
	info.delta = p_delta;
	info.seeked = p_seeked;
	info.is_external_seeking = p_is_external_seeking;
	info.weight = p_blend;
	info.looped_flag = p_looped_flag;

	blend_animation(*tls_process_state, *current_instance, p_animation, info);
}

double AnimationNode::blend_node_ex(const StringName &p_sub_path, const Ref<AnimationNode> &p_node, double p_time, bool p_seek, bool p_is_external_seeking, real_t p_blend, FilterAction p_filter, bool p_sync, bool p_test_only) {
	ERR_FAIL_NULL_V(tls_process_state, 0.0);
	ERR_FAIL_NULL_V(current_instance, 0.0);

	AnimationNodeInstance *other_instance = current_instance->get_child_instance_by_path_or_null(p_sub_path);
	ERR_FAIL_NULL_V_MSG(other_instance, 0.0, vformat("The sub-path '%s' does not exist under the current node instance.", String(p_sub_path)));

	AnimationMixer::PlaybackInfo info;
	info.time = p_time;
	info.seeked = p_seek;
	info.is_external_seeking = p_is_external_seeking;
	info.weight = p_blend;

	NodeTimeInfo nti = blend_node(*tls_process_state, *current_instance, other_instance, info, p_filter, p_sync, p_test_only);
	return nti.length - nti.position;
}

double AnimationNode::blend_input_ex(int p_input, double p_time, bool p_seek, bool p_is_external_seeking, real_t p_blend, FilterAction p_filter, bool p_sync, bool p_test_only) {
	ERR_FAIL_NULL_V(tls_process_state, 0.0);
	ERR_FAIL_NULL_V(current_instance, 0.0);

	AnimationMixer::PlaybackInfo info;
	info.time = p_time;
	info.seeked = p_seek;
	info.is_external_seeking = p_is_external_seeking;
	info.weight = p_blend;

	NodeTimeInfo nti = blend_input(*tls_process_state, *current_instance, p_input, info, p_filter, p_sync, p_test_only);
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
			for (const NodePath &E : filter) {
				r_options->push_back(String(E).quote());
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

	// For "tree_changed", wouldn't it be nice, if we could pass in the source?
	// That way we would be able to partially rebuild instances.
	ADD_SIGNAL(MethodInfo("tree_changed"));
	ADD_SIGNAL(MethodInfo("node_updated", PropertyInfo(Variant::INT, "object_id")));
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

void AnimationRootNode::_add_node(const Ref<AnimationNode> &p_node) {
	p_node->connect(SNAME("tree_changed"), callable_mp(this, &AnimationRootNode::_tree_changed), CONNECT_REFERENCE_COUNTED);
	p_node->connect(SNAME("node_updated"), callable_mp(this, &AnimationRootNode::_node_updated), CONNECT_REFERENCE_COUNTED);
	p_node->connect(SNAME("animation_node_renamed"), callable_mp(this, &AnimationRootNode::_animation_node_renamed), CONNECT_REFERENCE_COUNTED);
	p_node->connect(SNAME("animation_node_removed"), callable_mp(this, &AnimationRootNode::_animation_node_removed), CONNECT_REFERENCE_COUNTED);
}

void AnimationRootNode::_remove_node(const Ref<AnimationNode> &p_node) {
	p_node->disconnect(SNAME("tree_changed"), callable_mp(this, &AnimationRootNode::_tree_changed));
	p_node->disconnect(SNAME("node_updated"), callable_mp(this, &AnimationRootNode::_node_updated));
	p_node->disconnect(SNAME("animation_node_renamed"), callable_mp(this, &AnimationRootNode::_animation_node_renamed));
	p_node->disconnect(SNAME("animation_node_removed"), callable_mp(this, &AnimationRootNode::_animation_node_removed));
}

void AnimationRootNode::_tree_changed() {
	emit_signal(SNAME("tree_changed"));
}

void AnimationRootNode::_node_updated(const ObjectID &p_oid) {
	emit_signal(SNAME("node_updated"), p_oid);
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
		root_animation_node->disconnect(SNAME("node_updated"), callable_mp(this, &AnimationTree::_node_updated));
		root_animation_node->disconnect(SNAME("animation_node_renamed"), callable_mp(this, &AnimationTree::_animation_node_renamed));
		root_animation_node->disconnect(SNAME("animation_node_removed"), callable_mp(this, &AnimationTree::_animation_node_removed));
	}

	root_animation_node = p_animation_node;

	if (root_animation_node.is_valid()) {
		root_animation_node->connect(SNAME("tree_changed"), callable_mp(this, &AnimationTree::_tree_changed));
		root_animation_node->connect(SNAME("node_updated"), callable_mp(this, &AnimationTree::_node_updated));
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

	bool was_validation_dirty = validation_dirty;
	if (validation_dirty) {
		process_state.invalid_instances.clear();
		validation_successful = true;
		_validate_animation_graph(Animation::PARAMETERS_BASE_PATH, root_animation_node);
		validation_dirty = false;
	}
	if (!validation_successful) {
		return false;
	}
	if (was_validation_dirty) {
		_update_connections();
	}

	AnimationNodeInstance &instance = get_node_instance_by_path(SNAME(Animation::PARAMETERS_BASE_PATH.ascii().get_data()));

	{ // Setup.
		process_pass++;
		if (unlikely(process_pass == 0)) {
			process_pass = 1;
		}

		// Init process state.
		process_state = AnimationNode::ProcessState();
		process_state.tree = this;
		process_state.valid = true;
		process_state.invalid_instances.clear();
		process_state.last_pass = process_pass;
		process_state.track_map = &p_track_map;
		process_state.track_map_updated = track_map_version != last_track_map_version;
		process_state.is_testing = false;

		last_track_map_version = track_map_version;

		// Init node state for root AnimationNode.
		instance.track_weights.resize(p_track_count);
		real_t *src_blendsw = instance.track_weights.ptr();
		for (int i = 0; i < p_track_count; i++) {
			src_blendsw[i] = 1.0; // By default all go to 1 for the root input.
		}
		instance.path = SNAME(Animation::PARAMETERS_BASE_PATH.ascii().get_data());
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

		AnimationNode::tls_process_state = &process_state;
		root_animation_node->_pre_process(process_state, instance, pi, false);
		AnimationNode::tls_process_state = nullptr;
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

const AHashMap<StringName, AnimationNode::InvalidInstance> &AnimationTree::get_invalid_instances() const {
	return process_state.invalid_instances;
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

void AnimationTree::_node_updated(const ObjectID &p_oid) {
	// This is for when the animation in AnimationNodeAnimation changes.
	// or a connection in AnimationNodeBlendTree changes.

	// Ideally, we would only validate relevant instances, but for now, revalidate all.
	validation_dirty = true;
}

void AnimationTree::_animation_node_renamed(const ObjectID &p_oid, const String &p_old_name, const String &p_new_name) {
	//print_line("Node: " + ObjectDB::get_instance(p_oid)->get_class() + " (" + itos(p_oid) + ") renamed: " + p_old_name + " -> " + p_new_name);
	for (const StringName &pp : instance_paths[p_oid]) {
		String parent_path = pp;
		String old_base = parent_path + p_old_name;
		String new_base = parent_path + p_new_name;
		//print_line(" - Updating " + pp + ": " + old_base + " -> " + new_base);
		for (const PropertyInfo &E : properties) {
			if (E.name.begins_with(old_base)) {
				StringName old_name = E.name;
				StringName new_name = E.name.replace_first(old_base, new_base);
				//print_line("   - Property: " + String(old_name) + " -> " + String(new_name));
				property_map[new_name] = property_map[old_name];
				property_map.erase(old_name);
			}
		}
	}

	// Update tree second.
	properties_dirty = true;
	_update_properties();
}

void AnimationTree::_animation_node_removed(const ObjectID &p_oid, const StringName &p_node) {
	for (const StringName &parent_path : instance_paths[p_oid]) {
		String base_path = String(parent_path) + String(p_node);

		for (const PropertyInfo &E : properties) {
			if (E.name.begins_with(base_path)) {
				property_map.erase(E.name);
			}
		}
	}

	// Update tree second.
	properties_dirty = true;
	_update_properties();
}

void AnimationTree::_update_properties_for_node(const StringName &p_base_path, const Ref<AnimationNode> &p_node) const {
	ERR_FAIL_COND(p_node.is_null());

	instance_paths[p_node->get_instance_id()].insert(p_base_path);

	const String base_path_str = p_base_path;

	AnimationNodeInstance &instance = instance_map[p_base_path];
	instance.path = p_base_path;
	instance.resource = p_node;

#ifdef ENABLE_ACTIVITY_TRACKING
	if (p_node->get_input_count()) {
		for (int i = 0; i < p_node->get_input_count(); i++) {
			AnimationNodeInstance::Activity a;
			a.activity = 0;
			a.last_pass = 0;
			instance.input_activity.push_back(a);
		}
	}
#endif

	LocalVector<PropertyInfo> plist;
	p_node->get_parameter_list(&plist);
	for (PropertyInfo &pinfo : plist) {
		StringName pname = pinfo.name;
		StringName key = base_path_str + pname;

		Pair<Variant, bool> *param = property_map.getptr(key);
		if (!param) {
			param = &property_map.insert(key, Pair<Variant, bool>())->value;
		}
		param->first = p_node->get_parameter_default_value(pname);
		param->second = p_node->is_parameter_read_only(pname);

		pinfo.name = key;
		properties.push_back(pinfo);
	}

	LocalVector<AnimationNode::ChildNode> children;
	p_node->get_child_nodes(&children);

	// These have to be done in two passes, because _update_properties_for_node invalidates the instance reference.
	for (const AnimationNode::ChildNode &E : children) {
		ERR_CONTINUE(E.name.is_empty());
		instance.child_instances[E.name] = nullptr; // Will be set in _update_properties
	}

	for (const AnimationNode::ChildNode &E : children) {
		const StringName child_path = base_path_str + E.name + "/";
		_update_properties_for_node(child_path, E.node);
	}
}

void AnimationTree::_update_properties() const {
	if (!properties_dirty) {
		return;
	}

	// if properties are dirty, so is the validation state.
	validation_dirty = true;
	properties.clear();
	instance_map.clear();
	instance_paths.clear();

	if (root_animation_node.is_valid()) {
		_update_properties_for_node(Animation::PARAMETERS_BASE_PATH, root_animation_node);

		// Now that the properties and instances are stable, we can update them.
		for (KeyValue<StringName, AnimationNodeInstance> &E : instance_map) {
			const String &instance_path = E.key;
			AnimationNodeInstance &instance = E.value;

			// Update children.
			for (KeyValue<StringName, AnimationNodeInstance *> &kv : instance.child_instances) {
				const StringName child_path = instance_path + kv.key + "/";
				kv.value = instance_map.getptr(child_path);
				CRASH_COND(!kv.value); // Shouldn't ever happen.
			}

			// Now properties.
			const Ref<AnimationNode> &node = instance.resource;
			ERR_FAIL_COND(node.is_null());

			instance.parameter_ptrs_by_slot.resize_initialized(AnimationNodeInstance::SLOT_MAX);
			LocalVector<PropertyInfo> plist;
			node->get_parameter_list(&plist);
			for (const PropertyInfo &pinfo : plist) {
				StringName pname = pinfo.name;

				Pair<Variant, bool> *pair = property_map.getptr(instance_path + pname);
				CRASH_COND(!pair); // Shouldn't ever happen.

				instance.property_ptrs[pname] = &pair->first;

				// Some of these get special treatment.
				instance.maybe_bind_slot_property(pname, &pair->first);
			}
		}
	}

	properties_dirty = false;

	const_cast<AnimationTree *>(this)->notify_property_list_changed();
}

void AnimationTree::_validate_animation_graph(const StringName &p_path, const Ref<AnimationNode> &p_node) const {
	if (p_node.is_null()) {
		validation_successful = false;
		return;
	}

	p_node->validate_node(this, p_path);
	// We will continue even if the validation is not successful, to gather all errors.

	LocalVector<AnimationNode::ChildNode> children;
	p_node->get_child_nodes(&children);

	for (const AnimationNode::ChildNode &E : children) {
		const StringName child_path = String(p_path) + E.name + "/";
		_validate_animation_graph(child_path, E.node);
	}
}

void AnimationTree::_update_connections() {
	for (KeyValue<StringName, AnimationNodeInstance> &E : instance_map) {
		AnimationNodeInstance &parent_instance = E.value;

		const AnimationNodeBlendTree *blend_tree = Object::cast_to<AnimationNodeBlendTree>(parent_instance.resource.ptr());

		if (!blend_tree) {
			continue;
		}

		{
			const LocalVector<StringName> *output_connections = blend_tree->get_node_connection_array(SceneStringName(output));
			parent_instance.connection_instances.resize(1);
			AnimationNodeInstance *connected_instance = parent_instance.get_child_instance_by_path_or_null(output_connections->operator[](0));
			parent_instance.connection_instances[0] = connected_instance;
			CRASH_COND(!connected_instance); // Will never happen.
		}

		for (const KeyValue<StringName, AnimationNodeInstance *> &kv : parent_instance.child_instances) {
			AnimationNodeInstance *child_instance = kv.value;
			const LocalVector<StringName> &child_connections = *blend_tree->get_node_connection_array(kv.key);
			child_instance->connection_instances.clear();
			child_instance->connection_instances.resize(child_connections.size());
			for (uint32_t input = 0; input < child_connections.size(); input++) {
				const StringName &connected_node_name = child_connections[input];
				AnimationNodeInstance *connected_instance = parent_instance.get_child_instance_by_path_or_null(connected_node_name);
				child_instance->connection_instances[input] = connected_instance;
				CRASH_COND(!connected_instance); // Will never happen.
			}
		}
	}
}

void AnimationTree::_add_validation_error(const StringName &p_path, const String &p_error, int p_input_index) const {
	validation_successful = false;

	AnimationNode::InvalidInstance &invalid_instance = process_state.invalid_instances[p_path];

	if (p_input_index == -1) {
		invalid_instance.errors.push_back(p_error);
	} else {
		invalid_instance.input_errors.push_back({ p_input_index, p_error });
	}
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
	_update_properties();

	if (Pair<Variant, bool> *property_ptr = property_map.getptr(p_name)) {
		Pair<Variant, bool> *&pair = property_ptr;
		if (is_inside_tree() && pair->second) {
			return false; // Prevent to set property by user.
		}

		Variant &prop = pair->first;
		// Only copy variant if needed.
		if (Animation::needs_type_cast(prop, p_value)) {
			Variant value = p_value;
			if (Animation::validate_type_match(prop, value)) {
				prop = value;
			}
		} else {
			prop = p_value;
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
	_update_properties();

	if (const Pair<Variant, bool> *p = property_map.getptr(p_name)) {
		r_ret = p->first;
		return true;
	}

	return false;
}

void AnimationTree::_get_property_list(List<PropertyInfo> *p_list) const {
	_update_properties();

	for (const PropertyInfo &E : properties) {
		p_list->push_back(E);
	}
}

#ifdef ENABLE_ACTIVITY_TRACKING
real_t AnimationTree::get_connection_activity(const StringName &p_path, int p_connection) const {
	const AnimationNodeInstance *a = get_node_instance_by_path_or_null(p_path);
	if (!a) {
		return 0;
	}

	const LocalVector<AnimationNodeInstance::Activity> &activity = a->input_activity;
	if (p_connection < 0 || p_connection >= (int64_t)activity.size() || activity[p_connection].last_pass != process_pass) {
		return 0;
	}

	return activity[p_connection].activity;
}
#endif

#ifdef TOOLS_ENABLED
String AnimationTree::get_editor_error_message() const {
	if (!is_active()) {
		return TTR("The AnimationTree is inactive.\nActivate it in the inspector to enable playback; check node warnings if activation fails.");
	} else if (!is_enabled()) {
		return TTR("The AnimationTree node (or one of its ancestors) has its process mode set to Disabled.\nChange the process mode in the inspector to allow playback.");
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
