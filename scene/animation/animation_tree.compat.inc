/**************************************************************************/
/*  animation_tree.compat.inc                                             */
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

#ifndef DISABLE_DEPRECATED

// AnimationNode
void AnimationNode::set_filter_path(const NodePath &p_path, bool p_enable) {
	WARN_DEPRECATED;
	Ref<AnimationTrackFilter> track_filter = get_parameter(filter);
	ERR_FAIL_NULL(track_filter);
	track_filter->set_track(p_path, p_enable ? 1.0f : 0.0f);
}

void AnimationNode::set_filter_enabled(bool p_enable) {
	WARN_DEPRECATED;
}

bool AnimationNode::is_filter_enabled() const {
	WARN_DEPRECATED;
	return false;
}

bool AnimationNode::is_path_filtered(const NodePath &p_path) const {
	WARN_DEPRECATED;
	Ref<AnimationTrackFilter> track_filter = get_parameter(filter);
	if (track_filter.is_valid()) {
		return track_filter->has_track(p_path);
	}
	return false;
}

Array AnimationNode::_get_filters() const {
	WARN_DEPRECATED;
	Ref<AnimationTrackFilter> track_filter = get_parameter(filter);
	ERR_FAIL_NULL_V(track_filter, {});

	Array paths;
	for (const AnimationTrackFilter::TrackFilterInfo &E : track_filter->get_tracks()) {
		paths.push_back(String(E.track)); // Use strings, so sorting is possible.
	}
	paths.sort(); // Done so every time the scene is saved, it does not change.

	return paths;
}

void AnimationNode::_set_filters(const Array &p_filters) {
	WARN_DEPRECATED;
	ERR_FAIL_NULL(process_state);
	ERR_FAIL_COND(!process_state->tree->property_parent_map.has(node_state.base_path));
	ERR_FAIL_COND(!process_state->tree->property_parent_map[node_state.base_path].has(filter));
	StringName path = process_state->tree->property_parent_map[node_state.base_path][filter];

	Ref<AnimationTrackFilter> track_filter = process_state->tree->property_map[path].first;
	if (track_filter.is_null()) {
		track_filter.instantiate();
		process_state->tree->property_map[path].first = track_filter;
	}

	track_filter->clear_tracks();

	for (int i = 0; i < p_filters.size(); i++) {
		track_filter->set_track(p_filters[i], 1.0);
	}
}

void AnimationNode::_bind_compatibility_methods() {
	ClassDB::bind_method(D_METHOD("set_filter_path", "path", "enable"), &AnimationNode::set_filter_path);
	ClassDB::bind_method(D_METHOD("is_path_filtered", "path"), &AnimationNode::is_path_filtered);

	ClassDB::bind_method(D_METHOD("set_filter_enabled", "enable"), &AnimationNode::set_filter_enabled);
	ClassDB::bind_method(D_METHOD("is_filter_enabled"), &AnimationNode::is_filter_enabled);

	ClassDB::bind_method(D_METHOD("_set_filters", "filters"), &AnimationNode::_set_filters);
	ClassDB::bind_method(D_METHOD("_get_filters"), &AnimationNode::_get_filters);

	ADD_PROPERTY(PropertyInfo(Variant::BOOL, "filter_enabled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NONE), "set_filter_enabled", "is_filter_enabled");
	ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "filters", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR | PROPERTY_USAGE_INTERNAL), "_set_filters", "_get_filters");
}

// AnimationTree
void AnimationTree::_set_process_callback_bind_compat_80813(AnimationTree::AnimationProcessCallback p_mode) {
	set_callback_mode_process(static_cast<AnimationMixer::AnimationCallbackModeProcess>(static_cast<int>(p_mode)));
}

AnimationTree::AnimationProcessCallback AnimationTree::_get_process_callback_bind_compat_80813() const {
	return static_cast<AnimationProcessCallback>(static_cast<int>(get_callback_mode_process()));
}

void AnimationTree::_set_tree_root_bind_compat_80813(const Ref<AnimationNode> &p_root) {
	const Ref<AnimationRootNode> rn = Ref<AnimationRootNode>(p_root.ptr());
	if (rn.is_null()) {
		return;
	}
	return (set_root_animation_node(rn));
}

Ref<AnimationNode> AnimationTree::_get_tree_root_bind_compat_80813() const {
	const Ref<AnimationRootNode> rn = Ref<AnimationNode>(get_root_animation_node().ptr());
	return rn;
}

void AnimationTree::_bind_compatibility_methods() {
	ClassDB::bind_method(D_METHOD("set_process_callback", "mode"), &AnimationTree::_set_process_callback_bind_compat_80813);
	ClassDB::bind_method(D_METHOD("get_process_callback"), &AnimationTree::_get_process_callback_bind_compat_80813);
	ClassDB::bind_compatibility_method(D_METHOD("set_tree_root", "root"), &AnimationTree::_set_tree_root_bind_compat_80813);
	ClassDB::bind_compatibility_method(D_METHOD("get_tree_root"), &AnimationTree::_get_tree_root_bind_compat_80813);
	BIND_ENUM_CONSTANT(ANIMATION_PROCESS_PHYSICS);
	BIND_ENUM_CONSTANT(ANIMATION_PROCESS_IDLE);
	BIND_ENUM_CONSTANT(ANIMATION_PROCESS_MANUAL);
}

#endif // DISABLE_DEPRECATED
