/**************************************************************************/
/*  animation_tree.h                                                      */
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

#ifndef ANIMATION_TREE_H
#define ANIMATION_TREE_H

#include "animation_mixer.h"
#include "scene/resources/animation.h"

#define HUGE_LENGTH 31540000 // 31540000 seconds mean 1 year... is it too long? It must be longer than any Animation length and Transition xfade time to prevent time inversion for AnimationNodeStateMachine.

class AnimationNodeBlendTree;
class AnimationNodeStartState;
class AnimationNodeEndState;
class AnimationTree;

class AnimationNode : public Resource {
	GDCLASS(AnimationNode, Resource);

public:
	friend class AnimationTree;

	enum FilterAction {
		FILTER_IGNORE,
		FILTER_PASS,
		FILTER_STOP,
		FILTER_BLEND
	};

	struct Input {
		String name;
	};

	bool closable = false;
	Vector<Input> inputs;
	HashMap<NodePath, bool> filter;
	bool filter_enabled = false;

	// Temporary state for blending process which needs to be stored in each AnimationNodes.
	struct NodeState {
		StringName base_path;
		AnimationNode *parent = nullptr;
		Vector<StringName> connections;
		Vector<real_t> track_weights;
	} node_state;

	// Temporary state for blending process which needs to be started in the AnimationTree, pass through the AnimationNodes, and then return to the AnimationTree.
	struct ProcessState {
		AnimationTree *tree = nullptr;
		HashMap<NodePath, int> track_map; // TODO: Is there a better way to manage filter/tracks?
		bool is_testing = false;
		bool valid = false;
		String invalid_reasons;
		uint64_t last_pass = 0;
	} *process_state = nullptr;

	Array _get_filters() const;
	void _set_filters(const Array &p_filters);
	friend class AnimationNodeBlendTree;
	double _blend_node(Ref<AnimationNode> p_node, const StringName &p_subpath, AnimationNode *p_new_parent, AnimationMixer::PlaybackInfo p_playback_info, FilterAction p_filter = FILTER_IGNORE, bool p_sync = true, bool p_test_only = false, real_t *r_activity = nullptr);
	double _pre_process(ProcessState *p_process_state, AnimationMixer::PlaybackInfo p_playback_info);

protected:
	virtual double _process(const AnimationMixer::PlaybackInfo p_playback_info, bool p_test_only = false);
	double process(const AnimationMixer::PlaybackInfo p_playback_info, bool p_test_only = false);

	void blend_animation(const StringName &p_animation, AnimationMixer::PlaybackInfo p_playback_info);
	double blend_node(Ref<AnimationNode> p_node, const StringName &p_subpath, AnimationMixer::PlaybackInfo p_playback_info, FilterAction p_filter = FILTER_IGNORE, bool p_sync = true, bool p_test_only = false);
	double blend_input(int p_input, AnimationMixer::PlaybackInfo p_playback_info, FilterAction p_filter = FILTER_IGNORE, bool p_sync = true, bool p_test_only = false);

	// Bind-able methods to expose for compatibility, moreover AnimationMixer::PlaybackInfo is not exposed.
	void blend_animation_ex(const StringName &p_animation, double p_time, double p_delta, bool p_seeked, bool p_is_external_seeking, real_t p_blend, Animation::LoopedFlag p_looped_flag = Animation::LOOPED_FLAG_NONE);
	double blend_node_ex(const StringName &p_sub_path, Ref<AnimationNode> p_node, double p_time, bool p_seek, bool p_is_external_seeking, real_t p_blend, FilterAction p_filter = FILTER_IGNORE, bool p_sync = true, bool p_test_only = false);
	double blend_input_ex(int p_input, double p_time, bool p_seek, bool p_is_external_seeking, real_t p_blend, FilterAction p_filter = FILTER_IGNORE, bool p_sync = true, bool p_test_only = false);

	void make_invalid(const String &p_reason);
	AnimationTree *get_animation_tree() const;

	static void _bind_methods();

	void _validate_property(PropertyInfo &p_property) const;

	GDVIRTUAL0RC(Dictionary, _get_child_nodes)
	GDVIRTUAL0RC(Array, _get_parameter_list)
	GDVIRTUAL1RC(Ref<AnimationNode>, _get_child_by_name, StringName)
	GDVIRTUAL1RC(Variant, _get_parameter_default_value, StringName)
	GDVIRTUAL1RC(bool, _is_parameter_read_only, StringName)
	GDVIRTUAL4RC(double, _process, double, bool, bool, bool)
	GDVIRTUAL0RC(String, _get_caption)
	GDVIRTUAL0RC(bool, _has_filter)

public:
	virtual void get_parameter_list(List<PropertyInfo> *r_list) const;
	virtual Variant get_parameter_default_value(const StringName &p_parameter) const;
	virtual bool is_parameter_read_only(const StringName &p_parameter) const;

	void set_parameter(const StringName &p_name, const Variant &p_value);
	Variant get_parameter(const StringName &p_name) const;

	struct ChildNode {
		StringName name;
		Ref<AnimationNode> node;
	};

	virtual void get_child_nodes(List<ChildNode> *r_child_nodes);

	virtual String get_caption() const;

	virtual bool add_input(const String &p_name);
	virtual void remove_input(int p_index);
	virtual bool set_input_name(int p_input, const String &p_name);
	virtual String get_input_name(int p_input) const;
	int get_input_count() const;
	int find_input(const String &p_name) const;

	void set_filter_path(const NodePath &p_path, bool p_enable);
	bool is_path_filtered(const NodePath &p_path) const;

	void set_filter_enabled(bool p_enable);
	bool is_filter_enabled() const;

	void set_closable(bool p_closable);
	bool is_closable() const;

	virtual bool has_filter() const;

	virtual Ref<AnimationNode> get_child_by_name(const StringName &p_name) const;
	Ref<AnimationNode> find_node_by_path(const String &p_name) const;

	AnimationNode();
};

VARIANT_ENUM_CAST(AnimationNode::FilterAction)

// Root node does not allow inputs.
class AnimationRootNode : public AnimationNode {
	GDCLASS(AnimationRootNode, AnimationNode);

protected:
	virtual void _tree_changed();
	virtual void _animation_node_renamed(const ObjectID &p_oid, const String &p_old_name, const String &p_new_name);
	virtual void _animation_node_removed(const ObjectID &p_oid, const StringName &p_node);

public:
	AnimationRootNode() {}
};

class AnimationNodeStartState : public AnimationRootNode {
	GDCLASS(AnimationNodeStartState, AnimationRootNode);
};

class AnimationNodeEndState : public AnimationRootNode {
	GDCLASS(AnimationNodeEndState, AnimationRootNode);
};

class AnimationTree : public AnimationMixer {
	GDCLASS(AnimationTree, AnimationMixer);

#ifndef DISABLE_DEPRECATED
public:
	enum AnimationProcessCallback {
		ANIMATION_PROCESS_PHYSICS,
		ANIMATION_PROCESS_IDLE,
		ANIMATION_PROCESS_MANUAL,
	};
#endif // DISABLE_DEPRECATED

private:
	Ref<AnimationRootNode> root_animation_node;
	NodePath advance_expression_base_node = NodePath(String("."));

	AnimationNode::ProcessState process_state;
	uint64_t process_pass = 1;

	bool started = true;

	friend class AnimationNode;

	List<PropertyInfo> properties;
	HashMap<StringName, HashMap<StringName, StringName>> property_parent_map;
	HashMap<ObjectID, StringName> property_reference_map;
	HashMap<StringName, Pair<Variant, bool>> property_map; // Property value and read-only flag.

	bool properties_dirty = true;

	void _update_properties();
	void _update_properties_for_node(const String &p_base_path, Ref<AnimationNode> p_node);

	void _tree_changed();
	void _animation_node_renamed(const ObjectID &p_oid, const String &p_old_name, const String &p_new_name);
	void _animation_node_removed(const ObjectID &p_oid, const StringName &p_node);

	struct Activity {
		uint64_t last_pass = 0;
		real_t activity = 0.0;
	};
	HashMap<StringName, Vector<Activity>> input_activity_map;
	HashMap<StringName, Vector<Activity> *> input_activity_map_get;

	NodePath animation_player;

	void _setup_animation_player();
	void _animation_player_changed();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	virtual void _validate_property(PropertyInfo &p_property) const override;
	void _notification(int p_what);

	static void _bind_methods();

	virtual void _set_active(bool p_active) override;

	// Make animation instances.
	virtual bool _blend_pre_process(double p_delta, int p_track_count, const HashMap<NodePath, int> &p_track_map) override;

#ifndef DISABLE_DEPRECATED
	void _set_process_callback_bind_compat_80813(AnimationProcessCallback p_mode);
	AnimationProcessCallback _get_process_callback_bind_compat_80813() const;
	void _set_tree_root_bind_compat_80813(const Ref<AnimationNode> &p_root);
	Ref<AnimationNode> _get_tree_root_bind_compat_80813() const;

	static void _bind_compatibility_methods();
#endif // DISABLE_DEPRECATED

public:
	void set_animation_player(const NodePath &p_path);
	NodePath get_animation_player() const;

	void set_root_animation_node(const Ref<AnimationRootNode> &p_animation_node);
	Ref<AnimationRootNode> get_root_animation_node() const;

	void set_advance_expression_base_node(const NodePath &p_path);
	NodePath get_advance_expression_base_node() const;

	PackedStringArray get_configuration_warnings() const override;

	bool is_state_invalid() const;
	String get_invalid_state_reason() const;

	real_t get_connection_activity(const StringName &p_path, int p_connection) const;

	uint64_t get_last_process_pass() const;

	AnimationTree();
	~AnimationTree();
};

#ifndef DISABLE_DEPRECATED
VARIANT_ENUM_CAST(AnimationTree::AnimationProcessCallback);
#endif // DISABLE_DEPRECATED

#endif // ANIMATION_TREE_H
