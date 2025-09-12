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

#pragma once

#include "animation_mixer.h"
#include "scene/resources/animation.h"

#define HUGE_LENGTH 31540000 // 31540000 seconds mean 1 year... is it too long? It must be longer than any Animation length and Transition xfade time to prevent time inversion for AnimationNodeStateMachine.

class AnimationNodeBlendTree;
class AnimationNodeStartState;
class AnimationNodeEndState;
class AnimationTree;
struct AnimationNodeInstance;

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
	LocalVector<Input> inputs;
	AHashMap<NodePath, bool> filter;
	bool filter_enabled = false;

	// To propagate information from upstream for use in estimation of playback progress.
	// These values must be taken from the result of blend_node() or blend_input() and must be essentially read-only.
	// For example, if you want to change the position, you need to change the pi.time value of PlaybackInfo passed to blend_input(pi) and get the result.
	struct NodeTimeInfo {
		// Retain the previous frame values. These are stored into the AnimationTree's Map and exposing them as read-only values.
		double length = 0.0;
		double position = 0.0;
		double delta = 0.0;

		// Needs internally to estimate remain time, the previous frame values are not retained.
		Animation::LoopMode loop_mode = Animation::LOOP_NONE;
		bool will_end = false; // For breaking loop, it is true when just looped.
		bool is_infinity = false; // For unpredictable state machine's end.

		bool is_looping() {
			return loop_mode != Animation::LOOP_NONE;
		}
		double get_remain(bool p_break_loop = false) {
			if ((is_looping() && !p_break_loop) || is_infinity) {
				return HUGE_LENGTH;
			}
			if (is_looping() && p_break_loop && will_end) {
				return 0;
			}
			double remain = length - position;
			if (Math::is_zero_approx(remain)) {
				return 0;
			}
			return remain;
		}
	};

	// Temporary state for blending process which needs to be started in the AnimationTree, pass through the AnimationNodes, and then return to the AnimationTree.
	struct ProcessState {
		AnimationTree *tree = nullptr;
		const AHashMap<NodePath, int> *track_map; // TODO: Is there a better way to manage filter/tracks?
		bool is_testing = false;
		bool valid = false;
		String invalid_reasons;
		uint64_t last_pass = 0;
	};

	// For performance ProcessState needs to be passed down,
	// but the scripting api was already exposed before this optimization was made.
	// So to keep compatibility, we need this internal state, so that the scripting api can continue working as before.
	// It also must be thread_local, because multiple AnimationTrees can be processed in different threads.
	static thread_local ProcessState *tls_process_state;

public:
	Array _get_filters() const;
	void _set_filters(const Array &p_filters);
	friend class AnimationNodeBlendTree;

	// The time information is passed from upstream to downstream by AnimationMixer::PlaybackInfo::p_playback_info until AnimationNodeAnimation processes it.
	// Conversely, AnimationNodeAnimation returns the processed result as NodeTimeInfo from downstream to upstream.
	NodeTimeInfo _blend_node(ProcessState &p_process_state, AnimationNodeInstance &p_instance, AnimationNodeInstance &p_node_instance, const Ref<AnimationNode> &p_node, const StringName &p_subpath, AnimationNode *p_new_parent, AnimationMixer::PlaybackInfo p_playback_info, FilterAction p_filter = FILTER_IGNORE, bool p_sync = true, bool p_test_only = false, real_t *r_activity = nullptr);
	NodeTimeInfo _pre_process(ProcessState &p_process_state, AnimationNodeInstance &p_instance, const AnimationMixer::PlaybackInfo &p_playback_info, bool p_test_only = false);

protected:
	StringName current_length = "current_length";
	StringName current_position = "current_position";
	StringName current_delta = "current_delta";

	virtual NodeTimeInfo process(ProcessState &p_process_state, AnimationNodeInstance &p_instance, const AnimationMixer::PlaybackInfo &p_playback_info, bool p_test_only = false); // To organize time information. Virtualizing for especially AnimationNodeAnimation needs to take "backward" into account.
	virtual NodeTimeInfo _process(ProcessState &p_process_state, AnimationNodeInstance &p_instance, const AnimationMixer::PlaybackInfo &p_playback_info, bool p_test_only = false); // Main process.

	void blend_animation(ProcessState &p_process_state, AnimationNodeInstance &p_instance, const StringName &p_animation, AnimationMixer::PlaybackInfo &p_playback_info);
	NodeTimeInfo blend_node(ProcessState &p_process_state, AnimationNodeInstance &p_instance, const Ref<AnimationNode> &p_node, const StringName &p_subpath, const AnimationMixer::PlaybackInfo &p_playback_info, FilterAction p_filter = FILTER_IGNORE, bool p_sync = true, bool p_test_only = false);
	NodeTimeInfo blend_input(ProcessState &p_process_state, AnimationNodeInstance &p_instance, int p_input, const AnimationMixer::PlaybackInfo &p_playback_info, FilterAction p_filter = FILTER_IGNORE, bool p_sync = true, bool p_test_only = false);

	// Bind-able methods to expose for compatibility, moreover AnimationMixer::PlaybackInfo is not exposed.
	void blend_animation_ex(const StringName &p_animation, double p_time, double p_delta, bool p_seeked, bool p_is_external_seeking, real_t p_blend, Animation::LoopedFlag p_looped_flag = Animation::LOOPED_FLAG_NONE);
	double blend_node_ex(const StringName &p_sub_path, const Ref<AnimationNode> &p_node, double p_time, bool p_seek, bool p_is_external_seeking, real_t p_blend, FilterAction p_filter = FILTER_IGNORE, bool p_sync = true, bool p_test_only = false);
	double blend_input_ex(int p_input, double p_time, bool p_seek, bool p_is_external_seeking, real_t p_blend, FilterAction p_filter = FILTER_IGNORE, bool p_sync = true, bool p_test_only = false);

	void make_invalid(ProcessState &p_process_state, const String &p_reason);

	static void _bind_methods();

	void _validate_property(PropertyInfo &p_property) const;

	GDVIRTUAL0RC(Dictionary, _get_child_nodes)
	GDVIRTUAL0RC(Array, _get_parameter_list)
	GDVIRTUAL1RC(Ref<AnimationNode>, _get_child_by_name, StringName)
	GDVIRTUAL1RC(Variant, _get_parameter_default_value, StringName)
	GDVIRTUAL1RC(bool, _is_parameter_read_only, StringName)
	GDVIRTUAL4R(double, _process, double, bool, bool, bool)
	GDVIRTUAL0RC(String, _get_caption)
	GDVIRTUAL0RC(bool, _has_filter)

public:
	virtual void get_parameter_list(LocalVector<PropertyInfo> *r_list) const;
	virtual Variant get_parameter_default_value(const StringName &p_parameter) const;
	virtual bool is_parameter_read_only(const StringName &p_parameter) const;

	void set_parameter_ex(const StringName &p_name, const Variant &p_value);
	Variant get_parameter_ex(const StringName &p_name) const;

	void set_node_time_info(AnimationNodeInstance &instance, ProcessState &p_process_state, const NodeTimeInfo &p_node_time_info); // Wrapper of set_parameter().
	virtual NodeTimeInfo get_node_time_info(AnimationNodeInstance &instance, ProcessState &p_process_state) const; // Wrapper of get_parameter().

	struct ChildNode {
		StringName name;
		Ref<AnimationNode> node;
	};

	virtual void get_child_nodes(LocalVector<ChildNode> *r_child_nodes);

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

	void set_deletable(bool p_closable);
	bool is_deletable() const;

	ObjectID get_processing_animation_tree_instance_id() const;

	bool is_process_testing() const;

	virtual bool has_filter() const;

#ifdef TOOLS_ENABLED
	virtual void get_argument_options(const StringName &p_function, int p_idx, List<String> *r_options) const override;
#endif

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
};

class AnimationNodeStartState : public AnimationRootNode {
	GDCLASS(AnimationNodeStartState, AnimationRootNode);
};

class AnimationNodeEndState : public AnimationRootNode {
	GDCLASS(AnimationNodeEndState, AnimationRootNode);
};

// Per instance data, for a node.
struct AnimationNodeInstance {
	// TODO: Maybe ptr to parent AnimationNodeInstance for faster access???
	AnimationNode *parent = nullptr;
	Vector<StringName> connections;
	LocalVector<real_t> track_weights;
	StringName base_path;
	mutable AHashMap<StringName, int> property_cache;
	mutable AHashMap<StringName, StringName> child_base_cache; // child_name -> child_base
	mutable AHashMap<StringName, StringName> property_parent_map; // local property name -> full property path
	mutable AHashMap<StringName, Pair<Variant, bool> *> property_ptrs;

	// This makes it faster to access the most commonly used parameters, since we just index an array instead of doing a hash lookup.
	// But unfortunately, these are still Variants, which are quite slow
	// (ideally, get_parameter and set_parameter are removed, and everything is made strongly typed without variant).
	LocalVector<Pair<Variant, bool> *> parameter_ptrs_by_slot;

	// We can't put the members in here directly, because when AnimationNodeInstances are destroyed they do not persist data.
	// In an ideal world, we would create a structure like `AnimationNodeInstanceParameters`
	// It would contain the strongly typed members, and it would be persisted in AnimationTree.
	// But that is a ton of work, and this is a good enough optimization for now.
	uint32_t slot_current_length = static_cast<uint32_t>(-1);
	uint32_t slot_current_position = static_cast<uint32_t>(-1);
	uint32_t slot_current_delta = static_cast<uint32_t>(-1);

	_FORCE_INLINE_ void set_parameter(const StringName &p_name, const Variant &p_value, const bool p_test_only) {
		if (p_test_only) {
			return;
		}

		if (p_name == SNAME("current_length")) {
			ERR_FAIL_COND(p_value.get_type() != Variant::FLOAT);
			Pair<Variant, bool> *prop = parameter_ptrs_by_slot[slot_current_length];
			prop->first = p_value;
			return;
		}
		if (p_name == SNAME("current_position")) {
			ERR_FAIL_COND(p_value.get_type() != Variant::FLOAT);
			Pair<Variant, bool> *prop = parameter_ptrs_by_slot[slot_current_position];
			prop->first = p_value;
			return;
		}
		if (p_name == SNAME("current_delta")) {
			ERR_FAIL_COND(p_value.get_type() != Variant::FLOAT);
			Pair<Variant, bool> *prop = parameter_ptrs_by_slot[slot_current_delta];
			prop->first = p_value;
			return;
		}

		Pair<Variant, bool> **p = property_ptrs.getptr(p_name);
		ERR_FAIL_NULL(p);
		Pair<Variant, bool> &prop = **p;

		// Only copy variant if needed.
		if (Animation::needs_type_cast(prop.first, p_value)) {
			Variant value = p_value;
			if (Animation::validate_type_match(prop.first, value)) {
				prop.first = value;
			}
		} else {
			prop.first = p_value;
		}
	}

	_FORCE_INLINE_ Variant &get_parameter(const StringName &p_name) {
		static Variant dummy = Variant();

		if (p_name == SNAME("current_length")) {
			Pair<Variant, bool> *prop = parameter_ptrs_by_slot[slot_current_length];
			return prop->first;
		}
		if (p_name == SNAME("current_position")) {
			Pair<Variant, bool> *prop = parameter_ptrs_by_slot[slot_current_position];
			return prop->first;
		}
		if (p_name == SNAME("current_delta")) {
			Pair<Variant, bool> *prop = parameter_ptrs_by_slot[slot_current_delta];
			return prop->first;
		}

		Pair<Variant, bool> **p = property_ptrs.getptr(p_name);
		ERR_FAIL_NULL_V(p, dummy);
		Pair<Variant, bool> &prop = **p;
		return prop.first;
	}
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

	mutable LocalVector<PropertyInfo> properties;
	mutable AHashMap<StringName, Pair<Variant, bool>> property_map; // Property value and read-only flag.
	mutable AHashMap<ObjectID, AnimationNodeInstance> instance_map;

	mutable bool properties_dirty = true;

	void _update_properties() const;
	void _update_properties_for_node(const StringName &p_base_path, const Ref<AnimationNode> &p_node) const;

	void _tree_changed();
	void _animation_node_renamed(const ObjectID &p_oid, const String &p_old_name, const String &p_new_name);
	void _animation_node_removed(const ObjectID &p_oid, const StringName &p_node);

	struct Activity {
		uint64_t last_pass = 0;
		real_t activity = 0.0;
	};
	mutable AHashMap<StringName, LocalVector<Activity>> input_activity_map;
	mutable AHashMap<StringName, int> input_activity_map_get;

	NodePath animation_player;

	void _setup_animation_player();
	void _animation_player_changed();

	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	virtual uint32_t _get_libraries_property_usage() const override;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	virtual void _validate_property(PropertyInfo &p_property) const override;
	void _notification(int p_what);

	static void _bind_methods();

	virtual void _set_active(bool p_active) override;

	// Make animation instances.
	virtual bool _blend_pre_process(double p_delta, int p_track_count, const AHashMap<NodePath, int> &p_track_map) override;

#ifndef DISABLE_DEPRECATED
	void _set_process_callback_bind_compat_80813(AnimationProcessCallback p_mode);
	AnimationProcessCallback _get_process_callback_bind_compat_80813() const;
	void _set_tree_root_bind_compat_80813(const Ref<AnimationNode> &p_root);
	Ref<AnimationNode> _get_tree_root_bind_compat_80813() const;

	static void _bind_compatibility_methods();
#endif // DISABLE_DEPRECATED

public:
	AnimationNodeInstance &get_node_instance(const ObjectID p_id) {
		AnimationNodeInstance *instance = instance_map.getptr(p_id);
		CRASH_COND_MSG(instance == nullptr, "No instance found for id %s" + itos(p_id.operator uint64_t()));
		return *instance;
	}

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

#ifdef TOOLS_ENABLED
	String get_editor_error_message() const;
#endif

	AnimationTree();
	~AnimationTree();
};

#ifndef DISABLE_DEPRECATED
VARIANT_ENUM_CAST(AnimationTree::AnimationProcessCallback);
#endif // DISABLE_DEPRECATED
