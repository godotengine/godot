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

#ifdef TOOLS_ENABLED
#define ENABLE_ACTIVITY_TRACKING
#endif

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
	HashSet<NodePath> filter;
	bool filter_enabled = false;
	bool filters_dirty = true;

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

	struct InvalidInstance {
		LocalVector<String> errors;

		struct InputError {
			int index;
			String error;
		};

		LocalVector<InputError> input_errors;
	};

	// Temporary state for blending process which needs to be started in the AnimationTree, pass through the AnimationNodes, and then return to the AnimationTree.
	struct ProcessState {
		AnimationTree *tree = nullptr;
		const AHashMap<NodePath, int> *track_map; // TODO: Is there a better way to manage filter/tracks?
		bool track_map_updated = false;
		bool is_testing = false;
		bool valid = false;
		mutable AHashMap<StringName, InvalidInstance> invalid_instances;
		uint64_t last_pass = 0;
	};

	// For performance ProcessState needs to be passed down,
	// but the scripting api was already exposed before this optimization was made.
	// So to keep compatibility, we need this internal state, so that the scripting api can continue working as before.
	// It also must be thread_local, because multiple AnimationTrees can be processed in different threads.
	static thread_local ProcessState *tls_process_state;
	static thread_local AnimationNodeInstance *current_instance;

public:
	Array _get_filters() const;
	void _set_filters(const Array &p_filters);

	void _update_filter_cache(const ProcessState &p_process_state, const AnimationNodeInstance &p_instance);

	friend class AnimationNodeBlendTree;

	virtual void validate_node(const AnimationTree *p_tree, const StringName &p_path) const {}
	// The time information is passed from upstream to downstream by AnimationMixer::PlaybackInfo::p_playback_info until AnimationNodeAnimation processes it.
	// Conversely, AnimationNodeAnimation returns the processed result as NodeTimeInfo from downstream to upstream.
	NodeTimeInfo _blend_node(ProcessState &p_process_state, AnimationNodeInstance &p_instance, AnimationNodeInstance &p_other, AnimationMixer::PlaybackInfo p_playback_info, FilterAction p_filter = FILTER_IGNORE, bool p_sync = true, bool p_test_only = false, real_t *r_activity = nullptr);
	NodeTimeInfo _pre_process(ProcessState &p_process_state, AnimationNodeInstance &p_instance, const AnimationMixer::PlaybackInfo &p_playback_info, bool p_test_only = false);

protected:
	StringName current_length = "current_length";
	StringName current_position = "current_position";
	StringName current_delta = "current_delta";

	NodeTimeInfo process(ProcessState &p_process_state, AnimationNodeInstance &p_instance, const AnimationMixer::PlaybackInfo &p_playback_info, bool p_test_only = false);
	// Virtualizing for especially AnimationNodeAnimation needs to take "backward" into account.
	_FORCE_INLINE_ virtual double get_process_delta(AnimationNodeInstance &p_instance, const AnimationMixer::PlaybackInfo &p_playback_info) const {
		return p_playback_info.delta;
	}
	virtual NodeTimeInfo _process(ProcessState &p_process_state, AnimationNodeInstance &p_instance, const AnimationMixer::PlaybackInfo &p_playback_info, bool p_test_only = false); // Main process.

	void blend_animation(ProcessState &p_process_state, AnimationNodeInstance &p_instance, const StringName &p_animation, AnimationMixer::PlaybackInfo &p_playback_info);
	NodeTimeInfo blend_node(ProcessState &p_process_state, AnimationNodeInstance &p_instance, AnimationNodeInstance *p_other, const AnimationMixer::PlaybackInfo &p_playback_info, FilterAction p_filter = FILTER_IGNORE, bool p_sync = true, bool p_test_only = false);
	NodeTimeInfo blend_input(ProcessState &p_process_state, AnimationNodeInstance &p_instance, int p_input, const AnimationMixer::PlaybackInfo &p_playback_info, FilterAction p_filter = FILTER_IGNORE, bool p_sync = true, bool p_test_only = false);

	// Bind-able methods to expose for compatibility, moreover AnimationMixer::PlaybackInfo is not exposed.
	void blend_animation_ex(const StringName &p_animation, double p_time, double p_delta, bool p_seeked, bool p_is_external_seeking, real_t p_blend, Animation::LoopedFlag p_looped_flag = Animation::LOOPED_FLAG_NONE);
	double blend_node_ex(const StringName &p_sub_path, const Ref<AnimationNode> &p_node, double p_time, bool p_seek, bool p_is_external_seeking, real_t p_blend, FilterAction p_filter = FILTER_IGNORE, bool p_sync = true, bool p_test_only = false);
	double blend_input_ex(int p_input, double p_time, bool p_seek, bool p_is_external_seeking, real_t p_blend, FilterAction p_filter = FILTER_IGNORE, bool p_sync = true, bool p_test_only = false);

	void add_validation_error(const AnimationTree *p_tree, const StringName &p_path, const String &p_error, int p_input_index = -1) const;
	void make_invalid(ProcessState &p_process_state, AnimationNodeInstance &p_instance, const String &p_reason);

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
	void _add_node(const Ref<AnimationNode> &p_node);
	void _remove_node(const Ref<AnimationNode> &p_node);
	virtual void _tree_changed();
	void _node_updated(const ObjectID &p_oid);
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
// These get created/destroyed when the AnimationRootNode is updated, so you cannot use it to store persistent data.
struct AnimationNodeInstance {
	// This makes it faster to access the most commonly used parameters, since we just index an array instead of doing a hash lookup.
	// But unfortunately, these are still Variants, which are quite slow,
	// They are also pointers, which cause a level of indirection.
	//
	// In an ideal world, get_parameter and set_parameter are removed, and everything is made strongly typed (without variant).
	// But that is a ton of work, and this is a good enough optimization for now.
	LocalVector<Variant *> parameter_ptrs_by_slot;

	mutable LocalVector<AnimationNodeInstance *> connection_instances; // AnimationNodeInstance* | nullptr
	mutable LocalVector<real_t> track_weights;
	mutable LocalVector<int> filtered_track_indices_cache;
	mutable AHashMap<StringName, AnimationNodeInstance *> child_instances; // Child Name -> AnimationNodeInstance*

	// Multiple AnimationNodeInstances can share the same resource btw.
	Ref<AnimationNode> resource;

	// TODO: these fields are like only used during initialization, or editor, or already slow paths. Consider moving them out of this struct.
	//AnimationNodeInstance *parent = nullptr;
	StringName path; // e.g. "parameters/node_name/sub_node_name/"

	mutable AHashMap<StringName, Variant *> property_ptrs;

#ifdef ENABLE_ACTIVITY_TRACKING
	struct Activity {
		uint64_t last_pass = 0;
		real_t activity = 0.0;
	};

	LocalVector<Activity> input_activity;
#endif

	// Cache for AnimationNodeAnimation
	Ref<Animation> cached_animation;
	uint32_t cached_animation_version = 0;

	// Macro to define all slots.
	// slot index, enum name, member name, variant type, native type
	// you can reuse slot index for different nodes if they don't overlap.
	// Just make sure you update SLOT_MAX accordingly.
#define ANIM_SLOT_LIST(X)                                                              \
	/* Core parameters. */                                                             \
	X(0, CURRENT_LENGTH, current_length, Variant::FLOAT, double)                       \
	X(1, CURRENT_POSITION, current_position, Variant::FLOAT, double)                   \
	X(2, CURRENT_DELTA, current_delta, Variant::FLOAT, double)                         \
	/* AnimationNodeTimeScale. */                                                      \
	X(3, TIME_SCALE, scale, Variant::FLOAT, double)                                    \
	/* AnimationNodeTimeSeek. */                                                       \
	X(3, SEEK_POS_REQUEST, seek_pos_request, Variant::FLOAT, double)                   \
	/* AnimationNodeAdd2, AnimationNodeAdd3. */                                        \
	X(3, ADD_AMOUNT, add_amount, Variant::FLOAT, double)                               \
	/* AnimationNodeBlend2, AnimationNodeBlend3. */                                    \
	X(3, BLEND_AMOUNT, blend_amount, Variant::FLOAT, double)                           \
	/* AnimationNodeSub2. */                                                           \
	X(3, SUB_AMOUNT, sub_amount, Variant::FLOAT, double)                               \
	/* AnimationNodeOneShot. */                                                        \
	X(3, ONESHOT_TIME_TO_RESTART, time_to_restart, Variant::FLOAT, double)             \
	X(4, ONESHOT_FADE_IN_REMAINING, fade_in_remaining, Variant::FLOAT, double)         \
	X(5, ONESHOT_FADE_OUT_REMAINING, fade_out_remaining, Variant::FLOAT, double)       \
	X(6, ONESHOT_ACTIVE, active, Variant::BOOL, bool)                                  \
	X(7, ONESHOT_INTERNAL_ACTIVE, internal_active, Variant::BOOL, bool)                \
	X(8, ONESHOT_REQUEST, request, Variant::INT, int)                                  \
	/* AnimationNodeAnimation */                                                       \
	X(3, ANIMATION_BACKWARD, backward, Variant::BOOL, bool)                            \
	/* AnimationNodeTransition TODO: Make ref & */                                     \
	X(3, TRANSITION_REQUEST, transition_request, Variant::STRING, String)              \
	X(4, CURRENT_INDEX, current_index, Variant::INT, int)                              \
	X(5, PREV_INDEX, prev_index, Variant::INT, int)                                    \
	X(6, PREV_XFADING, prev_xfading, Variant::FLOAT, double)                           \
	X(7, CURRENT_STATE, current_state, Variant::STRING, String)                        \
	/* AnimationNodeBlendSpace1D and AnimationNodeBlendSpace2D,                        \
	We currently cannot do blend_position, due to type same name but different type */ \
	X(3, CLOSEST, closest, Variant::INT, int)

	enum Slot : uint8_t {
#define SLOT_ENUM(index, e, _member, _variant_type, _native_type) SLOT_##e = index,
		ANIM_SLOT_LIST(SLOT_ENUM)
#undef SLOT_ENUM
				SLOT_MAX = 9
	};

	void maybe_bind_slot_property(const StringName &p_name, Variant *p_property) {
#define HANDLE_MAYBE_BIND_SLOT_PARAMETER(_index, slot, name, _variant_type, _native_type) \
	if (p_name == SNAME(#name)) {                                                         \
		parameter_ptrs_by_slot[SLOT_##slot] = p_property;                                 \
		return;                                                                           \
	}
		ANIM_SLOT_LIST(HANDLE_MAYBE_BIND_SLOT_PARAMETER)
#undef HANDLE_MAYBE_BIND_SLOT_PARAMETER
	}

#define DEFINE_SET_PARAMETER_METHOD(_index, e, member, _variant_type, native_type)            \
	_FORCE_INLINE_ void set_parameter_##member(const native_type p_value, bool p_test_only) { \
		if (p_test_only) {                                                                    \
			return;                                                                           \
		}                                                                                     \
		Variant &prop = *parameter_ptrs_by_slot[SLOT_##e];                                    \
		prop = p_value;                                                                       \
	}
	ANIM_SLOT_LIST(DEFINE_SET_PARAMETER_METHOD)
#undef DEFINE_SET_PARAMETER_METHOD

#define DEFINE_GET_PARAMETER_METHOD(_index, e, member, _variant_type, native_type) \
	_FORCE_INLINE_ native_type get_parameter_##member() const {                    \
		return *parameter_ptrs_by_slot[SLOT_##e];                                  \
	}
	ANIM_SLOT_LIST(DEFINE_GET_PARAMETER_METHOD)
#undef DEFINE_GET_PARAMETER_METHOD

	_FORCE_INLINE_ void set_parameter(const StringName &p_name, const Variant &p_value, bool p_test_only) {
		if (p_test_only) {
			return;
		}

#define HANDLE_SET_SLOT_PARAMETER(_index, slot, name, variant_type, _native_type) \
	if (p_name == SNAME(#name)) {                                                 \
		ERR_FAIL_COND(p_value.get_type() != Variant::variant_type);               \
		Variant &prop = *parameter_ptrs_by_slot[SLOT_##slot];                     \
		prop = p_value;                                                           \
		return;                                                                   \
	}
		ANIM_SLOT_LIST(HANDLE_SET_SLOT_PARAMETER)
#undef HANDLE_SET_SLOT_PARAMETER

		Variant **p = property_ptrs.getptr(p_name);
		ERR_FAIL_NULL(p);
		Variant &prop = **p;

		// Only copy variant if needed.
		if (Animation::needs_type_cast(prop, p_value)) {
			Variant value = p_value;
			if (Animation::validate_type_match(prop, value)) {
				prop = value;
			}
		} else {
			prop = p_value;
		}
	}

	_FORCE_INLINE_ Variant &get_parameter(const StringName &p_name) {
		static Variant dummy = Variant();

#define HANDLE_GET_SLOT_PARAMETER(_index, slot, name, _variant_type, _native_type) \
	if (p_name == SNAME(#name)) {                                                  \
		return *parameter_ptrs_by_slot[SLOT_##slot];                               \
	}
		ANIM_SLOT_LIST(HANDLE_GET_SLOT_PARAMETER)
#undef HANDLE_GET_SLOT_PARAMETER

		Variant **p = property_ptrs.getptr(p_name);
		ERR_FAIL_NULL_V(p, dummy);
		Variant &prop = **p;
		return prop;
	}

	_FORCE_INLINE_ AnimationNodeInstance *get_child_instance_by_path_or_null(const StringName &p_path) {
		AnimationNodeInstance **instance_ptr = child_instances.getptr(p_path);
		if (!instance_ptr) {
			return nullptr;
		}
		AnimationNodeInstance *instance = *instance_ptr;
		return instance;
	}

	_FORCE_INLINE_ AnimationNodeInstance &get_child_instance_by_path(const StringName &p_path) {
		AnimationNodeInstance **instance_ptr = child_instances.getptr(p_path);
		CRASH_COND_MSG(!instance_ptr, "No child instance found for path: \"" + String(p_path) + "\".");
		AnimationNodeInstance *instance = *instance_ptr;
		CRASH_COND_MSG(!instance, "Child instance pointer is null for path: \"" + String(p_path) + "\".");
		return *instance;
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
	uint64_t last_track_map_version = 0;

	bool started = true;

	friend class AnimationNode;

	mutable LocalVector<PropertyInfo> properties;
	mutable AHashMap<StringName, Pair<Variant, bool>> property_map; // Property value and read-only flag.
	mutable AHashMap<StringName, AnimationNodeInstance> instance_map;
	mutable AHashMap<ObjectID, HashSet<StringName>> instance_paths;

	mutable bool properties_dirty = true;
	mutable bool validation_dirty = true;
	mutable bool validation_successful = false;

	void _update_properties() const;
	void _validate_animation_graph(const StringName &p_path, const Ref<AnimationNode> &p_node) const;
	void _update_connections();
	void _add_validation_error(const StringName &p_path, const String &p_error, int p_input_index = -1) const;
	void _update_properties_for_node(const StringName &p_base_path, const Ref<AnimationNode> &p_node) const;

	void _tree_changed();
	void _node_updated(const ObjectID &p_oid);
	void _animation_node_renamed(const ObjectID &p_oid, const String &p_old_name, const String &p_new_name);
	void _animation_node_removed(const ObjectID &p_oid, const StringName &p_node);

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
	AnimationNodeInstance &get_node_instance_by_path(const StringName &p_path) {
		AnimationNodeInstance *instance = instance_map.getptr(p_path);
		CRASH_COND_MSG(instance == nullptr, vformat(R"(No instance found for path "%s".)", String(p_path)));
		return *instance;
	}

	AnimationNodeInstance *get_node_instance_by_path_or_null(const StringName &p_path) const {
		return instance_map.getptr(p_path);
	}

#if TOOLS_ENABLED
	// Used in situations where instance_map is not built.
	// Intended for the editor, because its very slow.
	Ref<AnimationNode> get_animation_node_by_path(const StringName &p_path) const {
		Ref<AnimationNode> current = root_animation_node;

		int name_count = String(p_path).substr(0, String(p_path).length() - 1).count("/");
		for (int i = 0; i < name_count; i++) {
			if (current.is_null()) {
				return Ref<AnimationNode>();
			}

			StringName child_name = StringName(String(p_path).get_slicec('/', i + 1));
			current = current->get_child_by_name(child_name);
		}

		return current;
	}
#endif

	void set_animation_player(const NodePath &p_path);
	NodePath get_animation_player() const;

	void set_root_animation_node(const Ref<AnimationRootNode> &p_animation_node);
	Ref<AnimationRootNode> get_root_animation_node() const;

	void set_advance_expression_base_node(const NodePath &p_path);
	NodePath get_advance_expression_base_node() const;

	PackedStringArray get_configuration_warnings() const override;

	bool is_state_invalid() const;
	const AHashMap<StringName, AnimationNode::InvalidInstance> &get_invalid_instances() const;

	real_t get_connection_activity(const StringName &p_path, int p_connection) const;

#ifdef TOOLS_ENABLED
	String get_editor_error_message() const;
#endif

	AnimationTree();
	~AnimationTree();
};

#ifndef DISABLE_DEPRECATED
VARIANT_ENUM_CAST(AnimationTree::AnimationProcessCallback);
#endif // DISABLE_DEPRECATED
