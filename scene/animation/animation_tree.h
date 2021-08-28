/*************************************************************************/
/*  animation_tree.h                                                     */
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

#ifndef ANIMATION_GRAPH_PLAYER_H
#define ANIMATION_GRAPH_PLAYER_H

#include "animation_player.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/resources/animation.h"

class AnimationNodeBlendTree;
class AnimationPlayer;
class AnimationTree;

class AnimationNode : public Resource {
	GDCLASS(AnimationNode, Resource);

public:
	enum FilterAction {
		FILTER_IGNORE,
		FILTER_PASS,
		FILTER_STOP,
		FILTER_BLEND
	};

	struct Input {
		String name;
	};

	Vector<Input> inputs;

	real_t process_input(int p_input, real_t p_time, bool p_seek, real_t p_blend);

	friend class AnimationTree;

	struct AnimationState {
		Ref<Animation> animation;
		double time = 0.0;
		double delta = 0.0;
		const Vector<real_t> *track_blends = nullptr;
		real_t blend = 0.0;
		bool seeked = false;
	};

	struct State {
		int track_count = 0;
		HashMap<NodePath, int> track_map;
		List<AnimationState> animation_states;
		bool valid = false;
		AnimationPlayer *player = nullptr;
		AnimationTree *tree = nullptr;
		String invalid_reasons;
		uint64_t last_pass = 0;
	};

	Vector<real_t> blends;
	State *state = nullptr;

	real_t _pre_process(const StringName &p_base_path, AnimationNode *p_parent, State *p_state, real_t p_time, bool p_seek, const Vector<StringName> &p_connections);
	void _pre_update_animations(HashMap<NodePath, int> *track_map);

	//all this is temporary
	StringName base_path;
	Vector<StringName> connections;
	AnimationNode *parent = nullptr;

	HashMap<NodePath, bool> filter;
	bool filter_enabled = false;

	Array _get_filters() const;
	void _set_filters(const Array &p_filters);
	friend class AnimationNodeBlendTree;
	real_t _blend_node(const StringName &p_subpath, const Vector<StringName> &p_connections, AnimationNode *p_new_parent, Ref<AnimationNode> p_node, real_t p_time, bool p_seek, real_t p_blend, FilterAction p_filter = FILTER_IGNORE, bool p_optimize = true, real_t *r_max = nullptr);

protected:
	void blend_animation(const StringName &p_animation, real_t p_time, real_t p_delta, bool p_seeked, real_t p_blend);
	real_t blend_node(const StringName &p_sub_path, Ref<AnimationNode> p_node, real_t p_time, bool p_seek, real_t p_blend, FilterAction p_filter = FILTER_IGNORE, bool p_optimize = true);
	real_t blend_input(int p_input, real_t p_time, bool p_seek, real_t p_blend, FilterAction p_filter = FILTER_IGNORE, bool p_optimize = true);
	void make_invalid(const String &p_reason);

	static void _bind_methods();

	void _validate_property(PropertyInfo &property) const override;

	void _set_parent(Object *p_parent);

	GDVIRTUAL0RC(Dictionary, _get_child_nodes)
	GDVIRTUAL0RC(Array, _get_parameter_list)
	GDVIRTUAL1RC(Ref<AnimationNode>, _get_child_by_name, StringName)
	GDVIRTUAL1RC(Variant, _get_parameter_default_value, StringName)
	GDVIRTUAL2RC(double, _process, double, bool)
	GDVIRTUAL0RC(String, _get_caption)
	GDVIRTUAL0RC(bool, _has_filter)

public:
	virtual void get_parameter_list(List<PropertyInfo> *r_list) const;
	virtual Variant get_parameter_default_value(const StringName &p_parameter) const;

	void set_parameter(const StringName &p_name, const Variant &p_value);
	Variant get_parameter(const StringName &p_name) const;

	struct ChildNode {
		StringName name;
		Ref<AnimationNode> node;
	};

	virtual void get_child_nodes(List<ChildNode> *r_child_nodes);

	virtual double process(double p_time, bool p_seek);
	virtual String get_caption() const;

	int get_input_count() const;
	String get_input_name(int p_input);

	void add_input(const String &p_name);
	void set_input_name(int p_input, const String &p_name);
	void remove_input(int p_index);

	void set_filter_path(const NodePath &p_path, bool p_enable);
	bool is_path_filtered(const NodePath &p_path) const;

	void set_filter_enabled(bool p_enable);
	bool is_filter_enabled() const;

	virtual bool has_filter() const;

	virtual Ref<AnimationNode> get_child_by_name(const StringName &p_name);

	AnimationNode();
};

VARIANT_ENUM_CAST(AnimationNode::FilterAction)

//root node does not allow inputs
class AnimationRootNode : public AnimationNode {
	GDCLASS(AnimationRootNode, AnimationNode);

public:
	AnimationRootNode() {}
};

class AnimationTree : public Node {
	GDCLASS(AnimationTree, Node);

public:
	enum AnimationProcessCallback {
		ANIMATION_PROCESS_PHYSICS,
		ANIMATION_PROCESS_IDLE,
		ANIMATION_PROCESS_MANUAL,
	};

private:
	struct TrackCache {
		bool root_motion = false;
		uint64_t setup_pass = 0;
		uint64_t process_pass = 0;
		Animation::TrackType type = Animation::TrackType::TYPE_ANIMATION;
		Object *object = nullptr;
		ObjectID object_id;

		TrackCache() {
		}
		virtual ~TrackCache() {}
	};

	struct TrackCacheTransform : public TrackCache {
#ifndef _3D_DISABLED
		Node3D *node_3d = nullptr;
		Skeleton3D *skeleton = nullptr;
#endif // _3D_DISABLED
		int bone_idx = -1;
		Vector3 loc;
		Quaternion rot;
		real_t rot_blend_accum = 0.0;
		Vector3 scale;

		TrackCacheTransform() {
			type = Animation::TYPE_TRANSFORM3D;
		}
	};

	struct TrackCacheValue : public TrackCache {
		Variant value;
		Vector<StringName> subpath;
		TrackCacheValue() { type = Animation::TYPE_VALUE; }
	};

	struct TrackCacheMethod : public TrackCache {
		TrackCacheMethod() { type = Animation::TYPE_METHOD; }
	};

	struct TrackCacheBezier : public TrackCache {
		real_t value = 0.0;
		Vector<StringName> subpath;
		TrackCacheBezier() {
			type = Animation::TYPE_BEZIER;
		}
	};

	struct TrackCacheAudio : public TrackCache {
		bool playing = false;
		real_t start = 0.0;
		real_t len = 0.0;

		TrackCacheAudio() {
			type = Animation::TYPE_AUDIO;
		}
	};

	struct TrackCacheAnimation : public TrackCache {
		bool playing = false;

		TrackCacheAnimation() {
			type = Animation::TYPE_ANIMATION;
		}
	};

	HashMap<NodePath, TrackCache *> track_cache;
	Set<TrackCache *> playing_caches;

	Ref<AnimationNode> root;

	AnimationProcessCallback process_callback = ANIMATION_PROCESS_IDLE;
	bool active = false;
	NodePath animation_player;

	AnimationNode::State state;
	bool cache_valid = false;
	void _node_removed(Node *p_node);
	void _caches_cleared();

	void _clear_caches();
	bool _update_caches(AnimationPlayer *player);
	void _process_graph(real_t p_delta);

	uint64_t setup_pass = 1;
	uint64_t process_pass = 1;

	bool started = true;

	NodePath root_motion_track;
	Transform3D root_motion_transform;

	friend class AnimationNode;
	bool properties_dirty = true;
	void _tree_changed();
	void _update_properties();
	List<PropertyInfo> properties;
	HashMap<StringName, HashMap<StringName, StringName>> property_parent_map;
	HashMap<StringName, Variant> property_map;

	struct Activity {
		uint64_t last_pass = 0;
		real_t activity = 0.0;
	};

	HashMap<StringName, Vector<Activity>> input_activity_map;
	HashMap<StringName, Vector<Activity> *> input_activity_map_get;

	void _update_properties_for_node(const String &p_base_path, Ref<AnimationNode> node);

	ObjectID last_animation_player;

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_tree_root(const Ref<AnimationNode> &p_root);
	Ref<AnimationNode> get_tree_root() const;

	void set_active(bool p_active);
	bool is_active() const;

	void set_process_callback(AnimationProcessCallback p_mode);
	AnimationProcessCallback get_process_callback() const;

	void set_animation_player(const NodePath &p_player);
	NodePath get_animation_player() const;

	TypedArray<String> get_configuration_warnings() const override;

	bool is_state_invalid() const;
	String get_invalid_state_reason() const;

	void set_root_motion_track(const NodePath &p_track);
	NodePath get_root_motion_track() const;

	Transform3D get_root_motion_transform() const;

	real_t get_connection_activity(const StringName &p_path, int p_connection) const;
	void advance(real_t p_time);

	void rename_parameter(const String &p_base, const String &p_new_base);

	uint64_t get_last_process_pass() const;
	AnimationTree();
	~AnimationTree();
};

VARIANT_ENUM_CAST(AnimationTree::AnimationProcessCallback)

#endif // ANIMATION_GRAPH_PLAYER_H
