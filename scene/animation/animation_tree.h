#ifndef ANIMATION_GRAPH_PLAYER_H
#define ANIMATION_GRAPH_PLAYER_H

#include "animation_player.h"
#include "scene/3d/skeleton.h"
#include "scene/3d/spatial.h"
#include "scene/resources/animation.h"

class AnimationNodeBlendTree;
class AnimationPlayer;
class AnimationTree;

class AnimationNode : public Resource {
	GDCLASS(AnimationNode, Resource)
public:
	enum FilterAction {
		FILTER_IGNORE,
		FILTER_PASS,
		FILTER_STOP,
		FILTER_BLEND
	};

	struct Input {

		String name;
		StringName connected_to;
		float activity;
		uint64_t last_pass;
	};

	Vector<Input> inputs;

	float process_input(int p_input, float p_time, bool p_seek, float p_blend);

	friend class AnimationTree;

	struct AnimationState {

		Ref<Animation> animation;
		float time;
		float delta;
		const Vector<float> *track_blends;
		float blend;
		bool seeked;
	};

	struct State {

		int track_count;
		HashMap<NodePath, int> track_map;
		List<AnimationState> animation_states;
		bool valid;
		AnimationPlayer *player;
		String invalid_reasons;
		uint64_t last_pass;
	};

	Vector<float> blends;
	State *state;
	float _pre_process(State *p_state, float p_time, bool p_seek);
	void _pre_update_animations(HashMap<NodePath, int> *track_map);
	Vector2 position;

	AnimationNode *parent;
	AnimationTree *player;

	float _blend_node(Ref<AnimationNode> p_node, float p_time, bool p_seek, float p_blend, FilterAction p_filter = FILTER_IGNORE, bool p_optimize = true, float *r_max = NULL);

	HashMap<NodePath, bool> filter;
	bool filter_enabled;

	Array _get_filters() const;
	void _set_filters(const Array &p_filters);

protected:
	void blend_animation(const StringName &p_animation, float p_time, float p_delta, bool p_seeked, float p_blend);
	float blend_node(Ref<AnimationNode> p_node, float p_time, bool p_seek, float p_blend, FilterAction p_filter = FILTER_IGNORE, bool p_optimize = true);
	float blend_input(int p_input, float p_time, bool p_seek, float p_blend, FilterAction p_filter = FILTER_IGNORE, bool p_optimize = true);
	void make_invalid(const String &p_reason);

	static void _bind_methods();

	void _validate_property(PropertyInfo &property) const;

public:
	void set_parent(AnimationNode *p_parent);
	Ref<AnimationNode> get_parent() const;
	virtual void set_tree(AnimationTree *p_player);
	AnimationTree *get_tree() const;
	AnimationPlayer *get_player() const;

	virtual float process(float p_time, bool p_seek);
	virtual String get_caption() const;

	int get_input_count() const;
	String get_input_name(int p_input);
	StringName get_input_connection(int p_input);
	void set_input_connection(int p_input, const StringName &p_connection);
	float get_input_activity(int p_input) const;

	void add_input(const String &p_name);
	void set_input_name(int p_input, const String &p_name);
	void remove_input(int p_index);

	void set_filter_path(const NodePath &p_path, bool p_enable);
	bool is_path_filtered(const NodePath &p_path) const;

	void set_filter_enabled(bool p_enable);
	bool is_filter_enabled() const;

	virtual bool has_filter() const;

	void set_position(const Vector2 &p_position);
	Vector2 get_position() const;

	AnimationNode();
};

VARIANT_ENUM_CAST(AnimationNode::FilterAction)

//root node does not allow inputs
class AnimationRootNode : public AnimationNode {
	GDCLASS(AnimationRootNode, AnimationNode)
public:
	AnimationRootNode() {}
};

class AnimationTree : public Node {
	GDCLASS(AnimationTree, Node)
public:
	enum AnimationProcessMode {
		ANIMATION_PROCESS_PHYSICS,
		ANIMATION_PROCESS_IDLE,
	};

private:
	struct TrackCache {
		uint64_t setup_pass;
		uint64_t process_pass;
		Animation::TrackType type;
		Object *object;
		ObjectID object_id;

		TrackCache() {
			setup_pass = 0;
			process_pass = 0;
			object = NULL;
			object_id = 0;
		}
		virtual ~TrackCache() {}
	};

	struct TrackCacheTransform : public TrackCache {
		Spatial *spatial;
		Skeleton *skeleton;
		int bone_idx;
		Vector3 loc;
		Quat rot;
		Vector3 scale;

		TrackCacheTransform() {
			type = Animation::TYPE_TRANSFORM;
			spatial = NULL;
			bone_idx = -1;
			skeleton = NULL;
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

		float value;
		Vector<StringName> subpath;
		TrackCacheBezier() {
			type = Animation::TYPE_BEZIER;
			value = 0;
		}
	};

	struct TrackCacheAudio : public TrackCache {

		bool playing;
		float start;
		float len;

		TrackCacheAudio() {
			type = Animation::TYPE_AUDIO;
			playing = false;
			start = 0;
			len = 0;
		}
	};

	struct TrackCacheAnimation : public TrackCache {

		bool playing;

		TrackCacheAnimation() {
			type = Animation::TYPE_ANIMATION;
			playing = false;
		}
	};

	HashMap<NodePath, TrackCache *> track_cache;
	Set<TrackCache *> playing_caches;

	Ref<AnimationNode> root;

	AnimationProcessMode process_mode;
	bool active;
	NodePath animation_player;

	AnimationNode::State state;
	bool cache_valid;
	void _node_removed(Node *p_node);
	void _caches_cleared();

	void _clear_caches();
	bool _update_caches(AnimationPlayer *player);
	void _process_graph(float p_delta);

	uint64_t setup_pass;
	uint64_t process_pass;

	bool started;

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	void set_graph_root(const Ref<AnimationNode> &p_root);
	Ref<AnimationNode> get_graph_root() const;

	void set_active(bool p_active);
	bool is_active() const;

	void set_process_mode(AnimationProcessMode p_mode);
	AnimationProcessMode get_process_mode() const;

	void set_animation_player(const NodePath &p_player);
	NodePath get_animation_player() const;

	virtual String get_configuration_warning() const;

	bool is_state_invalid() const;
	String get_invalid_state_reason() const;

	uint64_t get_last_process_pass() const;
	AnimationTree();
	~AnimationTree();
};

VARIANT_ENUM_CAST(AnimationTree::AnimationProcessMode)

#endif // ANIMATION_GRAPH_PLAYER_H
