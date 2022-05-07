/*************************************************************************/
/*  animation_tree_player.h                                              */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef ANIMATION_TREE_PLAYER_H
#define ANIMATION_TREE_PLAYER_H

#include "animation_player.h"
#include "scene/3d/skeleton.h"
#include "scene/3d/spatial.h"
#include "scene/resources/animation.h"

class AnimationTreePlayer : public Node {
	GDCLASS(AnimationTreePlayer, Node);
	OBJ_CATEGORY("Animation Nodes");

public:
	enum AnimationProcessMode {
		ANIMATION_PROCESS_PHYSICS,
		ANIMATION_PROCESS_IDLE,
	};

	enum NodeType {

		NODE_OUTPUT,
		NODE_ANIMATION,
		NODE_ONESHOT,
		NODE_MIX,
		NODE_BLEND2,
		NODE_BLEND3,
		NODE_BLEND4,
		NODE_TIMESCALE,
		NODE_TIMESEEK,
		NODE_TRANSITION,

		NODE_MAX,
	};

	enum ConnectError {

		CONNECT_OK,
		CONNECT_INCOMPLETE,
		CONNECT_CYCLE
	};

private:
	enum {

		DISCONNECTED = -1,
	};

	struct TrackKey {
		uint32_t id;
		StringName subpath_concatenated;
		int bone_idx;

		inline bool operator<(const TrackKey &p_right) const {
			if (id == p_right.id) {
				if (bone_idx == p_right.bone_idx) {
					return subpath_concatenated < p_right.subpath_concatenated;
				} else {
					return bone_idx < p_right.bone_idx;
				}
			} else {
				return id < p_right.id;
			}
		}
	};

	struct Track {
		uint32_t id;
		Object *object;
		Spatial *spatial;
		Skeleton *skeleton;
		int bone_idx;
		Vector<StringName> subpath;

		Vector3 loc;
		Quat rot;
		Vector3 scale;

		Variant value;

		bool skip;

		Track() :
				id(0),
				object(nullptr),
				spatial(nullptr),
				skeleton(nullptr),
				bone_idx(-1),
				skip(false) {}
	};

	typedef Map<TrackKey, Track> TrackMap;

	TrackMap track_map;

	struct Input {
		StringName node;
		//Input() { node=-1;  }
	};

	struct NodeBase {
		bool cycletest;

		NodeType type;
		Point2 pos;

		Vector<Input> inputs;

		NodeBase() { cycletest = false; };
		virtual ~NodeBase() { cycletest = false; }
	};

	struct NodeOut : public NodeBase {
		NodeOut() {
			type = NODE_OUTPUT;
			inputs.resize(1);
		}
	};

	struct AnimationNode : public NodeBase {
		Ref<Animation> animation;

		struct TrackRef {
			int local_track;
			Track *track;
			float weight;
		};

		uint64_t last_version;
		List<TrackRef> tref;
		AnimationNode *next;
		float time;
		float step;
		String from;
		bool skip;

		HashMap<NodePath, bool> filter;

		AnimationNode() {
			type = NODE_ANIMATION;
			next = nullptr;
			last_version = 0;
			skip = false;
		}
	};

	struct OneShotNode : public NodeBase {
		bool active;
		bool start;
		float fade_in;
		float fade_out;

		bool autorestart;
		float autorestart_delay;
		float autorestart_random_delay;
		bool mix;

		float time;
		float remaining;
		float autorestart_remaining;

		HashMap<NodePath, bool> filter;

		OneShotNode() {
			type = NODE_ONESHOT;
			fade_in = 0;
			fade_out = 0;
			inputs.resize(2);
			autorestart = false;
			autorestart_delay = 1;
			autorestart_remaining = 0;
			mix = false;
			active = false;
			start = false;
		}
	};

	struct MixNode : public NodeBase {
		float amount;
		MixNode() {
			type = NODE_MIX;
			inputs.resize(2);
		}
	};

	struct Blend2Node : public NodeBase {
		float value;
		HashMap<NodePath, bool> filter;
		Blend2Node() {
			type = NODE_BLEND2;
			value = 0;
			inputs.resize(2);
		}
	};

	struct Blend3Node : public NodeBase {
		float value;
		Blend3Node() {
			type = NODE_BLEND3;
			value = 0;
			inputs.resize(3);
		}
	};

	struct Blend4Node : public NodeBase {
		Point2 value;
		Blend4Node() {
			type = NODE_BLEND4;
			inputs.resize(4);
		}
	};

	struct TimeScaleNode : public NodeBase {
		float scale;
		TimeScaleNode() {
			type = NODE_TIMESCALE;
			scale = 1;
			inputs.resize(1);
		}
	};

	struct TimeSeekNode : public NodeBase {
		float seek_pos;

		TimeSeekNode() {
			type = NODE_TIMESEEK;
			inputs.resize(1);
			seek_pos = -1;
		}
	};

	struct TransitionNode : public NodeBase {
		struct InputData {
			bool auto_advance;
			InputData() { auto_advance = false; }
		};

		Vector<InputData> input_data;

		float prev_time;
		float prev_xfading;
		int prev;
		bool switched;

		float time;
		int current;

		float xfade;

		TransitionNode() {
			type = NODE_TRANSITION;
			xfade = 0;
			inputs.resize(1);
			input_data.resize(1);
			current = 0;
			prev = -1;
			prev_time = 0;
			prev_xfading = 0;
			switched = false;
		}
		void set_current(int p_current);
	};

	void _update_sources();

	StringName out_name;
	NodeOut *out;

	NodePath base_path;
	NodePath master;

	ConnectError last_error;
	AnimationNode *active_list;
	AnimationProcessMode animation_process_mode;
	bool processing;
	bool active;
	bool dirty_caches;
	Map<StringName, NodeBase *> node_map;

	// return time left to finish animation
	float _process_node(const StringName &p_node, AnimationNode **r_prev_anim, float p_time, bool p_seek = false, float p_fallback_weight = 1.0, HashMap<NodePath, float> *p_weights = nullptr);
	void _process_animation(float p_delta);
	bool reset_request;

	ConnectError _cycle_test(const StringName &p_at_node);
	void _clear_cycle_test();

	Track *_find_track(const NodePath &p_path);
	void _recompute_caches();
	void _recompute_caches(const StringName &p_node);
	PoolVector<String> _get_node_list();

	void _compute_weights(float *p_fallback_weight, HashMap<NodePath, float> *p_weights, float p_coeff, const HashMap<NodePath, bool> *p_filter = nullptr, float p_filtered_coeff = 0);

protected:
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;
	void _notification(int p_what);

	static void _bind_methods();

public:
	void add_node(NodeType p_type, const StringName &p_node); // nodes must be >0 node 0 is built-in (exit)
	bool node_exists(const StringName &p_name) const;

	Error node_rename(const StringName &p_node, const StringName &p_new_name);
	int node_get_input_count(const StringName &p_node) const;
	StringName node_get_input_source(const StringName &p_node, int p_input) const;

	String get_configuration_warning() const;

	/* ANIMATION NODE */
	void animation_node_set_animation(const StringName &p_node, const Ref<Animation> &p_animation);
	Ref<Animation> animation_node_get_animation(const StringName &p_node) const;
	void animation_node_set_master_animation(const StringName &p_node, const String &p_master_animation);
	String animation_node_get_master_animation(const StringName &p_node) const;
	float animation_node_get_position(const StringName &p_node) const;

	void animation_node_set_filter_path(const StringName &p_node, const NodePath &p_track_path, bool p_filter);
	void animation_node_set_get_filtered_paths(const StringName &p_node, List<NodePath> *r_paths) const;
	bool animation_node_is_path_filtered(const StringName &p_node, const NodePath &p_path) const;

	/* ONE SHOT NODE */

	void oneshot_node_set_fadein_time(const StringName &p_node, float p_time);
	void oneshot_node_set_fadeout_time(const StringName &p_node, float p_time);

	float oneshot_node_get_fadein_time(const StringName &p_node) const;
	float oneshot_node_get_fadeout_time(const StringName &p_node) const;

	void oneshot_node_set_autorestart(const StringName &p_node, bool p_active);
	void oneshot_node_set_autorestart_delay(const StringName &p_node, float p_time);
	void oneshot_node_set_autorestart_random_delay(const StringName &p_node, float p_time);

	bool oneshot_node_has_autorestart(const StringName &p_node) const;
	float oneshot_node_get_autorestart_delay(const StringName &p_node) const;
	float oneshot_node_get_autorestart_random_delay(const StringName &p_node) const;

	void oneshot_node_set_mix_mode(const StringName &p_node, bool p_mix);
	bool oneshot_node_get_mix_mode(const StringName &p_node) const;

	void oneshot_node_start(const StringName &p_node);
	void oneshot_node_stop(const StringName &p_node);
	bool oneshot_node_is_active(const StringName &p_node) const;

	void oneshot_node_set_filter_path(const StringName &p_node, const NodePath &p_filter, bool p_enable);
	void oneshot_node_set_get_filtered_paths(const StringName &p_node, List<NodePath> *r_paths) const;
	bool oneshot_node_is_path_filtered(const StringName &p_node, const NodePath &p_path) const;

	/* MIX/BLEND NODES */

	void mix_node_set_amount(const StringName &p_node, float p_amount);
	float mix_node_get_amount(const StringName &p_node) const;

	void blend2_node_set_amount(const StringName &p_node, float p_amount);
	float blend2_node_get_amount(const StringName &p_node) const;
	void blend2_node_set_filter_path(const StringName &p_node, const NodePath &p_filter, bool p_enable);
	void blend2_node_set_get_filtered_paths(const StringName &p_node, List<NodePath> *r_paths) const;
	bool blend2_node_is_path_filtered(const StringName &p_node, const NodePath &p_path) const;

	void blend3_node_set_amount(const StringName &p_node, float p_amount);
	float blend3_node_get_amount(const StringName &p_node) const;

	void blend4_node_set_amount(const StringName &p_node, const Point2 &p_amount);
	Point2 blend4_node_get_amount(const StringName &p_node) const;

	/* TIMESCALE/TIMESEEK NODES */

	void timescale_node_set_scale(const StringName &p_node, float p_scale);
	float timescale_node_get_scale(const StringName &p_node) const;

	void timeseek_node_seek(const StringName &p_node, float p_pos);

	/* TRANSITION NODE */

	void transition_node_set_input_count(const StringName &p_node, int p_inputs); // used for transition node
	int transition_node_get_input_count(const StringName &p_node) const;
	void transition_node_delete_input(const StringName &p_node, int p_input); // used for transition node

	void transition_node_set_input_auto_advance(const StringName &p_node, int p_input, bool p_auto_advance); // used for transition node
	bool transition_node_has_input_auto_advance(const StringName &p_node, int p_input) const;

	void transition_node_set_xfade_time(const StringName &p_node, float p_time); // used for transition node
	float transition_node_get_xfade_time(const StringName &p_node) const;

	void transition_node_set_current(const StringName &p_node, int p_current);
	int transition_node_get_current(const StringName &p_node) const;

	void node_set_position(const StringName &p_node, const Vector2 &p_pos); //for display

	/* GETS */
	Point2 node_get_position(const StringName &p_node) const; //for display

	NodeType node_get_type(const StringName &p_node) const;

	void get_node_list(List<StringName> *p_node_list) const;
	void remove_node(const StringName &p_node);

	Error connect_nodes(const StringName &p_src_node, const StringName &p_dst_node, int p_dst_input);
	bool are_nodes_connected(const StringName &p_src_node, const StringName &p_dst_node, int p_dst_input) const;
	void disconnect_nodes(const StringName &p_node, int p_input);

	void set_base_path(const NodePath &p_path);
	NodePath get_base_path() const;

	void set_master_player(const NodePath &p_path);
	NodePath get_master_player() const;

	struct Connection {
		StringName src_node;
		StringName dst_node;
		int dst_input;
	};

	void get_connection_list(List<Connection> *p_connections) const;

	/* playback */

	void set_active(bool p_active);
	bool is_active() const;

	void reset();

	void recompute_caches();

	ConnectError get_last_error() const;

	void set_animation_process_mode(AnimationProcessMode p_mode);
	AnimationProcessMode get_animation_process_mode() const;

	void _set_process(bool p_process, bool p_force = false);

	void advance(float p_time);

	AnimationTreePlayer();
	~AnimationTreePlayer();
};

VARIANT_ENUM_CAST(AnimationTreePlayer::NodeType);
VARIANT_ENUM_CAST(AnimationTreePlayer::AnimationProcessMode);

#endif // ANIMATION_TREE_PLAYER_H
