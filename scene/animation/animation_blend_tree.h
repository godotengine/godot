#ifndef ANIMATION_BLEND_TREE_H
#define ANIMATION_BLEND_TREE_H

#include "scene/animation/animation_graph_player.h"

class AnimationNodeAnimation : public AnimationRootNode {

	GDCLASS(AnimationNodeAnimation, AnimationRootNode);

	StringName animation;

	uint64_t last_version;
	float time;
	float step;
	bool skip;

protected:
	void _validate_property(PropertyInfo &property) const;

	static void _bind_methods();

public:
	virtual String get_caption() const;
	virtual float process(float p_time, bool p_seek);

	void set_animation(const StringName &p_name);
	StringName get_animation() const;

	float get_playback_time() const;

	AnimationNodeAnimation();
};

class AnimationNodeOneShot : public AnimationNode {
	GDCLASS(AnimationNodeOneShot, AnimationNode);

public:
	enum MixMode {
		MIX_MODE_BLEND,
		MIX_MODE_ADD
	};

private:
	bool active;
	bool do_start;
	float fade_in;
	float fade_out;

	bool autorestart;
	float autorestart_delay;
	float autorestart_random_delay;
	MixMode mix;

	float time;
	float remaining;
	float autorestart_remaining;
	bool sync;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const;

	void set_fadein_time(float p_time);
	void set_fadeout_time(float p_time);

	float get_fadein_time() const;
	float get_fadeout_time() const;

	void set_autorestart(bool p_active);
	void set_autorestart_delay(float p_time);
	void set_autorestart_random_delay(float p_time);

	bool has_autorestart() const;
	float get_autorestart_delay() const;
	float get_autorestart_random_delay() const;

	void set_mix_mode(MixMode p_mix);
	MixMode get_mix_mode() const;

	void start();
	void stop();
	bool is_active() const;

	void set_use_sync(bool p_sync);
	bool is_using_sync() const;

	virtual bool has_filter() const;
	virtual float process(float p_time, bool p_seek);

	AnimationNodeOneShot();
};

VARIANT_ENUM_CAST(AnimationNodeOneShot::MixMode)

class AnimationNodeAdd : public AnimationNode {
	GDCLASS(AnimationNodeAdd, AnimationNode);

	float amount;
	bool sync;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const;

	void set_amount(float p_amount);
	float get_amount() const;

	void set_use_sync(bool p_sync);
	bool is_using_sync() const;

	virtual bool has_filter() const;
	virtual float process(float p_time, bool p_seek);

	AnimationNodeAdd();
};

class AnimationNodeBlend2 : public AnimationNode {
	GDCLASS(AnimationNodeBlend2, AnimationNode);

	float amount;
	bool sync;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const;
	virtual float process(float p_time, bool p_seek);

	void set_amount(float p_amount);
	float get_amount() const;

	void set_use_sync(bool p_sync);
	bool is_using_sync() const;

	virtual bool has_filter() const;
	AnimationNodeBlend2();
};

class AnimationNodeBlend3 : public AnimationNode {
	GDCLASS(AnimationNodeBlend3, AnimationNode);

	float amount;
	bool sync;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const;

	void set_amount(float p_amount);
	float get_amount() const;

	void set_use_sync(bool p_sync);
	bool is_using_sync() const;

	float process(float p_time, bool p_seek);
	AnimationNodeBlend3();
};

class AnimationNodeTimeScale : public AnimationNode {
	GDCLASS(AnimationNodeTimeScale, AnimationNode);

	float scale;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const;

	void set_scale(float p_scale);
	float get_scale() const;

	float process(float p_time, bool p_seek);

	AnimationNodeTimeScale();
};

class AnimationNodeTimeSeek : public AnimationNode {
	GDCLASS(AnimationNodeTimeSeek, AnimationNode);

	float seek_pos;

protected:
	static void _bind_methods();

public:
	virtual String get_caption() const;

	void set_seek_pos(float p_sec);
	float get_seek_pos() const;

	float process(float p_time, bool p_seek);

	AnimationNodeTimeSeek();
};

class AnimationNodeTransition : public AnimationNode {
	GDCLASS(AnimationNodeTransition, AnimationNode);

	enum {
		MAX_INPUTS = 32
	};
	struct InputData {

		String name;
		bool auto_advance;
		InputData() { auto_advance = false; }
	};

	InputData inputs[MAX_INPUTS];
	int enabled_inputs;

	float prev_time;
	float prev_xfading;
	int prev;
	bool switched;

	float time;
	int current;

	float xfade;

	void _update_inputs();

protected:
	static void _bind_methods();
	void _validate_property(PropertyInfo &property) const;

public:
	virtual String get_caption() const;

	void set_enabled_inputs(int p_inputs);
	int get_enabled_inputs();

	void set_input_as_auto_advance(int p_input, bool p_enable);
	bool is_input_set_as_auto_advance(int p_input) const;

	void set_input_caption(int p_input, const String &p_name);
	String get_input_caption(int p_input) const;

	void set_current(int p_current);
	int get_current() const;

	void set_cross_fade_time(float p_fade);
	float get_cross_fade_time() const;

	float process(float p_time, bool p_seek);

	AnimationNodeTransition();
};

class AnimationNodeOutput : public AnimationNode {
	GDCLASS(AnimationNodeOutput, AnimationNode)
public:
	virtual String get_caption() const;
	virtual float process(float p_time, bool p_seek);
	AnimationNodeOutput();
};

/////

class AnimationNodeBlendTree : public AnimationRootNode {
	GDCLASS(AnimationNodeBlendTree, AnimationRootNode)

	Map<StringName, Ref<AnimationNode> > nodes;

	Vector2 graph_offset;

protected:
	static void _bind_methods();
	bool _set(const StringName &p_name, const Variant &p_value);
	bool _get(const StringName &p_name, Variant &r_ret) const;
	void _get_property_list(List<PropertyInfo> *p_list) const;

public:
	enum ConnectionError {
		CONNECTION_OK,
		CONNECTION_ERROR_NO_INPUT,
		CONNECTION_ERROR_NO_INPUT_INDEX,
		CONNECTION_ERROR_NO_OUTPUT,
		CONNECTION_ERROR_SAME_NODE,
		CONNECTION_ERROR_CONNECTION_EXISTS,
		//no need to check for cycles due to tree topology
	};

	void add_node(const StringName &p_name, Ref<AnimationNode> p_node);
	Ref<AnimationNode> get_node(const StringName &p_name) const;
	void remove_node(const StringName &p_name);
	void rename_node(const StringName &p_name, const StringName &p_new_name);
	bool has_node(const StringName &p_name) const;
	StringName get_node_name(const Ref<AnimationNode> &p_node) const;

	void connect_node(const StringName &p_input_node, int p_input_index, const StringName &p_output_node);
	void disconnect_node(const StringName &p_node, int p_input_index);
	float get_connection_activity(const StringName &p_input_node, int p_input_index) const;

	struct NodeConnection {
		StringName input_node;
		int input_index;
		StringName output_node;
	};

	ConnectionError can_connect_node(const StringName &p_input_node, int p_input_index, const StringName &p_output_node) const;
	void get_node_connections(List<NodeConnection> *r_connections) const;

	virtual String get_caption() const;
	virtual float process(float p_time, bool p_seek);

	void get_node_list(List<StringName> *r_list);

	void set_graph_offset(const Vector2 &p_graph_offset);
	Vector2 get_graph_offset() const;

	virtual void set_graph_player(AnimationGraphPlayer *p_player);
	AnimationNodeBlendTree();
	~AnimationNodeBlendTree();
};

VARIANT_ENUM_CAST(AnimationNodeBlendTree::ConnectionError)

#endif // ANIMATION_BLEND_TREE_H
