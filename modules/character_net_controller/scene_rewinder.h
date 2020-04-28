/*************************************************************************/
/*  scene_rewinder.h                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "scene/main/node.h"

#include "core/hash_map.h"
#include "net_utilities.h"
#include <deque>
#include <vector>

#ifndef SCENE_REWINDER_H
#define SCENE_REWINDER_H

class Rewinder;
class CharacterNetController;

struct VarData {
	uint32_t id;
	StringName name;
	Variant value;
	bool enabled;

	VarData();
	VarData(StringName p_name);
	VarData(uint32_t p_id, StringName p_name, Variant p_val, bool p_enabled);

	bool operator==(const VarData &p_other) const;
};

struct NodeData {
	uint32_t id;
	ObjectID instance_id;
	bool is_controller;
	int registered_process_count;
	Vector<VarData> vars;

	// This is valid to use only inside the process function.
	Node *cached_node;

	NodeData();
	NodeData(uint32_t p_id, ObjectID p_instance_id, bool is_controller);

	// Returns the index to access the variable.
	int find_var_by_id(uint32_t p_id) const;
};

class SceneRewinder : public Node {
	GDCLASS(SceneRewinder, Node);

	friend class Rewinder;
	friend class ServerRewinder;
	friend class ClientRewinder;
	friend class NoNetRewinder;

	/// Used to set the amount of traced frames to determine the connection healt trend.
	///
	/// This parameter depends a lot on the physics iteration per second, and
	/// an optimal parameter, with 60 physics iteration per second, is 1200;
	/// that is equivalent of the latest 20 seconds frames.
	///
	/// A smaller value will make the recovery mechanism too noisy and so useless,
	/// on the other hand a too big value will make the recovery mechanism too
	/// slow.
	int network_traced_frames;

	/// Max tolerance for missing snapshots in the `network_traced_frames`.
	int missing_input_max_tolerance;

	/// Used to control the `player` tick acceleration, so to produce more
	/// inputs.
	real_t tick_acceleration;

	/// The "optimal input size" is dynamically updated and its size
	/// change at a rate that can be controlled by this parameter.
	real_t optimal_size_acceleration;

	/// The server is several frames behind the client, the maxim amount
	/// of these frames is defined by the value of this parameter.
	///
	/// To prevent introducing virtual lag.
	int server_input_storage_size;

	/// How much frames the doll is allowed to be processed out of sync.
	/// This is useful to avoid time jumps each rewinding.
	int out_of_sync_frames_tolerance;

	real_t server_notify_state_interval;
	real_t comparison_float_tolerance;

	Rewinder *rewinder;
	bool recover_in_progress;
	bool rewinding_in_progress;

	uint32_t node_counter;
	bool generate_id;
	HashMap<ObjectID, NodeData> data;
	CharacterNetController *main_controller;

	Vector<ObjectID> controllers;
	std::vector<CharacterNetController *> cached_controllers;

	real_t time_bank;
	real_t tick_additional_speed;

public:
	static void _bind_methods();

	virtual void _notification(int p_what);

public:
	SceneRewinder();
	~SceneRewinder();

	void set_network_traced_frames(int p_size);
	int get_network_traced_frames() const;

	void set_missing_snapshots_max_tolerance(int p_tolerance);
	int get_missing_snapshots_max_tolerance() const;

	void set_tick_acceleration(real_t p_acceleration);
	real_t get_tick_acceleration() const;

	void set_optimal_size_acceleration(real_t p_acceleration);
	real_t get_optimal_size_acceleration() const;

	void set_server_input_storage_size(int p_size);
	int get_server_input_storage_size() const;

	void set_out_of_sync_frames_tolerance(int p_tolerance);
	int get_out_of_sync_frames_tolerance() const;

	void set_server_notify_state_interval(real_t p_interval);
	real_t get_server_notify_state_interval() const;

	void set_comparison_float_tolerance(real_t p_tolerance);
	real_t get_comparison_float_tolerance() const;

	void register_variable(Node *p_node, StringName p_variable, StringName p_on_change_notify_to = StringName());
	void unregister_variable(Node *p_node, StringName p_variable);

	String get_changed_event_name(StringName p_variable);

	void track_variable_changes(Node *p_node, StringName p_variable, StringName p_method);
	void untrack_variable_changes(Node *p_node, StringName p_variable, StringName p_method);

	void register_controller(Node *p_controller);
	void unregister_controller(Node *p_controller);

	void _register_controller(CharacterNetController *p_controller);
	void _unregister_controller(CharacterNetController *p_controller);

	void register_process(Node *p_node, StringName p_function);
	void unregister_process(Node *p_node, StringName p_function);

	bool is_recovered() const;
	bool is_rewinding() const;

	/// This function works only on server.
	void force_state_notify();

	// TODO this MUST disapper and __reset MUST be called automatically when the
	// connection status of the game instance change
	// TODO improve the name. Reset is missleading
	/// Can only be called on the server
	void reset();
	void __reset();
	/// Can only be called on the server
	void clear();
	void __clear();

	void _rpc_send_state(Variant p_snapshot);
	void _rpc_send_tick_additional_speed(int p_speed);

private:
	void process();

	void cache_controllers();

	real_t get_pretended_delta() const;

	// Read the node variables and store the value if is different from the
	// previous one. Emit a signal for each changed variable.
	void pull_variable_changes(Vector<ObjectID> *r_null_objects = nullptr);

	void on_peer_connected(int p_peer_id);
	void on_peer_disconnected(int p_peer_id);

	NodeData *register_node(Node *p_node);

	// Returns true when the vectors are the same.
	bool vec2_evaluation(const Vector2 a, const Vector2 b);
	// Returns true when the vectors are the same.
	bool vec3_evaluation(const Vector3 a, const Vector3 b);
	// Returns true when the variants are the same.
	bool rewinder_variant_evaluation(const Variant &v_1, const Variant &v_2);

	bool is_client() const;
};

struct PeerData {
	int peer;
	real_t optimal_snapshots_size;
	real_t client_tick_additional_speed;
	// It goes from -100 to 100
	int client_tick_additional_speed_compressed;
	NetworkTracer network_tracer;

	PeerData();
	PeerData(int p_peer, int traced_frames);

	bool operator==(const PeerData &p_other) const;
};

struct Snapshot {
	// This is an utility variable that is used for fast comparisons.
	uint64_t player_controller_input_id;
	// TODO worth store the pointer instead of the Object ID?
	// TODO copy on write?
	HashMap<ObjectID, uint64_t> controllers_input_id;
	HashMap<ObjectID, NodeData> data;

	operator String() const;
};

struct ControllerRewinder {
	CharacterNetController *controller;
	uint64_t recovered_snapshot_input_id;
	int frames_to_skip;
	bool finished;

	void init(CharacterNetController *p_controller, int p_frames, uint64_t p_recovered_snapshot_input_id);
	void advance(int p_i, real_t p_delta);
	bool has_finished() const;
	CharacterNetController *get_controller() const;
	uint64_t get_processed_input_id(int p_i) const;
};

class Rewinder {
protected:
	SceneRewinder *scene_rewinder;

public:
	Rewinder(SceneRewinder *p_node);

	virtual void clear() = 0;

	virtual void process(real_t p_delta) = 0;
	virtual void post_process(real_t p_delta) = 0;
	virtual void receive_snapshot(Variant p_snapshot) = 0;
};

class NoNetRewinder : public Rewinder {
	friend class SceneRewinder;

public:
	NoNetRewinder(SceneRewinder *p_node);

	virtual void clear();

	virtual void process(real_t p_delta);
	virtual void post_process(real_t p_delta);
	virtual void receive_snapshot(Variant p_snapshot);
};

class ServerRewinder : public Rewinder {
	friend class SceneRewinder;

	real_t state_notifier_timer;
	Vector<PeerData> peers_data;
	uint64_t snapshot_count;

public:
	ServerRewinder(SceneRewinder *p_node);

	virtual void clear();

	void on_peer_connected(int p_peer_id);
	void on_peer_disconnected(int p_peer_id);

	Variant generate_snapshot();

	virtual void process(real_t p_delta);
	virtual void post_process(real_t p_delta);
	virtual void receive_snapshot(Variant p_snapshot);

private:
	/// This function updates the `tick_additional_speed` so that the `frames_inputs`
	/// size is enough to reduce the missing packets to 0.
	///
	/// When the internet connection is bad, the packets need more time to arrive.
	/// To heal this problem, the server tells the client to speed up a little bit
	/// so it send the inputs a bit earlier than the usual.
	///
	/// If the `frames_inputs` size is too big the input lag between the client and
	/// the server is artificial and no more dependent on the internet. For this
	/// reason the server tells the client to slowdown so to keep the `frames_inputs`
	/// size moderate to the needs.
	void adjust_player_tick_rate(real_t p_delta);
};

class ClientRewinder : public Rewinder {
	friend class SceneRewinder;

	HashMap<uint32_t, ObjectID> node_id_map;
	HashMap<uint32_t, NodePath> node_paths;

	// TODO can we get rid of this?
	uint64_t server_snapshot_id;
	uint64_t recovered_snapshot_id;
	Snapshot server_snapshot;
	std::deque<Snapshot> snapshots;

public:
	ClientRewinder(SceneRewinder *p_node);

	virtual void clear();

	virtual void process(real_t p_delta);
	virtual void post_process(real_t p_delta);
	virtual void receive_snapshot(Variant p_snapshot);

private:
	void store_snapshot();
	void update_snapshot(int p_i, int p_snapshot_index, ControllerRewinder *p_rewinders, int p_rewinder_count);

	void process_recovery(real_t p_delta);
	bool compare_and_recovery(const Snapshot &p_server_snapshot, const Snapshot &p_client_snapshot);
	void recovery_rewind(const Snapshot &p_server_snapshot, const Snapshot &p_client_snapshot, real_t p_delta);
	void parse_snapshot(Variant p_snapshot);
};

#endif
