/*************************************************************************/
/*  scene_synchronizer.h                                                 */
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

/**
	@author AndreaCatania
*/

#include "scene/main/node.h"

#include "core/local_vector.h"
#include "core/oa_hash_map.h"
#include "net_utilities.h"
#include <deque>

#ifndef SCENE_SYNCHRONIZER_H
#define SCENE_SYNCHRONIZER_H

class Synchronizer;
class NetworkedController;

/// # SceneSynchronizer
///
/// The `SceneSynchronizer` is responsible to keep the scene of all peers in sync.
/// Usually each peer has it istantiated, and depending if it's istantiated in
/// the server or in the client, it does a different thing.
///
/// ## The `Player` is playing the game on the server.
///
/// The server is authoritative and it can't never be wrong. For this reason
/// the `SceneSynchronizer` on the server sends at a fixed interval (defined by
/// `server_notify_state_interval`) a snapshot to all peers.
///
/// The clients receives the server snapshot, so it compares with the local
/// snapshot and if it's necessary perform the recovery.
///
/// ## Variable traking
///
/// The `SceneSynchronizer` is able to track any node variable. It's possible to specify
/// the variables to track using the function `register_variable`.
///
/// ## NetworkedController
/// The `NetworkedController` is able to aquire the `Player` input and perform
/// operation in sync with other peers. When a discrepancy is found by the
/// `SceneSynchronizer`, it will drive the `NetworkedController` so to recover that
/// missalignment.
///
///
/// ## Processing function
/// Some objects, that are not direclty controlled by a `Player`, may need to be
/// in sync between peers; since those are not controlled by a `Player` is
/// not necessary use the `NetworkedController`.
///
/// It's possible to specify some process functions using `register_process`.
/// The `SceneSynchronizer` will call these functions each frame, in sync with the
/// other peers.
///
/// As example object we may think about a moving platform, or a bridge that
/// opens and close, or even a simple timer to track the match time.
/// An example implementation would be:
/// ```
/// var time := 0.0
///
/// func _ready():
/// 	# Make sure this never go out of sync.
/// 	SceneSynchronizer.register_variable(self, "time")
///
/// 	# Make sure to call this in sync with other peers.
/// 	SceneSynchronizer.register_process(self, "in_sync_process")
///
/// func in_sync_process(delta: float):
/// 	time += delta
/// ```
/// In the above code the variable `time` will always be in sync.
///
//
// # Implementation details.
//
// The entry point of the above mechanism is the function `SceneSynchronizer::process()`.
// The server `SceneSynchronizer` code is inside the class `ServerSynchronizer`.
// The client `SceneSynchronizer` code is inside the class `ClientSynchronizer`.
// The no networking `SceneSynchronizer` code is inside the class `NoNetSynchronizer`.
class SceneSynchronizer : public Node {
	GDCLASS(SceneSynchronizer, Node);

	friend class Synchronizer;
	friend class ServerSynchronizer;
	friend class ClientSynchronizer;
	friend class NoNetSynchronizer;
	friend class SceneDiff;

public:
	enum SynchronizerType {
		SYNCHRONIZER_TYPE_NULL,
		SYNCHRONIZER_TYPE_NONETWORK,
		SYNCHRONIZER_TYPE_CLIENT,
		SYNCHRONIZER_TYPE_SERVER
	};

private:
	real_t server_notify_state_interval = 1.0;
	real_t comparison_float_tolerance = 0.001;

	SynchronizerType synchronizer_type = SYNCHRONIZER_TYPE_NULL;
	Synchronizer *synchronizer = nullptr;
	bool recover_in_progress = false;
	bool reset_in_progress = false;
	bool rewinding_in_progress = false;
	bool end_sync = false;

	bool peer_dirty = false;
	OAHashMap<int, NetUtility::PeerData> peer_data;

	bool generate_id = false;
	
	// All possible registered nodes.
	LocalVector<NetUtility::NodeData *> node_data;

	// All registered nodes, that have the NetworkNodeId assigned, organized per
	// NetworkNodeId.
	LocalVector<NetUtility::NodeData *> organized_node_data;

	// Controller nodes.
	LocalVector<NetUtility::NodeData *> node_data_controllers;

	// Just used to detect when the peer change. TODO Remove this and use a singnal instead.
	void *peer_ptr = nullptr;

	int event_flag;
	LocalVector<NetUtility::ChangeListener> event_listener;

public:
	static void _bind_methods();

	virtual void _notification(int p_what);

public:
	SceneSynchronizer();
	~SceneSynchronizer();

	void set_doll_desync_tolerance(int p_tolerance);
	int get_doll_desync_tolerance() const;

	void set_server_notify_state_interval(real_t p_interval);
	real_t get_server_notify_state_interval() const;

	void set_comparison_float_tolerance(real_t p_tolerance);
	real_t get_comparison_float_tolerance() const;

	/// Register a new node and returns its `NodeData`.
	NetUtility::NodeData *register_node(Node *p_node);
	uint32_t register_node_gdscript(Node *p_node);
	void unregister_node(Node *p_node);

	/// Returns the node ID.
	/// This may return `UINT32_MAX` in various cases:
	/// - The node is not registered.
	/// - The client doesn't know the ID yet.
	uint32_t get_node_id(Node *p_node);
	Node *get_node_from_id(uint32_t p_id);

	void register_variable(Node *p_node, StringName p_variable, StringName p_on_change_notify_to = StringName(), NetEventFlag p_flags = NetEventFlag::DEFAULT);
	void unregister_variable(Node *p_node, StringName p_variable);

	/// Start local node sync.
	void start_node_sync(const Node *p_node);
	/// Stop local node sync.
	void stop_node_sync(const Node *p_node);
	bool is_node_sync(const Node *p_node) const;

	/// Returns the variable ID relative to the `Node`.
	/// This may return `UINT32_MAX` in various cases:
	/// - The node is not registered.
	/// - The variable is not registered.
	/// - The client doesn't know the ID yet.
	uint32_t get_variable_id(Node *p_node, StringName p_variable);

	void set_skip_rewinding(Node *p_node, StringName p_variable, bool p_skip_rewinding);

	void track_variable_changes(Node *p_node, StringName p_variable, Object *p_object, StringName p_method, NetEventFlag p_flags = NetEventFlag::DEFAULT);
	void untrack_variable_changes(Node *p_node, StringName p_variable, Object *p_object, StringName p_method);

	void set_node_as_controlled_by(Node *p_node, Node *p_controller);

	/// Add a dependency to a controller, so that the rewinding mechanism can
	/// make sure to rewind that node when the controller is rewinded.
	/// You can remove and add dependency at any time. This operation
	/// don't need to be perfomed on server.
	void controller_add_dependency(Node *p_controller, Node *p_node);
	void controller_remove_dependency(Node *p_controller, Node *p_node);
	int controller_get_dependency_count(Node *p_controller) const;
	Node *controller_get_dependency(Node *p_controller, int p_index);

	void register_process(Node *p_node, StringName p_function);
	void unregister_process(Node *p_node, StringName p_function);

	void start_tracking_scene_changes(Object *p_diff_handle) const;
	void stop_tracking_scene_changes(Object *p_diff_handle) const;
	Variant pop_scene_changes(Object *p_diff_handle) const;
	void apply_scene_changes(Variant p_sync_data);

	bool is_recovered() const;
	bool is_resetted() const;
	bool is_rewinding() const;
	bool is_end_sync() const;

	/// This function works only on server.
	void force_state_notify();
	/// Make peers as dirty, so they will be reloaded next frame.
	void dirty_peers();

	void set_enabled(bool p_enable);
	bool is_enabled() const;

	void set_peer_networking_enable(int p_peer, bool p_enable);
	bool is_peer_networking_enable(int p_peer) const;

	void _on_peer_connected(int p_peer);
	void _on_peer_disconnected(int p_peer);

	void _on_node_removed(Node *p_node);

	void reset_synchronizer_mode();
	void clear();

	void _rpc_send_state(Variant p_snapshot);
	void _rpc_notify_need_full_snapshot();
	void _rpc_set_network_enabled(bool p_enabled);
	void _rpc_notify_peer_status(bool p_enabled);

	void update_peers();
	void clear_peers();

	void change_events_begin(int p_flag);
	void change_event_add(NetUtility::NodeData *p_node_data, NetVarId p_var_id, const Variant &p_old);
	void change_events_flush();

private:
	void expand_organized_node_data_vector(uint32_t p_size);

	/// This function is slow, but allow to take the node data even if the
	/// `NetNodeId` is not yet assigned.
	NetUtility::NodeData *find_node_data(const Node *p_node);
	const NetUtility::NodeData *find_node_data(const Node *p_node) const;

	/// This function is super fast, but only nodes with a `NetNodeId` assigned
	/// can be returned.
	NetUtility::NodeData *get_node_data(NetNodeId p_id);
	const NetUtility::NodeData *get_node_data(NetNodeId p_id) const;

	/// Returns the latest generated `NetNodeId`.
	NetNodeId get_biggest_node_id() const;

	void reset_controllers();
	void reset_controller(NetUtility::NodeData *p_controller);

	void process();
	
#ifdef DEBUG_ENABLED
	void validate_nodes();
#endif
	void purge_node_dependencies();

	real_t get_pretended_delta() const;

	/// Read the node variables and store the value if is different from the
	/// previous one and emits a signal.
	void pull_node_changes(NetUtility::NodeData *p_node_data);

	/// Add node data and generates the `NetNodeId` if allowed.
	void add_node_data(NetUtility::NodeData *p_node_data);
	void drop_node_data(NetUtility::NodeData *p_node_data);

	/// Set the node data net id.
	void set_node_data_id(NetUtility::NodeData *p_node_data, NetNodeId p_id);

public:
	/// Returns true when the vectors are the same.
	bool vec2_evaluation(const Vector2 &a, const Vector2 &b) const;
	/// Returns true when the vectors are the same.
	bool vec3_evaluation(const Vector3 &a, const Vector3 &b) const;
	/// Returns true when the variants are the same.
	bool synchronizer_variant_evaluation(const Variant &v_1, const Variant &v_2) const;

	/// Returns true if this peer is server.
	bool is_server() const;
	/// Returns true if this peer is client.
	bool is_client() const;
	/// Returns true if there is no network.
	bool is_no_network() const;
	/// Returns true if network is enabled.
	bool is_networked() const;
};

class Synchronizer {
protected:
	SceneSynchronizer *scene_synchronizer;

public:
	Synchronizer(SceneSynchronizer *p_node);
	virtual ~Synchronizer();

	virtual void clear() = 0;

	virtual void process() = 0;
	virtual void on_node_added(NetUtility::NodeData *p_node_data) {}
	virtual void on_node_removed(NetUtility::NodeData *p_node_data) {}
	virtual void on_variable_added(NetUtility::NodeData *p_node_data, StringName p_var_name) {}
	virtual void on_variable_changed(NetUtility::NodeData *p_node_data, NetVarId p_var_id, Variant p_old_value, int p_flag) {}
	virtual void on_controller_reset(NetUtility::NodeData *p_node_data) {}
};

class NoNetSynchronizer : public Synchronizer {
	friend class SceneSynchronizer;

	bool enabled = true;

public:
	NoNetSynchronizer(SceneSynchronizer *p_node);

	virtual void clear() override;
	virtual void process() override;

	void set_enabled(bool p_enabled);
	bool is_enabled() const;
};

class ServerSynchronizer : public Synchronizer {
	friend class SceneSynchronizer;

	real_t state_notifier_timer = 0.0;

	struct Change {
		bool not_known_before = false;
		Set<StringName> uknown_vars;
		Set<StringName> vars;
	};

	/// The changes; the order matters because the index is the NetNodeId.
	LocalVector<Change> changes;

public:
	ServerSynchronizer(SceneSynchronizer *p_node);

	virtual void clear() override;
	virtual void process() override;
	virtual void on_node_added(NetUtility::NodeData *p_node_data) override;
	virtual void on_variable_added(NetUtility::NodeData *p_node_data, StringName p_var_name) override;
	virtual void on_variable_changed(NetUtility::NodeData *p_node_data, NetVarId p_var_id, Variant p_old_value, int p_flag) override;

	void process_snapshot_notificator(real_t p_delta);
	Vector<Variant> global_nodes_generate_snapshot(bool p_force_full_snapshot) const;
	void controller_generate_snapshot(const NetUtility::NodeData *p_node_data, bool p_force_full_snapshot, Vector<Variant> &r_snapshot_result) const;
	void generate_snapshot_node_data(const NetUtility::NodeData *p_node_data, bool p_force_full_snapshot, Vector<Variant> &r_result) const;
};

class ClientSynchronizer : public Synchronizer {
	friend class SceneSynchronizer;

	NetUtility::NodeData *player_controller_node_data = nullptr;
	OAHashMap<NetNodeId, NodePath> node_paths;

	NetUtility::Snapshot last_received_snapshot;
	std::deque<NetUtility::Snapshot> client_snapshots;
	std::deque<NetUtility::Snapshot> server_snapshots;
	uint32_t last_checked_input = 0;
	bool enabled = true;
	bool want_to_enable = false;

	bool need_full_snapshot_notified = false;

	struct EndSyncEvent {
		NetUtility::NodeData *node_data;
		NetVarId var_id;
		Variant old_value;

		bool operator<(const EndSyncEvent &p_other) const {
			if (node_data->id == p_other.node_data->id) {
				return var_id < p_other.var_id;
			} else {
				return node_data->id < p_other.node_data->id;
			}
		}
	};

	Set<EndSyncEvent> sync_end_events;

public:
	ClientSynchronizer(SceneSynchronizer *p_node);

	virtual void clear() override;

	virtual void process() override;
	virtual void on_node_added(NetUtility::NodeData *p_node_data) override;
	virtual void on_node_removed(NetUtility::NodeData *p_node_data) override;
	virtual void on_variable_changed(NetUtility::NodeData *p_node_data, NetVarId p_var_id, Variant p_old_value, int p_flag) override;
	virtual void on_controller_reset(NetUtility::NodeData *p_node_data) override;

	void receive_snapshot(Variant p_snapshot);
	bool parse_sync_data(
			Variant p_snapshot,
			void *p_user_pointer,
			void (*p_node_parse)(void *p_user_pointer, NetUtility::NodeData *p_node_data),
			void (*p_controller_parse)(void *p_user_pointer, NetUtility::NodeData *p_node_data, uint32_t p_input_id),
			void (*p_variable_parse)(void *p_user_pointer, NetUtility::NodeData *p_node_data, NetVarId p_var_id, const Variant &p_value));

	void set_enabled(bool p_enabled);

private:
	/// Store node data organized per controller.
	void store_snapshot();

	void store_controllers_snapshot(
			const NetUtility::Snapshot &p_snapshot,
			std::deque<NetUtility::Snapshot> &r_snapshot_storage);

	void process_controllers_recovery(real_t p_delta);
	void process_paused_controller_recovery(real_t p_delta);
	bool parse_snapshot(Variant p_snapshot);
	bool compare_vars(
			const NetUtility::NodeData *p_synchronizer_node_data,
			const Vector<NetUtility::Var> &p_server_vars,
			const Vector<NetUtility::Var> &p_client_vars,
			Vector<NetUtility::Var> &r_postponed_recover);

	void notify_server_full_snapshot_is_needed();
};

VARIANT_ENUM_CAST(NetEventFlag)

#endif
