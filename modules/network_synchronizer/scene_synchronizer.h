/*************************************************************************/
/*  scene_synchronizer.h                                                     */
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

	/// Flags used to control when an event is executed.
	enum NetEventFlag {
		/// Called at the end of the frame, if the value is different.
		/// It's also called when a variable is modified by the
		/// `apply_scene_changes` function.
		CHANGE = 1 << 0,

		/// Called when the variable is modified by the `NetworkSynchronizer`
		/// because not in sync with the server.
		SYNC_RECOVER = 1 << 1,

		/// Called when the variable is modified by the `NetworkSynchronizer`
		/// because it's preparing the node for the rewinding.
		SYNC_RESET = 1 << 2,

		/// Called when the variable is modified during the rewinding phase.
		SYNC_REWIND = 1 << 3,

		/// Called at the end of the recovering phase, if the value was modified
		/// during the rewinding.
		SYNC_END = 1 << 4,

		DEFAULT = CHANGE | SYNC_END,
		ALWAYS = CHANGE | SYNC_RECOVER | SYNC_RESET | SYNC_REWIND | SYNC_END
	};

private:
	real_t server_notify_state_interval = 1.0;
	real_t comparison_float_tolerance = 0.001;

	SynchronizerType synchronizer_type = SYNCHRONIZER_TYPE_NULL;
	Synchronizer *synchronizer = nullptr;
	bool recover_in_progress = false;
	bool reset_in_progress = false;
	bool rewinding_in_progress = false;

	bool peer_dirty = false;
	OAHashMap<int, NetUtility::PeerData> peer_data;

	uint32_t node_counter = 1;
	bool generate_id = false; // TODO The id generator in this way is bad. Please make sure to regenerate all the ids only on server. Most important, each time a new reset is executed the id must be regenerated so also clients can become servers.
	// All possible registered nodes.
	LocalVector<Ref<NetUtility::NodeData>> node_data;
	// Controller nodes.
	LocalVector<Ref<NetUtility::NodeData>> controllers_node_data;
	// Global nodes.
	LocalVector<Ref<NetUtility::NodeData>> global_nodes_node_data;

	// Just used to detect when the peer change. TODO Remove this and use a singnal instead.
	void *peer_ptr = nullptr;

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

	void register_variable(Node *p_node, StringName p_variable, StringName p_on_change_notify_to = StringName(), NetEventFlag p_flags = NetEventFlag::DEFAULT);
	void unregister_variable(Node *p_node, StringName p_variable);

	void set_skip_rewinding(Node *p_node, StringName p_variable, bool p_skip_rewinding);

	String get_changed_event_name(StringName p_variable);

	void track_variable_changes(Node *p_node, StringName p_variable, StringName p_method, NetEventFlag p_flags = NetEventFlag::DEFAULT);
	void untrack_variable_changes(Node *p_node, StringName p_variable, StringName p_method);

	void set_node_as_controlled_by(Node *p_node, Node *p_controller);

	void register_process(Node *p_node, StringName p_function);
	void unregister_process(Node *p_node, StringName p_function);

	void start_tracking_scene_changes(Object *p_diff_handle) const;
	void stop_tracking_scene_changes(Object *p_diff_handle) const;
	Variant pop_scene_changes(Object *p_diff_handle) const;
	void apply_scene_changes(Variant p_sync_data);

	bool is_recovered() const;
	bool is_resetted() const;
	bool is_rewinding() const;

	/// This function works only on server.
	void force_state_notify();

	void _on_peer_connected(int p_peer);
	void _on_peer_disconnected(int p_peer);

	void reset_synchronizer_mode();
	/// Can only be called by the server
	void clear();
	void __clear();

	void _rpc_send_state(Variant p_snapshot);
	void _rpc_notify_need_full_snapshot();

	void update_peers();

private:
	NetUtility::NodeData *get_node_data(ObjectID p_object_id);
	uint32_t find_global_node(ObjectID p_object_id) const;
	NetUtility::NodeData *get_controller_node_data(ControllerID p_controller_id);

	void process();

	void validate_nodes();

	real_t get_pretended_delta() const;

	// Read the node variables and store the value if is different from the
	// previous one and emits a signal.
	void pull_node_changes(NetUtility::NodeData *p_node_data);

	NetUtility::NodeData *register_node(Node *p_node);

public:
	// Returns true when the vectors are the same.
	bool vec2_evaluation(const Vector2 a, const Vector2 b) const;
	// Returns true when the vectors are the same.
	bool vec3_evaluation(const Vector3 a, const Vector3 b) const;
	// Returns true when the variants are the same.
	bool synchronizer_variant_evaluation(const Variant &v_1, const Variant &v_2) const;

	bool is_client() const;
};

class Synchronizer {
protected:
	SceneSynchronizer *scene_synchronizer;

public:
	Synchronizer(SceneSynchronizer *p_node);
	virtual ~Synchronizer();

	virtual void clear() = 0;

	virtual void process() = 0;
	virtual void receive_snapshot(Variant p_snapshot) = 0;
	virtual void on_node_added(NetUtility::NodeData *p_node_data) {}
	virtual void on_node_removed(NetUtility::NodeData *p_node_data) {}
	virtual void on_variable_added(NetUtility::NodeData *p_node_data, StringName p_var_name) {}
	virtual void on_variable_changed(NetUtility::NodeData *p_node_data, StringName p_var_name) {}
};

class NoNetSynchronizer : public Synchronizer {
	friend class SceneSynchronizer;

public:
	NoNetSynchronizer(SceneSynchronizer *p_node);

	virtual void clear() override;

	virtual void process() override;
	virtual void receive_snapshot(Variant p_snapshot) override;
};

class ServerSynchronizer : public Synchronizer {
	friend class SceneSynchronizer;

	real_t state_notifier_timer = 0.0;

	struct Change {
		bool not_known_before = false;
		Set<StringName> uknown_vars;
		Set<StringName> vars;
	};
	OAHashMap<ObjectID, Change> changes;

public:
	ServerSynchronizer(SceneSynchronizer *p_node);

	virtual void clear() override;
	virtual void process() override;
	virtual void receive_snapshot(Variant p_snapshot) override;
	virtual void on_node_added(NetUtility::NodeData *p_node_data) override;
	virtual void on_variable_added(NetUtility::NodeData *p_node_data, StringName p_var_name) override;
	virtual void on_variable_changed(NetUtility::NodeData *p_node_data, StringName p_var_name) override;

	void process_snapshot_notificator(real_t p_delta);
	Vector<Variant> global_nodes_generate_snapshot(bool p_force_full_snapshot) const;
	void controller_generate_snapshot(const NetUtility::NodeData *p_node_data, bool p_force_full_snapshot, Vector<Variant> &r_snapshot_result) const;
	void generate_snapshot_node_data(const NetUtility::NodeData *p_node_data, bool p_force_full_snapshot, Vector<Variant> &r_result) const;
};

class ClientSynchronizer : public Synchronizer {
	friend class SceneSynchronizer;

	NetUtility::NodeData *player_controller_node_data = nullptr;
	OAHashMap<uint32_t, ObjectID> node_id_map;
	OAHashMap<uint32_t, NodePath> node_paths;

	NetUtility::Snapshot last_received_snapshot;
	std::deque<NetUtility::Snapshot> client_snapshots;
	std::deque<NetUtility::Snapshot> server_snapshots;

	bool need_full_snapshot_notified = false;

public:
	ClientSynchronizer(SceneSynchronizer *p_node);

	virtual void clear() override;

	virtual void process() override;
	virtual void receive_snapshot(Variant p_snapshot) override;
	virtual void on_node_added(NetUtility::NodeData *p_node_data) override;
	virtual void on_node_removed(NetUtility::NodeData *p_node_data) override;

	bool parse_sync_data(
			Variant p_snapshot,
			void *p_user_pointer,
			void (*p_node_parse)(void *p_user_pointer, NetUtility::NodeData *p_node_data),
			void (*p_controller_parse)(void *p_user_pointer, NetUtility::NodeData *p_node_data, uint32_t p_input_id),
			void (*p_variable_parse)(void *p_user_pointer, NetUtility::NodeData *p_node_data, uint32_t p_var_id, StringName p_variable_name, const Variant &p_value));

private:
	/// Store node data organized per controller.
	void store_snapshot();

	void store_controllers_snapshot(
			const NetUtility::Snapshot &p_snapshot,
			std::deque<NetUtility::Snapshot> &r_snapshot_storage);

	void process_controllers_recovery(real_t p_delta);
	void process_paused_controller_recovery(real_t p_delta);
	bool parse_snapshot(Variant p_snapshot);
	bool compare_vars(const NetUtility::NodeData *p_synchronizer_node_data, const Vector<NetUtility::VarData> &p_server_vars, const Vector<NetUtility::VarData> &p_client_vars, Vector<NetUtility::Var> &r_postponed_recover);

	void notify_server_full_snapshot_is_needed();
};

VARIANT_ENUM_CAST(SceneSynchronizer::NetEventFlag)

#endif
