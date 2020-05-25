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

#include "core/oa_hash_map.h"
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

public:
	enum SynchronizerType {
		SYNCHRONIZER_TYPE_NULL,
		SYNCHRONIZER_TYPE_NONETWORK,
		SYNCHRONIZER_TYPE_CLIENT,
		SYNCHRONIZER_TYPE_SERVER
	};

	typedef ObjectID ControllerID;

	struct Var {
		StringName name;
		Variant value;
	};

	struct VarData {
		uint32_t id = 0;
		Var var;
		bool skip_rewinding = false;
		bool enabled = false;

		VarData();
		VarData(StringName p_name);
		VarData(uint32_t p_id, StringName p_name, Variant p_val, bool p_skip_rewinding, bool p_enabled);

		bool operator==(const VarData &p_other) const;
	};

	struct NodeData {
		// ID used to reference this Node in the networked calls.
		uint32_t id = 0;
		ObjectID instance_id = ObjectID();
		bool is_controller = false;
		ControllerID controlled_by = ControllerID();
		ControllerID isle_id = ControllerID();
		Vector<ObjectID> controlled_nodes;
		Vector<VarData> vars;
		Vector<StringName> functions;

		// This is valid to use only inside the process function.
		Node *node = nullptr;

		NodeData();
		NodeData(uint32_t p_id, ObjectID p_instance_id, bool is_controller);

		// Returns the index to access the variable.
		int find_var_by_id(uint32_t p_id) const;
		bool can_be_part_of_isle(ControllerID p_controller_id, bool p_is_main_controller) const;
		void process(const real_t p_delta) const;
	};

	struct IsleData {
		ControllerID controller_instance_id = ControllerID();
		Vector<ObjectID> nodes;

		NetworkedController *controller = nullptr;
	};

	struct PeerData {
		// For new peers notify the state as soon as possible.
		bool force_notify_snapshot = true;
		// For new peers a full snapshot is needed.
		bool need_full_snapshot = true;
	};

	struct Snapshot {
		OAHashMap<ObjectID, uint64_t> controllers_input_id;
		OAHashMap<ObjectID, Vector<SceneSynchronizer::VarData>> node_vars;

		operator String() const;
	};

	struct IsleSnapshot {
		uint64_t input_id;
		OAHashMap<ObjectID, Vector<SceneSynchronizer::VarData>> node_vars;

		operator String() const;
	};

	struct PostponedRecover {
		SceneSynchronizer::NodeData *node_data = nullptr;
		Vector<SceneSynchronizer::Var> vars;
	};

private:
	/// How much frames the doll is allowed to be processed out of sync.
	/// This is useful to avoid time jumps each rewinding.
	int doll_desync_tolerance = 120;

	real_t server_notify_state_interval = 1.0;
	real_t comparison_float_tolerance = 0.001;

	SynchronizerType synchronizer_type = SYNCHRONIZER_TYPE_NULL;
	Synchronizer *synchronizer = nullptr;
	bool recover_in_progress = false;
	bool reset_in_progress = false;
	bool rewinding_in_progress = false;

	OAHashMap<int, PeerData> peer_data;

	uint32_t node_counter = 1;
	bool generate_id = false;
	OAHashMap<ControllerID, IsleData> isle_data;
	OAHashMap<ObjectID, NodeData> nodes_data;
	ObjectID main_controller_object_id = ObjectID();
	NetworkedController *main_controller = nullptr;

	// Just used to detect when the peer change. TODO Rimve this and use a singnal instead.
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

	void register_variable(Node *p_node, StringName p_variable, StringName p_on_change_notify_to = StringName(), bool p_skip_rewinding = false);
	void unregister_variable(Node *p_node, StringName p_variable);

	String get_changed_event_name(StringName p_variable);

	void track_variable_changes(Node *p_node, StringName p_variable, StringName p_method);
	void untrack_variable_changes(Node *p_node, StringName p_variable, StringName p_method);

	void set_node_as_controlled_by(Node *p_node, Node *p_controller);

	void unregister_controller(Node *p_controller);

	void _register_controller(NetworkedController *p_controller);
	void _unregister_controller(NetworkedController *p_controller);

	void register_process(Node *p_node, StringName p_function);
	void unregister_process(Node *p_node, StringName p_function);

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

private:
	void put_into_isle(ObjectID p_node_id, ControllerID p_isle_id);
	void remove_from_isle(ObjectID p_node_id, ControllerID p_isle_id);

	void process();

	void validate_nodes();

	real_t get_pretended_delta() const;

	// Read the node variables and store the value if is different from the
	// previous one and emits a signal.
	void pull_node_changes(NodeData *p_node_data);

	NodeData *register_node(Node *p_node);

	// Returns true when the vectors are the same.
	bool vec2_evaluation(const Vector2 a, const Vector2 b);
	// Returns true when the vectors are the same.
	bool vec3_evaluation(const Vector3 a, const Vector3 b);
	// Returns true when the variants are the same.
	bool synchronizer_variant_evaluation(const Variant &v_1, const Variant &v_2);

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
	virtual void on_node_added(ObjectID p_node_id) {}
	virtual void on_variable_added(ObjectID p_node_id, StringName p_var_name) {}
	virtual void on_variable_changed(ObjectID p_node_id, StringName p_var_name) {}
};

class NoNetSynchronizer : public Synchronizer {
	friend class SceneSynchronizer;

public:
	NoNetSynchronizer(SceneSynchronizer *p_node);

	virtual void clear();

	virtual void process();
	virtual void receive_snapshot(Variant p_snapshot);
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

	virtual void clear();
	virtual void process();
	virtual void receive_snapshot(Variant p_snapshot);
	virtual void on_node_added(ObjectID p_node_id);
	virtual void on_variable_added(ObjectID p_node_id, StringName p_var_name);
	virtual void on_variable_changed(ObjectID p_node_id, StringName p_var_name);

	void process_snapshot_notificator(real_t p_delta);
	Variant generate_snapshot(bool p_full_snapshot) const;
};

class ClientSynchronizer : public Synchronizer {
	friend class SceneSynchronizer;

	OAHashMap<uint32_t, ObjectID> node_id_map;
	OAHashMap<uint32_t, NodePath> node_paths;

	SceneSynchronizer::Snapshot server_snapshot;
	OAHashMap<SceneSynchronizer::ControllerID, std::deque<SceneSynchronizer::IsleSnapshot>> client_controllers_snapshots;
	OAHashMap<SceneSynchronizer::ControllerID, std::deque<SceneSynchronizer::IsleSnapshot>> server_controllers_snapshots;

	bool need_full_snapshot_notified = false;

public:
	ClientSynchronizer(SceneSynchronizer *p_node);

	virtual void clear();

	virtual void process();
	virtual void receive_snapshot(Variant p_snapshot);

private:
	/// Store node data organized per controller.
	void store_snapshot(const NetworkedController *p_controller);

	void store_controllers_snapshot(
			const SceneSynchronizer::Snapshot &p_snapshot,
			OAHashMap<SceneSynchronizer::ControllerID, std::deque<SceneSynchronizer::IsleSnapshot>> &r_snapshot_storage);

	void process_controllers_recovery(real_t p_delta);
	bool parse_snapshot(Variant p_snapshot);
	bool compare_vars(const SceneSynchronizer::NodeData *p_synchronizer_node_data, const Vector<SceneSynchronizer::VarData> &p_server_vars, const Vector<SceneSynchronizer::VarData> &p_client_vars, Vector<SceneSynchronizer::Var> &r_postponed_recover);

	void notify_server_full_snapshot_is_needed();
};

#endif
