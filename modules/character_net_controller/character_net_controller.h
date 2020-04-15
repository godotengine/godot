/*************************************************************************/
/*  character_net_controller.h                                           */
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

#include "core/math/transform.h"
#include "core/node_path.h"
#include "input_buffer.h"
#include "net_utilities.h"
#include <deque>
#include <vector>

#ifndef CHARACTER_NET_CONTROLLER_H
#define CHARACTER_NET_CONTROLLER_H

struct Controller;
class SceneRewinder;

class CharacterNetController : public Node {
	GDCLASS(CharacterNetController, Node);

public:
	enum InputCompressionLevel {
		INPUT_COMPRESSION_LEVEL_0 = InputsBuffer::COMPRESSION_LEVEL_0,
		INPUT_COMPRESSION_LEVEL_1 = InputsBuffer::COMPRESSION_LEVEL_1,
		INPUT_COMPRESSION_LEVEL_2 = InputsBuffer::COMPRESSION_LEVEL_2,
		INPUT_COMPRESSION_LEVEL_3 = InputsBuffer::COMPRESSION_LEVEL_3
	};

private:
	/// The snapshot storage size is used to cap the amount of inputs collected by
	/// the `Master`.
	///
	/// The server sends a message, to all the connected peers, notifing its
	/// status at a fixed interval.
	/// The peers, after receiving this update, removes all the old inputs until
	/// that moment.
	///
	/// If the `input_storage_size` is too small, the clients will collect inputs
	/// intermittently, but on the other side, a too large value may introduce
	/// virtual delay.
	///
	/// With 60 iteration per seconds a good value is `300`, but is adviced to
	/// perform some tests until you find a better suitable value for your needs.
	int player_snapshot_storage_size;

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

	/// Amount of time an inputs is re-sent to each node.
	/// Resend inputs is necessary because the packets may be lost since they
	/// are sent in an unreliable way.
	int max_redundant_inputs;

	/// The server is behing several frames behind the client, the maxim amount
	/// of these frames is defined by the value of this parameter.
	///
	/// To prevent introducing virtual lag.
	int server_snapshot_storage_size;

	/// The `server_snapshot_storage_size` is dynamically updated and its size
	/// change at a rate that can be controlled by this parameter.
	real_t optimal_size_acceleration;

	/// Max tollerance for missing snapshots in the `network_traced_frames`.
	int missing_snapshots_max_tollerance;

	/// Used to control the `Master` tick acceleration.
	real_t tick_acceleration;

	/// Interval in seconds of when the server sends the player states to the
	/// peers.
	///
	/// This must be enough to allow the clients to adjust its position, so must
	/// be tweek with the adjustment speed.
	real_t state_notify_interval;

	Controller *controller;
	InputsBuffer inputs_buffer;

	SceneRewinder *scene_rewinder;

	Vector<int> active_doll_peers;
	// Disabled peers is used to stop information propagation to a particular pear.
	Vector<int> disabled_doll_peers;

public:
	static void _bind_methods();

public:
	CharacterNetController();

	void set_player_snapshot_storage_size(int p_size);
	int get_player_snapshot_storage_size() const;

	void set_network_traced_frames(int p_size);
	int get_network_traced_frames() const;

	void set_max_redundant_inputs(int p_max);
	int get_max_redundant_inputs() const;

	void set_server_snapshot_storage_size(int p_size);
	int get_server_snapshot_storage_size() const;

	void set_optimal_size_acceleration(real_t p_acceleration);
	real_t get_optimal_size_acceleration() const;

	void set_missing_snapshots_max_tollerance(int p_tollerance);
	int get_missing_snapshots_max_tollerance() const;

	void set_tick_acceleration(real_t p_acceleration);
	real_t get_tick_acceleration() const;

	void set_state_notify_interval(real_t p_interval);
	real_t get_state_notify_interval() const;

	uint64_t get_current_input_id() const;

	/// Set bool
	/// Returns the same data.
	bool input_buffer_add_bool(bool p_input, InputCompressionLevel p_compression = INPUT_COMPRESSION_LEVEL_1);

	/// Get boolean value
	bool input_buffer_read_bool(InputCompressionLevel p_compression = INPUT_COMPRESSION_LEVEL_1);

	/// Set integer
	///
	/// Returns the stored values, you can store up to the max value for the
	/// compression.
	int64_t input_buffer_add_int(int64_t p_input, InputCompressionLevel p_compression = INPUT_COMPRESSION_LEVEL_1);

	/// Get integer
	int64_t input_buffer_read_int(InputCompressionLevel p_compression = INPUT_COMPRESSION_LEVEL_1);

	/// Set the unit real.
	///
	/// **Note:** Not unitary values lead to unexpected behaviour.
	///
	/// Returns the compressed value so both the client and the peers can use
	/// the same data.
	real_t input_buffer_add_unit_real(real_t p_input, InputCompressionLevel p_compression = INPUT_COMPRESSION_LEVEL_1);

	/// Returns the unit real
	real_t input_buffer_read_unit_real(InputCompressionLevel p_compression = INPUT_COMPRESSION_LEVEL_1);

	/// Add a normalized vector2 into the buffer.
	/// Note: The compression algorithm rely on the fact that this is a
	/// normalized vector. The behaviour is unexpected for not normalized vectors.
	///
	/// Returns the decompressed vector so both the client and the peers can use
	/// the same data.
	Vector2 input_buffer_add_normalized_vector2(Vector2 p_input, InputCompressionLevel p_compression = INPUT_COMPRESSION_LEVEL_1);

	/// Read a normalized vector2 from the input buffer.
	Vector2 input_buffer_read_normalized_vector2(InputCompressionLevel p_compression = INPUT_COMPRESSION_LEVEL_1);

	/// Add a normalized vector3 into the buffer.
	/// Note: The compression algorithm rely on the fact that this is a
	/// normalized vector. The behaviour is unexpected for not normalized vectors.
	///
	/// Returns the decompressed vector so both the client and the peers can use
	/// the same data.
	Vector3 input_buffer_add_normalized_vector3(Vector3 p_input, InputCompressionLevel p_compression = INPUT_COMPRESSION_LEVEL_1);

	/// Read a normalized vector3 from the input buffer.
	Vector3 input_buffer_read_normalized_vector3(InputCompressionLevel p_compression = INPUT_COMPRESSION_LEVEL_1);

	const InputsBuffer &get_inputs_buffer() const {
		return inputs_buffer;
	}

	InputsBuffer &get_inputs_buffer_mut() {
		return inputs_buffer;
	}

	void set_doll_peer_active(int p_peer_id, bool p_active);
	const Vector<int> &get_active_doll_peers() const;

	void _on_peer_connection_change(int p_peer_id);
	void update_active_doll_peers();

	/// This must be called on server whenever the character state is changed
	/// by anything not happened on client.
	void force_state_notify();

	void replay_snapshots();

	bool is_server_controller() const;
	bool is_player_controller() const;
	bool is_doll_controller() const;
	bool is_nonet_controller() const;

public:
	void set_inputs_buffer(const BitArray &p_new_buffer);

	void set_scene_rewinder(SceneRewinder *p_rewinder);
	SceneRewinder *get_scene_rewinder() const;
	bool has_scene_rewinder() const;

	/* On server rpc functions. */
	void _rpc_server_send_frames_snapshot(Vector<uint8_t> p_data);

	/* On master rpc functions. */
	void _rpc_player_send_tick_additional_speed(int p_additional_tick_speed);

	/* On puppet rpc functions. */
	void _rpc_doll_send_frames_snapshot(Vector<uint8_t> p_data);
	void _rpc_doll_notify_connection_status(bool p_open);

	/* On all peers rpc functions. */
	void _rpc_send_player_state(uint64_t p_snapshot_id, Variant p_data);

private:
	virtual void _notification(int p_what);
};

VARIANT_ENUM_CAST(CharacterNetController::InputCompressionLevel)

class PlayerInputsReference : public Object {
	GDCLASS(PlayerInputsReference, Object);

public:
	InputsBuffer inputs_buffer;

	static void _bind_methods();

	PlayerInputsReference() {}
	PlayerInputsReference(const InputsBuffer &p_ib) :
			inputs_buffer(p_ib) {}

	int get_size() const;

	bool read_bool(CharacterNetController::InputCompressionLevel p_compression = CharacterNetController::INPUT_COMPRESSION_LEVEL_1);
	int64_t read_int(CharacterNetController::InputCompressionLevel p_compression = CharacterNetController::INPUT_COMPRESSION_LEVEL_1);
	real_t read_unit_real(CharacterNetController::InputCompressionLevel p_compression = CharacterNetController::INPUT_COMPRESSION_LEVEL_1);
	Vector2 read_normalized_vector2(CharacterNetController::InputCompressionLevel p_compression = CharacterNetController::INPUT_COMPRESSION_LEVEL_1);
	Vector3 read_normalized_vector3(CharacterNetController::InputCompressionLevel p_compression = CharacterNetController::INPUT_COMPRESSION_LEVEL_1);

	void skip_bool(CharacterNetController::InputCompressionLevel p_compression = CharacterNetController::INPUT_COMPRESSION_LEVEL_1);
	void skip_int(CharacterNetController::InputCompressionLevel p_compression = CharacterNetController::INPUT_COMPRESSION_LEVEL_1);
	void skip_unit_real(CharacterNetController::InputCompressionLevel p_compression = CharacterNetController::INPUT_COMPRESSION_LEVEL_1);
	void skip_normalized_vector2(CharacterNetController::InputCompressionLevel p_compression = CharacterNetController::INPUT_COMPRESSION_LEVEL_1);
	void skip_normalized_vector3(CharacterNetController::InputCompressionLevel p_compression = CharacterNetController::INPUT_COMPRESSION_LEVEL_1);

	int get_bool_size(CharacterNetController::InputCompressionLevel p_compression = CharacterNetController::INPUT_COMPRESSION_LEVEL_1) const;
	int get_int_size(CharacterNetController::InputCompressionLevel p_compression = CharacterNetController::INPUT_COMPRESSION_LEVEL_1) const;
	int get_unit_real_size(CharacterNetController::InputCompressionLevel p_compression = CharacterNetController::INPUT_COMPRESSION_LEVEL_1) const;
	int get_normalized_vector2_size(CharacterNetController::InputCompressionLevel p_compression = CharacterNetController::INPUT_COMPRESSION_LEVEL_1) const;
	int get_normalized_vector3_size(CharacterNetController::InputCompressionLevel p_compression = CharacterNetController::INPUT_COMPRESSION_LEVEL_1) const;

	void begin();
	void set_inputs_buffer(const BitArray &p_new_buffer);
};

struct FrameSnapshotSkinny {
	uint64_t id;
	BitArray inputs_buffer;
};

struct FrameSnapshot {
	uint64_t id;
	BitArray inputs_buffer;
	Variant custom_data;
	uint64_t similarity;
};

struct Controller {
	CharacterNetController *node;

	Controller(CharacterNetController *p_node) :
			node(p_node) {}

	virtual ~Controller() {}

	virtual void physics_process(real_t p_delta) = 0;
	virtual void receive_snapshots(Vector<uint8_t> p_data) = 0;

	/// The server call this function on all peers with on server state.
	/// The peers can check if the state is the same or not and in this case
	/// recover its player state.
	virtual void player_state_check(uint64_t p_id, Variant p_data) = 0;
	virtual void replay_snapshots() = 0;
	virtual uint64_t get_current_snapshot_id() const = 0;
};

struct ServerController : public Controller {
	uint64_t current_input_buffer_id;
	uint32_t ghost_input_count;
	NetworkTracer network_tracer;
	std::deque<FrameSnapshotSkinny> snapshots;
	real_t optimal_snapshots_size;
	real_t client_tick_additional_speed;
	// It goes from -100 to 100
	int client_tick_additional_speed_compressed;
	real_t peers_state_checker_time;

	ServerController(CharacterNetController *p_node);

	virtual void physics_process(real_t p_delta);
	virtual void receive_snapshots(Vector<uint8_t> p_data);
	virtual void player_state_check(uint64_t p_snapshot_id, Variant p_data);
	virtual void replay_snapshots();
	virtual uint64_t get_current_snapshot_id() const;

	/// Fetch the next inputs, returns true if the input is new.
	bool fetch_next_input();

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

	/// This function is executed on server, and call a client function that
	/// checks if the player state is consistent between client and server.
	void check_peers_player_state(real_t p_delta, bool is_new_input);

	/// Forces the state check to be notified on next frame.
	void force_state_notify();
};

struct PlayerController : public Controller {
	real_t time_bank;
	real_t tick_additional_speed;
	uint64_t input_buffers_counter;
	std::deque<FrameSnapshot> frames_snapshot;
	std::vector<uint8_t> cached_packet_data;
	uint64_t recover_snapshot_id;
	uint64_t recovered_snapshot_id;
	Variant recover_state_data;

	PlayerController(CharacterNetController *p_node);

	virtual void physics_process(real_t p_delta);
	virtual void receive_snapshots(Vector<uint8_t> p_data);
	virtual void player_state_check(uint64_t p_snapshot_id, Variant p_data);
	virtual void replay_snapshots();
	virtual uint64_t get_current_snapshot_id() const;

	real_t get_pretended_delta() const;

	void store_input_buffer(uint64_t p_id);

	/// Sends an unreliable packet to the server, containing a packed array of
	/// frame snapshots.
	void send_frame_input_buffer_to_server();

	void process_recovery();

	void receive_tick_additional_speed(int p_speed);
};

/// The doll controller is kind of special controller, it's using a
/// `ServerController` + `MastertController`.
/// The `DollController` receives inputs from the client as the server does,
/// and fetch them exactly like the server.
/// After the execution of the inputs, the puppet start to act like the player,
/// because it wait the player status from the server to correct its motion.
///
/// There are some extra features available that allow the doll to stay in sync
/// with the server execution (see `soft_reset_to_server_state`) and the possibility
/// for the server to stop the data streaming.
struct DollController : public Controller {
	/// Used to perform server like operations
	ServerController server_controller;
	/// Used to perform master like operations
	PlayerController player_controller;
	bool is_server_communication_detected;
	bool is_server_state_update_received;
	bool is_flow_open;

	DollController(CharacterNetController *p_node);

	virtual void physics_process(real_t p_delta);
	virtual void receive_snapshots(Vector<uint8_t> p_data);
	virtual void player_state_check(uint64_t p_snapshot_id, Variant p_data);
	virtual void replay_snapshots();
	virtual uint64_t get_current_snapshot_id() const;

	void open_flow();
	void close_flow();

	/// Make sure the server and the client are is sync by dropping all the
	/// snapshots when a new state update invalidate these.
	void soft_reset_to_server_state();
	void hard_reset_to_server_state();
};

/// This controller is used when the game instance is not a peer of any kind.
/// This controller keeps the workflow as usual so it's possible to use the
/// `CharacterNetController` even without network.
struct NoNetController : public Controller {
	uint64_t frame_id;

	NoNetController(CharacterNetController *p_node);

	virtual void physics_process(real_t p_delta);
	virtual void receive_snapshots(Vector<uint8_t> p_data);
	virtual void player_state_check(uint64_t p_snapshot_id, Variant p_data);
	virtual void replay_snapshots();
	virtual uint64_t get_current_snapshot_id() const;
};

#endif
