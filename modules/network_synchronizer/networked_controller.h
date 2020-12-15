/*************************************************************************/
/*  networked_controller.h                                               */
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

#include "core/math/transform.h"
#include "core/node_path.h"
#include "data_buffer.h"
#include "interpolator.h"
#include "net_utilities.h"
#include <deque>
#include <vector>

#ifndef NETWORKED_CONTROLLER_H
#define NETWORKED_CONTROLLER_H

class SceneSynchronizer;
struct Controller;
struct ServerController;
struct PlayerController;
struct DollController;
struct NoNetController;

/// The `NetworkedController` is responsible to sync the `Player` inputs between
/// the peers. This allows to control a character, or an object with high precision
/// and replicates that movement on all connected peers.
///
/// The `NetworkedController` will sync inputs, based on those will perform
/// operations.
/// The result of these operations, are guaranteed to be the same accross the
/// peers, if we stay under the assumption that the initial state is the same.
///
/// Is possible to use the `SceneSynchronizer` to keep the state in sync with the
/// peers.
///
// # Implementation details
//
// The `NetworkedController` perform different operations depending where it's
// instantiated.
// The most important part is inside the `PlayerController`, `ServerController`,
// `DollController`, `NoNetController`.
class NetworkedController : public Node {
	GDCLASS(NetworkedController, Node);

	friend class SceneSynchronizer;

public:
	enum ControllerType {
		CONTROLLER_TYPE_NULL,
		CONTROLLER_TYPE_NONETWORK,
		CONTROLLER_TYPE_PLAYER,
		CONTROLLER_TYPE_SERVER,
		CONTROLLER_TYPE_DOLL
	};

private:
	/// The input storage size is used to cap the amount of inputs collected by
	/// the `PlayerController`.
	///
	/// The server sends a message, to all the connected peers, notifing its
	/// status at a fixed interval.
	/// The peers, after receiving this update, removes all the old inputs until
	/// that moment.
	///
	/// `input_storage_size`: is too small, the player may stop collect
	/// - Too small value makes the `PlayerController` stop collecting inputs
	///   too early, in case of lag.
	/// - Too big values may introduce too much latency, because the player keep
	///   pushing new inputs without receiving the server snapshot.
	///
	/// With 60 iteration per seconds a good value is `180` (60 * 3) so the
	/// `PlayerController` can be at max 3 seconds ahead the `ServerController`.
	int player_input_storage_size = 180;

	/// Amount of time an inputs is re-sent to each peer.
	/// Resenging inputs is necessary because the packets may be lost since as
	/// they are sent in an unreliable way.
	int max_redundant_inputs = 5;

	/// Time in seconds between each `tick_speedup` that the server sends to the
	/// client.
	real_t tick_speedup_notification_delay = 0.33;

	/// The connection quality is established by watching the time passed
	/// between each input is received.
	/// The more this time is the same the more the connection health is good.
	///
	/// The `network_traced_frames` defines how many frames have
	/// to be used to establish the connection quality.
	/// - Big values make the mechanism too slow.
	/// - Small values make the mechanism too sensible.
	int network_traced_frames = 120;

	/// Sensitivity to network oscillations. The value is in seconds and can be
	/// used to establish the connection quality.
	///
	/// For each input, the time needed for its arrival is traced; the standard
	/// deviation of these is divided by `net_sensitivity`: the result is
	/// the connection quality.
	///
	/// The more the time needed for each batch to arrive is different the
	/// bigger this value is: when this value approaches to
	/// `net_sensitivity` (or even surpasses it) the bad the connection is.
	///
	/// The result is the `connection_poorness` that goes from 0 to 1 and is
	/// used to decide the `optimal_frame_delay` that is interpolated between
	/// `min_frames_delay` and `max_frames_delay`.
	real_t net_sensitivity = 0.1;

	/// The `ServerController` will try to keep a margin of error, so that
	/// network oscillations doesn't leave the `ServerController` without
	/// inputs.
	///
	/// This margin of error is called `optimal_frame_delay` and it changes
	/// depending on the connection health:
	/// it can go from `min_frames_delay` to `max_frames_delay`.
	int min_frames_delay = 1;
	int max_frames_delay = 6;

	/// Rate at which the tick speed changes, so the `optimal_frame_delay` is
	/// matched.
	real_t tick_acceleration = 2.0;

	/// Collect rate (in frames) used by the server to estabish when to collect
	/// the state for a particular peer.
	/// It's possible to scale down this rate, for a particular peer,
	/// using the function: set_doll_collect_rate_factor(peer, factor);
	/// Current default is 10Hz.
	///
	/// The collected state is not immediatelly sent to the clients, rather it's
	/// delayed so to be sent in batch. The states marked as important are
	/// always collected.
	int doll_epoch_collect_rate = 1;

	/// The time rate at which a new batch is sent.
	real_t doll_epoch_batch_sync_rate = 0.25;

	/// The doll interpolator will try to keep a margin of error, so that network
	/// oscillations doesn't make the dolls freeze.
	///
	/// This margin of error is called `optimal_frame_delay` and it changes
	/// depending on the connection health:
	/// it can go from `doll_min_frames_delay` to `doll_max_frames_delay`.
	int doll_min_frames_delay = 1;
	int doll_max_frames_delay = 5;

	/// Max speedup / slowdown the doll can apply to recover its epoch buffer size.
	real_t doll_interpolation_max_speedup = 0.2;

	/// The connection quality is established by watching the time passed
	/// between each batch arrival.
	/// The more this time is the same the more the connection health is good.
	///
	/// The `doll_connection_stats_frame_span` defines how many frames have
	/// to be used to establish the connection quality.
	/// - Big values make the mechanism too slow.
	/// - Small values make the mechanism too sensible.
	/// The correct value should be give considering the
	/// `doll_epoch_batch_sync_rate`.
	int doll_connection_stats_frame_span = 30;

	/// Sensitivity to network oscillations. The value is in seconds and can be
	/// used to establish the connection quality.
	///
	/// For each batch, the time needed for its arrival is traced; the standard
	/// deviation of these is divided by `doll_net_sensitivity`: the result is
	/// the connection quality.
	///
	/// The more the time needed for each batch to arrive is different the
	/// bigger this value is: when this value approaches to
	/// `doll_net_sensitivity` (or even surpasses it) the bad the connection is.
	///
	/// The result is the `connection_poorness` that goes from 0 to 1 and is
	/// used to decide the `optimal_frames_delay` that is interpolated between
	/// `doll_min_frames_delay` and `doll_max_frames_delay`.
	real_t doll_net_sensitivity = 0.2;

	ControllerType controller_type = CONTROLLER_TYPE_NULL;
	Controller *controller = nullptr;
	DataBuffer inputs_buffer;

	SceneSynchronizer *scene_synchronizer = nullptr;

	bool packet_missing = false;
	bool has_player_new_input = false;

public:
	static void _bind_methods();

public:
	NetworkedController();

	void set_player_input_storage_size(int p_size);
	int get_player_input_storage_size() const;

	void set_max_redundant_inputs(int p_max);
	int get_max_redundant_inputs() const;

	void set_tick_speedup_notification_delay(real_t p_delay);
	real_t get_tick_speedup_notification_delay() const;

	void set_network_traced_frames(int p_size);
	int get_network_traced_frames() const;

	void set_min_frames_delay(int p_val);
	int get_min_frames_delay() const;

	void set_max_frames_delay(int p_val);
	int get_max_frames_delay() const;

	void set_net_sensitivity(real_t p_val);
	real_t get_net_sensitivity() const;

	void set_tick_acceleration(real_t p_acceleration);
	real_t get_tick_acceleration() const;

	void set_doll_epoch_collect_rate(int p_rate);
	int get_doll_epoch_collect_rate() const;

	void set_doll_epoch_batch_sync_rate(real_t p_rate);
	real_t get_doll_epoch_batch_sync_rate() const;

	void set_doll_min_frames_delay(int p_min);
	int get_doll_min_frames_delay() const;

	void set_doll_max_frames_delay(int p_max);
	int get_doll_max_frames_delay() const;

	void set_doll_interpolation_max_speedup(real_t p_speedup);
	real_t get_doll_interpolation_max_speedup() const;

	void set_doll_connection_stats_frame_span(int p_span);
	int get_doll_connection_stats_frame_span() const;

	void set_doll_net_sensitivity(real_t p_sensitivity);
	real_t get_doll_net_sensitivity() const;

	uint32_t get_current_input_id() const;

	const DataBuffer &get_inputs_buffer() const {
		return inputs_buffer;
	}

	DataBuffer &get_inputs_buffer_mut() {
		return inputs_buffer;
	}

	/// Returns the pretended delta used by the player.
	real_t player_get_pretended_delta(uint32_t p_iterations_per_seconds) const;

	void mark_epoch_as_important();

	void set_doll_collect_rate_factor(int p_peer, real_t p_factor);
	void set_doll_peer_active(int p_peer_id, bool p_active);
	void pause_notify_dolls();

	bool process_instant(int p_i, real_t p_delta);

	/// Returns the server controller or nullptr if this is not a server.
	ServerController *get_server_controller();
	const ServerController *get_server_controller() const;
	/// Returns the player controller or nullptr if this is not a player.
	PlayerController *get_player_controller();
	const PlayerController *get_player_controller() const;
	/// Returns the doll controller or nullptr if this is not a doll.
	DollController *get_doll_controller();
	const DollController *get_doll_controller() const;
	/// Returns the no net controller or nullptr if this is not a no net.
	NoNetController *get_nonet_controller();
	const NoNetController *get_nonet_controller() const;

	bool is_server_controller() const;
	bool is_player_controller() const;
	bool is_doll_controller() const;
	bool is_nonet_controller() const;

public:
	void set_inputs_buffer(const BitArray &p_new_buffer, uint32_t p_metadata_size_in_bit, uint32_t p_size_in_bit);

	void set_scene_synchronizer(SceneSynchronizer *p_synchronizer);
	SceneSynchronizer *get_scene_synchronizer() const;
	bool has_scene_synchronizer() const;

	/* On server rpc functions. */
	void _rpc_server_send_inputs(Vector<uint8_t> p_data);

	/* On client rpc functions. */
	void _rpc_send_tick_additional_speed(Vector<uint8_t> p_data);

	/* On puppet rpc functions. */
	void _rpc_doll_notify_sync_pause(uint32_t p_epoch);
	void _rpc_doll_send_epoch_batch(Vector<uint8_t> p_data);

	void process(real_t p_delta);

	void player_set_has_new_input(bool p_has);
	bool player_has_new_input() const;

private:
	virtual void _notification(int p_what);
};

struct FrameSnapshot {
	uint32_t id;
	BitArray inputs_buffer;
	uint32_t buffer_size_bit;
	uint32_t similarity;

	bool operator==(const FrameSnapshot &p_other) const {
		return p_other.id == id;
	}
};

struct Controller {
	NetworkedController *node;

	Controller(NetworkedController *p_node) :
			node(p_node) {}

	virtual ~Controller() {}

	virtual void ready() {}
	virtual uint32_t get_current_input_id() const = 0;

	virtual void clear_peers() {}
	virtual void activate_peer(int p_peer) {}
	virtual void deactivate_peer(int p_peer) {}
};

struct ServerController : public Controller {
	struct Peer {
		Peer() = default;
		Peer(int p_peer) :
				peer(p_peer) {}

		int peer = 0;
		bool active = true;
		real_t update_rate_factor = 1.0;
		int collect_timer = 0; // In frames
		int collect_threshold = 0; // In frames
		LocalVector<Vector<uint8_t>> epoch_batch;
		uint32_t batch_size = 0;
	};

	uint32_t current_input_buffer_id = UINT32_MAX;
	uint32_t ghost_input_count = 0;
	uint32_t last_sent_state_input_id = 0;
	real_t client_tick_additional_speed = 0.0;
	real_t additional_speed_notif_timer = 0.0;
	std::deque<FrameSnapshot> snapshots;
	bool streaming_paused = false;
	bool enabled = true;

	uint32_t input_arrival_time = UINT32_MAX;
	NetUtility::StatisticalRingBuffer<uint32_t> network_watcher;

	/// Used to sync the dolls.
	LocalVector<Peer> peers;
	DataBuffer epoch_state_data_cache;
	uint32_t epoch = 0;
	bool is_epoch_important = false;
	real_t batch_sync_timer = 0.0;

	ServerController(
			NetworkedController *p_node,
			int p_traced_frames);

	void process(real_t p_delta);
	uint32_t last_known_input() const;
	virtual uint32_t get_current_input_id() const override;

	void set_enabled(bool p_enable);

	virtual void clear_peers() override;
	virtual void activate_peer(int p_peer) override;
	virtual void deactivate_peer(int p_peer) override;

	void receive_inputs(Vector<uint8_t> p_data);
	int get_inputs_count() const;

	/// Fetch the next inputs, returns true if the input is new.
	bool fetch_next_input();

	void notify_send_state();

	void doll_sync(real_t p_delta);

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
	void calculates_player_tick_rate(real_t p_delta);
	void adjust_player_tick_rate(real_t p_delta);

	uint32_t find_peer(int p_peer) const;
};

struct PlayerController : public Controller {
	uint32_t current_input_id;
	uint32_t input_buffers_counter;
	real_t time_bank;
	real_t tick_additional_speed;
	bool streaming_paused = false;

	std::deque<FrameSnapshot> frames_snapshot;
	std::vector<uint8_t> cached_packet_data;

	PlayerController(NetworkedController *p_node);

	void process(real_t p_delta);
	int calculates_sub_ticks(real_t p_delta, real_t p_iteration_per_seconds);
	int notify_input_checked(uint32_t p_input_id);
	uint32_t last_known_input() const;
	uint32_t get_stored_input_id(int p_i) const;
	virtual uint32_t get_current_input_id() const override;

	bool process_instant(int p_i, real_t p_delta);
	real_t get_pretended_delta(real_t p_iteration_per_second) const;

	void store_input_buffer(uint32_t p_id);

	/// Sends an unreliable packet to the server, containing a packed array of
	/// frame snapshots.
	void send_frame_input_buffer_to_server();

	bool can_accept_new_inputs() const;
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
	Interpolator interpolator;
	real_t additional_speed = 0.0;
	uint32_t current_epoch = UINT32_MAX;
	real_t advancing_epoch = 0.0;
	uint32_t missing_epochs = 0;
	// Any received epoch prior to this one is discarded.
	uint32_t paused_epoch = 0;

	// Used to track the time taken for the next batch to arrive.
	uint32_t batch_receiver_timer = UINT32_MAX;
	/// Used to track how network is performing.
	NetUtility::StatisticalRingBuffer<uint32_t> network_watcher;

	DollController(NetworkedController *p_node);
	~DollController();

	virtual void ready() override;
	void process(real_t p_delta);
	// TODO consider make this non virtual
	virtual uint32_t get_current_input_id() const override;

	void receive_batch(Vector<uint8_t> p_data);
	uint32_t receive_epoch(Vector<uint8_t> p_data);

	uint32_t next_epoch(real_t p_delta);
	void pause(uint32_t p_epoch);
};

/// This controller is used when the game instance is not a peer of any kind.
/// This controller keeps the workflow as usual so it's possible to use the
/// `NetworkedController` even without network.
struct NoNetController : public Controller {
	uint32_t frame_id;

	NoNetController(NetworkedController *p_node);

	void process(real_t p_delta);
	virtual uint32_t get_current_input_id() const override;
};

#endif
