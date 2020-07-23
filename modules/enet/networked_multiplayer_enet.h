/*************************************************************************/
/*  networked_multiplayer_enet.h                                         */
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

#ifndef NETWORKED_MULTIPLAYER_ENET_H
#define NETWORKED_MULTIPLAYER_ENET_H

#include "core/crypto/crypto.h"
#include "core/io/compression.h"
#include "core/io/networked_multiplayer_peer.h"

#include <enet/enet.h>

class NetworkedMultiplayerENet : public NetworkedMultiplayerPeer {
	GDCLASS(NetworkedMultiplayerENet, NetworkedMultiplayerPeer);

public:
	enum CompressionMode {
		COMPRESS_NONE,
		COMPRESS_RANGE_CODER,
		COMPRESS_FASTLZ,
		COMPRESS_ZLIB,
		COMPRESS_ZSTD
	};

private:
	enum {
		SYSMSG_ADD_PEER,
		SYSMSG_REMOVE_PEER
	};

	enum {
		SYSCH_CONFIG,
		SYSCH_RELIABLE,
		SYSCH_UNRELIABLE,
		SYSCH_MAX
	};

	bool active;
	bool server;

	uint32_t unique_id;

	int target_peer;
	TransferMode transfer_mode;
	int transfer_channel;
	int channel_count;
	bool always_ordered;

	ENetEvent event;
	ENetPeer *peer;
	ENetHost *host;

	bool refuse_connections;
	bool server_relay;

	ConnectionStatus connection_status;

	Map<int, ENetPeer *> peer_map;

	struct Packet {
		ENetPacket *packet;
		int from;
		int channel;
	};

	CompressionMode compression_mode;

	List<Packet> incoming_packets;

	Packet current_packet;

	uint32_t _gen_unique_id() const;
	void _pop_current_packet();

	Vector<uint8_t> src_compressor_mem;
	Vector<uint8_t> dst_compressor_mem;

	ENetCompressor enet_compressor;
	static size_t enet_compress(void *context, const ENetBuffer *inBuffers, size_t inBufferCount, size_t inLimit, enet_uint8 *outData, size_t outLimit);
	static size_t enet_decompress(void *context, const enet_uint8 *inData, size_t inLimit, enet_uint8 *outData, size_t outLimit);
	static void enet_compressor_destroy(void *context);
	void _setup_compressor();

	IP_Address bind_ip;

	bool dtls_enabled;
	Ref<CryptoKey> dtls_key;
	Ref<X509Certificate> dtls_cert;
	bool dtls_verify;

protected:
	static void _bind_methods();

public:
	virtual void set_transfer_mode(TransferMode p_mode) override;
	virtual TransferMode get_transfer_mode() const override;
	virtual void set_target_peer(int p_peer) override;

	virtual int get_packet_peer() const override;

	virtual IP_Address get_peer_address(int p_peer_id) const;
	virtual int get_peer_port(int p_peer_id) const;

	Error create_server(int p_port, int p_max_clients = 32, int p_in_bandwidth = 0, int p_out_bandwidth = 0);
	Error create_client(const String &p_address, int p_port, int p_in_bandwidth = 0, int p_out_bandwidth = 0, int p_client_port = 0);

	void close_connection(uint32_t wait_usec = 100);

	void disconnect_peer(int p_peer, bool now = false);

	virtual void poll() override;

	virtual bool is_server() const override;

	virtual int get_available_packet_count() const override;
	virtual Error get_packet(const uint8_t **r_buffer, int &r_buffer_size) override; ///< buffer is GONE after next get_packet
	virtual Error put_packet(const uint8_t *p_buffer, int p_buffer_size) override;

	virtual int get_max_packet_size() const override;

	virtual ConnectionStatus get_connection_status() const override;

	virtual void set_refuse_new_connections(bool p_enable) override;
	virtual bool is_refusing_new_connections() const override;

	virtual int get_unique_id() const override;

	void set_compression_mode(CompressionMode p_mode);
	CompressionMode get_compression_mode() const;

	int get_packet_channel() const;
	int get_last_packet_channel() const;
	void set_transfer_channel(int p_channel);
	int get_transfer_channel() const;
	void set_channel_count(int p_channel);
	int get_channel_count() const;
	void set_always_ordered(bool p_ordered);
	bool is_always_ordered() const;
	void set_server_relay_enabled(bool p_enabled);
	bool is_server_relay_enabled() const;

	/// Reliable channel packet loss.
	real_t get_packet_loss(int p_peer) const;
	real_t get_packet_loss_variance(int p_peer) const;
	/// The epoch of the current packet loss.
	uint32_t get_packet_loss_epoch(int p_peer) const;

	/// Mean - Time taken for a reliable packet to do a round trip in ms.
	uint32_t get_round_trip_time(int p_peer) const;
	uint32_t get_round_trip_time_variance(int p_peer) const;
	/// Last packet - Time taken for a reliable packet to do a round trip in ms.
	uint32_t get_last_round_trip_time(int p_peer) const;
	uint32_t get_last_round_trip_time_variance(int p_peer) const;

	uint32_t get_packet_throttle(int p_peer) const;
	uint32_t get_packet_throttle_limit(int p_peer) const;
	uint32_t get_packet_throttle_counter(int p_peer) const;
	uint32_t get_packet_throttle_epoch(int p_peer) const;
	uint32_t get_packet_throttle_acceleration(int p_peer) const;
	uint32_t get_packet_throttle_deceleration(int p_peer) const;
	uint32_t get_packet_throttle_interval(int p_peer) const;

	/// Returns the total data sent from last time this function was called.
	uint32_t pop_total_sent_data();
	/// Returns the total packets sent from last time this function was called.
	uint32_t pop_total_sent_packets();
	/// Returns the total data received from last time this function was called.
	uint32_t pop_total_received_data();
	/// Returns the total packets received from last time this function was called.
	uint32_t pop_total_received_packets();

	/// Configures throttle parameter for a peer.
	///
	/// Unreliable packets are dropped by ENet in response to the varying conditions
	/// of the Internet connection to the peer.  The throttle represents a probability
	/// that an unreliable packet should not be dropped and thus sent by ENet to the peer.
	/// The lowest mean round trip time from the sending of a reliable packet to the
	/// receipt of its acknowledgement is measured over an amount of time specified by
	/// the interval parameter in milliseconds.  If a measured round trip time happens to
	/// be significantly less than the mean round trip time measured over the interval,
	/// then the throttle probability is increased to allow more traffic by an amount
	/// specified in the acceleration parameter, which is a ratio to the ENET_PEER_PACKET_THROTTLE_SCALE
	/// constant.  If a measured round trip time happens to be significantly greater than
	/// the mean round trip time measured over the interval, then the throttle probability
	/// is decreased to limit traffic by an amount specified in the deceleration parameter, which
	/// is a ratio to the ENET_PEER_PACKET_THROTTLE_SCALE constant.  When the throttle has
	/// a value of ENET_PEER_PACKET_THROTTLE_SCALE, no unreliable packets are dropped by
	/// ENet, and so 100% of all unreliable packets will be sent.  When the throttle has a
	/// value of 0, all unreliable packets are dropped by ENet, and so 0% of all unreliable
	/// packets will be sent.  Intermediate values for the throttle represent intermediate
	/// probabilities between 0% and 100% of unreliable packets being sent.  The bandwidth
	/// limits of the local and foreign hosts are taken into account to determine a
	/// sensible limit for the throttle probability above which it should not raise even in
	/// the best of conditions.
	///
	/// @param peer peer to configure.
	/// @param interval interval, in milliseconds, over which to measure lowest mean RTT; the default value is ENET_PEER_PACKET_THROTTLE_INTERVAL.
	/// @param acceleration rate at which to increase the throttle probability as mean RTT declines
	/// @param deceleration rate at which to decrease the throttle probability as mean RTT increases
	void configure_peer_throttle(int p_peer, uint32_t p_interval, uint32_t p_acceleration, uint32_t p_deceleration);

	NetworkedMultiplayerENet();
	~NetworkedMultiplayerENet();

	void set_bind_ip(const IP_Address &p_ip);
	void set_dtls_enabled(bool p_enabled);
	bool is_dtls_enabled() const;
	void set_dtls_verify_enabled(bool p_enabled);
	bool is_dtls_verify_enabled() const;
	void set_dtls_key(Ref<CryptoKey> p_key);
	void set_dtls_certificate(Ref<X509Certificate> p_cert);
};

VARIANT_ENUM_CAST(NetworkedMultiplayerENet::CompressionMode);

#endif // NETWORKED_MULTIPLAYER_ENET_H
