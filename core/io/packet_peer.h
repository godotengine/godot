/*************************************************************************/
/*  packet_peer.h                                                        */
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

#ifndef PACKET_PEER_H
#define PACKET_PEER_H

#include "core/io/stream_peer.h"
#include "core/object.h"
#include "core/ring_buffer.h"

class PacketPeer : public Reference {
	GDCLASS(PacketPeer, Reference);

	Variant _bnd_get_var(bool p_allow_objects = false);

	static void _bind_methods();

	Error _put_packet(const PoolVector<uint8_t> &p_buffer);
	PoolVector<uint8_t> _get_packet();
	Error _get_packet_error() const;

	mutable Error last_get_error;

	bool allow_object_decoding;

	int encode_buffer_max_size;
	PoolVector<uint8_t> encode_buffer;

public:
	virtual int get_available_packet_count() const = 0;
	virtual Error get_packet(const uint8_t **r_buffer, int &r_buffer_size) = 0; ///< buffer is GONE after next get_packet
	virtual Error put_packet(const uint8_t *p_buffer, int p_buffer_size) = 0;

	virtual int get_max_packet_size() const = 0;

	/* helpers / binders */

	virtual Error get_packet_buffer(PoolVector<uint8_t> &r_buffer);
	virtual Error put_packet_buffer(const PoolVector<uint8_t> &p_buffer);

	virtual Error get_var(Variant &r_variant, bool p_allow_objects = false);
	virtual Error put_var(const Variant &p_packet, bool p_full_objects = false);

	void set_allow_object_decoding(bool p_enable);
	bool is_object_decoding_allowed() const;

	void set_encode_buffer_max_size(int p_max_size);
	int get_encode_buffer_max_size() const;

	PacketPeer();
	~PacketPeer() {}
};

class PacketPeerStream : public PacketPeer {
	GDCLASS(PacketPeerStream, PacketPeer);

	//the way the buffers work sucks, will change later

	mutable Ref<StreamPeer> peer;
	mutable RingBuffer<uint8_t> ring_buffer;
	mutable Vector<uint8_t> input_buffer;
	mutable Vector<uint8_t> output_buffer;

	Error _poll_buffer() const;

protected:
	void _set_stream_peer(REF p_peer);
	static void _bind_methods();

public:
	virtual int get_available_packet_count() const;
	virtual Error get_packet(const uint8_t **r_buffer, int &r_buffer_size);
	virtual Error put_packet(const uint8_t *p_buffer, int p_buffer_size);

	virtual int get_max_packet_size() const;

	void set_stream_peer(const Ref<StreamPeer> &p_peer);
	Ref<StreamPeer> get_stream_peer() const;
	void set_input_buffer_max_size(int p_max_size);
	int get_input_buffer_max_size() const;
	void set_output_buffer_max_size(int p_max_size);
	int get_output_buffer_max_size() const;
	PacketPeerStream();
};

#endif // PACKET_STREAM_H
