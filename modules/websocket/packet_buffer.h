/*************************************************************************/
/*  packet_buffer.h                                                      */
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

#ifndef PACKET_BUFFER_H
#define PACKET_BUFFER_H

#include "core/templates/ring_buffer.h"

template <class T>
class PacketBuffer {
private:
	typedef struct {
		uint32_t size;
		T info;
	} _Packet;

	RingBuffer<_Packet> _packets;
	RingBuffer<uint8_t> _payload;

public:
	Error write_packet(const uint8_t *p_payload, uint32_t p_size, const T *p_info) {
#ifdef TOOLS_ENABLED
		// Verbose buffer warnings
		if (p_payload && _payload.space_left() < (int32_t)p_size) {
			ERR_PRINT("Buffer payload full! Dropping data.");
			ERR_FAIL_V(ERR_OUT_OF_MEMORY);
		}
		if (p_info && _packets.space_left() < 1) {
			ERR_PRINT("Too many packets in queue! Dropping data.");
			ERR_FAIL_V(ERR_OUT_OF_MEMORY);
		}
#else
		ERR_FAIL_COND_V(p_payload && (uint32_t)_payload.space_left() < p_size, ERR_OUT_OF_MEMORY);
		ERR_FAIL_COND_V(p_info && _packets.space_left() < 1, ERR_OUT_OF_MEMORY);
#endif

		// If p_info is nullptr, only the payload is written
		if (p_info) {
			_Packet p;
			p.size = p_size;
			memcpy(&p.info, p_info, sizeof(T));
			_packets.write(p);
		}

		// If p_payload is nullptr, only the packet information is written.
		if (p_payload) {
			_payload.write((const uint8_t *)p_payload, p_size);
		}

		return OK;
	}

	Error read_packet(uint8_t *r_payload, int p_bytes, T *r_info, int &r_read) {
		ERR_FAIL_COND_V(_packets.data_left() < 1, ERR_UNAVAILABLE);
		_Packet p;
		_packets.read(&p, 1);
		ERR_FAIL_COND_V(_payload.data_left() < (int)p.size, ERR_BUG);
		ERR_FAIL_COND_V(p_bytes < (int)p.size, ERR_OUT_OF_MEMORY);

		r_read = p.size;
		memcpy(r_info, &p.info, sizeof(T));
		_payload.read(r_payload, p.size);
		return OK;
	}

	void discard_payload(int p_size) {
		_packets.decrease_write(p_size);
	}

	void resize(int p_pkt_shift, int p_buf_shift) {
		_packets.resize(p_pkt_shift);
		_payload.resize(p_buf_shift);
	}

	int packets_left() const {
		return _packets.data_left();
	}

	void clear() {
		_payload.resize(0);
		_packets.resize(0);
	}

	PacketBuffer() {
		clear();
	}

	~PacketBuffer() {
		clear();
	}
};

#endif // PACKET_BUFFER_H
