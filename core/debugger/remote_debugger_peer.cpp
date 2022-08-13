/*************************************************************************/
/*  remote_debugger_peer.cpp                                             */
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

#include "remote_debugger_peer.h"

#include "core/config/project_settings.h"
#include "core/io/marshalls.h"
#include "core/os/os.h"

bool RemoteDebuggerPeerTCP::is_peer_connected() {
	return connected;
}

bool RemoteDebuggerPeerTCP::has_message(int p_channel) {
	MutexLock<Mutex> lock(channels[p_channel].shared.mutex);
	return channels[p_channel].shared.in_queue.size() > 0;
}

Array RemoteDebuggerPeerTCP::get_message(int p_channel) {
	MutexLock<Mutex> lock(channels[p_channel].shared.mutex);
	ERR_FAIL_COND_V(!has_message(p_channel), Array());
	Array out = channels[p_channel].shared.in_queue[0];
	channels[p_channel].shared.in_queue.pop_front();
	return out;
}

Error RemoteDebuggerPeerTCP::put_message(int p_channel, const Array &p_arr) {
	MutexLock lock(channels[p_channel].shared.mutex);
	if (channels[p_channel].shared.out_queue.size() >= max_queued_messages) {
		return ERR_OUT_OF_MEMORY;
	}
	channels[p_channel].shared.out_queue.push_back(p_arr);
	return OK;
}

int RemoteDebuggerPeerTCP::get_max_message_size() const {
	return MAX_MESSAGE_SIZE; // 8 MiB
}

void RemoteDebuggerPeerTCP::_close() {
	// FIXME non volatile bool may never be delivered
	running = false;
	thread.wait_to_finish();
	tcp_client->disconnect_from_host();
	for (Channel &channel : channels) {
		channel.out_buf.clear();
		channel.in_buf.clear();
	}
}

void RemoteDebuggerPeerTCP::close() {
	_close();
}

RemoteDebuggerPeerTCP::RemoteDebuggerPeerTCP(Ref<StreamPeerTCP> p_tcp) {
	for (Channel &channel : channels) {
		channel.in_buf.resize(MAX_MESSAGE_SIZE + 4); // Plus packet size | channel ID combo.
		channel.out_buf.resize(MAX_MESSAGE_SIZE + 4); // Plus packet size | channel ID combo.
	}
	tcp_client = p_tcp;
	if (tcp_client.is_valid()) { // Attaching to an already connected stream.
		connected = true;
#ifndef NO_THREADS
		running = true;
		thread.start(_thread_func, this);
#endif
	} else {
		tcp_client.instantiate();
	}
}

RemoteDebuggerPeerTCP::~RemoteDebuggerPeerTCP() {
	_close();
}

void RemoteDebuggerPeerTCP::_write_out() {
	while (tcp_client->get_status() == StreamPeerTCP::STATUS_CONNECTED && tcp_client->wait(NetSocket::POLL_TYPE_OUT) == OK) {
		if (current_write_channel < 0) {
			// check in priority order
			for (int channel_index : { CHANNEL_OTHER, CHANNEL_MAIN_THREAD, CHANNEL_DISCARDABLE }) {
				MutexLock<Mutex> lock(channels[channel_index].shared.mutex);
				if (channels[channel_index].shared.out_queue.size() == 0) {
					continue;
				}
				current_write_channel = channel_index;
				break;
			}
		}
		if (current_write_channel < 0) {
			// Nothing to do.
			return;
		}

		uint8_t *buf = channels[current_write_channel].out_buf.ptrw();

		if (channels[current_write_channel].out_left <= 0) {
			// Need to start next buffer
			channels[current_write_channel].shared.mutex.lock();
			Variant var = channels[current_write_channel].shared.out_queue[0];
			channels[current_write_channel].shared.out_queue.pop_front();
			channels[current_write_channel].shared.mutex.unlock();
			int size = 0;
			const int OVERHEAD = 4;
			Error err = encode_variant(var, nullptr, size);
			if (err != OK || size > channels[current_write_channel].out_buf.size() - OVERHEAD) {
				// Can't send, but we did service this item.
				WARN_PRINT(vformat("Failed to send debugger message to channel %d; error %d.", current_write_channel, err));
				current_write_channel = -1;
				continue;
			}
			static_assert(NUM_CHANNELS < 256);
			static_assert(MAX_MESSAGE_SIZE < (1 << 24));
			encode_uint32(size | (current_write_channel << 24), buf);
			encode_variant(var, buf + OVERHEAD, size);
			channels[current_write_channel].out_left = size + OVERHEAD;
			channels[current_write_channel].out_pos = 0;
		}

		int sent = 0;
		tcp_client->put_partial_data(buf + channels[current_write_channel].out_pos, channels[current_write_channel].out_left, sent);
		channels[current_write_channel].out_left -= sent;
		channels[current_write_channel].out_pos += sent;

		if (channels[current_write_channel].out_left <= 0) {
			// Done with the current transmission.
			current_write_channel = -1;
		}
	}
}

void RemoteDebuggerPeerTCP::_read_in() {
	while (tcp_client->get_status() == StreamPeerTCP::STATUS_CONNECTED && tcp_client->wait(NetSocket::POLL_TYPE_IN) == OK) {
		if (current_read_channel < 0) {
			// Could be reading any channel, so don't do it if any of them are blocked.
			// REVISIT: this isn't completely correct.  If Main is blocked on thread.join, it will never respond to any
			// requests and we need to give up, so we do need per-channel flow control, i.e. multiple sockets or our
			// own flow control (e.g. enet) or just conceptually tear down the Main connection and discard all Main
			// messages.
			int index = 0;
			for (Channel &channel : channels) {
				MutexLock<Mutex> lock(channel.shared.mutex);
				if (channel.in_left <= 0) {
					if (channel.shared.in_queue.size() > max_queued_messages) {
						if (index == CHANNEL_DISCARDABLE) {
							// Drop oldest message, since it is the most stale.
							channel.shared.in_queue.pop_front();
						} else {
							return; // Too many messages already in queue.
						}
					}
					if (tcp_client->get_available_bytes() < 4) {
						return; // Need 4 more bytes.
					}
				}
				++index;
			}
			uint32_t size = 0;
			int read = 0;
			const Error err = tcp_client->get_partial_data(reinterpret_cast<uint8_t *>(&size), 4, read);
			ERR_CONTINUE(read != 4 || err != OK);
			current_read_channel = static_cast<int>(size >> 24U & 0xffU);
			size = size & 0xffffffU;
			if (current_read_channel < 0 || current_read_channel >= NUM_CHANNELS) {
				WARN_PRINT(vformat("Ignored message with invalid channel ID: %d out of 0..%d", current_read_channel, NUM_CHANNELS - 1));
				current_read_channel = -1;
				continue;
			}
			ERR_CONTINUE(size > static_cast<uint32_t>(channels[current_read_channel].in_buf.size()));
			channels[current_read_channel].in_left = size;
			channels[current_read_channel].in_pos = 0;
		}
		uint8_t *buf = channels[current_read_channel].in_buf.ptrw();
		int read = 0;
		tcp_client->get_partial_data(buf + channels[current_read_channel].in_pos, channels[current_read_channel].in_left, read);
		channels[current_read_channel].in_left -= read;
		channels[current_read_channel].in_pos += read;
		if (channels[current_read_channel].in_left == 0) {
			// Restart with size marker, even if we error out below.
			const int was_reading_channel = current_read_channel;
			current_read_channel = -1;

			Variant var;
			const Error err = decode_variant(var, buf, channels[was_reading_channel].in_pos, &read);
			ERR_CONTINUE(read != channels[was_reading_channel].in_pos || err != OK);
			ERR_CONTINUE_MSG(var.get_type() != Variant::ARRAY, "Malformed packet received, not an Array.");
			MutexLock<Mutex> lock(channels[was_reading_channel].shared.mutex);
			channels[was_reading_channel].shared.in_queue.push_back(var);
		}
	}
}

Error RemoteDebuggerPeerTCP::connect_to_host(const String &p_host, uint16_t p_port) {
	IPAddress ip;
	if (p_host.is_valid_ip_address()) {
		ip = p_host;
	} else {
		ip = IP::get_singleton()->resolve_hostname(p_host);
	}

	const int port = p_port;

	const int tries = 6;
	const int waits[tries] = { 1, 10, 100, 1000, 1000, 1000 };

	tcp_client->connect_to_host(ip, port);

	for (int i = 0; i < tries; i++) {
		tcp_client->poll();
		if (tcp_client->get_status() == StreamPeerTCP::STATUS_CONNECTED) {
			print_verbose("Remote Debugger: Connected!");
			break;
		} else {
			const int ms = waits[i];
			OS::get_singleton()->delay_usec(ms * 1000);
			print_verbose("Remote Debugger: Connection failed with status: '" + String::num(tcp_client->get_status()) + "', retrying in " + String::num(ms) + " msec.");
		}
	}

	if (tcp_client->get_status() != StreamPeerTCP::STATUS_CONNECTED) {
		ERR_PRINT("Remote Debugger: Unable to connect. Status: " + String::num(tcp_client->get_status()) + ".");
		return FAILED;
	}
	connected = true;
#ifndef NO_THREADS
	running = true;
	thread.start(_thread_func, this);
#endif
	return OK;
}

void RemoteDebuggerPeerTCP::_thread_func(void *p_ud) {
	// Update in time for 144hz monitors
	// FIXME that precision is nonsense, we don't have that kind of resolution from sleeping (ms on Windows, scheduling intervals, etc.)
	const uint64_t min_tick = 6900;
	RemoteDebuggerPeerTCP *peer = static_cast<RemoteDebuggerPeerTCP *>(p_ud);
	while (peer->running && peer->is_peer_connected()) {
		uint64_t ticks_usec = OS::get_singleton()->get_ticks_usec();
		for (int channel_index = 0; channel_index < NUM_CHANNELS; ++channel_index) {
			peer->_poll(channel_index);
		}
		if (!peer->is_peer_connected()) {
			break;
		}
		ticks_usec = OS::get_singleton()->get_ticks_usec() - ticks_usec;
		if (ticks_usec < min_tick) {
			OS::get_singleton()->delay_usec(min_tick - ticks_usec);
		}
	}
}

void RemoteDebuggerPeerTCP::poll(int p_channel) {
#ifdef NO_THREADS
	if (p_channel == CHANNEL_MAIN_THREAD) {
		// Block only for the first channel, because they are actually all the same socket.
		_poll(p_channel);
	}
#else
	(void)p_channel;
#endif
}

void RemoteDebuggerPeerTCP::_poll(const int p_channel) {
	tcp_client->poll();
	if (connected) {
		// In this implementation, we only have one connection on the wire, so
		// we can't send from multiple channels fragmented.  So we have to
		// actually process all channels together.
		if (p_channel == CHANNEL_MAIN_THREAD) {
			_read_in();
			_write_out();
		}
		connected = tcp_client->get_status() == StreamPeerTCP::STATUS_CONNECTED;
	}
}

RemoteDebuggerPeer *RemoteDebuggerPeerTCP::create(const String &p_uri) {
	ERR_FAIL_COND_V(!p_uri.begins_with("tcp://"), nullptr);

	String debug_host = p_uri.replace("tcp://", "");
	uint16_t debug_port = 6007;

	if (debug_host.contains(":")) {
		const int sep_pos = debug_host.rfind(":");
		debug_port = debug_host.substr(sep_pos + 1).to_int();
		debug_host = debug_host.substr(0, sep_pos);
	}

	RemoteDebuggerPeerTCP *peer = memnew(RemoteDebuggerPeerTCP);
	const Error err = peer->connect_to_host(debug_host, debug_port);
	if (err != OK) {
		memdelete(peer);
		return nullptr;
	}
	return peer;
}

RemoteDebuggerPeer::RemoteDebuggerPeer() {
	max_queued_messages = (int)GLOBAL_GET("network/limits/debugger/max_queued_messages");
}
