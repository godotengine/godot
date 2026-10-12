/**************************************************************************/
/*  debug_transport_v2.cpp                                                */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "debug_transport_v2.h"

#include "core/io/ip.h"
#include "core/io/marshalls.h"
#include "core/os/os.h"

#include <stdint.h>

uint64_t DebugTransportV2::_make_session_id() {
	uint64_t random_bits = 0;
	if (OS::get_singleton()->get_entropy((uint8_t *)&random_bits, sizeof(random_bits)) != OK) {
		random_bits = OS::get_singleton()->get_ticks_usec() * UINT64_C(0x9E3779B97F4A7C15);
	}
	const uint64_t process_bits = (uint64_t)(uint32_t)OS::get_singleton()->get_process_id() << 32;
	uint64_t result = (random_bits ^ process_bits ^ OS::get_singleton()->get_ticks_usec()) & INT64_MAX;
	return result == 0 ? 1 : result;
}

Error DebugTransportV2::_wait_for_connections(uint32_t p_timeout_msec) {
	const uint64_t deadline = OS::get_singleton()->get_ticks_msec() + p_timeout_msec;
	while (OS::get_singleton()->get_ticks_msec() < deadline) {
		control_tcp->poll();
		data_tcp->poll();
		const StreamPeerTCP::Status control_status = control_tcp->get_status();
		const StreamPeerTCP::Status data_status = data_tcp->get_status();
		if (control_status == StreamPeerTCP::STATUS_CONNECTED && data_status == StreamPeerTCP::STATUS_CONNECTED) {
			return OK;
		}
		if (control_status == StreamPeerTCP::STATUS_ERROR || data_status == StreamPeerTCP::STATUS_ERROR) {
			return ERR_CANT_CONNECT;
		}
		OS::get_singleton()->delay_usec(1000);
	}
	return ERR_TIMEOUT;
}

Error DebugTransportV2::_poll(Ref<PacketPeerStream> p_peer, DebugProtocolV2::Channel p_channel, DebugProtocolV2::Envelope &r_message) {
	if (p_peer->get_available_packet_count() == 0) {
		return ERR_UNAVAILABLE;
	}

	Variant packet;
	Error error = p_peer->get_var(packet);
	if (error != OK) {
		return error;
	}

	DebugProtocolV2::Envelope message;
	String decode_error;
	error = DebugProtocolV2::decode(packet, message, decode_error);
	if (error != OK) {
		ERR_PRINT(decode_error);
		return error;
	}
	if (message.session_id != session_id || message.generation != generation || message.channel != p_channel) {
		ERR_PRINT("Debugger v2 received a packet for a different session, generation, or channel.");
		return ERR_INVALID_DATA;
	}

	r_message = message;
	trace.record("receive", p_channel, message, 0, 0, "accepted");
	return OK;
}

Error DebugTransportV2::_wait_for_hello_ack(DebugProtocolV2::Channel p_channel, uint32_t p_timeout_msec) {
	const uint64_t deadline = OS::get_singleton()->get_ticks_msec() + p_timeout_msec;
	while (OS::get_singleton()->get_ticks_msec() < deadline) {
		DebugProtocolV2::Envelope message;
		Error error = p_channel == DebugProtocolV2::CHANNEL_CONTROL ? poll_control(message) : poll_data(message);
		if (error == OK) {
			if (message.type != "hello_ack") {
				ERR_PRINT("Debugger v2 expected hello_ack, received '" + message.type + "'.");
				return ERR_INVALID_DATA;
			}
			if (message.payload.size() >= 3 && message.payload[0].get_type() == Variant::INT && message.payload[1].get_type() == Variant::INT && message.payload[2].get_type() == Variant::INT) {
				heartbeat_interval_msec = MAX(1, (int)message.payload[0]);
				heartbeat_timeout_msec = MAX(heartbeat_interval_msec, (uint32_t)(int)message.payload[1]);
				variable_page_size = MAX(1, (int)message.payload[2]);
			}
			return OK;
		}
		if (error != ERR_UNAVAILABLE) {
			return error;
		}
		OS::get_singleton()->delay_usec(1000);
	}
	return ERR_TIMEOUT;
}

Error DebugTransportV2::connect_to_host(const String &p_host, uint16_t p_port, uint32_t p_timeout_msec) {
	disconnect();
	reconnect_host = p_host;
	reconnect_port = p_port;
	session_id = _make_session_id();
	generation = 1;
	control_sequence = 0;
	data_sequence = 0;

	IPAddress ip;
	if (p_host.is_valid_ip_address()) {
		ip = p_host;
	} else {
		ip = IP::get_singleton()->resolve_hostname(p_host);
	}
	if (!ip.is_valid()) {
		ERR_PRINT("Debugger v2 could not resolve host '" + p_host + "'.");
		return ERR_CANT_RESOLVE;
	}

	Error error = control_tcp->connect_to_host(ip, p_port);
	if (error != OK) {
		return error;
	}
	error = data_tcp->connect_to_host(ip, p_port);
	if (error != OK) {
		disconnect();
		return error;
	}
	error = _wait_for_connections(p_timeout_msec);
	if (error != OK) {
		disconnect();
		return error;
	}

	control_tcp->set_no_delay(true);
	data_tcp->set_no_delay(true);
	control_peer->set_stream_peer(control_tcp);
	data_peer->set_stream_peer(data_tcp);

	Array hello_payload;
	hello_payload.push_back(OS::get_singleton()->get_process_id());
	hello_payload.push_back(heartbeat_interval_msec);
	hello_payload.push_back(variable_page_size);
	if (send_control("control_hello", hello_payload, 1) != OK || send_data("data_hello", hello_payload, 1) != OK || flush_data(1) != 1) {
		disconnect();
		return ERR_CANT_CONNECT;
	}

	const uint64_t handshake_begin = OS::get_singleton()->get_ticks_msec();
	error = _wait_for_hello_ack(DebugProtocolV2::CHANNEL_CONTROL, p_timeout_msec);
	if (error == OK) {
		const uint64_t elapsed = OS::get_singleton()->get_ticks_msec() - handshake_begin;
		const uint32_t remaining = elapsed >= p_timeout_msec ? 0 : p_timeout_msec - elapsed;
		error = _wait_for_hello_ack(DebugProtocolV2::CHANNEL_DATA, remaining);
	}
	if (error != OK) {
		ERR_PRINT("Debugger v2 dual-channel handshake failed.");
		disconnect();
		return error;
	}

	print_verbose("Remote Debugger v2: control and data channels connected.");
	return OK;
}

void DebugTransportV2::disconnect() {
	if (control_tcp.is_valid()) {
		control_tcp->disconnect_from_host();
	}
	if (data_tcp.is_valid()) {
		data_tcp->disconnect_from_host();
	}
	control_peer->set_stream_peer(Ref<StreamPeer>());
	data_peer->set_stream_peer(Ref<StreamPeer>());
	data_queue.clear();
	data_queue_bytes = 0;
	reconnecting = false;
	reconnect_hello_sent = false;
	reconnect_control_ack = false;
	reconnect_data_ack = false;
}

Error DebugTransportV2::start_reconnect() {
	ERR_FAIL_COND_V(reconnect_host.is_empty() || reconnect_port == 0, ERR_UNCONFIGURED);
	disconnect();
	control_tcp = Ref<StreamPeerTCP>(memnew(StreamPeerTCP));
	data_tcp = Ref<StreamPeerTCP>(memnew(StreamPeerTCP));
	generation++;
	if (generation == 0) {
		generation = 1;
	}
	control_sequence = 0;
	data_sequence = 0;

	IPAddress ip;
	if (reconnect_host.is_valid_ip_address()) {
		ip = reconnect_host;
	} else {
		ip = IP::get_singleton()->resolve_hostname(reconnect_host);
	}
	if (!ip.is_valid()) {
		return ERR_CANT_RESOLVE;
	}

	Error error = control_tcp->connect_to_host(ip, reconnect_port);
	if (error == OK) {
		error = data_tcp->connect_to_host(ip, reconnect_port);
	}
	if (error != OK) {
		disconnect();
		return error;
	}

	reconnecting = true;
	reconnect_started_msec = OS::get_singleton()->get_ticks_msec();
	return ERR_BUSY;
}

Error DebugTransportV2::poll_reconnect(uint32_t p_timeout_msec) {
	if (!reconnecting) {
		return is_connected() ? OK : ERR_UNCONFIGURED;
	}
	if (OS::get_singleton()->get_ticks_msec() - reconnect_started_msec > p_timeout_msec) {
		disconnect();
		return ERR_TIMEOUT;
	}

	control_tcp->poll();
	data_tcp->poll();
	const StreamPeerTCP::Status control_status = control_tcp->get_status();
	const StreamPeerTCP::Status data_status = data_tcp->get_status();
	if (control_status == StreamPeerTCP::STATUS_ERROR || data_status == StreamPeerTCP::STATUS_ERROR) {
		disconnect();
		return ERR_CANT_CONNECT;
	}
	if (control_status != StreamPeerTCP::STATUS_CONNECTED || data_status != StreamPeerTCP::STATUS_CONNECTED) {
		return ERR_BUSY;
	}

	if (!reconnect_hello_sent) {
		control_tcp->set_no_delay(true);
		data_tcp->set_no_delay(true);
		control_peer->set_stream_peer(control_tcp);
		data_peer->set_stream_peer(data_tcp);
		Array hello_payload;
		hello_payload.push_back(OS::get_singleton()->get_process_id());
		hello_payload.push_back(heartbeat_interval_msec);
		hello_payload.push_back(variable_page_size);
		if (send_control("control_hello", hello_payload, 1) != OK || send_data("data_hello", hello_payload, 1) != OK || flush_data(1) != 1) {
			disconnect();
			return ERR_CANT_CONNECT;
		}
		reconnect_hello_sent = true;
	}

	if (!reconnect_control_ack) {
		DebugProtocolV2::Envelope message;
		Error error = poll_control(message);
		if (error == OK) {
			if (message.type != "hello_ack") {
				disconnect();
				return ERR_INVALID_DATA;
			}
			reconnect_control_ack = true;
		} else if (error != ERR_UNAVAILABLE) {
			disconnect();
			return error;
		}
	}
	if (!reconnect_data_ack) {
		DebugProtocolV2::Envelope message;
		Error error = poll_data(message);
		if (error == OK) {
			if (message.type != "hello_ack") {
				disconnect();
				return ERR_INVALID_DATA;
			}
			reconnect_data_ack = true;
		} else if (error != ERR_UNAVAILABLE) {
			disconnect();
			return error;
		}
	}

	if (!reconnect_control_ack || !reconnect_data_ack) {
		return ERR_BUSY;
	}
	reconnecting = false;
	print_verbose("Remote Debugger v2: reconnected with generation " + String::num_int64(generation) + ".");
	return OK;
}

Error DebugTransportV2::send_control(const String &p_type, const Array &p_payload, uint64_t p_request_id, uint64_t p_stop_id) {
	ERR_FAIL_COND_V(!is_control_connected(), ERR_CONNECTION_ERROR);
	DebugProtocolV2::Envelope message;
	message.session_id = session_id;
	message.generation = generation;
	message.channel = DebugProtocolV2::CHANNEL_CONTROL;
	message.sequence = ++control_sequence;
	message.request_id = p_request_id;
	message.stop_id = p_stop_id;
	message.type = p_type;
	message.payload = p_payload;
	Error error = control_peer->put_var(DebugProtocolV2::encode(message));
	trace.record("send", DebugProtocolV2::CHANNEL_CONTROL, message, 0, 0, error == OK ? "sent" : "error");
	return error;
}

Error DebugTransportV2::_put_data_immediate(const DebugProtocolV2::Envelope &p_message) {
	Error error = data_peer->put_var(DebugProtocolV2::encode(p_message));
	trace.record("send", DebugProtocolV2::CHANNEL_DATA, p_message, 0, 0, error == OK ? "sent" : "error");
	return error;
}

Error DebugTransportV2::send_data(const String &p_type, const Array &p_payload, uint64_t p_request_id, uint64_t p_stop_id, DataPriority p_priority) {
	ERR_FAIL_COND_V(!is_data_connected(), ERR_CONNECTION_ERROR);
	DebugProtocolV2::Envelope message;
	message.session_id = session_id;
	message.generation = generation;
	message.channel = DebugProtocolV2::CHANNEL_DATA;
	message.sequence = ++data_sequence;
	message.request_id = p_request_id;
	message.stop_id = p_stop_id;
	message.type = p_type;
	message.payload = p_payload;

	int encoded_bytes = 0;
	Error error = encode_variant(DebugProtocolV2::encode(message), nullptr, encoded_bytes);
	if (error != OK) {
		return error;
	}
	encoded_bytes += 4;
	if (encoded_bytes > data_queue_max_bytes) {
		dropped_data_messages++;
		dropped_data_bytes += encoded_bytes;
		trace.record("queue", DebugProtocolV2::CHANNEL_DATA, message, encoded_bytes, 0, "oversized_drop");
		return ERR_OUT_OF_MEMORY;
	}

	while (data_queue.size() >= data_queue_max_messages || data_queue_bytes + encoded_bytes > data_queue_max_bytes) {
		List<QueuedData>::Element *drop = nullptr;
		for (List<QueuedData>::Element *element = data_queue.front(); element; element = element->next()) {
			if (!drop || element->get().priority > drop->get().priority) {
				drop = element;
			}
		}
		if (!drop || drop->get().priority < p_priority) {
			dropped_data_messages++;
			dropped_data_bytes += encoded_bytes;
			trace.record("queue", DebugProtocolV2::CHANNEL_DATA, message, encoded_bytes, 0, "backpressure_drop");
			return ERR_BUSY;
		}
		data_queue_bytes -= drop->get().bytes;
		dropped_data_messages++;
		dropped_data_bytes += drop->get().bytes;
		trace.record("queue", DebugProtocolV2::CHANNEL_DATA, drop->get().message, drop->get().bytes, 0, "backpressure_drop");
		data_queue.erase(drop);
	}

	QueuedData queued;
	queued.message = message;
	queued.priority = p_priority;
	queued.bytes = encoded_bytes;
	queued.enqueued_usec = OS::get_singleton()->get_ticks_usec();
	data_queue.push_back(queued);
	data_queue_bytes += encoded_bytes;
	trace.record("queue", DebugProtocolV2::CHANNEL_DATA, message, encoded_bytes, 0, "queued");
	return OK;
}

int DebugTransportV2::flush_data(int p_max_packets) {
	int flushed = 0;
	while (flushed < p_max_packets && data_queue.size() > 0 && is_data_connected()) {
		List<QueuedData>::Element *selected = data_queue.front();
		for (List<QueuedData>::Element *element = selected->next(); element; element = element->next()) {
			if (element->get().priority < selected->get().priority) {
				selected = element;
			}
		}
		const QueuedData queued = selected->get();
		data_queue_bytes -= queued.bytes;
		data_queue.erase(selected);
		const uint64_t queue_usec = OS::get_singleton()->get_ticks_usec() - queued.enqueued_usec;
		Error error = _put_data_immediate(queued.message);
		trace.record("flush", DebugProtocolV2::CHANNEL_DATA, queued.message, queued.bytes, queue_usec, error == OK ? "sent" : "error");
		if (error != OK) {
			break;
		}
		flushed++;
	}
	return flushed;
}

Error DebugTransportV2::poll_control(DebugProtocolV2::Envelope &r_message) {
	if (control_tcp.is_valid()) {
		control_tcp->poll();
	}
	if (!is_control_connected()) {
		return ERR_CONNECTION_ERROR;
	}
	return _poll(control_peer, DebugProtocolV2::CHANNEL_CONTROL, r_message);
}

Error DebugTransportV2::poll_data(DebugProtocolV2::Envelope &r_message) {
	if (data_tcp.is_valid()) {
		data_tcp->poll();
	}
	if (!is_data_connected()) {
		return ERR_CONNECTION_ERROR;
	}
	return _poll(data_peer, DebugProtocolV2::CHANNEL_DATA, r_message);
}

bool DebugTransportV2::is_control_connected() const {
	return control_tcp.is_valid() && control_tcp->get_status() == StreamPeerTCP::STATUS_CONNECTED;
}

bool DebugTransportV2::is_data_connected() const {
	return data_tcp.is_valid() && data_tcp->get_status() == StreamPeerTCP::STATUS_CONNECTED;
}

bool DebugTransportV2::is_connected() const {
	return !reconnecting && is_control_connected() && is_data_connected();
}

DebugTransportV2::DebugTransportV2() :
		control_tcp(Ref<StreamPeerTCP>(memnew(StreamPeerTCP))),
		control_peer(Ref<PacketPeerStream>(memnew(PacketPeerStream))),
		data_tcp(Ref<StreamPeerTCP>(memnew(StreamPeerTCP))),
		data_peer(Ref<PacketPeerStream>(memnew(PacketPeerStream))) {
	control_peer->set_output_buffer_max_size((1024 * 1024) - 4);
	data_peer->set_output_buffer_max_size((1024 * 1024 * 8) - 4);
}
