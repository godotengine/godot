/**************************************************************************/
/*  webrtc_lib_peer_connection.h                                          */
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

#ifndef WEBRTC_LIB_PEER_CONNECTION_H
#define WEBRTC_LIB_PEER_CONNECTION_H

#ifdef ENABLE_LIBDATACHANNEL

#include "core/os/mutex.h"
#include "core/templates/vector.h"
#include "webrtc_peer_connection.h"

#include "rtc/rtc.hpp"

class WebRTCLibPeerConnection : public WebRTCPeerConnection {
	GDCLASS(WebRTCLibPeerConnection, WebRTCPeerConnection);

private:
	std::shared_ptr<rtc::PeerConnection> peer_connection = nullptr;

	Error _create_pc(rtc::Configuration &r_config);
	Error _parse_ice_server(rtc::Configuration &r_config, Dictionary p_server);
	Error _parse_channel_config(rtc::DataChannelInit &r_config, const Dictionary &p_dict);

protected:
	static void _bind_methods() {}

public:
	ConnectionState get_connection_state() const override;
	GatheringState get_gathering_state() const override;
	SignalingState get_signaling_state() const override;

	Error initialize(Dictionary p_config) override;
	Ref<WebRTCDataChannel> create_data_channel(String p_channel, Dictionary p_channel_config) override;
	Error create_offer() override;
	Error set_remote_description(String type, String sdp) override;
	Error set_local_description(String type, String sdp) override;
	Error add_ice_candidate(String sdpMidName, int sdpMlineIndexName, String sdpName) override;
	Error poll() override;
	void close() override;

	WebRTCLibPeerConnection();
	~WebRTCLibPeerConnection();

private:
	struct QueuedSignal {
	private:
		String method;
		Variant argv[3];
		int argc = 0;

	public:
		QueuedSignal() {}
		QueuedSignal(String p_method, int p_argc, const Variant *p_argv) {
			method = p_method;
			argc = p_argc;
			for (int i = 0; i < argc; i++) {
				argv[i] = p_argv[i];
			}
		}

		void emit(Object *p_object) {
			if (argc == 0) {
				p_object->emit_signal(method);
			} else if (argc == 1) {
				p_object->emit_signal(method, argv[0]);
			} else if (argc == 2) {
				p_object->emit_signal(method, argv[0], argv[1]);
			} else if (argc == 3) {
				p_object->emit_signal(method, argv[0], argv[1], argv[2]);
			}
		}
	};

	Mutex mutex_signal_queue;
	Vector<QueuedSignal> signal_queue;
	Vector<QueuedSignal> signal_queue_processing;

	void queue_signal(String p_name, int p_argc, const Variant &p_arg1 = Variant(), const Variant &p_arg2 = Variant(), const Variant &p_arg3 = Variant());
};

#endif // ENABLE_LIBDATACHANNEL

#endif // WEBRTC_LIB_PEER_CONNECTION_H
