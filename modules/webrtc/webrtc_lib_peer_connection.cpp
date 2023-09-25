/**************************************************************************/
/*  webrtc_lib_peer_connection.cpp                                        */
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

#ifdef ENABLE_LIBDATACHANNEL

#include "webrtc_lib_peer_connection.h"
#include "rtc/exception_wrapper_godot.hpp"
#include "webrtc_lib_data_channel.h"

Error WebRTCLibPeerConnection::_parse_ice_server(rtc::Configuration &r_config, Dictionary p_server) {
	ERR_FAIL_COND_V(!p_server.has("urls"), ERR_INVALID_PARAMETER);

	// Parse mandatory URL
	Array urls;
	Variant urls_var = p_server["urls"];
	if (urls_var.get_type() == Variant::STRING) {
		urls.push_back(urls_var);
	} else if (urls_var.get_type() == Variant::ARRAY) {
		urls = urls_var;
	} else {
		ERR_FAIL_V(ERR_INVALID_PARAMETER);
	}
	// Parse credentials (only meaningful for TURN, only support password)
	String username;
	String credential;
	if (p_server.has("username") && p_server["username"].get_type() == Variant::STRING) {
		username = p_server["username"];
	}
	if (p_server.has("credential") && p_server["credential"].get_type() == Variant::STRING) {
		credential = p_server["credential"];
	}
	for (int i = 0; i < urls.size(); i++) {
		rtc::IceServer srv(urls[i].operator String().utf8().get_data());
		srv.username = username.utf8().get_data();
		srv.password = credential.utf8().get_data();
		r_config.iceServers.push_back(srv);
	}
	return OK;
}

Error WebRTCLibPeerConnection::_parse_channel_config(rtc::DataChannelInit &r_config, const Dictionary &p_dict) {
	Variant nil;
	Variant v;
	if (p_dict.has("negotiated")) {
		r_config.negotiated = p_dict["negotiated"].operator bool();
	}
	if (p_dict.has("id")) {
		r_config.id = uint16_t(p_dict["id"].operator int32_t());
	}
	// If negotiated it must have an ID, and ID only makes sense when negotiated.
	ERR_FAIL_COND_V(r_config.negotiated != r_config.id.has_value(), ERR_INVALID_PARAMETER);
	// Channels cannot be both time-constrained and retry-constrained.
	ERR_FAIL_COND_V(p_dict.has("maxPacketLifeTime") && p_dict.has("maxRetransmits"), ERR_INVALID_PARAMETER);
	if (p_dict.has("maxPacketLifeTime")) {
		r_config.reliability.type = rtc::Reliability::Type::Timed;
		r_config.reliability.rexmit = std::chrono::milliseconds(p_dict["maxPacketLifeTime"].operator int32_t());
	} else if (p_dict.has("maxRetransmits")) {
		r_config.reliability.type = rtc::Reliability::Type::Rexmit;
		r_config.reliability.rexmit = p_dict["maxRetransmits"].operator int32_t();
	}
	if (p_dict.has("ordered") && p_dict["ordered"].operator bool() == false) {
		r_config.reliability.unordered = true;
	}
	if (p_dict.has("protocol")) {
		r_config.protocol = p_dict["protocol"].operator String().utf8().get_data();
	}
	return OK;
}

WebRTCPeerConnection::ConnectionState WebRTCLibPeerConnection::get_connection_state() const {
	ERR_FAIL_COND_V(peer_connection == nullptr, STATE_CLOSED);

	rtc::PeerConnection::State state = peer_connection->state();
	switch (state) {
		case rtc::PeerConnection::State::New:
			return STATE_NEW;
		case rtc::PeerConnection::State::Connecting:
			return STATE_CONNECTING;
		case rtc::PeerConnection::State::Connected:
			return STATE_CONNECTED;
		case rtc::PeerConnection::State::Disconnected:
			return STATE_DISCONNECTED;
		case rtc::PeerConnection::State::Failed:
			return STATE_FAILED;
		default:
			return STATE_CLOSED;
	}
}

WebRTCLibPeerConnection::GatheringState WebRTCLibPeerConnection::get_gathering_state() const {
	ERR_FAIL_COND_V(peer_connection == nullptr, GATHERING_STATE_NEW);

	rtc::PeerConnection::GatheringState state = peer_connection->gatheringState();
	switch (state) {
		case rtc::PeerConnection::GatheringState::New:
			return GATHERING_STATE_NEW;
		case rtc::PeerConnection::GatheringState::InProgress:
			return GATHERING_STATE_GATHERING;
		case rtc::PeerConnection::GatheringState::Complete:
			return GATHERING_STATE_COMPLETE;
		default:
			return GATHERING_STATE_NEW;
	}
}

WebRTCLibPeerConnection::SignalingState WebRTCLibPeerConnection::get_signaling_state() const {
	ERR_FAIL_COND_V(peer_connection == nullptr, SIGNALING_STATE_CLOSED);

	rtc::PeerConnection::SignalingState state = peer_connection->signalingState();
	switch (state) {
		case rtc::PeerConnection::SignalingState::Stable:
			return SIGNALING_STATE_STABLE;
		case rtc::PeerConnection::SignalingState::HaveLocalOffer:
			return SIGNALING_STATE_HAVE_LOCAL_OFFER;
		case rtc::PeerConnection::SignalingState::HaveRemoteOffer:
			return SIGNALING_STATE_HAVE_REMOTE_OFFER;
		case rtc::PeerConnection::SignalingState::HaveLocalPranswer:
			return SIGNALING_STATE_HAVE_LOCAL_PRANSWER;
		case rtc::PeerConnection::SignalingState::HaveRemotePranswer:
			return SIGNALING_STATE_HAVE_REMOTE_PRANSWER;
		default:
			return SIGNALING_STATE_CLOSED;
	}
}

Error WebRTCLibPeerConnection::initialize(Dictionary p_config) {
	rtc::Configuration config = {};
	if (p_config.has("iceServers") && p_config["iceServers"].get_type() == Variant::ARRAY) {
		Array servers = p_config["iceServers"];
		for (int i = 0; i < servers.size(); i++) {
			ERR_FAIL_COND_V(servers[i].get_type() != Variant::DICTIONARY, ERR_INVALID_PARAMETER);
			Dictionary server = servers[i];
			Error err = _parse_ice_server(config, server);
			ERR_FAIL_COND_V(err != OK, FAILED);
		}
	}
	return _create_pc(config);
}

Ref<WebRTCDataChannel> WebRTCLibPeerConnection::create_data_channel(String p_channel, Dictionary p_channel_config) {
	ERR_FAIL_COND_V(!peer_connection, nullptr);

	// Read config from dictionary
	rtc::DataChannelInit config;

	Error err = _parse_channel_config(config, p_channel_config);
	ERR_FAIL_COND_V(err != OK, nullptr);

	std::string error;
	std::shared_ptr<rtc::DataChannel> ch = LibDataChannelExceptionWrapper::create_data_channel(peer_connection, p_channel.utf8().get_data(), config, error);
	ERR_FAIL_COND_V_MSG(ch == nullptr, nullptr, vformat("Failed to create peer connection. %s", error.c_str()));

	Ref<WebRTCLibDataChannel> out;
	out.instantiate();
	// Bind the library data channel to our object.
	bool negotiated = ch->id().has_value();
	out->bind_channel(ch, negotiated);
	return out;
}

Error WebRTCLibPeerConnection::create_offer() {
	ERR_FAIL_COND_V(!peer_connection, ERR_UNCONFIGURED);
	ERR_FAIL_COND_V(get_connection_state() != STATE_NEW, FAILED);
	std::string error;
	if (!LibDataChannelExceptionWrapper::create_offer(peer_connection, error)) {
		ERR_FAIL_V_MSG(FAILED, error.c_str());
	}
	return OK;
}

Error WebRTCLibPeerConnection::set_remote_description(String p_type, String p_sdp) {
	ERR_FAIL_COND_V(!peer_connection, ERR_UNCONFIGURED);
	std::string error;
	if (!LibDataChannelExceptionWrapper::set_remote_description(peer_connection, p_type.utf8().get_data(), p_sdp.utf8().get_data(), error)) {
		ERR_FAIL_V_MSG(FAILED, error.c_str());
	}
	return OK;
}

Error WebRTCLibPeerConnection::set_local_description(String p_type, String p_sdp) {
	ERR_FAIL_COND_V(!peer_connection, ERR_UNCONFIGURED);
	// XXX Library quirk. It doesn't seem possible to create offers/answers without setting the local description.
	// Ignore this call for now to avoid crash (it's already set automatically!).
	// peer_connection->setLocalDescription(p_type == String("offer") ? rtc::Description::Type::Offer : rtc::Description::Type::Answer);
	return OK;
}

Error WebRTCLibPeerConnection::add_ice_candidate(String sdpMidName, int sdpMlineIndexName, String sdpName) {
	ERR_FAIL_COND_V(!peer_connection, ERR_UNCONFIGURED);
	std::string error;
	if (!LibDataChannelExceptionWrapper::add_ice_candidate(peer_connection, sdpMidName.utf8().get_data(), sdpName.utf8().get_data(), error)) {
		ERR_FAIL_V_MSG(FAILED, error.c_str());
	}
	return OK;
}

Error WebRTCLibPeerConnection::poll() {
	ERR_FAIL_COND_V(!peer_connection, ERR_UNCONFIGURED);

	{
		MutexLock lock(mutex_signal_queue);
		// Vector is missing swap()
		Vector<QueuedSignal> tmp = signal_queue_processing;
		signal_queue_processing = signal_queue;
		signal_queue = tmp;
	}
	QueuedSignal *signal_ptr = signal_queue_processing.ptrw();
	for (int i = 0; i < signal_queue_processing.size(); i++) {
		signal_ptr[i].emit(this);
	}
	signal_queue_processing.clear();
	return OK;
}

void WebRTCLibPeerConnection::close() {
	if (peer_connection != nullptr) {
		LibDataChannelExceptionWrapper::close_peer_connection(peer_connection);
	}

	MutexLock lock(mutex_signal_queue);
	signal_queue.clear();
}

Error WebRTCLibPeerConnection::_create_pc(rtc::Configuration &r_config) {
	// Prevents libdatachannel from automatically creating offers.
	r_config.disableAutoNegotiation = true;

	std::string error;
	peer_connection = LibDataChannelExceptionWrapper::create_peer_connection(r_config, error);
	ERR_FAIL_COND_V_MSG(!peer_connection, FAILED, vformat("Failed to create peer connection. %s", error.c_str()));

	// Binding this should be fine as long as we call close when going out of scope.
	peer_connection->onLocalDescription([this](rtc::Description description) {
		String type = description.type() == rtc::Description::Type::Offer ? "offer" : "answer";
		queue_signal("session_description_created", 2, type, String(std::string(description).c_str()));
	});
	peer_connection->onLocalCandidate([this](rtc::Candidate candidate) {
		queue_signal("ice_candidate_created", 3, String(candidate.mid().c_str()), 0, String(candidate.candidate().c_str()));
	});
	peer_connection->onDataChannel([this](std::shared_ptr<rtc::DataChannel> channel) {
		Ref<WebRTCLibDataChannel> new_data_channel;
		new_data_channel.instantiate();
		new_data_channel->bind_channel(channel, false);
		queue_signal("data_channel_received", 1, new_data_channel);
	});
	/*
	peer_connection->onStateChange([](rtc::PeerConnection::State state) {
		std::cout << "[State: " << state << "]" << std::endl;
	});

	peer_connection->onGatheringStateChange([](rtc::PeerConnection::GatheringState state) {
		std::cout << "[Gathering State: " << state << "]" << std::endl;
	});
	*/
	return OK;
}

WebRTCLibPeerConnection::WebRTCLibPeerConnection() {
#ifdef DEBUG_ENABLED
	static bool debug_initialized = (rtc::InitLogger(rtc::LogLevel::Debug), true);
	(void)debug_initialized;
#endif
	initialize(Dictionary());
}

WebRTCLibPeerConnection::~WebRTCLibPeerConnection() {
	close();
}

void WebRTCLibPeerConnection::queue_signal(String p_name, int p_argc, const Variant &p_arg1, const Variant &p_arg2, const Variant &p_arg3) {
	MutexLock lock(mutex_signal_queue);
	const Variant argv[3] = { p_arg1, p_arg2, p_arg3 };
	signal_queue.push_back(QueuedSignal(p_name, p_argc, argv));
}

#endif // ENABLE_LIBDATACHANNEL
