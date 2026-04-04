/**************************************************************************/
/*  web_rtc_peer_connection.hpp                                           */
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

// THIS FILE IS GENERATED. EDITS WILL BE LOST.

#pragma once

#include <godot_cpp/classes/global_constants.hpp>
#include <godot_cpp/classes/ref.hpp>
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/variant/dictionary.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class String;
class StringName;
class WebRTCDataChannel;

class WebRTCPeerConnection : public RefCounted {
	GDEXTENSION_CLASS(WebRTCPeerConnection, RefCounted)

public:
	enum ConnectionState {
		STATE_NEW = 0,
		STATE_CONNECTING = 1,
		STATE_CONNECTED = 2,
		STATE_DISCONNECTED = 3,
		STATE_FAILED = 4,
		STATE_CLOSED = 5,
	};

	enum GatheringState {
		GATHERING_STATE_NEW = 0,
		GATHERING_STATE_GATHERING = 1,
		GATHERING_STATE_COMPLETE = 2,
	};

	enum SignalingState {
		SIGNALING_STATE_STABLE = 0,
		SIGNALING_STATE_HAVE_LOCAL_OFFER = 1,
		SIGNALING_STATE_HAVE_REMOTE_OFFER = 2,
		SIGNALING_STATE_HAVE_LOCAL_PRANSWER = 3,
		SIGNALING_STATE_HAVE_REMOTE_PRANSWER = 4,
		SIGNALING_STATE_CLOSED = 5,
	};

	static void set_default_extension(const StringName &p_extension_class);
	Error initialize(const Dictionary &p_configuration = Dictionary());
	Ref<WebRTCDataChannel> create_data_channel(const String &p_label, const Dictionary &p_options = Dictionary());
	Error create_offer();
	Error set_local_description(const String &p_type, const String &p_sdp);
	Error set_remote_description(const String &p_type, const String &p_sdp);
	Error add_ice_candidate(const String &p_media, int32_t p_index, const String &p_name);
	Error poll();
	void close();
	WebRTCPeerConnection::ConnectionState get_connection_state() const;
	WebRTCPeerConnection::GatheringState get_gathering_state() const;
	WebRTCPeerConnection::SignalingState get_signaling_state() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		RefCounted::register_virtuals<T, B>();
	}

public:
};

} // namespace godot

VARIANT_ENUM_CAST(WebRTCPeerConnection::ConnectionState);
VARIANT_ENUM_CAST(WebRTCPeerConnection::GatheringState);
VARIANT_ENUM_CAST(WebRTCPeerConnection::SignalingState);

