/**************************************************************************/
/*  web_rtc_peer_connection_extension.hpp                                 */
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
#include <godot_cpp/classes/web_rtc_peer_connection.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class Dictionary;
class String;
class WebRTCDataChannel;

class WebRTCPeerConnectionExtension : public WebRTCPeerConnection {
	GDEXTENSION_CLASS(WebRTCPeerConnectionExtension, WebRTCPeerConnection)

public:
	virtual WebRTCPeerConnection::ConnectionState _get_connection_state() const;
	virtual WebRTCPeerConnection::GatheringState _get_gathering_state() const;
	virtual WebRTCPeerConnection::SignalingState _get_signaling_state() const;
	virtual Error _initialize(const Dictionary &p_config);
	virtual Ref<WebRTCDataChannel> _create_data_channel(const String &p_label, const Dictionary &p_config);
	virtual Error _create_offer();
	virtual Error _set_remote_description(const String &p_type, const String &p_sdp);
	virtual Error _set_local_description(const String &p_type, const String &p_sdp);
	virtual Error _add_ice_candidate(const String &p_sdp_mid_name, int32_t p_sdp_mline_index, const String &p_sdp_name);
	virtual Error _poll();
	virtual void _close();

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		WebRTCPeerConnection::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_get_connection_state), decltype(&T::_get_connection_state)>) {
			BIND_VIRTUAL_METHOD(T, _get_connection_state, 2275710506);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_gathering_state), decltype(&T::_get_gathering_state)>) {
			BIND_VIRTUAL_METHOD(T, _get_gathering_state, 4262591401);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_signaling_state), decltype(&T::_get_signaling_state)>) {
			BIND_VIRTUAL_METHOD(T, _get_signaling_state, 3342956226);
		}
		if constexpr (!std::is_same_v<decltype(&B::_initialize), decltype(&T::_initialize)>) {
			BIND_VIRTUAL_METHOD(T, _initialize, 1494659981);
		}
		if constexpr (!std::is_same_v<decltype(&B::_create_data_channel), decltype(&T::_create_data_channel)>) {
			BIND_VIRTUAL_METHOD(T, _create_data_channel, 4111292546);
		}
		if constexpr (!std::is_same_v<decltype(&B::_create_offer), decltype(&T::_create_offer)>) {
			BIND_VIRTUAL_METHOD(T, _create_offer, 166280745);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_remote_description), decltype(&T::_set_remote_description)>) {
			BIND_VIRTUAL_METHOD(T, _set_remote_description, 852856452);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_local_description), decltype(&T::_set_local_description)>) {
			BIND_VIRTUAL_METHOD(T, _set_local_description, 852856452);
		}
		if constexpr (!std::is_same_v<decltype(&B::_add_ice_candidate), decltype(&T::_add_ice_candidate)>) {
			BIND_VIRTUAL_METHOD(T, _add_ice_candidate, 3958950400);
		}
		if constexpr (!std::is_same_v<decltype(&B::_poll), decltype(&T::_poll)>) {
			BIND_VIRTUAL_METHOD(T, _poll, 166280745);
		}
		if constexpr (!std::is_same_v<decltype(&B::_close), decltype(&T::_close)>) {
			BIND_VIRTUAL_METHOD(T, _close, 3218959716);
		}
	}

public:
};

} // namespace godot

