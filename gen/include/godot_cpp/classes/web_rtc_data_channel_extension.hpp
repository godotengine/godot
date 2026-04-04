/**************************************************************************/
/*  web_rtc_data_channel_extension.hpp                                    */
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
#include <godot_cpp/classes/web_rtc_data_channel.hpp>
#include <godot_cpp/variant/string.hpp>

#include <godot_cpp/core/class_db.hpp>

#include <type_traits>

namespace godot {

class WebRTCDataChannelExtension : public WebRTCDataChannel {
	GDEXTENSION_CLASS(WebRTCDataChannelExtension, WebRTCDataChannel)

public:
	virtual Error _get_packet(const uint8_t **r_buffer, int32_t *r_buffer_size);
	virtual Error _put_packet(const uint8_t *p_buffer, int32_t p_buffer_size);
	virtual int32_t _get_available_packet_count() const;
	virtual int32_t _get_max_packet_size() const;
	virtual Error _poll();
	virtual void _close();
	virtual void _set_write_mode(WebRTCDataChannel::WriteMode p_write_mode);
	virtual WebRTCDataChannel::WriteMode _get_write_mode() const;
	virtual bool _was_string_packet() const;
	virtual WebRTCDataChannel::ChannelState _get_ready_state() const;
	virtual String _get_label() const;
	virtual bool _is_ordered() const;
	virtual int32_t _get_id() const;
	virtual int32_t _get_max_packet_life_time() const;
	virtual int32_t _get_max_retransmits() const;
	virtual String _get_protocol() const;
	virtual bool _is_negotiated() const;
	virtual int32_t _get_buffered_amount() const;

protected:
	template <typename T, typename B>
	static void register_virtuals() {
		WebRTCDataChannel::register_virtuals<T, B>();
		if constexpr (!std::is_same_v<decltype(&B::_get_packet), decltype(&T::_get_packet)>) {
			BIND_VIRTUAL_METHOD(T, _get_packet, 3099858825);
		}
		if constexpr (!std::is_same_v<decltype(&B::_put_packet), decltype(&T::_put_packet)>) {
			BIND_VIRTUAL_METHOD(T, _put_packet, 3099858825);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_available_packet_count), decltype(&T::_get_available_packet_count)>) {
			BIND_VIRTUAL_METHOD(T, _get_available_packet_count, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_max_packet_size), decltype(&T::_get_max_packet_size)>) {
			BIND_VIRTUAL_METHOD(T, _get_max_packet_size, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_poll), decltype(&T::_poll)>) {
			BIND_VIRTUAL_METHOD(T, _poll, 166280745);
		}
		if constexpr (!std::is_same_v<decltype(&B::_close), decltype(&T::_close)>) {
			BIND_VIRTUAL_METHOD(T, _close, 3218959716);
		}
		if constexpr (!std::is_same_v<decltype(&B::_set_write_mode), decltype(&T::_set_write_mode)>) {
			BIND_VIRTUAL_METHOD(T, _set_write_mode, 1999768052);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_write_mode), decltype(&T::_get_write_mode)>) {
			BIND_VIRTUAL_METHOD(T, _get_write_mode, 2848495172);
		}
		if constexpr (!std::is_same_v<decltype(&B::_was_string_packet), decltype(&T::_was_string_packet)>) {
			BIND_VIRTUAL_METHOD(T, _was_string_packet, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_ready_state), decltype(&T::_get_ready_state)>) {
			BIND_VIRTUAL_METHOD(T, _get_ready_state, 3501143017);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_label), decltype(&T::_get_label)>) {
			BIND_VIRTUAL_METHOD(T, _get_label, 201670096);
		}
		if constexpr (!std::is_same_v<decltype(&B::_is_ordered), decltype(&T::_is_ordered)>) {
			BIND_VIRTUAL_METHOD(T, _is_ordered, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_id), decltype(&T::_get_id)>) {
			BIND_VIRTUAL_METHOD(T, _get_id, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_max_packet_life_time), decltype(&T::_get_max_packet_life_time)>) {
			BIND_VIRTUAL_METHOD(T, _get_max_packet_life_time, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_max_retransmits), decltype(&T::_get_max_retransmits)>) {
			BIND_VIRTUAL_METHOD(T, _get_max_retransmits, 3905245786);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_protocol), decltype(&T::_get_protocol)>) {
			BIND_VIRTUAL_METHOD(T, _get_protocol, 201670096);
		}
		if constexpr (!std::is_same_v<decltype(&B::_is_negotiated), decltype(&T::_is_negotiated)>) {
			BIND_VIRTUAL_METHOD(T, _is_negotiated, 36873697);
		}
		if constexpr (!std::is_same_v<decltype(&B::_get_buffered_amount), decltype(&T::_get_buffered_amount)>) {
			BIND_VIRTUAL_METHOD(T, _get_buffered_amount, 3905245786);
		}
	}

public:
};

} // namespace godot

