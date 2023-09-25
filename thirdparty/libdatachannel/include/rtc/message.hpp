/**
 * Copyright (c) 2019-2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_MESSAGE_H
#define RTC_MESSAGE_H

#include "common.hpp"
#include "reliability.hpp"

#include <functional>

namespace rtc {

struct RTC_CPP_EXPORT Message : binary {
	enum Type { Binary, String, Control, Reset };

	Message(const Message &message) = default;
	Message(size_t size, Type type_ = Binary) : binary(size), type(type_) {}

	template <typename Iterator>
	Message(Iterator begin_, Iterator end_, Type type_ = Binary)
	    : binary(begin_, end_), type(type_) {}

	Message(binary &&data, Type type_ = Binary) : binary(std::move(data)), type(type_) {}

	Type type;
	unsigned int stream = 0; // Stream id (SCTP stream or SSRC)
	unsigned int dscp = 0;   // Differentiated Services Code Point
	shared_ptr<Reliability> reliability;
};

using message_ptr = shared_ptr<Message>;
using message_callback = std::function<void(message_ptr message)>;

inline size_t message_size_func(const message_ptr &m) {
	return m->type == Message::Binary || m->type == Message::String ? m->size() : 0;
}

template <typename Iterator>
message_ptr make_message(Iterator begin, Iterator end, Message::Type type = Message::Binary,
                         unsigned int stream = 0, shared_ptr<Reliability> reliability = nullptr) {
	auto message = std::make_shared<Message>(begin, end, type);
	message->stream = stream;
	message->reliability = reliability;
	return message;
}

RTC_CPP_EXPORT message_ptr make_message(size_t size, Message::Type type = Message::Binary,
                                        unsigned int stream = 0,
                                        shared_ptr<Reliability> reliability = nullptr);

RTC_CPP_EXPORT message_ptr make_message(binary &&data, Message::Type type = Message::Binary,
                                        unsigned int stream = 0,
                                        shared_ptr<Reliability> reliability = nullptr);

RTC_CPP_EXPORT message_ptr make_message(message_variant data);

#if RTC_ENABLE_MEDIA

// Reconstructs a message_ptr from an opaque rtcMessage pointer that
// was allocated by rtcCreateOpaqueMessage().
message_ptr make_message_from_opaque_ptr(rtcMessage *&&message);

#endif

RTC_CPP_EXPORT message_variant to_variant(Message &&message);
RTC_CPP_EXPORT message_variant to_variant(const Message &message);

} // namespace rtc

#endif
