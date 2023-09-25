/**
 * Copyright (c) 2019-2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "message.hpp"

namespace rtc {

message_ptr make_message(size_t size, Message::Type type, unsigned int stream,
                         shared_ptr<Reliability> reliability) {
	auto message = std::make_shared<Message>(size, type);
	message->stream = stream;
	message->reliability = reliability;
	return message;
}

message_ptr make_message(binary &&data, Message::Type type, unsigned int stream,
                         shared_ptr<Reliability> reliability) {
	auto message = std::make_shared<Message>(std::move(data), type);
	message->stream = stream;
	message->reliability = reliability;
	return message;
}

message_ptr make_message(message_variant data) {
	return std::visit( //
	    overloaded{
	        [&](binary data) { return make_message(std::move(data), Message::Binary); },
	        [&](string data) {
		        auto b = reinterpret_cast<const byte *>(data.data());
		        return make_message(b, b + data.size(), Message::String);
	        },
	    },
	    std::move(data));
}

#if RTC_ENABLE_MEDIA

message_ptr make_message_from_opaque_ptr(rtcMessage *&&message) {
	auto ptr = std::unique_ptr<Message>(reinterpret_cast<Message *>(message));
	return message_ptr(std::move(ptr));
}

#endif

message_variant to_variant(Message &&message) {
	switch (message.type) {
	case Message::String:
		return string(reinterpret_cast<const char *>(message.data()), message.size());
	default:
		return std::move(message);
	}
}

message_variant to_variant(const Message &message) {
	switch (message.type) {
	case Message::String:
		return string(reinterpret_cast<const char *>(message.data()), message.size());
	default:
		return message;
	}
}

} // namespace rtc
