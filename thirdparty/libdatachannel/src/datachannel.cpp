/**
 * Copyright (c) 2019-2021 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "datachannel.hpp"
#include "common.hpp"
#include "peerconnection.hpp"

#include "impl/datachannel.hpp"
#include "impl/internals.hpp"
#include "impl/peerconnection.hpp"

#ifdef _WIN32
#include <winsock2.h>
#else
#include <arpa/inet.h>
#endif

namespace rtc {

DataChannel::DataChannel(impl_ptr<impl::DataChannel> impl)
    : CheshireCat<impl::DataChannel>(impl),
      Channel(std::dynamic_pointer_cast<impl::Channel>(impl)) {}

DataChannel::~DataChannel() {}

void DataChannel::close() { return impl()->close(); }

optional<uint16_t> DataChannel::stream() const { return impl()->stream(); }

optional<uint16_t> DataChannel::id() const { return impl()->stream(); }

string DataChannel::label() const { return impl()->label(); }

string DataChannel::protocol() const { return impl()->protocol(); }

Reliability DataChannel::reliability() const { return impl()->reliability(); }

bool DataChannel::isOpen(void) const { return impl()->isOpen(); }

bool DataChannel::isClosed(void) const { return impl()->isClosed(); }

size_t DataChannel::maxMessageSize() const { return impl()->maxMessageSize(); }

bool DataChannel::send(message_variant data) {
	return impl()->outgoing(make_message(std::move(data)));
}

bool DataChannel::send(const byte *data, size_t size) {
	return impl()->outgoing(std::make_shared<Message>(data, data + size, Message::Binary));
}

} // namespace rtc
