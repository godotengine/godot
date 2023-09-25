/**
 * Copyright (c) 2019 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_DATA_CHANNEL_H
#define RTC_DATA_CHANNEL_H

#include "channel.hpp"
#include "common.hpp"
#include "reliability.hpp"

#include <type_traits>

namespace rtc {

namespace impl {

struct DataChannel;
struct PeerConnection;

} // namespace impl

class RTC_CPP_EXPORT DataChannel final : private CheshireCat<impl::DataChannel>, public Channel {
public:
	DataChannel(impl_ptr<impl::DataChannel> impl);
	~DataChannel() override;

	optional<uint16_t> stream() const;
	optional<uint16_t> id() const;
	string label() const;
	string protocol() const;
	Reliability reliability() const;

	bool isOpen(void) const override;
	bool isClosed(void) const override;
	size_t maxMessageSize() const override;

	void close(void) override;
	bool send(message_variant data) override;
	bool send(const byte *data, size_t size) override;
	template <typename Buffer> bool sendBuffer(const Buffer &buf);
	template <typename Iterator> bool sendBuffer(Iterator first, Iterator last);

private:
	using CheshireCat<impl::DataChannel>::impl;
};

template <typename Buffer> std::pair<const byte *, size_t> to_bytes(const Buffer &buf) {
	using T = typename std::remove_pointer<decltype(buf.data())>::type;
	using E = typename std::conditional<std::is_void<T>::value, byte, T>::type;
	return std::make_pair(static_cast<const byte *>(static_cast<const void *>(buf.data())),
	                      buf.size() * sizeof(E));
}

template <typename Buffer> bool DataChannel::sendBuffer(const Buffer &buf) {
	auto [bytes, size] = to_bytes(buf);
	return send(bytes, size);
}

template <typename Iterator> bool DataChannel::sendBuffer(Iterator first, Iterator last) {
	size_t size = 0;
	for (Iterator it = first; it != last; ++it)
		size += it->size();

	binary buffer(size);
	byte *pos = buffer.data();
	for (Iterator it = first; it != last; ++it) {
		auto [bytes, len] = to_bytes(*it);
		pos = std::copy(bytes, bytes + len, pos);
	}
	return send(std::move(buffer));
}

} // namespace rtc

#endif
