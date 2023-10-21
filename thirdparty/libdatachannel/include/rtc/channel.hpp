/**
 * Copyright (c) 2019-2021 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_CHANNEL_H
#define RTC_CHANNEL_H

#include "common.hpp"

#include <atomic>
#include <functional>

namespace rtc {

namespace impl {
struct Channel;
}

class RTC_CPP_EXPORT Channel : private CheshireCat<impl::Channel> {
public:
	virtual ~Channel();

	virtual void close() = 0;
	virtual bool send(message_variant data) = 0; // returns false if buffered
	virtual bool send(const byte *data, size_t size) = 0;

	virtual bool isOpen() const = 0;
	virtual bool isClosed() const = 0;
	virtual size_t maxMessageSize() const; // max message size in a call to send
	virtual size_t bufferedAmount() const; // total size buffered to send

	void onOpen(std::function<void()> callback);
	void onClosed(std::function<void()> callback);
	void onError(std::function<void(string error)> callback);

	void onMessage(std::function<void(message_variant data)> callback);
	void onMessage(std::function<void(binary data)> binaryCallback,
	               std::function<void(string data)> stringCallback);

	void onBufferedAmountLow(std::function<void()> callback);
	void setBufferedAmountLowThreshold(size_t amount);

	void resetCallbacks();

	// Extended API
	optional<message_variant> receive(); // only if onMessage unset
	optional<message_variant> peek();    // only if onMessage unset
	size_t availableAmount() const;      // total size available to receive
	void onAvailable(std::function<void()> callback);

protected:
	Channel(impl_ptr<impl::Channel> impl);
};

} // namespace rtc

#endif // RTC_CHANNEL_H
