/**
 * Copyright (c) 2019-2021 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "channel.hpp"

#include "impl/channel.hpp"
#include "impl/internals.hpp"

namespace rtc {

Channel::~Channel() { impl()->resetCallbacks(); }

Channel::Channel(impl_ptr<impl::Channel> impl) : CheshireCat<impl::Channel>(std::move(impl)) {}

size_t Channel::maxMessageSize() const { return DEFAULT_MAX_MESSAGE_SIZE; }

size_t Channel::bufferedAmount() const { return impl()->bufferedAmount; }

void Channel::onOpen(std::function<void()> callback) { impl()->openCallback = callback; }

void Channel::onClosed(std::function<void()> callback) { impl()->closedCallback = callback; }

void Channel::onError(std::function<void(string error)> callback) {
	impl()->errorCallback = callback;
}

void Channel::onMessage(std::function<void(message_variant data)> callback) {
	impl()->messageCallback = callback;
	impl()->flushPendingMessages();
}

void Channel::onMessage(std::function<void(binary data)> binaryCallback,
                        std::function<void(string data)> stringCallback) {
	onMessage([binaryCallback, stringCallback](variant<binary, string> data) {
		std::visit(overloaded{binaryCallback, stringCallback}, std::move(data));
	});
}

void Channel::onBufferedAmountLow(std::function<void()> callback) {
	impl()->bufferedAmountLowCallback = callback;
}

void Channel::setBufferedAmountLowThreshold(size_t amount) {
	impl()->bufferedAmountLowThreshold = amount;
}

void Channel::resetCallbacks() { impl()->resetCallbacks(); }

optional<message_variant> Channel::receive() { return impl()->receive(); }

optional<message_variant> Channel::peek() { return impl()->peek(); }

size_t Channel::availableAmount() const { return impl()->availableAmount(); }

void Channel::onAvailable(std::function<void()> callback) { impl()->availableCallback = callback; }

} // namespace rtc
