/**
 * Copyright (c) 2020 Filip Klembara (in2core)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_MEDIA_CHAINABLE_HANDLER_H
#define RTC_MEDIA_CHAINABLE_HANDLER_H

#if RTC_ENABLE_MEDIA

#include "mediahandler.hpp"
#include "mediahandlerrootelement.hpp"

namespace rtc {

class RTC_CPP_EXPORT MediaChainableHandler : public MediaHandler {
	const shared_ptr<MediaHandlerRootElement> root;
	shared_ptr<MediaHandlerElement> leaf;
	mutable std::mutex mutex;

	message_ptr handleIncomingBinary(message_ptr);
	message_ptr handleIncomingControl(message_ptr);
	message_ptr handleOutgoingBinary(message_ptr);
	message_ptr handleOutgoingControl(message_ptr);
	bool sendProduct(ChainedOutgoingProduct product);
	shared_ptr<MediaHandlerElement> getLeaf() const;

public:
	MediaChainableHandler(shared_ptr<MediaHandlerRootElement> root);
	~MediaChainableHandler();
	message_ptr incoming(message_ptr ptr) override;
	message_ptr outgoing(message_ptr ptr) override;

	bool send(message_ptr msg);

	/// Adds element to chain
	/// @param chainable Chainable element
	void addToChain(shared_ptr<MediaHandlerElement> chainable);
};

} // namespace rtc

#endif // RTC_ENABLE_MEDIA

#endif // RTC_MEDIA_CHAINABLE_HANDLER_H
