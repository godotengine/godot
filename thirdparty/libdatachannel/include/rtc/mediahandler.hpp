/**
 * Copyright (c) 2020 Staz Modrzynski
 * Copyright (c) 2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_MEDIA_HANDLER_H
#define RTC_MEDIA_HANDLER_H

#include "common.hpp"
#include "message.hpp"

namespace rtc {

class RTC_CPP_EXPORT MediaHandler {
protected:
	// Use this callback when trying to send custom data (such as RTCP) to the client.
	synchronized_callback<message_ptr> outgoingCallback;

public:
	virtual ~MediaHandler() = default;

	// Called when there is traffic coming from the peer
	virtual message_ptr incoming(message_ptr ptr) = 0;

	// Called when there is traffic that needs to be sent to the peer
	virtual message_ptr outgoing(message_ptr ptr) = 0;

	// This callback is used to send traffic back to the peer.
	void onOutgoing(const std::function<void(message_ptr)> &cb) {
		this->outgoingCallback = synchronized_callback<message_ptr>(cb);
	}

	virtual bool requestKeyframe() { return false; }
};

} // namespace rtc

#endif // RTC_MEDIA_HANDLER_H
