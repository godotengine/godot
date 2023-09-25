/**
 * Copyright (c) 2019-2022 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "transport.hpp"

namespace rtc::impl {

Transport::Transport(shared_ptr<Transport> lower, state_callback callback)
    : mLower(std::move(lower)), mStateChangeCallback(std::move(callback)) {}

Transport::~Transport() {
	unregisterIncoming();

	if (mLower) {
		mLower->stop();
		mLower.reset();
	}
}

void Transport::registerIncoming() {
	if (mLower) {
		PLOG_VERBOSE << "Registering incoming callback";
		mLower->onRecv(std::bind(&Transport::incoming, this, std::placeholders::_1));
	}
}

void Transport::unregisterIncoming() {
	if (mLower) {
		PLOG_VERBOSE << "Unregistering incoming callback";
		mLower->onRecv(nullptr);
	}
}

Transport::State Transport::state() const { return mState; }

void Transport::onRecv(message_callback callback) { mRecvCallback = std::move(callback); }

void Transport::onStateChange(state_callback callback) {
	mStateChangeCallback = std::move(callback);
}

void Transport::start() { registerIncoming(); }

void Transport::stop() { unregisterIncoming(); }

bool Transport::send(message_ptr message) { return outgoing(message); }

void Transport::recv(message_ptr message) {
	try {
		mRecvCallback(message);
	} catch (const std::exception &e) {
		PLOG_WARNING << e.what();
	}
}

void Transport::changeState(State state) {
	try {
		if (mState.exchange(state) != state)
			mStateChangeCallback(state);
	} catch (const std::exception &e) {
		PLOG_WARNING << e.what();
	}
}

void Transport::incoming(message_ptr message) { recv(message); }

bool Transport::outgoing(message_ptr message) {
	if (mLower)
		return mLower->send(message);
	else
		return false;
}

} // namespace rtc::impl
