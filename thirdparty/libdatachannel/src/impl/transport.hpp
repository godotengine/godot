/**
 * Copyright (c) 2019-2022 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_IMPL_TRANSPORT_H
#define RTC_IMPL_TRANSPORT_H

#include "common.hpp"
#include "init.hpp"
#include "internals.hpp"
#include "message.hpp"

#include <atomic>
#include <functional>
#include <memory>

namespace rtc::impl {

class Transport {
public:
	enum class State { Disconnected, Connecting, Connected, Completed, Failed };
	using state_callback = std::function<void(State state)>;

	Transport(shared_ptr<Transport> lower = nullptr, state_callback callback = nullptr);
	virtual ~Transport();

	void registerIncoming();
	void unregisterIncoming();
	State state() const;

	void onRecv(message_callback callback);
	void onStateChange(state_callback callback);

	virtual void start();
	virtual void stop();
	virtual bool send(message_ptr message);

protected:
	void recv(message_ptr message);
	void changeState(State state);
	virtual void incoming(message_ptr message);
	virtual bool outgoing(message_ptr message);

private:
	const init_token mInitToken = Init::Instance().token();

	shared_ptr<Transport> mLower;
	synchronized_callback<State> mStateChangeCallback;
	synchronized_callback<message_ptr> mRecvCallback;

	std::atomic<State> mState = State::Disconnected;
};

} // namespace rtc::impl

#endif
