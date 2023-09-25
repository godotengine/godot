/**
 * Copyright (c) 2020-2021 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_GLOBAL_H
#define RTC_GLOBAL_H

#include "common.hpp"

#include <chrono>
#include <future>
#include <iostream>

namespace rtc {

enum class LogLevel { // Don't change, it must match plog severity
	None = 0,
	Fatal = 1,
	Error = 2,
	Warning = 3,
	Info = 4,
	Debug = 5,
	Verbose = 6
};

typedef std::function<void(LogLevel level, string message)> LogCallback;

RTC_CPP_EXPORT void InitLogger(LogLevel level, LogCallback callback = nullptr);

RTC_CPP_EXPORT void Preload();
RTC_CPP_EXPORT std::shared_future<void> Cleanup();

struct SctpSettings {
	// For the following settings, not set means optimized default
	optional<size_t> recvBufferSize;                // in bytes
	optional<size_t> sendBufferSize;                // in bytes
	optional<size_t> maxChunksOnQueue;              // in chunks
	optional<size_t> initialCongestionWindow;       // in MTUs
	optional<size_t> maxBurst;                      // in MTUs
	optional<unsigned int> congestionControlModule; // 0: RFC2581, 1: HSTCP, 2: H-TCP, 3: RTCC
	optional<std::chrono::milliseconds> delayedSackTime;
	optional<std::chrono::milliseconds> minRetransmitTimeout;
	optional<std::chrono::milliseconds> maxRetransmitTimeout;
	optional<std::chrono::milliseconds> initialRetransmitTimeout;
	optional<unsigned int> maxRetransmitAttempts;
	optional<std::chrono::milliseconds> heartbeatInterval;
};

RTC_CPP_EXPORT void SetSctpSettings(SctpSettings s);

} // namespace rtc

RTC_CPP_EXPORT std::ostream &operator<<(std::ostream &out, rtc::LogLevel level);

#endif
