/**
 * Copyright (c) 2021 Staz Modrzynski
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_SERVER_LOGCOUNTER_HPP
#define RTC_SERVER_LOGCOUNTER_HPP

#include "common.hpp"
#include "threadpool.hpp"

#include <atomic>
#include <chrono>

namespace rtc::impl {

class LogCounter {
private:
	struct LogData {
		plog::Severity mSeverity;
		std::string mText;
		std::chrono::steady_clock::duration mDuration;

		std::atomic<int> mCount = 0;
	};

	shared_ptr<LogData> mData;

public:
	LogCounter(plog::Severity severity, const std::string &text,
	           std::chrono::seconds duration = std::chrono::seconds(1));

	LogCounter &operator++(int);
};

} // namespace rtc::impl

#endif // RTC_SERVER_LOGCOUNTER_HPP
