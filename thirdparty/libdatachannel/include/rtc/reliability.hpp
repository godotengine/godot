/**
 * Copyright (c) 2019 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_RELIABILITY_H
#define RTC_RELIABILITY_H

#include "common.hpp"

#include <chrono>

namespace rtc {

struct Reliability {
	enum class Type { Reliable = 0, Rexmit, Timed };

	Type type = Type::Reliable;
	bool unordered = false;
	variant<int, std::chrono::milliseconds> rexmit = 0;
};

} // namespace rtc

#endif
