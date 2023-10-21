/**
 * Copyright (c) 2021 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_IMPL_SHA_H
#define RTC_IMPL_SHA_H

#if RTC_ENABLE_WEBSOCKET

#include "common.hpp"

namespace rtc::impl {

binary Sha1(const binary &input);
binary Sha1(const string &input);

} // namespace rtc::impl

#endif

#endif
