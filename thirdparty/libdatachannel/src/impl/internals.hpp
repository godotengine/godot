/**
 * Copyright (c) 2019-2021 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_IMPL_INTERNALS_H
#define RTC_IMPL_INTERNALS_H

#include "common.hpp"

// Disable warnings before including plog
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#elif defined(_MSC_VER)
#pragma warning(push, 0)
#endif

#include "plog/Log.h"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(_MSC_VER)
#pragma warning(pop)
#endif

namespace rtc {

const size_t MAX_NUMERICNODE_LEN = 48; // Max IPv6 string representation length
const size_t MAX_NUMERICSERV_LEN = 6;  // Max port string representation length

const uint16_t DEFAULT_SCTP_PORT = 5000; // SCTP port to use by default

const uint16_t MAX_SCTP_STREAMS_COUNT = 1024; // Max number of negotiated SCTP streams
                                              // RFC 8831 recommends 65535 but usrsctp needs a lot
                                              // of memory, Chromium historically limits to 1024.

const size_t DEFAULT_LOCAL_MAX_MESSAGE_SIZE = 256 * 1024; // Default local max message size
const size_t DEFAULT_MAX_MESSAGE_SIZE = 65536; // Remote max message size if not specified in SDP

const size_t RECV_QUEUE_LIMIT = 1024 * 1024; // Max per-channel queue size

const int MIN_THREADPOOL_SIZE = 4; // Minimum number of threads in the global thread pool (>= 2)

const size_t DEFAULT_MTU = RTC_DEFAULT_MTU; // defined in rtc.h

} // namespace rtc

#endif
