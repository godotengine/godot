/**
 * Copyright (c) 2020 Filip Klembara (in2core)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_OPUS_PACKETIZATION_HANDLER_H
#define RTC_OPUS_PACKETIZATION_HANDLER_H

#if RTC_ENABLE_MEDIA

#include "mediachainablehandler.hpp"
#include "opusrtppacketizer.hpp"

namespace rtc {

/// Handler for opus packetization
class RTC_CPP_EXPORT OpusPacketizationHandler final : public MediaChainableHandler {

public:
	/// Construct handler for opus packetization.
	/// @param packetizer RTP packetizer for opus
	OpusPacketizationHandler(shared_ptr<OpusRtpPacketizer> packetizer);
};

} // namespace rtc

#endif /* RTC_ENABLE_MEDIA */

#endif /* RTC_OPUS_PACKETIZATION_HANDLER_H */
