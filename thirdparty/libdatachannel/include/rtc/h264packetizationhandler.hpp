/**
 * Copyright (c) 2020 Filip Klembara (in2core)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_H264_PACKETIZATION_HANDLER_H
#define RTC_H264_PACKETIZATION_HANDLER_H

#if RTC_ENABLE_MEDIA

#include "h264rtppacketizer.hpp"
#include "mediachainablehandler.hpp"
#include "nalunit.hpp"

namespace rtc {

/// Handler for H264 packetization
class RTC_CPP_EXPORT H264PacketizationHandler final : public MediaChainableHandler {
public:
	/// Construct handler for H264 packetization.
	/// @param packetizer RTP packetizer for h264
	H264PacketizationHandler(shared_ptr<H264RtpPacketizer> packetizer);
};

} // namespace rtc

#endif /* RTC_ENABLE_MEDIA */

#endif /* RTC_H264_PACKETIZATION_HANDLER_H */
