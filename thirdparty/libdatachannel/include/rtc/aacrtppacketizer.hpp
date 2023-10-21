/**
 * Copyright (c) 2020 Filip Klembara (in2core)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_AAC_RTP_PACKETIZER_H
#define RTC_AAC_RTP_PACKETIZER_H

#if RTC_ENABLE_MEDIA

#include "mediachainablehandler.hpp"
#include "mediahandlerrootelement.hpp"
#include "rtppacketizer.hpp"

namespace rtc {

/// RTP packetizer for aac
class RTC_CPP_EXPORT AACRtpPacketizer final : public RtpPacketizer, public MediaHandlerRootElement {
public:
	/// default clock rate used in aac RTP communication
	inline static const uint32_t defaultClockRate = 48 * 1000;

	/// Constructs aac packetizer with given RTP configuration.
	/// @note RTP configuration is used in packetization process which may change some configuration
	/// properties such as sequence number.
	/// @param rtpConfig  RTP configuration
	AACRtpPacketizer(shared_ptr<RtpPacketizationConfig> rtpConfig);

	/// Creates RTP packet for given payload based on `rtpConfig`.
	/// @note This function increase sequence number after packetization.
	/// @param payload RTP payload
	/// @param setMark This needs to be `false` for all RTP packets with aac payload
	binary_ptr packetize(binary_ptr payload, bool setMark) override;

	/// Creates RTP packet for given samples (all samples share same RTP timesamp)
	/// @param messages aac samples
	/// @param control RTCP
	/// @returns RTP packets and unchanged `control`
	ChainedOutgoingProduct processOutgoingBinaryMessage(ChainedMessagesProduct messages,
	                                                    message_ptr control) override;
};

/// Handler for aac packetization
class RTC_CPP_EXPORT AACPacketizationHandler final : public MediaChainableHandler {

public:
	/// Construct handler for aac packetization.
	/// @param packetizer RTP packetizer for aac
	AACPacketizationHandler(shared_ptr<AACRtpPacketizer> packetizer)
	    : MediaChainableHandler(packetizer) {}
};

} // namespace rtc

#endif /* RTC_ENABLE_MEDIA */

#endif /* RTC_AAC_RTP_PACKETIZER_H */
