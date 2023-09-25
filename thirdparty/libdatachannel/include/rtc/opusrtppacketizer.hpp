/**
 * Copyright (c) 2020 Filip Klembara (in2core)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_OPUS_RTP_PACKETIZER_H
#define RTC_OPUS_RTP_PACKETIZER_H

#if RTC_ENABLE_MEDIA

#include "mediahandlerrootelement.hpp"
#include "rtppacketizer.hpp"

namespace rtc {

/// RTP packetizer for opus
class RTC_CPP_EXPORT OpusRtpPacketizer final : public RtpPacketizer,
                                               public MediaHandlerRootElement {
public:
	/// default clock rate used in opus RTP communication
	inline static const uint32_t defaultClockRate = 48 * 1000;

	/// Constructs opus packetizer with given RTP configuration.
	/// @note RTP configuration is used in packetization process which may change some configuration
	/// properties such as sequence number.
	/// @param rtpConfig  RTP configuration
	OpusRtpPacketizer(shared_ptr<RtpPacketizationConfig> rtpConfig);

	/// Creates RTP packet for given payload based on `rtpConfig`.
	/// @note This function increase sequence number after packetization.
	/// @param payload RTP payload
	/// @param setMark This needs to be `false` for all RTP packets with opus payload
	binary_ptr packetize(binary_ptr payload, bool setMark) override;

	/// Creates RTP packet for given samples (all samples share same RTP timesamp)
	/// @param messages opus samples
	/// @param control RTCP
	/// @returns RTP packets and unchanged `control`
	ChainedOutgoingProduct processOutgoingBinaryMessage(ChainedMessagesProduct messages,
	                                                    message_ptr control) override;
};

} // namespace rtc

#endif /* RTC_ENABLE_MEDIA */

#endif /* RTC_OPUS_RTP_PACKETIZER_H */
