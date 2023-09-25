/**
 * Copyright (c) 2023 Zita Liao (Dolby)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_H265_RTP_PACKETIZER_H
#define RTC_H265_RTP_PACKETIZER_H

#if RTC_ENABLE_MEDIA

#include "h265nalunit.hpp"
#include "mediahandlerrootelement.hpp"
#include "rtppacketizer.hpp"

namespace rtc {

/// RTP packetization of h265 payload
class RTC_CPP_EXPORT H265RtpPacketizer final : public RtpPacketizer,
                                               public MediaHandlerRootElement {
	shared_ptr<H265NalUnits> splitMessage(binary_ptr message);
	const uint16_t maximumFragmentSize;

public:
	using Separator = NalUnit::Separator;

	/// Default clock rate for H265 in RTP
	inline static const uint32_t defaultClockRate = 90 * 1000;

	H265RtpPacketizer(NalUnit::Separator separator, shared_ptr<RtpPacketizationConfig> rtpConfig,
	                  uint16_t maximumFragmentSize = H265NalUnits::defaultMaximumFragmentSize);

	/// Constructs h265 payload packetizer with given RTP configuration.
	/// @note RTP configuration is used in packetization process which may change some configuration
	/// properties such as sequence number.
	/// @param rtpConfig  RTP configuration
	/// @param maximumFragmentSize maximum size of one NALU fragment
	H265RtpPacketizer(shared_ptr<RtpPacketizationConfig> rtpConfig,
	                  uint16_t maximumFragmentSize = H265NalUnits::defaultMaximumFragmentSize);

	ChainedOutgoingProduct processOutgoingBinaryMessage(ChainedMessagesProduct messages,
	                                                    message_ptr control) override;

private:
	const NalUnit::Separator separator;
};

} // namespace rtc

#endif /* RTC_ENABLE_MEDIA */

#endif /* RTC_H265_RTP_PACKETIZER_H */
