/**
 * Copyright (c) 2020 Filip Klembara (in2core)
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_RTP_PACKETIZER_H
#define RTC_RTP_PACKETIZER_H

#if RTC_ENABLE_MEDIA

#include "message.hpp"
#include "rtppacketizationconfig.hpp"

namespace rtc {

/// Class responsible for RTP packetization
class RTC_CPP_EXPORT RtpPacketizer {
	static const auto rtpHeaderSize = 12;
	static const auto rtpExtHeaderCvoSize = 8;

public:
	virtual ~RtpPacketizer() = default;

	// RTP configuration
	const shared_ptr<RtpPacketizationConfig> rtpConfig;

	/// Constructs packetizer with given RTP configuration.
	/// @note RTP configuration is used in packetization process which may change some configuration
	/// properties such as sequence number.
	/// @param rtpConfig  RTP configuration
	RtpPacketizer(shared_ptr<RtpPacketizationConfig> rtpConfig);

	/// Creates RTP packet for given payload based on `rtpConfig`.
	/// @note This function increase sequence number after packetization.
	/// @param payload RTP payload
	/// @param setMark Set marker flag in RTP packet if true
	virtual shared_ptr<binary> packetize(shared_ptr<binary> payload, bool setMark);
};

} // namespace rtc

#endif /* RTC_ENABLE_MEDIA */

#endif /* RTC_RTP_PACKETIZER_H */
