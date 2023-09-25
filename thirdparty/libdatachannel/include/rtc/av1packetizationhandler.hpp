/**
 * Copyright (c) 2023 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_AV1_PACKETIZATION_HANDLER_H
#define RTC_AV1_PACKETIZATION_HANDLER_H

#if RTC_ENABLE_MEDIA

#include "av1rtppacketizer.hpp"
#include "mediachainablehandler.hpp"
#include "nalunit.hpp"

namespace rtc {

/// Handler for AV1 packetization
class RTC_CPP_EXPORT AV1PacketizationHandler final : public MediaChainableHandler {
public:
	/// Construct handler for AV1 packetization.
	/// @param packetizer RTP packetizer for AV1
	AV1PacketizationHandler(shared_ptr<AV1RtpPacketizer> packetizer);
};

} // namespace rtc

#endif /* RTC_ENABLE_MEDIA */

#endif /* RTC_AV1_PACKETIZATION_HANDLER_H */
