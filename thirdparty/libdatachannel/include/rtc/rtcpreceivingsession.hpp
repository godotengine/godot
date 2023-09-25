/**
 * Copyright (c) 2020 Staz Modrzynski
 * Copyright (c) 2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_RTCP_RECEIVING_SESSION_H
#define RTC_RTCP_RECEIVING_SESSION_H

#if RTC_ENABLE_MEDIA

#include "common.hpp"
#include "mediahandler.hpp"
#include "message.hpp"
#include "rtp.hpp"

namespace rtc {

// An RtcpSession can be plugged into a Track to handle the whole RTCP session
class RTC_CPP_EXPORT RtcpReceivingSession : public MediaHandler {
public:
	message_ptr incoming(message_ptr ptr) override;
	message_ptr outgoing(message_ptr ptr) override;
	bool send(message_ptr ptr);

	void requestBitrate(unsigned int newBitrate);

	bool requestKeyframe() override;

protected:
	void pushREMB(unsigned int bitrate);
	void pushRR(unsigned int lastSR_delay);

	void pushPLI();

	unsigned int mRequestedBitrate = 0;
	SSRC mSsrc = 0;
	uint32_t mGreatestSeqNo = 0;
	uint64_t mSyncRTPTS, mSyncNTPTS;
};

} // namespace rtc

#endif // RTC_ENABLE_MEDIA

#endif // RTC_RTCP_RECEIVING_SESSION_H
