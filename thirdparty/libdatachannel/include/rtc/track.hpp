/**
 * Copyright (c) 2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_TRACK_H
#define RTC_TRACK_H

#include "channel.hpp"
#include "common.hpp"
#include "description.hpp"
#include "mediahandler.hpp"

namespace rtc {

namespace impl {

class Track;

} // namespace impl

class RTC_CPP_EXPORT Track final : private CheshireCat<impl::Track>, public Channel {
public:
	Track(impl_ptr<impl::Track> impl);
	~Track() override;

	string mid() const;
	Description::Direction direction() const;
	Description::Media description() const;

	void setDescription(Description::Media description);

	void close(void) override;
	bool send(message_variant data) override;
	bool send(const byte *data, size_t size) override;

	bool isOpen(void) const override;
	bool isClosed(void) const override;
	size_t maxMessageSize() const override;

	bool requestKeyframe();

	void setMediaHandler(shared_ptr<MediaHandler> handler);
	shared_ptr<MediaHandler> getMediaHandler();

	// Deprecated, use setMediaHandler() and getMediaHandler()
	inline void setRtcpHandler(shared_ptr<MediaHandler> handler) { setMediaHandler(handler); }
	inline shared_ptr<MediaHandler> getRtcpHandler() { return getMediaHandler(); }

private:
	using CheshireCat<impl::Track>::impl;
};

} // namespace rtc

#endif
