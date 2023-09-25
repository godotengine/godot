/**
 * Copyright (c) 2020-2021 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "track.hpp"

#include "impl/internals.hpp"
#include "impl/track.hpp"

namespace rtc {

Track::Track(impl_ptr<impl::Track> impl)
    : CheshireCat<impl::Track>(impl), Channel(std::dynamic_pointer_cast<impl::Channel>(impl)) {}

Track::~Track() {}

string Track::mid() const { return impl()->mid(); }

Description::Direction Track::direction() const { return impl()->direction(); }

Description::Media Track::description() const { return impl()->description(); }

void Track::setDescription(Description::Media description) {
	impl()->setDescription(std::move(description));
}

void Track::close() { impl()->close(); }

bool Track::send(message_variant data) { return impl()->outgoing(make_message(std::move(data))); }

bool Track::send(const byte *data, size_t size) { return send(binary(data, data + size)); }

bool Track::isOpen(void) const { return impl()->isOpen(); }

bool Track::isClosed(void) const { return impl()->isClosed(); }

size_t Track::maxMessageSize() const { return impl()->maxMessageSize(); }

void Track::setMediaHandler(shared_ptr<MediaHandler> handler) {
	impl()->setMediaHandler(std::move(handler));
}

bool Track::requestKeyframe() {
	// only push PLI for video
	if (description().type() == "video") {
		if (auto handler = impl()->getMediaHandler()) {
			return handler->requestKeyframe();
		}
	}
	return false;
}

shared_ptr<MediaHandler> Track::getMediaHandler() { return impl()->getMediaHandler(); }

} // namespace rtc
