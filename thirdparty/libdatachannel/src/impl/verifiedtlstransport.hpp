/**
 * Copyright (c) 2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_IMPL_VERIFIED_TLS_TRANSPORT_H
#define RTC_IMPL_VERIFIED_TLS_TRANSPORT_H

#include "tlstransport.hpp"

#if RTC_ENABLE_WEBSOCKET

namespace rtc::impl {

class VerifiedTlsTransport final : public TlsTransport {
public:
	VerifiedTlsTransport(variant<shared_ptr<TcpTransport>, shared_ptr<HttpProxyTransport>> lower,
	                     string host, certificate_ptr certificate, state_callback callback);
	~VerifiedTlsTransport();
};

} // namespace rtc::impl

#endif

#endif
