/**
 * Copyright (c) 2020-2021 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef RTC_IMPL_WS_HANDSHAKE_H
#define RTC_IMPL_WS_HANDSHAKE_H

#include "common.hpp"

#if RTC_ENABLE_WEBSOCKET

#include <list>
#include <map>
#include <stdexcept>
#include <vector>

namespace rtc::impl {

class WsHandshake final {
public:
	WsHandshake();
	WsHandshake(string host, string path = "/", std::vector<string> protocols = {});

	string host() const;
	string path() const;
	std::vector<string> protocols() const;

	string generateHttpRequest();
	string generateHttpResponse();
	string generateHttpError(int responseCode = 400);

	class Error : public std::runtime_error {
	public:
		explicit Error(const string &w);
	};

	class RequestError : public Error {
	public:
		explicit RequestError(const string &w, int responseCode = 400);
		int responseCode() const;

	private:
		const int mResponseCode;
	};

	size_t parseHttpRequest(const byte *buffer, size_t size);
	size_t parseHttpResponse(const byte *buffer, size_t size);

private:
	static string generateKey();
	static string computeAcceptKey(const string &key);

	string mHost;
	string mPath;
	std::vector<string> mProtocols;
	string mKey;
	mutable std::mutex mMutex;
};

} // namespace rtc::impl

#endif

#endif
