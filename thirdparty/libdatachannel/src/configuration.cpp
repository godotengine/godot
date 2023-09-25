/**
 * Copyright (c) 2019 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "configuration.hpp"

#include "impl/utils.hpp"

#include <cassert>
#include <regex>

namespace {

bool parse_url(const std::string &url, std::vector<std::optional<std::string>> &result) {
	// Modified regex from RFC 3986, see https://www.rfc-editor.org/rfc/rfc3986.html#appendix-B
	static const char *rs =
	    R"(^(([^:.@/?#]+):)?(/{0,2}((([^:@]*)(:([^@]*))?)@)?(([^:/?#]*)(:([^/?#]*))?))?([^?#]*)(\?([^#]*))?(#(.*))?)";
	static const std::regex r(rs, std::regex::extended);

	std::smatch m;
	if (!std::regex_match(url, m, r) || m[10].length() == 0)
		return false;

	result.resize(m.size());
	std::transform(m.begin(), m.end(), result.begin(), [](const auto &sm) {
		return sm.length() > 0 ? std::make_optional(std::string(sm)) : std::nullopt;
	});

	assert(result.size() == 18);
	return true;
}

} // namespace

namespace rtc {

namespace utils = impl::utils;

IceServer::IceServer(const string &url) {
	std::vector<optional<string>> opt;
	if (!parse_url(url, opt))
		throw std::invalid_argument("Invalid ICE server URL: " + url);

	string scheme = opt[2].value_or("stun");
	relayType = RelayType::TurnUdp;
	if (scheme == "stun" || scheme == "STUN")
		type = Type::Stun;
	else if (scheme == "turn" || scheme == "TURN")
		type = Type::Turn;
	else if (scheme == "turns" || scheme == "TURNS") {
		type = Type::Turn;
		relayType = RelayType::TurnTls;
	} else
		throw std::invalid_argument("Unknown ICE server protocol: " + scheme);

	if (auto &query = opt[15]) {
		if (query->find("transport=udp") != string::npos)
			relayType = RelayType::TurnUdp;
		if (query->find("transport=tcp") != string::npos)
			relayType = RelayType::TurnTcp;
		if (query->find("transport=tls") != string::npos)
			relayType = RelayType::TurnTls;
	}

	username = utils::url_decode(opt[6].value_or(""));
	password = utils::url_decode(opt[8].value_or(""));

	hostname = opt[10].value();
	if (hostname.front() == '[' && hostname.back() == ']') {
		// IPv6 literal
		hostname.erase(hostname.begin());
		hostname.pop_back();
	} else {
		hostname = utils::url_decode(hostname);
	}

	string service = opt[12].value_or(relayType == RelayType::TurnTls ? "5349" : "3478");
	try {
		port = uint16_t(std::stoul(service));
	} catch (...) {
		throw std::invalid_argument("Invalid ICE server port in URL: " + service);
	}
}

IceServer::IceServer(string hostname_, uint16_t port_)
    : hostname(std::move(hostname_)), port(port_), type(Type::Stun) {}

IceServer::IceServer(string hostname_, string service_)
    : hostname(std::move(hostname_)), type(Type::Stun) {
	try {
		port = uint16_t(std::stoul(service_));
	} catch (...) {
		throw std::invalid_argument("Invalid ICE server port: " + service_);
	}
}

IceServer::IceServer(string hostname_, uint16_t port_, string username_, string password_,
                     RelayType relayType_)
    : hostname(std::move(hostname_)), port(port_), type(Type::Turn), username(std::move(username_)),
      password(std::move(password_)), relayType(relayType_) {}

IceServer::IceServer(string hostname_, string service_, string username_, string password_,
                     RelayType relayType_)
    : hostname(std::move(hostname_)), type(Type::Turn), username(std::move(username_)),
      password(std::move(password_)), relayType(relayType_) {
	try {
		port = uint16_t(std::stoul(service_));
	} catch (...) {
		throw std::invalid_argument("Invalid ICE server port: " + service_);
	}
}

ProxyServer::ProxyServer(const string &url) {
	std::vector<optional<string>> opt;
	if (!parse_url(url, opt))
		throw std::invalid_argument("Invalid proxy server URL: " + url);

	string scheme = opt[2].value_or("http");
	if (scheme == "http" || scheme == "HTTP")
		type = Type::Http;
	else if (scheme == "socks5" || scheme == "SOCKS5")
		type = Type::Socks5;
	else
		throw std::invalid_argument("Unknown proxy server protocol: " + scheme);

	username = opt[6];
	password = opt[8];

	hostname = opt[10].value();
	while (!hostname.empty() && hostname.front() == '[')
		hostname.erase(hostname.begin());
	while (!hostname.empty() && hostname.back() == ']')
		hostname.pop_back();

	string service = opt[12].value_or(type == Type::Socks5 ? "1080" : "3128");
	try {
		port = uint16_t(std::stoul(service));
	} catch (...) {
		throw std::invalid_argument("Invalid proxy server port in URL: " + service);
	}
}

ProxyServer::ProxyServer(Type type_, string hostname_, uint16_t port_)
    : type(type_), hostname(std::move(hostname_)), port(port_) {}

ProxyServer::ProxyServer(Type type_, string hostname_, uint16_t port_, string username_,
                         string password_)
    : type(type_), hostname(std::move(hostname_)), port(port_), username(std::move(username_)),
      password(std::move(password_)) {}

} // namespace rtc
