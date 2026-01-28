/**************************************************************************/
/*  test_ip_address.h                                                     */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "core/io/ip_address.h"

#include "tests/test_macros.h"

namespace TestIPAddress {

struct IPTester : public IPAddress {
public:
	static bool _test_v4(const char *p_ip) {
		uint8_t ip[4];
		return _parse_ipv4(String::utf8(p_ip), 0, ip);
	}

	static bool _test_v6(const char *p_ip) {
		IPAddress addr;
		return _parse_ipv6(String::utf8(p_ip), addr);
	}

	static bool test_v4(const char *p_ip, bool p_valid) {
		bool is_6 = _test_v6(p_ip);
		bool is_4 = _test_v4(p_ip);
		bool is_valid = IPAddress::is_valid_ip_address(String::utf8(p_ip));
		IPAddress ip(p_ip);
		return is_4 == p_valid && is_6 == false && is_valid == p_valid && ip.is_valid() == p_valid && ip.is_wildcard() == false;
	}

	static bool test_v6(const char *p_ip, bool p_valid) {
		bool is_6 = _test_v6(p_ip);
		bool is_4 = _test_v4(p_ip);
		bool is_valid = is_valid_ip_address(String::utf8(p_ip));
		IPAddress ip(p_ip);
		return is_4 == false && is_6 == p_valid && is_valid == p_valid && ip.is_valid() == p_valid && ip.is_wildcard() == false;
	}

	static bool test_wildcard(const char *p_ip) {
		bool is_6 = _test_v6(p_ip);
		bool is_4 = _test_v4(p_ip);
		bool is_valid = is_valid_ip_address(String::utf8(p_ip));
		IPAddress ip(p_ip);
		return is_6 == false && is_4 == false && ip.is_valid() == false && is_valid == false && ip.is_wildcard() == true;
	}
};

TEST_CASE("[IPAddress] Wildcard and misc") {
	ERR_PRINT_OFF;

	auto test_ip = [](const char *l_ip, bool l_valid) {
		return IPAddress(l_ip).is_wildcard() == false && l_valid == (IPTester::test_v4(l_ip, true) || IPTester::test_v6(l_ip, true));
	};

	CHECK(IPTester::test_wildcard("*"));

	CHECK(test_ip("", false));

	CHECK(test_ip(" ", false));

	CHECK(test_ip("::", true));
	CHECK(test_ip("0.0.0.0", true));

	CHECK(test_ip("not an ip", false));
	CHECK(test_ip("surely.not:an:ip", false));

	ERR_PRINT_ON;
}

TEST_CASE("[IPAddress] IPv4 is_valid") {
	ERR_PRINT_OFF;

	auto test_ip = [](const char *l_ip, bool l_valid) {
		return IPTester::test_v4(l_ip, l_valid);
	};

	// Valid IPs
	CHECK(test_ip("127.0.0.1", true));
	CHECK(test_ip("255.255.255.255", true));
	CHECK(test_ip("0.0.0.0", true));

	// Invalid IPs
	CHECK(test_ip(" 127.0.0.1", false));
	CHECK(test_ip("127.0.0.1 ", false));
	CHECK(test_ip(" 127.0.0.1 ", false));
	CHECK(test_ip("127.0.0.-1", false));
	CHECK(test_ip("127.0.0.256", false));
	CHECK(test_ip("127.0.0.", false));
	CHECK(test_ip(".0.0.1", false));
	CHECK(test_ip("127.0.0.1.", false));
	CHECK(test_ip(".127.0.0.1", false));
	CHECK(test_ip("0.127.0.0.1", false));
	CHECK(test_ip("127.0.0.1.0", false));
	CHECK(test_ip(".....", false));
	CHECK(test_ip("....", false));
	CHECK(test_ip("...", false));
	CHECK(test_ip("..", false));
	CHECK(test_ip(".", false));
	CHECK(test_ip("", false));
	ERR_PRINT_ON;
}

TEST_CASE("[IPAddress] IPv4 Parsing") {
	auto test_ip = [](const char *l_ip, uint32_t l_test) {
		l_test = BSWAP32(l_test);
		return memcmp(IPAddress(l_ip).get_ipv4(), &l_test, 4) == 0 && IPTester::test_v4(l_ip, true);
	};
	CHECK(test_ip("127.0.0.1", 2130706433));
	CHECK(test_ip("255.0.0.0", 255 << 24));
	CHECK(test_ip("0.255.0.0", 255 << 16));
	CHECK(test_ip("0.0.255.0", 255 << 8));
	CHECK(test_ip("0.0.0.255", 255));
	CHECK(test_ip("127.0.0.0", 127 << 24));
	CHECK(test_ip("0.127.0.0", 127 << 16));
	CHECK(test_ip("0.0.127.0", 127 << 8));
	CHECK(test_ip("0.0.0.127", 127));
	CHECK(test_ip("1.0.0.0", 1 << 24));
	CHECK(test_ip("0.1.0.0", 1 << 16));
	CHECK(test_ip("0.0.1.0", 1 << 8));
	CHECK(test_ip("0.0.0.1", 1));
}

TEST_CASE("[IPAddress] IPv6 is_valid") {
	ERR_PRINT_OFF;

	auto test_ip = [](const char *l_ip, bool l_valid) {
		return IPTester::test_v6(l_ip, l_valid);
	};

	// Valid IPs
	CHECK(test_ip("::", true));

	CHECK(test_ip("::1", true));
	CHECK(test_ip("::1:1", true));
	CHECK(test_ip("::1:1:1", true));
	CHECK(test_ip("::1:1:1:1", true));
	CHECK(test_ip("::1:1:1:1:1", true));
	CHECK(test_ip("::1:1:1:1:1:1", true));
	CHECK(test_ip("::1:1:1:1:1:1:1", true));
	CHECK(test_ip("1:1:1:1:1:1:1:1", true));
	CHECK(test_ip("1:1:1:1:1:1:1::", true));
	CHECK(test_ip("1:1:1:1:1:1::", true));
	CHECK(test_ip("1:1:1:1:1::", true));
	CHECK(test_ip("1:1:1:1::", true));
	CHECK(test_ip("1:1:1::", true));
	CHECK(test_ip("1:1::", true));
	CHECK(test_ip("1::", true));

	CHECK(test_ip("1::1:1:1:1:1:1", true));
	CHECK(test_ip("1:1::1:1:1:1:1", true));
	CHECK(test_ip("1:1:1::1:1:1:1", true));
	CHECK(test_ip("1:1:1:1::1:1:1", true));
	CHECK(test_ip("1:1:1:1:1::1:1", true));
	CHECK(test_ip("1:1:1:1:1:1::1", true));
	CHECK(test_ip("ffff:ffff:ffff:ffff:ffff:ffff:ffff:ffff", true));
	CHECK(test_ip("FFFF:FFFF:FFFF:FFFF:FFFF:FFFF:FFFF:FFFF", true));
	CHECK(test_ip("::ffff", true));
	CHECK(test_ip("::FFFF", true));
	CHECK(test_ip("ffff::", true));
	CHECK(test_ip("FFFF::", true));

	// IPv4-mapped address
	CHECK(test_ip("::ffff:127.0.0.1", true));
	CHECK(test_ip("::FFFF:127.0.0.1", true));
	CHECK(test_ip("0::ffff:127.0.0.1", true));
	CHECK(test_ip("0:0::ffff:127.0.0.1", true));
	CHECK(test_ip("0:0:0::ffff:127.0.0.1", true));
	CHECK(test_ip("0:0:0:0::ffff:127.0.0.1", true));
	CHECK(test_ip("0:0:0:0:0:ffff:127.0.0.1", true));

	// Invalid IPs
	CHECK(test_ip(" ::", false));
	CHECK(test_ip(":: ", false));
	CHECK(test_ip("::-0", false));
	CHECK(test_ip("1:1:1:1:1:1:1:g", false));
	CHECK(test_ip(" 1:1:1:1:1:1:1:1", false));
	CHECK(test_ip("1:1:1:1:1:1:1:1 ", false));
	CHECK(test_ip(" 1:1:1:1:1:1:1:1 ", false));
	CHECK(test_ip("1:1:1:1:1:1:1:1:1", false));
	CHECK(test_ip("1:1:1:1:1:1:1:1::", false));
	CHECK(test_ip("::1:1:1:1:1:1:1:1", false));
	CHECK(test_ip("::1::1", false));
	CHECK(test_ip(":", false));
	CHECK(test_ip(":::", false));
	CHECK(test_ip("::::", false));
	CHECK(test_ip(":::::", false));
	CHECK(test_ip("::::::", false));
	CHECK(test_ip(":::::::", false));
	CHECK(test_ip("::::::::", false));

	// IPv4-mapped address
	CHECK(test_ip("1::ffff:127.0.0.1", false));
	CHECK(test_ip("::1:ffff:127.0.0.1", false));
	CHECK(test_ip("::ffff:127.0.256.1", false));
	CHECK(test_ip("::ffff:127.0.0.256", false));
	CHECK(test_ip("::ffff:256.0.0.1", false));
	CHECK(test_ip("::ffff:127.0.0.1.1", false));
	CHECK(test_ip("::ffff:127.0.0.1.", false));
	CHECK(test_ip("::ffff:127.0.0.", false));
	CHECK(test_ip("::ffff:127.0.0", false));
	CHECK(test_ip("::ffff:127.0.", false));
	CHECK(test_ip("::ffff:127.0", false));
	CHECK(test_ip("::ffff:127.", false));
	CHECK(test_ip("::ffff:", false));

	// This is a valid IPv6 address (non IPv4-mapped)
	CHECK(test_ip("::ffff:127", true));

	ERR_PRINT_ON;
}

TEST_CASE("[IPAddress] IPv6 Parsing") {
	struct InitIP {
		uint16_t data[8];
	};
	auto test_ip = [](const char *l_ip, InitIP l_test) {
		for (int i = 0; i < 8; i++) {
			l_test.data[i] = BSWAP16(l_test.data[i]);
		}
		return memcmp(IPAddress(l_ip).get_ipv6(), l_test.data, 16) == 0 && IPTester::test_v6(l_ip, true);
	};
	CHECK(IPAddress("::").is_valid() == true);

	CHECK(test_ip("::1", { 0, 0, 0, 0, 0, 0, 0, 1 }));
	CHECK(test_ip("::1:1", { 0, 0, 0, 0, 0, 0, 1, 1 }));
	CHECK(test_ip("::1:1:1", { 0, 0, 0, 0, 0, 1, 1, 1 }));
	CHECK(test_ip("::1:1:1:1", { 0, 0, 0, 0, 1, 1, 1, 1 }));
	CHECK(test_ip("::1:1:1:1:1", { 0, 0, 0, 1, 1, 1, 1, 1 }));
	CHECK(test_ip("::1:1:1:1:1:1", { 0, 0, 1, 1, 1, 1, 1, 1 }));
	CHECK(test_ip("::1:1:1:1:1:1:1", { 0, 1, 1, 1, 1, 1, 1, 1 }));
	CHECK(test_ip("1:1:1:1:1:1:1:1", { 1, 1, 1, 1, 1, 1, 1, 1 }));
	CHECK(test_ip("1:1:1:1:1:1:1::", { 1, 1, 1, 1, 1, 1, 1, 0 }));
	CHECK(test_ip("1:1:1:1:1:1::", { 1, 1, 1, 1, 1, 1, 0, 0 }));
	CHECK(test_ip("1:1:1:1:1::", { 1, 1, 1, 1, 1, 0, 0, 0 }));
	CHECK(test_ip("1:1:1:1::", { 1, 1, 1, 1, 0, 0, 0, 0 }));
	CHECK(test_ip("1:1:1::", { 1, 1, 1, 0, 0, 0, 0, 0 }));
	CHECK(test_ip("1:1::", { 1, 1, 0, 0, 0, 0, 0, 0 }));
	CHECK(test_ip("1::", { 1, 0, 0, 0, 0, 0, 0, 0 }));

	CHECK(test_ip("ffff::", { 0xFFFF, 0, 0, 0, 0, 0, 0, 0 }));
	CHECK(test_ip("::ffff", { 0, 0, 0, 0, 0, 0, 0, 0xFFFF }));
	CHECK(test_ip("::fffe:0", { 0, 0, 0, 0, 0, 0, 0xFFFE, 0 }));
	CHECK(test_ip("0:fffe::", { 0, 0xFFFE, 0, 0, 0, 0, 0, 0 }));

	// IPv4-mapped address
	CHECK(test_ip("::ffff:127.0.0.1", { 0, 0, 0, 0, 0, 0xFFFF, 0x7F00, 1 }));
	CHECK(test_ip("::FFFF:127.0.0.1", { 0, 0, 0, 0, 0, 0xFFFF, 0x7F00, 1 }));
	CHECK(test_ip("0::ffff:127.0.0.1", { 0, 0, 0, 0, 0, 0xFFFF, 0x7F00, 1 }));
	CHECK(test_ip("0:0::ffff:127.0.0.1", { 0, 0, 0, 0, 0, 0xFFFF, 0x7F00, 1 }));
	CHECK(test_ip("0:0:0::ffff:127.0.0.1", { 0, 0, 0, 0, 0, 0xFFFF, 0x7F00, 1 }));
	CHECK(test_ip("0:0:0:0::ffff:127.0.0.1", { 0, 0, 0, 0, 0, 0xFFFF, 0x7F00, 1 }));
	CHECK(test_ip("0:0:0:0:0:ffff:127.0.0.1", { 0, 0, 0, 0, 0, 0xFFFF, 0x7F00, 1 }));
}

} // namespace TestIPAddress
