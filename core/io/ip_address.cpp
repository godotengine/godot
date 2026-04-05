/**************************************************************************/
/*  ip_address.cpp                                                        */
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

#include "ip_address.h"
/*
IPAddress::operator Variant() const {
	return operator String();
}*/

IPAddress::operator String() const {
	if (wildcard) {
		return "*";
	}

	if (!valid) {
		return "";
	}

	if (is_ipv4()) {
		// IPv4 address mapped to IPv6.
		return itos(field8[12]) + "." + itos(field8[13]) + "." + itos(field8[14]) + "." + itos(field8[15]);
	}
	String ret;
	for (int i = 0; i < 8; i++) {
		if (i > 0) {
			ret = ret + ":";
		}
		uint16_t num = (field8[i * 2] << 8) + field8[i * 2 + 1];
		ret = ret + String::num_int64(num, 16);
	}

	return ret;
}

bool IPAddress::_parse_ipv6(const String &p_string, IPAddress &r_ip) {
	int len = p_string.length();
	const char32_t *buf = p_string.ptr();

	int cur = 0;
	int shift = -1;
	for (int i = 0; i < len; i++) {
		for (int j = i; j < len; j++) {
			char32_t c = buf[j];
			if (c == ':') {
				if (j + 1 == len) {
					return false; // Can't end with a column (unless part of shortening).
				}
				if (buf[j + 1] == ':') {
					if (shift > -1) {
						return false; // Only one shortening allowed.
					} else if (j == 0) {
						shift = cur;
					} else {
						shift = cur + 1;
					}
					j++;
				} else if (i == j) {
					return false; // Stray column.
				}
				i = j;
				break;
			}
			if (j - i > 3) {
				return false;
			}
			if (c >= '0' && c <= '9') {
				r_ip.field16[cur] = r_ip.field16[cur] << 4;
				r_ip.field16[cur] |= c - '0';
			} else if (c >= 'a' && c <= 'f') {
				r_ip.field16[cur] = r_ip.field16[cur] << 4;
				r_ip.field16[cur] |= 10 + (c - 'a');
			} else if (c >= 'A' && c <= 'F') {
				r_ip.field16[cur] = r_ip.field16[cur] << 4;
				r_ip.field16[cur] |= 10 + (c - 'A');
			} else if (c == '.') {
				// IPv4 mapped IPv6 (e.g. "::FFFF:127.0.0.1").
				if (cur < 1 || r_ip.field16[cur - 1] != 0xFFFF) {
					return false; // IPv6 part must end with FFFF.
				}
				if (shift < 0 && cur != 6) {
					return false; // Needs 5 zeros, and FFFF "0:0:0:0:0:FFFF:127.0.0.1".
				}
				// Only empty bytes allowed before FFFF.
				r_ip.field16[cur] = 0;
				r_ip.field16[cur - 1] = 0;
				while (cur > 0) {
					cur--;
					if (r_ip.field16[cur] != 0) {
						return false;
					}
				}
				r_ip.field16[5] = 0xFFFF;
				return _parse_ipv4(p_string, i, &r_ip.field8[12]);
			} else {
				return false;
			}
			if (j + 1 == len) {
				i = j;
			}
		}
		r_ip.field16[cur] = BSWAP16(r_ip.field16[cur]);
		cur += 1;
		if (cur > 8 || (cur == 8 && i + 1 != len)) {
			return false;
		}
	}
	if (shift < 0) {
		return cur == 8; // Should have parsed 8 16-bits ints.
	} else if (shift > 7) {
		return false; // Can't shorten more than this.
	} else if (shift == cur) {
		return true; // Nothing to do, end is assumed zeroized.
	}
	// Shift bytes.
	int pad = 8 - cur;
	int blank_end = shift + pad;
	for (int i = 7; i > shift; i--) {
		if (i < blank_end) {
			r_ip.field16[i] = 0;
		} else {
			r_ip.field16[i] = r_ip.field16[i - pad];
		}
	}
	return true;
}

bool IPAddress::_parse_ipv4(const String &p_string, int p_start, uint8_t *r_dest) {
	int len = p_string.length();
	const char32_t *buf = p_string.ptr();

	int cur = 0;
	uint16_t next = 0;
	bool parsed = false;
	for (int i = p_start; i < len; i++) {
		char32_t c = buf[i];
		if (c == '.') {
			if (!parsed) {
				return false;
			}
			parsed = false;
			r_dest[cur] = next;
			next = 0;
			cur++;
			if (cur > 3) {
				return false;
			}
		} else if (c >= '0' && c <= '9') {
			parsed = true;
			next *= 10;
			next += c - '0';
			if (next > 255) {
				return false;
			}
		} else {
			return false; // Invalid char.
		}
	}
	if (!parsed) {
		return false;
	}
	r_dest[cur] = next;
	return parsed && cur == 3;
}

void IPAddress::clear() {
	memset(&field8[0], 0, sizeof(field8));
	valid = false;
	wildcard = false;
}

bool IPAddress::is_ipv4() const {
	return (field32[0] == 0 && field32[1] == 0 && field16[4] == 0 && field16[5] == 0xffff);
}

const uint8_t *IPAddress::get_ipv4() const {
	ERR_FAIL_COND_V_MSG(!is_ipv4(), &(field8[12]), "IPv4 requested, but current IP is IPv6."); // Not the correct IPv4 (it's an IPv6), but we don't want to return a null pointer risking an engine crash.
	return &(field8[12]);
}

void IPAddress::set_ipv4(const uint8_t *p_ip) {
	clear();
	valid = true;
	field16[5] = 0xffff;
	field32[3] = *((const uint32_t *)p_ip);
}

const uint8_t *IPAddress::get_ipv6() const {
	return field8;
}

void IPAddress::set_ipv6(const uint8_t *p_buf) {
	clear();
	valid = true;
	for (int i = 0; i < 16; i++) {
		field8[i] = p_buf[i];
	}
}

bool IPAddress::is_valid_ip_address(const String &p_string) {
	IPAddress addr;
	if (p_string.length() < IPV6_MAX_STRING_LENGTH && p_string.contains_char(':')) {
		return _parse_ipv6(p_string, addr);
	} else if (p_string.length() < IPV4_MAX_STRING_LENGTH) { // Try IPv4.
		return _parse_ipv4(p_string, 0, &addr.field8[12]);
	}
	return false;
}

IPAddress::IPAddress(const String &p_string) {
	clear();

	if (p_string == "*") {
		// Wildcard (not a valid IP).
		wildcard = true;

	} else if (p_string.length() < IPV6_MAX_STRING_LENGTH && p_string.contains_char(':')) {
		// IPv6.
		valid = _parse_ipv6(p_string, *this);
		ERR_FAIL_COND_MSG(!valid, "Invalid IPv6 address: " + p_string);

	} else if (p_string.length() < IPV4_MAX_STRING_LENGTH) {
		// IPv4 (mapped to IPv6 internally).
		field16[5] = 0xffff;
		valid = _parse_ipv4(p_string, 0, &field8[12]);
		ERR_FAIL_COND_MSG(!valid, "Invalid IPv4 address: " + p_string);

	} else {
		ERR_PRINT("Invalid IP address.");
	}
}

_FORCE_INLINE_ static void _32_to_buf(uint8_t *p_dst, uint32_t p_n) {
	p_dst[0] = (p_n >> 24) & 0xff;
	p_dst[1] = (p_n >> 16) & 0xff;
	p_dst[2] = (p_n >> 8) & 0xff;
	p_dst[3] = (p_n >> 0) & 0xff;
}

IPAddress::IPAddress(uint32_t p_a, uint32_t p_b, uint32_t p_c, uint32_t p_d, bool is_v6) {
	clear();
	valid = true;
	if (!is_v6) {
		// Mapped to IPv6.
		field16[5] = 0xffff;
		field8[12] = p_a;
		field8[13] = p_b;
		field8[14] = p_c;
		field8[15] = p_d;
	} else {
		_32_to_buf(&field8[0], p_a);
		_32_to_buf(&field8[4], p_b);
		_32_to_buf(&field8[8], p_c);
		_32_to_buf(&field8[12], p_d);
	}
}
