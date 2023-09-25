/**
 * Copyright (c) 2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "stun.h"
#include "base64.h"
#include "const_time.h"
#include "crc32.h"
#include "juice.h"
#include "log.h"
#include "udp.h"

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STUN_MAGIC 0x2112A442
#define STUN_FINGERPRINT_XOR 0x5354554E // "STUN"
#define STUN_ATTR_SIZE sizeof(struct stun_attr)

// STUN_MAX_PASSWORD_LEN > HASH_SHA256_SIZE > HASH_MD5_SIZE
#define MAX_HMAC_KEY_LEN STUN_MAX_PASSWORD_LEN

#define MAX_HMAC_INPUT_LEN (STUN_MAX_USERNAME_LEN + STUN_MAX_REALM_LEN + STUN_MAX_PASSWORD_LEN + 2)

#define MAX_USERHASH_INPUT_LEN (STUN_MAX_USERNAME_LEN + STUN_MAX_REALM_LEN + 1)

#ifndef htonll
#define htonll(x)                                                                                  \
	((uint64_t)(((uint64_t)htonl((uint32_t)(x))) << 32) | (uint64_t)htonl((uint32_t)((x) >> 32)))
#endif
#ifndef ntohll
#define ntohll(x) htonll(x)
#endif

static size_t align32(size_t len) {
	while (len & 0x03)
		++len;
	return len;
}

static size_t generate_hmac_key(const stun_message_t *msg, const char *password, void *key) {
	if (*msg->credentials.realm != '\0') {
		// long-term credentials
		if (*msg->credentials.username == '\0')
			JLOG_WARN("Generating HMAC key for long-term credentials with empty STUN username");

		char input[MAX_HMAC_INPUT_LEN];
		int input_len = snprintf(input, MAX_HMAC_INPUT_LEN, "%s:%s:%s", msg->credentials.username,
		                         msg->credentials.realm, password ? password : "");
		if (input_len < 0)
			return 0;

		if (input_len >= MAX_HMAC_INPUT_LEN)
			input_len = MAX_HMAC_INPUT_LEN - 1;

		switch (msg->credentials.password_algorithm) {
		case STUN_PASSWORD_ALGORITHM_SHA256:
			hash_sha256(input, input_len, key);
			return HASH_SHA256_SIZE;
		default:
			hash_md5(input, input_len, key);
			return HASH_MD5_SIZE;
		}
	} else {
		// short-term credentials
		int key_len = snprintf((char *)key, MAX_HMAC_KEY_LEN, "%s", password ? password : "");
		if (key_len < 0)
			return 0;

		if (key_len >= MAX_HMAC_KEY_LEN)
			key_len = MAX_HMAC_KEY_LEN - 1;

		return key_len;
	}
}

static size_t generate_password_algorithms_attr(uint8_t *attr) {
	// attr size must be at least STUN_PASSWORD_ALGORITHMS_ATTR_MAX_SIZE
	struct stun_value_password_algorithm *pwa = (struct stun_value_password_algorithm *)attr;
	pwa->algorithm = htons(STUN_PASSWORD_ALGORITHM_SHA256);
	pwa->parameters_length = 0;
	++pwa;
	pwa->algorithm = htons(STUN_PASSWORD_ALGORITHM_MD5);
	pwa->parameters_length = 0;
	++pwa;
	return (uint8_t *)pwa - attr;
}

int stun_write(void *buf, size_t size, const stun_message_t *msg, const char *password) {
	uint8_t *begin = buf;
	uint8_t *pos = begin;
	uint8_t *end = begin + size;

	JLOG_VERBOSE("Writing STUN message, class=0x%X, method=0x%X", (unsigned int)msg->msg_class,
	             (unsigned int)msg->msg_method);

	size_t len =
	    stun_write_header(pos, end - pos, msg->msg_class, msg->msg_method, msg->transaction_id);
	if (len <= 0)
		goto overflow;
	pos += len;
	uint8_t *attr_begin = pos;

	if (msg->error_code) {
		const char *reason = stun_get_error_reason(msg->error_code);
		char buffer[sizeof(struct stun_value_error_code) + STUN_MAX_ERROR_REASON_LEN + 1];
		struct stun_value_error_code *error = (struct stun_value_error_code *)buffer;
		memset(error, 0, sizeof(*error));
		error->code_class = (msg->error_code / 100) & 0x07;
		error->code_number = msg->error_code % 100;
		strcpy((char *)error->reason, reason);
		len = stun_write_attr(pos, end - pos, STUN_ATTR_ERROR_CODE, error,
		                      sizeof(struct stun_value_error_code) + strlen(reason));
		if (len <= 0)
			goto overflow;
		pos += len;
	}
	if (msg->mapped.len) {
		JLOG_VERBOSE("Writing XOR mapped address");
		uint8_t value[32];
		uint8_t mask[16];
		*((uint32_t *)mask) = htonl(STUN_MAGIC);
		memcpy(mask + 4, msg->transaction_id, 12);
		int value_len = stun_write_value_mapped_address(
		    value, 32, (const struct sockaddr *)&msg->mapped.addr, msg->mapped.len, mask);
		if (value_len > 0) {
			len = stun_write_attr(pos, end - pos, STUN_ATTR_XOR_MAPPED_ADDRESS, value, value_len);
			if (len <= 0)
				goto overflow;
			pos += len;
		}
	}
	if (msg->priority) {
		uint32_t priority = htonl(msg->priority);
		len = stun_write_attr(pos, end - pos, STUN_ATTR_PRIORITY, &priority, 4);
		if (len <= 0)
			goto overflow;
		pos += len;
	}
	if (msg->use_candidate) {
		len = stun_write_attr(pos, end - pos, STUN_ATTR_USE_CANDIDATE, NULL, 0);
		if (len <= 0)
			goto overflow;
		pos += len;
	}
	if (msg->ice_controlling) {
		uint64_t ice_controlling = htonll(msg->ice_controlling);
		len = stun_write_attr(pos, end - pos, STUN_ATTR_ICE_CONTROLLING, &ice_controlling, 8);
		if (len <= 0)
			goto overflow;
		pos += len;
	}
	if (msg->ice_controlled) {
		uint64_t ice_controlled = htonll(msg->ice_controlled);
		len = stun_write_attr(pos, end - pos, STUN_ATTR_ICE_CONTROLLED, &ice_controlled, 8);
		if (len <= 0)
			goto overflow;
		pos += len;
	}
	if (msg->channel_number) {
		struct stun_value_channel_number channel_number;
		memset(&channel_number, 0, sizeof(channel_number));
		channel_number.channel_number = htons(msg->channel_number);
		len = stun_write_attr(pos, end - pos, STUN_ATTR_CHANNEL_NUMBER, &channel_number,
		                      sizeof(channel_number));
		if (len <= 0)
			goto overflow;
		pos += len;
	}
	if (msg->lifetime_set || msg->lifetime) {
		uint32_t lifetime = htonl(msg->lifetime);
		len = stun_write_attr(pos, end - pos, STUN_ATTR_LIFETIME, &lifetime, 4);
		if (len <= 0)
			goto overflow;
		pos += len;
	}
	if (msg->peer.len) {
		JLOG_VERBOSE("Writing XOR peer address");
		uint8_t value[32];
		uint8_t mask[16];
		*((uint32_t *)mask) = htonl(STUN_MAGIC);
		memcpy(mask + 4, msg->transaction_id, 12);
		int value_len = stun_write_value_mapped_address(
		    value, 32, (const struct sockaddr *)&msg->peer.addr, msg->peer.len, mask);
		if (value_len > 0) {
			len = stun_write_attr(pos, end - pos, STUN_ATTR_XOR_PEER_ADDRESS, value, value_len);
			if (len <= 0)
				goto overflow;
			pos += len;
		}
	}
	if (msg->relayed.len) {
		JLOG_VERBOSE("Writing XOR relay address");
		uint8_t value[32];
		uint8_t mask[16];
		*((uint32_t *)mask) = htonl(STUN_MAGIC);
		memcpy(mask + 4, msg->transaction_id, 12);
		int value_len = stun_write_value_mapped_address(
		    value, 32, (const struct sockaddr *)&msg->relayed.addr, msg->relayed.len, mask);
		if (value_len > 0) {
			len = stun_write_attr(pos, end - pos, STUN_ATTR_XOR_RELAYED_ADDRESS, value, value_len);
			if (len <= 0)
				goto overflow;
			pos += len;
		}
	}
	if (msg->data) {
		len = stun_write_attr(pos, end - pos, STUN_ATTR_DATA, (const uint8_t *)msg->data,
		                      msg->data_size);
		if (len <= 0)
			goto overflow;
		pos += len;
	}
	if (msg->even_port) {
		struct stun_value_even_port even_port;
		memset(&even_port, 0, sizeof(even_port));
		if (msg->next_port)
			even_port.r |= 0x80;
		len = stun_write_attr(pos, end - pos, STUN_ATTR_CHANNEL_NUMBER, &even_port,
		                      sizeof(even_port));
		if (len <= 0)
			goto overflow;
		pos += len;
	}
	if (msg->requested_transport) {
		struct stun_value_requested_transport requested_transport;
		memset(&requested_transport, 0, sizeof(requested_transport));
		requested_transport.protocol = 17;
		len = stun_write_attr(pos, end - pos, STUN_ATTR_REQUESTED_TRANSPORT, &requested_transport,
		                      sizeof(requested_transport));
		if (len <= 0)
			goto overflow;
		pos += len;
	}
	if (msg->dont_fragment) {
		len = stun_write_attr(pos, end - pos, STUN_ATTR_DONT_FRAGMENT, NULL, 0);
		if (len <= 0)
			goto overflow;
		pos += len;
	}
	if (msg->reservation_token) {
		uint64_t reservation_token = htonll(msg->reservation_token);
		len = stun_write_attr(pos, end - pos, STUN_ATTR_RESERVATION_TOKEN, &reservation_token, 8);
		if (len <= 0)
			goto overflow;
		pos += len;
	}

	const char *software = "libjuice";
	len = stun_write_attr(pos, end - pos, STUN_ATTR_SOFTWARE, software, strlen(software));
	if (len <= 0)
		goto overflow;
	pos += len;

	if (msg->msg_class == STUN_CLASS_REQUEST) {
		if (msg->credentials.enable_userhash) {
			len = stun_write_attr(pos, end - pos, STUN_ATTR_USERHASH, msg->credentials.userhash,
			                      USERHASH_SIZE);
			if (len <= 0)
				goto overflow;
			pos += len;

		} else if (*msg->credentials.username != '\0') {
			len = stun_write_attr(pos, end - pos, STUN_ATTR_USERNAME, msg->credentials.username,
			                      strlen(msg->credentials.username));
			if (len <= 0)
				goto overflow;
			pos += len;
		}
	}
	if (msg->msg_class == STUN_CLASS_REQUEST ||
	    (msg->msg_class == STUN_CLASS_RESP_ERROR &&
	     (msg->error_code == 401 || msg->error_code == 438) // Unauthenticated or Stale Nonce
	     )) {
		if (*msg->credentials.realm != '\0') {
			len = stun_write_attr(pos, end - pos, STUN_ATTR_REALM, msg->credentials.realm,
			                      strlen(msg->credentials.realm));
			if (len <= 0)
				goto overflow;
			pos += len;
		}
		if (*msg->credentials.nonce != '\0') {
			len = stun_write_attr(pos, end - pos, STUN_ATTR_NONCE, msg->credentials.nonce,
			                      strlen(msg->credentials.nonce));
			if (len <= 0)
				goto overflow;
			pos += len;

			if (msg->credentials.password_algorithm > 0) {
				len = stun_write_attr(pos, end - pos, STUN_ATTR_PASSWORD_ALGORITHMS,
				                      msg->credentials.password_algorithms_value,
				                      msg->credentials.password_algorithms_value_size);
				if (len <= 0)
					goto overflow;
				pos += len;

			} else if (msg->msg_class != STUN_CLASS_REQUEST) {
				uint8_t pwa_value[STUN_MAX_PASSWORD_ALGORITHMS_VALUE_SIZE];
				size_t pwa_size = generate_password_algorithms_attr(pwa_value);
				len = stun_write_attr(pos, end - pos, STUN_ATTR_PASSWORD_ALGORITHMS, pwa_value,
				                      pwa_size);
				if (len <= 0)
					goto overflow;
				pos += len;
			}

			if (msg->msg_class == STUN_CLASS_REQUEST &&
			    msg->credentials.password_algorithm != STUN_PASSWORD_ALGORITHM_UNSET) {
				struct stun_value_password_algorithm pwa;
				pwa.algorithm = htons(msg->credentials.password_algorithm);
				len = stun_write_attr(pos, end - pos, STUN_ATTR_PASSWORD_ALGORITHM, &pwa,
				                      sizeof(pwa));
				if (len <= 0)
					goto overflow;
				pos += len;
			}
		}
	}
	if (msg->msg_class != STUN_CLASS_INDICATION && password) {
		uint8_t key[MAX_HMAC_KEY_LEN];
		size_t key_len = generate_hmac_key(msg, password, key);

		size_t tmp_length = pos - attr_begin + STUN_ATTR_SIZE + HMAC_SHA1_SIZE;
		stun_update_header_length(begin, tmp_length);

		uint8_t hmac[HMAC_SHA1_SIZE];
		hmac_sha1(begin, pos - begin, key, key_len, hmac);
		len = stun_write_attr(pos, end - pos, STUN_ATTR_MESSAGE_INTEGRITY, hmac, HMAC_SHA1_SIZE);
		if (len <= 0)
			goto overflow;
		pos += len;

		// According to RFC 8489, the agent must include both MESSAGE-INTEGRITY and
		// MESSAGE-INTEGRITY-SHA256. However, this makes legacy agents and servers fail with error
		// 420 Unknown Attribute. Therefore, unless the password algorithm SHA-256 is enabled, only
		// MESSAGE-INTEGRITY is included in the message for compatibility.
		if (msg->credentials.password_algorithm != STUN_PASSWORD_ALGORITHM_UNSET) {
			// If the response contains a PASSWORD-ALGORITHMS attribute, all the
			// subsequent requests MUST be authenticated using MESSAGE-INTEGRITY-
			// SHA256 only.
			size_t tmp_length = pos - attr_begin + STUN_ATTR_SIZE + HMAC_SHA256_SIZE;
			stun_update_header_length(begin, tmp_length);

			uint8_t hmac[HMAC_SHA256_SIZE];
			hmac_sha256(begin, pos - begin, key, key_len, hmac);
			len = stun_write_attr(pos, end - pos, STUN_ATTR_MESSAGE_INTEGRITY_SHA256, hmac,
			                      HMAC_SHA256_SIZE);
			if (len <= 0)
				goto overflow;
			pos += len;
		}
	}

	size_t length = pos - attr_begin + STUN_ATTR_SIZE + 4;
	if (length & 0x03) {
		JLOG_ERROR("Written STUN message length is not multiple of 4, length=%zu", length);
		return -1;
	}
	stun_update_header_length(begin, length);

	uint32_t fingerprint = htonl(CRC32(buf, pos - begin) ^ STUN_FINGERPRINT_XOR);
	len = stun_write_attr(pos, end - pos, STUN_ATTR_FINGERPRINT, &fingerprint, 4);
	if (len <= 0)
		goto overflow;
	pos += len;

	return (int)(pos - begin);

overflow:
	JLOG_ERROR("Not enough space in buffer for STUN message, size=%zu", size);
	return -1;
}

int stun_write_header(void *buf, size_t size, stun_class_t class, stun_method_t method,
                      const uint8_t *transaction_id) {
	if (size < sizeof(struct stun_header))
		return -1;

	uint16_t type = (uint16_t) class | (uint16_t)method;

	struct stun_header *header = buf;
	header->type = htons(type);
	header->length = htons(0);
	header->magic = htonl(STUN_MAGIC);
	memcpy(header->transaction_id, transaction_id, STUN_TRANSACTION_ID_SIZE);

	return sizeof(struct stun_header);
}

size_t stun_update_header_length(void *buf, size_t length) {
	struct stun_header *header = buf;
	size_t previous = ntohs(header->length);
	header->length = htons((uint16_t)length);
	return previous;
}

int stun_write_attr(void *buf, size_t size, uint16_t type, const void *value, size_t length) {
	JLOG_VERBOSE("Writing STUN attribute type 0x%X, length=%zu", (unsigned int)type, length);

	if (size < sizeof(struct stun_attr) + length)
		return -1;

	struct stun_attr *attr = buf;
	attr->type = htons(type);
	attr->length = htons((uint16_t)length);

	if (length > 0) {
		memcpy(attr->value, value, length);

		// Pad to align on 4 bytes
		while (length & 0x03)
			attr->value[length++] = 0;
	}

	return (int)(sizeof(struct stun_attr) + length);
}

int stun_write_value_mapped_address(void *buf, size_t size, const struct sockaddr *addr,
                                    socklen_t addrlen, const uint8_t *mask) {
	if (size < sizeof(struct stun_value_mapped_address))
		return -1;

	struct stun_value_mapped_address *value = buf;
	value->padding = 0;
	switch (addr->sa_family) {
	case AF_INET: {
		value->family = STUN_ADDRESS_FAMILY_IPV4;
		if (size < sizeof(struct stun_value_mapped_address) + 4)
			return -1;
		if (addrlen < (socklen_t)sizeof(struct sockaddr_in))
			return -1;
		JLOG_VERBOSE("Writing IPv4 address");
		const struct sockaddr_in *sin = (const struct sockaddr_in *)addr;
		value->port = sin->sin_port ^ *((uint16_t *)mask);
		const uint8_t *bytes = (const uint8_t *)&sin->sin_addr;
		for (int i = 0; i < 4; ++i)
			value->address[i] = bytes[i] ^ mask[i];
		return sizeof(struct stun_value_mapped_address) + 4;
	}
	case AF_INET6: {
		value->family = STUN_ADDRESS_FAMILY_IPV6;
		if (size < sizeof(struct stun_value_mapped_address) + 16)
			return -1;
		if (addrlen < (socklen_t)sizeof(struct sockaddr_in6))
			return -1;
		JLOG_VERBOSE("Writing IPv6 address");
		const struct sockaddr_in6 *sin6 = (const struct sockaddr_in6 *)addr;
		value->port = sin6->sin6_port ^ *((uint16_t *)mask);
		const uint8_t *bytes = (const uint8_t *)&sin6->sin6_addr;
		for (int i = 0; i < 16; ++i)
			value->address[i] = bytes[i] ^ mask[i];
		return sizeof(struct stun_value_mapped_address) + 16;
	}
	default: {
		JLOG_DEBUG("Unknown address family %u", (unsigned int)addr->sa_family);
		return -1;
	}
	}
}

bool is_stun_datagram(const void *data, size_t size) {
	// RFC 8489: The most significant 2 bits of every STUN message MUST be zeroes. This can be used
	// to differentiate STUN packets from other protocols when STUN is multiplexed with other
	// protocols on the same port.
	if (!size || *((uint8_t *)data) & 0xC0) {
		JLOG_VERBOSE("Not a STUN message: first 2 bits are not zeroes");
		return false;
	}

	if (size < sizeof(struct stun_header)) {
		JLOG_VERBOSE("Not a STUN message: message too short, size=%zu", size);
		return false;
	}

	const struct stun_header *header = data;
	if (ntohl(header->magic) != STUN_MAGIC) {
		JLOG_VERBOSE("Not a STUN message: magic number invalid");
		return false;
	}

	// RFC 8489: The message length MUST contain the size of the message in bytes, not including the
	// 20-byte STUN header.  Since all STUN attributes are padded to a multiple of 4 bytes, the last
	// 2 bits of this field are always zero.  This provides another way to distinguish STUN packets
	// from packets of other protocols.
	const size_t length = ntohs(header->length);
	if (length & 0x03) {
		JLOG_VERBOSE("Not a STUN message: invalid length %zu not multiple of 4", length);
		return false;
	}
	if (size != sizeof(struct stun_header) + length) {
		JLOG_VERBOSE("Not a STUN message: invalid length %zu while expecting %zu", length,
		             size - sizeof(struct stun_header));
		return false;
	}

	return true;
}

int stun_read(void *data, size_t size, stun_message_t *msg) {
	memset(msg, 0, sizeof(*msg));

	if (size < sizeof(struct stun_header)) {
		JLOG_ERROR("STUN message too short, size=%zu", size);
		return -1;
	}

	const struct stun_header *header = data;
	const size_t length = ntohs(header->length);
	if (size < sizeof(struct stun_header) + length) {
		JLOG_ERROR("Invalid STUN message length, length=%zu, available=%zu", length,
		           size - sizeof(struct stun_header));
		return -1;
	}

	uint16_t type = ntohs(header->type);
	msg->msg_class = (stun_class_t)(type & STUN_CLASS_MASK);
	msg->msg_method = (stun_method_t)(type & ~STUN_CLASS_MASK);
	memcpy(msg->transaction_id, header->transaction_id, STUN_TRANSACTION_ID_SIZE);
	JLOG_VERBOSE("Reading STUN message, class=0x%X, method=0x%X", (unsigned int)msg->msg_class,
	             (unsigned int)msg->msg_method);

	uint32_t security_bits = 0;

	uint8_t *begin = data;
	uint8_t *attr_begin = begin + sizeof(struct stun_header);
	uint8_t *end = attr_begin + length;
	const uint8_t *pos = attr_begin;
	while (pos < end) {
		int ret = stun_read_attr(pos, end - pos, msg, begin, attr_begin, &security_bits);
		if (ret <= 0) {
			JLOG_DEBUG("Reading STUN attribute failed");
			return -1;
		}
		pos += ret;
	}

	JLOG_VERBOSE("Finished reading STUN attributes");

	stun_credentials_t *credentials = &msg->credentials;

	// RFC 8489: If the response is an error response with an error code of 401 (Unauthenticated) or
	// 438 (Stale Nonce), the client MUST test if the NONCE attribute value starts with the "nonce
	// cookie". If so and the "nonce cookie" has the STUN Security Feature "Password algorithms"
	// bit set to 1 but no PASSWORD-ALGORITHMS attribute is present, then the client MUST NOT retry
	// the request with a new transaction. See
	// https://www.rfc-editor.org/rfc/rfc8489.html#section-9.2.5
	if (msg->msg_class == STUN_CLASS_RESP_ERROR &&
	    (msg->error_code == 401 || msg->error_code == 438) &&
	    security_bits & STUN_SECURITY_PASSWORD_ALGORITHMS_BIT &&
	    credentials->password_algorithms_value_size == 0) {
		JLOG_INFO("STUN Security Feature \"Password algorithms\" bit is set in %u error response "
		          "but the corresponding attribute is missing",
		          msg->error_code);
		msg->error_code = STUN_ERROR_INTERNAL_VALIDATION_FAILED; // so the agent will give up
	}

	// RFC 8489: If the request contains neither the PASSWORD-ALGORITHMS nor the
	// PASSWORD-ALGORITHM algorithm, then the request is processed as though
	// PASSWORD-ALGORITHM were MD5.
	// Otherwise, unless (1) PASSWORD-ALGORITHM and PASSWORD-ALGORITHMS are both
	// present, (2) PASSWORD-ALGORITHMS  matches the value sent in the response that sent
	// this NONCE, and (3) PASSWORD-ALGORITHM matches one of the entries in
	// PASSWORD-ALGORITHMS, the server MUST generate an error response with an error code of
	// 400 (Bad Request). See https://www.rfc-editor.org/rfc/rfc8489.html#section-9.2.4
	if (!STUN_IS_RESPONSE(msg->msg_class)) {
		if (credentials->password_algorithms_value_size == 0 &&
		    credentials->password_algorithm == STUN_PASSWORD_ALGORITHM_UNSET) {
			credentials->password_algorithm = STUN_PASSWORD_ALGORITHM_MD5;

		} else if (credentials->password_algorithm == STUN_PASSWORD_ALGORITHM_UNSET) {
			JLOG_INFO("No suitable password algorithm in STUN request");
			msg->error_code = STUN_ERROR_INTERNAL_VALIDATION_FAILED;

		} else if (credentials->password_algorithms_value_size == 0) {
			JLOG_INFO("Missing password algorithms list in STUN request");
			msg->error_code = STUN_ERROR_INTERNAL_VALIDATION_FAILED;

		} else {
			uint8_t pwa_value[STUN_MAX_PASSWORD_ALGORITHMS_VALUE_SIZE];
			size_t pwa_size = generate_password_algorithms_attr(pwa_value);
			if (pwa_size != credentials->password_algorithms_value_size ||
			    memcmp(credentials->password_algorithms_value, pwa_value, pwa_size) != 0) {
				JLOG_INFO("Password algorithms list is invalid in STUN request");
				msg->error_code = STUN_ERROR_INTERNAL_VALIDATION_FAILED;
			}
		}
	}

	if (security_bits & STUN_SECURITY_USERNAME_ANONYMITY_BIT) {
		JLOG_DEBUG("Remote agent supports user anonymity");
		credentials->enable_userhash = true;
	}

	return (int)(sizeof(struct stun_header) + length);
}

int stun_read_attr(const void *data, size_t size, stun_message_t *msg, uint8_t *begin,
                   uint8_t *attr_begin, uint32_t *security_bits) {
	// RFC 8489: When present, the FINGERPRINT attribute MUST be the last attribute in the
	// message and thus will appear after MESSAGE-INTEGRITY and MESSAGE-INTEGRITY-SHA256.
	if (msg->has_fingerprint) {
		JLOG_DEBUG("Invalid STUN attribute after fingerprint");
		return -1;
	}

	if (size < sizeof(struct stun_attr)) {
		JLOG_VERBOSE("STUN attribute too short");
		return -1;
	}

	const struct stun_attr *attr = data;
	size_t length = ntohs(attr->length);
	stun_attr_type_t type = (stun_attr_type_t)ntohs(attr->type);
	JLOG_VERBOSE("Reading attribute 0x%X, length=%zu", (unsigned int)type, length);
	if (size < sizeof(struct stun_attr) + length) {
		JLOG_DEBUG("STUN attribute length invalid, length=%zu, available=%zu", length,
		           size - sizeof(struct stun_attr));
		return -1;
	}

	// RFC 8489: Note that agents MUST ignore all attributes that follow MESSAGE-INTEGRITY, with
	// the exception of the MESSAGE-INTEGRITY-SHA256 and FINGERPRINT attributes.
	if (msg->has_integrity && type != STUN_ATTR_MESSAGE_INTEGRITY &&
	    type != STUN_ATTR_MESSAGE_INTEGRITY_SHA256 && type != STUN_ATTR_FINGERPRINT) {
		JLOG_DEBUG("Ignoring STUN attribute 0x%X after message integrity", (unsigned int)type);
		while (length & 0x03)
			++length; // attributes are aligned on 4 bytes
		return (int)(sizeof(struct stun_attr) + length);
	}

	switch (type) {
	case STUN_ATTR_MAPPED_ADDRESS: {
		JLOG_VERBOSE("Reading mapped address");
		uint8_t zero_mask[16] = {0};
		if (stun_read_value_mapped_address(attr->value, length, &msg->mapped, zero_mask) < 0)
			return -1;
		break;
	}
	case STUN_ATTR_XOR_MAPPED_ADDRESS: {
		JLOG_VERBOSE("Reading XOR mapped address");
		uint8_t mask[16];
		*((uint32_t *)mask) = htonl(STUN_MAGIC);
		memcpy(mask + 4, msg->transaction_id, 12);
		if (stun_read_value_mapped_address(attr->value, length, &msg->mapped, mask) < 0)
			return -1;
		break;
	}
	case STUN_ATTR_ALTERNATE_SERVER: {
		JLOG_VERBOSE("Reading alternate server");
		uint8_t zero_mask[16] = {0};
		if (stun_read_value_mapped_address(attr->value, length, &msg->alternate_server, zero_mask) <
		    0)
			return -1;
		break;
	}
	case STUN_ATTR_ERROR_CODE: {
		JLOG_VERBOSE("Reading error code");
		if (length < sizeof(struct stun_value_error_code)) {
			JLOG_DEBUG("STUN error code value too short, length=%zu", length);
			return -1;
		}
		const struct stun_value_error_code *error =
		    (const struct stun_value_error_code *)attr->value;
		msg->error_code = (error->code_class & 0x07) * 100 + error->code_number;

		if (msg->error_code == 401 || msg->error_code == 438) { // Unauthenticated or Stale Nonce
			JLOG_DEBUG("Got STUN error code %u", msg->error_code);

		} else if (JLOG_INFO_ENABLED) {
			size_t reason_length = length - sizeof(struct stun_value_error_code);
			if (reason_length >= STUN_MAX_ERROR_REASON_LEN)
				reason_length = STUN_MAX_ERROR_REASON_LEN - 1;

			char buffer[STUN_MAX_ERROR_REASON_LEN];
			memcpy(buffer, (const char *)error->reason, reason_length);
			buffer[reason_length] = '\0';

			JLOG_INFO("Got STUN error code %u, reason \"%s\"", msg->error_code, buffer);
		}
		break;
	}
	case STUN_ATTR_UNKNOWN_ATTRIBUTES: {
		JLOG_VERBOSE("Reading STUN unknown attributes");
		const uint16_t *attributes = (const uint16_t *)attr->value;
		for (int i = 0; i < (int)ntohs(attr->length) / 2; ++i) {
			stun_attr_type_t type = (stun_attr_type_t)ntohs(attributes[i]);
			JLOG_INFO("Got unknown attribute response for attribute 0x%X", (unsigned int)type);
		}
		break;
	}
	case STUN_ATTR_USERNAME: {
		JLOG_VERBOSE("Reading username");
		if (length + 1 > STUN_MAX_USERNAME_LEN) {
			JLOG_WARN("STUN username attribute value too long, length=%zu", length);
			return -1;
		}
		memcpy(msg->credentials.username, (const char *)attr->value, length);
		msg->credentials.username[length] = '\0';
		JLOG_VERBOSE("Got username: %s", msg->credentials.username);
		break;
	}
	case STUN_ATTR_MESSAGE_INTEGRITY: {
		JLOG_VERBOSE("Reading message integrity");
		if (length != HMAC_SHA1_SIZE) {
			JLOG_DEBUG("STUN message integrity length invalid, length=%zu", length);
			return -1;
		}
		msg->has_integrity = true;
		break;
	}
	case STUN_ATTR_MESSAGE_INTEGRITY_SHA256: {
		JLOG_VERBOSE("Reading message integrity SHA256");
		if (length != HMAC_SHA256_SIZE) {
			JLOG_DEBUG("STUN message integrity SHA256 length invalid, length=%zu", length);
			return -1;
		}
		msg->has_integrity = true;
		break;
	}
	case STUN_ATTR_FINGERPRINT: {
		JLOG_VERBOSE("Reading fingerprint");
		if (length != 4) {
			JLOG_DEBUG("STUN fingerprint length invalid, length=%zu", length);
			return -1;
		}
		size_t tmp_length = (uint8_t *)data - attr_begin + STUN_ATTR_SIZE + 4;
		size_t prev_length = stun_update_header_length(begin, tmp_length);
		uint32_t expected = CRC32(begin, (uint8_t *)data - begin) ^ STUN_FINGERPRINT_XOR;
		stun_update_header_length(begin, prev_length);

		uint32_t fingerprint = ntohl(*((uint32_t *)attr->value));
		if (fingerprint != expected) {
			JLOG_ERROR("STUN fingerprint check failed, expected=%lX, actual=%lX",
			           (unsigned long)expected, (unsigned long)fingerprint);
			return -1;
		}
		JLOG_VERBOSE("STUN fingerprint check succeeded");
		msg->has_fingerprint = true;
		break;
	}
	case STUN_ATTR_REALM: {
		JLOG_VERBOSE("Reading realm");
		if (length + 1 > STUN_MAX_REALM_LEN) {
			JLOG_WARN("STUN realm attribute value too long, length=%zu", length);
			return -1;
		}
		memcpy(msg->credentials.realm, (const char *)attr->value, length);
		msg->credentials.realm[length] = '\0';
		JLOG_VERBOSE("Got realm: %s", msg->credentials.realm);
		break;
	}
	case STUN_ATTR_NONCE: {
		JLOG_VERBOSE("Reading nonce");
		if (length + 1 > STUN_MAX_NONCE_LEN) {
			JLOG_WARN("STUN nonce attribute value too long, length=%zu", length);
			return -1;
		}
		memcpy(msg->credentials.nonce, (const char *)attr->value, length);
		msg->credentials.nonce[length] = '\0';
		JLOG_VERBOSE("Got nonce: %s", msg->credentials.nonce);

		// If the nonce of a response starts with the nonce cookie, decode the Security Feature bits
		// See https://www.rfc-editor.org/rfc/rfc8489.html#section-9.2
		if (STUN_IS_RESPONSE(msg->msg_class) &&
		    strlen(msg->credentials.nonce) > STUN_NONCE_COOKIE_LEN + 4 &&
		    strncmp(msg->credentials.nonce, STUN_NONCE_COOKIE, STUN_NONCE_COOKIE_LEN) == 0) {
			char encoded_security_bits[5];
			memcpy(encoded_security_bits, msg->credentials.nonce + STUN_NONCE_COOKIE_LEN, 4);
			encoded_security_bits[4] = '\0';

			uint8_t bytes[4];
			bytes[0] = 0;
			int len = BASE64_DECODE(encoded_security_bits, bytes + 1, 3);
			if (len == 3) {
				*security_bits = ntohl(*((uint32_t *)bytes));
				JLOG_VERBOSE("Nonce has cookie, Security Feature bits are 0x%lX",
				             (unsigned long)*security_bits);
			} else {
				JLOG_WARN("Nonce has cookie, but the encoded Security Feature bits field \"%s\" is "
				          "invalid",
				          encoded_security_bits);
				security_bits = 0;
			}
		} else if (msg->msg_class == STUN_CLASS_RESP_ERROR) {
			JLOG_DEBUG("Remote agent does not support RFC 8489");
		}
		break;
	}
	case STUN_ATTR_PASSWORD_ALGORITHM: {
		JLOG_VERBOSE("Reading password algorithm");
		if (length < sizeof(struct stun_value_password_algorithm)) {
			JLOG_WARN("STUN password algorithm value too short, length=%zu", length);
			return -1;
		}
		if (!STUN_IS_RESPONSE(msg->msg_class)) {
			const struct stun_value_password_algorithm *pwa =
			    (const struct stun_value_password_algorithm *)attr->value;
			stun_password_algorithm_t algorithm = ntohs(pwa->algorithm);
			if (algorithm == STUN_PASSWORD_ALGORITHM_MD5 ||
			    algorithm == STUN_PASSWORD_ALGORITHM_SHA256)
				msg->credentials.password_algorithm = algorithm;
			else
				JLOG_WARN("Unknown password algorithm 0x%hX", algorithm);
		} else {
			JLOG_WARN("Found password algorithm in response, ignoring");
		}
		break;
	}
	case STUN_ATTR_PASSWORD_ALGORITHMS: {
		JLOG_VERBOSE("Reading password algorithms list");
		if (length < sizeof(struct stun_value_password_algorithm)) {
			JLOG_WARN("STUN password algorithms list too short, length=%zu", length);
			return -1;
		}
		if (length > STUN_MAX_PASSWORD_ALGORITHMS_VALUE_SIZE) {
			JLOG_WARN("STUN password algorithms list too long, length=%zu", length);
			return -1;
		}

		memcpy(msg->credentials.password_algorithms_value, attr->value, length);
		msg->credentials.password_algorithms_value_size = length;

		if (!STUN_IS_RESPONSE(msg->msg_class)) {
			const uint8_t *pos = attr->value;
			const uint8_t *end = pos + length;
			while (pos < end) {
				if ((size_t)(end - pos) < sizeof(struct stun_value_password_algorithm)) {
					JLOG_WARN("STUN password algorithms list truncated, available=%zu", end - pos);
					return -1;
				}
				const struct stun_value_password_algorithm *pwa =
				    (const struct stun_value_password_algorithm *)pos;
				stun_password_algorithm_t algorithm = ntohs(pwa->algorithm);
				size_t parameters_length = ntohs(pwa->parameters_length);
				size_t padded_length = align32(parameters_length);

				pos += sizeof(struct stun_value_password_algorithm);

				if ((size_t)(end - pos) < padded_length) {
					JLOG_WARN(
					    "STUN password algorithm parameters too long, length=%zu, padded=%zu, "
					    "available=%zu",
					    parameters_length, padded_length, end - pos);
					return -1;
				}

				pos += padded_length;

				if (algorithm == STUN_PASSWORD_ALGORITHM_MD5 ||
				    algorithm == STUN_PASSWORD_ALGORITHM_SHA256) {
					msg->credentials.password_algorithm = algorithm;
					break;
				}

				JLOG_DEBUG("Unknown password algorithm 0x%hX", algorithm);
			}
		}
		break;
	}
	case STUN_ATTR_USERHASH: {
		JLOG_VERBOSE("Reading user hash");
		if (length != USERHASH_SIZE) {
			JLOG_WARN("STUN user hash value too long, length=%zu", length);
			return -1;
		}
		memcpy(msg->credentials.userhash, attr->value, USERHASH_SIZE);
		msg->credentials.enable_userhash = true;
		break;
	}
	case STUN_ATTR_SOFTWARE: {
		JLOG_VERBOSE("Reading software");
		if (length + 1 > STUN_MAX_SOFTWARE_LEN) {
			JLOG_WARN("STUN software attribute value too long, length=%zu", length);
			return -1;
		}
		char buffer[STUN_MAX_SOFTWARE_LEN];
		memcpy(buffer, (const char *)attr->value, length);
		buffer[length] = '\0';
		JLOG_VERBOSE("Remote agent is \"%s\"", buffer);
		break;
	}
	case STUN_ATTR_PRIORITY: {
		JLOG_VERBOSE("Reading priority");
		if (length != 4) {
			JLOG_DEBUG("STUN priority length invalid, length=%zu", length);
			return -1;
		}
		msg->priority = ntohl(*((uint32_t *)attr->value));
		JLOG_VERBOSE("Got priority: %lu", (unsigned long)msg->priority);
		break;
	}
	case STUN_ATTR_USE_CANDIDATE: {
		JLOG_VERBOSE("Found use candidate flag");
		msg->use_candidate = true;
		break;
	}
	case STUN_ATTR_ICE_CONTROLLING: {
		JLOG_VERBOSE("Found ICE controlling attribute");
		if (length != 8) {
			JLOG_DEBUG("STUN ICE controlling attribute length invalid, length=%zu", length);
			return -1;
		}
		msg->ice_controlling = ntohll(*((uint64_t *)attr->value));
		break;
	}
	case STUN_ATTR_ICE_CONTROLLED: {
		JLOG_VERBOSE("Found ICE controlled attribute");
		if (length != 8) {
			JLOG_DEBUG("STUN ICE controlled attribute length invalid, length=%zu", length);
			return -1;
		}
		msg->ice_controlled = ntohll(*((uint64_t *)attr->value));
		break;
	}
	case STUN_ATTR_CHANNEL_NUMBER: {
		JLOG_VERBOSE("Reading channel number attribute");
		if (length < sizeof(struct stun_value_channel_number)) {
			JLOG_DEBUG("STUN channel number attribute value too short, length=%zu", length);
			return -1;
		}
		const struct stun_value_channel_number *channel_number =
		    (const struct stun_value_channel_number *)attr->value;
		msg->channel_number = ntohs(channel_number->channel_number);
		break;
	}
	case STUN_ATTR_LIFETIME: {
		JLOG_VERBOSE("Reading lifetime attribute");
		if (length != 4) {
			JLOG_DEBUG("STUN lifetime attribute length invalid, length=%zu", length);
			return -1;
		}
		msg->lifetime = ntohl(*((uint32_t *)attr->value));
		msg->lifetime_set = true;
		break;
	}
	case STUN_ATTR_XOR_PEER_ADDRESS: {
		JLOG_VERBOSE("Reading XOR peer address");
		uint8_t mask[16];
		*((uint32_t *)mask) = htonl(STUN_MAGIC);
		memcpy(mask + 4, msg->transaction_id, 12);
		if (stun_read_value_mapped_address(attr->value, length, &msg->peer, mask) < 0)
			return -1;
		break;
	}
	case STUN_ATTR_XOR_RELAYED_ADDRESS: {
		JLOG_VERBOSE("Reading XOR relayed address");
		uint8_t mask[16];
		*((uint32_t *)mask) = htonl(STUN_MAGIC);
		memcpy(mask + 4, msg->transaction_id, 12);
		if (stun_read_value_mapped_address(attr->value, length, &msg->relayed, mask) < 0)
			return -1;
		break;
	}
	case STUN_ATTR_DATA: {
		JLOG_VERBOSE("Found data");
		msg->data = (const char *)attr->value;
		msg->data_size = length;
		break;
	}
	case STUN_ATTR_EVEN_PORT: {
		JLOG_VERBOSE("Found even port attribute");
		if (length < 1) {
			JLOG_DEBUG("STUN even port attribute length invalid, length=%zu", length);
			return -1;
		}
		msg->even_port = true;
		msg->next_port = ((struct stun_value_even_port *)attr->value)->r & 0x80;
		break;
	}
	case STUN_ATTR_REQUESTED_TRANSPORT: {
		JLOG_VERBOSE("Found requested transport attribute");
		if (length < sizeof(struct stun_value_requested_transport)) {
			JLOG_DEBUG("STUN requested transport attribute length invalid, length=%zu", length);
			return -1;
		}
		const struct stun_value_requested_transport *requested_transport =
		    (const struct stun_value_requested_transport *)attr->value;
		if (requested_transport->protocol != 17) { // UDP
			JLOG_WARN("Unexpected requested transport protocol: %d",
			          (int)requested_transport->protocol);
			return -1;
		}
		msg->requested_transport = true;
		break;
	}
	case STUN_ATTR_DONT_FRAGMENT: {
		JLOG_VERBOSE("Found don't fragment attribute");
		msg->dont_fragment = true;
		break;
	}
	case STUN_ATTR_RESERVATION_TOKEN: {
		JLOG_VERBOSE("Found reservation token");
		if (length != 8) {
			JLOG_DEBUG("STUN reservation token length invalid, length=%zu", length);
			return -1;
		}
		msg->reservation_token = ntohll(*((uint64_t *)attr->value));
		break;
	}
	default: {
		// Ignore
		if (STUN_IS_OPTIONAL_ATTR(type))
			JLOG_DEBUG("Ignoring unknown optional STUN attribute type 0x%X", (unsigned int)type);
		else
			JLOG_WARN("Unknown STUN attribute type 0x%X, ignoring", (unsigned int)type);
		break;
	}
	}
	return (int)(sizeof(struct stun_attr) + align32(length));
}

int stun_read_value_mapped_address(const void *data, size_t size, addr_record_t *mapped,
                                   const uint8_t *mask) {
	size_t len = sizeof(struct stun_value_mapped_address);
	if (size < len) {
		JLOG_VERBOSE("STUN mapped address value too short, size=%zu", size);
		return -1;
	}
	const struct stun_value_mapped_address *value = data;
	stun_address_family_t family = (stun_address_family_t)value->family;
	switch (family) {
	case STUN_ADDRESS_FAMILY_IPV4: {
		len += 4;
		if (size < len) {
			JLOG_DEBUG("IPv4 mapped address value too short, size=%zu", size);
			return -1;
		}
		JLOG_VERBOSE("Reading IPv4 address");
		mapped->len = sizeof(struct sockaddr_in);
		struct sockaddr_in *sin = (struct sockaddr_in *)&mapped->addr;
		sin->sin_family = AF_INET;
		sin->sin_port = value->port ^ *((uint16_t *)mask);
		uint8_t *bytes = (uint8_t *)&sin->sin_addr;
		for (int i = 0; i < 4; ++i)
			bytes[i] = value->address[i] ^ mask[i];
		break;
	}
	case STUN_ADDRESS_FAMILY_IPV6: {
		len += 16;
		if (size < len) {
			JLOG_DEBUG("IPv6 mapped address value too short, size=%zu", size);
			return -1;
		}
		JLOG_VERBOSE("Reading IPv6 address");
		mapped->len = sizeof(struct sockaddr_in6);
		struct sockaddr_in6 *sin6 = (struct sockaddr_in6 *)&mapped->addr;
		sin6->sin6_family = AF_INET6;
		sin6->sin6_port = value->port ^ *((uint16_t *)mask);
		uint8_t *bytes = (uint8_t *)&sin6->sin6_addr;
		for (int i = 0; i < 16; ++i)
			bytes[i] = value->address[i] ^ mask[i];
		break;
	}
	default: {
		JLOG_DEBUG("Unknown STUN address family 0x%X", (unsigned int)family);
		len = size;
		break;
	}
	}
	return (int)len;
}

bool stun_check_integrity(void *buf, size_t size, const stun_message_t *msg, const char *password) {
	if (!msg->has_integrity)
		return false;

	const struct stun_header *header = buf;
	const size_t length = ntohs(header->length);
	if (size < sizeof(struct stun_header) + length)
		return false;

	uint8_t key[MAX_HMAC_KEY_LEN];
	size_t key_len = generate_hmac_key(msg, password, key);

	bool success = false;
	uint8_t *begin = buf;
	const uint8_t *attr_begin = begin + sizeof(struct stun_header);
	const uint8_t *end = attr_begin + length;
	const uint8_t *pos = attr_begin;
	while (pos < end) {
		const struct stun_attr *attr = (const struct stun_attr *)pos;
		size_t attr_length = ntohs(attr->length);
		if (size < sizeof(struct stun_attr) + attr_length)
			return false;

		stun_attr_type_t type = (stun_attr_type_t)ntohs(attr->type);
		switch (type) {
		case STUN_ATTR_MESSAGE_INTEGRITY: {
			if (attr_length != HMAC_SHA1_SIZE)
				return false;

			size_t tmp_length = pos - attr_begin + STUN_ATTR_SIZE + HMAC_SHA1_SIZE;
			size_t prev_length = stun_update_header_length(begin, tmp_length);
			uint8_t hmac[HMAC_SHA1_SIZE];
			hmac_sha1(begin, pos - begin, key, key_len, hmac);
			stun_update_header_length(begin, prev_length);

			const uint8_t *expected_hmac = attr->value;
			if (const_time_memcmp(hmac, expected_hmac, HMAC_SHA1_SIZE) != 0) {
				JLOG_DEBUG("STUN message integrity SHA1 check failed");
				return false;
			}

			success = true;
			break;
		}
		case STUN_ATTR_MESSAGE_INTEGRITY_SHA256: {
			if (attr_length != HMAC_SHA256_SIZE)
				return false;

			size_t tmp_length = pos - attr_begin + STUN_ATTR_SIZE + HMAC_SHA256_SIZE;
			size_t prev_length = stun_update_header_length(begin, tmp_length);
			uint8_t hmac[HMAC_SHA256_SIZE];
			hmac_sha256(begin, pos - begin, key, key_len, hmac);
			stun_update_header_length(begin, prev_length);

			const uint8_t *expected_hmac = attr->value;
			if (const_time_memcmp(hmac, expected_hmac, HMAC_SHA256_SIZE) != 0) {
				JLOG_DEBUG("STUN message integrity SHA256 check failed");
				return false;
			}

			success = true;
			break;
		}
		default:
			// Ignore
			break;
		}

		pos += sizeof(struct stun_attr) + align32(attr_length);
	}

	if (!success)
		return false;

	JLOG_VERBOSE("STUN message integrity check succeeded");
	return true;
}

void stun_prepend_nonce_cookie(char *nonce) {
	// RFC 8489: To indicate that it supports this specification, a server MUST prepend the
	// NONCE attribute value with the character string composed of "obMatJos2" concatenated with
	// the (4-character) base64 [RFC4648] encoding of the 24-bit STUN Security Features See
	// https://www.rfc-editor.org/rfc/rfc8489.html#section-9.2
	char copy[STUN_MAX_NONCE_LEN];
	strcpy(copy, nonce);

	char encoded_security_bits[5];
	uint32_t security_bits =
	    htonl(STUN_SECURITY_PASSWORD_ALGORITHMS_BIT | STUN_SECURITY_USERNAME_ANONYMITY_BIT);
	BASE64_ENCODE((uint8_t *)&security_bits + 1, 3, encoded_security_bits, 5);

	snprintf(nonce, STUN_MAX_NONCE_LEN, "%s%s%.*s", STUN_NONCE_COOKIE, encoded_security_bits,
	         STUN_MAX_NONCE_LEN - (STUN_NONCE_COOKIE_LEN + 5), copy);
}

void stun_compute_userhash(const char *username, const char *realm, uint8_t *out) {
	char input[MAX_USERHASH_INPUT_LEN];
	int input_len = snprintf(input, MAX_USERHASH_INPUT_LEN, "%s:%s", username, realm);
	if (input_len < 0)
		return;

	if (input_len >= MAX_USERHASH_INPUT_LEN)
		input_len = MAX_USERHASH_INPUT_LEN - 1;

	hash_sha256(input, input_len, out);
}

void stun_process_credentials(const stun_credentials_t *credentials, stun_credentials_t *dst) {
	char username[STUN_MAX_USERNAME_LEN];
	strcpy(username, dst->username);
	*dst = *credentials;
	strcpy(dst->username, username);

	if (credentials->enable_userhash)
		stun_compute_userhash(username, credentials->realm, dst->userhash);
}

const char *stun_get_error_reason(unsigned int code) {
	switch (code) {
	case 0:
		return "";
	case 300:
		return "Try Alternate";
	case 400:
		return "Bad Request";
	case 401:
		return "Unauthenticated";
	case 403:
		return "Forbidden";
	case 420:
		return "Unknown Attribute";
	case 437:
		return "Allocation Mismatch";
	case 438:
		return "Stale Nonce";
	case 440:
		return "Address Family not Supported";
	case 441:
		return "Wrong credentials";
	case 442:
		return "Unsupported Transport Protocol";
	case 443:
		return "Peer Address Family Mismatch";
	case 486:
		return "Allocation Quota Reached";
	case 500:
		return "Server Error";
	case 508:
		return "Insufficient Capacity";
	default:
		return "Error";
	}
}

JUICE_EXPORT bool _juice_is_stun_datagram(const void *data, size_t size) {
	return is_stun_datagram(data, size);
}

JUICE_EXPORT int _juice_stun_read(void *data, size_t size, stun_message_t *msg) {
	return stun_read(data, size, msg);
}

JUICE_EXPORT bool _juice_stun_check_integrity(void *buf, size_t size, const stun_message_t *msg,
                                              const char *password) {
	return stun_check_integrity(buf, size, msg, password);
}
