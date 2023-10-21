/**
 * Copyright (c) 2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef JUICE_STUN_H
#define JUICE_STUN_H

#include "juice.h"

#include "addr.h"
#include "hash.h"
#include "hmac.h"

#include <stdbool.h>
#include <stdint.h>

#pragma pack(push, 1)
/*
 * STUN message header (20 bytes)
 * See https://www.rfc-editor.org/rfc/rfc8489.html#section-5
 *
 *  0                   1                   2                   3
 *  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |0 0|     STUN Message Type     |         Message Length        |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                    Magic Cookie = 0x2112A442                  |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                                                               |
 * |                     Transaction ID (96 bits)                  |
 * |                                                               |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 */
#define STUN_TRANSACTION_ID_SIZE 12

struct stun_header {
	uint16_t type;
	uint16_t length;
	uint32_t magic;
	uint8_t transaction_id[STUN_TRANSACTION_ID_SIZE];
};

/*
 * Format of STUN Message Type Field
 *
 *  0                 1
 *  2  3  4 5 6 7 8 9 0 1 2 3 4 5
 * +--+--+-+-+-+-+-+-+-+-+-+-+-+-+
 * |M |M |M|M|M|C|M|M|M|C|M|M|M|M|
 * |11|10|9|8|7|1|6|5|4|0|3|2|1|0|
 * +--+--+-+-+-+-+-+-+-+-+-+-+-+-+
 * Request:    C=b00
 * Indication: C=b01
 * Response:   C=b10 (success)
 *             C=b11 (error)
 */
#define STUN_CLASS_MASK 0x0110

typedef enum stun_class {
	STUN_CLASS_REQUEST = 0x0000,
	STUN_CLASS_INDICATION = 0x0010,
	STUN_CLASS_RESP_SUCCESS = 0x0100,
	STUN_CLASS_RESP_ERROR = 0x0110
} stun_class_t;

typedef enum stun_method {
	STUN_METHOD_BINDING = 0x0001,

	// Methods for TURN
	// See https://www.rfc-editor.org/rfc/rfc8656.html#section-17
	STUN_METHOD_ALLOCATE = 0x003,
	STUN_METHOD_REFRESH = 0x004,
	STUN_METHOD_SEND = 0x006,
	STUN_METHOD_DATA = 0x007,
	STUN_METHOD_CREATE_PERMISSION = 0x008,
	STUN_METHOD_CHANNEL_BIND = 0x009
} stun_method_t;

#define STUN_IS_RESPONSE(msg_class) (msg_class & 0x0100)

/*
 * STUN attribute header
 *
 *  0                   1                   2                   3
 *  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |             Type              |            Length             |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                        Value (variable)                     ...
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 */
struct stun_attr {
	uint16_t type;
	uint16_t length;
	uint8_t value[];
};

typedef enum stun_attr_type {
	// Comprehension-required
	STUN_ATTR_MAPPED_ADDRESS = 0x0001,
	STUN_ATTR_USERNAME = 0x0006,
	STUN_ATTR_MESSAGE_INTEGRITY = 0x0008,
	STUN_ATTR_ERROR_CODE = 0x0009,
	STUN_ATTR_UNKNOWN_ATTRIBUTES = 0x000A,
	STUN_ATTR_REALM = 0x0014,
	STUN_ATTR_NONCE = 0x0015,
	STUN_ATTR_MESSAGE_INTEGRITY_SHA256 = 0x001C,
	STUN_ATTR_PASSWORD_ALGORITHM = 0x001D,
	STUN_ATTR_USERHASH = 0x001E,
	STUN_ATTR_XOR_MAPPED_ADDRESS = 0x0020,
	STUN_ATTR_PRIORITY = 0x0024,
	STUN_ATTR_USE_CANDIDATE = 0x0025,

	// Comprehension-optional
	STUN_ATTR_PASSWORD_ALGORITHMS = 0x8002,
	STUN_ATTR_ALTERNATE_DOMAIN = 0x8003,
	STUN_ATTR_SOFTWARE = 0x8022,
	STUN_ATTR_ALTERNATE_SERVER = 0x8023,
	STUN_ATTR_FINGERPRINT = 0x8028,
	STUN_ATTR_ICE_CONTROLLED = 0x8029,
	STUN_ATTR_ICE_CONTROLLING = 0x802A,

	// Attributes for TURN
	// See https://www.rfc-editor.org/rfc/rfc8656.html#section-18
	STUN_ATTR_CHANNEL_NUMBER = 0x000C,
	STUN_ATTR_LIFETIME = 0x000D,
	STUN_ATTR_XOR_PEER_ADDRESS = 0x0012,
	STUN_ATTR_DATA = 0x0013,
	STUN_ATTR_XOR_RELAYED_ADDRESS = 0x0016,
	STUN_ATTR_EVEN_PORT = 0x0018,
	STUN_ATTR_REQUESTED_TRANSPORT = 0x0019,
	STUN_ATTR_DONT_FRAGMENT = 0x001A,
	STUN_ATTR_RESERVATION_TOKEN = 0x0022
} stun_attr_type_t;

#define STUN_IS_OPTIONAL_ATTR(attr_type) (attr_type & 0x8000)

/*
 * STUN attribute value for MAPPED-ADDRESS or XOR-MAPPED-ADDRESS
 *
 *  0                   1                   2                   3
 *  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |X X X X X X X X|    Family     |        Port or X-Port         |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                                                               |
 * |           Address or X-Address (32 bits or 128 bits)          |
 * |                                                               |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 */
struct stun_value_mapped_address {
	uint8_t padding;
	uint8_t family;
	uint16_t port;
	uint8_t address[];
};

typedef enum stun_address_family {
	STUN_ADDRESS_FAMILY_IPV4 = 0x01,
	STUN_ADDRESS_FAMILY_IPV6 = 0x02,
} stun_address_family_t;

/*
 * STUN attribute value for ERROR-CODE
 *
 *  0                   1                   2                   3
 *  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |           Reserved, should be 0         |Class|     Number    |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |      Reason Phrase (variable)                               ...
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 */
struct stun_value_error_code {
	uint16_t reserved;
	uint8_t code_class; // lower 3 bits only, higher bits are reserved
	uint8_t code_number;
	uint8_t reason[];
};

#define STUN_ERROR_INTERNAL_VALIDATION_FAILED 599

/*
 * STUN attribute for CHANNEL-NUMBER
 *
 *  0                   1                   2                   3
 *  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |        Channel Number         |         RFFU = 0              |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 */
struct stun_value_channel_number {
	uint16_t channel_number;
	uint16_t reserved;
};

/*
 * STUN attribute for EVEN-PORT
 *
 *  0
 *  0 1 2 3 4 5 6 7
 * +-+-+-+-+-+-+-+-+
 * |R|    RFFU     |
 * +-+-+-+-+-+-+-+-+
 */
struct stun_value_even_port {
	uint8_t r;
};

/*
 * STUN attribute for REQUESTED-TRANSPORT
 *
 *  0                   1                   2                   3
 *  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |    Protocol   |                    RFFU                       |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 */
struct stun_value_requested_transport {
	uint8_t protocol;
	uint8_t reserved1;
	uint16_t reserved2;
};

/*
 * STUN attribute value for PASSWORD-ALGORITHM and PASSWORD-ALGORITHMS
 *
 *  0                   1                   2                   3
 *  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |         Algorithm 1           | Algorithm 1 Parameters Length |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                    Algorithm 1 Parameters (variable)
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |         Algorithm 2           | Algorithm 2 Parameters Length |
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                    Algorithm 2 Parameters (variable)
 * +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
 * |                                                             ...
 */
struct stun_value_password_algorithm {
	uint16_t algorithm;
	uint16_t parameters_length;
	uint8_t parameters[];
};

typedef enum stun_password_algorithm {
	STUN_PASSWORD_ALGORITHM_UNSET = 0x0000,
	STUN_PASSWORD_ALGORITHM_MD5 = 0x0001,
	STUN_PASSWORD_ALGORITHM_SHA256 = 0x0002,
} stun_password_algorithm_t;

#pragma pack(pop)

// The value of USERNAME is a variable-length value. It MUST contain a UTF-8 [RFC3629] encoded
// sequence of less than 513 bytes [...]
#define STUN_MAX_USERNAME_LEN 513 + 1

// The REALM attribute [...] MUST be a UTF-8 [RFC3629] encoded sequence of less than 128 characters
// (which can be as long as 763 bytes)
#define STUN_MAX_REALM_LEN 763 + 1

// The NONCE attribute may be present in requests and responses. It [...] MUST be less than 128
// characters (which can be as long as 763 bytes)
#define STUN_MAX_NONCE_LEN 763 + 1

// The value of SOFTWARE is variable length. It MUST be a UTF-8 [RFC3629] encoded sequence of less
// than 128 characters (which can be as long as 763 bytes)
#define STUN_MAX_SOFTWARE_LEN 763 + 1

// The reason phrase MUST be a UTF-8-encoded [RFC3629] sequence of fewer than 128 characters (which
// can be as long as 509 bytes when encoding them or 763 bytes when decoding them).
#define STUN_MAX_ERROR_REASON_LEN 763 + 1

#define STUN_MAX_PASSWORD_LEN STUN_MAX_USERNAME_LEN

// Nonce cookie prefix as specified in https://www.rfc-editor.org/rfc/rfc8489.html#section-9.2
#define STUN_NONCE_COOKIE "obMatJos2"
#define STUN_NONCE_COOKIE_LEN 9

// USERHASH is a SHA256 digest
#define USERHASH_SIZE HASH_SHA256_SIZE

// STUN Security Feature bits as defined in https://www.rfc-editor.org/rfc/rfc8489.html#section-18.1
// See errata about bit order: https://www.rfc-editor.org/errata_search.php?rfc=8489
// Bits are assigned starting from the least significant side of the bit set, so Bit 0 is the rightmost bit, and Bit 23 is the leftmost bit.
// Bit 0: Password algorithms
// Bit 1: Username anonymity
// Bit 2-23: Unassigned

#define STUN_SECURITY_PASSWORD_ALGORITHMS_BIT 0x01
#define STUN_SECURITY_USERNAME_ANONYMITY_BIT 0x02

#define STUN_MAX_PASSWORD_ALGORITHMS_VALUE_SIZE 256

typedef struct stun_credentials {
	char username[STUN_MAX_USERNAME_LEN];
	char realm[STUN_MAX_REALM_LEN];
	char nonce[STUN_MAX_NONCE_LEN];
	uint8_t userhash[USERHASH_SIZE];
	bool enable_userhash;
	stun_password_algorithm_t password_algorithm;
	uint8_t password_algorithms_value[STUN_MAX_PASSWORD_ALGORITHMS_VALUE_SIZE];
	size_t password_algorithms_value_size;
} stun_credentials_t;

typedef struct stun_message {
	stun_class_t msg_class;
	stun_method_t msg_method;
	uint8_t transaction_id[STUN_TRANSACTION_ID_SIZE];
	unsigned int error_code;
	uint32_t priority;
	uint64_t ice_controlling;
	uint64_t ice_controlled;
	bool use_candidate;
	addr_record_t mapped;

	stun_credentials_t credentials;

	// Only for reading
	bool has_integrity;
	bool has_fingerprint;

	// TURN
	addr_record_t peer;
	addr_record_t relayed;
	addr_record_t alternate_server;
	const char *data;
	size_t data_size;
	uint32_t lifetime;
	uint16_t channel_number;
	bool lifetime_set;
	bool even_port;
	bool next_port;
	bool dont_fragment;
	bool requested_transport;
	uint64_t reservation_token;

} stun_message_t;

int stun_write(void *buf, size_t size, const stun_message_t *msg,
               const char *password); // password may be NULL
int stun_write_header(void *buf, size_t size, stun_class_t class, stun_method_t method,
                      const uint8_t *transaction_id);
size_t stun_update_header_length(void *buf, size_t length);
int stun_write_attr(void *buf, size_t size, uint16_t type, const void *value, size_t length);
int stun_write_value_mapped_address(void *buf, size_t size, const struct sockaddr *addr,
                                    socklen_t addrlen, const uint8_t *mask);

bool is_stun_datagram(const void *data, size_t size);

int stun_read(void *data, size_t size, stun_message_t *msg);
int stun_read_attr(const void *data, size_t size, stun_message_t *msg, uint8_t *begin,
                   uint8_t *attr_begin, uint32_t *security_bits);
int stun_read_value_mapped_address(const void *data, size_t size, addr_record_t *mapped,
                                   const uint8_t *mask);

bool stun_check_integrity(void *buf, size_t size, const stun_message_t *msg, const char *password);

void stun_compute_userhash(const char *username, const char *realm, uint8_t *out);
void stun_prepend_nonce_cookie(char *nonce);
void stun_process_credentials(const stun_credentials_t *credentials, stun_credentials_t *dst);

const char *stun_get_error_reason(unsigned int code);

// Export for tests
JUICE_EXPORT bool _juice_is_stun_datagram(const void *data, size_t size);
JUICE_EXPORT int _juice_stun_read(void *data, size_t size, stun_message_t *msg);
JUICE_EXPORT bool _juice_stun_check_integrity(void *buf, size_t size, const stun_message_t *msg,
                                              const char *password);

#endif
