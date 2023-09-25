/**
 * Copyright (c) 2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef JUICE_ADDR_H
#define JUICE_ADDR_H

#include "socket.h"

#include <stdbool.h>
#include <stdint.h>

// IPv6 max representation length is 45 plus 4 for potential zone index
#define ADDR_MAX_NUMERICHOST_LEN 56 // 45 + 4 + 1 rounded up
#define ADDR_MAX_NUMERICSERV_LEN 8 // 5 + 1 rounded up
#define ADDR_MAX_STRING_LEN 64

socklen_t addr_get_len(const struct sockaddr *sa);
uint16_t addr_get_port(const struct sockaddr *sa);
int addr_set_port(struct sockaddr *sa, uint16_t port);
bool addr_is_any(const struct sockaddr *sa);
bool addr_is_local(const struct sockaddr *sa);
bool addr_unmap_inet6_v4mapped(struct sockaddr *sa, socklen_t *len);
bool addr_map_inet6_v4mapped(struct sockaddr_storage *ss, socklen_t *len);
bool addr_is_equal(const struct sockaddr *a, const struct sockaddr *b, bool compare_ports);
int addr_to_string(const struct sockaddr *sa, char *buffer, size_t size);
unsigned long addr_hash(const struct sockaddr *sa, bool with_port);

typedef struct addr_record {
	struct sockaddr_storage addr;
	socklen_t len;
} addr_record_t;

int addr_resolve(const char *hostname, const char *service, addr_record_t *records, size_t count);
bool addr_is_numeric_hostname(const char *hostname);

bool addr_record_is_equal(const addr_record_t *a, const addr_record_t *b, bool compare_ports);
int addr_record_to_string(const addr_record_t *record, char *buffer, size_t size);
unsigned long addr_record_hash(const addr_record_t *record, bool with_port);

#endif // JUICE_ADDR_H
