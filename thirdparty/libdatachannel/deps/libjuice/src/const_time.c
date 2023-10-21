/**
 * Copyright (c) 2021 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#include "const_time.h"

int const_time_memcmp(const void *a, const void *b, size_t len) {
	const unsigned char *ca = a;
	const unsigned char *cb = b;
	unsigned char x = 0;
	for (size_t i = 0; i < len; i++)
		x |= ca[i] ^ cb[i];

	return x;
}

int const_time_strcmp(const void *a, const void *b) {
	const unsigned char *ca = a;
	const unsigned char *cb = b;
	unsigned char x = 0;
	size_t i = 0;
	for(;;) {
		x |= ca[i] ^ cb[i];
		if (!ca[i] || !cb[i])
			break;
		++i;
	}

	return x;
}
