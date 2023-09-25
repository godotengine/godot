/**
 * Copyright (c) 2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef JUICE_HMAC_H
#define JUICE_HMAC_H

#include <stdint.h>
#include <stdlib.h>

#define HMAC_SHA1_SIZE 20
#define HMAC_SHA256_SIZE 32

void hmac_sha1(const void *message, size_t size, const void *key, size_t key_size, void *digest);
void hmac_sha256(const void *message, size_t size, const void *key, size_t key_size, void *digest);

#endif
