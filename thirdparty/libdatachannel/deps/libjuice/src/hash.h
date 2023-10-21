/**
 * Copyright (c) 2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef JUICE_HASH_H
#define JUICE_HASH_H

#include <stdint.h>
#include <stdlib.h>

#define HASH_MD5_SIZE 16
#define HASH_SHA1_SIZE 24
#define HASH_SHA256_SIZE 32

void hash_md5(const void *message, size_t size, void *digest);
void hash_sha1(const void *message, size_t size, void *digest);
void hash_sha256(const void *message, size_t size, void *digest);

#endif
