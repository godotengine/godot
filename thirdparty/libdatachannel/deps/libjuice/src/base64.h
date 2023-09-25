/**
 * Copyright (c) 2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef JUICE_BASE64_H
#define JUICE_BASE64_H

#include "juice.h"

#include <stdint.h>
#include <stdlib.h>

// RFC4648-compliant base64 encoder and decoder
JUICE_EXPORT int juice_base64_encode(const void *data, size_t size, char *out, size_t out_size);
JUICE_EXPORT int juice_base64_decode(const char *str, void *out, size_t out_size);

#define BASE64_ENCODE(data, size, out, out_size) juice_base64_encode(data, size, out, out_size)
#define BASE64_DECODE(str, out, out_size) juice_base64_decode(str, out, out_size)

#endif // JUICE_BASE64_H
