/**
 * Copyright (c) 2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef JUICE_CRC32_H
#define JUICE_CRC32_H

#include "juice.h"

#include <stdint.h>
#include <stdlib.h>

JUICE_EXPORT uint32_t juice_crc32(const void *data, size_t size);

#define CRC32(data, size) juice_crc32(data, size)

#endif // JUICE_CRC32_H
