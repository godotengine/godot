/**
 * Copyright (c) 2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef JUICE_RANDOM_H
#define JUICE_RANDOM_H

#include <stdint.h>
#include <stdlib.h>

void juice_random(void *buf, size_t size);
void juice_random_str64(char *buf, size_t size);

uint32_t juice_rand32(void);
uint64_t juice_rand64(void);

#endif // JUICE_RANDOM_H
