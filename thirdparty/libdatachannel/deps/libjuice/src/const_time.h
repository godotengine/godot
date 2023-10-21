/**
 * Copyright (c) 2021 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef JUICE_CONST_TIME_H
#define JUICE_CONST_TIME_H

#include <stdint.h>
#include <stdlib.h>

int const_time_memcmp(const void *a, const void *b, size_t len);
int const_time_strcmp(const void *a, const void *b);

#endif
