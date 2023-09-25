/**
 * Copyright (c) 2020 Paul-Louis Ageneau
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

#ifndef JUICE_TIMESTAMP_H
#define JUICE_TIMESTAMP_H

#include <stdint.h>
#include <stdlib.h>

typedef int64_t timestamp_t;
typedef timestamp_t timediff_t;

timestamp_t current_timestamp();

#endif
