#ifndef foodirectionhfoo
#define foodirectionhfoo

/***
  This file is part of PulseAudio.

  Copyright 2014 Intel Corporation

  PulseAudio is free software; you can redistribute it and/or modify
  it under the terms of the GNU Lesser General Public License as published
  by the Free Software Foundation; either version 2.1 of the License,
  or (at your option) any later version.

  PulseAudio is distributed in the hope that it will be useful, but
  WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
  General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with PulseAudio; if not, see <http://www.gnu.org/licenses/>.
***/

#include <pulse/def.h>

/** \file
 * Utility functions for \ref pa_direction_t. */

/** Return non-zero if the given value is a valid direction (either input,
 * output or bidirectional). \since 6.0 */
int pa_direction_valid(pa_direction_t direction) PA_GCC_CONST;

/** Return a textual representation of the direction. \since 6.0 */
const char *pa_direction_to_string(pa_direction_t direction);

#endif
