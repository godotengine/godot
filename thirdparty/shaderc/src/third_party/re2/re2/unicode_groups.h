// Copyright 2008 The RE2 Authors.  All Rights Reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef RE2_UNICODE_GROUPS_H_
#define RE2_UNICODE_GROUPS_H_

// Unicode character groups.

// The codes get split into ranges of 16-bit codes
// and ranges of 32-bit codes.  It would be simpler
// to use only 32-bit ranges, but these tables are large
// enough to warrant extra care.
//
// Using just 32-bit ranges gives 27 kB of data.
// Adding 16-bit ranges gives 18 kB of data.
// Adding an extra table of 16-bit singletons would reduce
// to 16.5 kB of data but make the data harder to use;
// we don't bother.

#include <stdint.h>

#include "util/util.h"
#include "util/utf.h"

namespace re2 {

struct URange16
{
  uint16_t lo;
  uint16_t hi;
};

struct URange32
{
  Rune lo;
  Rune hi;
};

struct UGroup
{
  const char *name;
  int sign;  // +1 for [abc], -1 for [^abc]
  const URange16 *r16;
  int nr16;
  const URange32 *r32;
  int nr32;
};

// Named by property or script name (e.g., "Nd", "N", "Han").
// Negated groups are not included.
extern const UGroup unicode_groups[];
extern const int num_unicode_groups;

// Named by POSIX name (e.g., "[:alpha:]", "[:^lower:]").
// Negated groups are included.
extern const UGroup posix_groups[];
extern const int num_posix_groups;

// Named by Perl name (e.g., "\\d", "\\D").
// Negated groups are included.
extern const UGroup perl_groups[];
extern const int num_perl_groups;

}  // namespace re2

#endif  // RE2_UNICODE_GROUPS_H_
