/* infcodes.h -- header to use infcodes.c
 * Copyright (C) 1995-2002 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

/* WARNING: this file should *not* be used by applications. It is
   part of the implementation of the compression library and is
   subject to change. Applications should only use zlib.h.
 */

#ifndef _INFCODES_H
#define _INFCODES_H

struct inflate_codes_state;
typedef struct inflate_codes_state FAR inflate_codes_statef;

local inflate_codes_statef *inflate_codes_new OF((
    uInt, uInt,
    inflate_huft *, inflate_huft *,
    z_streamp ));

local int inflate_codes OF((
    inflate_blocks_statef *,
    z_streamp ,
    int));

local void inflate_codes_free OF((
    inflate_codes_statef *,
    z_streamp ));

#endif /* _INFCODES_H */
