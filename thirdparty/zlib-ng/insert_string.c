/* insert_string.c -- insert_string integer hash variant
 *
 * Copyright (C) 1995-2024 Jean-loup Gailly and Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 *
 */

#include "zbuild.h"
#include "deflate.h"

#define HASH_SLIDE           16

#define HASH_CALC(h, val)    h = ((val * 2654435761U) >> HASH_SLIDE);
#define HASH_CALC_VAR        h
#define HASH_CALC_VAR_INIT   uint32_t h

#define UPDATE_HASH          update_hash
#define INSERT_STRING        insert_string
#define QUICK_INSERT_STRING  quick_insert_string

#include "insert_string_tpl.h"
