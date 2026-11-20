/* insert_string_roll.c -- insert_string rolling hash variant
 *
 * Copyright (C) 1995-2024 Jean-loup Gailly and Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 *
 */

#include "zbuild.h"
#include "deflate.h"

#define HASH_SLIDE           5

#define HASH_CALC(h, val)    h = ((h << HASH_SLIDE) ^ ((uint8_t)val))
#define HASH_CALC_VAR        s->ins_h
#define HASH_CALC_VAR_INIT
#define HASH_CALC_READ       val = strstart[0]
#define HASH_CALC_MASK       (32768u - 1u)
#define HASH_CALC_OFFSET     (STD_MIN_MATCH-1)

#define UPDATE_HASH          update_hash_roll
#define INSERT_STRING        insert_string_roll
#define QUICK_INSERT_STRING  quick_insert_string_roll

#include "insert_string_tpl.h"
