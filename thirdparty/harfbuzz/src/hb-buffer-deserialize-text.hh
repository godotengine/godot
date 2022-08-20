
#line 1 "hb-buffer-deserialize-text.rl"
/*
 * Copyright © 2013  Google, Inc.
 *
 *  This is part of HarfBuzz, a text shaping library.
 *
 * Permission is hereby granted, without written agreement and without
 * license or royalty fees, to use, copy, modify, and distribute this
 * software and its documentation for any purpose, provided that the
 * above copyright notice and the following two paragraphs appear in
 * all copies of this software.
 *
 * IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE TO ANY PARTY FOR
 * DIRECT, INDIRECT, SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES
 * ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN
 * IF THE COPYRIGHT HOLDER HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH
 * DAMAGE.
 *
 * THE COPYRIGHT HOLDER SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE.  THE SOFTWARE PROVIDED HEREUNDER IS
 * ON AN "AS IS" BASIS, AND THE COPYRIGHT HOLDER HAS NO OBLIGATION TO
 * PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR MODIFICATIONS.
 *
 * Google Author(s): Behdad Esfahbod
 */

#ifndef HB_BUFFER_DESERIALIZE_TEXT_HH
#define HB_BUFFER_DESERIALIZE_TEXT_HH

#include "hb.hh"


#line 33 "hb-buffer-deserialize-text.hh"
static const unsigned char _deserialize_text_trans_keys[] = {
	0u, 0u, 9u, 91u, 85u, 85u, 43u, 43u, 48u, 102u, 9u, 85u, 48u, 57u, 45u, 57u, 
	48u, 57u, 48u, 57u, 48u, 57u, 45u, 57u, 48u, 57u, 44u, 44u, 45u, 57u, 48u, 57u, 
	44u, 57u, 43u, 124u, 45u, 57u, 48u, 57u, 9u, 124u, 9u, 124u, 0u, 0u, 9u, 85u, 
	9u, 124u, 9u, 124u, 9u, 124u, 9u, 124u, 9u, 124u, 9u, 124u, 9u, 124u, 9u, 124u, 
	9u, 124u, 9u, 124u, 9u, 124u, 9u, 124u, 9u, 124u, 9u, 124u, 9u, 124u, 9u, 124u, 
	9u, 124u, 9u, 124u, 9u, 124u, 0
};

static const char _deserialize_text_key_spans[] = {
	0, 83, 1, 1, 55, 77, 10, 13, 
	10, 10, 10, 13, 10, 1, 13, 10, 
	14, 82, 13, 10, 116, 116, 0, 77, 
	116, 116, 116, 116, 116, 116, 116, 116, 
	116, 116, 116, 116, 116, 116, 116, 116, 
	116, 116, 116
};

static const short _deserialize_text_index_offsets[] = {
	0, 0, 84, 86, 88, 144, 222, 233, 
	247, 258, 269, 280, 294, 305, 307, 321, 
	332, 347, 430, 444, 455, 572, 689, 690, 
	768, 885, 1002, 1119, 1236, 1353, 1470, 1587, 
	1704, 1821, 1938, 2055, 2172, 2289, 2406, 2523, 
	2640, 2757, 2874
};

static const char _deserialize_text_indicies[] = {
	0, 0, 0, 0, 0, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	0, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 2, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 3, 1, 4, 1, 5, 
	1, 6, 6, 6, 6, 6, 6, 6, 
	6, 6, 6, 1, 1, 1, 1, 1, 
	1, 1, 6, 6, 6, 6, 6, 6, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 6, 6, 6, 6, 6, 6, 
	1, 7, 7, 7, 7, 7, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	7, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 4, 1, 8, 
	9, 9, 9, 9, 9, 9, 9, 9, 
	9, 1, 10, 1, 1, 11, 12, 12, 
	12, 12, 12, 12, 12, 12, 12, 1, 
	13, 14, 14, 14, 14, 14, 14, 14, 
	14, 14, 1, 15, 16, 16, 16, 16, 
	16, 16, 16, 16, 16, 1, 17, 18, 
	18, 18, 18, 18, 18, 18, 18, 18, 
	1, 19, 1, 1, 20, 21, 21, 21, 
	21, 21, 21, 21, 21, 21, 1, 22, 
	23, 23, 23, 23, 23, 23, 23, 23, 
	23, 1, 24, 1, 25, 1, 1, 26, 
	27, 27, 27, 27, 27, 27, 27, 27, 
	27, 1, 28, 29, 29, 29, 29, 29, 
	29, 29, 29, 29, 1, 24, 1, 1, 
	1, 23, 23, 23, 23, 23, 23, 23, 
	23, 23, 23, 1, 30, 30, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 30, 1, 
	1, 30, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 30, 30, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 30, 1, 31, 
	1, 1, 32, 33, 33, 33, 33, 33, 
	33, 33, 33, 33, 1, 34, 35, 35, 
	35, 35, 35, 35, 35, 35, 35, 1, 
	36, 36, 36, 36, 36, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 36, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 37, 
	37, 37, 37, 37, 37, 37, 37, 37, 
	37, 1, 1, 1, 38, 39, 1, 1, 
	37, 37, 37, 37, 37, 37, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	37, 37, 37, 37, 37, 37, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 40, 1, 41, 41, 41, 
	41, 41, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 41, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 42, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	43, 1, 1, 7, 7, 7, 7, 7, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 7, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 4, 
	1, 44, 44, 44, 44, 44, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	44, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 45, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 46, 1, 44, 44, 
	44, 44, 44, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 44, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 47, 47, 47, 
	47, 47, 47, 47, 47, 47, 47, 1, 
	1, 1, 1, 45, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 46, 1, 49, 49, 49, 49, 49, 
	48, 48, 48, 48, 48, 48, 48, 48, 
	48, 48, 48, 48, 48, 48, 48, 48, 
	48, 48, 49, 48, 48, 50, 48, 48, 
	48, 48, 48, 48, 48, 51, 1, 48, 
	48, 48, 48, 48, 48, 48, 48, 48, 
	48, 48, 48, 48, 48, 48, 48, 52, 
	48, 48, 53, 48, 48, 48, 48, 48, 
	48, 48, 48, 48, 48, 48, 48, 48, 
	48, 48, 48, 48, 48, 48, 48, 48, 
	48, 48, 48, 48, 48, 48, 54, 55, 
	48, 48, 48, 48, 48, 48, 48, 48, 
	48, 48, 48, 48, 48, 48, 48, 48, 
	48, 48, 48, 48, 48, 48, 48, 48, 
	48, 48, 48, 48, 48, 48, 56, 48, 
	57, 57, 57, 57, 57, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 57, 
	30, 30, 58, 30, 30, 30, 30, 30, 
	30, 30, 59, 1, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 60, 30, 30, 61, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 62, 63, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 64, 30, 57, 57, 57, 
	57, 57, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 57, 30, 30, 58, 
	30, 30, 30, 30, 30, 30, 30, 59, 
	1, 30, 30, 30, 65, 66, 66, 66, 
	66, 66, 66, 66, 66, 66, 30, 30, 
	30, 60, 30, 30, 61, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	62, 63, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	64, 30, 67, 67, 67, 67, 67, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 67, 1, 1, 68, 1, 1, 1, 
	1, 1, 1, 1, 1, 69, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 70, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 71, 1, 72, 
	72, 72, 72, 72, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 72, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 42, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 73, 1, 74, 74, 74, 74, 
	74, 48, 48, 48, 48, 48, 48, 48, 
	48, 48, 48, 48, 48, 48, 48, 48, 
	48, 48, 48, 74, 48, 48, 50, 48, 
	48, 48, 48, 48, 48, 48, 51, 1, 
	48, 48, 48, 48, 48, 48, 48, 48, 
	48, 48, 48, 48, 48, 48, 48, 48, 
	52, 48, 48, 53, 48, 48, 48, 48, 
	48, 48, 48, 48, 48, 48, 48, 48, 
	48, 48, 48, 48, 48, 48, 48, 48, 
	48, 48, 48, 48, 48, 48, 48, 54, 
	55, 48, 48, 48, 48, 48, 48, 48, 
	48, 48, 48, 48, 48, 48, 48, 48, 
	48, 48, 48, 48, 48, 48, 48, 48, 
	48, 48, 48, 48, 48, 48, 48, 56, 
	48, 75, 75, 75, 75, 75, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	75, 1, 1, 76, 1, 1, 1, 1, 
	1, 1, 1, 77, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	78, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 45, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 79, 1, 80, 80, 
	80, 80, 80, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 80, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 81, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 82, 1, 80, 80, 80, 80, 80, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 80, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 83, 83, 83, 83, 83, 83, 
	83, 83, 83, 83, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 81, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 82, 1, 
	84, 84, 84, 84, 84, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 84, 
	1, 1, 85, 1, 1, 1, 1, 1, 
	1, 1, 86, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 87, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 88, 1, 84, 84, 84, 
	84, 84, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 84, 1, 1, 85, 
	1, 1, 1, 1, 1, 1, 1, 86, 
	1, 1, 1, 1, 29, 29, 29, 29, 
	29, 29, 29, 29, 29, 29, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 87, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	88, 1, 75, 75, 75, 75, 75, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 75, 1, 1, 76, 1, 1, 1, 
	1, 1, 1, 1, 77, 1, 1, 1, 
	1, 89, 89, 89, 89, 89, 89, 89, 
	89, 89, 89, 1, 1, 1, 1, 1, 
	1, 78, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 45, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 79, 1, 90, 
	90, 90, 90, 90, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 90, 1, 
	1, 91, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 92, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 93, 1, 90, 90, 90, 90, 
	90, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 90, 1, 1, 91, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 35, 35, 35, 35, 35, 
	35, 35, 35, 35, 35, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	92, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 93, 
	1, 67, 67, 67, 67, 67, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	67, 1, 1, 68, 1, 1, 1, 1, 
	1, 1, 1, 1, 69, 1, 1, 1, 
	14, 14, 14, 14, 14, 14, 14, 14, 
	14, 14, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 70, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 71, 1, 94, 94, 
	94, 94, 94, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 94, 30, 30, 
	58, 30, 30, 30, 30, 30, 30, 30, 
	59, 1, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 60, 30, 30, 61, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 62, 95, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 96, 30, 94, 94, 94, 94, 94, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 94, 30, 30, 58, 30, 30, 
	30, 30, 30, 30, 30, 59, 1, 30, 
	30, 30, 97, 97, 97, 97, 97, 97, 
	97, 97, 97, 97, 30, 30, 30, 60, 
	30, 30, 61, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 62, 95, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 96, 30, 
	0
};

static const char _deserialize_text_trans_targs[] = {
	1, 0, 2, 26, 3, 4, 20, 5, 
	24, 25, 8, 29, 40, 29, 40, 32, 
	37, 33, 34, 12, 13, 16, 13, 16, 
	14, 15, 35, 36, 35, 36, 27, 19, 
	38, 39, 38, 39, 21, 20, 6, 22, 
	23, 21, 22, 23, 21, 22, 23, 25, 
	27, 27, 28, 7, 9, 11, 17, 22, 
	31, 27, 28, 7, 9, 11, 17, 22, 
	31, 41, 42, 30, 10, 18, 22, 31, 
	30, 31, 31, 30, 10, 7, 11, 31, 
	30, 22, 31, 34, 30, 10, 7, 22, 
	31, 37, 30, 10, 22, 31, 27, 22, 
	31, 42
};

static const char _deserialize_text_trans_actions[] = {
	0, 0, 0, 0, 1, 0, 2, 0, 
	2, 2, 3, 4, 4, 5, 5, 4, 
	4, 4, 4, 3, 3, 3, 0, 0, 
	6, 3, 4, 4, 5, 5, 5, 3, 
	4, 4, 5, 5, 7, 8, 9, 7, 
	7, 0, 0, 0, 10, 10, 10, 8, 
	12, 13, 14, 15, 15, 15, 16, 11, 
	11, 18, 19, 20, 20, 20, 0, 17, 
	17, 4, 4, 21, 22, 22, 21, 21, 
	0, 0, 13, 10, 23, 23, 23, 10, 
	24, 24, 24, 5, 25, 26, 26, 25, 
	25, 5, 27, 28, 27, 27, 30, 29, 
	29, 5
};

static const char _deserialize_text_eof_actions[] = {
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 7, 0, 0, 0, 
	10, 10, 11, 17, 17, 21, 0, 11, 
	10, 24, 24, 25, 25, 10, 27, 27, 
	21, 29, 29
};

static const int deserialize_text_start = 1;
static const int deserialize_text_first_final = 20;
static const int deserialize_text_error = 0;

static const int deserialize_text_en_main = 1;


#line 117 "hb-buffer-deserialize-text.rl"


static hb_bool_t
_hb_buffer_deserialize_text (hb_buffer_t *buffer,
				    const char *buf,
				    unsigned int buf_len,
				    const char **end_ptr,
				    hb_font_t *font)
{
  const char *p = buf, *pe = buf + buf_len;

  /* Ensure we have positions. */
  (void) hb_buffer_get_glyph_positions (buffer, nullptr);

  while (p < pe && ISSPACE (*p))
    p++;

  const char *eof = pe, *tok = nullptr;
  int cs;
  hb_glyph_info_t info = {0};
  hb_glyph_position_t pos = {0};
  
#line 506 "hb-buffer-deserialize-text.hh"
	{
	cs = deserialize_text_start;
	}

#line 509 "hb-buffer-deserialize-text.hh"
	{
	int _slen;
	int _trans;
	const unsigned char *_keys;
	const char *_inds;
	if ( p == pe )
		goto _test_eof;
	if ( cs == 0 )
		goto _out;
_resume:
	_keys = _deserialize_text_trans_keys + (cs<<1);
	_inds = _deserialize_text_indicies + _deserialize_text_index_offsets[cs];

	_slen = _deserialize_text_key_spans[cs];
	_trans = _inds[ _slen > 0 && _keys[0] <=(*p) &&
		(*p) <= _keys[1] ?
		(*p) - _keys[0] : _slen ];

	cs = _deserialize_text_trans_targs[_trans];

	if ( _deserialize_text_trans_actions[_trans] == 0 )
		goto _again;

	switch ( _deserialize_text_trans_actions[_trans] ) {
	case 1:
#line 38 "hb-buffer-deserialize-text.rl"
	{
	memset (&info, 0, sizeof (info));
	memset (&pos , 0, sizeof (pos ));
}
	break;
	case 3:
#line 51 "hb-buffer-deserialize-text.rl"
	{
	tok = p;
}
	break;
	case 5:
#line 55 "hb-buffer-deserialize-text.rl"
	{ if (unlikely (!buffer->ensure_glyphs ())) return false; }
	break;
	case 8:
#line 56 "hb-buffer-deserialize-text.rl"
	{ if (unlikely (!buffer->ensure_unicode ())) return false; }
	break;
	case 20:
#line 58 "hb-buffer-deserialize-text.rl"
	{
	/* TODO Unescape delimiters. */
	if (!hb_font_glyph_from_string (font,
					tok, p - tok,
					&info.codepoint))
	  return false;
}
	break;
	case 9:
#line 66 "hb-buffer-deserialize-text.rl"
	{if (!parse_hex (tok, p, &info.codepoint )) return false; }
	break;
	case 23:
#line 68 "hb-buffer-deserialize-text.rl"
	{ if (!parse_uint (tok, p, &info.cluster )) return false; }
	break;
	case 6:
#line 69 "hb-buffer-deserialize-text.rl"
	{ if (!parse_int  (tok, p, &pos.x_offset )) return false; }
	break;
	case 26:
#line 70 "hb-buffer-deserialize-text.rl"
	{ if (!parse_int  (tok, p, &pos.y_offset )) return false; }
	break;
	case 22:
#line 71 "hb-buffer-deserialize-text.rl"
	{ if (!parse_int  (tok, p, &pos.x_advance)) return false; }
	break;
	case 28:
#line 72 "hb-buffer-deserialize-text.rl"
	{ if (!parse_int  (tok, p, &pos.y_advance)) return false; }
	break;
	case 16:
#line 38 "hb-buffer-deserialize-text.rl"
	{
	memset (&info, 0, sizeof (info));
	memset (&pos , 0, sizeof (pos ));
}
#line 51 "hb-buffer-deserialize-text.rl"
	{
	tok = p;
}
	break;
	case 4:
#line 51 "hb-buffer-deserialize-text.rl"
	{
	tok = p;
}
#line 55 "hb-buffer-deserialize-text.rl"
	{ if (unlikely (!buffer->ensure_glyphs ())) return false; }
	break;
	case 2:
#line 51 "hb-buffer-deserialize-text.rl"
	{
	tok = p;
}
#line 56 "hb-buffer-deserialize-text.rl"
	{ if (unlikely (!buffer->ensure_unicode ())) return false; }
	break;
	case 17:
#line 58 "hb-buffer-deserialize-text.rl"
	{
	/* TODO Unescape delimiters. */
	if (!hb_font_glyph_from_string (font,
					tok, p - tok,
					&info.codepoint))
	  return false;
}
#line 43 "hb-buffer-deserialize-text.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 19:
#line 58 "hb-buffer-deserialize-text.rl"
	{
	/* TODO Unescape delimiters. */
	if (!hb_font_glyph_from_string (font,
					tok, p - tok,
					&info.codepoint))
	  return false;
}
#line 55 "hb-buffer-deserialize-text.rl"
	{ if (unlikely (!buffer->ensure_glyphs ())) return false; }
	break;
	case 7:
#line 66 "hb-buffer-deserialize-text.rl"
	{if (!parse_hex (tok, p, &info.codepoint )) return false; }
#line 43 "hb-buffer-deserialize-text.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 10:
#line 68 "hb-buffer-deserialize-text.rl"
	{ if (!parse_uint (tok, p, &info.cluster )) return false; }
#line 43 "hb-buffer-deserialize-text.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 25:
#line 70 "hb-buffer-deserialize-text.rl"
	{ if (!parse_int  (tok, p, &pos.y_offset )) return false; }
#line 43 "hb-buffer-deserialize-text.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 21:
#line 71 "hb-buffer-deserialize-text.rl"
	{ if (!parse_int  (tok, p, &pos.x_advance)) return false; }
#line 43 "hb-buffer-deserialize-text.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 27:
#line 72 "hb-buffer-deserialize-text.rl"
	{ if (!parse_int  (tok, p, &pos.y_advance)) return false; }
#line 43 "hb-buffer-deserialize-text.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 24:
#line 73 "hb-buffer-deserialize-text.rl"
	{ if (!parse_uint (tok, p, &info.mask    )) return false; }
#line 43 "hb-buffer-deserialize-text.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 12:
#line 38 "hb-buffer-deserialize-text.rl"
	{
	memset (&info, 0, sizeof (info));
	memset (&pos , 0, sizeof (pos ));
}
#line 51 "hb-buffer-deserialize-text.rl"
	{
	tok = p;
}
#line 55 "hb-buffer-deserialize-text.rl"
	{ if (unlikely (!buffer->ensure_glyphs ())) return false; }
	break;
	case 15:
#line 38 "hb-buffer-deserialize-text.rl"
	{
	memset (&info, 0, sizeof (info));
	memset (&pos , 0, sizeof (pos ));
}
#line 51 "hb-buffer-deserialize-text.rl"
	{
	tok = p;
}
#line 58 "hb-buffer-deserialize-text.rl"
	{
	/* TODO Unescape delimiters. */
	if (!hb_font_glyph_from_string (font,
					tok, p - tok,
					&info.codepoint))
	  return false;
}
	break;
	case 18:
#line 58 "hb-buffer-deserialize-text.rl"
	{
	/* TODO Unescape delimiters. */
	if (!hb_font_glyph_from_string (font,
					tok, p - tok,
					&info.codepoint))
	  return false;
}
#line 55 "hb-buffer-deserialize-text.rl"
	{ if (unlikely (!buffer->ensure_glyphs ())) return false; }
#line 43 "hb-buffer-deserialize-text.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 29:
#line 58 "hb-buffer-deserialize-text.rl"
	{
	/* TODO Unescape delimiters. */
	if (!hb_font_glyph_from_string (font,
					tok, p - tok,
					&info.codepoint))
	  return false;
}
#line 73 "hb-buffer-deserialize-text.rl"
	{ if (!parse_uint (tok, p, &info.mask    )) return false; }
#line 43 "hb-buffer-deserialize-text.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 11:
#line 38 "hb-buffer-deserialize-text.rl"
	{
	memset (&info, 0, sizeof (info));
	memset (&pos , 0, sizeof (pos ));
}
#line 51 "hb-buffer-deserialize-text.rl"
	{
	tok = p;
}
#line 58 "hb-buffer-deserialize-text.rl"
	{
	/* TODO Unescape delimiters. */
	if (!hb_font_glyph_from_string (font,
					tok, p - tok,
					&info.codepoint))
	  return false;
}
#line 43 "hb-buffer-deserialize-text.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 14:
#line 38 "hb-buffer-deserialize-text.rl"
	{
	memset (&info, 0, sizeof (info));
	memset (&pos , 0, sizeof (pos ));
}
#line 51 "hb-buffer-deserialize-text.rl"
	{
	tok = p;
}
#line 58 "hb-buffer-deserialize-text.rl"
	{
	/* TODO Unescape delimiters. */
	if (!hb_font_glyph_from_string (font,
					tok, p - tok,
					&info.codepoint))
	  return false;
}
#line 55 "hb-buffer-deserialize-text.rl"
	{ if (unlikely (!buffer->ensure_glyphs ())) return false; }
	break;
	case 30:
#line 58 "hb-buffer-deserialize-text.rl"
	{
	/* TODO Unescape delimiters. */
	if (!hb_font_glyph_from_string (font,
					tok, p - tok,
					&info.codepoint))
	  return false;
}
#line 73 "hb-buffer-deserialize-text.rl"
	{ if (!parse_uint (tok, p, &info.mask    )) return false; }
#line 55 "hb-buffer-deserialize-text.rl"
	{ if (unlikely (!buffer->ensure_glyphs ())) return false; }
#line 43 "hb-buffer-deserialize-text.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 13:
#line 38 "hb-buffer-deserialize-text.rl"
	{
	memset (&info, 0, sizeof (info));
	memset (&pos , 0, sizeof (pos ));
}
#line 51 "hb-buffer-deserialize-text.rl"
	{
	tok = p;
}
#line 58 "hb-buffer-deserialize-text.rl"
	{
	/* TODO Unescape delimiters. */
	if (!hb_font_glyph_from_string (font,
					tok, p - tok,
					&info.codepoint))
	  return false;
}
#line 55 "hb-buffer-deserialize-text.rl"
	{ if (unlikely (!buffer->ensure_glyphs ())) return false; }
#line 43 "hb-buffer-deserialize-text.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
#line 826 "hb-buffer-deserialize-text.hh"
	}

_again:
	if ( cs == 0 )
		goto _out;
	if ( ++p != pe )
		goto _resume;
	_test_eof: {}
	if ( p == eof )
	{
	switch ( _deserialize_text_eof_actions[cs] ) {
	case 17:
#line 58 "hb-buffer-deserialize-text.rl"
	{
	/* TODO Unescape delimiters. */
	if (!hb_font_glyph_from_string (font,
					tok, p - tok,
					&info.codepoint))
	  return false;
}
#line 43 "hb-buffer-deserialize-text.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 7:
#line 66 "hb-buffer-deserialize-text.rl"
	{if (!parse_hex (tok, p, &info.codepoint )) return false; }
#line 43 "hb-buffer-deserialize-text.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 10:
#line 68 "hb-buffer-deserialize-text.rl"
	{ if (!parse_uint (tok, p, &info.cluster )) return false; }
#line 43 "hb-buffer-deserialize-text.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 25:
#line 70 "hb-buffer-deserialize-text.rl"
	{ if (!parse_int  (tok, p, &pos.y_offset )) return false; }
#line 43 "hb-buffer-deserialize-text.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 21:
#line 71 "hb-buffer-deserialize-text.rl"
	{ if (!parse_int  (tok, p, &pos.x_advance)) return false; }
#line 43 "hb-buffer-deserialize-text.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 27:
#line 72 "hb-buffer-deserialize-text.rl"
	{ if (!parse_int  (tok, p, &pos.y_advance)) return false; }
#line 43 "hb-buffer-deserialize-text.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 24:
#line 73 "hb-buffer-deserialize-text.rl"
	{ if (!parse_uint (tok, p, &info.mask    )) return false; }
#line 43 "hb-buffer-deserialize-text.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 29:
#line 58 "hb-buffer-deserialize-text.rl"
	{
	/* TODO Unescape delimiters. */
	if (!hb_font_glyph_from_string (font,
					tok, p - tok,
					&info.codepoint))
	  return false;
}
#line 73 "hb-buffer-deserialize-text.rl"
	{ if (!parse_uint (tok, p, &info.mask    )) return false; }
#line 43 "hb-buffer-deserialize-text.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 11:
#line 38 "hb-buffer-deserialize-text.rl"
	{
	memset (&info, 0, sizeof (info));
	memset (&pos , 0, sizeof (pos ));
}
#line 51 "hb-buffer-deserialize-text.rl"
	{
	tok = p;
}
#line 58 "hb-buffer-deserialize-text.rl"
	{
	/* TODO Unescape delimiters. */
	if (!hb_font_glyph_from_string (font,
					tok, p - tok,
					&info.codepoint))
	  return false;
}
#line 43 "hb-buffer-deserialize-text.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
#line 953 "hb-buffer-deserialize-text.hh"
	}
	}

	_out: {}
	}

#line 141 "hb-buffer-deserialize-text.rl"


  *end_ptr = p;

  return p == pe && *(p-1) != ']';
}

#endif /* HB_BUFFER_DESERIALIZE_TEXT_HH */
