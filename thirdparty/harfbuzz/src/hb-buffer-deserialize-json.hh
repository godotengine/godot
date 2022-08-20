
#line 1 "hb-buffer-deserialize-json.rl"
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

#ifndef HB_BUFFER_DESERIALIZE_JSON_HH
#define HB_BUFFER_DESERIALIZE_JSON_HH

#include "hb.hh"


#line 33 "hb-buffer-deserialize-json.hh"
static const unsigned char _deserialize_json_trans_keys[] = {
	0u, 0u, 9u, 123u, 9u, 34u, 97u, 117u, 120u, 121u, 34u, 34u, 9u, 58u, 9u, 57u, 
	48u, 57u, 9u, 125u, 9u, 125u, 9u, 125u, 34u, 34u, 9u, 58u, 9u, 57u, 48u, 57u, 
	9u, 125u, 9u, 125u, 108u, 108u, 34u, 34u, 9u, 58u, 9u, 57u, 9u, 125u, 9u, 125u, 
	120u, 121u, 34u, 34u, 9u, 58u, 9u, 57u, 48u, 57u, 9u, 125u, 9u, 125u, 34u, 34u, 
	9u, 58u, 9u, 57u, 48u, 57u, 9u, 125u, 9u, 125u, 108u, 108u, 34u, 34u, 9u, 58u, 
	9u, 57u, 9u, 125u, 9u, 125u, 34u, 34u, 9u, 58u, 9u, 57u, 34u, 92u, 9u, 125u, 
	34u, 92u, 9u, 125u, 9u, 125u, 34u, 34u, 9u, 58u, 9u, 57u, 9u, 125u, 9u, 93u, 
	9u, 123u, 0u, 0u, 0
};

static const char _deserialize_json_key_spans[] = {
	0, 115, 26, 21, 2, 1, 50, 49, 
	10, 117, 117, 117, 1, 50, 49, 10, 
	117, 117, 1, 1, 50, 49, 117, 117, 
	2, 1, 50, 49, 10, 117, 117, 1, 
	50, 49, 10, 117, 117, 1, 1, 50, 
	49, 117, 117, 1, 50, 49, 59, 117, 
	59, 117, 117, 1, 50, 49, 117, 85, 
	115, 0
};

static const short _deserialize_json_index_offsets[] = {
	0, 0, 116, 143, 165, 168, 170, 221, 
	271, 282, 400, 518, 636, 638, 689, 739, 
	750, 868, 986, 988, 990, 1041, 1091, 1209, 
	1327, 1330, 1332, 1383, 1433, 1444, 1562, 1680, 
	1682, 1733, 1783, 1794, 1912, 2030, 2032, 2034, 
	2085, 2135, 2253, 2371, 2373, 2424, 2474, 2534, 
	2652, 2712, 2830, 2948, 2950, 3001, 3051, 3169, 
	3255, 3371
};

static const char _deserialize_json_indicies[] = {
	0, 0, 0, 0, 0, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	0, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 2, 1, 3, 3, 3, 
	3, 3, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 3, 1, 4, 1, 
	5, 1, 6, 7, 1, 8, 9, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 10, 1, 11, 12, 
	1, 13, 1, 13, 13, 13, 13, 13, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 13, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 14, 1, 14, 14, 
	14, 14, 14, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 14, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 15, 1, 1, 16, 17, 17, 
	17, 17, 17, 17, 17, 17, 17, 1, 
	18, 19, 19, 19, 19, 19, 19, 19, 
	19, 19, 1, 20, 20, 20, 20, 20, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 20, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 21, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 22, 
	1, 23, 23, 23, 23, 23, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	23, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 3, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 24, 1, 20, 
	20, 20, 20, 20, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 20, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 21, 1, 1, 1, 19, 19, 
	19, 19, 19, 19, 19, 19, 19, 19, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 22, 1, 25, 1, 25, 
	25, 25, 25, 25, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 25, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	26, 1, 26, 26, 26, 26, 26, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 26, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 27, 1, 
	1, 28, 29, 29, 29, 29, 29, 29, 
	29, 29, 29, 1, 30, 31, 31, 31, 
	31, 31, 31, 31, 31, 31, 1, 32, 
	32, 32, 32, 32, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 32, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 33, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 34, 1, 32, 32, 32, 
	32, 32, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 32, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	33, 1, 1, 1, 31, 31, 31, 31, 
	31, 31, 31, 31, 31, 31, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 34, 1, 35, 1, 36, 1, 36, 
	36, 36, 36, 36, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 36, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	37, 1, 37, 37, 37, 37, 37, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 37, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 38, 39, 39, 39, 39, 39, 39, 
	39, 39, 39, 1, 40, 40, 40, 40, 
	40, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 40, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 41, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	42, 1, 40, 40, 40, 40, 40, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 40, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 41, 1, 1, 
	1, 43, 43, 43, 43, 43, 43, 43, 
	43, 43, 43, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 42, 1, 
	44, 45, 1, 46, 1, 46, 46, 46, 
	46, 46, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 46, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 47, 1, 
	47, 47, 47, 47, 47, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 47, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 48, 1, 1, 49, 
	50, 50, 50, 50, 50, 50, 50, 50, 
	50, 1, 51, 52, 52, 52, 52, 52, 
	52, 52, 52, 52, 1, 53, 53, 53, 
	53, 53, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 53, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	54, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 55, 1, 53, 53, 53, 53, 53, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 53, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 54, 1, 
	1, 1, 52, 52, 52, 52, 52, 52, 
	52, 52, 52, 52, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 55, 
	1, 56, 1, 56, 56, 56, 56, 56, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 56, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 57, 1, 57, 57, 
	57, 57, 57, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 57, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 58, 1, 1, 59, 60, 60, 
	60, 60, 60, 60, 60, 60, 60, 1, 
	61, 62, 62, 62, 62, 62, 62, 62, 
	62, 62, 1, 63, 63, 63, 63, 63, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 63, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 64, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 65, 
	1, 63, 63, 63, 63, 63, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	63, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 64, 1, 1, 1, 
	62, 62, 62, 62, 62, 62, 62, 62, 
	62, 62, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 65, 1, 66, 
	1, 67, 1, 67, 67, 67, 67, 67, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 67, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 68, 1, 68, 68, 
	68, 68, 68, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 68, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 69, 70, 70, 
	70, 70, 70, 70, 70, 70, 70, 1, 
	71, 71, 71, 71, 71, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 71, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 72, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 73, 1, 71, 71, 
	71, 71, 71, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 71, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 72, 1, 1, 1, 74, 74, 74, 
	74, 74, 74, 74, 74, 74, 74, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 73, 1, 75, 1, 75, 75, 
	75, 75, 75, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 75, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 76, 
	1, 76, 76, 76, 76, 76, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	76, 1, 77, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	78, 79, 79, 79, 79, 79, 79, 79, 
	79, 79, 1, 81, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 80, 80, 80, 
	80, 80, 80, 80, 80, 82, 80, 83, 
	83, 83, 83, 83, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 83, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 84, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 85, 1, 80, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 80, 
	1, 86, 86, 86, 86, 86, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	86, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 87, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 88, 1, 86, 
	86, 86, 86, 86, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 86, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 87, 1, 1, 1, 89, 89, 
	89, 89, 89, 89, 89, 89, 89, 89, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 88, 1, 90, 1, 90, 
	90, 90, 90, 90, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 90, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	91, 1, 91, 91, 91, 91, 91, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 91, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 92, 93, 93, 93, 93, 93, 93, 
	93, 93, 93, 1, 86, 86, 86, 86, 
	86, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 86, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 87, 
	1, 1, 1, 94, 94, 94, 94, 94, 
	94, 94, 94, 94, 94, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	88, 1, 95, 95, 95, 95, 95, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 95, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 96, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 97, 1, 
	0, 0, 0, 0, 0, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 0, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 2, 1, 1, 0
};

static const char _deserialize_json_trans_targs[] = {
	1, 0, 2, 2, 3, 4, 18, 24, 
	37, 43, 51, 5, 12, 6, 7, 8, 
	9, 11, 9, 11, 10, 2, 55, 10, 
	55, 13, 14, 15, 16, 17, 16, 17, 
	10, 2, 55, 19, 20, 21, 22, 23, 
	10, 2, 55, 23, 25, 31, 26, 27, 
	28, 29, 30, 29, 30, 10, 2, 55, 
	32, 33, 34, 35, 36, 35, 36, 10, 
	2, 55, 38, 39, 40, 41, 42, 10, 
	2, 55, 42, 44, 45, 46, 49, 50, 
	46, 47, 48, 10, 2, 55, 10, 2, 
	55, 50, 52, 53, 49, 54, 54, 55, 
	56, 57
};

static const char _deserialize_json_trans_actions[] = {
	0, 0, 1, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 0, 2, 
	2, 2, 0, 0, 3, 3, 4, 0, 
	5, 0, 0, 2, 2, 2, 0, 0, 
	6, 6, 7, 0, 0, 0, 2, 2, 
	8, 8, 9, 0, 0, 0, 0, 0, 
	2, 2, 2, 0, 0, 10, 10, 11, 
	0, 0, 2, 2, 2, 0, 0, 12, 
	12, 13, 0, 0, 0, 2, 2, 14, 
	14, 15, 0, 0, 0, 2, 16, 16, 
	0, 17, 0, 18, 18, 19, 20, 20, 
	21, 17, 0, 0, 22, 22, 23, 0, 
	0, 0
};

static const int deserialize_json_start = 1;
static const int deserialize_json_first_final = 55;
static const int deserialize_json_error = 0;

static const int deserialize_json_en_main = 1;


#line 111 "hb-buffer-deserialize-json.rl"


static hb_bool_t
_hb_buffer_deserialize_json (hb_buffer_t *buffer,
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
  if (p < pe && *p == (buffer->len ? ',' : '['))
  {
    *end_ptr = ++p;
  }

  const char *tok = nullptr;
  int cs;
  hb_glyph_info_t info = {0};
  hb_glyph_position_t pos = {0};
  
#line 554 "hb-buffer-deserialize-json.hh"
	{
	cs = deserialize_json_start;
	}

#line 557 "hb-buffer-deserialize-json.hh"
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
	_keys = _deserialize_json_trans_keys + (cs<<1);
	_inds = _deserialize_json_indicies + _deserialize_json_index_offsets[cs];

	_slen = _deserialize_json_key_spans[cs];
	_trans = _inds[ _slen > 0 && _keys[0] <=(*p) &&
		(*p) <= _keys[1] ?
		(*p) - _keys[0] : _slen ];

	cs = _deserialize_json_trans_targs[_trans];

	if ( _deserialize_json_trans_actions[_trans] == 0 )
		goto _again;

	switch ( _deserialize_json_trans_actions[_trans] ) {
	case 1:
#line 38 "hb-buffer-deserialize-json.rl"
	{
	memset (&info, 0, sizeof (info));
	memset (&pos , 0, sizeof (pos ));
}
	break;
	case 5:
#line 43 "hb-buffer-deserialize-json.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 2:
#line 51 "hb-buffer-deserialize-json.rl"
	{
	tok = p;
}
	break;
	case 17:
#line 55 "hb-buffer-deserialize-json.rl"
	{ if (unlikely (!buffer->ensure_glyphs ())) return false; }
	break;
	case 23:
#line 56 "hb-buffer-deserialize-json.rl"
	{ if (unlikely (!buffer->ensure_unicode ())) return false; }
	break;
	case 18:
#line 58 "hb-buffer-deserialize-json.rl"
	{
	/* TODO Unescape \" and \\ if found. */
	if (!hb_font_glyph_from_string (font,
					tok+1, p - tok - 2, /* Skip "" */
					&info.codepoint))
	  return false;
}
	break;
	case 20:
#line 66 "hb-buffer-deserialize-json.rl"
	{ if (!parse_uint (tok, p, &info.codepoint)) return false; }
	break;
	case 8:
#line 67 "hb-buffer-deserialize-json.rl"
	{ if (!parse_uint (tok, p, &info.cluster )) return false; }
	break;
	case 10:
#line 68 "hb-buffer-deserialize-json.rl"
	{ if (!parse_int  (tok, p, &pos.x_offset )) return false; }
	break;
	case 12:
#line 69 "hb-buffer-deserialize-json.rl"
	{ if (!parse_int  (tok, p, &pos.y_offset )) return false; }
	break;
	case 3:
#line 70 "hb-buffer-deserialize-json.rl"
	{ if (!parse_int  (tok, p, &pos.x_advance)) return false; }
	break;
	case 6:
#line 71 "hb-buffer-deserialize-json.rl"
	{ if (!parse_int  (tok, p, &pos.y_advance)) return false; }
	break;
	case 14:
#line 72 "hb-buffer-deserialize-json.rl"
	{ if (!parse_uint (tok, p, &info.mask    )) return false; }
	break;
	case 16:
#line 51 "hb-buffer-deserialize-json.rl"
	{
	tok = p;
}
#line 55 "hb-buffer-deserialize-json.rl"
	{ if (unlikely (!buffer->ensure_glyphs ())) return false; }
	break;
	case 22:
#line 51 "hb-buffer-deserialize-json.rl"
	{
	tok = p;
}
#line 56 "hb-buffer-deserialize-json.rl"
	{ if (unlikely (!buffer->ensure_unicode ())) return false; }
	break;
	case 19:
#line 58 "hb-buffer-deserialize-json.rl"
	{
	/* TODO Unescape \" and \\ if found. */
	if (!hb_font_glyph_from_string (font,
					tok+1, p - tok - 2, /* Skip "" */
					&info.codepoint))
	  return false;
}
#line 43 "hb-buffer-deserialize-json.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 21:
#line 66 "hb-buffer-deserialize-json.rl"
	{ if (!parse_uint (tok, p, &info.codepoint)) return false; }
#line 43 "hb-buffer-deserialize-json.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 9:
#line 67 "hb-buffer-deserialize-json.rl"
	{ if (!parse_uint (tok, p, &info.cluster )) return false; }
#line 43 "hb-buffer-deserialize-json.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 11:
#line 68 "hb-buffer-deserialize-json.rl"
	{ if (!parse_int  (tok, p, &pos.x_offset )) return false; }
#line 43 "hb-buffer-deserialize-json.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 13:
#line 69 "hb-buffer-deserialize-json.rl"
	{ if (!parse_int  (tok, p, &pos.y_offset )) return false; }
#line 43 "hb-buffer-deserialize-json.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 4:
#line 70 "hb-buffer-deserialize-json.rl"
	{ if (!parse_int  (tok, p, &pos.x_advance)) return false; }
#line 43 "hb-buffer-deserialize-json.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 7:
#line 71 "hb-buffer-deserialize-json.rl"
	{ if (!parse_int  (tok, p, &pos.y_advance)) return false; }
#line 43 "hb-buffer-deserialize-json.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
	case 15:
#line 72 "hb-buffer-deserialize-json.rl"
	{ if (!parse_uint (tok, p, &info.mask    )) return false; }
#line 43 "hb-buffer-deserialize-json.rl"
	{
	buffer->add_info (info);
	if (unlikely (!buffer->successful))
	  return false;
	buffer->pos[buffer->len - 1] = pos;
	*end_ptr = p;
}
	break;
#line 735 "hb-buffer-deserialize-json.hh"
	}

_again:
	if ( cs == 0 )
		goto _out;
	if ( ++p != pe )
		goto _resume;
	_test_eof: {}
	_out: {}
	}

#line 139 "hb-buffer-deserialize-json.rl"


  *end_ptr = p;

  return p == pe && *(p-1) != ']';
}

#endif /* HB_BUFFER_DESERIALIZE_JSON_HH */
