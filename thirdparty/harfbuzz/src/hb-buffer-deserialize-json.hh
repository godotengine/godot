
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


#line 36 "hb-buffer-deserialize-json.hh"
static const unsigned char _deserialize_json_trans_keys[] = {
	0u, 0u, 9u, 123u, 9u, 34u, 97u, 117u, 120u, 121u, 34u, 34u, 9u, 58u, 9u, 57u, 
	48u, 57u, 9u, 125u, 9u, 125u, 9u, 125u, 34u, 34u, 9u, 58u, 9u, 57u, 48u, 57u, 
	9u, 125u, 9u, 125u, 108u, 108u, 34u, 34u, 9u, 58u, 9u, 57u, 9u, 125u, 9u, 125u, 
	120u, 121u, 34u, 34u, 9u, 58u, 9u, 57u, 48u, 57u, 9u, 125u, 9u, 125u, 34u, 34u, 
	9u, 58u, 9u, 57u, 48u, 57u, 9u, 125u, 9u, 125u, 34u, 34u, 9u, 58u, 9u, 57u, 
	34u, 92u, 9u, 125u, 34u, 92u, 9u, 125u, 9u, 125u, 34u, 34u, 9u, 58u, 9u, 57u, 
	9u, 125u, 9u, 93u, 9u, 123u, 0u, 0u, 0
};

static const char _deserialize_json_key_spans[] = {
	0, 115, 26, 21, 2, 1, 50, 49, 
	10, 117, 117, 117, 1, 50, 49, 10, 
	117, 117, 1, 1, 50, 49, 117, 117, 
	2, 1, 50, 49, 10, 117, 117, 1, 
	50, 49, 10, 117, 117, 1, 50, 49, 
	59, 117, 59, 117, 117, 1, 50, 49, 
	117, 85, 115, 0
};

static const short _deserialize_json_index_offsets[] = {
	0, 0, 116, 143, 165, 168, 170, 221, 
	271, 282, 400, 518, 636, 638, 689, 739, 
	750, 868, 986, 988, 990, 1041, 1091, 1209, 
	1327, 1330, 1332, 1383, 1433, 1444, 1562, 1680, 
	1682, 1733, 1783, 1794, 1912, 2030, 2032, 2083, 
	2133, 2193, 2311, 2371, 2489, 2607, 2609, 2660, 
	2710, 2828, 2914, 3030
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
	5, 1, 6, 7, 1, 1, 8, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 9, 1, 10, 11, 
	1, 12, 1, 12, 12, 12, 12, 12, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 12, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 13, 1, 13, 13, 
	13, 13, 13, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 13, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 14, 1, 1, 15, 16, 16, 
	16, 16, 16, 16, 16, 16, 16, 1, 
	17, 18, 18, 18, 18, 18, 18, 18, 
	18, 18, 1, 19, 19, 19, 19, 19, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 19, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 20, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 21, 
	1, 22, 22, 22, 22, 22, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	22, 1, 1, 1, 1, 1, 1, 1, 
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
	1, 1, 1, 1, 1, 23, 1, 19, 
	19, 19, 19, 19, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 19, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 20, 1, 1, 1, 18, 18, 
	18, 18, 18, 18, 18, 18, 18, 18, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 21, 1, 24, 1, 24, 
	24, 24, 24, 24, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 24, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	25, 1, 25, 25, 25, 25, 25, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 25, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 26, 1, 
	1, 27, 28, 28, 28, 28, 28, 28, 
	28, 28, 28, 1, 29, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 1, 31, 
	31, 31, 31, 31, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 31, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 32, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 33, 1, 31, 31, 31, 
	31, 31, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 31, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	32, 1, 1, 1, 30, 30, 30, 30, 
	30, 30, 30, 30, 30, 30, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 33, 1, 34, 1, 35, 1, 35, 
	35, 35, 35, 35, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 35, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	36, 1, 36, 36, 36, 36, 36, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 36, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 37, 38, 38, 38, 38, 38, 38, 
	38, 38, 38, 1, 39, 39, 39, 39, 
	39, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 39, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 40, 
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
	41, 1, 39, 39, 39, 39, 39, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 39, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 40, 1, 1, 
	1, 42, 42, 42, 42, 42, 42, 42, 
	42, 42, 42, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 41, 1, 
	43, 44, 1, 45, 1, 45, 45, 45, 
	45, 45, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 45, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 46, 1, 
	46, 46, 46, 46, 46, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 46, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 47, 1, 1, 48, 
	49, 49, 49, 49, 49, 49, 49, 49, 
	49, 1, 50, 51, 51, 51, 51, 51, 
	51, 51, 51, 51, 1, 52, 52, 52, 
	52, 52, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 52, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	53, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 54, 1, 52, 52, 52, 52, 52, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 52, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 53, 1, 
	1, 1, 51, 51, 51, 51, 51, 51, 
	51, 51, 51, 51, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 54, 
	1, 55, 1, 55, 55, 55, 55, 55, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 55, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 56, 1, 56, 56, 
	56, 56, 56, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 56, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 57, 1, 1, 58, 59, 59, 
	59, 59, 59, 59, 59, 59, 59, 1, 
	60, 61, 61, 61, 61, 61, 61, 61, 
	61, 61, 1, 62, 62, 62, 62, 62, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 62, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 63, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 64, 
	1, 62, 62, 62, 62, 62, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	62, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 63, 1, 1, 1, 
	61, 61, 61, 61, 61, 61, 61, 61, 
	61, 61, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 64, 1, 65, 
	1, 65, 65, 65, 65, 65, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	65, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 66, 1, 66, 66, 66, 66, 
	66, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 66, 1, 67, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 68, 69, 69, 69, 69, 
	69, 69, 69, 69, 69, 1, 71, 70, 
	70, 70, 70, 70, 70, 70, 70, 70, 
	70, 70, 70, 70, 70, 70, 70, 70, 
	70, 70, 70, 70, 70, 70, 70, 70, 
	70, 70, 70, 70, 70, 70, 70, 70, 
	70, 70, 70, 70, 70, 70, 70, 70, 
	70, 70, 70, 70, 70, 70, 70, 70, 
	70, 70, 70, 70, 70, 70, 70, 70, 
	72, 70, 73, 73, 73, 73, 73, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 73, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 74, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 75, 1, 
	70, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 70, 1, 76, 76, 76, 76, 
	76, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 76, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 77, 
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
	78, 1, 76, 76, 76, 76, 76, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 76, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 77, 1, 1, 
	1, 79, 79, 79, 79, 79, 79, 79, 
	79, 79, 79, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 78, 1, 
	80, 1, 80, 80, 80, 80, 80, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 80, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 81, 1, 81, 81, 81, 
	81, 81, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 81, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 82, 83, 83, 83, 
	83, 83, 83, 83, 83, 83, 1, 76, 
	76, 76, 76, 76, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 76, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 77, 1, 1, 1, 84, 84, 
	84, 84, 84, 84, 84, 84, 84, 84, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 78, 1, 85, 85, 85, 
	85, 85, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 85, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	86, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 87, 1, 0, 0, 0, 0, 0, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 1, 1, 1, 1, 1, 1, 
	1, 1, 0, 1, 1, 1, 1, 1, 
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
	1, 1, 1, 1, 1, 2, 1, 1, 
	0
};

static const char _deserialize_json_trans_targs[] = {
	1, 0, 2, 2, 3, 4, 18, 24, 
	37, 45, 5, 12, 6, 7, 8, 9, 
	11, 9, 11, 10, 2, 49, 10, 49, 
	13, 14, 15, 16, 17, 16, 17, 10, 
	2, 49, 19, 20, 21, 22, 23, 10, 
	2, 49, 23, 25, 31, 26, 27, 28, 
	29, 30, 29, 30, 10, 2, 49, 32, 
	33, 34, 35, 36, 35, 36, 10, 2, 
	49, 38, 39, 40, 43, 44, 40, 41, 
	42, 10, 2, 49, 10, 2, 49, 44, 
	46, 47, 43, 48, 48, 49, 50, 51
};

static const char _deserialize_json_trans_actions[] = {
	0, 0, 1, 0, 0, 0, 0, 0, 
	0, 0, 0, 0, 0, 0, 2, 2, 
	2, 0, 0, 3, 3, 4, 0, 5, 
	0, 0, 2, 2, 2, 0, 0, 6, 
	6, 7, 0, 0, 0, 2, 2, 8, 
	8, 9, 0, 0, 0, 0, 0, 2, 
	2, 2, 0, 0, 10, 10, 11, 0, 
	0, 2, 2, 2, 0, 0, 12, 12, 
	13, 0, 0, 2, 14, 14, 0, 15, 
	0, 16, 16, 17, 18, 18, 19, 15, 
	0, 0, 20, 20, 21, 0, 0, 0
};

static const int deserialize_json_start = 1;
static const int deserialize_json_first_final = 49;
static const int deserialize_json_error = 0;

static const int deserialize_json_en_main = 1;


#line 108 "hb-buffer-deserialize-json.rl"


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
  
#line 512 "hb-buffer-deserialize-json.hh"
	{
	cs = deserialize_json_start;
	}

#line 517 "hb-buffer-deserialize-json.hh"
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
	case 15:
#line 55 "hb-buffer-deserialize-json.rl"
	{ if (unlikely (!buffer->ensure_glyphs ())) return false; }
	break;
	case 21:
#line 56 "hb-buffer-deserialize-json.rl"
	{ if (unlikely (!buffer->ensure_unicode ())) return false; }
	break;
	case 16:
#line 58 "hb-buffer-deserialize-json.rl"
	{
	/* TODO Unescape \" and \\ if found. */
	if (!hb_font_glyph_from_string (font,
					tok, p - tok,
					&info.codepoint))
	  return false;
}
	break;
	case 18:
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
#line 51 "hb-buffer-deserialize-json.rl"
	{
	tok = p;
}
#line 55 "hb-buffer-deserialize-json.rl"
	{ if (unlikely (!buffer->ensure_glyphs ())) return false; }
	break;
	case 20:
#line 51 "hb-buffer-deserialize-json.rl"
	{
	tok = p;
}
#line 56 "hb-buffer-deserialize-json.rl"
	{ if (unlikely (!buffer->ensure_unicode ())) return false; }
	break;
	case 17:
#line 58 "hb-buffer-deserialize-json.rl"
	{
	/* TODO Unescape \" and \\ if found. */
	if (!hb_font_glyph_from_string (font,
					tok, p - tok,
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
	case 19:
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
#line 713 "hb-buffer-deserialize-json.hh"
	}

_again:
	if ( cs == 0 )
		goto _out;
	if ( ++p != pe )
		goto _resume;
	_test_eof: {}
	_out: {}
	}

#line 136 "hb-buffer-deserialize-json.rl"


  *end_ptr = p;

  return p == pe && *(p-1) != ']';
}

#endif /* HB_BUFFER_DESERIALIZE_JSON_HH */
