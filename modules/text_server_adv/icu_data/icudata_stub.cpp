/*************************************************************************/
/*  icudata_stub.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/

#include "unicode/udata.h"
#include "unicode/utypes.h"
#include "unicode/uversion.h"

typedef struct {
	uint16_t header_size;
	uint8_t magic_1, magic_2;
	UDataInfo info;
	char padding[8];
	uint32_t count, reserved;
	int fake_name_and_data[4];
} ICU_data_header;

extern "C" U_EXPORT const ICU_data_header U_ICUDATA_ENTRY_POINT = {
	32,
	0xDA, 0x27,
	{ sizeof(UDataInfo),
			0,
#if U_IS_BIG_ENDIAN
			1,
#else
			0,
#endif
			U_CHARSET_FAMILY,
			sizeof(UChar),
			0,
			{ 0x54, 0x6F, 0x43, 0x50 },
			{ 1, 0, 0, 0 },
			{ 0, 0, 0, 0 } },
	{ 0, 0, 0, 0, 0, 0, 0, 0 },
	0, 0,
	{ 0, 0, 0, 0 }
};
