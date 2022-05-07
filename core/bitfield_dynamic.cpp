/*************************************************************************/
/*  bitfield_dynamic.cpp                                                 */
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

#include "bitfield_dynamic.h"

#include "core/os/memory.h"

#include <string.h>

void BitFieldDynamic::copy_from(const BitFieldDynamic &p_source) {
	create(p_source.get_num_bits(), false);
	memcpy(_data, p_source.get_data(), p_source.get_num_bytes());
}

void BitFieldDynamic::create(uint32_t p_num_bits, bool p_blank) {
	// first delete any initial
	destroy();

	_num_bits = p_num_bits;
	if (p_num_bits) {
		_num_bytes = (p_num_bits / 8) + 1;
		_data = (uint8_t *)memalloc(_num_bytes);

		if (p_blank) {
			blank(false);
		}
	}
}

void BitFieldDynamic::destroy() {
	if (_data) {
		memfree(_data);
		_data = nullptr;
	}

	_num_bytes = 0;
	_num_bits = 0;
}

void BitFieldDynamic::blank(bool p_set_or_zero) {
	if (p_set_or_zero) {
		memset(_data, 255, _num_bytes);
	} else {
		memset(_data, 0, _num_bytes);
	}
}

void BitFieldDynamic::invert() {
	for (uint32_t n = 0; n < _num_bytes; n++) {
		_data[n] = ~_data[n];
	}
}
