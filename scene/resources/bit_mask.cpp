/*************************************************************************/
/*  bit_mask.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "bit_mask.h"
#include "io/image_loader.h"

void BitMap::create(const Size2 &p_size) {

	ERR_FAIL_COND(p_size.width < 1);
	ERR_FAIL_COND(p_size.height < 1);

	width = p_size.width;
	height = p_size.height;
	bitmask.resize(((width * height) / 8) + 1);
	zeromem(bitmask.ptr(), bitmask.size());
}

void BitMap::create_from_image_alpha(const Ref<Image> &p_image) {

	ERR_FAIL_COND(p_image.is_null() || p_image->empty());
	Ref<Image> img = p_image->duplicate();
	img->convert(Image::FORMAT_LA8);
	ERR_FAIL_COND(img->get_format() != Image::FORMAT_LA8);

	create(Size2(img->get_width(), img->get_height()));

	PoolVector<uint8_t>::Read r = img->get_data().read();
	uint8_t *w = bitmask.ptr();

	for (int i = 0; i < width * height; i++) {

		int bbyte = i / 8;
		int bbit = i % 8;
		if (r[i * 2])
			w[bbyte] |= (1 << bbit);
	}
}

void BitMap::set_bit_rect(const Rect2 &p_rect, bool p_value) {

	Rect2i current = Rect2i(0, 0, width, height).clip(p_rect);
	uint8_t *data = bitmask.ptr();

	for (int i = current.position.x; i < current.position.x + current.size.x; i++) {

		for (int j = current.position.y; j < current.position.y + current.size.y; j++) {

			int ofs = width * j + i;
			int bbyte = ofs / 8;
			int bbit = ofs % 8;

			uint8_t b = data[bbyte];

			if (p_value)
				b |= (1 << bbit);
			else
				b &= !(1 << bbit);

			data[bbyte] = b;
		}
	}
}

int BitMap::get_true_bit_count() const {

	int ds = bitmask.size();
	const uint8_t *d = bitmask.ptr();
	int c = 0;

	//fast, almot branchless version

	for (int i = 0; i < ds; i++) {

		c += (d[i] & (1 << 7)) >> 7;
		c += (d[i] & (1 << 6)) >> 6;
		c += (d[i] & (1 << 5)) >> 5;
		c += (d[i] & (1 << 4)) >> 4;
		c += (d[i] & (1 << 3)) >> 3;
		c += (d[i] & (1 << 2)) >> 2;
		c += d[i] & 1;
	}

	return c;
}

void BitMap::set_bit(const Point2 &p_pos, bool p_value) {

	int x = Math::fast_ftoi(p_pos.x);
	int y = Math::fast_ftoi(p_pos.y);

	ERR_FAIL_INDEX(x, width);
	ERR_FAIL_INDEX(y, height);

	int ofs = width * y + x;
	int bbyte = ofs / 8;
	int bbit = ofs % 8;

	uint8_t b = bitmask[bbyte];

	if (p_value)
		b |= (1 << bbit);
	else
		b &= !(1 << bbit);

	bitmask[bbyte] = b;
}

bool BitMap::get_bit(const Point2 &p_pos) const {

	int x = Math::fast_ftoi(p_pos.x);
	int y = Math::fast_ftoi(p_pos.y);
	ERR_FAIL_INDEX_V(x, width, false);
	ERR_FAIL_INDEX_V(y, height, false);

	int ofs = width * y + x;
	int bbyte = ofs / 8;
	int bbit = ofs % 8;

	return (bitmask[bbyte] & (1 << bbit)) != 0;
}

Size2 BitMap::get_size() const {

	return Size2(width, height);
}

void BitMap::_set_data(const Dictionary &p_d) {

	ERR_FAIL_COND(!p_d.has("size"));
	ERR_FAIL_COND(!p_d.has("data"));

	create(p_d["size"]);
	bitmask = p_d["data"];
}

Dictionary BitMap::_get_data() const {

	Dictionary d;
	d["size"] = get_size();
	d["data"] = bitmask;
	return d;
}

void BitMap::_bind_methods() {

	ClassDB::bind_method(D_METHOD("create", "size"), &BitMap::create);
	ClassDB::bind_method(D_METHOD("create_from_image_alpha", "image"), &BitMap::create_from_image_alpha);

	ClassDB::bind_method(D_METHOD("set_bit", "position", "bit"), &BitMap::set_bit);
	ClassDB::bind_method(D_METHOD("get_bit", "position"), &BitMap::get_bit);

	ClassDB::bind_method(D_METHOD("set_bit_rect", "p_rect", "bit"), &BitMap::set_bit_rect);
	ClassDB::bind_method(D_METHOD("get_true_bit_count"), &BitMap::get_true_bit_count);

	ClassDB::bind_method(D_METHOD("get_size"), &BitMap::get_size);

	ClassDB::bind_method(D_METHOD("_set_data"), &BitMap::_set_data);
	ClassDB::bind_method(D_METHOD("_get_data"), &BitMap::_get_data);

	ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "data", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NOEDITOR), "_set_data", "_get_data");
}

BitMap::BitMap() {

	width = 0;
	height = 0;
}

//////////////////////////////////////
