/*************************************************************************/
/*  bit_mask.h                                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef BIT_MASK_H
#define BIT_MASK_H

#include "resource.h"
#include "io/resource_loader.h"


class BitMap : public Resource {

	OBJ_TYPE(BitMap,Resource);

	Vector<uint8_t> bitmask;
	int width;
	int height;

protected:
	void _set_data(const Dictionary& p_d);
	Dictionary _get_data() const;

	static void _bind_methods();
public:


	void create(const Size2& p_size);
	void create_from_image_alpha(const Image& p_image);

	void set_bit(const Point2& p_pos,bool p_value);
	bool get_bit(const Point2& p_pos) const;
	void set_bit_rect(const Rect2& p_rect,bool p_value);
	int get_true_bit_count() const;

	Size2 get_size() const;

	BitMap();
};

class ResourceFormatLoaderBitMap : public ResourceFormatLoader {

public:

	virtual RES load(const String &p_path,const String& p_original_path="",Error *r_error=NULL);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String& p_type) const;
	virtual String get_resource_type(const String &p_path) const;

	ResourceFormatLoaderBitMap();
};

#endif // BIT_MASK_H
