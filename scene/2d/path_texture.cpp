/*************************************************************************/
/*  path_texture.cpp                                                     */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
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
#include "path_texture.h"

void PathTexture::set_begin_texture(const Ref<Texture> &p_texture) {

	begin = p_texture;
	update();
}

Ref<Texture> PathTexture::get_begin_texture() const {

	return begin;
}

void PathTexture::set_repeat_texture(const Ref<Texture> &p_texture) {

	repeat = p_texture;
	update();
}
Ref<Texture> PathTexture::get_repeat_texture() const {

	return repeat;
}

void PathTexture::set_end_texture(const Ref<Texture> &p_texture) {

	end = p_texture;
	update();
}
Ref<Texture> PathTexture::get_end_texture() const {

	return end;
}

void PathTexture::set_subdivisions(int p_amount) {

	ERR_FAIL_INDEX(p_amount, 32);
	subdivs = p_amount;
	update();
}

int PathTexture::get_subdivisions() const {

	return subdivs;
}

void PathTexture::set_overlap(int p_amount) {

	overlap = p_amount;
	update();
}
int PathTexture::get_overlap() const {

	return overlap;
}

PathTexture::PathTexture() {

	overlap = 0;
	subdivs = 1;
}
