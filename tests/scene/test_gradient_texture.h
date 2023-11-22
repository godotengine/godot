/**************************************************************************/
/*  test_gradient.h                                                       */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#ifndef TEST_GRADIENT_TEXTURE_H
#define TEST_GRADIENT_IEXTURE_H

#include "scene/resources/gradient_texture.h"

#include "thirdparty/doctest/doctest.h"

namespace TestGradientTexture {

TEST_CASE("[GradientTexture1D] Create gradienttexture1D") {
	Ref<GradientTexture1D> gradient_texture = memnew(GradientTexture1D);

	CHECK(
		Ref<Gradient> test_gradient = memnew(Gradient);
		gradient_texture.set_gradient(test_gradient);
		gradient_texture.get_gradient() == test_gradient;
	)

	CHECK(
		gradient_texture.set_width(82)
		gradient_texture.get_width() == 82;
	)

	CHECK(
		graident_texture.set_use_hdr(true);
		graident_texture.get_using_hdr() == true;
	)
}

TEST_CASE("[GradientTexture1D] Update") {
	Ref<GradientTexture1D> gradient_texture = memnew(GradientTexture1D);
	Ref<Gradient> test_gradient = memnew(Gradient);
	gradient_texture.set_gradient(test_gradient);
	gradient_texture.update_now();

	REQUIRE_FALSE(
		gradient_texture.is_null();
	)

	CHECK(
		gradient_texture.get_gradient() == test_gradient;
	)

	CHECK(
		gradient_texture.get_width() == 256;
	)

	CHECK(
		graident_texture.get_using_hdr() == false;
	)
}

TEST_CASE("[GradientTexture2D] Create gradienttexture2D"){
	Ref<GradientTexture2D> gradient_texture = memnew(GradientTexture2D);

	CHECK()
}
}


    