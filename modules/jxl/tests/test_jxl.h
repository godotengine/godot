/**************************************************************************/
/*  test_jxl.h                                                            */
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

#pragma once

#include "../image_loader_jxl.h"

#include "core/io/file_access.h"
#include "core/io/image.h"
#include "tests/test_macros.h"
#include "tests/test_utils.h"

namespace TestJXL {

static Ref<Image> load_jxl(const String &p_file, Error &r_error, BitField<ImageFormatLoader::LoaderFlags> p_flags = ImageFormatLoader::FLAG_NONE) {
	Ref<ImageLoaderJXL> loader;
	loader.instantiate();
	Ref<FileAccess> f = FileAccess::open(TestUtils::get_data_path("images/" + p_file), FileAccess::READ, &r_error);
	if (f.is_null()) {
		return Ref<Image>();
	}
	Ref<Image> image;
	image.instantiate();
	r_error = loader->load_image(image, f, p_flags, 1.0f);
	return image;
}

TEST_CASE("[JXL] Recognized extension") {
	Ref<ImageLoaderJXL> loader;
	loader.instantiate();
	List<String> extensions;
	loader->get_recognized_extensions(&extensions);
	CHECK(extensions.size() == 1);
	CHECK(extensions.front()->get() == "jxl");
}

TEST_CASE("[JXL] Decode test images") {
	// The number of channels (color plus optional alpha) and the precision decide
	// the resulting Godot format: 8-bit originals decode to integers, 32-bit floats
	// to full floats, and everything in between to half floats. These fixtures cover
	// every supported combination, in both lossless and lossy variants.
	struct Case {
		const char *file;
		int width;
		int height;
		Image::Format format;
	};
	const Case cases[] = {
		// 8-bit integer.
		{ "jxl_gray_8bit.jxl", 16, 12, Image::FORMAT_L8 },
		{ "jxl_gray_alpha_8bit.jxl", 18, 12, Image::FORMAT_LA8 },
		{ "jxl_rgb_8bit.jxl", 20, 12, Image::FORMAT_RGB8 },
		{ "jxl_rgb_alpha_8bit.jxl", 22, 12, Image::FORMAT_RGBA8 },
		// 16-bit integer.
		{ "jxl_gray_16bit.jxl", 24, 12, Image::FORMAT_RH },
		{ "jxl_gray_alpha_16bit.jxl", 26, 12, Image::FORMAT_RGH },
		{ "jxl_rgb_16bit.jxl", 28, 12, Image::FORMAT_RGBH },
		{ "jxl_rgb_alpha_16bit.jxl", 30, 12, Image::FORMAT_RGBAH },
		// 32-bit floating point.
		{ "jxl_gray_float.jxl", 32, 12, Image::FORMAT_RF },
		{ "jxl_rgb_float.jxl", 34, 12, Image::FORMAT_RGBF },
		// Lossy.
		{ "jxl_gray_8bit_lossy.jxl", 36, 16, Image::FORMAT_L8 },
		{ "jxl_rgb_8bit_lossy.jxl", 38, 16, Image::FORMAT_RGB8 },
		{ "jxl_rgb_alpha_16bit_lossy.jxl", 40, 16, Image::FORMAT_RGBAH },
	};

	for (const Case &c : cases) {
		Error err = FAILED;
		Ref<Image> image = load_jxl(c.file, err);
		CHECK_MESSAGE(err == OK, vformat("'%s' should load successfully.", c.file));
		REQUIRE_MESSAGE(image.is_valid(), vformat("'%s' should produce a valid image.", c.file));
		CHECK_MESSAGE(image->get_width() == c.width, vformat("'%s' should have width %d.", c.file, c.width));
		CHECK_MESSAGE(image->get_height() == c.height, vformat("'%s' should have height %d.", c.file, c.height));
		CHECK_MESSAGE(image->get_format() == c.format, vformat("'%s' should decode to the expected format.", c.file));
		CHECK_FALSE_MESSAGE(image->is_empty(), vformat("'%s' should not be empty.", c.file));
	}
}

TEST_CASE("[JXL] Invalid data is rejected") {
	const String path = TestUtils::get_temp_path("invalid.jxl");
	{
		Ref<FileAccess> w = FileAccess::open(path, FileAccess::WRITE);
		REQUIRE(w.is_valid());
		w->store_string("This is not a JPEG XL file.");
	}

	Ref<ImageLoaderJXL> loader;
	loader.instantiate();
	Error err = OK;
	Ref<FileAccess> f = FileAccess::open(path, FileAccess::READ, &err);
	REQUIRE(f.is_valid());
	Ref<Image> image;
	image.instantiate();

	ERR_PRINT_OFF;
	const Error load_err = loader->load_image(image, f, ImageFormatLoader::FLAG_NONE, 1.0f);
	ERR_PRINT_ON;
	CHECK(load_err != OK);
}

TEST_CASE("[JXL] Force linear conversion") {
	Error err_default = FAILED;
	Error err_linear = FAILED;
	Ref<Image> as_stored = load_jxl("jxl_rgb_8bit.jxl", err_default);
	Ref<Image> as_linear = load_jxl("jxl_rgb_8bit.jxl", err_linear, ImageFormatLoader::FLAG_FORCE_LINEAR);

	CHECK(err_default == OK);
	CHECK(err_linear == OK);
	REQUIRE(as_stored.is_valid());
	REQUIRE(as_linear.is_valid());
	// Converting the sRGB original to linear changes its non-trivial samples.
	CHECK(as_stored->get_data() != as_linear->get_data());
}

} // namespace TestJXL
