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
#include "../resource_saver_jxl.h"

#include "core/io/file_access.h"
#include "core/io/image.h"
#include "core/io/resource_saver.h"
#include "scene/resources/compressed_texture.h"
#include "scene/resources/image_texture.h"
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
	CHECK(as_stored->get_data() != as_linear->get_data());
}

TEST_CASE("[JXL] Load from buffer") {
	const Vector<uint8_t> buffer = FileAccess::get_file_as_bytes(TestUtils::get_data_path("images/jxl_rgb_alpha_16bit.jxl"));
	REQUIRE_FALSE(buffer.is_empty());

	Ref<Image> image;
	image.instantiate();
	CHECK(image->load_jxl_from_buffer(buffer) == OK);
	CHECK(image->get_width() == 30);
	CHECK(image->get_height() == 12);
	CHECK(image->get_format() == Image::FORMAT_RGBAH);

	Error err = FAILED;
	Ref<Image> from_file = load_jxl("jxl_rgb_alpha_16bit.jxl", err);
	REQUIRE(err == OK);
	CHECK(image->get_data() == from_file->get_data());

	Vector<uint8_t> invalid;
	invalid.resize(16);
	invalid.fill(0);
	Ref<Image> invalid_image;
	invalid_image.instantiate();
	ERR_PRINT_OFF;
	CHECK(invalid_image->load_jxl_from_buffer(invalid) != OK);
	ERR_PRINT_ON;
	CHECK(invalid_image->is_empty());
}

TEST_CASE("[JXL] Load from file") {
	Ref<Image> image = Image::load_from_file(TestUtils::get_data_path("images/jxl_rgb_8bit.jxl"));
	REQUIRE(image.is_valid());
	CHECK(image->get_width() == 20);
	CHECK(image->get_height() == 12);
	CHECK(image->get_format() == Image::FORMAT_RGB8);
}

static Ref<Image> make_gradient(int p_width, int p_height, Image::Format p_format) {
	Ref<Image> image = Image::create_empty(p_width, p_height, false, p_format);
	for (int y = 0; y < p_height; y++) {
		for (int x = 0; x < p_width; x++) {
			// Use values above 1.0 in float formats.
			const float scale = Image::get_format_pixel_size(p_format) > 4 ? 4.0f : 1.0f;
			image->set_pixel(x, y, Color(scale * x / p_width, scale * y / p_height, 0.5f, 0.25f + 0.75f * x / p_width));
		}
	}
	return image;
}

TEST_CASE("[JXL] Lossless save round trip") {
	struct Case {
		Image::Format source;
		Image::Format decoded;
	};
	const Case cases[] = {
		{ Image::FORMAT_L8, Image::FORMAT_L8 },
		{ Image::FORMAT_LA8, Image::FORMAT_LA8 },
		{ Image::FORMAT_RGB8, Image::FORMAT_RGB8 },
		{ Image::FORMAT_RGBA8, Image::FORMAT_RGBA8 },
		{ Image::FORMAT_RH, Image::FORMAT_RH },
		{ Image::FORMAT_RGBH, Image::FORMAT_RGBH },
		{ Image::FORMAT_RGBAH, Image::FORMAT_RGBAH },
		{ Image::FORMAT_RF, Image::FORMAT_RF },
		{ Image::FORMAT_RGBF, Image::FORMAT_RGBF },
		{ Image::FORMAT_RGBAF, Image::FORMAT_RGBAF },
		// These come back in a wider format.
		{ Image::FORMAT_RGH, Image::FORMAT_RGBH },
		{ Image::FORMAT_RGF, Image::FORMAT_RGBF },
		{ Image::FORMAT_RGBE9995, Image::FORMAT_RGBF },
	};

	for (const Case &c : cases) {
		const String name = Image::get_format_name(c.source);
		Ref<Image> image = make_gradient(19, 7, c.source);

		const Vector<uint8_t> buffer = image->save_jxl_to_buffer();
		REQUIRE_MESSAGE(!buffer.is_empty(), vformat("%s should encode.", name));

		Ref<Image> decoded;
		decoded.instantiate();
		CHECK_MESSAGE(decoded->load_jxl_from_buffer(buffer) == OK, vformat("%s should decode.", name));
		CHECK_MESSAGE(decoded->get_format() == c.decoded, vformat("%s should keep its format.", name));
		decoded->convert(c.source);
		CHECK_MESSAGE(decoded->get_data() == image->get_data(), vformat("%s should round trip losslessly.", name));
	}
}

TEST_CASE("[JXL] Lossless save of signed and special float values") {
	// Mixed signs in one float channel used to fail.
	const float values[] = { -1.5f, 100.0f, 0.0f, -0.0f, 1e-30f, -3e38f, 3e38f, INFINITY, -INFINITY, 0.5f };
	for (int size : { 1, 2, 3, 8, 33, 64 }) {
		for (Image::Format format : { Image::FORMAT_RF, Image::FORMAT_RGBF, Image::FORMAT_RGBAF }) {
			Ref<Image> image = Image::create_empty(size, size, false, format);
			int index = size;
			for (int y = 0; y < size; y++) {
				for (int x = 0; x < size; x++) {
					image->set_pixel(x, y, Color(values[index % 10], values[(index * 3 + 1) % 10], values[(index * 7 + 2) % 10], values[(index * 9 + 3) % 10]));
					index++;
				}
			}

			const Vector<uint8_t> buffer = image->save_jxl_to_buffer();
			REQUIRE_MESSAGE(!buffer.is_empty(), vformat("%dx%d %s should encode.", size, size, Image::get_format_name(format)));
			Ref<Image> decoded;
			decoded.instantiate();
			CHECK(decoded->load_jxl_from_buffer(buffer) == OK);
			CHECK_MESSAGE(decoded->get_data() == image->get_data(), vformat("%dx%d %s should round trip losslessly.", size, size, Image::get_format_name(format)));
		}
	}
}

TEST_CASE("[JXL] Lossy save") {
	Ref<Image> image = make_gradient(64, 48, Image::FORMAT_RGBA8);
	const Vector<uint8_t> low = image->save_jxl_to_buffer(true, 0.1f);
	const Vector<uint8_t> high = image->save_jxl_to_buffer(true, 0.95f);
	REQUIRE_FALSE(low.is_empty());
	REQUIRE_FALSE(high.is_empty());
	CHECK(low.size() < high.size());

	Ref<Image> decoded;
	decoded.instantiate();
	CHECK(decoded->load_jxl_from_buffer(high) == OK);
	CHECK(decoded->get_width() == 64);
	CHECK(decoded->get_height() == 48);
	CHECK(decoded->get_format() == Image::FORMAT_RGBA8);
	const Color expected = image->get_pixel(40, 30);
	const Color actual = decoded->get_pixel(40, 30);
	CHECK(Math::abs(actual.r - expected.r) < 0.05f);
	CHECK(Math::abs(actual.g - expected.g) < 0.05f);
	CHECK(Math::abs(actual.b - expected.b) < 0.05f);

	Ref<Image> hdr = make_gradient(32, 32, Image::FORMAT_RGBF);
	Ref<Image> hdr_decoded;
	hdr_decoded.instantiate();
	CHECK(hdr_decoded->load_jxl_from_buffer(hdr->save_jxl_to_buffer(true, 0.9f)) == OK);
	CHECK(hdr_decoded->get_format() == Image::FORMAT_RGBF);
	CHECK(hdr_decoded->get_pixel(31, 31).r > 2.0f);

	CHECK_FALSE(image->save_jxl_to_buffer(true, 1.0f).is_empty());
	CHECK_FALSE(hdr->save_jxl_to_buffer(true, 1.0f).is_empty());

	// Bright HDR values can overflow to infinity in half float.
	Ref<Image> overflow = make_gradient(32, 32, Image::FORMAT_RGBH);
	overflow->set_pixel(5, 5, Color(INFINITY, 1.0f, 0.5f));
	Ref<Image> overflow_decoded;
	overflow_decoded.instantiate();
	CHECK(overflow_decoded->load_jxl_from_buffer(overflow->save_jxl_to_buffer(true, 0.9f)) == OK);

	ERR_PRINT_OFF;
	CHECK(image->save_jxl_to_buffer(true, 1.5f).is_empty());
	ERR_PRINT_ON;
}

TEST_CASE("[JXL] Save to file") {
	Ref<Image> image = make_gradient(21, 13, Image::FORMAT_RGBA8);
	const String path = TestUtils::get_temp_path("saved.jxl");
	CHECK(image->save_jxl(path) == OK);

	Ref<Image> loaded = Image::load_from_file(path);
	REQUIRE(loaded.is_valid());
	CHECK(loaded->get_data() == image->get_data());

	const String texture_path = TestUtils::get_temp_path("saved_texture.jxl");
	CHECK(ResourceSaver::save(ImageTexture::create_from_image(image), texture_path) == OK);
	Ref<Image> texture_loaded = Image::load_from_file(texture_path);
	REQUIRE(texture_loaded.is_valid());
	CHECK(texture_loaded->get_data() == image->get_data());
}

TEST_CASE("[JXL] HDR CompressedTexture2D data") {
	Ref<Image> image = make_gradient(16, 8, Image::FORMAT_RGBAH);
	image->generate_mipmaps();

	// Same layout the texture importer writes.
	const String path = TestUtils::get_temp_path("hdr_texture.bin");
	{
		Ref<FileAccess> f = FileAccess::open(path, FileAccess::WRITE);
		REQUIRE(f.is_valid());
		f->store_32(CompressedTexture2D::DATA_FORMAT_JXL);
		f->store_16(image->get_width());
		f->store_16(image->get_height());
		f->store_32(image->get_mipmap_count());
		f->store_32(image->get_format());
		for (int i = 0; i < image->get_mipmap_count() + 1; i++) {
			const Vector<uint8_t> data = image->get_image_from_mipmap(i)->save_jxl_to_buffer();
			f->store_32(data.size());
			f->store_buffer(data);
		}
	}

	Ref<FileAccess> f = FileAccess::open(path, FileAccess::READ);
	REQUIRE(f.is_valid());
	Ref<Image> loaded = CompressedTexture2D::load_image_from_file(f);
	REQUIRE(loaded.is_valid());
	CHECK(loaded->get_format() == Image::FORMAT_RGBAH);
	CHECK(loaded->has_mipmaps());
	CHECK(loaded->get_data() == image->get_data());
}

} // namespace TestJXL
