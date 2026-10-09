/**************************************************************************/
/*  image_loader_jxl.cpp                                                  */
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

#include "image_loader_jxl.h"

#include <jxl/cms.h>
#include <jxl/decode.h>

static Error _jxl_decode(Image *p_image, JxlDecoder *p_dec, const uint8_t *p_buffer, size_t p_buffer_len, bool p_force_linear) {
	ERR_FAIL_COND_V(JxlDecoderSetCms(p_dec, *JxlGetDefaultCms()) != JXL_DEC_SUCCESS, ERR_CANT_OPEN);
	ERR_FAIL_COND_V(JxlDecoderSubscribeEvents(p_dec, JXL_DEC_BASIC_INFO | JXL_DEC_FULL_IMAGE) != JXL_DEC_SUCCESS, ERR_CANT_OPEN);
	ERR_FAIL_COND_V(JxlDecoderSetInput(p_dec, p_buffer, p_buffer_len) != JXL_DEC_SUCCESS, ERR_FILE_CORRUPT);
	JxlDecoderCloseInput(p_dec);

	JxlBasicInfo info;
	JxlPixelFormat format = {};
	Image::Format dest_format = Image::FORMAT_RGBA8;
	Vector<uint8_t> data;

	for (;;) {
		JxlDecoderStatus status = JxlDecoderProcessInput(p_dec);

		switch (status) {
			case JXL_DEC_ERROR:
				ERR_FAIL_V_MSG(ERR_FILE_CORRUPT, "JXL decoding failed.");
			case JXL_DEC_NEED_MORE_INPUT:
				ERR_FAIL_V_MSG(ERR_FILE_CORRUPT, "JXL file is truncated.");

			case JXL_DEC_BASIC_INFO: {
				ERR_FAIL_COND_V(JxlDecoderGetBasicInfo(p_dec, &info) != JXL_DEC_SUCCESS, ERR_FILE_CORRUPT);

				const uint32_t channels = info.num_color_channels + (info.alpha_bits > 0 ? 1 : 0);

				// Preserve the original precision: 8-bit integers stay 8 bits, 32-bit floats
				// stay 32 bits, and everything in between decodes to half floats.
				if (info.exponent_bits_per_sample > 0 && info.bits_per_sample > 16) {
					format.data_type = JXL_TYPE_FLOAT;
					const Image::Format formats[4] = { Image::FORMAT_RF, Image::FORMAT_RGF, Image::FORMAT_RGBF, Image::FORMAT_RGBAF };
					dest_format = formats[channels - 1];
				} else if (info.exponent_bits_per_sample > 0 || info.bits_per_sample > 8) {
					format.data_type = JXL_TYPE_FLOAT16;
					const Image::Format formats[4] = { Image::FORMAT_RH, Image::FORMAT_RGH, Image::FORMAT_RGBH, Image::FORMAT_RGBAH };
					dest_format = formats[channels - 1];
				} else {
					format.data_type = JXL_TYPE_UINT8;
					const Image::Format formats[4] = { Image::FORMAT_L8, Image::FORMAT_LA8, Image::FORMAT_RGB8, Image::FORMAT_RGBA8 };
					dest_format = formats[channels - 1];
				}

				format.num_channels = channels;
				format.endianness = JXL_NATIVE_ENDIAN;
				format.align = 0;
			} break;

			case JXL_DEC_NEED_IMAGE_OUT_BUFFER: {
				size_t buffer_size = 0;
				ERR_FAIL_COND_V(JxlDecoderImageOutBufferSize(p_dec, &format, &buffer_size) != JXL_DEC_SUCCESS, ERR_FILE_CORRUPT);
				ERR_FAIL_COND_V(data.resize(buffer_size) != OK, ERR_OUT_OF_MEMORY);
				ERR_FAIL_COND_V(JxlDecoderSetImageOutBuffer(p_dec, &format, data.ptrw(), buffer_size) != JXL_DEC_SUCCESS, ERR_FILE_CORRUPT);
			} break;

			case JXL_DEC_FULL_IMAGE: {
				// Only the first frame is imported, ignoring any animation.
				p_image->set_data(info.xsize, info.ysize, false, dest_format, data);

				if (p_force_linear) {
					// Treat the decoded samples as sRGB and convert them to linear, like the EXR loader.
					for (int y = 0; y < p_image->get_height(); y++) {
						for (int x = 0; x < p_image->get_width(); x++) {
							p_image->set_pixel(x, y, p_image->get_pixel(x, y).srgb_to_linear());
						}
					}
				}
				return OK;
			}

			default:
				ERR_FAIL_V_MSG(ERR_FILE_CORRUPT, "Unexpected JXL decoder status.");
		}
	}
}

Error ImageLoaderJXL::load_image(Ref<Image> p_image, Ref<FileAccess> f, BitField<ImageFormatLoader::LoaderFlags> p_flags, float p_scale) {
	Vector<uint8_t> src_image;
	uint64_t src_image_len = f->get_length();
	ERR_FAIL_COND_V(src_image_len == 0, ERR_FILE_CORRUPT);
	ERR_FAIL_COND_V(src_image.resize(src_image_len) != OK, ERR_OUT_OF_MEMORY);
	f->get_buffer(src_image.ptrw(), src_image_len);

	JxlDecoder *dec = JxlDecoderCreate(nullptr);
	ERR_FAIL_NULL_V(dec, ERR_CANT_CREATE);
	Error err = _jxl_decode(p_image.ptr(), dec, src_image.ptr(), src_image_len, p_flags & FLAG_FORCE_LINEAR);
	JxlDecoderDestroy(dec);

	return err;
}

void ImageLoaderJXL::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("jxl");
}
