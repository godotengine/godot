/*************************************************************************/
/*  image_loader_ies.cpp                                                 */
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

#include "image_loader_ies.h"

#include "core/io/file_access.h"
#include "core/string/ustring.h"

#include "thirdparty/ies/ies_loader.h"
#include <ios>

Error ImageLoaderIES::load_image(Ref<Image> p_image, Ref<FileAccess> f, bool p_force_linear, float p_scale) {
	IESFileInfo info = {};
	std::string ies_file = f->get_as_utf8_string(true).utf8().get_data();

	IESLoadHelper IESLoader;
	if (!IESLoader.load(ies_file, info)) {
		ERR_PRINT(vformat("IES image loader error: %s", info.error().c_str()));
		return FAILED;
	}
	int32_t width = 512;
	width *= p_scale;
	int32_t height = 1;
	int32_t channel_count = 3;
	Vector<uint8_t> imgdata;
	imgdata.resize(width * height * channel_count * sizeof(float));
	if (!IESLoader.saveAs1D(info, (float *)imgdata.ptrw(), width, channel_count)) {
		ERR_PRINT(vformat("IES save error: %s", info.error().c_str()));
		return FAILED;
	}
	p_image->create(width, height, false, Image::FORMAT_RGBF, imgdata);
	p_image->generate_mipmaps();

	return OK;
}

void ImageLoaderIES::get_recognized_extensions(List<String> *p_extensions) const {
	p_extensions->push_back("ies");
}

ImageLoaderIES::ImageLoaderIES() {
}
