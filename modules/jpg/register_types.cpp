/**************************************************************************/
/*  register_types.cpp                                                    */
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

#include "register_types.h"

#include "image_loader_libjpeg_turbo.h"
#include "movie_writer_mjpeg.h"

static Ref<ImageLoaderLibJPEGTurbo> image_loader_libjpeg_turbo;
static MovieWriterMJPEG *writer_mjpeg = nullptr;

void initialize_jpg_module(ModuleInitializationLevel p_level) {
	switch (p_level) {
		case MODULE_INITIALIZATION_LEVEL_SERVERS: {
			if constexpr (GD_IS_CLASS_ENABLED(MovieWriterMJPEG)) {
				writer_mjpeg = memnew(MovieWriterMJPEG);
				MovieWriter::add_writer(writer_mjpeg);
			}
		} break;

		case MODULE_INITIALIZATION_LEVEL_SCENE: {
			image_loader_libjpeg_turbo.instantiate();
			ImageLoader::add_image_format_loader(image_loader_libjpeg_turbo);
		} break;

		default:
			break;
	}
}

void uninitialize_jpg_module(ModuleInitializationLevel p_level) {
	switch (p_level) {
		case MODULE_INITIALIZATION_LEVEL_SCENE: {
			ImageLoader::remove_image_format_loader(image_loader_libjpeg_turbo);
			image_loader_libjpeg_turbo.unref();
		} break;

		case MODULE_INITIALIZATION_LEVEL_SERVERS: {
			if constexpr (GD_IS_CLASS_ENABLED(MovieWriterMJPEG)) {
				memdelete(writer_mjpeg);
			}
		} break;

		default:
			break;
	}
}
