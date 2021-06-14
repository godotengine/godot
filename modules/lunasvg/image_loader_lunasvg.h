/*************************************************************************/
/*  image_loader_lunasvg.h                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef RESOURCE_IMPORTER_LUNASVG
#define RESOURCE_IMPORTER_LUNASVG

#include "core/error/error_macros.h"
#include "core/io/image_loader.h"
#include "core/string/ustring.h"

#include "core/templates/local_vector.h"
#include "thirdparty/lunasvg/include/document.h"
#include <stdint.h>
#include <iostream>
#include <memory>
#include <sstream>

class ImageLoaderLunaSVG : public ImageFormatLoader {
public:
	static void create_image_from_string(Ref<Image> p_image, String p_string, float p_scale, bool upsample, bool p_convert_color);
	virtual Error load_image(Ref<Image> p_image, FileAccess *p_fileaccess,
			bool p_force_linear, float p_scale) override;
	virtual void get_recognized_extensions(List<String> *p_extensions) const override;

public:
	virtual ~ImageLoaderLunaSVG() {}
};
#endif // RESOURCE_IMPORTER_LUNASVG
