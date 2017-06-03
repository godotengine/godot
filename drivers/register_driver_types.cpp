/*************************************************************************/
/*  register_driver_types.cpp                                            */
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
#include "register_driver_types.h"

#include "core/math/geometry.h"
#include "png/image_loader_png.h"
#include "png/resource_saver_png.h"

#ifdef TOOLS_ENABLED
#include "convex_decomp/b2d_decompose.h"
#endif

#ifdef TOOLS_ENABLED
#include "platform/windows/export/export.h"
#endif

static ImageLoaderPNG *image_loader_png = NULL;
static ResourceSaverPNG *resource_saver_png = NULL;

void register_core_driver_types() {

	image_loader_png = memnew(ImageLoaderPNG);
	ImageLoader::add_image_format_loader(image_loader_png);

	resource_saver_png = memnew(ResourceSaverPNG);
	ResourceSaver::add_resource_format_saver(resource_saver_png);
}

void unregister_core_driver_types() {

	if (image_loader_png)
		memdelete(image_loader_png);
	if (resource_saver_png)
		memdelete(resource_saver_png);
}

void register_driver_types() {

#ifdef TOOLS_ENABLED
	Geometry::_decompose_func = b2d_decompose;
#endif
}

void unregister_driver_types() {
}
