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

#include "image_loader_svg.h"

#include <thorvg.h>

#ifdef THREADS_ENABLED
#define TVG_THREADS 1
#else
#define TVG_THREADS 0
#endif

#ifdef LOTTIE_ENABLED
#include "lottie_texture.h"

static Ref<ResourceFormatLoaderLottie> resource_loader_lottie;
static Ref<ResourceFormatSaverLottie> resource_saver_lottie;

#ifdef TOOLS_ENABLED
#include "editor/resource_importer_lottie.h"

static Ref<ResourceImporterLottieCTEX> resource_importer_lottie_ctex;
static Ref<ResourceImporterLottieJSON> resource_importer_lottie_json;
#endif // TOOLS_ENABLED
#endif // LOTTIE_ENABLED

static Ref<ImageLoaderSVG> image_loader_svg;

void initialize_svg_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	tvg::CanvasEngine tvgEngine = tvg::CanvasEngine::Sw;

	if (tvg::Initializer::init(tvgEngine, TVG_THREADS) != tvg::Result::Success) {
		return;
	}

	image_loader_svg.instantiate();
	ImageLoader::add_image_format_loader(image_loader_svg);

#ifdef LOTTIE_ENABLED
	resource_loader_lottie.instantiate();
	resource_saver_lottie.instantiate();

	// Lottie loader should be at the front of JSON loader.
	ResourceLoader::add_resource_format_loader(resource_loader_lottie, true);
	ResourceSaver::add_resource_format_saver(resource_saver_lottie);
	ClassDB::register_class<LottieTexture2D>();

#ifdef TOOLS_ENABLED
	resource_importer_lottie_ctex.instantiate();
	resource_importer_lottie_json.instantiate();
	ResourceFormatImporter::get_singleton()->add_importer(resource_importer_lottie_ctex);
	ResourceFormatImporter::get_singleton()->add_importer(resource_importer_lottie_json);
#endif // TOOLS_ENABLED
#endif // LOTTIE_ENABLED
}

void uninitialize_svg_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}

	if (image_loader_svg.is_null()) {
		// It failed to initialize so it was not added.
		return;
	}

	ImageLoader::remove_image_format_loader(image_loader_svg);
	image_loader_svg.unref();

#ifdef LOTTIE_ENABLED
	ResourceLoader::remove_resource_format_loader(resource_loader_lottie);
	ResourceSaver::remove_resource_format_saver(resource_saver_lottie);
	resource_loader_lottie.unref();
	resource_saver_lottie.unref();

#ifdef TOOLS_ENABLED
	ResourceFormatImporter::get_singleton()->remove_importer(resource_importer_lottie_ctex);
	ResourceFormatImporter::get_singleton()->remove_importer(resource_importer_lottie_json);
	resource_importer_lottie_ctex.unref();
	resource_importer_lottie_json.unref();
#endif // TOOLS_ENABLED
#endif // LOTTIE_ENABLED
	tvg::Initializer::term(tvg::CanvasEngine::Sw);
}
