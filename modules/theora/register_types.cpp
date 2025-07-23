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

#include "video_stream_theora.h"

#ifdef TOOLS_ENABLED
#include "core/config/project_settings.h"
#include "editor/editor_node.h"
#include "editor/movie_writer_ogv.h"
#include "editor/resource_importer_video.h"
#endif

static Ref<ResourceFormatLoaderTheora> resource_loader_theora;
#ifdef TOOLS_ENABLED
static MovieWriterOGV *writer_ogv = nullptr;
#endif

#ifdef TOOLS_ENABLED
static void _editor_init() {
	Ref<ResourceImporterVideo> video_importer;
	video_importer.instantiate();
	ResourceFormatImporter::get_singleton()->add_importer(video_importer);
}
#endif

void initialize_theora_module(ModuleInitializationLevel p_level) {
	switch (p_level) {
		case MODULE_INITIALIZATION_LEVEL_SERVERS: {
#ifdef TOOLS_ENABLED
			if (GD_IS_CLASS_ENABLED(MovieWriterOGV)) {
				writer_ogv = memnew(MovieWriterOGV);
				MovieWriter::add_writer(writer_ogv);
			}
#endif
		} break;

		case MODULE_INITIALIZATION_LEVEL_SCENE: {
			resource_loader_theora.instantiate();
			ResourceLoader::add_resource_format_loader(resource_loader_theora, true);
			GDREGISTER_CLASS(VideoStreamTheora);
		} break;
		default:
			break;
	}

#ifdef TOOLS_ENABLED
	if (p_level == MODULE_INITIALIZATION_LEVEL_EDITOR) {
		GLOBAL_DEF_RST_BASIC("filesystem/import/ffmpeg/enabled", true);
		if (GLOBAL_GET_CACHED(bool, "filesystem/import/ffmpeg/enabled")) {
			GDREGISTER_CLASS(ResourceImporterVideo);
			EditorNode::add_init_callback(_editor_init);
		}
	}
#endif
}

void uninitialize_theora_module(ModuleInitializationLevel p_level) {
	switch (p_level) {
		case MODULE_INITIALIZATION_LEVEL_SCENE: {
			ResourceLoader::remove_resource_format_loader(resource_loader_theora);
			resource_loader_theora.unref();
		} break;

		case MODULE_INITIALIZATION_LEVEL_SERVERS: {
#ifdef TOOLS_ENABLED
			if (GD_IS_CLASS_ENABLED(MovieWriterOGV)) {
				memdelete(writer_ogv);
			}
#endif
		} break;
		default:
			break;
	}
}
