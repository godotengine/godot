/*************************************************************************/
/*  register_types.cpp                                                   */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2019 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2019 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "register_types.h"
#include "audio_stream_music.h"
#include "resource_importer_music.h"

static Ref<ResourceFormatLoaderAudioStreamMusic> music_stream_loader;

void register_music_types() {

//#ifdef TOOLS_ENABLED
	//Ref<ResourceImporterOGGVorbis> ogg_import;
	//ogg_import.instance();
	//ResourceFormatImporter::get_singleton()->add_importer(ogg_import);
    
    Ref<ResourceImporterXM> import_xm;
	import_xm.instance();
	ResourceFormatImporter::get_singleton()->add_importer(import_xm);

    music_stream_loader.instance();
    ResourceLoader::add_resource_format_loader(music_stream_loader);

//#endif
	//ClassDB::register_class<AudioStreamOGGVorbis>();
    ClassDB::register_class<AudioStreamMusic>();
}

void unregister_music_types() {
	        music_stream_loader.unref();
}
