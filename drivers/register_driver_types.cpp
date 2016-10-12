/*************************************************************************/
/*  register_driver_types.cpp                                            */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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

#include "png/image_loader_png.h"
#include "png/resource_saver_png.h"
#include "chibi/event_stream_chibi.h"


#ifdef TOOLS_ENABLED
#include "squish/image_compress_squish.h"
#endif

#ifdef TOOLS_ENABLED
#include "convex_decomp/b2d_decompose.h"
#endif

#ifdef TOOLS_ENABLED
#include "platform/windows/export/export.h"
#endif

#ifdef THEORA_ENABLED
#include "theora/video_stream_theora.h"
#endif


#include "drivers/nrex/regex.h"

#ifdef MUSEPACK_ENABLED
#include "mpc/audio_stream_mpc.h"
#endif

static ImageLoaderPNG *image_loader_png=NULL;
static ResourceSaverPNG *resource_saver_png=NULL;

#ifdef THEORA_ENABLED
static ResourceFormatLoaderVideoStreamTheora* theora_stream_loader = NULL;
#endif

#ifdef MUSEPACK_ENABLED
static ResourceFormatLoaderAudioStreamMPC * mpc_stream_loader=NULL;
#endif

#ifdef OPENSSL_ENABLED
#include "openssl/register_openssl.h"
#endif


void register_core_driver_types() {

	image_loader_png = memnew( ImageLoaderPNG );
	ImageLoader::add_image_format_loader( image_loader_png );

	resource_saver_png = memnew( ResourceSaverPNG );
	ResourceSaver::add_resource_format_saver(resource_saver_png);

	ObjectTypeDB::register_type<RegEx>();
}

void unregister_core_driver_types() {

	if (image_loader_png)
		memdelete( image_loader_png );
	if (resource_saver_png)
		memdelete( resource_saver_png );
}


void register_driver_types() {

#ifdef TOOLS_ENABLED

	Geometry::_decompose_func=b2d_decompose;
#endif

#ifdef MUSEPACK_ENABLED

	mpc_stream_loader=memnew( ResourceFormatLoaderAudioStreamMPC );
	ResourceLoader::add_resource_format_loader(mpc_stream_loader);
	ObjectTypeDB::register_type<AudioStreamMPC>();

#endif

#ifdef OPENSSL_ENABLED

	register_openssl();
#endif

#ifdef THEORA_ENABLED
	theora_stream_loader = memnew( ResourceFormatLoaderVideoStreamTheora );
	ResourceLoader::add_resource_format_loader(theora_stream_loader);
	ObjectTypeDB::register_type<VideoStreamTheora>();
#endif


#ifdef TOOLS_ENABLED
#ifdef SQUISH_ENABLED

	Image::set_compress_bc_func(image_compress_squish);

#endif
#endif

	initialize_chibi();
}

void unregister_driver_types() {

#ifdef THEORA_ENABLED
	memdelete (theora_stream_loader);
#endif


#ifdef MUSEPACK_ENABLED

	memdelete (mpc_stream_loader);
#endif

#ifdef OPENSSL_ENABLED

	unregister_openssl();
#endif

	finalize_chibi();
}
