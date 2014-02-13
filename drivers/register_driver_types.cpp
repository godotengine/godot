/*************************************************/
/*  register_driver_types.cpp                    */
/*************************************************/
/*            This file is part of:              */
/*                GODOT ENGINE                   */
/*************************************************/
/*       Source code within this file is:        */
/*  (c) 2007-2010 Juan Linietsky, Ariel Manzur   */
/*             All Rights Reserved.              */
/*************************************************/

#include "register_driver_types.h"

#include "drivers/png/image_loader_png.h"
#include "drivers/webp/image_loader_webp.h"
#include "drivers/png/resource_saver_png.h"
#include "drivers/jpg/image_loader_jpg.h"
#include "drivers/dds/texture_loader_dds.h"
#include "drivers/pvr/texture_loader_pvr.h"
#include "drivers/etc1/image_etc.h"
#include "drivers/chibi/event_stream_chibi.h"

#ifdef TOOLS_ENABLED
#ifdef SQUISH_ENABLED
#include "drivers/squish/image_compress_squish.h"
#endif
#endif

#ifdef TOOLS_ENABLED
#include "drivers/convex_decomp/b2d_decompose.h"
#endif

#ifdef TREMOR_ENABLED
#include "drivers/teora/audio_stream_ogg.h"
#endif

#ifdef VORBIS_ENABLED
#include "drivers/vorbis/audio_stream_ogg_vorbis.h"
#endif


#ifdef SPEEX_ENABLED
#include "drivers/speex/audio_stream_speex.h"
#endif

#ifdef THEORA_ENABLED
#include "drivers/theora/video_stream_theora.h"
#endif

#include "drivers/trex/regex.h"

#ifdef MUSEPACK_ENABLED
#include "drivers/mpc/audio_stream_mpc.h"
#endif

#ifdef PNG_ENABLED
static ImageLoaderPNG *image_loader_png=NULL;
static ResourceSaverPNG *resource_saver_png=NULL;
#endif

#ifdef WEBP_ENABLED
static ImageLoaderWEBP *image_loader_webp=NULL;
//static ResourceSaverPNG *resource_saver_png=NULL;
#endif

#ifdef JPG_ENABLED
static ImageLoaderJPG *image_loader_jpg=NULL;
#endif

#ifdef DDS_ENABLED
static ResourceFormatDDS *resource_loader_dds=NULL;
#endif


#ifdef PVR_ENABLED
static ResourceFormatPVR *resource_loader_pvr=NULL;
#endif

#ifdef TREMOR_ENABLED
static ResourceFormatLoaderAudioStreamOGG *vorbis_stream_loader=NULL;
#endif

#ifdef VORBIS_ENABLED
static ResourceFormatLoaderAudioStreamOGGVorbis *vorbis_stream_loader=NULL;
#endif

#ifdef SPEEX_ENABLED
static ResourceFormatLoaderAudioStreamSpeex *speex_stream_loader=NULL;
#endif

#ifdef THEORA_ENABLED
static ResourceFormatLoaderVideoStreamTheora* theora_stream_loader = NULL;
#endif

#ifdef MUSEPACK_ENABLED
static ResourceFormatLoaderAudioStreamMPC * mpc_stream_loader=NULL;
#endif

void register_core_driver_types() {

#ifdef PNG_ENABLED
	image_loader_png = memnew( ImageLoaderPNG );
	ImageLoader::add_image_format_loader( image_loader_png );

	resource_saver_png = memnew( ResourceSaverPNG );
	ResourceSaver::add_resource_format_saver(resource_saver_png);

#endif

#ifdef WEBP_ENABLED
	image_loader_webp = memnew( ImageLoaderWEBP );
	ImageLoader::add_image_format_loader( image_loader_webp );

//	resource_saver_png = memnew( ResourceSaverPNG );
//	ResourceSaver::add_resource_format_saver(resource_saver_png);

#endif

#ifdef JPG_ENABLED

	image_loader_jpg = memnew( ImageLoaderJPG );
	ImageLoader::add_image_format_loader( image_loader_jpg );
#endif

	ObjectTypeDB::register_type<RegEx>();
}

void unregister_core_driver_types() {

#ifdef PNG_ENABLED
	if (image_loader_png)
		memdelete( image_loader_png );
	if (resource_saver_png)
		memdelete( resource_saver_png );
#endif

#ifdef WEBP_ENABLED
	if (image_loader_webp)
		memdelete( image_loader_webp );
//	if (resource_saver_png)
//		memdelete( resource_saver_png );
#endif

#ifdef JPG_ENABLED
	if (image_loader_jpg)
		memdelete( image_loader_jpg );
#endif

}


void register_driver_types() {

#ifdef TREMOR_ENABLED
	vorbis_stream_loader=memnew( ResourceFormatLoaderAudioStreamOGG );
	ResourceLoader::add_resource_format_loader(vorbis_stream_loader );
	ObjectTypeDB::register_type<AudioStreamOGG>();
#endif

#ifdef VORBIS_ENABLED
	vorbis_stream_loader=memnew( ResourceFormatLoaderAudioStreamOGGVorbis );
	ResourceLoader::add_resource_format_loader(vorbis_stream_loader );
	ObjectTypeDB::register_type<AudioStreamOGGVorbis>();
#endif


#ifdef DDS_ENABLED
	resource_loader_dds = memnew( ResourceFormatDDS );
	ResourceLoader::add_resource_format_loader(resource_loader_dds );
#endif

#ifdef PVR_ENABLED
	resource_loader_pvr = memnew( ResourceFormatPVR );
	ResourceLoader::add_resource_format_loader(resource_loader_pvr );
#endif

#ifdef TOOLS_ENABLED

	Geometry::_decompose_func=b2d_decompose;
#endif

#ifdef SPEEX_ENABLED
	speex_stream_loader=memnew( ResourceFormatLoaderAudioStreamSpeex );
	ResourceLoader::add_resource_format_loader(speex_stream_loader);
	ObjectTypeDB::register_type<AudioStreamSpeex>();
#endif

#ifdef MUSEPACK_ENABLED

	mpc_stream_loader=memnew( ResourceFormatLoaderAudioStreamMPC );
	ResourceLoader::add_resource_format_loader(mpc_stream_loader);
	ObjectTypeDB::register_type<AudioStreamMPC>();

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

	_register_etc1_compress_func();
	initialize_chibi();
}

void unregister_driver_types() {

#ifdef TREMOR_ENABLED
	memdelete( vorbis_stream_loader );
#endif

#ifdef VORBIS_ENABLED
	memdelete( vorbis_stream_loader );
#endif

#ifdef SPEEX_ENABLED
	memdelete( speex_stream_loader );
#endif

#ifdef THEORA_ENABLED

	memdelete (theora_stream_loader);
#endif

#ifdef MUSEPACK_ENABLED

	memdelete (mpc_stream_loader);
#endif

#ifdef DDS_ENABLED
	memdelete(resource_loader_dds);
#endif

#ifdef PVR_ENABLED
	memdelete(resource_loader_pvr);
#endif

	finalize_chibi();
}
