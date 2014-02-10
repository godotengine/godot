/*************************************************/
/*  image_loader_webp.h                          */
/*************************************************/
/*            This file is part of:              */
/*                GODOT ENGINE                   */
/*************************************************/
/*       Source code within this file is:        */
/*  (c) 2007-2010 Juan Linietsky, Ariel Manzur   */
/*             All Rights Reserved.              */
/*************************************************/

#ifndef IMAGE_LOADER_WEBP_H
#define IMAGE_LOADER_WEBP_H

#include "io/image_loader.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class ImageLoaderWEBP : public ImageFormatLoader {


public:

	virtual Error load_image(Image *p_image,FileAccess *f);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;	
	ImageLoaderWEBP();
};



#endif
