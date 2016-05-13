/*************************************************/
/*  image_loader_jpg.h                           */
/*************************************************/
/*            This file is part of:              */
/*                GODOT ENGINE                   */
/*************************************************/
/*       Source code within this file is:        */
/*  (c) 2007-2016 Juan Linietsky, Ariel Manzur   */
/*             All Rights Reserved.              */
/*************************************************/

#ifndef BITMAP_LOADER_PNM_H
#define BITMAP_LOADER_PNM_H

#include "io/resource_loader.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class ResourceFormatPBM : public ResourceFormatLoader {


public:

	virtual RES load(const String &p_path,const String& p_original_path="",Error *r_error=NULL);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String& p_type) const;
	virtual String get_resource_type(const String &p_path) const;
};



#endif
