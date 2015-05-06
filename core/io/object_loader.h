/*************************************************************************/
/*  object_loader.h                                                      */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef OBJECT_LOADER_H
#define OBJECT_LOADER_H

#include "object.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
#ifdef OLD_SCENE_FORMAT_ENABLED
class ObjectFormatLoader {
public:	
	
	virtual Error load(Object **p_object,Variant &p_meta)=0;

	virtual ~ObjectFormatLoader() {}
};

class ObjectFormatLoaderInstancer {
public:	

	virtual ObjectFormatLoader* instance(const String& p_file,const String& p_magic)=0;	
	virtual void get_recognized_extensions(List<String> *p_extensions) const=0;		
	bool recognize(const String& p_extension) const;
		
	virtual ~ObjectFormatLoaderInstancer() {}
};

class ObjectLoader {
	
	enum {
		MAX_LOADERS=64
	};
	
	static ObjectFormatLoaderInstancer *loader[MAX_LOADERS];
	static int loader_count;
	
public:
	
	static ObjectFormatLoader *instance_format_loader(const String& p_path,const String& p_magic,String p_force_extension="");
	static void add_object_format_loader_instancer(ObjectFormatLoaderInstancer *p_format_loader_instancer);
	static void get_recognized_extensions(List<String> *p_extensions);

		

};

#endif
#endif
