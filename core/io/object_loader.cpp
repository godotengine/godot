/*************************************************************************/
/*  object_loader.cpp                                                    */
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
#include "object_loader.h"

#ifdef OLD_SCENE_FORMAT_ENABLED

bool ObjectFormatLoaderInstancer::recognize(const String& p_extension) const {
	
	
	List<String> extensions;
	get_recognized_extensions(&extensions);
	for (List<String>::Element *E=extensions.front();E;E=E->next()) {

		if (E->get().nocasecmp_to(p_extension)==0)
			return true;
	}
	
	return false;
}

ObjectFormatLoaderInstancer *ObjectLoader::loader[MAX_LOADERS];
int ObjectLoader::loader_count=0;


ObjectFormatLoader *ObjectLoader::instance_format_loader(const String& p_path,const String& p_magic,String p_force_extension) {
	
	String extension=p_force_extension.length()?p_force_extension:p_path.extension();

	for (int i=0;i<loader_count;i++) {
		
		if (!loader[i]->recognize(extension))
			continue;
		ObjectFormatLoader *format_loader = loader[i]->instance(p_path,p_magic);
		if (format_loader)
			return format_loader;
	}
	
	return NULL;
}

void ObjectLoader::get_recognized_extensions(List<String> *p_extensions)  {
	
	for (int i=0;i<loader_count;i++) {
		
		loader[i]->get_recognized_extensions(p_extensions);
	}	
}



void ObjectLoader::add_object_format_loader_instancer(ObjectFormatLoaderInstancer *p_format_loader_instancer) {
	
	ERR_FAIL_COND(loader_count>=MAX_LOADERS );
	loader[loader_count++]=p_format_loader_instancer;	
}


#endif
