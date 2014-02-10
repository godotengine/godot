/*************************************************************************/
/*  object_saver.cpp                                                     */
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
#include "object_saver.h"
#ifdef OLD_SCENE_FORMAT_ENABLED

void OptimizedSaver::add_property(const StringName& p_name, const Variant& p_value) {

	ERR_FAIL_COND(!_list);
	Property p;
	p.name=p_name;
	p.value=p_value;
	_list->push_back(p);
}

bool OptimizedSaver::optimize_object(const Object *p_object) {

	return false; //not optimize
}

void OptimizedSaver::get_property_list(const Object* p_object,List<Property>  *p_properties) {


	_list=p_properties;

	bool res = call("optimize_object",p_object);

	if (!res) {

		List<PropertyInfo> plist;
		p_object->get_property_list(&plist);
		for(List<PropertyInfo>::Element *E=plist.front();E;E=E->next()) {

			PropertyInfo pinfo=E->get();
			if ((pinfo.usage&PROPERTY_USAGE_STORAGE) || (is_bundle_resources_enabled() && pinfo.usage&PROPERTY_USAGE_BUNDLE)) {

				add_property(pinfo.name,p_object->get(pinfo.name));
			}
		}

	}

	_list=NULL;
}

void OptimizedSaver::set_target_platform(const String& p_platform) {

	ERR_FAIL_COND(p_platform!="" && !p_platform.is_valid_identifier());
	platform=p_platform;
}

String OptimizedSaver::get_target_platform() const {

	return platform;
}

void OptimizedSaver::set_target_name(const String& p_name) {

	name=p_name;
}

String OptimizedSaver::get_target_name() const {

	return name;
}

void OptimizedSaver::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_target_platform","name"),&OptimizedSaver::set_target_platform);
	ObjectTypeDB::bind_method(_MD("get_target_platform"),&OptimizedSaver::get_target_platform);
	ObjectTypeDB::bind_method(_MD("set_target_name","name"),&OptimizedSaver::set_target_name);
	ObjectTypeDB::bind_method(_MD("add_property","name","value"),&OptimizedSaver::add_property);
	ObjectTypeDB::bind_method(_MD("optimize_object","obj"),&OptimizedSaver::optimize_object);
}

OptimizedSaver::OptimizedSaver() {

	_list=NULL;
}

ObjectFormatSaverInstancer *ObjectSaver::saver[MAX_LOADERS];
int ObjectSaver::saver_count=0;

bool ObjectFormatSaverInstancer::recognize(const String& p_extension) const {
	
	
	List<String> extensions;
	get_recognized_extensions(&extensions);
	for (List<String>::Element *E=extensions.front();E;E=E->next()) {
		
		if (E->get().nocasecmp_to(p_extension.extension())==0)
			return true;
	}
	
	return false;
}

ObjectFormatSaver *ObjectSaver::instance_format_saver(const String& p_path,const String& p_magic,String p_force_extension,uint32_t p_flags,const Ref<OptimizedSaver>& p_optimizer) {
	
	String extension=p_force_extension.length()?p_force_extension:p_path.extension();
	
	for (int i=0;i<saver_count;i++) {
		
		if (!saver[i]->recognize(extension))
			continue;
		ObjectFormatSaver *format_saver = saver[i]->instance(p_path,p_magic,p_flags,p_optimizer);
		if (format_saver)
			return format_saver;
	}
	
	return NULL;
}

void ObjectSaver::get_recognized_extensions(List<String> *p_extensions)  {
	
	for (int i=0;i<saver_count;i++) {
		
		saver[i]->get_recognized_extensions(p_extensions);
	}	
}



void ObjectSaver::add_object_format_saver_instancer(ObjectFormatSaverInstancer *p_format_saver_instancer) {
	
	ERR_FAIL_COND(saver_count>=MAX_LOADERS );
	saver[saver_count++]=p_format_saver_instancer;	
}



#endif
