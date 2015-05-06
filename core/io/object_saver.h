/*************************************************************************/
/*  object_saver.h                                                       */
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
#ifndef OBJECT_SAVER_H
#define OBJECT_SAVER_H

#include "object.h"
#include "resource.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

#ifdef OLD_SCENE_FORMAT_ENABLED

class OptimizedSaver : public Reference {

	OBJ_TYPE(OptimizedSaver,Reference);
public:

	struct Property {

		StringName name;
		Variant value;
	};

private:

	String name;
	String platform;
	List<Property>  *_list;
protected:


	void set_target_platform(const String& p_platform);
	void set_target_name(const String& p_name);
	void add_property(const StringName& p_name, const Variant& p_value);
	static void _bind_methods();

	virtual bool optimize_object(const Object *p_object);

public:


	virtual bool is_bundle_resources_enabled() const { return false; }

	String get_target_platform() const;
	String get_target_name() const;
	void get_property_list(const Object* p_object, List<Property>  *p_properties);


	OptimizedSaver();

};


class ObjectFormatSaver {
public:	
	
	virtual Error save(const Object *p_object,const Variant &p_meta=Variant())=0;
	
	virtual ~ObjectFormatSaver() {}
};

class ObjectFormatSaverInstancer {
public:	

	virtual void get_recognized_extensions(List<String> *p_extensions) const=0;
	virtual ObjectFormatSaver* instance(const String& p_file,const String& p_magic="",uint32_t p_flags=0,const Ref<OptimizedSaver>& p_optimizer=Ref<OptimizedSaver>())=0;
	bool recognize(const String& p_extension) const;
		
	virtual ~ObjectFormatSaverInstancer() {}
};

class ObjectSaver {
	
	enum {
		MAX_LOADERS=64
	};
	
	static ObjectFormatSaverInstancer *saver[MAX_LOADERS];
	static int saver_count;
	
public:

	enum SaverFlags {

		FLAG_RELATIVE_PATHS=1,
		FLAG_BUNDLE_RESOURCES=2,
		FLAG_OMIT_EDITOR_PROPERTIES=4,
		FLAG_SAVE_BIG_ENDIAN=8
	};

	
	static ObjectFormatSaver *instance_format_saver(const String& p_path,const String& p_magic,String p_force_extension="",uint32_t p_flags=0,const Ref<OptimizedSaver>& p_optimizer=Ref<OptimizedSaver>());
	static void get_recognized_extensions(List<String> *p_extensions);
	
	static void add_object_format_saver_instancer(ObjectFormatSaverInstancer *p_format_saver_instancer);
	
	
};

#endif
#endif
