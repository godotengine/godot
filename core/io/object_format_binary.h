/*************************************************************************/
/*  object_format_binary.h                                               */
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
#ifndef OBJECT_FORMAT_BINARY_H
#define OBJECT_FORMAT_BINARY_H

#include "object_loader.h"
#include "object_saver_base.h"
#include "dvector.h"
#include "core/os/file_access.h"

#ifdef OLD_SCENE_FORMAT_ENABLED
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/


class ObjectFormatSaverBinary : public ObjectFormatSaver {

	String local_path;


	Ref<OptimizedSaver> optimizer;

	bool relative_paths;
	bool bundle_resources;
	bool skip_editor;
	bool big_endian;
	int bin_meta_idx;
	FileAccess *f;
	String magic;
	Map<RES,int> resource_map;
	Map<StringName,int> string_map;
	Vector<StringName> strings;

	struct SavedObject {

		Variant meta;
		String type;


		struct SavedProperty {

			int name_idx;
			Variant value;
		};

		List<SavedProperty> properties;
	};


	int get_string_index(const String& p_string);
	void save_unicode_string(const String& p_string);

	List<SavedObject*> saved_objects;
	List<SavedObject*> saved_resources;

	void _pad_buffer(int p_bytes);
	Error _save_obj(const Object *p_object,SavedObject *so);
	void _find_resources(const Variant& p_variant);
	void write_property(int p_idx,const Variant& p_property);


public:

	virtual Error save(const Object *p_object,const Variant &p_meta);

	ObjectFormatSaverBinary(FileAccess *p_file,const String& p_magic,const String& p_local_path,uint32_t p_flags,const Ref<OptimizedSaver>& p_optimizer);
	~ObjectFormatSaverBinary();
};

class ObjectFormatSaverInstancerBinary : public ObjectFormatSaverInstancer {
public:

	virtual ObjectFormatSaver* instance(const String& p_file,const String& p_magic,uint32_t p_flags=0,const Ref<OptimizedSaver>& p_optimizer=Ref<OptimizedSaver>());
	virtual void get_recognized_extensions(List<String> *p_extensions) const;

	virtual ~ObjectFormatSaverInstancerBinary();
};




/***********************************/
/***********************************/
/***********************************/
/***********************************/

class ObjectFormatLoaderBinary : public ObjectFormatLoader {

	String local_path;

	FileAccess *f;

	bool endian_swap;
	bool use_real64;

	Vector<char> str_buf;
	List<RES> resource_cache;

	Map<int,StringName> string_map;

	String get_unicode_string();
	void _advance_padding(uint32_t p_len);

friend class ObjectFormatLoaderInstancerBinary;


	Error parse_property(Variant& r_v, int& r_index);

public:


	virtual Error load(Object **p_object,Variant &p_meta);

	ObjectFormatLoaderBinary(FileAccess *f,bool p_endian_swap,bool p_use64);
	virtual ~ObjectFormatLoaderBinary();
};

class ObjectFormatLoaderInstancerBinary : public ObjectFormatLoaderInstancer {
public:

	virtual ObjectFormatLoaderBinary* instance(const String& p_file,const String& p_magic);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;



};



#endif // OBJECT_FORMAT_BINARY_H
#endif
