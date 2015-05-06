/*************************************************************************/
/*  object_format_xml.h                                                  */
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
#ifndef OBJECT_FORMAT_XML_H
#define OBJECT_FORMAT_XML_H

#ifdef XML_ENABLED
#ifdef OLD_SCENE_FORMAT_ENABLED
#include "io/object_loader.h"
#include "io/object_saver.h"
#include "os/file_access.h"
#include "map.h"
#include "resource.h"
#include "xml_parser.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class ObjectFormatSaverXML : public ObjectFormatSaver {
	
	String local_path;


	Ref<OptimizedSaver> optimizer;

	bool relative_paths;
	bool bundle_resources;
	bool skip_editor;
	FileAccess *f;
	String magic;
	int depth;
	Map<RES,int> resource_map;

	struct SavedObject {

		Variant meta;
		String type;


		struct SavedProperty {

			String name;
			Variant value;
		};

		List<SavedProperty> properties;
	};

	List<RES> saved_resources;

	List<SavedObject*> saved_objects;
	
	void enter_tag(const String& p_section,const String& p_args="");
	void exit_tag(const String& p_section);
	
	void _find_resources(const Variant& p_variant);
	void write_property(const String& p_name,const Variant& p_property,bool *r_ok=NULL);

	
	void escape(String& p_str);
	void write_tabs(int p_diff=0);
	void write_string(String p_str,bool p_escape=true);
		
public:	
	
	virtual Error save(const Object *p_object,const Variant &p_meta);
	
	ObjectFormatSaverXML(FileAccess *p_file,const String& p_magic,const String& p_local_path,uint32_t p_flags,const Ref<OptimizedSaver>& p_optimizer);
	~ObjectFormatSaverXML();
};

class ObjectFormatSaverInstancerXML : public ObjectFormatSaverInstancer {
public:	 

	virtual ObjectFormatSaver* instance(const String& p_file,const String& p_magic,uint32_t p_flags=0,const Ref<OptimizedSaver>& p_optimizer=Ref<OptimizedSaver>());
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	
	virtual ~ObjectFormatSaverInstancerXML();
};

/***********************************/
/***********************************/
/***********************************/
/***********************************/

//#define OPTIMIZED_XML_LOADER

#ifdef OPTIMIZED_XML_LOADER

class ObjectFormatLoaderXML : public ObjectFormatLoader {

	Ref<XMLParser> parser;
	String local_path;

	Error _close_tag(const String& p_tag);
	Error _parse_property(Variant& r_property,String& r_name);

friend class ObjectFormatLoaderInstancerXML;

	List<RES> resource_cache;
public:


	virtual Error load(Object **p_object,Variant &p_meta);


};

class ObjectFormatLoaderInstancerXML : public ObjectFormatLoaderInstancer {
public:

	virtual ObjectFormatLoaderXML* instance(const String& p_file,const String& p_magic);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;

};

#else


class ObjectFormatLoaderXML : public ObjectFormatLoader {

	String local_path;

	FileAccess *f;

	struct Tag {

		String name;
		HashMap<String,String> args;
	};

	_FORCE_INLINE_ Error _parse_array_element(Vector<char> &buff,bool p_number_only,FileAccess *f,bool *end);

	mutable int lines;
	uint8_t get_char() const;
	int get_current_line() const;

friend class ObjectFormatLoaderInstancerXML;
	List<Tag> tag_stack;

	List<RES> resource_cache;
	Tag* parse_tag(bool* r_exit=NULL);
	Error close_tag(const String& p_name);
	void unquote(String& p_str);
	Error goto_end_of_tag();
	Error parse_property_data(String &r_data);
	Error parse_property(Variant& r_v, String &r_name);

public:


	virtual Error load(Object **p_object,Variant &p_meta);

	virtual ~ObjectFormatLoaderXML();
};

class ObjectFormatLoaderInstancerXML : public ObjectFormatLoaderInstancer {
public:

	virtual ObjectFormatLoaderXML* instance(const String& p_file,const String& p_magic);
	virtual void get_recognized_extensions(List<String> *p_extensions) const;



};

#endif
#endif
#endif
#endif
