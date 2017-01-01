/*************************************************************************/
/*  resource_format_xml.h                                                */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef RESOURCE_FORMAT_XML_H
#define RESOURCE_FORMAT_XML_H

#include "io/resource_loader.h"
#include "io/resource_saver.h"
#include "os/file_access.h"



class ResourceInteractiveLoaderXML : public ResourceInteractiveLoader {

	String local_path;
	String res_path;

	FileAccess *f;

	struct Tag {

		String name;
		HashMap<String,String> args;

	};

	_FORCE_INLINE_ Error _parse_array_element(Vector<char> &buff,bool p_number_only,FileAccess *f,bool *end);


	struct ExtResource {
		String path;
		String type;
	};


	Map<String,String> remaps;

	Map<int,ExtResource> ext_resources;

	int resources_total;
	int resource_current;
	String resource_type;

	mutable int lines;
	uint8_t get_char() const;
	int get_current_line() const;

friend class ResourceFormatLoaderXML;
	List<Tag> tag_stack;

	List<RES> resource_cache;
	Tag* parse_tag(bool* r_exit=NULL,bool p_printerr=true,List<String> *r_order=NULL);
	Error close_tag(const String& p_name);
	_FORCE_INLINE_ void unquote(String& p_str);
	Error goto_end_of_tag();
	Error parse_property_data(String &r_data);
	Error parse_property(Variant& r_v, String &r_name);

	Error error;

	RES resource;

public:

	virtual void set_local_path(const String& p_local_path);
	virtual Ref<Resource> get_resource();
	virtual Error poll();
	virtual int get_stage() const;
	virtual int get_stage_count() const;

	void open(FileAccess *p_f);
	String recognize(FileAccess *p_f);
	void get_dependencies(FileAccess *p_f, List<String> *p_dependencies, bool p_add_types);
	Error rename_dependencies(FileAccess *p_f, const String &p_path,const Map<String,String>& p_map);


	~ResourceInteractiveLoaderXML();

};

class ResourceFormatLoaderXML : public ResourceFormatLoader {
public:

	virtual Ref<ResourceInteractiveLoader> load_interactive(const String &p_path,Error *r_error=NULL);
	virtual void get_recognized_extensions_for_type(const String& p_type,List<String> *p_extensions) const;
	virtual void get_recognized_extensions(List<String> *p_extensions) const;
	virtual bool handles_type(const String& p_type) const;
	virtual String get_resource_type(const String &p_path) const;
	virtual void get_dependencies(const String& p_path, List<String> *p_dependencies, bool p_add_types=false);
	virtual Error rename_dependencies(const String &p_path,const Map<String,String>& p_map);


};


////////////////////////////////////////////////////////////////////////////////////////////


class ResourceFormatSaverXMLInstance  {

	String local_path;



	bool takeover_paths;
	bool relative_paths;
	bool bundle_resources;
	bool skip_editor;
	FileAccess *f;
	int depth;
	Set<RES> resource_set;
	List<RES> saved_resources;
	Map<RES,int> external_resources;

	void enter_tag(const char* p_tag,const String& p_args=String());
	void exit_tag(const char* p_tag);

	void _find_resources(const Variant& p_variant,bool p_main=false);
	void write_property(const String& p_name,const Variant& p_property,bool *r_ok=NULL);


	void escape(String& p_str);
	void write_tabs(int p_diff=0);
	void write_string(String p_str,bool p_escape=true);


public:

	Error save(const String &p_path,const RES& p_resource,uint32_t p_flags=0);


};

class ResourceFormatSaverXML : public ResourceFormatSaver {
public:
	static ResourceFormatSaverXML* singleton;
	virtual Error save(const String &p_path,const RES& p_resource,uint32_t p_flags=0);
	virtual bool recognize(const RES& p_resource) const;
	virtual void get_recognized_extensions(const RES& p_resource,List<String> *p_extensions) const;

	ResourceFormatSaverXML();
};


#endif // RESOURCE_FORMAT_XML_H
