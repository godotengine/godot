/*************************************************************************/
/*  raw_text.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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

#include "raw_text.h"
#include "core/object_type_db.h"
#include "core/os/file_access.h"

void RawText::set_text(const String& text) {
	if(text.length() != m_text.length() && text!=m_text) {
		m_text = text;
		emit_changed();
	}
}

void RawText::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_text","text"),&RawText::set_text);
	ObjectTypeDB::bind_method(_MD("get_text"),&RawText::get_text);
	ADD_PROPERTY( PropertyInfo(Variant::STRING,"text"), _SCS("set_text"), _SCS("get_text"));

}

void RawText::get_recognized_extensions(List<String> *p_extensions) {
	String extensions[] = {
		"txt", "md", "json", "toml", "cfg", "ini", "inf", "c", "cpp", "cxx",
		"c++", "h", "hpp",  "m", "mm", "css", "htm", "html", "py", "java", "js",
		"sh", "cmd", "csv", "cs", "as", "bash", "hlsl", "glsl", "diff", "php",
		"lua", "rb", "yml", "yaml", "go"
#ifndef XML_ENABLED
		,"xml"
#endif
	};
	for(size_t i=0; i<sizeof(extensions)/sizeof (String); i++) {
		p_extensions->push_back(extensions[i]);
	}
}

Error RawText::load_text(const String& p_path) {

	DVector<uint8_t> sourcef;
	Error err;
	FileAccess *f=FileAccess::open(p_path,FileAccess::READ,&err);
	if (err) {

		ERR_FAIL_COND_V(err,err);
	}

	int len = f->get_len();
	sourcef.resize(len+1);
	DVector<uint8_t>::Write w = sourcef.write();
	int r = f->get_buffer(w.ptr(),len);
	f->close();
	memdelete(f);
	ERR_FAIL_COND_V(r!=len,ERR_CANT_OPEN);
	w[len]=0;

	String s;
	if (s.parse_utf8((const char*)w.ptr())) {

		ERR_EXPLAIN("RawText '"+p_path+"' contains invalid unicode (utf-8), so it was not loaded. Please ensure that files are saved in valid utf-8 unicode.");
		ERR_FAIL_V(ERR_INVALID_DATA);
	}

	set_text(s);

	return OK;
}

/*************** RESOURCE ***************/

RES ResourceFormatLoaderRawText::load(const String &p_path, const String& p_original_path, Error *r_error) {

	if (r_error)
		*r_error=ERR_FILE_CANT_OPEN;

	RawText *rawtext = memnew( RawText );
	Ref<RawText> text(rawtext);

	Error err = text->load_text(p_path);
	if (err!=OK) {
		ERR_FAIL_COND_V(err!=OK, RES());
	}
	text->set_path(p_path);

	return text;
}

void ResourceFormatLoaderRawText::get_recognized_extensions(List<String> *p_extensions) const {
	RawText::get_recognized_extensions(p_extensions);
}


bool ResourceFormatLoaderRawText::handles_type(const String& p_type) const {
	return "RawText";
}

String ResourceFormatLoaderRawText::get_resource_type(const String &p_path) const {
	String el = p_path.extension().to_lower();
	if (el=="txt" || el=="md" || el=="json")
		return "RawText";
	return "";
}


Error ResourceFormatSaverRawText::save(const String &p_path,const RES& p_resource,uint32_t p_flags) {

	Ref<RawText> srtext = p_resource;
	ERR_FAIL_COND_V(srtext.is_null(),ERR_INVALID_PARAMETER);

	String text = srtext->get_text();

	Error err;
	FileAccess *file = FileAccess::open(p_path,FileAccess::WRITE,&err);

	if (err) {
		ERR_FAIL_COND_V(err,err);
	}

	file->store_string(text);
	if (file->get_error()!=OK && file->get_error()!=ERR_FILE_EOF) {
		memdelete(file);
		return ERR_CANT_CREATE;
	}
	file->close();
	memdelete(file);

	return OK;
}

void ResourceFormatSaverRawText::get_recognized_extensions(const RES& p_resource,List<String> *p_extensions) const {

	if (p_resource->cast_to<RawText>()) {
		RawText::get_recognized_extensions(p_extensions);
	}

}

bool ResourceFormatSaverRawText::recognize(const RES& p_resource) const {

	return p_resource->cast_to<RawText>()!=NULL;
}

