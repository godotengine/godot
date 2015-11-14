/*************************************************************************/
/*  editor_translation.cpp                                               */
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

#include "editor_translation.h"

//#include "globals.h"
//#include "io/resource_loader.h"
#include "os/os.h"
#include "os/file_access.h"
#include "os/dir_access.h"
#include "core/io/translation_loader_po.h"
#include "tools/editor/editor_settings.h"

EditorTranslationServer *EditorTranslationServer::singleton=NULL;

void EditorTranslationServer::load() {
	clear_translations();

	// load settings
	if (!EditorSettings::get_singleton())
		EditorSettings::create();

	// load locale
	String locale = OS::get_singleton()->get_locale();
	if (EditorSettings::get_singleton()->has("editor_language/locale"))
		locale = EditorSettings::get_singleton()->get("editor_language/locale");
	set_locale(locale);

	// load translation
	String po =  EditorSettings::get_singleton()->get_settings_path() + "/lang/" + locale +"/editor.po";
	if (!FileAccess::exists(po))
		return;

	TranslationLoaderPO *po_loader = memnew(TranslationLoaderPO);
	Ref<Translation> tran = po_loader->load(po);
	if (tran.is_valid())
		add_translation(tran);
	memdelete(po_loader);
}

String EditorTranslationServer::install(String p_file) {

	// get locale
	TranslationLoaderPO *po_loader = memnew(TranslationLoaderPO);
	Ref<Translation> tran = po_loader->load(p_file);
	if (tran.is_null())
		return "";

	String locale = tran->get_locale();
	if (!Translation::is_valid_locale(locale))
		return "";

	// make folder $SETTING/lang/$LOCALE
	String lan_dir = EditorSettings::get_singleton()->get_settings_path() + "/lang";
	DirAccess *da = DirAccess::open(EditorSettings::get_singleton()->get_settings_path());
	Error err;
	err = da->make_dir("lang");
	if (err!=OK && err!=ERR_ALREADY_EXISTS)
		return "";

	da->change_dir("lang");

	err = da->make_dir(locale);
	if (err!=OK && err!=ERR_ALREADY_EXISTS)
		return "";

	// copy po
	String po =  EditorSettings::get_singleton()->get_settings_path() + "/lang/" + locale +"/editor.po";
	FileAccess *src = FileAccess::open(p_file, FileAccess::READ, &err);
	ERR_FAIL_COND_V(err,"");
	Vector<uint8_t> buf;
	buf.resize(src->get_len());
	src->get_buffer(buf.ptr(),src->get_len());
	src->close();
	memdelete(src);

	FileAccess *dst = FileAccess::open(po, FileAccess::WRITE, &err);
	ERR_FAIL_COND_V(err,"");
	dst->store_buffer(buf.ptr(),buf.size());
	dst->close();
	memdelete(dst);

	return locale;
}

void EditorTranslationServer::create() {
	if (singleton)
		return;
	memnew(EditorTranslationServer);
}

void EditorTranslationServer::destroy() {

	if (!singleton)
		return;
	memdelete(singleton);
}

EditorTranslationServer::EditorTranslationServer() {
	singleton=this;
}
