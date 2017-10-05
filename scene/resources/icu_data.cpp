/*************************************************************************/
/*  icu_data.cpp                                                         */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2018 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2018 Godot Engine contributors (cf. AUTHORS.md)    */
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

#include "scene/resources/icu_data.h"
#include "core/os/file_access.h"
#include "core/project_settings.h"

#ifdef TOOLS_ENABLED
#include "core/engine.h"
#include "main/main.h"
#endif

/*************************************************************************/
/*  ICUData                                                              */
/*************************************************************************/

#ifdef USE_TEXT_SHAPING
bool ICUData::icu_data_loaded = false;
Vector<uint8_t> ICUData::icu_data;
#endif

void ICUData::_bind_methods() {
	//NOP
}

bool ICUData::is_loaded() {

#ifdef USE_TEXT_SHAPING
	return icu_data_loaded;
#else
	return false;
#endif
}

void ICUData::init() {

#ifdef USE_TEXT_SHAPING
#ifdef ICU_STATIC_DATA
	//Initalize ICU data (static)
	err = U_ZERO_ERROR;
	u_init(&err);
	if (U_FAILURE(err)) {
		ERR_PRINT(u_errorName(err));
		return;
	}
	icu_data_loaded = true;
#else

#ifdef TOOLS_ENABLED
	if (Engine::get_singleton()->is_editor_hint() ||
			ProjectSettings::get_singleton()->get_resource_path().empty() ||
			Main::is_project_manager()) {
		return;
	}
#endif

	//Initialize ICU data from resource, should be done at most once in a process, before the first ICU operation (e.g., u_init())
	String datapath = ProjectSettings::get_singleton()->get("application/config/icudata");
	if (datapath != String() && FileAccess::exists(datapath)) {
		FileAccess *f = FileAccess::open(datapath, FileAccess::READ);
		if (f) {
			UErrorCode err = U_ZERO_ERROR;

			//ICU data found
			size_t len = f->get_len();
			icu_data.resize(len);
			f->get_buffer(icu_data.ptrw(), len);
			f->close();

			udata_setCommonData(icu_data.ptr(), &err);
			if (U_FAILURE(err)) {
				ERR_PRINT(u_errorName(err));
				return;
			}

			err = U_ZERO_ERROR;
			u_init(&err);
			if (U_FAILURE(err)) {
				ERR_PRINT(u_errorName(err));
				return;
			}

			icu_data_loaded = true;
		}
	}
#endif
#endif
}

void ICUData::finish() {

#ifdef USE_TEXT_SHAPING
	//Cleanup ICU data
	u_cleanup();
#endif
}

/*************************************************************************/
/*  ResourceFormatLoaderICUData                                          */
/*************************************************************************/

RES ResourceFormatLoaderICUData::load(const String &p_path, const String &p_original_path, Error *r_error) {

	if (r_error)
		*r_error = ERR_FILE_CANT_OPEN;

	Ref<ICUData> dummy;
	dummy.instance();

	if (!FileAccess::exists(p_path)) {
		return RES();
	}

	if (r_error)
		*r_error = OK;

	return dummy;
}

void ResourceFormatLoaderICUData::get_recognized_extensions(List<String> *p_extensions) const {

	p_extensions->push_back("icudt");
}

bool ResourceFormatLoaderICUData::handles_type(const String &p_type) const {

	return (p_type == "ICUData");
}

String ResourceFormatLoaderICUData::get_resource_type(const String &p_path) const {

	String el = p_path.get_extension().to_lower();
	if (el == "icudt")
		return "ICUData";
	return "";
}
