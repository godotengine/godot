/**************************************************************************/
/*  resource_saver_elf.cpp                                                */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#include "resource_saver_elf.h"
#include "../register_types.h"
#include "core/error/error_list.h"
#include "core/io/file_access.h"
#include "core/io/resource.h"
#include "core/io/resource_uid.h"
#include "core/object/object.h"
#include "core/object/ref_counted.h"
#include "core/string/ustring.h"
#include "core/templates/list.h"
#include "core/typedefs.h"
#include "script_elf.h"
#include "script_language_elf.h"

Error ResourceFormatSaverELF::save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags) {
	// Do not save, revert instead
	Ref<ELFScript> elf_model = Object::cast_to<ELFScript>(p_resource.ptr());
	if (elf_model.is_null()) {
		return Error::OK;
	}
	elf_model->set_file(p_path);
	elf_model->reload(true);
	return Error::OK;
}

Error ResourceFormatSaverELF::set_uid(const String &p_path, ResourceUID::ID p_uid) {
	return Error::OK;
}

bool ResourceFormatSaverELF::recognize(const Ref<Resource> &p_resource) const {
	return Object::cast_to<ELFScript>(p_resource.ptr()) != nullptr;
}

void ResourceFormatSaverELF::get_recognized_extensions(const Ref<Resource> &p_resource, List<String> *p_extensions) const {
	if (Object::cast_to<ELFScript>(p_resource.ptr()) != nullptr) {
		p_extensions->push_back("elf");
	}
}

bool ResourceFormatSaverELF::recognize_path(const Ref<Resource> &p_resource, const String &p_path) const {
	return Object::cast_to<ELFScript>(p_resource.ptr()) != nullptr;
}
