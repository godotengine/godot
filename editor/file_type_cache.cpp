/*************************************************************************/
/*  file_type_cache.cpp                                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "file_type_cache.h"

#include "global_config.h"
#include "os/file_access.h"

FileTypeCache *FileTypeCache::singleton = NULL;

bool FileTypeCache::has_file(const String &p_path) const {

	GLOBAL_LOCK_FUNCTION
	return file_type_map.has(p_path);
}

String FileTypeCache::get_file_type(const String &p_path) const {

	GLOBAL_LOCK_FUNCTION
	ERR_FAIL_COND_V(!file_type_map.has(p_path), "");
	return file_type_map[p_path];
}
void FileTypeCache::set_file_type(const String &p_path, const String &p_type) {

	GLOBAL_LOCK_FUNCTION
	file_type_map[p_path] = p_type;
}

void FileTypeCache::load() {

	GLOBAL_LOCK_FUNCTION
	String project = GlobalConfig::get_singleton()->get_resource_path();
	FileAccess *f = FileAccess::open(project + "/file_type_cache.cch", FileAccess::READ);

	if (!f) {

		WARN_PRINT("Can't open file_type_cache.cch.");
		return;
	}

	file_type_map.clear();
	while (!f->eof_reached()) {

		String path = f->get_line();
		if (f->eof_reached())
			break;
		String type = f->get_line();
		set_file_type(path, type);
	}

	memdelete(f);
}

void FileTypeCache::save() {

	GLOBAL_LOCK_FUNCTION
	String project = GlobalConfig::get_singleton()->get_resource_path();
	FileAccess *f = FileAccess::open(project + "/file_type_cache.cch", FileAccess::WRITE);
	if (!f) {

		ERR_EXPLAIN(TTR("Can't open file_type_cache.cch for writing, not saving file type cache!"));
		ERR_FAIL();
	}

	const String *K = NULL;

	while ((K = file_type_map.next(K))) {

		f->store_line(*K);
		f->store_line(file_type_map[*K]);
	}

	memdelete(f);
}

FileTypeCache::FileTypeCache() {

	ERR_FAIL_COND(singleton);
	singleton = this;
}
