/*************************************************************************/
/*  resource_saver.h                                                     */
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
#ifndef RESOURCE_SAVER_H
#define RESOURCE_SAVER_H

#include "resource.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/

class ResourceFormatSaver {
public:
	virtual Error save(const String &p_path, const RES &p_resource, uint32_t p_flags = 0) = 0;
	virtual bool recognize(const RES &p_resource) const = 0;
	virtual void get_recognized_extensions(const RES &p_resource, List<String> *p_extensions) const = 0;

	virtual ~ResourceFormatSaver() {}
};

typedef void (*ResourceSavedCallback)(const String &p_path);

class ResourceSaver {

	enum {
		MAX_SAVERS = 64
	};

	static ResourceFormatSaver *saver[MAX_SAVERS];
	static int saver_count;
	static bool timestamp_on_save;
	static ResourceSavedCallback save_callback;

public:
	enum SaverFlags {

		FLAG_RELATIVE_PATHS = 1,
		FLAG_BUNDLE_RESOURCES = 2,
		FLAG_CHANGE_PATH = 4,
		FLAG_OMIT_EDITOR_PROPERTIES = 8,
		FLAG_SAVE_BIG_ENDIAN = 16,
		FLAG_COMPRESS = 32,
		FLAG_REPLACE_SUBRESOURCE_PATHS = 64,
	};

	static Error save(const String &p_path, const RES &p_resource, uint32_t p_flags = 0);
	static void get_recognized_extensions(const RES &p_resource, List<String> *p_extensions);
	static void add_resource_format_saver(ResourceFormatSaver *p_format_saver, bool p_at_front = false);

	static void set_timestamp_on_save(bool p_timestamp) { timestamp_on_save = p_timestamp; }
	static void set_save_callback(ResourceSavedCallback p_callback);
};

#endif
