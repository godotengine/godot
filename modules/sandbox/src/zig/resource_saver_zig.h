/**************************************************************************/
/*  resource_saver_zig.h                                                  */
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

#pragma once

#include "core/io/resource_saver.h"
#include <godot_cpp/classes/resource_saver.hpp>


class ResourceFormatSaverZig : public ResourceFormatSaver {
	GDCLASS(ResourceFormatSaverZig, ResourceFormatSaver);

protected:
	static void _bind_methods() {}

public:
	static void init();
	static void deinit();
	virtual Error _save(const Ref<Resource> &p_resource, const String &p_path, uint32_t p_flags) override;
	virtual Error _set_uid(const String &p_path, int64_t p_uid) override;
	virtual bool _recognize(const Ref<Resource> &p_resource) const override;
	virtual PackedStringArray _get_recognized_extensions(const Ref<Resource> &p_resource) const override;
	virtual bool _recognize_path(const Ref<Resource> &p_resource, const String &p_path) const override;
};
