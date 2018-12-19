/*************************************************************************/
/*  godotsharp_export.h                                                  */
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

#ifndef GODOTSHARP_EXPORT_H
#define GODOTSHARP_EXPORT_H

#include <mono/metadata/image.h>

#include "editor/editor_export.h"

#include "../mono_gd/gd_mono_header.h"

class GodotSharpExport : public EditorExportPlugin {

	MonoAssemblyName *aname_prealloc;

	bool _add_file(const String &p_src_path, const String &p_dst_path, bool p_remap = false);

	Error _get_assembly_dependencies(GDMonoAssembly *p_assembly, const Vector<String> &p_search_dirs, Map<String, String> &r_dependencies);

protected:
	virtual void _export_file(const String &p_path, const String &p_type, const Set<String> &p_features);
	virtual void _export_begin(const Set<String> &p_features, bool p_debug, const String &p_path, int p_flags);

public:
	static void register_internal_calls();

	GodotSharpExport();
	~GodotSharpExport();
};

#endif // GODOTSHARP_EXPORT_H
