/*************************************************************************/
/*  editor_import_plugin.h                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef EDITOR_IMPORT_PLUGIN_H
#define EDITOR_IMPORT_PLUGIN_H

#include "core/io/resource_importer.h"

class EditorImportPlugin : public ResourceImporter {
	GDCLASS(EditorImportPlugin, ResourceImporter);

protected:
	static void _bind_methods();

	GDVIRTUAL0RC(String, _get_importer_name)
	GDVIRTUAL0RC(String, _get_visible_name)
	GDVIRTUAL0RC(int, _get_preset_count)
	GDVIRTUAL1RC(String, _get_preset_name, int)
	GDVIRTUAL0RC(Vector<String>, _get_recognized_extensions)
	GDVIRTUAL2RC(Array, _get_import_options, String, int)
	GDVIRTUAL0RC(String, _get_save_extension)
	GDVIRTUAL0RC(String, _get_resource_type)
	GDVIRTUAL0RC(float, _get_priority)
	GDVIRTUAL0RC(int, _get_import_order)
	GDVIRTUAL3RC(bool, _get_option_visibility, String, StringName, Dictionary)
	GDVIRTUAL5RC(int, _import, String, String, Dictionary, Array, Array)

public:
	EditorImportPlugin();
	virtual String get_importer_name() const override;
	virtual String get_visible_name() const override;
	virtual void get_recognized_extensions(List<String> *p_extensions) const override;
	virtual String get_preset_name(int p_idx) const override;
	virtual int get_preset_count() const override;
	virtual String get_save_extension() const override;
	virtual String get_resource_type() const override;
	virtual float get_priority() const override;
	virtual int get_import_order() const override;
	virtual void get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const override;
	virtual bool get_option_visibility(const String &p_path, const String &p_option, const Map<StringName, Variant> &p_options) const override;
	virtual Error import(const String &p_source_file, const String &p_save_path, const Map<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata = nullptr) override;
};

#endif //EDITOR_IMPORT_PLUGIN_H
