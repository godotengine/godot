/*************************************************************************/
/*  editor_sample_import_plugin.h                                        */
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
#ifndef EDITOR_SAMPLE_IMPORT_PLUGIN_H
#define EDITOR_SAMPLE_IMPORT_PLUGIN_H

#if 0
#include "editor/editor_import_export.h"
#include "scene/resources/font.h"

class EditorNode;
class EditorSampleImportDialog;

class EditorSampleImportPlugin : public EditorImportPlugin {

	GDCLASS(EditorSampleImportPlugin,EditorImportPlugin);

	EditorSampleImportDialog *dialog;
	void _compress_ima_adpcm(const Vector<float>& p_data,PoolVector<uint8_t>& dst_data);
public:

	static EditorSampleImportPlugin *singleton;

	virtual String get_name() const;
	virtual String get_visible_name() const;
	virtual void import_dialog(const String& p_from="");
	virtual Error import(const String& p_path, const Ref<ResourceImportMetadata>& p_from);
	void import_from_drop(const Vector<String>& p_drop, const String &p_dest_path);
	virtual void reimport_multiple_files(const Vector<String>& p_list);
	virtual bool can_reimport_multiple_files() const;


	EditorSampleImportPlugin(EditorNode* p_editor);
};

class EditorSampleExportPlugin : public EditorExportPlugin {

	GDCLASS( EditorSampleExportPlugin, EditorExportPlugin);


public:

	virtual Vector<uint8_t> custom_export(String& p_path,const Ref<EditorExportPlatform> &p_platform);

	EditorSampleExportPlugin();
};

#endif // EDITOR_SAMPLE_IMPORT_PLUGIN_H
#endif
