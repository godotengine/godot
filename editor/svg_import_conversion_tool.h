/**************************************************************************/
/*  svg_import_conversion_tool.h                                          */
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

#include "core/object/class_db.h"

class ConfirmationDialog;
class EditorFileSystemDirectory;

class SvgImportConversionTool : public Object {
	GDCLASS(SvgImportConversionTool, Object);

	ConfirmationDialog *convert_dialog = nullptr;

	void _find_svg_files(EditorFileSystemDirectory *p_dir, Vector<String> &r_svg_paths);
	bool _is_texture2d_import(const String &p_import_path);
	void _convert_svg_file(const String &p_path);
	void _on_dialog_confirmed();

	const String META_REIMPORT_PATHS = "reimport_paths";

public:
	const String META_SVG_IMPORT_CONVERSION_TOOL = "svg_import_conversion_tool";
	const String META_RUN_ON_RESTART = "run_on_restart";
	const StringName CONVERSION_FINISHED = "conversion_finished";

protected:
	static void _bind_methods();

public:
	void popup_dialog();
	void prepare_conversion();
	void begin_conversion();
	void finish_conversion();
	void convert_svgs();
};
