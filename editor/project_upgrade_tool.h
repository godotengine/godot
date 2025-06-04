/**************************************************************************/
/*  project_upgrade_tool.h                                                */
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

class ProjectUpgradeTool : public Object {
	GDCLASS(ProjectUpgradeTool, Object);

	ConfirmationDialog *upgrade_dialog = nullptr;

	void _add_files(EditorFileSystemDirectory *p_dir, Vector<String> &r_reimport_paths, Vector<String> &r_resave_scenes, Vector<String> &r_resave_resources);

	const String META_REIMPORT_PATHS = "reimport_paths";
	const String META_RESAVE_SCENES = "resave_scenes";
	const String META_RESAVE_RESOURCES = "resave_resources";

public:
	const String META_PROJECT_UPGRADE_TOOL = "project_upgrade_tool";
	const String META_RUN_ON_RESTART = "run_on_restart";
	const StringName UPGRADE_FINISHED = "upgrade_finished";

protected:
	static void _bind_methods();

public:
	void popup_dialog();
	void prepare_upgrade();
	void begin_upgrade();
	void finish_upgrade();
};
