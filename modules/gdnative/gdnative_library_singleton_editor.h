/*************************************************************************/
/*  gdnative_library_singleton_editor.h                                  */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2021 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2021 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef GD_NATIVE_LIBRARY_EDITOR_H
#define GD_NATIVE_LIBRARY_EDITOR_H

#ifdef TOOLS_ENABLED
#include "editor/editor_file_system.h"
#include "editor/project_settings_editor.h"

class GDNativeLibrarySingletonEditor : public VBoxContainer {
	GDCLASS(GDNativeLibrarySingletonEditor, VBoxContainer);

private:
	Tree *libraries;
	UndoRedo *undo_redo;

	bool updating;

	static Set<String> _find_singletons_recursive(EditorFileSystemDirectory *p_dir);

protected:
	void _notification(int p_what);
	static void _bind_methods();

	void _discover_singletons();
	void _item_edited();
	void _update_libraries();

public:
	GDNativeLibrarySingletonEditor();
};

#endif
#endif // GD_NATIVE_LIBRARY_EDITOR_H
