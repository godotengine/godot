/*************************************************************************/
/*  dependency_editor.h                                                  */
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
#ifndef DEPENDENCY_EDITOR_H
#define DEPENDENCY_EDITOR_H

#include "editor_file_dialog.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/tree.h"

class EditorFileSystemDirectory;

class DependencyEditor : public AcceptDialog {
	GDCLASS(DependencyEditor, AcceptDialog);

	Tree *tree;
	Button *fixdeps;

	EditorFileDialog *search;

	String replacing;
	String editing;
	List<String> missing;

	void _fix_and_find(EditorFileSystemDirectory *efsd, Map<String, Map<String, String> > &candidates);

	void _searched(const String &p_path);
	void _load_pressed(Object *p_item, int p_cell, int p_button);
	void _fix_all();
	void _update_list();

	void _update_file();

protected:
	static void _bind_methods();
	void _notification(int p_what);

public:
	void edit(const String &p_path);
	DependencyEditor();
};

class DependencyEditorOwners : public AcceptDialog {
	GDCLASS(DependencyEditorOwners, AcceptDialog);

	ItemList *owners;
	String editing;
	void _fill_owners(EditorFileSystemDirectory *efsd);

public:
	void show(const String &p_path);
	DependencyEditorOwners();
};

class DependencyRemoveDialog : public ConfirmationDialog {
	GDCLASS(DependencyRemoveDialog, ConfirmationDialog);

	Label *text;
	Tree *owners;
	bool exist;
	Map<String, TreeItem *> files;
	void _fill_owners(EditorFileSystemDirectory *efsd);

	void ok_pressed();

public:
	void show(const Vector<String> &to_erase);
	DependencyRemoveDialog();
};

class DependencyErrorDialog : public ConfirmationDialog {
	GDCLASS(DependencyErrorDialog, ConfirmationDialog);

	String for_file;
	Button *fdep;
	Label *text;
	Tree *files;
	void ok_pressed();
	void custom_action(const String &);

public:
	void show(const String &p_for, const Vector<String> &report);
	DependencyErrorDialog();
};

class OrphanResourcesDialog : public ConfirmationDialog {
	GDCLASS(OrphanResourcesDialog, ConfirmationDialog);

	DependencyEditor *dep_edit;
	Tree *files;
	ConfirmationDialog *delete_confirm;
	void ok_pressed();

	bool _fill_owners(EditorFileSystemDirectory *efsd, HashMap<String, int> &refs, TreeItem *p_parent);

	List<String> paths;
	void _find_to_delete(TreeItem *p_item, List<String> &paths);
	void _delete_confirm();
	void _button_pressed(Object *p_item, int p_column, int p_id);

	void refresh();
	static void _bind_methods();

public:
	void show();
	OrphanResourcesDialog();
};

#endif // DEPENDENCY_EDITOR_H
