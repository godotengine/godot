/*************************************************************************/
/*  import_settings.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2016 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef IMPORT_SETTINGS_H
#define IMPORT_SETTINGS_H

#include "object.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/tree.h"
#include "scene/gui/label.h"
#include "scene/gui/option_button.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/progress_bar.h"
#include "scene/gui/slider.h"
#include "scene/gui/spin_box.h"
#include "scene/resources/mesh.h"
#include "editor_import_export.h"
#include "editor_file_system.h"
#include "editor_dir_dialog.h"
class EditorNode;


class ImportSettingsDialog : public ConfirmationDialog {

	OBJ_TYPE(ImportSettingsDialog,ConfirmationDialog);

	TreeItem *edited;
	EditorNode *editor;
	Tree *tree;
	bool updating;

	void _button_pressed(Object *p_button, int p_col, int p_id);
	void _item_pressed(int p_idx);
	bool _generate_fs(TreeItem *p_parent,EditorFileSystemDirectory *p_dir);

	String texformat;

	void _item_edited();
	virtual void ok_pressed();

protected:


	void _notification(int p_what);
	static void _bind_methods();
public:

	void update_tree();


	void popup_import_settings();
	ImportSettingsDialog(EditorNode *p_editor);

};

#endif // IMPORT_SETTINGS_H
