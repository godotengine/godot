/*************************************************************************/
/*  project_manager.h                                                    */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef PROJECT_MANAGER_H
#define PROJECT_MANAGER_H

#include "scene/gui/dialogs.h"
#include "scene/gui/tree.h"
#include "scene/gui/scroll_container.h"
#include "scene/gui/file_dialog.h"

class NewProjectDialog;

class ProjectManager : public Control {
	OBJ_TYPE( ProjectManager, Control );

	Button *erase_btn;
	Button *open_btn;
	Button *run_btn;

	FileDialog *scan_dir;

	ConfirmationDialog *erase_ask;
	ConfirmationDialog *multi_open_ask;
	ConfirmationDialog *multi_run_ask;
	NewProjectDialog *npdialog;
	ScrollContainer *scroll;
	VBoxContainer *scroll_childs;
	Map<String, String> selected_list; // name -> main_scene
	String last_clicked;
	String single_selected_main;
	bool importing;

	void _item_doubleclicked();



	void _scan_projects();
	void _run_project();
	void _run_project_confirm();
	void _open_project();
	void _open_project_confirm();
	void _import_project();
	void _new_project();
	void _erase_project();
	void _erase_project_confirm();
	void _exit_dialog();
	void _scan_begin(const String& p_base);

	void _load_recent_projects();
	void _scan_dir(DirAccess *da,float pos, float total,List<String> *r_projects);

	void _panel_draw(Node *p_hb);
	void _panel_input(const InputEvent& p_ev,Node *p_hb);
	void _favorite_pressed(Node *p_hb);

protected:

	static void _bind_methods();
public:
	ProjectManager();
	~ProjectManager();
};

#endif // PROJECT_MANAGER_H
