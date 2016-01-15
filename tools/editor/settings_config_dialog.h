/*************************************************************************/
/*  settings_config_dialog.h                                             */
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
#ifndef SETTINGS_CONFIG_DIALOG_H
#define SETTINGS_CONFIG_DIALOG_H


#include "property_editor.h"
#include "scene/gui/tab_container.h"
#include "scene/gui/rich_text_label.h"

class EditorSettingsDialog : public AcceptDialog {

	OBJ_TYPE(EditorSettingsDialog,AcceptDialog);



	ConfirmationDialog *install_confirm;
	bool updating;
	ConfirmationDialog *plugin_setting;
	String plugin_setting_edit;

	RichTextLabel *plugin_description;

	TabContainer *tabs;

	Button *rescan_plugins;
	Tree *plugins;
	SectionedPropertyEditor *property_editor;

	Timer *timer;

	virtual void cancel_pressed();
	virtual void ok_pressed();

	void _plugin_edited();

	void _plugin_settings(Object *p_obj,int p_cell,int p_index);
	void _settings_changed();
	void _settings_save();

	void _plugin_install();

	void _notification(int p_what);

	void _rescan_plugins();
	void _update_plugins();

	void _clear_search_box();

protected:

	static void _bind_methods();
public:

	void popup_edit_settings();

	EditorSettingsDialog();
};

#endif // SETTINGS_CONFIG_DIALOG_H
