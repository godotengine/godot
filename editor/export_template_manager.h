/*************************************************************************/
/*  export_template_manager.h                                            */
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

#ifndef EXPORT_TEMPLATE_MANAGER_H
#define EXPORT_TEMPLATE_MANAGER_H

#include "editor/editor_settings.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/menu_button.h"
#include "scene/gui/progress_bar.h"
#include "scene/gui/scroll_container.h"
#include "scene/main/http_request.h"

class ExportTemplateVersion;

class ExportTemplateManager : public AcceptDialog {
	GDCLASS(ExportTemplateManager, AcceptDialog);

	bool current_version_exists = false;
	bool downloads_available = true;
	bool mirrors_available = false;
	bool is_refreshing_mirrors = false;
	bool is_downloading_templates = false;
	float update_countdown = 0;

	Label *current_value;
	Label *current_missing_label;
	Label *current_installed_label;

	HBoxContainer *current_installed_hb;
	LineEdit *current_installed_path;
	Button *current_open_button;
	Button *current_uninstall_button;

	VBoxContainer *install_options_vb;
	OptionButton *mirrors_list;

	enum MirrorAction {
		VISIT_WEB_MIRROR,
		COPY_MIRROR_URL,
	};

	MenuButton *mirror_options_button;
	HBoxContainer *download_progress_hb;
	ProgressBar *download_progress_bar;
	Label *download_progress_label;
	HTTPRequest *download_templates;
	Button *install_file_button;
	HTTPRequest *request_mirrors;

	enum TemplatesAction {
		OPEN_TEMPLATE_FOLDER,
		UNINSTALL_TEMPLATE,
	};

	Tree *installed_table;

	ConfirmationDialog *uninstall_confirm;
	String uninstall_version;
	FileDialog *install_file_dialog;
	AcceptDialog *hide_dialog_accept;

	void _update_template_status();

	void _download_current();
	void _download_template(const String &p_url, bool p_skip_check = false);
	void _download_template_completed(int p_status, int p_code, const PackedStringArray &headers, const PackedByteArray &p_data);
	void _cancel_template_download();
	void _refresh_mirrors();
	void _refresh_mirrors_completed(int p_status, int p_code, const PackedStringArray &headers, const PackedByteArray &p_data);

	bool _humanize_http_status(HTTPRequest *p_request, String *r_status, int *r_downloaded_bytes, int *r_total_bytes);
	void _set_current_progress_status(const String &p_status, bool p_error = false);
	void _set_current_progress_value(float p_value, const String &p_status);

	void _install_file();
	bool _install_file_selected(const String &p_file, bool p_skip_progress = false);

	void _uninstall_template(const String &p_version);
	void _uninstall_template_confirmed();

	String _get_selected_mirror() const;
	void _mirror_options_button_cbk(int p_id);
	void _installed_table_button_cbk(Object *p_item, int p_column, int p_id);

	void _open_template_folder(const String &p_version);

	virtual void ok_pressed() override;
	void _hide_dialog();

protected:
	void _notification(int p_what);
	static void _bind_methods();

public:
	bool can_install_android_template();
	Error install_android_template();

	Error install_android_template_from_file(const String &p_file);

	void popup_manager();

	ExportTemplateManager();
};

#endif // EXPORT_TEMPLATE_MANAGER_H
