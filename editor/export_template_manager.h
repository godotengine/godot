#ifndef EXPORT_TEMPLATE_MANAGER_H
#define EXPORT_TEMPLATE_MANAGER_H

#include "editor/editor_settings.h"
#include "scene/gui/dialogs.h"
#include "scene/gui/file_dialog.h"
#include "scene/gui/scroll_container.h"

class ExportTemplateVersion;

class ExportTemplateManager : public ConfirmationDialog {
	GDCLASS(ExportTemplateManager, ConfirmationDialog)

	ScrollContainer *installed_scroll;
	VBoxContainer *installed_vb;
	HBoxContainer *current_hb;
	FileDialog *template_open;

	ConfirmationDialog *remove_confirm;
	String to_remove;

	void _update_template_list();

	void _download_template(const String &p_version);
	void _uninstall_template(const String &p_version);
	void _uninstall_template_confirm();

	virtual void ok_pressed();
	void _install_from_file(const String &p_file);

protected:
	static void _bind_methods();

public:
	void popup_manager();

	ExportTemplateManager();
};

#endif // EXPORT_TEMPLATE_MANAGER_H
