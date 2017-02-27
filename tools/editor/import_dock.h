#ifndef IMPORTDOCK_H
#define IMPORTDOCK_H

#include "io/resource_import.h"
#include "editor_file_system.h"
#include "scene/gui/box_container.h"
#include "scene/gui/option_button.h"
#include "scene/gui/popup_menu.h"
#include "property_editor.h"

class ImportDockParameters;
class ImportDock : public VBoxContainer {
	GDCLASS(ImportDock,VBoxContainer)

	LineEdit *imported;
	OptionButton *import_as;
	MenuButton *preset;
	PropertyEditor *import_opts;

	List<PropertyInfo> properties;
	Map<StringName,Variant> property_values;

	Button *import;

	ImportDockParameters *params;

	void _preset_selected(int p_idx);

	void _reimport();
protected:
	static void _bind_methods();
public:

	void set_edit_path(const String& p_path);
	void set_edit_multiple_paths(const Vector<String>& p_paths);
	void clear();

	ImportDock();
	~ImportDock();
};

#endif // IMPORTDOCK_H
