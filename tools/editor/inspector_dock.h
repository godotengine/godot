#ifndef INSPECTOR_DOCK_H
#define INSPECTOR_DOCK_H

#include "scene/gui/box_container.h"
#include "property_editor.h"


//this is for now bundled in EditorNode, will be moved away here eventually

#if 0
class InspectorDock : public VBoxContainer
{
	OBJ_TYPE(InspectorDock,VBoxContainer);

	PropertyEditor *property_editor;

	EditorHistory editor_history;

	void _go_next();
	void _go_prev();

protected:

	static void _bind_methods();
public:

	EditorHistory &get_editor_history();

	PropertyEditor *get_property_editor();

	InspectorDock();
};

#endif
#endif // INSPECTOR_DOCK_H
