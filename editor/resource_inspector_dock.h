#ifndef RESOURCE_INSPECTOR_DOCK_H
#define RESOURCE_INSPECTOR_DOCK_H

#include "scene/gui/box_container.h"
#include "scene/gui/tab_container.h"

class ResourceInspectorDock : public VBoxContainer {
	GDCLASS(ResourceInspectorDock, VBoxContainer);

	static ResourceInspectorDock *singleton;
	TabContainer *tab_container = nullptr;

	void _tab_gui_input(const Ref<InputEvent> &p_event);
	void _cleanup_tabs();

protected:
	static void _bind_methods();

public:
	static ResourceInspectorDock *get_singleton() { return singleton; }
	Error inspect_resource(const String &p_resource);
	Error edit_resource(const Ref<Resource> &p_resource);

	ResourceInspectorDock();
	~ResourceInspectorDock();
};

#endif // RESOURCE_INSPECTOR_DOCK_H