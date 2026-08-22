#ifndef GODOT_EXPORT_PLUGIN_3DS_H
#define GODOT_EXPORT_PLUGIN_3DS_H

#include "editor/export/editor_export_platform.h"

class EditorExportPlatform3DS : public EditorExportPlatform {
	GDCLASS(EditorExportPlatform3DS, EditorExportPlatform);

protected:
	static void _bind_methods();

public:
	virtual String get_name() const override;
	virtual String get_os_name() const override;
	virtual List<String> get_binary_extensions(const Ref<EditorExportPreset> &p_preset) const override;

	EditorExportPlatform3DS();
};

#endif
