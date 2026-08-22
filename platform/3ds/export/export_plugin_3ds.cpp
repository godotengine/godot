#include "export_plugin_3ds.h"

#include "core/object/class_db.h"

void EditorExportPlatform3DS::_bind_methods() {
}

EditorExportPlatform3DS::EditorExportPlatform3DS() {
}

String EditorExportPlatform3DS::get_name() const {
	return "Nintendo 3DS";
}

String EditorExportPlatform3DS::get_os_name() const {
	return "3ds";
}

List<String> EditorExportPlatform3DS::get_binary_extensions(
		const Ref<EditorExportPreset> &p_preset) const {

	List<String> extensions;

	extensions.push_back("elf");
	extensions.push_back("3dsx");

	return extensions;
}
