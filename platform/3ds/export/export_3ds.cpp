#include "export_3ds.h"

#include "core/error/error_list.h"
#include "core/os/os.h"

void Export3DS::_bind_methods() {
}

Export3DS::Export3DS() {
}

Error Export3DS::export_project(
		const String &project_path,
		const String &output_path) {

	print_line("Godot 3DS Exporter");
	print_line("Project: " + project_path);
	print_line("Output: " + output_path);

	return OK;
}
