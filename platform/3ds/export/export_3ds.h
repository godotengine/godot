#ifndef GODOT_EXPORT_3DS_H
#define GODOT_EXPORT_3DS_H

#include "core/object/ref_counted.h"
#include "core/string/ustring.h"

class Export3DS : public RefCounted {
	GDCLASS(Export3DS, RefCounted);

protected:
	static void _bind_methods();

public:
	Error export_project(
			const String &project_path,
			const String &output_path);

	Export3DS();
};

#endif
