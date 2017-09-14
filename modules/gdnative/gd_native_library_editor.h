#ifndef GD_NATIVE_LIBRARY_EDITOR_H
#define GD_NATIVE_LIBRARY_EDITOR_H

#ifdef TOOLS_ENABLED
#include "editor/project_settings_editor.h"
#include "editor/editor_file_system.h"

class GDNativeLibraryEditor : public VBoxContainer
{
	Tree *libraries;

	bool updating;
	void _update_libraries();

	void _find_gdnative_singletons(EditorFileSystemDirectory *p_dir,const Set<String>& enabled_list);
	void _item_edited();
protected:

	void _notification(int p_what);
	static void _bind_methods();
public:
	GDNativeLibraryEditor();
};

#endif
#endif // GD_NATIVE_LIBRARY_EDITOR_H
