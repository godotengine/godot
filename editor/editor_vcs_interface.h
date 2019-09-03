#ifndef EDITOR_VCS_INTERFACE_H
#define EDITOR_VCS_INTERFACE_H

#include "core/object.h"
#include "core/ustring.h"
#include "scene/gui/panel_container.h"

class EditorVCSInterface : public Object {

	GDCLASS(EditorVCSInterface, Object)

	bool is_initialized;

protected:
	static EditorVCSInterface *singleton;

	static void _bind_methods();

	// Implemented by addons as end points for the proxy functions
	bool _initialize(String p_project_root_path);
	bool _get_is_vcs_intialized();
	Dictionary _get_modified_files_data();
	void _stage_file(String p_file_path);
	void _unstage_file(String p_file_path);
	void _commit(String p_msg);
	Array _get_file_diff(String p_file_path);
	bool _shut_down();
	String _get_project_name();
	String _get_vcs_name();

public:
	static EditorVCSInterface *get_singleton();
	static void set_singleton(EditorVCSInterface *p_singleton);

	bool is_addon_ready();

	// Proxy functions to the editor for use
	bool initialize(String p_project_root_path);
	bool get_is_vcs_intialized();
	Dictionary get_modified_files_data();
	void stage_file(String p_file_path);
	void unstage_file(String p_file_path);
	void commit(String p_msg);
	Array get_file_diff(String p_file_path);
	bool shut_down();
	String get_project_name();
	String get_vcs_name();

	EditorVCSInterface();
	virtual ~EditorVCSInterface();
};

#endif // !EDITOR_VCS_INTERFACE_H
