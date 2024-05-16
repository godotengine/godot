
#ifdef TOOLS_ENABLED
#include "../unity/unity_link_server.h"
#include "scene/resources/texture.h"
#include "unity_link_server_editor_plugin.h"
#include "editor/editor_log.h"
#include "editor/editor_node.h"
#include "editor/editor_settings.h"
#include "editor/dependency_editor.h"
#include "editor/editor_file_system.h"

#include "scene/resources/texture.h"
#include "editor/plugins/editor_plugin.h"
#include "../unity/unity_link_server.h"

class UnityLinkServerEditorPlugin : public EditorPlugin {
	GDCLASS(UnityLinkServerEditorPlugin, EditorPlugin);

	UnityLinkServer server;

	bool started = false;

private:
	void _notification(int p_what);

public:
	UnityLinkServerEditorPlugin();
	void start();
	void stop();
};
void UnityLinkServerEditorPluginRegister::initialize()
{
    
	EditorPlugins::add_by_type<UnityLinkServerEditorPlugin>();
}

UnityLinkServerEditorPlugin::UnityLinkServerEditorPlugin() {
}

void UnityLinkServerEditorPlugin::_notification(int p_what) {
	switch (p_what) {
		case NOTIFICATION_ENTER_TREE: {
			start();
		} break;

		case NOTIFICATION_EXIT_TREE: {
			stop();
		} break;

		case NOTIFICATION_INTERNAL_PROCESS: {
			// The main loop can be run again during request processing, which modifies internal state of the protocol.
			// Thus, "polling" is needed to prevent it from parsing other requests while the current one isn't finished.
			if (started ) {
				server.poll();
			}
		} break;

		case EditorSettings::NOTIFICATION_EDITOR_SETTINGS_CHANGED: {
		} break;
	}
}

void UnityLinkServerEditorPlugin::start() {
	server.start() ;
	{
		EditorNode::get_log()->add_message("--- unity link server started port 9010---", EditorLog::MSG_TYPE_EDITOR);
		set_process_internal(true);
		started = true;
	}
}

void UnityLinkServerEditorPlugin::stop() {
	server.stop();
	started = false;
	EditorNode::get_log()->add_message("--- unity link server stopped ---", EditorLog::MSG_TYPE_EDITOR);
}
#endif