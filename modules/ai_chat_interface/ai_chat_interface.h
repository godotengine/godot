#ifndef AI_CHAT_INTERFACE_H
#define AI_CHAT_INTERFACE_H

#include "editor/editor_plugin.h"
#include "scene/gui/button.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/panel_container.h"
#include "scene/gui/text_edit.h"
#include "scene/gui/box_container.h"
#include "core/io/websocket_client.h"
#include "core/config/project_settings.h" // Required for ProjectSettings
#include "core/io/dir_access.h" // Required for DirAccess

// Forward declaration
class CodeEdit;

class AIChatInterfacePlugin : public EditorPlugin {
    GDCLASS(AIChatInterfacePlugin, EditorPlugin);

private:
    Button *chat_button;
    PanelContainer *chat_dock;
    TextEdit *chat_history;
    LineEdit *message_input;
    Button *send_button;

    Ref<WebSocketClient> ws_client;

    void _chat_button_pressed();
    void _send_button_pressed();

    void _on_websocket_connected(String p_protocol);
    void _on_websocket_error();
    void _on_websocket_connection_closed(bool p_was_clean_close);
    void _on_websocket_data_received();

    // Helper for scene manipulation
    void _handle_create_node_command(const PackedStringArray &p_parts);
    // Helper for scripting assistance
    void _handle_insert_code_snippet_command(const String &p_code_snippet);
    // Helpers for project understanding
    void _handle_get_project_setting_command(const String &p_setting_name);
    void _handle_list_scene_files_command();
    void _scan_dir_for_scenes(const String &p_path, PackedStringArray &r_scene_files);


public:
    AIChatInterfacePlugin();
    ~AIChatInterfacePlugin();

protected:
    void _notification(int p_what);
    static void _bind_methods();

    void _enter_tree() override;
    void _exit_tree() override;
    void _process(double p_delta) override;
};

#endif // AI_CHAT_INTERFACE_H
