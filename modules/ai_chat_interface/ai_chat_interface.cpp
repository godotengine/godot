#include "ai_chat_interface.h"

#include "core/io/json.h"
#include "editor/editor_interface.h"
#include "editor/editor_paths.h"
#include "editor/editor_undo_redo_manager.h"
#include "editor/gui/editor_bottom_panel.h"
#include "editor/plugins/script_editor_plugin.h"
#include "scene/gui/code_edit.h"
#include "scene/main/node.h"
#include "scene/gui/label.h"
#include "scene/gui/margin_container.h"
// ProjectSettings is now included in the header
// DirAccess is now included in the header

const String CLOUDFLARE_WEBSOCKET_URL = "wss://your-cloudflare-worker-url"; // Placeholder URL

AIChatInterfacePlugin::AIChatInterfacePlugin() {
    chat_button = nullptr;
    chat_dock = nullptr;
    chat_history = nullptr;
    message_input = nullptr;
    send_button = nullptr;
    ws_client.instantiate();
}

AIChatInterfacePlugin::~AIChatInterfacePlugin() {
}

void AIChatInterfacePlugin::_chat_button_pressed() {
    if (chat_dock) {
        chat_dock->set_visible(!chat_dock->is_visible());
    }
}

void AIChatInterfacePlugin::_send_button_pressed() {
    if (message_input && chat_history && ws_client.is_valid() && ws_client->get_connection_status() == WebSocketPeer::CONNECTION_CONNECTED) {
        String text = message_input->get_text();
        if (!text.is_empty()) {
            chat_history->append_text(vformat("User: %s\n", text));
            Error err = ws_client->get_peer(1)->put_packet(text.to_utf8_buffer());
            if (err != OK) {
                chat_history->append_text("System: Error sending message.\n");
            }
            message_input->clear();
        }
    } else if (message_input && chat_history) {
         String text = message_input->get_text();
         if (!text.is_empty()) {
            chat_history->append_text(vformat("User (offline): %s\n", text));
            message_input->clear();
        }
        chat_history->append_text("System: Not connected to server.\n");
    }
}

void AIChatInterfacePlugin::_on_websocket_connected(String p_protocol) {
    if (chat_history) {
        chat_history->append_text("System: Connected to Cloudflare.\n");
    }
}

void AIChatInterfacePlugin::_on_websocket_error() {
    if (chat_history) {
        chat_history->append_text("System: WebSocket connection error.\n");
    }
}

void AIChatInterfacePlugin::_on_websocket_connection_closed(bool p_was_clean_close) {
    if (chat_history) {
        chat_history->append_text(vformat("System: WebSocket connection closed (Cleanly: %s).\n", p_was_clean_close ? "Yes" : "No"));
    }
}

void AIChatInterfacePlugin::_handle_create_node_command(const PackedStringArray &p_parts) {
    if (p_parts.size() < 4) {
        chat_history->append_text("System: Error - CREATE_NODE command requires 3 arguments (NodeType, NodeName, ParentPath).\n");
        return;
    }

    String node_type = p_parts[1];
    String node_name = p_parts[2];
    NodePath parent_path = NodePath(p_parts[3]);

    Node *edited_scene_root = EditorInterface::get_singleton()->get_edited_scene_root();
    if (!edited_scene_root) {
        chat_history->append_text("System: Error - No scene is currently being edited.\n");
        return;
    }

    Node *parent_node = edited_scene_root->get_node_or_null(parent_path);
    if (!parent_node) {
        chat_history->append_text(vformat("System: Error - Parent node not found at path: %s\n", parent_path));
        return;
    }

    Object *obj = ClassDB::instantiate(node_type);
    if (!obj) {
        chat_history->append_text(vformat("System: Error - Failed to instantiate node of type: %s\n", node_type));
        return;
    }

    Node *new_node = Object::cast_to<Node>(obj);
    if (!new_node) {
        memdelete(obj);
        chat_history->append_text(vformat("System: Error - Instantiated object is not a Node (type: %s)\n", node_type));
        return;
    }

    new_node->set_name(node_name);

    EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
    undo_redo->create_action(vformat(TTR("AI: Create Node '%s'"), node_name));
    undo_redo->add_do_method(parent_node, "add_child", new_node, true);
    undo_redo->add_do_method(new_node, "set_owner", edited_scene_root);
    undo_redo->add_undo_method(parent_node, "remove_child", new_node);
    undo_redo->commit_action();

    chat_history->append_text(vformat("System: Created node '%s' of type '%s' under '%s'.\n", node_name, node_type, parent_path));
}

void AIChatInterfacePlugin::_handle_insert_code_snippet_command(const String &p_code_snippet) {
    ScriptEditor *script_editor = ScriptEditor::get_singleton();
    if (!script_editor) {
         if (chat_history) chat_history->append_text("System: Error - ScriptEditor not found.\n");
        return;
    }

    ScriptEditorBase *current_editor = script_editor->get_current_editor();
    if (!current_editor) {
        if (chat_history) chat_history->append_text("System: Error - No script is currently open or active.\n");
        return;
    }

    CodeEdit *code_edit = current_editor->get_code_edit();
    if (!code_edit) {
        if (chat_history) chat_history->append_text("System: Error - Active editor does not have a CodeEdit control.\n");
        return;
    }

    EditorUndoRedoManager *undo_redo = EditorUndoRedoManager::get_singleton();
    undo_redo->create_action(TTR("AI: Insert Code Snippet"));
    String original_text = code_edit->get_text();
    int original_cursor_line = code_edit->get_caret_line();
    int original_cursor_col = code_edit->get_caret_column();

    code_edit->insert_text_at_caret(p_code_snippet);

    undo_redo->add_do_method(code_edit, "set_text", code_edit->get_text());
    undo_redo->add_do_method(code_edit, "set_caret_line", code_edit->get_caret_line());
    undo_redo->add_do_method(code_edit, "set_caret_column", code_edit->get_caret_column());
    undo_redo->add_undo_method(code_edit, "set_text", original_text);
    undo_redo->add_undo_method(code_edit, "set_caret_line", original_cursor_line);
    undo_redo->add_undo_method(code_edit, "set_caret_column", original_cursor_col);
    undo_redo->commit_action();

    if (chat_history) chat_history->append_text("System: Inserted code snippet into the active script.\n");
}

void AIChatInterfacePlugin::_handle_get_project_setting_command(const String &p_setting_name) {
    if (!ProjectSettings::get_singleton()->has_setting(p_setting_name)) {
        String response = vformat("SETTING_NOT_FOUND|%s", p_setting_name);
        if (ws_client.is_valid() && ws_client->get_peer(1).is_valid() && ws_client->get_connection_status() == WebSocketPeer::CONNECTION_CONNECTED) {
            ws_client->get_peer(1)->put_packet(response.to_utf8_buffer());
        }
        if (chat_history) chat_history->append_text(vformat("System: AI requested non-existent project setting '%s'.\n", p_setting_name));
        return;
    }
    Variant value = ProjectSettings::get_singleton()->get_setting(p_setting_name);
    String response = vformat("SETTING_VALUE|%s|%s", p_setting_name, String(value)); // Convert variant to string
    if (ws_client.is_valid() && ws_client->get_peer(1).is_valid() && ws_client->get_connection_status() == WebSocketPeer::CONNECTION_CONNECTED) {
        ws_client->get_peer(1)->put_packet(response.to_utf8_buffer());
    }
    if (chat_history) chat_history->append_text(vformat("System: AI requested project setting '%s', value: '%s'.\n", p_setting_name, String(value)));
}

void AIChatInterfacePlugin::_scan_dir_for_scenes(const String &p_path, PackedStringArray &r_scene_files) {
    Ref<DirAccess> da = DirAccess::open(p_path);
    if (!da.is_valid()) {
        return;
    }
    da->list_dir_begin();
    String file_name = da->get_next();
    while (!file_name.is_empty()) {
        if (da->current_is_dir() && file_name != "." && file_name != ".." && file_name != ".git" && file_name != ".import") { // Exclude common non-project dirs
            _scan_dir_for_scenes(p_path.path_join(file_name), r_scene_files);
        } else if (file_name.ends_with(".tscn") || file_name.ends_with(".scn")) {
            // Removed .res for now as it's too generic, can be added back with PackedScene check if needed
            r_scene_files.push_back(p_path.path_join(file_name));
        }
        file_name = da->get_next();
    }
    // list_dir_end is not strictly needed for Ref<DirAccess> as it closes on destruction, but good practice.
    // da->list_dir_end();
}

void AIChatInterfacePlugin::_handle_list_scene_files_command() {
    PackedStringArray scene_files;
    _scan_dir_for_scenes("res://", scene_files);

    String response_data;
    // Limit the number of files to avoid overly large packets, if necessary
    // const int max_files_to_send = 100;
    // for (int i = 0; i < MIN(scene_files.size(), max_files_to_send); ++i) {
    for (int i = 0; i < scene_files.size(); ++i) {
        if (i > 0) {
            response_data += ",";
        }
        response_data += scene_files[i];
    }

    String response = vformat("SCENE_FILES_LIST|%s", response_data);
    if (ws_client.is_valid() && ws_client->get_peer(1).is_valid() && ws_client->get_connection_status() == WebSocketPeer::CONNECTION_CONNECTED) {
        ws_client->get_peer(1)->put_packet(response.to_utf8_buffer());
    }
    if (chat_history) chat_history->append_text(vformat("System: AI requested scene file list. Found %d scenes.\n", scene_files.size()));
}


void AIChatInterfacePlugin::_on_websocket_data_received() {
    if (ws_client.is_valid() && ws_client->get_peer(1).is_valid() && ws_client->get_peer(1)->get_available_packet_count() > 0) {
        PackedByteArray packet = ws_client->get_peer(1)->get_packet();
        String data = packet.get_string_from_utf8();
        if (chat_history) {
            chat_history->append_text(vformat("AI: %s\n", data));

            PackedStringArray parts = data.split("|", false, 2);
            if (parts.size() > 0) {
                String command = parts[0].strip_edges(); // Trim whitespace from command
                if (command == "CREATE_NODE") {
                    if (parts.size() > 1) {
                         _handle_create_node_command(data.split("|"));
                    } else {
                         if (chat_history) chat_history->append_text("System: Error - CREATE_NODE command missing arguments.\n");
                    }
                } else if (command == "INSERT_CODE_SNIPPET") {
                    if (parts.size() > 1) {
                        _handle_insert_code_snippet_command(parts[1].strip_edges());
                    } else {
                         if (chat_history) chat_history->append_text("System: Error - INSERT_CODE_SNIPPET command missing code.\n");
                    }
                } else if (command == "GET_PROJECT_SETTING") {
                     if (parts.size() > 1) {
                        _handle_get_project_setting_command(parts[1].strip_edges());
                    } else {
                        if (chat_history) chat_history->append_text("System: Error - GET_PROJECT_SETTING command missing setting name.\n");
                    }
                } else if (command == "LIST_SCENE_FILES") {
                    _handle_list_scene_files_command();
                }
                // Else, not a recognized command, could be a plain text response
            }
        }
    }
}

void AIChatInterfacePlugin::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_PROCESS: {
            if (ws_client.is_valid() && ws_client->get_connection_status() != WebSocketPeer::CONNECTION_DISCONNECTED) {
                ws_client->poll();
            }
        } break;
    }
}

void AIChatInterfacePlugin::_bind_methods() {
    ClassDB::bind_method(D_METHOD("_chat_button_pressed"), &AIChatInterfacePlugin::_chat_button_pressed);
    ClassDB::bind_method(D_METHOD("_send_button_pressed"), &AIChatInterfacePlugin::_send_button_pressed);
}

void AIChatInterfacePlugin::_enter_tree() {
    set_process_notification(true);

    chat_button = memnew(Button);
    chat_button->set_text("AI Chat");
    chat_button->connect("pressed", callable_mp(this, &AIChatInterfacePlugin::_chat_button_pressed));
    add_control_to_header_menu(chat_button);

    chat_dock = memnew(PanelContainer);
    chat_dock->set_name("AIChat");
    chat_dock->set_custom_minimum_size(Size2(250, 350) * EDSCALE);

    VBoxContainer *chat_vbox = memnew(VBoxContainer);
    chat_dock->add_child(chat_vbox);

    chat_history = memnew(TextEdit);
    chat_history->set_v_size_flags(Control::SIZE_EXPAND_FILL);
    chat_history->set_readonly(true);
    chat_history->set_focus_mode(Control::FOCUS_CLICK);
    chat_vbox->add_child(chat_history);

    HBoxContainer *input_hbox = memnew(HBoxContainer);
    chat_vbox->add_child(input_hbox);

    message_input = memnew(LineEdit);
    message_input->set_h_size_flags(Control::SIZE_EXPAND_FILL);
    input_hbox->add_child(message_input);
    message_input->connect("text_submitted", callable_mp(this, &AIChatInterfacePlugin::_send_button_pressed));

    send_button = memnew(Button);
    send_button->set_text("Send");
    input_hbox->add_child(send_button);
    send_button->connect("pressed", callable_mp(this, &AIChatInterfacePlugin::_send_button_pressed));

    add_control_to_dock(EditorDockManager::DOCK_SLOT_RIGHT_UL, chat_dock);
    chat_dock->hide();

    if (ws_client.is_valid()) {
        ws_client->connect("connection_established", callable_mp(this, &AIChatInterfacePlugin::_on_websocket_connected));
        ws_client->connect("connection_error", callable_mp(this, &AIChatInterfacePlugin::_on_websocket_error));
        ws_client->connect("connection_closed", callable_mp(this, &AIChatInterfacePlugin::_on_websocket_connection_closed));
        ws_client->connect("data_received", callable_mp(this, &AIChatInterfacePlugin::_on_websocket_data_received));

        Error err = ws_client->connect_to_url(CLOUDFLARE_WEBSOCKET_URL);
        if (err != OK) {
            if (chat_history) {
                chat_history->append_text("System: Failed to connect to WebSocket URL.\n");
            }
        } else {
             if (chat_history) {
                chat_history->append_text("System: Attempting to connect to Cloudflare...\n");
            }
        }
    }
}

void AIChatInterfacePlugin::_exit_tree() {
    set_process_notification(false);

    if (chat_button) {
        remove_control_from_header_menu(chat_button);
        memdelete(chat_button);
        chat_button = nullptr;
    }
    if (chat_dock) {
        remove_control_from_docks(chat_dock);
        memdelete(chat_dock);
        chat_dock = nullptr;
        chat_history = nullptr;
        message_input = nullptr;
        send_button = nullptr;
    }

    if (ws_client.is_valid()) {
        if (ws_client->get_connection_status() != WebSocketPeer::CONNECTION_DISCONNECTED) {
            ws_client->disconnect_from_host();
        }
    }
}

void AIChatInterfacePlugin::_process(double p_delta) {
    // Using _notification(NOTIFICATION_PROCESS) for polling.
}
