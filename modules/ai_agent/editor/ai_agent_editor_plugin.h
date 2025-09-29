/**************************************************************************/
/*  ai_agent_editor_plugin.h                                              */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#pragma once

#ifdef TOOLS_ENABLED

#include "editor/editor_plugin.h"
#include "editor/editor_interface.h"
#include "scene/gui/control.h"
#include "scene/gui/split_container.h"
#include "scene/gui/text_edit.h"
#include "scene/gui/button.h"
#include "scene/gui/line_edit.h"
#include "scene/gui/label.h"
#include "scene/gui/option_button.h"
#include "scene/gui/check_box.h"
#include "../ai_agent.h"

class AIAgentDock;

class AIAgentEditorPlugin : public EditorPlugin {
    GDCLASS(AIAgentEditorPlugin, EditorPlugin);

private:
    AIAgentDock *ai_agent_dock;

protected:
    static void _bind_methods();

public:
    AIAgentEditorPlugin();
    ~AIAgentEditorPlugin();

    virtual String get_name() const override;
    virtual void make_visible(bool p_visible) override;
    virtual bool has_main_screen() const override;
    virtual void edit(Object *p_object) override;
    virtual bool handles(Object *p_object) const override;
};

class AIAgentDock : public Control {
    GDCLASS(AIAgentDock, Control);

private:
    // UI Components
    VSplitContainer *main_split;
    VBoxContainer *config_container;
    VBoxContainer *chat_container;
    HBoxContainer *input_container;
    
    // Configuration controls
    LineEdit *api_key_input;
    OptionButton *model_selector;
    CheckBox *auto_execute_checkbox;
    Button *save_config_button;
    Button *load_config_button;
    
    // Chat interface
    TextEdit *conversation_display;
    LineEdit *message_input;
    Button *send_button;
    Button *clear_button;
    
    // Agent controls
    Button *start_agent_button;
    Button *stop_agent_button;
    Label *status_label;
    
    // TypeScript editor
    TextEdit *typescript_editor;
    Button *run_typescript_button;
    Button *compile_typescript_button;
    
    // Current AI agent reference
    AIAgent *current_agent;
    
    void _setup_ui();
    void _on_send_button_pressed();
    void _on_clear_button_pressed();
    void _on_message_input_entered(const String &text);
    void _on_start_agent_pressed();
    void _on_stop_agent_pressed();
    void _on_save_config_pressed();
    void _on_load_config_pressed();
    void _on_run_typescript_pressed();
    void _on_compile_typescript_pressed();
    void _on_agent_response_received(const String &response);
    void _on_agent_error_occurred(const String &error);
    void _update_status();
    void _load_default_config();
    void _apply_config_to_agent();

protected:
    static void _bind_methods();
    void _notification(int p_what);

public:
    AIAgentDock();
    ~AIAgentDock();
    
    void set_current_agent(AIAgent *p_agent);
    AIAgent *get_current_agent() const;
    void refresh_ui();
};

#endif // TOOLS_ENABLED