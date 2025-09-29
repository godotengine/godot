/**************************************************************************/
/*  ai_agent.h                                                            */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#pragma once

#include "scene/main/node.h"
#include "core/object/script_language.h"
#include "gemini_client.h"
#include "ai_conversation.h"
#include "typescript/typescript_runner.h"

class AIAgent : public Node {
    GDCLASS(AIAgent, Node);

public:
    enum AgentState {
        AGENT_STATE_IDLE,
        AGENT_STATE_THINKING,
        AGENT_STATE_EXECUTING,
        AGENT_STATE_WAITING,
        AGENT_STATE_ERROR,
    };

private:
    Ref<GeminiClient> gemini_client;
    Ref<AIConversation> conversation;
    Ref<TypeScriptRunner> typescript_runner;
    
    AgentState current_state;
    String agent_name;
    String agent_role;
    String current_task;
    Dictionary agent_memory;
    Array available_actions;
    
    // Configuration
    bool auto_execute;
    float thinking_delay;
    int max_conversation_turns;
    
    // Internal state
    Timer *thinking_timer;
    bool is_processing;
    String last_user_input;
    String pending_script;
    
    void _setup_default_actions();
    void _process_ai_response(const String &response);
    void _execute_action(const Dictionary &action);
    void _on_thinking_timer_timeout();
    void _on_gemini_response(const String &response);
    void _on_gemini_error(const String &error);

protected:
    static void _bind_methods();
    void _notification(int p_what);

public:
    AIAgent();
    ~AIAgent();

    // Configuration methods
    void set_agent_name(const String &p_name);
    String get_agent_name() const;
    
    void set_agent_role(const String &p_role);
    String get_agent_role() const;
    
    void set_gemini_api_key(const String &p_api_key);
    String get_gemini_api_key() const;
    
    void set_auto_execute(bool p_auto_execute);
    bool get_auto_execute() const;
    
    void set_thinking_delay(float p_delay);
    float get_thinking_delay() const;
    
    void set_max_conversation_turns(int p_max_turns);
    int get_max_conversation_turns() const;

    // Core AI functionality
    void send_message(const String &message);
    void send_message_with_context(const String &message, const Dictionary &context);
    void execute_typescript_code(const String &code);
    void load_typescript_file(const String &file_path);
    
    // Action system
    void add_action(const String &name, const Callable &callback, const String &description = "");
    void remove_action(const String &name);
    Array get_available_actions() const;
    
    // Memory system
    void remember(const String &key, const Variant &value);
    Variant recall(const String &key) const;
    void forget(const String &key);
    void clear_memory();
    Dictionary get_memory() const;
    
    // State management
    AgentState get_agent_state() const;
    String get_current_task() const;
    void set_current_task(const String &p_task);
    
    // Conversation management
    void clear_conversation();
    Array get_conversation_history() const;
    
    // Utility methods
    void start_thinking();
    void stop_thinking();
    bool is_agent_busy() const;
    void reset_agent();

    // Getters for sub-components
    Ref<GeminiClient> get_gemini_client() const;
    Ref<AIConversation> get_conversation() const;
    Ref<TypeScriptRunner> get_typescript_runner() const;
};

VARIANT_ENUM_CAST(AIAgent::AgentState)