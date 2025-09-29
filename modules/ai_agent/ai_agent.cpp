/**************************************************************************/
/*  ai_agent.cpp                                                          */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#include "ai_agent.h"

#include "scene/main/timer.h"
#include "core/io/json.h"

AIAgent::AIAgent() {
    current_state = AGENT_STATE_IDLE;
    agent_name = "AI Agent";
    agent_role = "Assistant";
    auto_execute = false;
    thinking_delay = 1.0f;
    max_conversation_turns = 50;
    is_processing = false;
    
    // Initialize components
    gemini_client.instantiate();
    conversation.instantiate();
    typescript_runner.instantiate();
    
    thinking_timer = memnew(Timer);
    thinking_timer->set_wait_time(thinking_delay);
    thinking_timer->set_one_shot(true);
    add_child(thinking_timer);
    thinking_timer->connect("timeout", callable_mp(this, &AIAgent::_on_thinking_timer_timeout));
    
    _setup_default_actions();
}

AIAgent::~AIAgent() {
}

void AIAgent::_bind_methods() {
    // Configuration methods
    ClassDB::bind_method(D_METHOD("set_agent_name", "name"), &AIAgent::set_agent_name);
    ClassDB::bind_method(D_METHOD("get_agent_name"), &AIAgent::get_agent_name);
    
    ClassDB::bind_method(D_METHOD("set_agent_role", "role"), &AIAgent::set_agent_role);
    ClassDB::bind_method(D_METHOD("get_agent_role"), &AIAgent::get_agent_role);
    
    ClassDB::bind_method(D_METHOD("set_gemini_api_key", "api_key"), &AIAgent::set_gemini_api_key);
    ClassDB::bind_method(D_METHOD("get_gemini_api_key"), &AIAgent::get_gemini_api_key);
    
    ClassDB::bind_method(D_METHOD("set_auto_execute", "auto_execute"), &AIAgent::set_auto_execute);
    ClassDB::bind_method(D_METHOD("get_auto_execute"), &AIAgent::get_auto_execute);
    
    ClassDB::bind_method(D_METHOD("set_thinking_delay", "delay"), &AIAgent::set_thinking_delay);
    ClassDB::bind_method(D_METHOD("get_thinking_delay"), &AIAgent::get_thinking_delay);
    
    ClassDB::bind_method(D_METHOD("set_max_conversation_turns", "max_turns"), &AIAgent::set_max_conversation_turns);
    ClassDB::bind_method(D_METHOD("get_max_conversation_turns"), &AIAgent::get_max_conversation_turns);

    // Core AI functionality
    ClassDB::bind_method(D_METHOD("send_message", "message"), &AIAgent::send_message);
    ClassDB::bind_method(D_METHOD("send_message_with_context", "message", "context"), &AIAgent::send_message_with_context);
    ClassDB::bind_method(D_METHOD("execute_typescript_code", "code"), &AIAgent::execute_typescript_code);
    ClassDB::bind_method(D_METHOD("load_typescript_file", "file_path"), &AIAgent::load_typescript_file);
    
    // Action system
    ClassDB::bind_method(D_METHOD("add_action", "name", "callback", "description"), &AIAgent::add_action, DEFVAL(""));
    ClassDB::bind_method(D_METHOD("remove_action", "name"), &AIAgent::remove_action);
    ClassDB::bind_method(D_METHOD("get_available_actions"), &AIAgent::get_available_actions);
    
    // Memory system
    ClassDB::bind_method(D_METHOD("remember", "key", "value"), &AIAgent::remember);
    ClassDB::bind_method(D_METHOD("recall", "key"), &AIAgent::recall);
    ClassDB::bind_method(D_METHOD("forget", "key"), &AIAgent::forget);
    ClassDB::bind_method(D_METHOD("clear_memory"), &AIAgent::clear_memory);
    ClassDB::bind_method(D_METHOD("get_memory"), &AIAgent::get_memory);
    
    // State management
    ClassDB::bind_method(D_METHOD("get_agent_state"), &AIAgent::get_agent_state);
    ClassDB::bind_method(D_METHOD("get_current_task"), &AIAgent::get_current_task);
    ClassDB::bind_method(D_METHOD("set_current_task", "task"), &AIAgent::set_current_task);
    
    // Conversation management
    ClassDB::bind_method(D_METHOD("clear_conversation"), &AIAgent::clear_conversation);
    ClassDB::bind_method(D_METHOD("get_conversation_history"), &AIAgent::get_conversation_history);
    
    // Utility methods
    ClassDB::bind_method(D_METHOD("start_thinking"), &AIAgent::start_thinking);
    ClassDB::bind_method(D_METHOD("stop_thinking"), &AIAgent::stop_thinking);
    ClassDB::bind_method(D_METHOD("is_agent_busy"), &AIAgent::is_agent_busy);
    ClassDB::bind_method(D_METHOD("reset_agent"), &AIAgent::reset_agent);

    // Getters for sub-components
    ClassDB::bind_method(D_METHOD("get_gemini_client"), &AIAgent::get_gemini_client);
    ClassDB::bind_method(D_METHOD("get_conversation"), &AIAgent::get_conversation);
    ClassDB::bind_method(D_METHOD("get_typescript_runner"), &AIAgent::get_typescript_runner);

    // Properties
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "agent_name"), "set_agent_name", "get_agent_name");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "agent_role"), "set_agent_role", "get_agent_role");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_execute"), "set_auto_execute", "get_auto_execute");
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT, "thinking_delay", PROPERTY_HINT_RANGE, "0.1,10.0,0.1"), "set_thinking_delay", "get_thinking_delay");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_conversation_turns", PROPERTY_HINT_RANGE, "1,200,1"), "set_max_conversation_turns", "get_max_conversation_turns");

    // Enums
    BIND_ENUM_CONSTANT(AGENT_STATE_IDLE);
    BIND_ENUM_CONSTANT(AGENT_STATE_THINKING);
    BIND_ENUM_CONSTANT(AGENT_STATE_EXECUTING);
    BIND_ENUM_CONSTANT(AGENT_STATE_WAITING);
    BIND_ENUM_CONSTANT(AGENT_STATE_ERROR);

    // Signals
    ADD_SIGNAL(MethodInfo("response_received", PropertyInfo(Variant::STRING, "response")));
    ADD_SIGNAL(MethodInfo("error_occurred", PropertyInfo(Variant::STRING, "error")));
    ADD_SIGNAL(MethodInfo("state_changed", PropertyInfo(Variant::INT, "new_state")));
    ADD_SIGNAL(MethodInfo("thinking_started"));
    ADD_SIGNAL(MethodInfo("thinking_finished"));
}

void AIAgent::_notification(int p_what) {
    switch (p_what) {
        case NOTIFICATION_READY: {
            if (gemini_client.is_valid()) {
                gemini_client->connect("response_received", callable_mp(this, &AIAgent::_on_gemini_response));
                gemini_client->connect("error_occurred", callable_mp(this, &AIAgent::_on_gemini_error));
            }
        } break;
        
        case NOTIFICATION_PROCESS: {
            if (gemini_client.is_valid()) {
                gemini_client->poll();
            }
        } break;
    }
}

void AIAgent::_setup_default_actions() {
    Dictionary log_action;
    log_action["name"] = "log";
    log_action["description"] = "Log a message to the console";
    log_action["callback"] = callable_mp(this, &AIAgent::_execute_action);
    available_actions.push_back(log_action);
    
    Dictionary wait_action;
    wait_action["name"] = "wait";
    wait_action["description"] = "Wait for a specified amount of time";
    wait_action["callback"] = callable_mp(this, &AIAgent::_execute_action);
    available_actions.push_back(wait_action);
}

void AIAgent::_process_ai_response(const String &response) {
    // Simple response processing - in a full implementation, this would parse 
    // structured responses and execute actions
    current_state = AGENT_STATE_IDLE;
    emit_signal("response_received", response);
    emit_signal("state_changed", current_state);
}

void AIAgent::_execute_action(const Dictionary &action) {
    String action_name = action.get("name", "");
    
    if (action_name == "log") {
        String message = action.get("message", "AI Agent executing action");
        print_line("[AI Agent] " + message);
    } else if (action_name == "wait") {
        float wait_time = action.get("time", 1.0f);
        thinking_timer->set_wait_time(wait_time);
        thinking_timer->start();
        current_state = AGENT_STATE_WAITING;
        emit_signal("state_changed", current_state);
    }
}

void AIAgent::_on_thinking_timer_timeout() {
    if (current_state == AGENT_STATE_THINKING) {
        emit_signal("thinking_finished");
    }
    current_state = AGENT_STATE_IDLE;
    emit_signal("state_changed", current_state);
}

void AIAgent::_on_gemini_response(const String &response) {
    _process_ai_response(response);
}

void AIAgent::_on_gemini_error(const String &error) {
    current_state = AGENT_STATE_ERROR;
    emit_signal("error_occurred", error);
    emit_signal("state_changed", current_state);
}

// Configuration methods
void AIAgent::set_agent_name(const String &p_name) {
    agent_name = p_name;
}

String AIAgent::get_agent_name() const {
    return agent_name;
}

void AIAgent::set_agent_role(const String &p_role) {
    agent_role = p_role;
    if (gemini_client.is_valid()) {
        gemini_client->set_system_prompt("You are " + agent_role + " named " + agent_name);
    }
}

String AIAgent::get_agent_role() const {
    return agent_role;
}

void AIAgent::set_gemini_api_key(const String &p_api_key) {
    if (gemini_client.is_valid()) {
        gemini_client->set_api_key(p_api_key);
    }
}

String AIAgent::get_gemini_api_key() const {
    if (gemini_client.is_valid()) {
        return gemini_client->get_api_key();
    }
    return "";
}

void AIAgent::set_auto_execute(bool p_auto_execute) {
    auto_execute = p_auto_execute;
}

bool AIAgent::get_auto_execute() const {
    return auto_execute;
}

void AIAgent::set_thinking_delay(float p_delay) {
    thinking_delay = MAX(0.1f, p_delay);
    if (thinking_timer) {
        thinking_timer->set_wait_time(thinking_delay);
    }
}

float AIAgent::get_thinking_delay() const {
    return thinking_delay;
}

void AIAgent::set_max_conversation_turns(int p_max_turns) {
    max_conversation_turns = MAX(1, p_max_turns);
    if (conversation.is_valid()) {
        conversation->set_max_history(max_conversation_turns);
    }
}

int AIAgent::get_max_conversation_turns() const {
    return max_conversation_turns;
}

// Core AI functionality
void AIAgent::send_message(const String &message) {
    if (is_processing) {
        return;
    }
    
    is_processing = true;
    last_user_input = message;
    current_state = AGENT_STATE_THINKING;
    
    if (conversation.is_valid()) {
        conversation->add_user_message(message);
    }
    
    if (gemini_client.is_valid()) {
        gemini_client->send_message(message);
    }
    
    emit_signal("state_changed", current_state);
}

void AIAgent::send_message_with_context(const String &message, const Dictionary &context) {
    // Store context in memory
    for (int i = 0; i < context.size(); i++) {
        Array keys = context.keys();
        for (int j = 0; j < keys.size(); j++) {
            agent_memory[keys[j]] = context[keys[j]];
        }
    }
    
    send_message(message);
}

void AIAgent::execute_typescript_code(const String &code) {
    if (typescript_runner.is_valid()) {
        current_state = AGENT_STATE_EXECUTING;
        emit_signal("state_changed", current_state);
        
        Error err = typescript_runner->execute_typescript_code(code, this);
        if (err != OK) {
            current_state = AGENT_STATE_ERROR;
            emit_signal("error_occurred", "Failed to execute TypeScript code");
        } else {
            current_state = AGENT_STATE_IDLE;
        }
        emit_signal("state_changed", current_state);
    }
}

void AIAgent::load_typescript_file(const String &file_path) {
    if (typescript_runner.is_valid()) {
        current_state = AGENT_STATE_EXECUTING;
        emit_signal("state_changed", current_state);
        
        Error err = typescript_runner->execute_typescript_file(file_path, this);
        if (err != OK) {
            current_state = AGENT_STATE_ERROR;
            emit_signal("error_occurred", "Failed to load TypeScript file: " + file_path);
        } else {
            current_state = AGENT_STATE_IDLE;
        }
        emit_signal("state_changed", current_state);
    }
}

// Action system
void AIAgent::add_action(const String &name, const Callable &callback, const String &description) {
    Dictionary action;
    action["name"] = name;
    action["callback"] = callback;
    action["description"] = description;
    available_actions.push_back(action);
}

void AIAgent::remove_action(const String &name) {
    for (int i = 0; i < available_actions.size(); i++) {
        Dictionary action = available_actions[i];
        if (action.get("name", "") == name) {
            available_actions.remove_at(i);
            break;
        }
    }
}

Array AIAgent::get_available_actions() const {
    return available_actions;
}

// Memory system
void AIAgent::remember(const String &key, const Variant &value) {
    agent_memory[key] = value;
}

Variant AIAgent::recall(const String &key) const {
    return agent_memory.get(key, Variant());
}

void AIAgent::forget(const String &key) {
    agent_memory.erase(key);
}

void AIAgent::clear_memory() {
    agent_memory.clear();
}

Dictionary AIAgent::get_memory() const {
    return agent_memory;
}

// State management
AIAgent::AgentState AIAgent::get_agent_state() const {
    return current_state;
}

String AIAgent::get_current_task() const {
    return current_task;
}

void AIAgent::set_current_task(const String &p_task) {
    current_task = p_task;
}

// Conversation management
void AIAgent::clear_conversation() {
    if (conversation.is_valid()) {
        conversation->clear_messages();
    }
}

Array AIAgent::get_conversation_history() const {
    if (conversation.is_valid()) {
        return conversation->get_messages();
    }
    return Array();
}

// Utility methods
void AIAgent::start_thinking() {
    current_state = AGENT_STATE_THINKING;
    emit_signal("thinking_started");
    emit_signal("state_changed", current_state);
    thinking_timer->start();
}

void AIAgent::stop_thinking() {
    if (current_state == AGENT_STATE_THINKING) {
        thinking_timer->stop();
        current_state = AGENT_STATE_IDLE;
        emit_signal("thinking_finished");
        emit_signal("state_changed", current_state);
    }
}

bool AIAgent::is_agent_busy() const {
    return current_state != AGENT_STATE_IDLE;
}

void AIAgent::reset_agent() {
    stop_thinking();
    clear_conversation();
    clear_memory();
    current_task = "";
    is_processing = false;
    current_state = AGENT_STATE_IDLE;
    emit_signal("state_changed", current_state);
}

// Getters for sub-components
Ref<GeminiClient> AIAgent::get_gemini_client() const {
    return gemini_client;
}

Ref<AIConversation> AIAgent::get_conversation() const {
    return conversation;
}

Ref<TypeScriptRunner> AIAgent::get_typescript_runner() const {
    return typescript_runner;
}