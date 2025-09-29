/**************************************************************************/
/*  ai_conversation.cpp                                                   */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#include "ai_conversation.h"

#include "core/io/json.h"
#include "core/os/time.h"

AIConversation::AIConversation() {
    conversation_id = "";
    max_history = 100;
    auto_save = false;
    save_path = "";
}

AIConversation::~AIConversation() {
}

void AIConversation::_bind_methods() {
    // Message management
    ClassDB::bind_method(D_METHOD("add_message", "role", "content", "metadata"), &AIConversation::add_message, DEFVAL(Dictionary()));
    ClassDB::bind_method(D_METHOD("add_user_message", "content"), &AIConversation::add_user_message);
    ClassDB::bind_method(D_METHOD("add_assistant_message", "content"), &AIConversation::add_assistant_message);
    ClassDB::bind_method(D_METHOD("add_system_message", "content"), &AIConversation::add_system_message);
    
    ClassDB::bind_method(D_METHOD("get_message", "index"), &AIConversation::get_message);
    ClassDB::bind_method(D_METHOD("get_messages"), &AIConversation::get_messages);
    ClassDB::bind_method(D_METHOD("get_message_count"), &AIConversation::get_message_count);
    ClassDB::bind_method(D_METHOD("clear_messages"), &AIConversation::clear_messages);
    ClassDB::bind_method(D_METHOD("remove_message", "index"), &AIConversation::remove_message);
    
    // Properties
    ClassDB::bind_method(D_METHOD("set_conversation_id", "conversation_id"), &AIConversation::set_conversation_id);
    ClassDB::bind_method(D_METHOD("get_conversation_id"), &AIConversation::get_conversation_id);
    
    ClassDB::bind_method(D_METHOD("set_max_history", "max_history"), &AIConversation::set_max_history);
    ClassDB::bind_method(D_METHOD("get_max_history"), &AIConversation::get_max_history);
    
    ClassDB::bind_method(D_METHOD("set_auto_save", "auto_save"), &AIConversation::set_auto_save);
    ClassDB::bind_method(D_METHOD("get_auto_save"), &AIConversation::get_auto_save);
    
    ClassDB::bind_method(D_METHOD("set_save_path", "save_path"), &AIConversation::set_save_path);
    ClassDB::bind_method(D_METHOD("get_save_path"), &AIConversation::get_save_path);
    
    // Search and filtering
    ClassDB::bind_method(D_METHOD("find_messages_by_role", "role"), &AIConversation::find_messages_by_role);
    ClassDB::bind_method(D_METHOD("find_messages_containing", "text"), &AIConversation::find_messages_containing);
    ClassDB::bind_method(D_METHOD("get_last_message"), &AIConversation::get_last_message);
    ClassDB::bind_method(D_METHOD("get_last_message_by_role", "role"), &AIConversation::get_last_message_by_role);
    
    // Persistence
    ClassDB::bind_method(D_METHOD("save_to_file", "file_path"), &AIConversation::save_to_file, DEFVAL(""));
    ClassDB::bind_method(D_METHOD("load_from_file", "file_path"), &AIConversation::load_from_file);
    ClassDB::bind_method(D_METHOD("to_dictionary"), &AIConversation::to_dictionary);
    ClassDB::bind_method(D_METHOD("from_dictionary", "data"), &AIConversation::from_dictionary);
    
    // Context management
    ClassDB::bind_method(D_METHOD("get_context_window", "window_size"), &AIConversation::get_context_window);
    ClassDB::bind_method(D_METHOD("get_formatted_conversation", "format"), &AIConversation::get_formatted_conversation, DEFVAL("simple"));
    
    // Statistics
    ClassDB::bind_method(D_METHOD("get_conversation_stats"), &AIConversation::get_conversation_stats);

    // Properties
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "conversation_id"), "set_conversation_id", "get_conversation_id");
    ADD_PROPERTY(PropertyInfo(Variant::INT, "max_history", PROPERTY_HINT_RANGE, "1,1000,1"), "set_max_history", "get_max_history");
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "auto_save"), "set_auto_save", "get_auto_save");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "save_path", PROPERTY_HINT_GLOBAL_FILE, "*.json"), "set_save_path", "get_save_path");

    // Enums
    BIND_ENUM_CONSTANT(ROLE_USER);
    BIND_ENUM_CONSTANT(ROLE_ASSISTANT);
    BIND_ENUM_CONSTANT(ROLE_SYSTEM);
}

Dictionary AIConversation::_create_message(MessageRole role, const String &content, const Dictionary &metadata) {
    Dictionary message;
    message["role"] = _role_to_string(role);
    message["content"] = content;
    message["timestamp"] = Time::get_unix_time_from_system();
    message["metadata"] = metadata;
    return message;
}

void AIConversation::_trim_history() {
    if (max_history > 0 && messages.size() > max_history) {
        int to_remove = messages.size() - max_history;
        for (int i = 0; i < to_remove; i++) {
            messages.pop_front();
        }
    }
}

String AIConversation::_role_to_string(MessageRole role) const {
    switch (role) {
        case ROLE_USER:
            return "user";
        case ROLE_ASSISTANT:
            return "assistant";
        case ROLE_SYSTEM:
            return "system";
        default:
            return "user";
    }
}

AIConversation::MessageRole AIConversation::_string_to_role(const String &role_str) const {
    if (role_str == "assistant") {
        return ROLE_ASSISTANT;
    } else if (role_str == "system") {
        return ROLE_SYSTEM;
    }
    return ROLE_USER;
}

void AIConversation::add_message(MessageRole role, const String &content, const Dictionary &metadata) {
    Dictionary message = _create_message(role, content, metadata);
    messages.push_back(message);
    _trim_history();
    
    if (auto_save && !save_path.is_empty()) {
        save_to_file();
    }
}

void AIConversation::add_user_message(const String &content) {
    add_message(ROLE_USER, content);
}

void AIConversation::add_assistant_message(const String &content) {
    add_message(ROLE_ASSISTANT, content);
}

void AIConversation::add_system_message(const String &content) {
    add_message(ROLE_SYSTEM, content);
}

Dictionary AIConversation::get_message(int index) const {
    if (index >= 0 && index < messages.size()) {
        return messages[index];
    }
    return Dictionary();
}

Array AIConversation::get_messages() const {
    return messages;
}

int AIConversation::get_message_count() const {
    return messages.size();
}

void AIConversation::clear_messages() {
    messages.clear();
    
    if (auto_save && !save_path.is_empty()) {
        save_to_file();
    }
}

void AIConversation::remove_message(int index) {
    if (index >= 0 && index < messages.size()) {
        messages.remove_at(index);
        
        if (auto_save && !save_path.is_empty()) {
            save_to_file();
        }
    }
}

void AIConversation::set_conversation_id(const String &p_id) {
    conversation_id = p_id;
}

String AIConversation::get_conversation_id() const {
    return conversation_id;
}

void AIConversation::set_max_history(int p_max_history) {
    max_history = MAX(1, p_max_history);
    _trim_history();
}

int AIConversation::get_max_history() const {
    return max_history;
}

void AIConversation::set_auto_save(bool p_auto_save) {
    auto_save = p_auto_save;
}

bool AIConversation::get_auto_save() const {
    return auto_save;
}

void AIConversation::set_save_path(const String &p_save_path) {
    save_path = p_save_path;
}

String AIConversation::get_save_path() const {
    return save_path;
}

Array AIConversation::find_messages_by_role(MessageRole role) const {
    Array filtered_messages;
    String role_str = _role_to_string(role);
    
    for (int i = 0; i < messages.size(); i++) {
        Dictionary message = messages[i];
        if (message.get("role", "") == role_str) {
            filtered_messages.push_back(message);
        }
    }
    
    return filtered_messages;
}

Array AIConversation::find_messages_containing(const String &text) const {
    Array filtered_messages;
    String lower_text = text.to_lower();
    
    for (int i = 0; i < messages.size(); i++) {
        Dictionary message = messages[i];
        String content = message.get("content", "");
        if (content.to_lower().contains(lower_text)) {
            filtered_messages.push_back(message);
        }
    }
    
    return filtered_messages;
}

Dictionary AIConversation::get_last_message() const {
    if (messages.size() > 0) {
        return messages[messages.size() - 1];
    }
    return Dictionary();
}

Dictionary AIConversation::get_last_message_by_role(MessageRole role) const {
    String role_str = _role_to_string(role);
    
    for (int i = messages.size() - 1; i >= 0; i--) {
        Dictionary message = messages[i];
        if (message.get("role", "") == role_str) {
            return message;
        }
    }
    
    return Dictionary();
}

Error AIConversation::save_to_file(const String &file_path) {
    String path = file_path.is_empty() ? save_path : file_path;
    if (path.is_empty()) {
        return ERR_FILE_NOT_FOUND;
    }
    
    Dictionary data = to_dictionary();
    JSON json;
    String json_string = json.stringify(data);
    
    Ref<FileAccess> file = FileAccess::open(path, FileAccess::WRITE);
    if (file.is_null()) {
        return ERR_CANT_OPEN;
    }
    
    file->store_string(json_string);
    file->close();
    
    return OK;
}

Error AIConversation::load_from_file(const String &file_path) {
    Ref<FileAccess> file = FileAccess::open(file_path, FileAccess::READ);
    if (file.is_null()) {
        return ERR_CANT_OPEN;
    }
    
    String json_string = file->get_as_text();
    file->close();
    
    JSON json;
    Error err = json.parse(json_string);
    if (err != OK) {
        return err;
    }
    
    Dictionary data = json.data;
    from_dictionary(data);
    
    save_path = file_path;
    return OK;
}

Dictionary AIConversation::to_dictionary() const {
    Dictionary data;
    data["conversation_id"] = conversation_id;
    data["max_history"] = max_history;
    data["auto_save"] = auto_save;
    data["save_path"] = save_path;
    data["messages"] = messages;
    data["created_at"] = Time::get_unix_time_from_system();
    return data;
}

void AIConversation::from_dictionary(const Dictionary &data) {
    conversation_id = data.get("conversation_id", "");
    max_history = data.get("max_history", 100);
    auto_save = data.get("auto_save", false);
    save_path = data.get("save_path", "");
    messages = data.get("messages", Array());
}

Array AIConversation::get_context_window(int window_size) const {
    Array context;
    int start_index = MAX(0, messages.size() - window_size);
    
    for (int i = start_index; i < messages.size(); i++) {
        context.push_back(messages[i]);
    }
    
    return context;
}

String AIConversation::get_formatted_conversation(const String &format) const {
    String formatted = "";
    
    if (format == "markdown") {
        for (int i = 0; i < messages.size(); i++) {
            Dictionary message = messages[i];
            String role = message.get("role", "");
            String content = message.get("content", "");
            
            if (role == "user") {
                formatted += "**User:** " + content + "\n\n";
            } else if (role == "assistant") {
                formatted += "**Assistant:** " + content + "\n\n";
            } else if (role == "system") {
                formatted += "*System: " + content + "*\n\n";
            }
        }
    } else {
        // Simple format
        for (int i = 0; i < messages.size(); i++) {
            Dictionary message = messages[i];
            String role = message.get("role", "");
            String content = message.get("content", "");
            formatted += role.capitalize() + ": " + content + "\n";
        }
    }
    
    return formatted;
}

Dictionary AIConversation::get_conversation_stats() const {
    Dictionary stats;
    stats["total_messages"] = messages.size();
    
    int user_count = 0;
    int assistant_count = 0;
    int system_count = 0;
    int total_characters = 0;
    
    for (int i = 0; i < messages.size(); i++) {
        Dictionary message = messages[i];
        String role = message.get("role", "");
        String content = message.get("content", "");
        
        if (role == "user") {
            user_count++;
        } else if (role == "assistant") {
            assistant_count++;
        } else if (role == "system") {
            system_count++;
        }
        
        total_characters += content.length();
    }
    
    stats["user_messages"] = user_count;
    stats["assistant_messages"] = assistant_count;
    stats["system_messages"] = system_count;
    stats["total_characters"] = total_characters;
    stats["average_message_length"] = messages.size() > 0 ? total_characters / messages.size() : 0;
    
    return stats;
}