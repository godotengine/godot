/**************************************************************************/
/*  ai_conversation.h                                                     */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/

#pragma once

#include "core/object/ref_counted.h"
#include "core/io/file_access.h"

class AIConversation : public RefCounted {
    GDCLASS(AIConversation, RefCounted);

public:
    enum MessageRole {
        ROLE_USER,
        ROLE_ASSISTANT,
        ROLE_SYSTEM,
    };

private:
    Array messages;
    String conversation_id;
    int max_history;
    bool auto_save;
    String save_path;
    
    Dictionary _create_message(MessageRole role, const String &content, const Dictionary &metadata = Dictionary());
    void _trim_history();
    String _role_to_string(MessageRole role) const;
    MessageRole _string_to_role(const String &role_str) const;

protected:
    static void _bind_methods();

public:
    AIConversation();
    ~AIConversation();

    // Message management
    void add_message(MessageRole role, const String &content, const Dictionary &metadata = Dictionary());
    void add_user_message(const String &content);
    void add_assistant_message(const String &content);
    void add_system_message(const String &content);
    
    Dictionary get_message(int index) const;
    Array get_messages() const;
    int get_message_count() const;
    void clear_messages();
    void remove_message(int index);
    
    // Conversation properties
    void set_conversation_id(const String &p_id);
    String get_conversation_id() const;
    
    void set_max_history(int p_max_history);
    int get_max_history() const;
    
    void set_auto_save(bool p_auto_save);
    bool get_auto_save() const;
    
    void set_save_path(const String &p_save_path);
    String get_save_path() const;
    
    // Search and filtering
    Array find_messages_by_role(MessageRole role) const;
    Array find_messages_containing(const String &text) const;
    Dictionary get_last_message() const;
    Dictionary get_last_message_by_role(MessageRole role) const;
    
    // Persistence
    Error save_to_file(const String &file_path = "");
    Error load_from_file(const String &file_path);
    Dictionary to_dictionary() const;
    void from_dictionary(const Dictionary &data);
    
    // Context management
    Array get_context_window(int window_size) const;
    String get_formatted_conversation(const String &format = "simple") const;
    
    // Statistics
    Dictionary get_conversation_stats() const;
};

VARIANT_ENUM_CAST(AIConversation::MessageRole)