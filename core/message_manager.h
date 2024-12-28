#pragma once
#include "core/object/ref_counted.h"
class MessageManager : public Object
{
    GDCLASS(MessageManager, Object);
    static void _bind_methods();
public:
	static MessageManager* singleton;
    
    MessageManager();
    ~MessageManager();
    static MessageManager* get_singleton()
    {
        return singleton;
    }

    void emit(const StringName &p_message, Array p_args);
    void emit_deferred(const StringName &p_message, Array p_args);
    
    void emit_enum(int64_t p_message, Array p_args);
    void emit_enum_deferred(int64_t p_message, Array p_args);

    void process(int p_max_count = 10);
    void register_message(const StringName &p_message, const Callable &p_callable);
    void unregister_message(const StringName &p_message, const Callable &p_callable);


    void register_enum_message(int64_t p_message, const Callable &p_callable);
    void unregister_enum_message(int64_t p_message, const Callable &p_callable);

    void clear_messages();
    void clear();

    HashMap<StringName, List<Callable>> messages;
    HashMap<int64_t, List<Callable>> enum_messages;

    List<Pair<StringName,Array>> deferred_messages_list;
    List<Pair<int64_t,Array>> deferred_enum_messages_list;
};
