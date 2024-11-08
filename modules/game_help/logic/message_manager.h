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
    void register_message(const StringName &p_message, const Callable &p_callable);
    void unregister_message(const StringName &p_message, const Callable &p_callable);
    void clear();

    HashMap<StringName, List<Callable>> messages;
};
