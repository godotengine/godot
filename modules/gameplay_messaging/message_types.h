// message_types.h
#ifndef MESSAGE_TYPES_H
#define MESSAGE_TYPES_H

#include "core/object/ref_counted.h"
#include "core/string/string_name.h"
#include "core/variant/variant.h"
#include "core/object/object.h"

using namespace godot;

// Base message listener handle that can be passed to/from GDScript
class MessageListenerHandle : public RefCounted {
    GDCLASS(MessageListenerHandle, RefCounted)

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("is_valid"), &MessageListenerHandle::is_valid);
        ClassDB::bind_method(D_METHOD("get_channel"), &MessageListenerHandle::get_channel);
    }

    StringName channel;
    int32_t id = 0;
    bool valid = false;

public:
    MessageListenerHandle() {}
    MessageListenerHandle(const StringName& p_channel, int32_t p_id)
        : channel(p_channel), id(p_id), valid(true) {}

    bool is_valid() const { return valid; }
    void invalidate() { valid = false; }

    const StringName& get_channel() const { return channel; }
    int32_t get_id() const { return id; }
};

#endif // MESSAGE_TYPES_H
