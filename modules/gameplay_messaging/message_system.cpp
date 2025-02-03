// message_system.cpp
#include "message_system.h"

MessageSystem *MessageSystem::singleton;


void MessageSystem::_bind_methods() {
    ClassDB::bind_method(D_METHOD("broadcast_message", "channel", "message"), &MessageSystem::broadcast_message);
    ClassDB::bind_method(D_METHOD("register_listener", "channel", "callback", "partial_match"), &MessageSystem::register_listener, DEFVAL(false));
    ClassDB::bind_method(D_METHOD("unregister_listener", "handle"), &MessageSystem::unregister_listener);
}

void MessageSystem::broadcast_message(const StringName& channel, const Variant& message) {
    StringName current_channel = channel;

    while (!current_channel.is_empty()) {
        HashMap<StringName, Vector<ListenerEntry>>::Iterator it = listeners.find(current_channel);
        if (it) {
            _dispatch_to_channel(channel, current_channel, message, it->value);
        }

        // Get parent channel by removing the last segment
        String channel_str = String(current_channel);
        int last_dot = channel_str.rfind(".");
        if (last_dot == -1) {
            break;
        }
        current_channel = StringName(channel_str.substr(0, last_dot));
    }
}

void MessageSystem::_dispatch_to_channel(
    const StringName& broadcast_channel,
    const StringName& listener_channel,
    const Variant& message,
    const Vector<ListenerEntry>& channel_listeners
) {
    // Make a copy of listeners to prevent issues if callbacks modify the list
    Vector<ListenerEntry> current_listeners = channel_listeners;

    for (const ListenerEntry& listener : current_listeners) {
        if (listener_channel == broadcast_channel || listener.partial_match) {
            Array args;
            args.push_back(broadcast_channel);
            args.push_back(message);
            listener.callback.callv(args);
        }
    }
}

Ref<MessageListenerHandle> MessageSystem::register_listener(
    const StringName& channel,
    const Callable& callback,
    bool partial_match
) {
    if (!listeners.has(channel)) {
        listeners[channel] = Vector<ListenerEntry>();
        next_handle_ids[channel] = 0;
    }

    int32_t handle_id = ++next_handle_ids[channel];

    ListenerEntry entry;
    entry.callback = callback;
    entry.partial_match = partial_match;
    entry.handle_id = handle_id;

    listeners[channel].push_back(entry);
    return Ref<MessageListenerHandle>(memnew(MessageListenerHandle(channel, handle_id)));
}

void MessageSystem::unregister_listener(const Ref<MessageListenerHandle>& handle) {
    if (!handle.is_valid() || !handle->is_valid()) {
        return;
    }

    auto channel = handle->get_channel();
    auto handle_id = handle->get_id();

    HashMap<StringName, Vector<ListenerEntry>>::Iterator it = listeners.find(channel);
    if (it) {
        Vector<ListenerEntry>& channel_listeners = it->value;
        for (int i = 0; i < channel_listeners.size(); i++) {
            if (channel_listeners[i].handle_id == handle_id) {
                channel_listeners.remove_at(i);
                break;
            }
        }

        if (channel_listeners.is_empty()) {
            listeners.erase(channel);
        }
    }

    handle->invalidate();
}
