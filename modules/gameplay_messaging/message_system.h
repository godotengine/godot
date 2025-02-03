// message_System.h
#ifndef MESSAGE_SYSTEM_H
#define MESSAGE_SYSTEM_H

#include "message_types.h"
#include "scene/main/node.h"
#include "core/templates/hash_map.h"
#include "core/templates/vector.h"

class MessageSystem : public Object {
	GDCLASS(MessageSystem, Object);

private:
	static MessageSystem* singleton;
public:

	static MessageSystem* get_singleton() {
		return singleton;

	}

	MessageSystem() { singleton = this; }

protected:
	static void _bind_methods();

private:
	struct ListenerEntry {
		Callable callback;
		bool partial_match;
		int32_t handle_id;
	};

	HashMap<StringName, Vector<ListenerEntry>> listeners;
	HashMap<StringName, int32_t> next_handle_ids;

public:

	void broadcast_message(const StringName& channel, const Variant& message);
	Ref<MessageListenerHandle> register_listener(const StringName& channel, const Callable& callback, bool partial_match = false);
	void unregister_listener(const Ref<MessageListenerHandle>& handle);

private:
	void _dispatch_to_channel(const StringName& broadcast_channel, const StringName& listener_channel, const Variant& message, const Vector<ListenerEntry>& channel_listeners);
};

#endif // MESSAGE_SYSTEM_H
