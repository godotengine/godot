#pragma once

#include "beehave_node.h"

class BeehaveListener : public RefCounted
{
    GDCLASS(BeehaveListener, RefCounted);

    static void _bind_methods()
    {
	    GDVIRTUAL_BIND(_process, "owenr_node", "blackboard");
    }

public:
    virtual void process(Node * actor, Blackboard* blackboard)
    {
        #if !TOOLS_ENABLED
           if(is_editor_only)
           {
               return;
           }
        #endif
        GDVIRTUAL_CALL(_process, actor, blackboard);
    }
	GDVIRTUAL2(_process,Node*,Blackboard*);
    bool is_editor_only = false;

};

class BeehaveTree : public Resource
{
    GDCLASS(BeehaveTree, Resource);
    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_root_node", "root_node"), &BeehaveTree::set_root_node);
        ClassDB::bind_method(D_METHOD("get_root_node"), &BeehaveTree::get_root_node);

        ClassDB::bind_method(D_METHOD("set_listener", "listener"), &BeehaveTree::set_listener);
        ClassDB::bind_method(D_METHOD("get_listener"), &BeehaveTree::get_listener);

        ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "root_node", PROPERTY_HINT_RESOURCE_TYPE, "BeehaveNode"), "set_root_node", "get_root_node");
        ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "listener", PROPERTY_HINT_ARRAY_TYPE, MAKE_RESOURCE_TYPE_HINT("BeehaveListener")), "set_listener", "get_listener");
    }
    enum Status { SUCCESS, FAILURE, RUNNING };
public:
    // 初始化
    void init(Node * actor, Blackboard* blackboard)
    {
        last_tick = 0;
        _can_send_message = false;
        _process_time_metric_value = 0;
        status = RUNNING;
        if(root_node.is_valid())
        {
            root_node->interrupt(actor,blackboard);
        }
    }

    void process(Node * actor, Blackboard* blackboard, double delta)
    {
        if (last_tick < tick_rate - 1)
        {
            last_tick += 1;
            return;

        }
        last_tick = 0;
        float start_time = OS::get_singleton()->get_ticks_usec();

	    blackboard->set_var("can_send_message", _can_send_message);

        // if _can_send_message:
        // 	BeehaveDebuggerMessages.process_begin(get_instance_id())
        tick(actor, blackboard);

        // if _can_send_message:
        // 	BeehaveDebuggerMessages.process_end(get_instance_id())

	    _process_time_metric_value = OS::get_singleton()->get_ticks_usec() - start_time;
    }
    int tick(Node * actor, Blackboard* blackboard)
    {	
        if (actor == nullptr || root_node.is_null())
		{
            return FAILURE;
        }
        if (status != RUNNING)
        {
            root_node->before_run(actor, blackboard);
        }
        status = root_node->tick(actor, blackboard);

        // 	if _can_send_message:
		// BeehaveDebuggerMessages.process_tick(child.get_instance_id(), status)
		// BeehaveDebuggerMessages.process_tick(get_instance_id(), status)
        if (status != RUNNING)
        {
            root_node->after_run(actor, blackboard);
        }
        root_node->set_status(status);
        return status;
    }

    void set_root_node(const Ref<BeehaveNode> &p_root_node)
    {
        root_node = p_root_node;
    }

    Ref<BeehaveNode> get_root_node()
    {
        return root_node;
    }

    void set_listener(const TypedArray<BeehaveListener> &p_listener)
    {
        listeners.clear();
        for (int i = 0; i < p_listener.size(); i++)
        {
            listeners.push_back(p_listener[i]);
        }
    }

    TypedArray<BeehaveListener> get_listener()
    {
        TypedArray<BeehaveListener> ret;
        for (int i = 0; i < listeners.size(); i++)
        {
            ret.push_back(listeners[i]);
        }
        return ret;
    }

public:

    ObjectID last_editor_id;
    LocalVector<Ref<BeehaveListener>> listeners;
    Ref<BeehaveNode> root_node;
    float tick_rate = 0.1f;
    float last_tick = 0.0f;
    float _process_time_metric_value = 0.0f;
    int status = -1;
    bool _can_send_message = false;
};

