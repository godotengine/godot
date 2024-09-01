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
    virtual void start(const Ref<BeehaveRuncontext>& run_context)
    {
        #if !TOOLS_ENABLED
           if(is_editor_only)
           {
               return;
           }
        #endif
        GDVIRTUAL_CALL(_start, run_context);

    }
    virtual void process(const Ref<BeehaveRuncontext>& run_context)
    {
        #if !TOOLS_ENABLED
           if(is_editor_only)
           {
               return;
           }
        #endif
        GDVIRTUAL_CALL(_process, run_context);
    }
    virtual void stop(const Ref<BeehaveRuncontext>& run_context)
    {
        #if !TOOLS_ENABLED
           if(is_editor_only)
           {
               return;
           }
        #endif
        GDVIRTUAL_CALL(_stop, run_context);
    }
	GDVIRTUAL1(_start, const Ref<BeehaveRuncontext>&);
	GDVIRTUAL1(_process, const Ref<BeehaveRuncontext>&);
	GDVIRTUAL1(_stop, const Ref<BeehaveRuncontext>&);
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
        
        
        ADD_SUBGROUP("BeehaveTree","");
        ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "listener", PROPERTY_HINT_ARRAY_TYPE, MAKE_RESOURCE_TYPE_HINT("BeehaveListener")), "set_listener", "get_listener");
    }
    enum Status { SUCCESS, FAILURE, RUNNING };
public:
    // 初始化
    void init(const Ref<BeehaveRuncontext>& run_context)
    {
        last_tick = 0;
        _can_send_message = false;
        _process_time_metric_value = 0;
        status = RUNNING;
        run_context->tree = this;
        
        for (uint32_t i = 0; i < listeners.size(); i++)
        {
            listeners[i]->start(run_context);
        }
        if(root_node.is_valid())
        {
            root_node->interrupt(run_context);
        }
    }

    int process(const Ref<BeehaveRuncontext>& run_context)
    {
        if (last_tick < tick_rate - 1)
        {
            last_tick += 1;
            return status;

        }
        if(debug_break_node != nullptr)
        {
            return status;            
        }
        if(status != RUNNING)
        {
            return status;
        }
        run_context->tree = this;
        last_tick = 0;
        float start_time = OS::get_singleton()->get_ticks_usec();

        
        for (uint32_t i = 0; i < listeners.size(); i++)
        {
            listeners[i]->process(run_context);
        }
        // if _can_send_message:
        // 	BeehaveDebuggerMessages.process_begin(get_instance_id())
        status = tick(run_context);

        if(status != RUNNING)
        {
            on_stop(run_context);
        }
        // if _can_send_message:
        // 	BeehaveDebuggerMessages.process_end(get_instance_id())

	    _process_time_metric_value = OS::get_singleton()->get_ticks_usec() - start_time;
        return status;
    }
    void stop(const Ref<BeehaveRuncontext>& run_context)
    {
        if(status != RUNNING)
        {
            return;
        }
        on_stop(run_context);
		run_context->reset();
        status = SUCCESS;
    }
protected:
    void on_stop(const Ref<BeehaveRuncontext>& run_context)
    {        
        for (uint32_t i = 0; i < listeners.size(); i++)
        {
            listeners[i]->stop(run_context);
        }
    }
    int tick(const Ref<BeehaveRuncontext>& run_context)
    {	
        if (run_context->actor == nullptr || root_node.is_null())
		{
            return FAILURE;
        }
        if (status != RUNNING)
        {
            root_node->before_run(run_context);
        }
        status = root_node->tick(run_context);

        // 	if _can_send_message:
		// BeehaveDebuggerMessages.process_tick(child.get_instance_id(), status)
		// BeehaveDebuggerMessages.process_tick(get_instance_id(), status)
        if (status != RUNNING)
        {
            root_node->after_run(run_context);
        }
		run_context->set_run_state(this,status);
        return status;
    }
public:

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
        for (uint32_t i = 0; i < listeners.size(); i++)
        {
            ret.push_back(listeners[i]);
        }
        return ret;
    }
public:
    void set_debug_break_node(BeehaveNode *p_node)
    {
        debug_break_node = p_node;
    }
    BeehaveNode *get_debug_break_node()
    {
        return debug_break_node;
    }

public:
    Ref<BeehaveNode> root_node;

    LocalVector<Ref<BeehaveListener>> listeners;
    // 当前中断的节点
    class BeehaveNode *debug_break_node = nullptr;

    ObjectID last_editor_id;
    float tick_rate = 0.1f;
    float last_tick = 0.0f;
    float _process_time_metric_value = 0.0f;
    int status = -1;
    bool _can_send_message = false;
};

