#pragma once

#include "scene/main/node.h"
#include "core/io/resource.h"
#include "modules/limboai/bt/bt_player.h"

// 行为树运行上下文
class BeehaveRuncontext : public RefCounted
{
public:

    // 获取属性
    Dictionary get_property(Object* curr_this_node)
    {
        if(properties.has(curr_this_node->get_instance_id()))
        {
            return properties[curr_this_node->get_instance_id()];
        }
        Dictionary rs;
        properties[curr_this_node->get_instance_id()] = rs;
        return rs;
    }
    void set_run_state(Object* curr_this_node,int state)
    {
        Dictionary rs = get_property(curr_this_node);
        rs[SNAME("run_state")] = state;
    }
    int get_run_state(Object* curr_this_node)
    {
        Dictionary rs = get_property(curr_this_node);
        return rs.get(SNAME("run_state"),-1);
    }

    void init_child_state(Object* curr_this_node,int child_count)
    {
        Dictionary rs = get_property(curr_this_node);
        Vector<int32_t> child_state = rs.get("child_status", Vector<int32_t>());
        child_state.resize(child_count);
        child_state.fill(0);
        rs[SNAME("run_state")] = -1;
    }
    Vector<int32_t> get_child_state(Object* curr_this_node)
    {
        Dictionary rs = get_property(curr_this_node);
        Vector<int32_t> child_state = rs.get("child_status", Vector<int32_t>());
        return child_state;
    }
    HashMap<uint64_t,Dictionary> properties;
    double time = 0.0;
    double delta = 0.0;
    Node* actor = nullptr;
    class BeehaveTree* tree = nullptr;
    Ref<class Blackboard> blackboard;
};
/// 序列节点
class BeehaveNode : public RefCounted
{
    GDCLASS(BeehaveNode, RefCounted);
    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_name", "name"), &BeehaveNode::set_name);
        ClassDB::bind_method(D_METHOD("get_name"), &BeehaveNode::get_name);

        ClassDB::bind_method(D_METHOD("set_annotation", "annotation"), &BeehaveNode::set_annotation);
        ClassDB::bind_method(D_METHOD("get_annotation"), &BeehaveNode::get_annotation);

        ClassDB::bind_method(D_METHOD("set_children", "children"), &BeehaveNode::set_children);
        ClassDB::bind_method(D_METHOD("get_children"), &BeehaveNode::get_children);


        ClassDB::bind_method(D_METHOD("set_enable", "enable"), &BeehaveNode::set_enable);
        ClassDB::bind_method(D_METHOD("get_enable"), &BeehaveNode::get_enable);

        ClassDB::bind_method(D_METHOD("set_editor_collapsed_children", "enable"), &BeehaveNode::set_editor_collapsed_children);
        ClassDB::bind_method(D_METHOD("get_editor_collapsed_children"), &BeehaveNode::get_editor_collapsed_children);

        ClassDB::bind_method(D_METHOD("set_debug_enabled", "enable"), &BeehaveNode::set_debug_enabled);
        ClassDB::bind_method(D_METHOD("get_debug_enabled"), &BeehaveNode::get_debug_enabled);


        ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "name"), "set_name", "get_name");
        ADD_PROPERTY(PropertyInfo(Variant::STRING, "annotation"), "set_annotation", "get_annotation");
        ADD_PROPERTY(PropertyInfo(Variant::BOOL, "enable", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_enable", "get_enable");
        ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editor_collapsed_children", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_editor_collapsed_children", "get_editor_collapsed_children");
        ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "children", PROPERTY_HINT_ARRAY_TYPE, MAKE_RESOURCE_TYPE_HINT("BeehaveNode"), PROPERTY_USAGE_NO_EDITOR), "set_children", "get_children");
        ADD_PROPERTY(PropertyInfo(Variant::BOOL, "debug_enabled", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_NO_EDITOR), "set_debug_enabled", "get_debug_enabled");
    }
public:
    enum SequenceRunState
    {
        NONE_PROCESS = -1,SUCCESS, FAILURE, RUNNING
    };
    int get_child_count() const { return children.size(); }

    Ref<BeehaveNode> get_child(uint32_t p_index) const
    {
        if(p_index < children.size())
        {
            return children[p_index];
        }
        return Ref<BeehaveNode>(); 
    }


    // Called when this node needs to be interrupted before it can return FAILURE or SUCCESS.
    virtual void interrupt(const Ref<BeehaveRuncontext>& run_context) 
    {
        run_context->init_child_state(this, children.size());
        for(uint32_t i = 0; i < children.size(); ++i)
        {
            children[i]->interrupt(run_context);
        }
    }
    // Called before the first time it ticks by the parent.
    virtual void  before_run(const Ref<BeehaveRuncontext>& run_context)  
    {

    }
    int process(const Ref<BeehaveRuncontext>& run_context);
    virtual int tick(const Ref<BeehaveRuncontext>& run_context) 
    {
        run_context->set_run_state(this,SequenceRunState::SUCCESS);
        return SUCCESS;

    }
    // Called after the last time it ticks and returns
    // [code]SUCCESS[/code] or [code]FAILURE[/code].
    virtual void after_run(const Ref<BeehaveRuncontext>& run_context)  
    {
        
    }
    virtual TypedArray<StringName> get_class_name()
    {
        return {"BeehaveNode"};
    }
    bool can_send_message(Blackboard* blackboard)
	{
        return blackboard->get_var("can_send_message", false);
    }
    virtual String get_tooltip()
    {
        return String(L"行为树节点");
    }
    virtual String get_lable_name()
    {
        return String(L"");
    }
    virtual StringName get_icon()
    {
        return SNAME("BezierHandlesFree");
    }
    // 獲取支持放几个子节点,-1 是任意多子节点
    virtual int get_supper_child_count()
    {
        return 0;
    }
	StringName get_id()
	{
		return StringName(String::num_int64((uint64_t)get_instance_id()));
	}
    void add_child(Ref<BeehaveNode> p_child)
    {
        children.push_back(p_child);
    }
public:
    void set_children(TypedArray<BeehaveNode> p_children)
    {
        children.clear();
        for(uint32_t i = 0; i < p_children.size(); ++i)
        {
            children.push_back(p_children[i]);
        }
    }

    TypedArray<BeehaveNode> get_children()
    {
        TypedArray<BeehaveNode> rs;
        for(uint32_t i = 0; i < children.size(); ++i)
        {
            rs.push_back(children[i]);
        }
        return rs;
    }
    void set_name(StringName p_name)
    {
        name = p_name;
    }

    StringName get_name()
    {
        return name;
    }

	void set_annotation(String p_annotation)
    {
        annotation = p_annotation;
    }
    String get_annotation()
    {
        return annotation;
    }
    void set_enable(bool p_enable)
    {
        enabled = p_enable;
    }

    bool get_enable()
    {
        return enabled;
    }
    void set_editor_collapsed_children(bool p_enable)
    {
        editor_collapsed_children = p_enable;
    }

    bool get_editor_collapsed_children()
    {
        return editor_collapsed_children;
    }
    void set_debug_enabled(bool p_enable)
    {
        debug_enabled = p_enable;
    }

    bool get_debug_enabled()
    {
        return debug_enabled;
    }
public:
    // 上移子节点
    void move_child_up(const Ref<BeehaveNode>& p_child)
    {
        for(uint32_t i = 0; i < children.size(); ++i)
        {
            if(children[i] == p_child)
            {
                if(i == 0)
                {
                    return;
                }
                Ref<BeehaveNode> tmp = children[i];
                children[i] = children[i - 1];
                children[i - 1] = tmp;
                break;
            }
        }
    }
    // 下移子节点
    void move_child_down(const Ref<BeehaveNode>& p_child)
    {
        for(uint32_t i = 0; i < children.size(); ++i)
        {
            if(children[i] == p_child)
            {
                if(i == children.size() - 1)
                {
                    return;
                }
                Ref<BeehaveNode> tmp = children[i];
                children[i] = children[i + 1];
                children[i + 1] = tmp;
                break;
            }
        }
    }
    // 删除子节点
    void remove_child(const Ref<BeehaveNode>& p_child)
    {
        for(uint32_t i = 0; i < children.size(); ++i)
        {
            if(children[i] == p_child)
            {
                children.remove_at(i);
                break;
            }
        }
    }
protected:
    LocalVector<Ref<BeehaveNode>> children;
	StringName name;
	// 获取描述
	String annotation;
    bool enabled = true;
    // 是否编辑器折叠子节点
    bool editor_collapsed_children = false;
    bool debug_enabled = false;
};



class BeehaveComposite : public BeehaveNode
{
    GDCLASS(BeehaveComposite, BeehaveNode);
    
    static void _bind_methods()
    {

    }
public:
    virtual String get_tooltip()override
    {
        return String(L"复合节点以特定方式控制其子节点的执行流。");
    }
    virtual String get_lable_name()
    {
        return String(L"组合节点");
    }
    virtual void interrupt(const Ref<BeehaveRuncontext>& run_context)override
    {
        base_class_type::interrupt(run_context);
    }
    virtual TypedArray<StringName> get_class_name()override
    {
        TypedArray<StringName> rs = base_class_type::get_class_name();
        rs.push_back("BeehaveComposite");
        return rs;
    }
    // 獲取支持放几个子节点,-1 是任意多子节点
    virtual int get_supper_child_count()
    {
        return -1;
    }
};

class BeehaveDecorator : public BeehaveNode
{
    GDCLASS(BeehaveDecorator, BeehaveNode);
    static void _bind_methods()
    {
        
    }
public:
    virtual String get_tooltip()override
    {
        return String(L"装饰器节点用于转换其子节点接收到的结果。\n只能有一个子节点。");
    }
    virtual String get_lable_name()
    {
        return String(L"装饰器节点");
    }
    virtual void interrupt(const Ref<BeehaveRuncontext>& run_context)override
    {
        base_class_type::interrupt(run_context);
    }
    virtual TypedArray<StringName> get_class_name()override
    {
        TypedArray<StringName> rs = base_class_type::get_class_name();
        rs.push_back("BeehaveDecorator");
        return rs;
    }
    // 獲取支持放几个子节点,-1 是任意多子节点
    int get_supper_child_count()override
    {
        return 1;
    }
};

class BeehaveLeaf : public BeehaveNode
{
    GDCLASS(BeehaveLeaf, BeehaveNode);
    static void _bind_methods()
    {
        
    }
public:
    virtual String get_tooltip()override
    {
        return String(L"树的所有叶节点的基类。");
    }
    virtual String get_lable_name()
    {
        return String(L"叶节点");
    }
    virtual void interrupt(const Ref<BeehaveRuncontext>& run_context)override
    {
        base_class_type::interrupt(run_context);
    }
    virtual TypedArray<StringName> get_class_name()override
    {
        TypedArray<StringName> rs = base_class_type::get_class_name();
        rs.push_back("BeehaveLeaf");
        return rs;
    }
    // 獲取支持放几个子节点,-1 是任意多子节点
    int get_supper_child_count()override
    {
        return 0;
    }
};

class BeehaveAction : public BeehaveLeaf
{
    GDCLASS(BeehaveAction, BeehaveLeaf);
    static void _bind_methods()
    {
	    GDVIRTUAL_BIND(_interrupt, "run_context");
	    GDVIRTUAL_BIND(_before_run, "run_context");
	    GDVIRTUAL_BIND(_after_run, "run_context");
	    GDVIRTUAL_BIND(_tick, "run_context");
    }
public:
    virtual StringName get_icon()
    {
        return SNAME("action");
    }
    virtual void interrupt(const Ref<BeehaveRuncontext>& run_context)  override
    {
        base_class_type::interrupt(run_context);	
        GDVIRTUAL_CALL(_interrupt, run_context);
    }
    virtual void before_run(const Ref<BeehaveRuncontext>& run_context)  override
    {
        base_class_type::before_run(run_context);
        GDVIRTUAL_CALL(_before_run, run_context);
    }
    virtual void after_run(const Ref<BeehaveRuncontext>& run_context)  override
    {
        base_class_type::after_run(run_context);
        GDVIRTUAL_CALL(_after_run, run_context);
    }
    virtual int tick(const Ref<BeehaveRuncontext>& run_context)  override
    {
        if (GDVIRTUAL_IS_OVERRIDDEN(_tick))
        {            
            int rs;
            GDVIRTUAL_CALL(_tick, run_context,rs);
			run_context->set_run_state(this,rs);
            return rs;
        }
        int rs = base_class_type::tick(run_context);
		run_context->set_run_state(this, rs);
        return rs;
    }

    
	GDVIRTUAL1(_interrupt,const Ref<BeehaveRuncontext>&);
	GDVIRTUAL1(_before_run,const Ref<BeehaveRuncontext>&);
	GDVIRTUAL1(_after_run,const Ref<BeehaveRuncontext>&);
	GDVIRTUAL1R(int,_tick,const Ref<BeehaveRuncontext>&);
};

// 模板
class BeehaveNodeTemplate : public RefCounted
{
    GDCLASS(BeehaveNodeTemplate, RefCounted);
    static void _bind_methods()
    {
    }
public:

protected:
    StringName group;

    Ref<BeehaveNode> node;
};

class BeehaveNodeTemplateManager : public RefCounted
{
    GDCLASS(BeehaveNodeTemplateManager, RefCounted);
    static void _bind_methods()
    {
    }
public:
    Ref<BeehaveNodeTemplate> get_template(const String& group, const String& name)
    {
        return nullptr;
    }


protected:
	HashMap<StringName, HashMap<StringName, Ref<BeehaveNodeTemplate> > > templates;
};


