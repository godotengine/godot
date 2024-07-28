#pragma once
#include "scene/3d/node_3d.h"
#include "modules/limboai/bt/bt_player.h"

class CharacterAnimatorLayer;
// 动画逻辑执行的任务
class CharacterTaskBase : public Resource
{
    GDCLASS(CharacterTaskBase,Resource)
    static void _bind_methods()
    {

    }
public:
    void process_task(CharacterAnimatorLayer* animator,Blackboard* blackboard)
    {
        _process_task(animator,blackboard);
    }
    virtual void _process_task(CharacterAnimatorLayer* animator,Blackboard* blackboard)
    {

    }
protected:
    bool is_value_by_property = false;
    StringName value_property_name;
};
class CharacterTaskActionBase : public Resource
{
    GDCLASS(CharacterTaskActionBase,Resource)
    static void _bind_methods()
    {

    }
    public:
    virtual void _process_task(CharacterAnimatorLayer* animator,Blackboard* blackboard)
    {

    }
    virtual void update_name()
    {

    }
    Array get_blackbord_propertys()
    {
        return _get_blackbord_propertys();
    }
    virtual Array _get_blackbord_propertys()
    {
        return Array();
    }
protected:
    bool is_value_by_property = false;
    StringName value_property_name;
};
class CharacterAnimationLogicNode;
class CharacterAnimationLogicRoot : public RefCounted
{
    GDCLASS(CharacterAnimationLogicRoot,RefCounted)
    static void _bind_methods();
public:
    void sort();
    void set_node_list(const Array& p_node_list)
    {
        for(int32_t i = 0; i < p_node_list.size(); ++i)
        {
            node_list.push_back(p_node_list[i]);
        }
    }
    Array get_node_list() 
    { 
        Array ret;
        for(uint32_t i = 0; i < node_list.size(); ++i)
        {
            ret.push_back(node_list[i]);
        }
        return ret; 
    }

    void set_bt_sort(int id){}
    int get_bt_sort() { return 0; }
    Ref<CharacterAnimationLogicNode> process_logic(Blackboard* blackboard);
public:
    bool is_need_sort = true;
    LocalVector<Ref<CharacterAnimationLogicNode>>  node_list;

};
// 动画逻辑层信息
/*
    每一个层里面可以处理多个动画状态
    每一个状态都有一套动画配置
*/
class CharacterAnimationLogicLayer : public Resource
{
    GDCLASS(CharacterAnimationLogicLayer, Resource);
    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_default_state_name","p_default_state_name"),&CharacterAnimationLogicLayer::set_default_state_name);
        ClassDB::bind_method(D_METHOD("get_default_state_name"),&CharacterAnimationLogicLayer::get_default_state_name);

        ClassDB::bind_method(D_METHOD("set_state_map","p_state_map"),&CharacterAnimationLogicLayer::set_state_map);
        ClassDB::bind_method(D_METHOD("get_state_map"),&CharacterAnimationLogicLayer::get_state_map);

        ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "default_state_name"), "set_default_state_name", "get_default_state_name");
        ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "state_map"), "set_state_map", "get_state_map");
    }
public:

    void set_default_state_name(const StringName& p_default_state_name) { default_state_name = p_default_state_name; }
    StringName get_default_state_name() { return default_state_name; }

    void set_state_map(const Dictionary& p_state_map) 
    {
        state_map.clear();
        Array key = p_state_map.keys();
        Array value = p_state_map.values();
        for(int32_t i = 0; i < key.size(); ++i)
        {
            state_map[key[i]] = value[i];
        }
        if(state_map.size() > 0 && !state_map.has(default_state_name))
        {
            default_state_name = key[0];
        }
    }
    Dictionary get_state_map() 
    { 
        Dictionary ret;
        for(auto & a : state_map)
        {
            ret[a.key] = a.value;
        }
        return ret; 
    }

    Ref<CharacterAnimationLogicNode> process_logic(StringName default_state_name,Blackboard* blackboard);


public:
    //  默认状态名称
    StringName default_state_name;
    HashMap<StringName, Ref<CharacterAnimationLogicRoot>> state_map;

};
