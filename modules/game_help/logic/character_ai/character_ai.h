#ifndef _CHARACTER_AI_H_
#define _CHARACTER_AI_H_
#include "scene/3d/node_3d.h"
#include "core/object/ref_counted.h"
#include "modules/limboai/bt/bt_player.h"
#include "scene/3d/physics/character_body_3d.h"
#include "../body_main.h"


class CharacterBodyMain;
// 用来检测角色的一些状态
class CharacterAI_CheckBase : public RefCounted
{
    GDCLASS(CharacterAI_CheckBase,RefCounted);
    static void _bind_methods(){}
    public:
    // 返回true 代表立即执行决策,否则大脑根据其他条件决策
    virtual bool execute(CharacterBodyMain *node, Blackboard* blackboard)
    {
        bool rs = false;
        if(_execute_check(node,blackboard))
        {
            rs = true ;
        }
        if (GDVIRTUAL_IS_OVERRIDDEN(_execute)) {
            bool is_update_brain = false;
            GDVIRTUAL_CALL(_execute, node,blackboard,is_update_brain);
            rs = true ;
        }
        return rs;
    }
    virtual bool _execute_check(CharacterBodyMain *node,Blackboard* blackboard)
    {
        return false;
    }
	GDVIRTUAL2R(bool,_execute,CharacterBodyMain*,Blackboard*)
    // 优先级
    int priority = 0;

};

// 检测角色是否在地面上
class CharacterAI_CheckGround : public CharacterAI_CheckBase
{
    GDCLASS(CharacterAI_CheckGround, CharacterAI_CheckBase);
    static void _bind_methods(){}
public:
    bool _execute_check(CharacterBodyMain *node, Blackboard* blackboard);

	PhysicsDirectSpaceState3D::RayResult result;
    float check_move_height;
    float check_max_distance = 0.0;
    float ground_min_distance = 0.0;
    uint64_t ground_mask = 0;

};

// 检测角色敌人
class CharacterAI_CheckEnemy : public CharacterAI_CheckBase
{
    GDCLASS(CharacterAI_CheckEnemy, CharacterAI_CheckBase);
    static void _bind_methods(){}
    public:
    bool _execute_check(CharacterBodyMain *node, Blackboard* blackboard)
    {
        return false;
    }
    
};
// 检测角色跳跃
class CharacterAI_CheckJump : public CharacterAI_CheckBase
{
    GDCLASS(CharacterAI_CheckJump, CharacterAI_CheckBase);
    static void _bind_methods(){}
    public:
    bool _execute_check(CharacterBodyMain *node, Blackboard* blackboard)
    {
        if(blackboard->get("is_fall"))
        {
            return false;
        }
        // 检测玩家是否请求跳跃
        if(!blackboard->get("is_jump") && blackboard->get("is_ground") && blackboard->get("request_jump"))
        {
            blackboard->set("is_jump", false);
            return true;
        }
        return false;
        
    }
    
};
// 检测角色二次跳跃
class CharacterAI_CheckJump2 : public CharacterAI_CheckBase
{
    GDCLASS(CharacterAI_CheckJump2, CharacterAI_CheckBase);
    static void _bind_methods(){}
    public:
    bool _execute_check(CharacterBodyMain *node, Blackboard* blackboard)
    {
        // 检测玩家是否请求跳跃
        if(blackboard->get("is_jump") && blackboard->get("request_jump") && !blackboard->get("is_jump2"))
        {
            float distance =  blackboard->get("to_ground_distance");
            if(distance > min_ground_distance)
            {
                blackboard->set("is_jump2", false);
                return true;
            }
        }
        return false;
        
    }
    
    // 触发二次跳跃最小距离
    float min_ground_distance = 0.0;
};

// 检测角色是否超越巡逻范围
class CharacterAI_CheckPatrol : public CharacterAI_CheckBase
{
    GDCLASS(CharacterAI_CheckPatrol, CharacterAI_CheckBase);
    static void _bind_methods(){}
    public:
    virtual bool _execute_check(Blackboard* blackboard) 
    {
        return false;
    }
    LocalVector<Ref<CharacterAI_CheckBase>> checks;
};

// 角色感应器
class CharacterAI_Inductor : public RefCounted
{
    GDCLASS(CharacterAI_Inductor,RefCounted);
    static void _bind_methods(){}
public:
    struct SortCharacterCheck {
        bool operator()(const Ref<CharacterAI_CheckBase> &l, const Ref<CharacterAI_CheckBase> &r) const {
            int lp = 0;
            int rp = 0;
            if(l.is_valid()){
                lp = l->priority;
            }
            if(r.is_valid()){
                rp = r->priority;
            }

            return lp > lp;
        }
    };
    virtual bool execute(CharacterBodyMain *node,Blackboard* blackboard) 
    {
        sort_check();
        bool rs = false;
        for(uint32_t i=0;i<checks.size();i++)
        {
            if(checks[i].is_valid() && checks[i]->execute(node,blackboard))
            {
                rs = true;
            }
        }
        return rs;
    }

    void sort_check()
    {
        if(!is_sort)
        {
            return;
        }
        checks.sort_custom<SortCharacterCheck>();
        is_sort = true;
    }

    LocalVector<Ref<CharacterAI_CheckBase>> checks;
    bool is_sort = false;
};
// 角色 AI 逻辑节点
class CharacterAILogicNode : public Resource
{
    GDCLASS(CharacterAILogicNode,Resource);

    static void _bind_methods()
    {

    }
public:
    void enter(CharacterBodyMain *node,Blackboard* blackboard)
    {
        _enter_logic(node,blackboard);
        if (GDVIRTUAL_IS_OVERRIDDEN(_enter)) {
            GDVIRTUAL_CALL(_enter, node,blackboard);
        }

    }
    virtual void _enter_logic(CharacterBodyMain *node,Blackboard* blackboard)
    {

    }
    bool execute(CharacterBodyMain *node,Blackboard* blackboard)
    {
        bool rs = false;
        if(_execute_logic(node,blackboard))
        {
            rs = true ;
        }
        if (GDVIRTUAL_IS_OVERRIDDEN(_execute)) {
            bool is_stop = false;
            GDVIRTUAL_CALL(_execute, node,blackboard,is_stop);
            rs = true ;
        }
        return rs;
    }
    virtual bool _execute_logic(CharacterBodyMain *node,Blackboard* blackboard)
    {
        return false;
    }
    void exit(CharacterBodyMain *node,Blackboard* blackboard)
    {
        _stop_logic(node,blackboard);
        if (GDVIRTUAL_IS_OVERRIDDEN(_stop)) {

            GDVIRTUAL_CALL(_stop, node,blackboard);
        }

    }
    virtual void _stop_logic(CharacterBodyMain *node,Blackboard* blackboard)
    {

    }
	GDVIRTUAL2(_enter,CharacterBodyMain*,Blackboard*)
	GDVIRTUAL2R(bool,_execute,CharacterBodyMain*,Blackboard*)
	GDVIRTUAL2(_stop,CharacterBodyMain*,Blackboard*)
};
// 巡逻 AI 逻辑节点
class CharacterAILogicNode_Patrol : public CharacterAILogicNode
{
    GDCLASS(CharacterAILogicNode_Patrol,CharacterAILogicNode);

    static void _bind_methods()
    {

    }
 public:
    virtual void _enter_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {

    }
    virtual bool _execute_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {
        return false;
    }
    virtual void _stop_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {

    }
};


class CharacterAILogicNode_Jump : public CharacterAILogicNode
{
    GDCLASS(CharacterAILogicNode_Jump,CharacterAILogicNode);
    static void _bind_methods()
    {

    }
public:
    virtual void _enter_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {

    }
    virtual bool _execute_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {
        return false;
    }
    virtual void _stop_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {
        blackboard->set("is_fall",true);
    }
};

class CharacterAILogicNode_Jump2 : public CharacterAILogicNode
{
    GDCLASS(CharacterAILogicNode_Jump2,CharacterAILogicNode);
    static void _bind_methods()
    {

    }
    public:
    virtual void _enter_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {

    }
    virtual bool _execute_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {
        return false;
    }
    virtual void _stop_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {
        // 
        blackboard->set("is_fall",true);

    }
};

// 跟随目标 AI 逻辑节点
class CharacterAILogicNode_Follow : public CharacterAILogicNode
{
    GDCLASS(CharacterAILogicNode_Follow,CharacterAILogicNode);
    static void _bind_methods()
    {

    }
    public:
    virtual void _enter_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {

    }
    virtual bool _execute_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {
        return false;
    }
    virtual void _stop_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {

    }
};

// 逃跑 AI 逻辑节点
class CharacterAILogicNode_Escape : public CharacterAILogicNode
{
    GDCLASS(CharacterAILogicNode_Escape,CharacterAILogicNode);
    static void _bind_methods()
    {

    }
    public:
    virtual void _enter_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {

    }
    virtual bool _execute_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {
        return false;
    }
    virtual void _stop_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {

    }
};

// 战斗 AI 逻辑节点
class CharacterAILogicNode_Battle : public CharacterAILogicNode
{
    GDCLASS(CharacterAILogicNode_Battle,CharacterAILogicNode);
    static void _bind_methods()
    {

    }
    public:
    virtual void _enter_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {

    }
    virtual bool _execute_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {
        return false;
    }
    virtual void _stop_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {

    }
};

// 重生 AI 逻辑节点
class CharacterAILogicNode_Respawn : public CharacterAILogicNode
{
    GDCLASS(CharacterAILogicNode_Respawn,CharacterAILogicNode);
    static void _bind_methods()
    {

    }
    public:
    virtual void _enter_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {

    }
    virtual bool _execute_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {
        return false;
    }
    virtual void _stop_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {

    }
};

// 挑衅 AI 逻辑节点
class CharacterAILogicNode_Provoke  : public CharacterAILogicNode
{
    GDCLASS(CharacterAILogicNode_Provoke,CharacterAILogicNode);
    static void _bind_methods()
    {

    }
    public:
    virtual void _enter_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {
    }
    virtual bool _execute_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {
        return false;
    }
    virtual void _stop_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {

    }
};

// 发呆 AI 逻辑节点
class CharacterAILogicNode_Idle : public CharacterAILogicNode
{
    GDCLASS(CharacterAILogicNode_Idle,CharacterAILogicNode);
    static void _bind_methods()
    {

    }
    public:
    virtual void _enter_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {
        int ident_index = 0;
        if(is_random)
        {
            ident_index = Math::rand()%idle_time_list.size();
        }
        blackboard->set_var("Idextity_Index",ident_index);
        current_idle_start_time = OS::get_singleton()->get_unix_time();
    }
    virtual bool _execute_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {
        int ident_index = blackboard->get_var("Idextity_Index",0);
        float idle_time = idle_time_list[ident_index];
        if(OS::get_singleton()->get_unix_time() - current_idle_start_time < idle_time)
        {
            return false;
        }
        ++ident_index;
        if(ident_index >= idle_time_list.size())
        {
            ident_index = 0;
        }
        if(is_random)
        {
            ident_index = Math::rand()%idle_time_list.size();
        }
        blackboard->set_var("Idextity_Index",ident_index);
        current_idle_start_time = OS::get_singleton()->get_unix_time();
        return false;
    }
    virtual void _stop_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {

    }

    TypedArray<float> idle_time_list;
    bool is_random;
    float current_idle_start_time;
    
};

// 死亡 AI 逻辑节点
class CharacterAILogicNode_Dead : public CharacterAILogicNode
{
    GDCLASS(CharacterAILogicNode_Dead,CharacterAILogicNode);
    static void _bind_methods()
    {

    }
    public:
    virtual void _enter_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {
        if(!blackboard->get("dead/is_dead"))
        {
            blackboard->set("is_dead",true);
        }
    }
    virtual bool _execute_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {
        if(!blackboard->get("dead/request_revive"))
        {
            blackboard->set("is_dead",false);
            // 设置复活的生命
            blackboard->set("prop/curr_life",blackboard->get("dead/request_revive_life"));
            return true;
        }
        return false;
    }
    virtual void _stop_logic(CharacterBodyMain *node,Blackboard* blackboard)override
    {

    }
    
};
// 角色的阵营 枚举
enum CharacterCamp
{
    CharacterCamp_Player,
    CharacterCamp_Enemy,
    CharacterCamp_Friend,
};

struct CharacterAIContext
{
    Ref<CharacterAILogicNode> logic_node;
    StringName logic_name;
    CharacterCamp camp;
    
};


// AI 大脑
class CharacterAI_Brain : public RefCounted
{
    GDCLASS(CharacterAI_Brain,RefCounted);
    static void _bind_methods()
    {

    }
public:

    void execute(CharacterBodyMain *node,Blackboard* blackboard) 
    {
        _execute_brain(node,blackboard);
        if (GDVIRTUAL_IS_OVERRIDDEN(_execute)) {
            GDVIRTUAL_CALL(_execute, node,blackboard);
        }
    }
    virtual void _execute_brain(CharacterBodyMain *node,Blackboard* blackboard) 
    {

    }
	GDVIRTUAL2(_execute,CharacterBodyMain*,Blackboard*)

};

class CharacterAI : public Resource
{
    GDCLASS(CharacterAI,Resource);
    static void _bind_methods()
    {

    }
public:
    void set_inductor(Ref<CharacterAI_Brain> p_inductor)
    {
        inductor = p_inductor;
    }

    Ref<CharacterAI_Brain> get_inductor()
    {
        return inductor;
    }

    void set_brain(Ref<CharacterAI_Brain> p_brain)
    {
        brain = p_brain;
    }

    Ref<CharacterAI_Brain> get_brain()
    {
        return brain;
    }

    void set_logic_node(TypedArray<CharacterAILogicNode> p_logic_node)
    {
        logic_nodes.clear();
        logic_node_array.clear();
        for(auto &kv : p_logic_node)
        {
            Ref<CharacterAILogicNode> logic_node = kv;
            logic_node_array.push_back(logic_node);
            if(logic_node.is_valid())
            {
                logic_nodes[logic_node->get_name()] = logic_node;
            }
        }
    }
    TypedArray<CharacterAILogicNode> get_logic_node()
    {
        TypedArray<CharacterAILogicNode> ret;
        for(auto &kv : logic_node_array)
        {
            ret.push_back(kv);
        }
        return ret;
    }
public:
    void execute(CharacterBodyMain *node,Blackboard* blackboard,CharacterAIContext* p_context)
    {
        bool is_run_brain = false;
        if(p_context->logic_node.is_valid())
        {
            if(p_context->logic_node->execute(node,blackboard))
            {
                is_run_brain = true;
                p_context->logic_node->exit(node,blackboard);
                p_context->logic_node = Ref<CharacterAILogicNode>();
                p_context->logic_name = StringName();
            }
        }
        else
        {
            is_run_brain = true;
        }
        if(inductor.is_valid())
        {
            inductor->execute(node,blackboard);
        }

        if(is_run_brain && brain.is_valid())
        {
            brain->execute(node,blackboard);
            StringName logic_name = blackboard->get_var("ai/curr_logic_node_name",ident_node_name);
            if(logic_name != StringName() && logic_name != p_context->logic_name)
            {
                if(logic_nodes.has(logic_name))
                {
                    if(p_context->logic_node.is_valid())
                    {
                        p_context->logic_node->exit(node,blackboard);
                    }
                    p_context->logic_name = logic_name;
                    p_context->logic_node = logic_nodes[logic_name];
                    if(p_context->logic_node.is_valid())
                    {
                        p_context->logic_node->enter(node,blackboard);
                    }
                    return;
                }
            }
            
        }
        // 没有判断出任何状态,强制进入休闲状态
        if(p_context->logic_node.is_null())
        {
            p_context->logic_name = ident_node_name;
            if(logic_nodes.has(ident_node_name))
            {
                p_context->logic_node = logic_nodes[ident_node_name];
                if(p_context->logic_node.is_valid())
                {
                    p_context->logic_node->enter(node,blackboard);
                }
            }
        }

    }
    StringName ident_node_name = "ident";
    // 角色感應器
    Ref<CharacterAI_Inductor> inductor;
    Ref<CharacterAI_Brain> brain;
    LocalVector<Ref<CharacterAILogicNode>> logic_node_array;
    HashMap<StringName,Ref<CharacterAILogicNode>> logic_nodes;

};
#endif