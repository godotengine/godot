#ifndef _CHARACTER_AI_H_
#define _CHARACTER_AI_H_
#include "scene/3d/node_3d.h"
#include "core/object/ref_counted.h"
#include "modules/limboai/bt/bt_player.h"
#include "scene/3d/physics/character_body_3d.h"
#include "animator_condition.h"
#include "../body_main.h"


class CharacterBodyMain;
// 用来检测角色的一些状态
class CharacterAI_CheckBase : public RefCounted
{
    GDCLASS(CharacterAI_CheckBase,RefCounted);
    static void _bind_methods() {

        ClassDB::bind_method(D_METHOD("set_priority","priority"),&CharacterAI_CheckBase::set_priority);
        ClassDB::bind_method(D_METHOD("get_priority"),&CharacterAI_CheckBase::get_priority);
        
        ClassDB::bind_method(D_METHOD("set_name","name"),&CharacterAI_CheckBase::set_name);
        ClassDB::bind_method(D_METHOD("get_name"),&CharacterAI_CheckBase::get_name);


        ClassDB::bind_method(D_METHOD("set_enable_condition","enable_condition"),&CharacterAI_CheckBase::set_enable_condition);
        ClassDB::bind_method(D_METHOD("get_enable_condition"),&CharacterAI_CheckBase::get_enable_condition);

        ClassDB::bind_method(D_METHOD("set_blackboard_plan","blackboard_plan"),&CharacterAI_CheckBase::set_blackboard_plan);
        ClassDB::bind_method(D_METHOD("get_blackboard_plan"),&CharacterAI_CheckBase::get_blackboard_plan);

        ADD_PROPERTY(PropertyInfo(Variant::INT,"priority"), "set_priority","get_priority");
        ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME,"name"), "set_name","get_name");
        ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"enable_condition",PROPERTY_HINT_RESOURCE_TYPE,"CharacterAnimatorCondition"), "set_enable_condition","get_enable_condition");
        ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"blackboard_plan",PROPERTY_HINT_RESOURCE_TYPE,"BlackboardPlan"), "set_blackboard_plan","get_blackboard_plan");
    }
    public:
    // 返回true 代表立即执行决策,否则大脑根据其他条件决策
    virtual bool execute(CharacterBodyMain *node, Blackboard* blackboard)
    {
        bool rs = false;
        if(enable_condition.is_valid())
        {
            if( !enable_condition->is_enable(blackboard) )
            {
                return rs;
            }
        }
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


    public:
    void set_priority(int p_priority)
    {
        this->priority = p_priority;
    }

    int get_priority()
    {
        return priority;
    }

    void set_name(StringName p_name)
    {
        this->name = p_name;
    }

    StringName get_name()
    {
        return name;
    }

    void set_enable_condition(Ref<CharacterAnimatorCondition> p_enable_condition)
    {
        this->enable_condition = p_enable_condition;
        update_blackboard_plan();
    }

    Ref<CharacterAnimatorCondition> get_enable_condition()
    {
        return enable_condition;
    }

    void set_blackboard_plan(Ref<BlackboardPlan> p_blackboard_plan)
    {
        this->blackboard_plan = p_blackboard_plan;
        update_blackboard_plan();
    }

    Ref<BlackboardPlan> get_blackboard_plan()
    {
        return blackboard_plan;
    }
    void update_blackboard_plan()
    {
        if(blackboard_plan.is_valid() && enable_condition.is_valid())
        {
            enable_condition->set_blackboard_plan(blackboard_plan);
        }
    }
    protected:
    // 优先级
    int priority = 0;
    StringName name;
    // 检查器的激活条件
    Ref<CharacterAnimatorCondition> enable_condition;
    
    Ref<BlackboardPlan> blackboard_plan;

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
    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_enemy_layer","layer"),&CharacterAI_CheckEnemy::set_enemy_layer);
        ClassDB::bind_method(D_METHOD("get_enemy_layer"),&CharacterAI_CheckEnemy::get_enemy_layer);

        ClassDB::bind_method(D_METHOD("set_is_form_angle","is_form_angle"),&CharacterAI_CheckEnemy::set_is_form_angle);
        ClassDB::bind_method(D_METHOD("get_is_form_angle"),&CharacterAI_CheckEnemy::get_is_form_angle);

        ClassDB::bind_method(D_METHOD("set_form_angle","form_angle"),&CharacterAI_CheckEnemy::set_form_angle);
        ClassDB::bind_method(D_METHOD("get_form_angle"),&CharacterAI_CheckEnemy::get_form_angle);

        ClassDB::bind_method(D_METHOD("set_body_area_name","body_area_name"),&CharacterAI_CheckEnemy::set_body_area_name);
        ClassDB::bind_method(D_METHOD("get_body_area_name"),&CharacterAI_CheckEnemy::get_body_area_name);

        ADD_PROPERTY(PropertyInfo(Variant::INT,"enemy_layer",PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_enemy_layer","get_enemy_layer");
        ADD_PROPERTY(PropertyInfo(Variant::BOOL,"is_form_angle"), "set_is_form_angle","get_is_form_angle");
        ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"form_angle"), "set_form_angle","get_form_angle");
        ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME,"body_area_name"), "set_body_area_name","get_body_area_name");
    }
    public:
    bool _execute_check(CharacterBodyMain *node, Blackboard* blackboard);
    void set_enemy_layer(int32_t p_layer)
    {
        enemy_layer = p_layer;
    }

    int32_t get_enemy_layer()
    {
        return enemy_layer;
    }

    void set_is_form_angle(bool p_is_form_angle)
    {
        is_form_angle = p_is_form_angle;
    }

    bool get_is_form_angle()
    {
        return is_form_angle;
    }

    void set_form_angle(float p_form_angle)
    {
        form_angle = p_form_angle;
    }

    float get_form_angle()
    {
        return form_angle;
    }   

    void set_body_area_name(StringName p_body_area_name)
    {
        body_area_name = p_body_area_name;
    }

    StringName get_body_area_name()
    {
        return body_area_name;
    }

    StringName body_area_name;
    int32_t enemy_layer = 0;
    bool is_form_angle = false;
    float form_angle = 1;
    bool is_sort_check = false;
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
    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_check","check"),&CharacterAI_Inductor::set_check);
        ClassDB::bind_method(D_METHOD("get_check"),&CharacterAI_Inductor::get_check);

        ADD_PROPERTY(PropertyInfo(Variant::ARRAY,"check",PROPERTY_HINT_ARRAY_TYPE,MAKE_RESOURCE_TYPE_HINT("CharacterAI_CheckBase")), "set_check","get_check");
    }
public:
    struct SortCharacterCheck {
        bool operator()(const Ref<CharacterAI_CheckBase> &l, const Ref<CharacterAI_CheckBase> &r) const {
            int lp = 0;
            int rp = 0;
            if(l.is_valid()){
                lp = l->get_priority();
            }
            if(r.is_valid()){
                rp = r->get_priority();
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
    void set_check(TypedArray<CharacterAI_CheckBase> check)
    {
        checks.clear();
        for(int32_t i=0;i<check.size();i++)
        {
            checks.push_back(check[i]);
        }
    }
    TypedArray<CharacterAI_CheckBase> get_check()
    {
        TypedArray<CharacterAI_CheckBase> ret;
        for(uint32_t i=0;i<checks.size();i++)
        {
            ret.push_back(checks[i]);
        }
        return ret;
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
class CharacterAILogicNode : public RefCounted
{
    GDCLASS(CharacterAILogicNode,RefCounted);

    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_name","name"),&CharacterAILogicNode::set_name);
        ClassDB::bind_method(D_METHOD("get_name"),&CharacterAILogicNode::get_name);

        ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME,"name"), "set_name","get_name");
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

    void set_name(StringName p_name)
    {
        this->name = p_name;
    }

    StringName get_name()
    {
        return name;
    }
	GDVIRTUAL2(_enter,CharacterBodyMain*,Blackboard*)
	GDVIRTUAL2R(bool,_execute,CharacterBodyMain*,Blackboard*)
	GDVIRTUAL2(_stop,CharacterBodyMain*,Blackboard*)
    StringName name;
};
// 巡逻 AI 逻辑节点
class CharacterAILogicNode_Patrol : public CharacterAILogicNode
{
    GDCLASS(CharacterAILogicNode_Patrol,CharacterAILogicNode);

    static void _bind_methods()
    {

    }
 public:
    CharacterAILogicNode_Patrol()
    {
        name = "CharacterAILogicNode_Patrol";
    }
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
    CharacterAILogicNode_Jump()
    {
        name = "CharacterAILogicNode_Jump";
    }
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
    CharacterAILogicNode_Jump2()
    {
        name = "CharacterAILogicNode_Jump2";
    }
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
    CharacterAILogicNode_Follow()
    {
        name = "CharacterAILogicNode_Follow";
    }
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
    CharacterAILogicNode_Escape()
    {
        name = "CharacterAILogicNode_Escape";
    }
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
    CharacterAILogicNode_Battle()
    {
        name = "CharacterAILogicNode_Battle";
    }
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
    CharacterAILogicNode_Respawn()
    {
        name = "CharacterAILogicNode_Respawn";
    }
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
    CharacterAILogicNode_Provoke()
    {
        name = "CharacterAILogicNode_Provoke";
    }
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
    CharacterAILogicNode_Idle()
    {
        name = "CharacterAILogicNode_Idle";
    }
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
    CharacterAILogicNode_Dead()
    {
        name = "CharacterAILogicNode_Dead";
    }
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
	void set_name(const StringName &p_name) {
		name = p_name;
	}

	StringName get_name() const {
		return name;
	}
	GDVIRTUAL2(_execute,CharacterBodyMain*,Blackboard*)

	StringName name;
};

class CharacterAI : public Resource
{
    GDCLASS(CharacterAI,Resource);
    static void _bind_methods();
public:
    void set_inductor(const Ref<CharacterAI_Inductor>& p_inductor)
    {
        inductor = p_inductor;
    }

    Ref<CharacterAI_Inductor> get_inductor()
    {
        return inductor;
    }

    void set_brain(const Ref<CharacterAI_Brain>& p_brain)
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
                StringName sn = logic_node->get_name();
                if(sn != StringName())
                {
                    logic_nodes[logic_node->get_name()] = logic_node;
                }
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
    void execute(CharacterBodyMain *node,Blackboard* blackboard,struct CharacterAIContext* p_context);
    StringName ident_node_name = "ident";
    // 角色感應器
    Ref<CharacterAI_Inductor> inductor;
    Ref<CharacterAI_Brain> brain;
    LocalVector<Ref<CharacterAILogicNode>> logic_node_array;
    HashMap<StringName,Ref<CharacterAILogicNode>> logic_nodes;

};
#endif