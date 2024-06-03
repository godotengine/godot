#ifndef _CHARACTER_AI_H_
#define _CHARACTER_AI_H_
#include "scene/3d/node_3d.h"
#include "core/object/ref_counted.h"
#include "modules/limboai/bt/bt_player.h"

// 用来检测角色的一些状态
class CharacterAI_CheckBase : public RefCounted
{
    public:
    virtual void execute(Node3D *node, Blackboard* blackboard)
    {

    }

};

// 检测角色是否在地面上
class CharacterAI_CheckGround : public CharacterAI_CheckBase
{

public:
    void execute(Node3D *node, Blackboard* blackboard);

	PhysicsDirectSpaceState3D::RayResult result;
    float check_move_height;
    float check_max_distance = 0.0;
    float ground_min_distance = 0.0;
    uint64_t ground_mask = 0;

};

// 检测角色敌人
class CharacterAI_CheckEnemy : public CharacterAI_CheckBase
{
    
};
// 检测角色跳跃
class CharacterAI_CheckJump : public CharacterAI_CheckBase
{
    
};

// 检测角色是否超越巡逻范围
class CharacterAI_CheckPatrol : public CharacterAI_CheckBase
{
    
};

// 角色感应器
class CharacterAI_Inductor : public RefCounted
{
public:
    virtual void execute(Blackboard* blackboard) 
    {

    }

    LocalVector<Ref<CharacterAI_CheckBase>> checks;
};
// 角色 AI 逻辑节点
class CharacterAILogicNode : public Resource
{
    void enter(Blackboard* blackboard)
    {

    }
    void execute(Blackboard* blackboard)
    {

    }
    void exit(Blackboard* blackboard)
    {

    }
};
// 巡逻 AI 逻辑节点
class CharacterAILogicNode_Patrol : public CharacterAILogicNode
{
    void enter(Blackboard* blackboard)
    {

    }
    void execute(Blackboard* blackboard)
    {

    }
    void exit(Blackboard* blackboard)
    {

    }
};

// 跟随目标 AI 逻辑节点
class CharacterAILogicNode_Follow : public CharacterAILogicNode
{
    void enter(Blackboard* blackboard)
    {

    }
    void execute(Blackboard* blackboard)
    {

    }
    void exit(Blackboard* blackboard)
    {

    }
};

// 逃跑 AI 逻辑节点
class CharacterAILogicNode_Escape : public CharacterAILogicNode
{
    void enter(Blackboard* blackboard)
    {

    }
    void execute(Blackboard* blackboard)
    {

    }
    void exit(Blackboard* blackboard)
    {

    }
};

// 战斗 AI 逻辑节点
class CharacterAILogicNode_Battle : public CharacterAILogicNode
{
    void enter(Blackboard* blackboard)
    {

    }
    void execute(Blackboard* blackboard)
    {

    }
    void exit(Blackboard* blackboard)
    {

    }
};

// 重生 AI 逻辑节点
class CharacterAILogicNode_Respawn : public CharacterAILogicNode
{
    void enter(Blackboard* blackboard)
    {

    }
    void execute(Blackboard* blackboard)
    {

    }
    void exit(Blackboard* blackboard)
    {

    }
};

// 挑衅 AI 逻辑节点
class CharacterAILogicNode_Provoke  : public CharacterAILogicNode
{
    void enter(Blackboard* blackboard)
    {
    
}
};

// 发呆 AI 逻辑节点
class CharacterAILogicNode_Idle : public CharacterAILogicNode
{
    void enter(Blackboard* blackboard)
    {

    }
    void execute(Blackboard* blackboard)
    {

    }
    void exit(Blackboard* blackboard)
    {

    }
};

// 死亡 AI 逻辑节点
class CharacterAILogicNode_Dead : public CharacterAILogicNode
{
    void enter(Blackboard* blackboard)
    {

    }
    void execute(Blackboard* blackboard)
    {

    }
    void exit(Blackboard* blackboard)
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
public:
    virtual void execute(Blackboard* blackboard) 
    {
    }
    void run_logic(Blackboard* blackboard,StringName p_logic_name)
    {

    }

};

class CharacterAI : public Resource
{

public:
    void execute(Blackboard* blackboard)
    {
        if(inductor.is_valid())
        {
            inductor->execute(blackboard);
        }
    }
    void run_logic(Blackboard* blackboard,StringName p_logic_name)
    {
    }
    // 角色感應器
    Ref<CharacterAI_Inductor> inductor;
    Ref<CharacterAI_Brain> brain;
    HashMap<StringName,Ref<CharacterAILogicNode>> logic_nodes;

};
#endif