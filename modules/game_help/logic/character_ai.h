#ifndef _CHARACTER_AI_H_
#define _CHARACTER_AI_H_
#include "body_animator_logic.h"
#include "body_main.h"

// 用来检测角色的一些状态
class CharacterAI_CheckBase : public RefCounted
{

};

// 检测角色是否在地面上
class CharacterAI_CheckGround : public CharacterAI_CheckBase
{

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
// AI 大脑
class CharacterAI_Brain : public RefCounted
{
public:
    virtual void execute(Blackboard* blackboard) 
    {

    }

};

#endif