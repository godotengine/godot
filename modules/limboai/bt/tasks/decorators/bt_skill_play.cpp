#include "bt_skill_play.h"
#include "modules/game_help/logic/body_main.h"

void BTPlaySkill::initialize(Node *p_agent, const Ref<Blackboard> &p_blackboard)
{
    BTNewScope::initialize(p_agent, p_blackboard);
    CharacterBodyMain *body = Object::cast_to<CharacterBodyMain>(p_agent->get_owner());
    if(body == nullptr)
    {
        return;
    }
    if(!skillTree.is_empty())
    {
        body->play_skill(skillTree);
    }
}
PackedStringArray BTPlaySkill::get_configuration_warnings() 
{
    
    PackedStringArray warnings = BTTask::get_configuration_warnings(); // ! BTDecorator skipped intentionally
    if (skillTree.is_empty()) {
        warnings.append("skill needs to be assigned.");
    }
    else if(!FileAccess::exists(skillTree))
    {
        warnings.append("skill file not found.");
    }
    return warnings;
}
void BTPlaySkill::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("set_skill", "skill"), &BTPlaySkill::set_skill);
    ClassDB::bind_method(D_METHOD("get_skill"), &BTPlaySkill::get_skill);


    
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "skill"), "set_skill", "get_skill");
}