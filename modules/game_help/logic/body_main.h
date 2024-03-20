
#ifndef BODY_MAIN_H
#define BODY_MAIN_H
#include "scene/resources/packed_scene.h"
#include "scene/animation/animation_player.h"
#include "scene/animation/animation_tree.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/3d/physics/character_body_3d.h"
#include "body_part.h"
#include "animation_help.h"
#include "body_animator.h"


#include "modules/limboai/bt/bt_player.h"
#include "modules/limboai/bt/tasks/decorators/bt_new_scope.h"

// 身体的插槽信息
class BodySocket
{
    Transform3D localPose;
    Transform3D globalPose;

    void on_bone_pose_update(Skeleton3D *p_skeleton, int p_bone_index)
    {
        globalPose = p_skeleton->get_bone_global_pose(p_bone_index) * localPose;
    }

};
// 身体主要部件部分
class CharacterBodyMain : public CharacterBody3D {
    GDCLASS(CharacterBodyMain, CharacterBody3D);
    static void _bind_methods();

public:
    // 初始化身體
    void init_main_body(String p_skeleton_file_path,StringName p_animation_group);
    void clear_all();

public:
	void set_behavior_tree(const Ref<BehaviorTree> &p_tree)
    {
        get_bt_player()->set_behavior_tree(p_tree);
    }
	Ref<BehaviorTree> get_behavior_tree()  { return get_bt_player()->get_behavior_tree(); };

	void set_blackboard_plan(const Ref<BlackboardPlan> &p_plan)
    {
        get_bt_player()->set_blackboard_plan(p_plan);
    }
	Ref<BlackboardPlan> get_blackboard_plan() { return get_bt_player()->get_blackboard_plan(); }

	void set_update_mode(int p_mode)
    {
        get_bt_player()->set_update_mode((BTPlayer::UpdateMode)p_mode);
    }
	int get_update_mode() { return (int)(get_bt_player()->get_update_mode()); }

	Ref<Blackboard> get_blackboard()  { return player_blackboard; }
    // 设置黑板
	void set_blackboard(const Ref<Blackboard> &p_blackboard) 
    { 
        player_blackboard = p_blackboard; 
        get_bt_player()->get_blackboard()->set_parent(p_blackboard);
        if(btSkillPlayer != nullptr)
        {
            btSkillPlayer->get_blackboard()->set_parent(player_blackboard); 
        }
    }

	void restart()
    {
        get_bt_player()->restart();
    }
	int get_last_status() { return get_bt_player()->get_last_status(); }

	Ref<BTTask> get_tree_instance() { return get_bt_player()->get_tree_instance(); }
public:
    bool play_skill(String p_skill_name)
    {
        if(btSkillPlayer != nullptr)
        {
            return false;
        }
        btSkillPlayer = memnew(BTPlayer);
        btSkillPlayer->set_owner((Node*)this);
        btSkillPlayer->set_name("BTPlayer_Skill");
        btSkillPlayer->connect("behavior_tree_finished", callable_mp(this, &CharacterBodyMain::skill_tree_finished));
        btSkillPlayer->connect("updated", callable_mp(this, &CharacterBodyMain::skill_tree_update));
        add_child(btSkillPlayer);
        if(has_method("skill_tree_init"))
        {
            call("skill_tree_init",p_skill_name);
        }
        btSkillPlayer->get_blackboard()->set_parent(player_blackboard);

        get_blackboard()->set_var("skill_name",p_skill_name);
        get_blackboard()->set_var("skill_play",true);
        return true;
    }
    void stop_skill()
    {
        get_blackboard()->set_var("skill_name","");
        get_blackboard()->set_var("skill_play",false);
		callable_mp(this, &CharacterBodyMain::_stop_skill).call_deferred();
    }

    CharacterBodyMain();
    ~CharacterBodyMain();

protected:
    void load_skeleton();
    void load_mesh(const StringName& part_name,String p_mesh_file_path);
    BTPlayer * get_bt_player();
    void _stop_skill()
    {
        if(btSkillPlayer != nullptr)
        {
            memdelete(btSkillPlayer);
            btSkillPlayer = nullptr;
        }
    }
    
protected:
    void behavior_tree_finished(int last_status);
    void behavior_tree_update(int last_status);

    
    void skill_tree_finished(int last_status);
    void skill_tree_update(int last_status);


protected:
    Skeleton3D *skeleton = nullptr;
    AnimationPlayer *player = nullptr;
    AnimationTree *tree = nullptr;
    mutable BTPlayer *btPlayer = nullptr;
    bool is_skill_stop = false;
    // 技能播放器
    mutable BTPlayer *btSkillPlayer = nullptr;
    // 角色自己的黑板
    Ref<Blackboard> player_blackboard;
    // 骨架配置文件
    String skeleton_res;
    // 动画组配置
    String animation_group;
    // 身體部件列表
    PackedStringArray   partList;
    // 插槽信息
    HashMap<StringName,BodySocket> socket;
    // 身体部件信息
    HashMap<StringName,CharacterBodyPartInstane> bodyPart;
    Ref<CharacterAnimator>    animator;
    


};


#endif