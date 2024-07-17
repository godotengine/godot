
#ifndef BODY_MAIN_H
#define BODY_MAIN_H
#include "scene/resources/packed_scene.h"
#include "scene/animation/animation_player.h"
#include "scene/animation/animation_tree.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/3d/physics/character_body_3d.h"
#include "scene/3d/physics/collision_shape_3d.h"
#include "animator/animation_help.h"
#include "animator/body_animator.h"
#include "character_movement.h"
#include "character_check_area_3d.h"
#include "./character_shape/character_body_part.h"
#include "./character_shape/character_body_prefab.h"
#include "navigation/character_navigation_agent.h"
#include "modules/renik/renik.h"


#include "modules/limboai/bt/bt_player.h"
#include "modules/limboai/bt/tasks/decorators/bt_new_scope.h"

class CharacterAI;
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
// 角色的阵营 枚举
enum CharacterCamp
{
    CharacterCamp_Player,
    CharacterCamp_Enemy,
    CharacterCamp_Friend,
};
class CharacterAILogicNode;
struct CharacterAIContext
{
    Ref<CharacterAILogicNode> logic_node;
    StringName logic_name;
    CharacterCamp camp;
    
};

// 身体主要部件部分
class CharacterBodyMain : public CharacterBody3D {
    GDCLASS(CharacterBodyMain, CharacterBody3D);
    static void _bind_methods();
public:
    void _update_ai();
    void _process_move();
    void _process_animator();
    void _process_animation();
    void _process_ik();

public:
    // 初始化身體
    void init_main_body(String p_skeleton_file_path,StringName p_animation_group);
    void clear_all();
    void _notification( int p_notification );
    CharacterBodyMain();
    ~CharacterBodyMain();

public:
	void set_behavior_tree(const Ref<BehaviorTree> &p_tree)
    {
        get_bt_player()->set_behavior_tree(p_tree);
    }
	Ref<BehaviorTree> get_behavior_tree()  { return get_bt_player()->get_behavior_tree(); };

	void set_blackboard_plan(const Ref<BlackboardPlan> &p_plan)
    {
        init_blackboard_plan(p_plan);
        get_bt_player()->set_blackboard_plan(p_plan);
    }
	Ref<BlackboardPlan> get_blackboard_plan() { return get_bt_player()->get_blackboard_plan(); }
    static void init_blackboard_plan(Ref<BlackboardPlan> p_plan);

	void set_update_mode(int p_mode)
    {
        get_bt_player()->set_update_mode((BTPlayer::UpdateMode)p_mode);
    }
	int get_update_mode() { return (int)(get_bt_player()->get_update_mode()); }

    void set_skeleton_resource(const String& p_skeleton_path);

    String get_skeleton_resource() { return skeleton_res; }

    // 设置黑板
	void set_blackboard(const Ref<Blackboard> &p_blackboard) ;
    // 
    // 可编辑属性,必须初始类返回一个空对象
	Ref<Blackboard> _get_blackboard()  { return player_blackboard; }
	Ref<Blackboard> get_blackboard() 
     {
        if(player_blackboard.is_null())
        {
            player_blackboard.instantiate();
            ERR_PRINT_ED("get_blackboard");
        }
         return player_blackboard; 
    }

	void restart()
    {
        get_bt_player()->restart();
    }
	int get_last_status() { return get_bt_player()->get_last_status(); }

	Ref<BTTask> get_tree_instance() { return get_bt_player()->get_tree_instance(); }

    void set_navigation_agent(const Ref<CharacterNavigationAgent3D> &p_navigation_agent);
    Ref<CharacterNavigationAgent3D> get_navigation_agent();
public:
    void set_main_shape(const Ref<CollisionObject3DConnectionShape>& p_shape) {
        if(mainShape == p_shape)
        {
            return;
        }
        if(mainShape.is_valid())
        {            
            mainShape->set_link_target(nullptr);
        }
        mainShape = p_shape;
        if(mainShape.is_valid())
        {            
            mainShape->set_link_target(this);
        }
    }
    Ref<CollisionObject3DConnectionShape> get_main_shape()
     {
        return mainShape; 
    }
    void set_check_area(const TypedArray<CharacterCheckArea3D> &p_check_area)
    {
        check_area.clear();
        for(int i = 0;i<p_check_area.size();i++)
        {
            check_area.push_back(p_check_area[i]);
        }
        on_update_area();
    }
    TypedArray<CharacterCheckArea3D> get_check_area()
    {
        TypedArray<CharacterCheckArea3D> ret;
        for(uint32_t i = 0;i<check_area.size();i++)
        {
            ret.push_back(check_area[i]);
        }
        return ret;
    }
    Ref<CharacterCheckArea3D> get_check_area_by_name(const StringName &p_name)
    {
        for(uint32_t i = 0;i<check_area.size();i++)
        {
            if(check_area[i]->get_name() == p_name)
            {
                return check_area[i];
            }
        }
        return Ref<CharacterCheckArea3D>();
    }

public:
    // 初始化身體分組信息
    void init_body_part_array(const Array& p_part_array);
    // 身體部位
    void set_body_part(const Dictionary& part);
    Dictionary get_body_part();
    
    void set_body_prefab(const Ref<CharacterBodyPrefab> &p_body_prefab);
    Ref<CharacterBodyPrefab> get_body_prefab();
    void load_prefab();

    // 技能相关
public:
    bool play_skill(String p_skill_name);
    void stop_skill();

public:
    void set_character_ai(const Ref<CharacterAI> &p_ai);
    Ref<CharacterAI> get_character_ai();

    // 动画相关
public:
    void set_animation_library(const Ref<CharacterAnimationLibrary> &p_library) 
    {
        if(animation_library.is_valid() || p_library.is_null())
        {
            return;
        }
        animation_library = p_library;
    }
    Ref<CharacterAnimationLibrary> get_animation_library()
    {
        if(animation_library.is_valid())
        {
            animation_library.instantiate();
        }
        return animation_library;
    }
    void set_animator(const Ref<CharacterAnimator> &p_animator)
    {
        if(animator.is_valid() || p_animator.is_null())
        {
            return;
        }
        animator = p_animator;
		animator->set_body(this);
    }

    Ref<CharacterAnimator> get_animator()
    {
        if(animator.is_valid())
        {
            animator.instantiate();
            animator->set_body(this);
        }
        return animator;
    }
    // 可编辑属性,必须初始类返回一个空对象
    Ref<CharacterAnimator> _get_animator()
    {
        return animator;
    }

    void set_speed(float p_speed)
    {
        Ref<Blackboard> blackboard = _get_blackboard();
        blackboard->set_var(StringName("Speed"),p_speed);

    }

    void set_is_moveing(bool p_is_moveing)
    {
        Ref<Blackboard> blackboard = _get_blackboard();
        blackboard->set_var(StringName("IsMoveing"),p_is_moveing);
    }

    void set_move_target(const Vector3& target)
    {
        Ref<Blackboard> blackboard = _get_blackboard();
        blackboard->set_var(StringName("MoveTarget"),target);

    }
    // 设置是否蹲下
    void set_is_jump(bool p_is_jump)
    {
        Ref<Blackboard> blackboard = _get_blackboard();
        blackboard->set_var(StringName("IsJump"),p_is_jump);
    }

    // 更新角色的朝向
    void update_forward(const Vector3 & old_forward,const Vector3& curr_forward,Blackboard * p_blocakboard)
    {
        Quaternion old_quat = Quaternion(old_forward,curr_forward);
        Vector3 angle = old_quat.get_euler();
        p_blocakboard->set_var(StringName("Pitch"),angle.x);
        p_blocakboard->set_var(StringName("Yaw"),angle.y);

        p_blocakboard->set_var(StringName("OldForward"),old_forward);
        p_blocakboard->set_var(StringName("CurrForward"),curr_forward);
    }
    void change_animator_state(StringName p_state_name)
    {
        get_blackboard()->set_var("CurrState",p_state_name);
        if(animator.is_valid())
        {
            animator->change_state(p_state_name);
        }
    }
    void set_ik(const Ref<RenIK>& p_ik)
    {
        if(p_ik == ik)
        {
            return;
        }
        ik = p_ik;
        Skeleton3D * skeleton = get_skeleton();
        if(skeleton && ik.is_valid())
        {
            ik->_initialize(skeleton);
        }
    }
    Ref<RenIK> get_ik()
    {
        return ik;
    }
    Skeleton3D* get_skeleton()
    {
        Skeleton3D* skeleton = Object::cast_to<Skeleton3D>(ObjectDB::get_instance(skeletonID));
        return skeleton;
    }
	GDVIRTUAL0(_update_player_position)
public:
    
	virtual void input(const Ref<InputEvent> &p_event) override
    {
    }
	virtual void shortcut_input(const Ref<InputEvent> &p_key_event) override
    {

    }
	virtual void unhandled_input(const Ref<InputEvent> &p_event)override
    {

    }
	virtual void unhandled_key_input(const Ref<InputEvent> &p_key_event)override
    {

    }
protected:
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

    void _update(double p_delta);

protected:
	virtual void update_world_transform(const Transform3D & trans) override
	{
        if(character_movement.is_valid())
        {
            character_movement->set_world_transform(trans);
        }
        else
        {
		    set_global_transform(trans);
        }
	}
    void on_update_area()
    {
        for(uint32_t i = 0; i < check_area.size();++i)
        {
            if(check_area[i].is_valid())
            {
                check_area[i]->set_body_main(this);
            }
        }
    }
    void on_blockbora_value_changed(Blackboard *p_blackboard,const StringName& p_property)
    {
        String name = p_property;
        if(name.begins_with("anim_state/"))
        {
            if(animator.is_valid())
            {
                // 動畫狀態相關屬性，通知執行當前狀態匹配的動作
            }
        }
    }
protected:
    LocalVector<Ref<CharacterCheckArea3D>> check_area;
    Ref<CollisionObject3DConnectionShape> mainShape;
    // 初始化数据
    Dictionary init_data;


    mutable BTPlayer *btPlayer = nullptr;
    bool is_skill_stop = false;
    // 技能播放器
    mutable BTPlayer *btSkillPlayer = nullptr;


    // 角色自己的黑板
    Ref<Blackboard> player_blackboard;
    Ref<CharacterMovement> character_movement;
    Ref<CharacterNavigationAgent3D> character_agent;

    CharacterAIContext ai_context;
    Ref<CharacterAI> character_ai;


    
    // 骨架配置文件
    String skeleton_res;
    ObjectID skeletonID;
    Ref<CharacterAnimationLibrary> animation_library;
    Ref<CharacterAnimator>    animator;
    // 动画组配置
    String animation_group;
    // 角色的IK信息
    Ref<RenIK> ik;


    Ref<CharacterBodyPrefab> body_prefab;
    // 身体部件信息
    HashMap<StringName,Ref<CharacterBodyPartInstane>> bodyPart;
    // 身體部件列表
    PackedStringArray   partList;
    // 插槽信息
    HashMap<StringName,BodySocket> socket;
};

#endif
