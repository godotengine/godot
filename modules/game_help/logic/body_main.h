
#ifndef BODY_MAIN_H
#define BODY_MAIN_H
#include "scene/resources/packed_scene.h"
#include "scene/animation/animation_player.h"
#include "scene/animation/animation_tree.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/3d/physics/character_body_3d.h"
#include "scene/3d/physics/collision_shape_3d.h"
#include "scene/3d/label_3d.h"

#include "animator/animation_help.h"
#include "animator/body_animator.h"
#include "character_movement.h"
#include "character_check_area_3d.h"
#include "./character_shape/character_body_part.h"
#include "./character_shape/character_body_prefab.h"
#include "navigation/character_navigation_agent.h"
#include "modules/renik/renik.h"
#include "beehave/beehave_tree.h"


#include "core/object/worker_thread_pool.h"

#include "./blackboard/blackboard_plan.h"

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
class CharacterAIContext : public RefCounted
{
    GDCLASS(CharacterAIContext,RefCounted);
public:
	CharacterAIContext();
public:
    Ref<CharacterAILogicNode> logic_node;
    StringName logic_name;
    CharacterCamp camp;
    // 行为树运行上下文,只有逻辑节点使用,每个逻辑节点执行结束后悔进行重置
    Ref<BeehaveRuncontext> beehave_run_context;
    
};
// 人形骨骼显示
class HumanBoneVisble  : public Node3D
{
    GDCLASS(HumanBoneVisble,Node3D);

public:
    void init(Skeleton3D* p_skeleton, Ref<CharacterBoneMap> p_bone_map) {
        clear();
        skeleton = p_skeleton;
        bone_map = p_bone_map;
        if(skeleton == nullptr || bone_map.is_null()) {
            return;
        }
		Dictionary bm = p_bone_map->get_bone_map();
		Array keys = bm.keys();
        for(auto E : keys) {
			String value = bm[E];
			int bone_index = skeleton->find_bone(value);
            if(bone_index  == -1) {
                continue;
            }
            Label3D* label = memnew(Label3D);
            label->set_dont_save(true);
            label->set_global_position(skeleton->get_bone_global_pose(bone_index).origin);
            label->set_text(get_bone_label(value));
			label->set_scale(Vector3(0.1, 0.1, 0.1));
            add_child(label);
            //label->set_owner(get_tree()->get_edited_scene_root());
            bone_label[bone_index] = label;
        }

    }
    String get_bone_label(StringName p_bone) {
        static HashMap<String,String> label_map = {
            {"Hips",L"臀部"},
            {"Spine",L"脊柱"},
            {"Chest",L"颈部"},
            {"UpperChest",L"上颈部"},
            {"Neck",L"颈部"},
            {"Head",L"头部"},
            {"Jaw",L"下巴"},

            {"LeftShoulder",L"左肩"},
            {"RightShoulder",L"右肩"},

            {"LeftUpperArm",L"左上臂"},
            {"RightUpperArm",L"右上臂"},

            {"LeftLowerArm",L"左下臂"},
            {"RightLowerArm",L"右下臂"},

            {"LeftHand",L"左手"},
            {"RightHand",L"右手"},

            {"LeftUpperLeg",L"左上腿"},
            {"RightUpperLeg",L"右上腿"},

            {"LeftLowerLeg",L"左下腿"},
            {"RightLowerLeg",L"右下腿"},

            {"LeftFoot",L"左脚"},
            {"RightFoot",L"右脚"},

            {"LeftEye",L"左眼"},
            {"RightEye",L"右眼"},

            {"LeftToes",L"左足"},
            {"RightToes",L"右足"},

            {"LeftThumbMetacarpal",L"左拇指"},
            {"LeftThumbProximal",L"左拇指近端"},
            {"LeftThumbDistal",L"左拇指远端"},

            {"LeftIndexProximal",L"左食指近端"},
            {"LeftIndexIntermediate",L"左食指中间"},
            {"LeftIndexDistal",L"左食指远端"},

            {"LeftMiddleProximal",L"左中指近端"},
            {"LeftMiddleIntermediate",L"左中指中间"},
            {"LeftMiddleDistal",L"左中指远端"},

            {"LeftRingProximal",L"左无名指近端"},
            {"LeftRingIntermediate",L"左无名指中间"},
            {"LeftRingDistal",L"左无名指远端"},

            {"LeftLittleProximal",L"左小拇指近端"},
            {"LeftLittleIntermediate",L"左小拇指中间"},
            {"LeftLittleDistal",L"左小拇指远端"},


            {"RightThumbMetacarpal",L"右拇指"},
            {"RightThumbProximal",L"右拇指近端"},
            {"RightThumbDistal",L"右拇指远端"},

            {"RightIndexProximal",L"右食指近端"},
            {"RightIndexIntermediate",L"右食指中间"},
            {"RightIndexDistal",L"右食指远端"},

            {"RightMiddleProximal",L"右中指近端"},
            {"RightMiddleIntermediate",L"右中指中间"},
            {"RightMiddleDistal",L"右中指远端"},

            {"RightRingProximal",L"右无名指近端"},
            {"RightRingIntermediate",L"右无名指中间"},
            {"RightRingDistal",L"右无名指远端"},

            {"RightLittleProximal",L"右小拇指近端"},
            {"RightLittleIntermediate",L"右小拇指中间"},
            {"RightLittleDistal",L"右小拇指远端"},

        };
        if(label_map.has(p_bone)) {
            return label_map[p_bone];
        }
        return p_bone;
    }
    void clear() {
        
        for(auto E : bone_label) {
            remove_child(E.value);
            E.value->queue_free();
        }
		bone_label.clear();
        skeleton = nullptr;
    }

    void update() {
        if(skeleton == nullptr) {
            return;
        }
        for(auto E : bone_label) {
            E.value->set_position(skeleton->get_bone_global_pose(E.key).origin);
        }
    }

protected:
    Skeleton3D* skeleton = nullptr;
    Ref<CharacterBoneMap> bone_map;
    HashMap<int, Label3D*> bone_label;
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
    void init();
    void clear_all();
    void _notification( int p_notification );
    CharacterBodyMain();
    ~CharacterBodyMain();

public:

	void set_blackboard_plan(const Ref<BlackboardPlan> &p_plan)
    {
        blackboard_plan = p_plan;
        init_ai_context();
        init_blackboard_plan(p_plan);
    }
	Ref<BlackboardPlan> get_blackboard_plan() { return blackboard_plan; }

	// 可编辑属性,必须初始类返回一个空对象
	Ref<Blackboard> _get_blackboard()  { return player_blackboard; }
	Ref<Blackboard> get_blackboard() 
     {
        if(player_blackboard.is_null())
        {
            player_blackboard.instantiate();
            //ERR_PRINT_ED("get_blackboard");
        }
         return player_blackboard; 
    }

	void restart()
    {
    }

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
    void init_ai_context();
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
        if(p_animator.is_null())
        {
            return;
        }
        if(animator.is_valid())
        {
            animator->set_body(nullptr);
        }
        animator = p_animator;
		animator->set_body(this);
    }

    Ref<CharacterAnimator> get_animator()
    {
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
	AABB get_mesh_aabb() {
		AABB aabb;
		for (auto& part : bodyPart) {
			aabb = aabb.merge(part.value->get_mesh_aabb());
		}
		return aabb;
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
    void behavior_tree_finished(int last_status);
    void behavior_tree_update(int last_status);

    
    void skill_tree_finished(int last_status);
    void skill_tree_update(int last_status);

    void _update(double p_delta);

	void _init_body();
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
public:
    void set_editor_form_mesh_file_path(const String& p_file_path)
    {
        editor_form_mesh_file_path = p_file_path;
    }
    String get_editor_form_mesh_file_path()
    {
        return editor_form_mesh_file_path;
    }

    void set_editor_run_ai(bool p_run_ai)
    {
        ai_context.unref();
        run_ai = p_run_ai;
    }

    bool get_editor_run_ai()
    {
        return run_ai;
    }
	static Ref<CharacterBodyPrefab> build_prefab(const String& mesh_path);
    DECL_MEMBER_BUTTON(editor_build_form_mesh_file_path);

    // 生成动画资产帮助类
    void set_editor_ref_bone_map(Ref<CharacterBoneMap> p_bone_map) {
        editor_ref_bone_map = p_bone_map;
    }

    Ref<CharacterBoneMap> get_editor_ref_bone_map() {
        return editor_ref_bone_map;
    }
    void set_editor_animation_file_path(const String& p_file_path)
    {
		editor_animation_file_path = p_file_path;
    }

    String get_editor_animation_file_path()
    {
        return editor_animation_file_path;
    }
    Ref<CharacterBoneMap> editor_ref_bone_map;
    String editor_animation_file_path;
    DECL_MEMBER_BUTTON(editor_build_animation);

    void set_editor_animation_speed(float p_speed) {
        editor_animation_speed = p_speed;
    }

    float get_editor_animation_speed() {
        return editor_animation_speed;
    }

    void set_editor_pause_animation(bool p_pause) {
        editor_pause_animation = p_pause;
    }

    bool get_editor_pause_animation() {
        return editor_pause_animation;
    }

    void set_play_animation(const Ref<Animation>& p_play_animation)
    {
        play_animation = p_play_animation;
        update_bone_visble();
    }

    Ref<Animation> get_play_animation()
    {
        return play_animation;
    }

	bool editor_show_mesh = true;

    void set_editor_show_mesh(bool p_show) {
        editor_show_mesh = p_show;
        for(auto& part : bodyPart) {
            part.value->set_show_mesh(p_show);
        }
    }

    bool get_editor_show_mesh() {
        return editor_show_mesh;
    }


    Ref<Animation> play_animation;
    DECL_MEMBER_BUTTON(editor_play_select_animation);
    void update_bone_visble();

public:
    static ObjectID& get_curr_editor_player();
    // 获取当前编辑的角色
    static CharacterBodyMain* get_current_editor_player()
    {
        return Object::cast_to<CharacterBodyMain>(ObjectDB::get_instance(get_curr_editor_player()));
    }
    static void init_blackboard_plan(Ref<BlackboardPlan> p_plan);
protected:
    LocalVector<Ref<CharacterCheckArea3D>> check_area;
    Ref<CollisionObject3DConnectionShape> mainShape;
    // 初始化数据
    Dictionary init_data;
    // 角色的编辑器模型
    String editor_form_mesh_file_path;


    bool is_skill_stop = false;

    Ref<BlackboardPlan> blackboard_plan;
    // 角色自己的黑板
    Ref<Blackboard> player_blackboard;
    Ref<CharacterMovement> character_movement;
    Ref<CharacterNavigationAgent3D> character_agent;

    Ref<CharacterAIContext> ai_context;
    Ref<CharacterAI> character_ai;


    
    // 骨架配置文件
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
    // 插槽信息
    HashMap<StringName,BodySocket> socket;

    HumanBoneVisble* bone_label = nullptr;

    bool run_ai = true;
    bool editor_pause_animation = false;
    float editor_animation_speed = 1.0;
};

#endif
