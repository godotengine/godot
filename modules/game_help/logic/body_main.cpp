#include "body_main.h"
#include "data_table_manager.h"
#include "scene/3d/path_3d.h"
#include "scene/resources/3d/capsule_shape_3d.h"
#include "character_ai/character_ai.h"
#include "character_manager.h"
#include "core/io/file_access.h"
#include "core/io/dir_access.h"

CharacterAIContext::CharacterAIContext()
{
	beehave_run_context.instantiate();
}

void CharacterBodyMain::init()
{
    if(character_ai.is_null())
    {
		character_ai.instantiate();
    }
	character_ai->init();
    if(animator.is_null())
    {
        animator.instantiate();
    }
    animator->set_body(this);
    animator->init();

    // 创建外形
    if(mainShape.is_null())
    {
        mainShape.instantiate();
        Ref<CapsuleShape3D> shape;
        shape.instantiate();
        shape->set_radius(0.5f);
        shape->set_height(2.0f);

        mainShape->set_shape(shape);
		mainShape->set_position(Vector3(0, 1, 0));
        mainShape->set_link_target(this);        
    }
}
void CharacterBodyMain::clear_all()
{
    if(bone_label != nullptr) {
        bone_label->clear();
    }
    Skeleton3D* skeleton = Object::cast_to<Skeleton3D>(ObjectDB::get_instance(skeletonID));
    if(skeleton == nullptr)
    {
        return;
    }
    if(skeleton)
    {
        remove_child(skeleton);
        skeleton->queue_free();
        skeleton = nullptr;
        
    }
    bodyPart.clear();
    if(animator.is_valid()) {
        animator->set_body(nullptr);
    }
}
void CharacterBodyMain::_notification( int p_notification )
{
    if(character_agent.is_valid())
    {
        character_agent->set_body_main(this);
        character_agent->_notification(p_notification);

    }
	switch (p_notification) {
        case NOTIFICATION_ENTER_TREE: {
			if(CharacterManager::get_singleton() != nullptr)
				CharacterManager::get_singleton()->register_character(this);
        } break;
        case NOTIFICATION_EXIT_TREE: {
			if (CharacterManager::get_singleton() != nullptr)
				CharacterManager::get_singleton()->unregister_character(this);            
        }
        break;
    }

}
void CharacterBodyMain::init_ai_context()
{
	if (ai_context.is_null()) {
		ai_context.instantiate();
		ai_context->beehave_run_context->blackboard = get_blackboard();
		ai_context->beehave_run_context->actor = this;
	}

}
void CharacterBodyMain::_update(double p_delta)
{

    // 更新玩家位置
    GDVIRTUAL_CALL(_update_player_position);
    for(uint32_t i = 0; i < check_area.size();++i)
    {
        if(check_area[i].is_valid())
        {
            check_area[i]->update_world_move(get_global_position());
        }
    }
    _process_move();

}
void CharacterBodyMain::_update_ai()
{
#ifdef TOOLS_ENABLED
    if(!run_ai)
    {
        return;
    }
#endif
	init_ai_context();
    if(character_ai.is_valid())
    {
        character_ai->execute(this,ai_context.ptr());
    }
}
void CharacterBodyMain::_process_animator(double time_delta)
{
    if(editor_pause_animation) {
        return ;
    }
    if(animator.is_valid())
    {
        animator->_thread_update_animator(time_delta * editor_animation_speed);
    }
}
void CharacterBodyMain::_process_animation()
{
    if(animator.is_valid())
    {
        animator->_thread_update_animation(get_process_delta_time());
    }
}
void CharacterBodyMain::_process_ik()
{
    float delta = get_process_delta_time();
    update_track_target();
    if(ik.is_valid())
    {
        ik->update_ik();
        ik->update_placement(delta);
    }

    if(bone_label != nullptr) 
    {
        bone_label->update();
    }
}

void CharacterBodyMain::update_track_target() {

    if(track_target.is_empty()) {
        return;
    }
    
    CharacterBodyMain* src_track_target = Object::cast_to<CharacterBodyMain>(get_parent()->find_child(track_target));
    if(src_track_target == nullptr) {
        return;
    }
    LocalVector<String> human_bones = get_human_bones();
    Skeleton3D * skeleton = Object::cast_to<Skeleton3D>(ObjectDB::get_instance(skeletonID));
    
    Skeleton3D * src_skeleton = src_track_target->get_skeleton();
    if(skeleton == nullptr || src_skeleton == nullptr) {
        return;
    }
	temp_last_bone_pose.resize(human_bones.size());
    LocalVector<Vector3> temp_src_bone_pose_angle;
    LocalVector<Vector3> temp_src_bone_rest_angle;
    LocalVector<Vector3> temp_bone_rest_angle;
    temp_src_bone_pose_angle.resize(human_bones.size());
    temp_src_bone_rest_angle.resize(human_bones.size());
    temp_bone_rest_angle.resize(human_bones.size());
	for (int i = 0; i < human_bones.size(); i++) {
		int bone_index = skeleton->find_bone(human_bones[i]);
		int src_bone_index = src_skeleton->find_bone(human_bones[i]);
		if (bone_index == -1) {
			continue;
		}
        temp_src_bone_pose_angle[i] = src_skeleton->get_bone_pose(src_bone_index).basis.get_euler() * (float)(180.0 / Math_PI);
        temp_src_bone_rest_angle[i] = src_skeleton->get_bone_rest(src_bone_index).basis.get_euler() * (float)(180.0 / Math_PI);
        temp_bone_rest_angle[i] = skeleton->get_bone_rest(bone_index).basis.get_euler() * (float)(180.0 / Math_PI);
		temp_last_bone_pose[i] = src_skeleton->get_bone_pose(src_bone_index).basis.get_rotation_quaternion() * src_skeleton->get_bone_rest(src_bone_index).basis.get_rotation_quaternion().inverse();
        temp_last_bone_pose[i] = temp_last_bone_pose[i] * skeleton->get_bone_rest(bone_index).basis.get_rotation_quaternion();
	}
    for (int i = 0; i < human_bones.size(); i++) {
        int bone_index = skeleton->find_bone(human_bones[i]);
        int src_bone_index = src_skeleton->find_bone(human_bones[i]);
        if(bone_index == -1 || src_bone_index == -1) {
            continue;
        }
        skeleton->set_bone_pose_rotation(bone_index, temp_last_bone_pose[i]);
    }
}
void CharacterBodyMain::_process_move()
{
    // 处理角色移动
    bool is_walk = get_blackboard()->get_var("move/using_navigation_target",false);
    if(is_walk )
    {
        // 处理导航行走
        if(character_agent.is_valid())
        {
            if(is_walk)
            {
                Vector3 target_pos = get_blackboard()->get_var("move/navigation_target_pos",Vector3());
                // 设置角色移动速度
                character_agent->set_velocity(get_velocity());
                character_agent->set_target_position(target_pos);
            }
            else
            {
                character_agent->set_navigation_finished(true);
            }
        }
    }
    else
    {
        move_and_slide();
    }

}
void CharacterBodyMain::set_character_ai(const Ref<CharacterAI> &p_ai)
{
    character_ai = p_ai;
}
Ref<CharacterAI> CharacterBodyMain::get_character_ai()
{
    return character_ai;
} 

void CharacterBodyMain::behavior_tree_finished(int last_status)
{
    emit_signal("behavior_tree_finished", last_status);
}
void CharacterBodyMain::behavior_tree_update(int last_status)
{
    emit_signal("updated", last_status);
}
void CharacterBodyMain::skill_tree_finished(int last_status)
{
    emit_signal("skill_tree_finished", last_status);
}
void CharacterBodyMain::skill_tree_update(int last_status)
{
    emit_signal("skill_tree_updated", last_status);
}
void CharacterBodyMain::set_navigation_agent(const Ref<CharacterNavigationAgent3D> &p_navigation_agent)
{
    if(character_agent == p_navigation_agent)
    {
        return;
    }
    if(character_agent.is_valid())
    {
        character_agent->set_body_main(nullptr);
    }
    character_agent = p_navigation_agent;
    if(character_agent.is_valid())
    {
        character_agent->set_body_main(this);
    }
}
Ref<CharacterNavigationAgent3D> CharacterBodyMain::get_navigation_agent()
{
    return character_agent;
}



void CharacterBodyMain::set_body_prefab(const Ref<CharacterBodyPrefab> &p_body_prefab)
{
    if(p_body_prefab.is_null() || p_body_prefab == body_prefab)
    {
        return;
    }
    if(body_prefab.is_valid())
    {
        body_prefab->disconnect_changed(callable_mp(this, &CharacterBodyMain::load_prefab));
    }
    body_prefab = p_body_prefab;
    
    load_prefab();


}
void CharacterBodyMain::load_prefab()
{
	_init_body();
}
void CharacterBodyMain::_init_body()
{
	// 存储一下引用,避免当前再次引用的资产不会立即释放
	clear_all();
	HashMap<StringName, Ref<CharacterBodyPartInstane>> old_part = bodyPart;
	if (body_prefab.is_valid())
	{
		body_prefab->connect_changed(callable_mp(this, &CharacterBodyMain::load_prefab));
		Ref<PackedScene> scene;
		String skeleton_res = body_prefab->skeleton_path;
		if (skeleton_res.begins_with("res://"))
			scene = ResourceLoader::load(skeleton_res);
		else
			scene = ResourceLoader::load(CharacterManager::get_singleton()->get_skeleton_root_path().path_join(skeleton_res));
		if (!scene.is_valid())
		{
			ERR_FAIL_MSG("load skeleton failed:" + skeleton_res);
			skeleton_res = "";
			return;
		}
		Node* ins = scene->instantiate(PackedScene::GEN_EDIT_STATE_INSTANCE);
		if (ins == nullptr) {
			ERR_FAIL_MSG("init skeleton instantiate failed:" + skeleton_res);
			skeleton_res = "";
			return;
		}
		Skeleton3D* skeleton = Object::cast_to<Skeleton3D>(ins);
		if (skeleton == nullptr)
		{
			ERR_FAIL_MSG("scene is not Skeleton3D:" + skeleton_res);
			skeleton->queue_free();
			skeleton_res = "";
			return;
		}
		skeleton->set_name("Skeleton3D");

		add_child(skeleton);
		skeleton->set_owner(this);
		skeleton->set_dont_save(true);


		if (skeleton && ik.is_valid())
		{
			ik->_initialize(skeleton);
		}
		skeletonID = skeleton->get_instance_id();
		// 
		TypedArray<CharacterBodyPart> part_array = body_prefab->load_part();
		int size = part_array.size();
		for (int i = 0; i < size; i++)
		{

			Ref<CharacterBodyPartInstane> p;
			p.instantiate();
			p->set_skeleton(skeleton);
			p->set_part(part_array[i]);
            p->set_show_mesh(editor_show_mesh);
			Ref< CharacterBodyPart> part = part_array[i];
			bodyPart[part->get_name()] = p;
		}
        if(animator.is_valid()) {
            animator->set_body(this);
        }
		update_bone_visble();
		notify_property_list_changed();
	}

}
Ref<CharacterBodyPrefab> CharacterBodyMain::get_body_prefab()
{
    return body_prefab;
}



CharacterBodyMain::CharacterBodyMain()
{
    character_movement.instantiate();
}
CharacterBodyMain::~CharacterBodyMain()
{
    
    for(uint32_t i = 0; i < check_area.size();++i)
    {
        if(check_area[i].is_valid())
        {
            check_area[i]->set_body_main(nullptr);
        }
    }
    check_area.clear();
}

void CharacterBodyMain::_bind_methods()
{
    
	ClassDB::bind_method(D_METHOD("restart"), &CharacterBodyMain::restart);

	ClassDB::bind_method(D_METHOD("set_blackboard_plan", "plan"), &CharacterBodyMain::set_blackboard_plan);
	ClassDB::bind_method(D_METHOD("get_blackboard_plan"), &CharacterBodyMain::get_blackboard_plan);


    ClassDB::bind_method(D_METHOD("set_navigation_agent", "navigation_agent"), &CharacterBodyMain::set_navigation_agent);
    ClassDB::bind_method(D_METHOD("get_navigation_agent"), &CharacterBodyMain::get_navigation_agent);

    ClassDB::bind_method(D_METHOD("set_animation_library", "animation_library"), &CharacterBodyMain::set_animation_library);
    ClassDB::bind_method(D_METHOD("get_animation_library"), &CharacterBodyMain::get_animation_library);

    ClassDB::bind_method(D_METHOD("set_animator", "animator"), &CharacterBodyMain::set_animator);
    ClassDB::bind_method(D_METHOD("get_animator"), &CharacterBodyMain::get_animator);

    ClassDB::bind_method(D_METHOD("set_ik", "ik"), &CharacterBodyMain::set_ik);
    ClassDB::bind_method(D_METHOD("get_ik"), &CharacterBodyMain::get_ik);

    ClassDB::bind_method(D_METHOD("set_main_shape", "shape"), &CharacterBodyMain::set_main_shape);
    ClassDB::bind_method(D_METHOD("get_main_shape"), &CharacterBodyMain::get_main_shape);

    ClassDB::bind_method(D_METHOD("set_check_area", "check_area"), &CharacterBodyMain::set_check_area);
    ClassDB::bind_method(D_METHOD("get_check_area"), &CharacterBodyMain::get_check_area);
    ClassDB::bind_method(D_METHOD("get_check_area_by_name", "name"), &CharacterBodyMain::get_check_area_by_name);


    ClassDB::bind_method(D_METHOD("set_body_prefab", "body_prefab"), &CharacterBodyMain::set_body_prefab);
    ClassDB::bind_method(D_METHOD("get_body_prefab"), &CharacterBodyMain::get_body_prefab);


    ClassDB::bind_method(D_METHOD("set_character_ai", "ai"), &CharacterBodyMain::set_character_ai);
    ClassDB::bind_method(D_METHOD("get_character_ai"), &CharacterBodyMain::get_character_ai);

    
    ClassDB::bind_method(D_METHOD("set_editor_form_mesh_file_path", "editor_form_mesh_file_path"), &CharacterBodyMain::set_editor_form_mesh_file_path);
    ClassDB::bind_method(D_METHOD("get_editor_form_mesh_file_path"), &CharacterBodyMain::get_editor_form_mesh_file_path);

    ClassDB::bind_method(D_METHOD("set_editor_ref_bone_map", "editor_ref_bone_map"), &CharacterBodyMain::set_editor_ref_bone_map);
    ClassDB::bind_method(D_METHOD("get_editor_ref_bone_map"), &CharacterBodyMain::get_editor_ref_bone_map);

    ClassDB::bind_method(D_METHOD("set_editor_animation_file_path", "path"), &CharacterBodyMain::set_editor_animation_file_path);
    ClassDB::bind_method(D_METHOD("get_editor_animation_file_path"), &CharacterBodyMain::get_editor_animation_file_path);


	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "blackboard_plan", PROPERTY_HINT_RESOURCE_TYPE, "BlackboardPlan", PROPERTY_USAGE_DEFAULT ), "set_blackboard_plan", "get_blackboard_plan");



    ADD_GROUP("editor", "editor_");

    ClassDB::bind_method(D_METHOD("set_editor_show_mesh", "editor_show_mesh"), &CharacterBodyMain::set_editor_show_mesh);
    ClassDB::bind_method(D_METHOD("get_editor_show_mesh"), &CharacterBodyMain::get_editor_show_mesh);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL, "editor_show_mesh", PROPERTY_HINT_NONE, "", PROPERTY_USAGE_EDITOR), "set_editor_show_mesh", "get_editor_show_mesh");

	ADD_PROPERTY(PropertyInfo(Variant::STRING, "editor_form_mesh_file_path"), "set_editor_form_mesh_file_path", "get_editor_form_mesh_file_path");
    ADD_MEMBER_BUTTON(editor_build_form_mesh_file_path,L"根据模型初始化",CharacterBodyMain);

    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "editor_ref_bone_map", PROPERTY_HINT_RESOURCE_TYPE, "CharacterBoneMap", PROPERTY_USAGE_DEFAULT ), "set_editor_ref_bone_map", "get_editor_ref_bone_map");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "editor_animation_file_path",PROPERTY_HINT_FILE,"tres,*.tres"), "set_editor_animation_file_path", "get_editor_animation_file_path");

    ADD_MEMBER_BUTTON(editor_build_animation,L"构建动画文件信息",CharacterBodyMain);

    
    ClassDB::bind_method(D_METHOD("set_play_animation", "play_animation"), &CharacterBodyMain::set_play_animation);
    ClassDB::bind_method(D_METHOD("get_play_animation"), &CharacterBodyMain::get_play_animation);
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "editor_play_animation", PROPERTY_HINT_RESOURCE_TYPE, "Animation"), "set_play_animation", "get_play_animation");
    ADD_MEMBER_BUTTON(editor_play_select_animation,L"播放动画",CharacterBodyMain);


    ClassDB::bind_method(D_METHOD("set_track_target", "track_target"), &CharacterBodyMain::set_track_target);
    ClassDB::bind_method(D_METHOD("get_track_target"), &CharacterBodyMain::get_track_target);
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "editor_track_target", PROPERTY_HINT_NONE, "",PROPERTY_USAGE_EDITOR), "set_track_target", "get_track_target");



    ADD_GROUP("show", "");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "body_prefab", PROPERTY_HINT_RESOURCE_TYPE, "CharacterBodyPrefab",PROPERTY_USAGE_DEFAULT ), "set_body_prefab", "get_body_prefab");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "animator", PROPERTY_HINT_RESOURCE_TYPE, "CharacterAnimator",PROPERTY_USAGE_DEFAULT ), "set_animator", "get_animator"); 
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "animation_library", PROPERTY_HINT_RESOURCE_TYPE, "CharacterAnimationLibrary",PROPERTY_USAGE_DEFAULT ), "set_animation_library", "get_animation_library");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "ik", PROPERTY_HINT_RESOURCE_TYPE, "RenIK",PROPERTY_USAGE_DEFAULT ), "set_ik", "get_ik");


    ADD_GROUP("logic", "");

    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "character_ai", PROPERTY_HINT_RESOURCE_TYPE, "CharacterAI"), "set_character_ai", "get_character_ai");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "navigation_agent", PROPERTY_HINT_RESOURCE_TYPE, "CharacterNavigationAgent3D"), "set_navigation_agent", "get_navigation_agent");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "main_shape", PROPERTY_HINT_RESOURCE_TYPE, "CollisionObject3DConnectionShape",PROPERTY_USAGE_DEFAULT), "set_main_shape", "get_main_shape");
    ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "check_area", PROPERTY_HINT_RESOURCE_TYPE, MAKE_RESOURCE_TYPE_HINT("CharacterCheckArea3D"),PROPERTY_USAGE_DEFAULT), "set_check_area", "get_check_area");


	ADD_SIGNAL(MethodInfo("behavior_tree_finished", PropertyInfo(Variant::INT, "status")));
	ADD_SIGNAL(MethodInfo("behavior_tree_updated", PropertyInfo(Variant::INT, "status")));
    
	ADD_SIGNAL(MethodInfo("skill_tree_finished", PropertyInfo(Variant::INT, "status")));
	ADD_SIGNAL(MethodInfo("skill_tree_updated", PropertyInfo(Variant::INT, "status")));

    

}

/*********************************************************************************************************************************************************************************************************************************/


ObjectID& CharacterBodyMain::get_curr_editor_player()
{
    static ObjectID curr_editor_player;
    return curr_editor_player;
}
void CharacterBodyMain::init_blackboard_plan(Ref<BlackboardPlan> p_plan)
{
    if(!p_plan.is_valid())
    {
        return;
    }
    if(!p_plan->has_var("prop/curr_life"))
    {
        p_plan->add_var("prop/curr_life",BBVariable(Variant::FLOAT,0.0f));
    }
    if(!p_plan->has_var("prop/max_life"))
    {
        p_plan->add_var("prop/max_life",BBVariable(Variant::FLOAT,0.0f));
    }
    if(!p_plan->has_var("ai/curr_logic_node_name"))
    {
        p_plan->add_var("ai/curr_logic_node_name",BBVariable(Variant::STRING_NAME,StringName()));
    }


    if(!p_plan->has_var("move/look_target_pos"))
    {
        p_plan->add_var("move/look_target_pos",BBVariable(Variant::VECTOR3,Vector3()));
    }
    if(!p_plan->has_var("move/move_path"))
    {
        p_plan->add_var("move/move_path",BBVariable(Variant::OBJECT,Ref<Curve3D>()));
    }
    if(!p_plan->has_var("move/atack_target_pos"))
    {
        p_plan->add_var("move/atack_target_pos",BBVariable(Variant::VECTOR3,Vector3()));
    }
    if(!p_plan->has_var("move/navigation_target_pos"))
    {
        p_plan->add_var("move/navigation_target_pos",BBVariable(Variant::VECTOR3,Vector3()));
    }
    if(!p_plan->has_var("move/using_navigation_target"))
    {
        p_plan->add_var("move/using_navigation_target",BBVariable(Variant::BOOL,false));
    }

    if(!p_plan->has_var("move/request"))
    {
        p_plan->add_var("move/request",BBVariable(Variant::BOOL,false));
    }

    if(!p_plan->has_var("move/speed"))
    {
        p_plan->add_var("move/speed",BBVariable(Variant::FLOAT,0));
    }
    if(!p_plan->has_var("move/curr_speed"))
    {
        p_plan->add_var("move/curr_speed",BBVariable(Variant::FLOAT,0));
    }
    // 出生点
    if(!p_plan->has_var("move/birth_point"))
    {
        p_plan->add_var("move/birth_point",BBVariable(Variant::FLOAT,0));
    }
    // 巡逻范围
    if(!p_plan->has_var("move/patrol_range"))
    {
        p_plan->add_var("move/patrol_range",BBVariable(Variant::FLOAT,0));
    }

    if(!p_plan->has_var("phys/is_ground"))
    {
        p_plan->add_var("phys/is_ground",BBVariable(Variant::BOOL,false));
    }
    if(!p_plan->has_var("phys/to_ground_distance"))
    {
        p_plan->add_var("phys/to_ground_distance",BBVariable(Variant::FLOAT,0));
    }
    if(!p_plan->has_var("phys/ground_pos"))
    {
        p_plan->add_var("phys/ground_pos",BBVariable(Variant::VECTOR3,Vector3()));
    }
    if(!p_plan->has_var("phys/ground_normal"))
    {
        p_plan->add_var("phys/ground_normal",BBVariable(Variant::VECTOR3,Vector3()));
    }
    if(!p_plan->has_var("phys/ground_object_id"))
    {
        p_plan->add_var("phys/ground_object_id",BBVariable(Variant::INT,0));
    }
    if(!p_plan->has_var("phys/ground_collider_layer"))
    {
        p_plan->add_var("phys/ground_collider_layer",BBVariable(Variant::INT,0));
    }
    if(!p_plan->has_var("phys/is_fall"))
    {
        p_plan->add_var("phys/is_fall",BBVariable(Variant::BOOL,false));
    }
    if(!p_plan->has_var("phys/is_on_air"))
    {
        p_plan->add_var("phys/is_on_air",BBVariable(Variant::BOOL,false));
    }

    
    if(!p_plan->has_var("jump/request_jump"))
    {
        p_plan->add_var("jump/request_jump",BBVariable(Variant::BOOL,false));
    }  

    if(!p_plan->has_var("jump/is_jump"))
    {
        p_plan->add_var("jump/is_jump",BBVariable(Variant::BOOL,false));
    }   
    if(!p_plan->has_var("jump/is_jump2"))
    {
        p_plan->add_var("jump/is_jump2",BBVariable(Variant::BOOL,false));
    }  
    if(!p_plan->has_var("jump/jump_count"))
    {
        p_plan->add_var("jump/jump_count",BBVariable(Variant::INT,0));
    }
    // 请求角色复活
    if(!p_plan->has_var("dead/is_dead"))
    {
        p_plan->add_var("dead/is_dead",BBVariable(Variant::BOOL,false));
    }
    if(!p_plan->has_var("dead/request_revive"))
    {
        p_plan->add_var("dead/request_revive",BBVariable(Variant::BOOL,false));
    }
    if(!p_plan->has_var("dead/request_revive_life"))
    {
        p_plan->add_var("dead/request_revive_life",BBVariable(Variant::FLOAT,0.0f));
    }

    // 人物情绪
    if(!p_plan->has_var("ai/emotion"))
    {
        p_plan->add_var("ai/emotion",BBVariable(Variant::INT,0,PROPERTY_HINT_ENUM,L"平静,感激,愤怒,激动,开心,恐惧"));
    }

    // 人物性格
    if(!p_plan->has_var("ai/personality"))
    {
        p_plan->add_var("ai/personality",BBVariable(Variant::INT,0,PROPERTY_HINT_ENUM,L"和平,好斗,胆小,忠诚,复仇心强,狡猾"));
    }
}



// 保存模型资源
static void save_fbx_res( const String& group_name,const String& sub_path,const Ref<Resource>& p_resource,String& save_path, bool is_resource = true)
{
	String export_root_path = "res://Assets/public";
	if (!DirAccess::exists("res://Assets"))
	{
		DirAccess::make_dir_absolute("res://Assets");
	}
	if (!DirAccess::exists(export_root_path))
	{
		DirAccess::make_dir_absolute(export_root_path);
	}
	export_root_path  = export_root_path.path_join(group_name);
	if (!DirAccess::exists(export_root_path))
	{
		DirAccess::make_dir_absolute(export_root_path);
	}
	export_root_path = export_root_path.path_join(sub_path);
	if (!DirAccess::exists(export_root_path))
	{
		DirAccess::make_dir_absolute(export_root_path);
	}
	save_path = export_root_path.path_join(p_resource->get_name() + (is_resource ? ".res" :".scn"));
	ResourceSaver::save(p_resource, save_path);
	print_line(L"CharacterBodyMain.save_fbx_res: 存储资源 :" + save_path);
    save_path = sub_path.path_join(p_resource->get_name() + (is_resource ? ".res" :".scn"));
}
static void get_fbx_meshs(Node *p_node,HashMap<String,MeshInstance3D* > &meshs)
{

	for(int i=0;i<p_node->get_child_count();i++)
	{
		Node * child = p_node->get_child(i);
		if(child->get_class() == "MeshInstance3D")
		{
			MeshInstance3D * mesh = Object::cast_to<MeshInstance3D>(child);
			if(!meshs.has(mesh->get_name())){
				meshs[mesh->get_name()] = mesh;
			}
			else{
				String name = mesh->get_name();
				int index = 1;
				while(meshs.has(name +"_"+ itos(index))){
					name = mesh->get_name().str() + "_" + itos(index);
					index++;
				}
				meshs[name] = mesh;
			}
		}
		get_fbx_meshs(child,meshs);
	}
}
void reset_owenr(Node* node, Node* owenr)
{
	for (int i = 0; i < node->get_child_count(); ++i)
	{
		Node* c = node->get_child(i);
		c->set_owner(nullptr);
		reset_owenr(c, owenr);
	}
}
Ref<CharacterBodyPrefab> CharacterBodyMain::build_prefab(const String& mesh_path)
{
	if (!FileAccess::exists(mesh_path))
	{
		return Ref<CharacterBodyPrefab>();
	}

	// 加载模型
	Ref<PackedScene> scene = ResourceLoader::load(mesh_path);

	if (scene.is_null())
	{
		print_line(L"CharacterBodyMain: 路径不存在 :" + mesh_path);
		return Ref<CharacterBodyPrefab>();
	}
	Node* p_node = scene->instantiate(PackedScene::GEN_EDIT_STATE_DISABLED);
	String p_group = mesh_path.get_file().get_basename();

	Node* node = p_node->find_child("Skeleton3D");
	Skeleton3D* skeleton = Object::cast_to<Skeleton3D>(node);

	Dictionary bone_map;
	String ske_save_path, bone_map_save_path;
	if (skeleton != nullptr)
	{
		bone_map = skeleton->get_human_bone_mapping();
        Vector<String> bone_names = skeleton->get_bone_names();
		skeleton->set_human_bone_mapping(bone_map);
		skeleton->set_owner(nullptr);
		reset_owenr(skeleton, skeleton);

		// 存儲骨架信息
		Ref<PackedScene> packed_scene;
		packed_scene.instantiate();
		packed_scene->pack(skeleton);
		packed_scene->set_name("skeleton");
		save_fbx_res("skeleton", p_group, packed_scene, ske_save_path, false);

		// 存储骨骼映射
		Ref<CharacterBoneMap> bone_map_ref;
		bone_map_ref.instantiate();
		bone_map_ref->set_name("bone_map");
		bone_map_ref->set_bone_map(bone_map);
        bone_map_ref->set_bone_names(bone_names);
		save_fbx_res("bone_map", p_group, bone_map_ref, bone_map_save_path, true);
	}
	// 生成预制体
	Ref<CharacterBodyPrefab> body_prefab;
	body_prefab.instantiate();
	body_prefab->set_name(p_group);
	HashMap<String, MeshInstance3D* > meshs;
	// 便利存儲模型文件
	get_fbx_meshs(p_node, meshs);
	for (auto it = meshs.begin(); it != meshs.end(); ++it) {
		Ref<CharacterBodyPart> part;
		part.instantiate();
		MeshInstance3D* mesh = it->value;
		part->init_form_mesh_instance(mesh, bone_map);
        
		part->set_name(it->key);
		String save_path;
		save_fbx_res("meshs", p_group, part, save_path, true);
		body_prefab->parts[save_path] = true;
	}
	// 保存预制体
	body_prefab->skeleton_path = ske_save_path;
	save_fbx_res("prefab", p_group, body_prefab, bone_map_save_path, true);


	p_node->queue_free();
	return body_prefab;
}
void CharacterBodyMain::editor_build_form_mesh_file_path()
{
	Ref<CharacterBodyPrefab> prefab = build_prefab(editor_form_mesh_file_path);
    // 设置预制体
    set_body_prefab(prefab);
    
}
void CharacterBodyMain::editor_play_select_animation() {
    init();
    if(play_animation.is_null()) {
        return;
    }
    if(animator.is_null()) {
        return;
    }
    animator->editor_play_animation(play_animation);
}
void CharacterBodyMain::editor_to_human_animation() {

}
void CharacterBodyMain::update_bone_visble()
{
    if(bone_label != nullptr) {
        bone_label->clear();
    }
    Skeleton3D* skeleton = get_skeleton();
    if (skeleton == nullptr)
    {
        return;
    }
    if(play_animation.is_null()) {
        return;
    }
    if(bone_label == nullptr) {
        bone_label = memnew(HumanBoneVisble);
        add_child(bone_label);
        bone_label->set_owner(this);
    }
	bone_label->init(get_skeleton(), play_animation->get_bone_map());
    
}
void CharacterBodyMain::editor_build_animation()
{
    if(!FileAccess::exists(editor_animation_file_path))
    {
		print_line(L"CharacterBodyMain: 路径不存在 :" + editor_animation_file_path);
        return;
    }
	Ref<PackedScene> scene = ResourceLoader::load(editor_animation_file_path);
	if (scene.is_null())
	{
		print_line(L"CharacterBodyMain: 路径不存在 :" + editor_animation_file_path);
        return;
	}
	Node* p_node = scene->instantiate(PackedScene::GEN_EDIT_STATE_DISABLED);
	Ref<CharacterBoneMap> bone_map;
    Node* node = p_node->find_child("Skeleton3D");
    Skeleton3D* skeleton = Object::cast_to<Skeleton3D>(node);
	if (skeleton != nullptr)
	{
		bone_map.instantiate();
		bone_map->set_bone_map(skeleton->get_human_bone_mapping());
		bone_map->set_bone_names(skeleton->get_bone_names());
	}
	else 
	{
		if (editor_ref_bone_map.is_null())
		{
			print_error(L"CharacterBodyMain: 路径不存在骨架信息,必须要设置骨骼映射:" + editor_animation_file_path);
			return;
		}
		else
		{
			bone_map = editor_ref_bone_map;
		}
	}

	Node* anim_node = p_node->find_child("AnimationPlayer");
    if(anim_node == nullptr)
    {
        print_line(L"CharacterBodyMain: 路径不存在动画信息:" + editor_animation_file_path);
        return;
    }

    AnimationPlayer* player = Object::cast_to<AnimationPlayer>(anim_node);
    if(player == nullptr)
    {
        print_line(L"CharacterBodyMain: 路径不存在动画信息:" + editor_animation_file_path);
        return;
    }
	String p_group = editor_animation_file_path.get_file().get_basename();
    List<StringName> p_animations;
    player->get_animation_list(&p_animations);
    for (const StringName &E : p_animations) {
        Ref<Animation> animation = player->get_animation(E);
        if(animation.is_valid())
        {
            Ref<Animation> new_animation = animation->duplicate();
            new_animation->set_bone_map(bone_map);
            if(skeleton == nullptr)
            {
                new_animation->remap_node_to_bone_name(bone_map->get_bone_names());
            }
            new_animation->optimize();
            new_animation->compress();
			if (p_animations.size() == 1)
			{
				Vector<String> names = p_group.split("@");
				String name;
				if (names.size() > 0)
				{
					name = names[names.size() - 1];
				}
				else
				{
					name = E;
				}
				new_animation->set_name(name);
			}
			else
			{
				new_animation->set_name(E);
			}
            String save_path;
			save_fbx_res("animation", p_group, new_animation, save_path, true);
            
        }
    }
}



