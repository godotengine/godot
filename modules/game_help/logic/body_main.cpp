#include "body_main.h"
#include "data_table_manager.h"
#include "scene/3d/path_3d.h"
#include "character_ai/character_ai.h"
#include "character_manager.h"
#include "core/io/file_access.h"
#include "core/io/dir_access.h"

CharacterAIContext::CharacterAIContext()
{
	beehave_run_context.instantiate();
}

ObjectID& CharacterBodyMain::get_curr_editor_player()
{
    static ObjectID curr_editor_player;
    return curr_editor_player;
}
void CharacterBodyMain::_bind_methods()
{
    
	ClassDB::bind_method(D_METHOD("restart"), &CharacterBodyMain::restart);
	ClassDB::bind_method(D_METHOD("init_main_body","p_skeleton_file_path","p_animation_group"), &CharacterBodyMain::init_main_body);




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

    ClassDB::bind_method(D_METHOD("init_body_part_array", "part_array"), &CharacterBodyMain::init_body_part_array);
    ClassDB::bind_method(D_METHOD("set_body_part", "part"), &CharacterBodyMain::set_body_part);
    ClassDB::bind_method(D_METHOD("get_body_part"), &CharacterBodyMain::get_body_part);

    ClassDB::bind_method(D_METHOD("set_character_ai", "ai"), &CharacterBodyMain::set_character_ai);
    ClassDB::bind_method(D_METHOD("get_character_ai"), &CharacterBodyMain::get_character_ai);

    ClassDB::bind_method(D_METHOD("set_skeleton_resource", "skeleton"), &CharacterBodyMain::set_skeleton_resource);
    ClassDB::bind_method(D_METHOD("get_skeleton_resource"), &CharacterBodyMain::get_skeleton_resource);

    
    ClassDB::bind_method(D_METHOD("set_editor_form_mesh_file_path", "editor_form_mesh_file_path"), &CharacterBodyMain::set_editor_form_mesh_file_path);
    ClassDB::bind_method(D_METHOD("get_editor_form_mesh_file_path"), &CharacterBodyMain::get_editor_form_mesh_file_path);


	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "blackboard_plan", PROPERTY_HINT_RESOURCE_TYPE, "BlackboardPlan", PROPERTY_USAGE_DEFAULT ), "set_blackboard_plan", "get_blackboard_plan");
    ADD_GROUP("editor", "editor_");
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "editor_form_mesh_file_path"), "set_editor_form_mesh_file_path", "get_editor_form_mesh_file_path");
    ADD_MEMBER_BUTTON(editor_build_form_mesh_file_path,L"根据模型初始化",CharacterBodyMain);


    //ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "behavior_tree", PROPERTY_HINT_RESOURCE_TYPE, "BehaviorTree"), "set_behavior_tree", "get_behavior_tree");
	//ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "blackboard", PROPERTY_HINT_RESOURCE_TYPE, "Blackboard",PROPERTY_USAGE_DEFAULT ), "set_blackboard", "get_blackboard");
    ADD_GROUP("logic", "");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "character_ai", PROPERTY_HINT_RESOURCE_TYPE, "CharacterAI"), "set_character_ai", "get_character_ai");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "navigation_agent", PROPERTY_HINT_RESOURCE_TYPE, "CharacterNavigationAgent3D"), "set_navigation_agent", "get_navigation_agent");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "main_shape", PROPERTY_HINT_RESOURCE_TYPE, "CollisionObject3DConnectionShape",PROPERTY_USAGE_DEFAULT), "set_main_shape", "get_main_shape");
    ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "check_area", PROPERTY_HINT_RESOURCE_TYPE, RESOURCE_TYPE_HINT("CharacterCheckArea3D"),PROPERTY_USAGE_DEFAULT), "set_check_area", "get_check_area");
    

    ADD_GROUP("show", "");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "body_prefab", PROPERTY_HINT_RESOURCE_TYPE, "CharacterBodyPrefab",PROPERTY_USAGE_DEFAULT ), "set_body_prefab", "get_body_prefab");
    ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "body_part", PROPERTY_HINT_NONE,"",PROPERTY_USAGE_DEFAULT), "set_body_part", "get_body_part");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "animator", PROPERTY_HINT_RESOURCE_TYPE, "CharacterAnimator",PROPERTY_USAGE_DEFAULT ), "set_animator", "get_animator"); 
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "animation_library", PROPERTY_HINT_RESOURCE_TYPE, "AnimationLibrary",PROPERTY_USAGE_DEFAULT ), "set_animation_library", "get_animation_library");
    ADD_PROPERTY(PropertyInfo(Variant::STRING, "skeleton", PROPERTY_HINT_FILE, "*.tscn,*.scn",PROPERTY_USAGE_DEFAULT ), "set_skeleton_resource", "get_skeleton_resource");
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "ik", PROPERTY_HINT_RESOURCE_TYPE, "RenIK",PROPERTY_USAGE_DEFAULT ), "set_ik", "get_ik");


	ADD_SIGNAL(MethodInfo("behavior_tree_finished", PropertyInfo(Variant::INT, "status")));
	ADD_SIGNAL(MethodInfo("behavior_tree_updated", PropertyInfo(Variant::INT, "status")));
    
	ADD_SIGNAL(MethodInfo("skill_tree_finished", PropertyInfo(Variant::INT, "status")));
	ADD_SIGNAL(MethodInfo("skill_tree_updated", PropertyInfo(Variant::INT, "status")));

    

}

void CharacterBodyMain::clear_all()
{
    Skeleton3D* skeleton = Object::cast_to<Skeleton3D>(ObjectDB::get_instance(skeletonID));
    if(skeleton == nullptr)
    {
        return;
    }
    if(skeleton)
    {
        memdelete(skeleton);
        skeleton = nullptr;
        
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
	save_path = export_root_path.path_join(p_resource->get_name() + (is_resource ? ".tres" :".tscn"));
	ResourceSaver::save(p_resource, save_path);
	print_line(L"CharacterBodyMain.save_fbx_res: 存储资源 :" + save_path);
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

void CharacterBodyMain::editor_build_animation_form_file_path()
{
    if(editor_ref_bone_map.is_null()) {
        return;
    }
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

	Node* node = p_node->find_child("AnimationPlayer");
    if(node == nullptr)
    {
        print_line(L"CharacterBodyMain: 路径不存在动画信息:" + editor_animation_file_path);
        return;
    }

    AnimationPlayer* player = Object::cast_to<AnimationPlayer>(node);
    if(player == nullptr)
    {
        print_line(L"CharacterBodyMain: 路径不存在动画信息:" + editor_animation_file_path);
        return;
    }
    List<StringName> p_animations;
    player->get_animation_list(&p_animations);
    for (const StringName &E : p_animations) {
        Ref<Animation> animation = player->get_animation(E);
        if(animation.is_valid())
        {
            // 保存动画
            Ref<CharacterAnimation> character_animation;
            character_animation.instantiate();
            character_animation->set_bone_map(editor_ref_bone_map);

            List<PropertyInfo> plist;
            animation->get_property_list(&plist);

            Ref<RefCounted> r = static_cast<RefCounted *>(ClassDB::instantiate(get_class()));
            ERR_FAIL_COND(r.is_null());

            for (const PropertyInfo &E : plist) {
                if (!(E.usage & PROPERTY_USAGE_STORAGE)) {
                    continue;
                }
                Variant p = get(E.name);

                switch (p.get_type()) {
                    case Variant::Type::DICTIONARY:
                    case Variant::Type::ARRAY:
                    case Variant::Type::PACKED_BYTE_ARRAY:
                    case Variant::Type::PACKED_COLOR_ARRAY:
                    case Variant::Type::PACKED_INT32_ARRAY:
                    case Variant::Type::PACKED_INT64_ARRAY:
                    case Variant::Type::PACKED_FLOAT32_ARRAY:
                    case Variant::Type::PACKED_FLOAT64_ARRAY:
                    case Variant::Type::PACKED_STRING_ARRAY:
                    case Variant::Type::PACKED_VECTOR2_ARRAY:
                    case Variant::Type::PACKED_VECTOR3_ARRAY:
                    case Variant::Type::PACKED_VECTOR4_ARRAY: {
                       character_animation->set(E.name, p.duplicate(true));
                    } break;

                    case Variant::Type::OBJECT: {
                        character_animation->set(E.name, p);
                    } break;

                    default: {
                        character_animation->set(E.name, p);
                    }
                }
            }
            character_animation->set_name(E);
        
            
            String save_path;
            save_fbx_res("animation", editor_ref_bone_map->get_name(), character_animation, save_path, true);
            
        }
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
void CharacterBodyMain::_process_animator()
{
    if(animator.is_valid())
    {
        animator->_thread_update_animator(get_process_delta_time());
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
    if(ik.is_valid())
    {
        ik->update_ik();
        ik->update_placement(delta);
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
}
// 初始化身體
void CharacterBodyMain::init_main_body(String p_skeleton_file_path,StringName p_animation_group)
{
    skeleton_res = p_skeleton_file_path;
    animation_group = p_animation_group;

}

void CharacterBodyMain::set_character_ai(const Ref<CharacterAI> &p_ai)
{
    character_ai = p_ai;
}
Ref<CharacterAI> CharacterBodyMain::get_character_ai()
{
    return character_ai;
} 


void CharacterBodyMain::set_skeleton_resource(const String& p_skeleton_path)
{        
    skeleton_res = p_skeleton_path;
    clear_all();
    Ref<PackedScene> scene = ResourceLoader::load(skeleton_res);
    if(!scene.is_valid())
    {
        ERR_FAIL_MSG("load skeleton failed:" + skeleton_res);
        skeleton_res = "";
        return ;
    }
    Node* ins = scene->instantiate(PackedScene::GEN_EDIT_STATE_INSTANCE);
    if (ins == nullptr) {
        ERR_FAIL_MSG("init skeleton instantiate failed:" + skeleton_res);
        skeleton_res = "";
        return;
    }
    Skeleton3D* skeleton = Object::cast_to<Skeleton3D>(ins); 
    if(skeleton == nullptr)
    {
        ERR_FAIL_MSG("scene is not Skeleton3D:" + skeleton_res);
        memdelete(ins);
        skeleton_res = "";
        return ;
    }
    skeleton->set_name("Skeleton3D");

    add_child(skeleton);
    skeleton->set_owner(this);
    skeleton->set_dont_save(true);

    
    if(skeleton && ik.is_valid())
    {
        ik->_initialize(skeleton);
    }
    skeletonID = skeleton->get_instance_id();

    
}
void CharacterBodyMain::load_mesh(const StringName& part_name,String p_mesh_file_path)
{
    auto old_ins = bodyPart.find(part_name);
    if(old_ins == bodyPart.end())
    {
        ERR_FAIL_MSG("not found body part:" + part_name.str());
        return;
    }
    Ref<CharacterBodyPart> mesh = ResourceLoader::load(p_mesh_file_path);
    if(!mesh.is_valid())
    {
        return;
    }
    Ref<CharacterBodyPartInstane> ins = old_ins->value;
    ins->set_part(mesh);
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


void CharacterBodyMain::init_body_part_array(const Array& p_part_array)
{
    for(auto & a : bodyPart)
    {
        a.value->clear();
    }
    bodyPart.clear();
    Skeleton3D* skeleton = Object::cast_to<Skeleton3D>(ObjectDB::get_instance(skeletonID));
    if(skeleton == nullptr)
    {
        return;
    }

    for(int i = 0;i < p_part_array.size();i++)
    {
        StringName part_name = p_part_array[i];    
        if(part_name.is_empty())
        {
            continue;
        }
        Ref<CharacterBodyPartInstane> p;
        p.instantiate();
        p->set_skeleton(skeleton);
        bodyPart[part_name] = p;
    }
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
    bodyPart.clear();
    if(body_prefab.is_valid())
    {
        body_prefab->connect_changed(callable_mp(this, &CharacterBodyMain::load_prefab));
        set_skeleton_resource(body_prefab->get_skeleton_path());
        Skeleton3D* skeleton = Object::cast_to<Skeleton3D>(ObjectDB::get_instance(skeletonID));
        // 
        TypedArray<CharacterBodyPart> part_array = body_prefab->load_part();
        int size = part_array.size();
        for(int i = 0; i < size; i++)
        {
            
            Ref<CharacterBodyPartInstane> p;
            p.instantiate();
            p->set_skeleton(skeleton);
            p->set_part(part_array[i]);
			Ref< CharacterBodyPart> part = part_array[i];
            bodyPart[part->get_name()] = p;
        }
        notify_property_list_changed();
    }

}
Ref<CharacterBodyPrefab> CharacterBodyMain::get_body_prefab()
{
    return body_prefab;
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

void CharacterBodyMain::set_body_part(const Dictionary& part)
{
    Array keys = part.keys();
    
    HashMap<StringName,Ref<CharacterBodyPartInstane>> old_bodyPart = bodyPart;
    bodyPart.clear();
    Skeleton3D* skeleton = Object::cast_to<Skeleton3D>(ObjectDB::get_instance(skeletonID));
    if(skeleton == nullptr)
    {
        return;
    }
    for(int i = 0;i < keys.size();i++)
    {
        StringName part_name = keys[i];

        Ref<CharacterBodyPartInstane> p = part[part_name];
        if(old_bodyPart.has(part_name))
        {
            Ref<CharacterBodyPartInstane> mesh = old_bodyPart[part_name];
            if(p->get_part() != mesh->get_part())
            {
                mesh->set_part(p->get_part());
                mesh->set_skeleton(skeleton);
            }
            bodyPart[part_name] = mesh;        
            old_bodyPart.erase(part_name);       

        }
        else
        {
            // 克隆一份
            if(!p.is_valid())
            {
                p.instantiate();
            }
            else
            {
                p = p->duplicate();
            }
            p->set_skeleton(skeleton);
            bodyPart[part_name] = p; 
        }
    }
    for(auto & a : old_bodyPart)
    {
        a.value->clear();
    }
}
Dictionary CharacterBodyMain::get_body_part()
{
    Dictionary ret;
    for(auto & a : bodyPart)
    {
        ret[a.key] = a.value;
    }
    return ret;
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







