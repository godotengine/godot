#include "animation_process_panel.h"
#include "scene/gui/separator.h"
#include "core/io/json.h"

#include "scene/resources/animation.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/animation/animation_tree.h"
#include "scene/animation/animation_player.h"
#include "scene/resources/packed_scene.h"
#include "../../logic/animator/human_animation.h"




HBoxContainer* AnimationProcessPanel::create_line(Control* control , bool is_side_separator ) {
    HBoxContainer* hb = memnew(HBoxContainer);
    if(is_side_separator) {
        HSeparator* sep = memnew(HSeparator);
        sep->set_h_size_flags(SIZE_EXPAND_FILL);
        hb->add_child(sep);
    }
    hb->add_child(control);

    if(is_side_separator) {
        HSeparator* sep = memnew(HSeparator);
        sep->set_h_size_flags(SIZE_EXPAND_FILL);
        hb->add_child(sep);
    }
    return hb;

}
HBoxContainer* AnimationProcessPanel::create_line(Control* control0 , Control* control1,  bool is_side_separator ) {
    HBoxContainer* hb = memnew(HBoxContainer);
    if(is_side_separator) {
        HSeparator* sep = memnew(HSeparator);
        sep->set_h_size_flags(SIZE_EXPAND_FILL);
        hb->add_child(sep);
    }
    hb->add_child(control0);
    hb->add_child(control1);

    if(is_side_separator) {
        HSeparator* sep = memnew(HSeparator);
        sep->set_h_size_flags(SIZE_EXPAND_FILL);
        hb->add_child(sep);
    }
    return hb;

}

HBoxContainer* AnimationProcessPanel::create_line(Control* control0 , Control* control1, Control* control2,  bool is_side_separator ) {
    HBoxContainer* hb = memnew(HBoxContainer);
    if(is_side_separator) {
        HSeparator* sep = memnew(HSeparator);
        sep->set_h_size_flags(SIZE_EXPAND_FILL);
        hb->add_child(sep);
    }
    hb->add_child(control0);
    hb->add_child(control1);
    hb->add_child(control2);

    if(is_side_separator) {
        HSeparator* sep = memnew(HSeparator);
        sep->set_h_size_flags(SIZE_EXPAND_FILL);
        hb->add_child(sep);
    }
    return hb;

}

HBoxContainer* AnimationProcessPanel::create_line(Control* control0 , Control* control1, Control* control2, Control* control3,  bool is_side_separator ) {
    HBoxContainer* hb = memnew(HBoxContainer);
    if(is_side_separator) {
        HSeparator* sep = memnew(HSeparator);
        sep->set_h_size_flags(SIZE_EXPAND_FILL);
        hb->add_child(sep);
    }
    hb->add_child(control0);
    hb->add_child(control1);
    hb->add_child(control2);
    hb->add_child(control3);

    if(is_side_separator) {
        HSeparator* sep = memnew(HSeparator);
        sep->set_h_size_flags(SIZE_EXPAND_FILL);
        hb->add_child(sep);
    }
    return hb;

}

AnimationProcessPanel::AnimationProcessPanel() {
    load_animation_config();

    HBoxContainer* hb = memnew(HBoxContainer);
    hb->set_h_size_flags(SIZE_EXPAND_FILL);
    add_child(hb);
    {
        VBoxContainer* vb = memnew(VBoxContainer);
        vb->set_custom_minimum_size(Vector2(300, 0));
        hb->add_child(vb);

        property_preview_prefab_path = memnew(EditorPropertyPath);
        property_preview_prefab_path->set_label(L"選擇预制件：");
        property_preview_prefab_path->set_object_and_property(this, SNAME("preview_prefab_path"));
        property_preview_prefab_path->setup({ "*.res", "*.tres" }, false, false);
        property_preview_prefab_path->set_h_size_flags(SIZE_EXPAND_FILL);
		property_preview_prefab_path->set_custom_property(true);
        property_preview_prefab_path->update_property();
        vb->add_child(property_preview_prefab_path);

        preview = memnew(AnimationNodePreview);
        preview->set_custom_minimum_size(Vector2(400, 400));
        vb->add_child(preview);
        if(preview_prefab_path != "") {
            preview->set_prefab_path(preview_prefab_path);
        }
    }

    {
        VBoxContainer* vb = memnew(VBoxContainer);
        vb->set_h_size_flags(SIZE_EXPAND_FILL);
        vb->set_custom_minimum_size(Vector2(300, 0));
        hb->add_child(vb);

        {

        
            Label* label = memnew(Label);
            label->set_h_size_flags(SIZE_EXPAND_FILL);
            label->set_text(L"单个动画处理");
            label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
            label->set_modulate(Color(1,0.8,0.7,1));
            vb->add_child(create_line(  label,true));
                                    
            VSeparator* sep = memnew(VSeparator);
            vb->add_child(sep);

            {
                single_path = memnew(EditorPropertyPath);
                single_path->set_label(L"选择动画文件");
                single_path->set_object_and_property(this, SNAME("single_animation_file_path"));
                single_path->setup({ "res", "tres" }, false, false);
                single_path->set_h_size_flags(SIZE_EXPAND_FILL);
				single_path->set_custom_property(true);
                single_path->update_property();

                single_animation_group = memnew(EditorPropertyTextEnum);
                single_animation_group->set_label(L"动画组");
                single_animation_group->set_custom_minimum_size(Vector2(300, 0));
                single_animation_group->set_object_and_property(this, "single_animation_group");
                single_animation_group->set_dynamic(true, "get_animation_groups");
                single_animation_group->setup(Vector<String>());
				single_animation_group->set_custom_property(true);
                single_animation_group->update_property();

                single_animation_tags = memnew(EditorPropertyTextEnum);
                single_animation_tags->set_label(L"动画标签");
                single_animation_tags->set_custom_minimum_size(Vector2(300, 0));
                single_animation_tags->set_object_and_property(this, "single_animation_tags");
                single_animation_tags->set_dynamic(true, "get_animation_tags");
                single_animation_tags->setup(Vector<String>());
				single_animation_tags->set_custom_property(true);
                single_animation_tags->update_property();

                conver_single_button = memnew(Button);
                conver_single_button->set_text(L"转换");
                conver_single_button->connect(SceneStringName(pressed), callable_mp(this, &AnimationProcessPanel::_on_conver_single_pressed));


                
                vb->add_child(create_line(  single_path,single_animation_group,single_animation_tags));

                vb->add_child(conver_single_button);
            }

        
        }



        {
            Label* label = memnew(Label);
            label->set_h_size_flags(SIZE_EXPAND_FILL);
            label->set_text(L"多个动画处理");
            label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
            label->set_modulate(Color(1,0.8,0.7,1));

            vb->add_child(create_line(label,true));
                                    
            VSeparator* sep = memnew(VSeparator);
            vb->add_child(sep);
            {
                multe_path = memnew(EditorPropertyPath);
                multe_path->set_label(L"选择文件夹");
                multe_path->set_object_and_property(this, "multe_animation_file_path");
                multe_path->set_h_size_flags(SIZE_EXPAND_FILL);
                multe_path->setup(Vector<String>(), true, false);
				multe_path->set_custom_property(true);
                multe_path->update_property();

                multe_animation_group = memnew(EditorPropertyTextEnum);
                multe_animation_group->set_label(L"动画组");
                multe_animation_group->set_custom_minimum_size(Vector2(300, 0));
                multe_animation_group->set_object_and_property(this, "multe_animation_group");
                multe_animation_group->set_dynamic(true, "get_animation_groups");
                multe_animation_group->setup(Vector<String>());
				multe_animation_group->set_custom_property(true);
                multe_animation_group->update_property();

                multe_animation_tags = memnew(EditorPropertyTextEnum);
                multe_animation_tags->set_label(L"动画标签");
                multe_animation_tags->set_custom_minimum_size(Vector2(300, 0));
                multe_animation_tags->set_object_and_property(this, "multe_animation_tags");
                multe_animation_tags->set_dynamic(true, "get_animation_tags");
                multe_animation_tags->setup(Vector<String>());
				multe_animation_tags->set_custom_property(true);
                multe_animation_tags->update_property();

                conver_multe_button = memnew(Button);
                conver_multe_button->set_text(L"转换");
                conver_multe_button->connect(SceneStringName(pressed), callable_mp(this, &AnimationProcessPanel::_on_conver_multe_pressed));
                
                vb->add_child(create_line(  multe_path,multe_animation_group,multe_animation_tags));

                vb->add_child(conver_multe_button);

            }

            


        }

    }

}
void AnimationProcessPanel::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_preview_prefab_path", "path"), &AnimationProcessPanel::set_preview_prefab_path);
    ClassDB::bind_method(D_METHOD("get_preview_prefab_path"), &AnimationProcessPanel::get_preview_prefab_path);

    ClassDB::bind_method(D_METHOD("get_animation_groups"), &AnimationProcessPanel::get_animation_groups);
    ClassDB::bind_method(D_METHOD("get_animation_tags"), &AnimationProcessPanel::get_animation_tags);

    ClassDB::bind_method(D_METHOD("set_single_animation_file_path", "path"), &AnimationProcessPanel::set_single_animation_file_path);
    ClassDB::bind_method(D_METHOD("get_single_animation_file_path"), &AnimationProcessPanel::get_single_animation_file_path);

    ClassDB::bind_method(D_METHOD("set_single_animation_group", "group"), &AnimationProcessPanel::set_single_animation_group);
    ClassDB::bind_method(D_METHOD("get_single_animation_group"), &AnimationProcessPanel::get_single_animation_group);
    ClassDB::bind_method(D_METHOD("set_single_animation_tags", "tag"), &AnimationProcessPanel::set_single_animation_tags);
    ClassDB::bind_method(D_METHOD("get_single_animation_tags"), &AnimationProcessPanel::get_single_animation_tags);

    ClassDB::bind_method(D_METHOD("set_multe_animation_file_path", "path"), &AnimationProcessPanel::set_multe_animation_file_path);
    ClassDB::bind_method(D_METHOD("get_multe_animation_file_path"), &AnimationProcessPanel::get_multe_animation_file_path);
    ClassDB::bind_method(D_METHOD("set_multe_animation_group", "group"), &AnimationProcessPanel::set_multe_animation_group);
    ClassDB::bind_method(D_METHOD("get_multe_animation_group"), &AnimationProcessPanel::get_multe_animation_group);
    ClassDB::bind_method(D_METHOD("set_multe_animation_tags", "tag"), &AnimationProcessPanel::set_multe_animation_tags);
    ClassDB::bind_method(D_METHOD("get_multe_animation_tags"), &AnimationProcessPanel::get_multe_animation_tags);

    ADD_PROPERTY(PropertyInfo(Variant::STRING, "preview_prefab_path"), "set_preview_prefab_path", "get_preview_prefab_path");

    ADD_PROPERTY(PropertyInfo(Variant::STRING, "single_animation_file_path"), "set_single_animation_file_path", "get_single_animation_file_path");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "single_animation_group"), "set_single_animation_group", "get_single_animation_group");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "single_animation_tags"), "set_single_animation_tags", "get_single_animation_tags");

    ADD_PROPERTY(PropertyInfo(Variant::STRING, "multe_animation_file_path"), "set_multe_animation_file_path", "get_multe_animation_file_path");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "multe_animation_group"), "set_multe_animation_group", "get_multe_animation_group");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "multe_animation_tags"), "set_multe_animation_tags", "get_multe_animation_tags");


}

void AnimationProcessPanel::set_preview_prefab_path(const String& path) {
    preview_prefab_path = path;
    if(preview != nullptr) {
        preview->set_prefab_path(path);
    }
    save_animation_config();
}
String AnimationProcessPanel::get_preview_prefab_path() {
    return preview_prefab_path;
}

void AnimationProcessPanel::set_single_animation_file_path(const String& path) {
    single_animation_file_path = path;
    save_animation_config();
    single_path->update_property();
}
String AnimationProcessPanel::get_single_animation_file_path() {
    return single_animation_file_path;
}
void AnimationProcessPanel::set_single_animation_group(const String& group) {
    single_animation_group_name = group;     
    save_animation_config();   
}
StringName AnimationProcessPanel::get_single_animation_group() {
    return single_animation_group_name;
}
void AnimationProcessPanel::set_single_animation_tags(const String& tag) {
    single_animation_tag_name = tag;
    save_animation_config();
    
}
StringName AnimationProcessPanel::get_single_animation_tags() {
    return single_animation_tag_name;
}


void AnimationProcessPanel::set_multe_animation_file_path(const String& path) {
    multe_animation_file_path = path;
    save_animation_config();
    multe_path->update_property();
}
String AnimationProcessPanel::get_multe_animation_file_path() {
    return multe_animation_file_path;
}
void AnimationProcessPanel::set_multe_animation_group(const String& group) {
    multe_animation_group_name = group;
    save_animation_config();
}
StringName AnimationProcessPanel::get_multe_animation_group() {
    return multe_animation_group_name;
}
void AnimationProcessPanel::set_multe_animation_tags(const String& tag) {
    multe_animation_tag_name = tag;
    save_animation_config();
}
StringName AnimationProcessPanel::get_multe_animation_tags() {
    return multe_animation_tag_name;
}

Array AnimationProcessPanel::get_animation_groups() {
    Array arr;
    CharacterManager::get_singleton()->get_animation_groups(&arr);
    return arr;
}
Array AnimationProcessPanel::get_animation_tags() {
    Array arr;
    CharacterManager::get_singleton()->get_animation_tags(&arr);
    return arr;
}
void AnimationProcessPanel::save_animation_config() {
    String path = "res://.godot/animation_process_panel_config.json";

    Dictionary dict;
    dict["preview_prefab_path"] = preview_prefab_path;

    dict["single_animation_file_path"] = single_animation_file_path;
    dict["single_animation_group"] = single_animation_group_name;
    dict["single_animation_tags"] = single_animation_tag_name;

    dict["multe_animation_file_path"] = multe_animation_file_path;
    dict["multe_animation_group"] = multe_animation_group_name;
    dict["multe_animation_tags"] = multe_animation_tag_name;

    Ref<FileAccess> file = FileAccess::open(path, FileAccess::WRITE);
    file->store_string(JSON::stringify(dict));
    file->close();
}

void AnimationProcessPanel::load_animation_config() {
    String path = "res://.godot/animation_process_panel_config.json";
    Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
    if (file == nullptr) {
        return;
    }
    String json = file->get_as_text();
    file->close();
    Dictionary dict = JSON::parse_string(json);

    preview_prefab_path = dict["preview_prefab_path"];

    single_animation_file_path = dict["single_animation_file_path"];
    single_animation_group_name = dict["single_animation_group"];
    single_animation_tag_name = dict["single_animation_tags"];

    multe_animation_file_path = dict["multe_animation_file_path"];
    multe_animation_group_name = dict["multe_animation_group"];
    multe_animation_tag_name = dict["multe_animation_tags"];
}



// 保存模型资源
static void save_fbx_res(const String& group_name, const String& sub_path, const Ref<Resource>& p_resource, String& save_path, bool is_resource = true)
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
	export_root_path = export_root_path.path_join(group_name);
	if (!DirAccess::exists(export_root_path))
	{
		DirAccess::make_dir_absolute(export_root_path);
	}
	export_root_path = export_root_path.path_join(sub_path);
	if (!DirAccess::exists(export_root_path))
	{
		DirAccess::make_dir_absolute(export_root_path);
	}
	save_path = export_root_path.path_join(p_resource->get_name() + (is_resource ? ".res" : ".scn"));
	ResourceSaver::save(p_resource, save_path, ResourceSaver::FLAG_CHANGE_PATH);
	ResourceCache::set_ref(save_path, p_resource.ptr());
	print_line(L"CharacterBodyMain.save_fbx_res: 存储资源 :" + save_path);
	save_path = sub_path.path_join(p_resource->get_name() + (is_resource ? ".res" : ".scn"));
}
static void save_fbx_tres(const String& group_name, const String& sub_path, const Ref<Resource>& p_resource, String& save_path, bool is_resource = true)
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
	export_root_path = export_root_path.path_join(group_name);
	if (!DirAccess::exists(export_root_path))
	{
		DirAccess::make_dir_absolute(export_root_path);
	}
	export_root_path = export_root_path.path_join(sub_path);
	if (!DirAccess::exists(export_root_path))
	{
		DirAccess::make_dir_absolute(export_root_path);
	}
	save_path = export_root_path.path_join(p_resource->get_name() + (is_resource ? ".tres" : ".tscn"));
	ResourceSaver::save(p_resource, save_path, ResourceSaver::FLAG_CHANGE_PATH);
	print_line(L"CharacterBodyMain.save_fbx_res: 存储资源 :" + save_path);
	save_path = sub_path.path_join(p_resource->get_name() + (is_resource ? ".tres" : ".tscn"));
}


static void node_to_bone_skeleton(Skeleton3D* p_ske, Node3D* p_node, int bode_parent) {
	int index = bode_parent;
	index = p_ske->add_bone(p_node->get_name());
	p_ske->set_bone_parent(index, bode_parent);
	Transform3D trans = p_node->get_transform();
	p_ske->set_bone_pose(index, trans);


	for (int i = 0; i < p_node->get_child_count(); ++i) {
		Node3D* node = Object::cast_to<Node3D>(p_node->get_child(i));
		if (node != nullptr) {
			node_to_bone_skeleton(p_ske, node, index);

		}
	}

}

void AnimationProcessPanel::editor_build_animation_form_path(String p_file_path,const StringName& animation_group, const StringName& animation_tag)
{
    if(!FileAccess::exists(p_file_path))
    {
		print_line(L"AnimationProcessPanel: 路径不存在 :" + p_file_path);
        return;
    }
	Ref<PackedScene> scene = ResourceLoader::load(p_file_path);
	if (scene.is_null())
	{
		print_line(L"AnimationProcessPanel: 路径不存在 :" + p_file_path);
        return;
	}
	Node* p_node = scene->instantiate(PackedScene::GEN_EDIT_STATE_DISABLED);
    Node* node = p_node->find_child("Skeleton3D");
    Skeleton3D* skeleton = Object::cast_to<Skeleton3D>(node);

	Node* anim_node = p_node->find_child("AnimationPlayer");
    if(anim_node == nullptr)
    {
        print_line(L"AnimationProcessPanel: 路径不存在动画信息:" + p_file_path);
        return;
    }

    AnimationPlayer* player = Object::cast_to<AnimationPlayer>(anim_node);
    if(player == nullptr)
    {
        print_line(L"AnimationProcessPanel: 路径不存在动画信息:" + p_file_path);
        return;
    }
	String p_group = p_file_path.get_file().get_basename();
    List<StringName> p_animations;
    player->get_animation_list(&p_animations);


	bool is_node_skeleton = false;
	Skeleton3D* bone_map_skeleton;

	HashMap<String, int> human_bone_name_index;
	Dictionary bone_map;
	Vector<String> bone_names;
	Ref<HumanBoneConfig> animation_human_config;
	//if (skeleton == nullptr)
	{
		is_node_skeleton = true;

		HashSet<String> node_name;
		for (const StringName& E : p_animations) {
			Ref<Animation> animation = player->get_animation(E);
			animation->get_node_names(node_name);
		}

		bone_map_skeleton = memnew(Skeleton3D);

		for (int i = 0; i < p_node->get_child_count(); ++i) {
			Node3D* child = Object::cast_to<Node3D>(p_node->get_child(i));
			if (child != nullptr) {
				if (child->get_child_count() > 0) {
					node_to_bone_skeleton(bone_map_skeleton, child, -1);
					break;
				}
			}
		}
		bone_names = bone_map_skeleton->get_bone_names();

		bone_map = bone_map_skeleton->get_human_bone_mapping();
		bone_map_skeleton->set_human_bone_mapping(bone_map);
	}
	//else

	// 有些动画的骨架可能存在多份,选择骨头最多的当做身体
	if(skeleton != nullptr && skeleton->get_bone_count() > bone_map.size())
	{
		auto new_bone_map = skeleton->get_human_bone_mapping();
		if (new_bone_map.size() > bone_map.size()) {
			bone_map = new_bone_map;
			bone_names = skeleton->get_bone_names();

			skeleton->set_human_bone_mapping(bone_map);
			bone_map_skeleton = skeleton;
		}
	}
	if (bone_map.size() < 2) {
			print_line(L"CharacterBodyMain: 动画的骨架不支持:" + p_file_path);
		return;
	}




    animation_human_config.instantiate();
    HashMap<String, String> _bone_label = HumanAnim::HumanAnimmation::get_bone_label();
    HumanAnim::HumanAnimmation::build_virtual_pose(bone_map_skeleton, *animation_human_config.ptr(), _bone_label);
	for (int i = 0; i < bone_names.size(); ++i) {
		human_bone_name_index[bone_names[i]] = i;
	}



    for (const StringName &E : p_animations) {
        Ref<Animation> animation = player->get_animation(E);
        if(animation.is_valid())
        {
            Ref<Animation> new_animation;
			new_animation = animation->duplicate();
            if(skeleton == nullptr)
            {
                new_animation->remap_node_to_bone_name(bone_names);
            }

			// 如果存在人形动作配置,转换动画为人形动画
			if (animation_human_config.is_valid()) {
				new_animation = HumanAnim::HumanAnimmation::build_human_animation(bone_map_skeleton, *animation_human_config.ptr(), new_animation, bone_map, true);
			}
            new_animation->set_animation_group(animation_group);
			new_animation->set_animation_tag(animation_tag);
            new_animation->optimize();
#if EDITOR_OPTIMIZE_ANIMATION
            new_animation->compress();
#endif
			String group = p_group;
			if (p_animations.size() == 1)
			{
				Vector<String> names = p_group.split("@");
				if (names.size() == 2)
				{
					group = names[0];
				}
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
            if(animation_human_config.is_valid())  {
			    save_fbx_res("human_animation", group, new_animation, save_path, true);
            }
            else {
			    save_fbx_res("animation", group, new_animation, save_path, true);
            }
            
        }
    }
	if (is_node_skeleton) {
		memdelete(bone_map_skeleton);
		bone_map_skeleton = nullptr;
	}
    p_node->queue_free();
}


void AnimationProcessPanel::_on_conver_single_pressed() {
    if( !FileAccess::exists(single_animation_file_path) ) {
        return;
    }
    if(single_animation_group_name.str().is_empty()) {
        WARN_PRINT("请先设置动画组名");
        return;
    }
    if(single_animation_tag_name.str().is_empty()) {
        WARN_PRINT("请先设置动画标签名");
        return;
    }
    editor_build_animation_form_path(single_animation_file_path,single_animation_group_name,single_animation_tag_name);    
}



void AnimationProcessPanel::editor_convert_animations(String p_file_path, const StringName& animation_group, const StringName& animation_tag)
{

    PackedStringArray files = DirAccess::get_files_at(p_file_path);

    for (int i = 0; i < files.size(); ++i) {
        String file = files[i];
        String ext = file.get_extension().to_lower();
        if (ext == "fbx" || ext == "gltf" || ext == "glb") {
            editor_build_animation_form_path(p_file_path.path_join(file), animation_group, animation_tag);
        }
    }
    PackedStringArray dirs = DirAccess::get_directories_at(p_file_path);
    for (int i = 0; i < dirs.size(); ++i) {
        String dir = p_file_path.path_join(dirs[i]);
        editor_convert_animations(dir, animation_group, animation_tag);
    }

}

void AnimationProcessPanel::_on_conver_multe_pressed() {
    if( !DirAccess::exists(multe_animation_file_path) ) {
        return;
    }
    if(multe_animation_group_name.str().is_empty()) {
        WARN_PRINT("请先设置动画组名");
        return;
    }
    if(multe_animation_tag_name.str().is_empty()) {
        WARN_PRINT("请先设置动画标签名");
        return;
    }
    editor_convert_animations(multe_animation_file_path,multe_animation_group_name,multe_animation_tag_name);
    
}
