#include "charcter_prefab_process_panel.h"
#include "../../logic/animator/human_animation.h"
#include "../../logic/animator/character_animation_node.h"

#include "scene/resources/packed_scene.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/gui/separator.h"


#include "core/io/file_access.h"
#include "core/io/dir_access.h"
#include "core/io/json.h"


HBoxContainer* CharacterPrefabProcessPanel::create_line(Control* control , bool is_side_separator ) {
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
HBoxContainer* CharacterPrefabProcessPanel::create_line(Control* control0 , Control* control1,  bool is_side_separator ) {
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

HBoxContainer* CharacterPrefabProcessPanel::create_line(Control* control0 , Control* control1, Control* control2,  bool is_side_separator ) {
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

HBoxContainer* CharacterPrefabProcessPanel::create_line(Control* control0 , Control* control1, Control* control2, Control* control3,  bool is_side_separator ) {
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

CharacterPrefabProcessPanel::CharacterPrefabProcessPanel() {
    load_charcter_prefab_config();

    HBoxContainer* hb = memnew(HBoxContainer);
    hb->set_h_size_flags(SIZE_EXPAND_FILL);
    add_child(hb);
    {
        VBoxContainer* vb = memnew(VBoxContainer);
        vb->set_custom_minimum_size(Vector2(300, 0));
        hb->add_child(vb);

        property_preview_mesh_path = memnew(EditorPropertyPath);
        property_preview_mesh_path->set_label(L"選擇预制件：");
        property_preview_mesh_path->set_object_and_property(this, SNAME("preview_mesh_path"));
		property_preview_mesh_path->setup({ "*.fbx", "*.gltf","*.glb" }, false, false);
        property_preview_mesh_path->set_h_size_flags(SIZE_EXPAND_FILL);
		property_preview_mesh_path->set_custom_property(true);
        property_preview_mesh_path->update_property();
        vb->add_child(property_preview_mesh_path);

        preview = memnew(SceneViewPanel);
        preview->set_custom_minimum_size(Vector2(400, 400));
        vb->add_child(preview);
        if(preview_mesh_path != "") {
            preview->set_scene_path(preview_mesh_path);
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
            label->set_text(L"单个角色处理");
            label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
            label->set_modulate(Color(1,0.8,0.7,1));
            vb->add_child(create_line(  label,true));
                                    
            VSeparator* sep = memnew(VSeparator);
            vb->add_child(sep);

            {

                single_charcter_prefab_group = memnew(EditorPropertyTextEnum);
                single_charcter_prefab_group->set_label(L"动画组");
                single_charcter_prefab_group->set_custom_minimum_size(Vector2(300, 0));
                single_charcter_prefab_group->set_object_and_property(this, "single_charcter_prefab_group");
                single_charcter_prefab_group->set_dynamic(true, "get_charcter_prefab_groups");
                single_charcter_prefab_group->setup(Vector<String>());
				single_charcter_prefab_group->set_custom_property(true);
                single_charcter_prefab_group->update_property();

                conver_single_button = memnew(Button);
                conver_single_button->set_text(L"转换");
                conver_single_button->connect(SceneStringName(pressed), callable_mp(this, &CharacterPrefabProcessPanel::_on_conver_single_pressed));


                
                vb->add_child(create_line(  single_charcter_prefab_group,true));

                vb->add_child(conver_single_button);
            }

        
        }



        {
            Label* label = memnew(Label);
            label->set_h_size_flags(SIZE_EXPAND_FILL);
            label->set_text(L"多个角色处理");
            label->set_horizontal_alignment(HORIZONTAL_ALIGNMENT_CENTER);
            label->set_modulate(Color(1,0.8,0.7,1));

            vb->add_child(create_line(label,true));
                                    
            VSeparator* sep = memnew(VSeparator);
            vb->add_child(sep);
            {
                multe_path = memnew(EditorPropertyPath);
                multe_path->set_label(L"选择文件夹");
                multe_path->set_object_and_property(this, "multe_charcter_prefab_file_path");
                multe_path->set_h_size_flags(SIZE_EXPAND_FILL);
                multe_path->setup(Vector<String>(), true, false);
				multe_path->set_custom_property(true);

                multe_charcter_prefab_group = memnew(EditorPropertyTextEnum);
                multe_charcter_prefab_group->set_label(L"动画组");
                multe_charcter_prefab_group->set_custom_minimum_size(Vector2(300, 0));
                multe_charcter_prefab_group->set_object_and_property(this, "multe_charcter_prefab_group");
                multe_charcter_prefab_group->set_dynamic(true, "get_charcter_prefab_groups");
                multe_charcter_prefab_group->setup(Vector<String>());
				multe_charcter_prefab_group->set_custom_property(true);
                multe_charcter_prefab_group->set_modulate(Color(1,0.8,0.7,1));
                multe_charcter_prefab_group->update_property();


                conver_multe_button = memnew(Button);
                conver_multe_button->set_text(L"转换");
                conver_multe_button->connect(SceneStringName(pressed), callable_mp(this, &CharacterPrefabProcessPanel::_on_conver_multe_pressed));
                
                vb->add_child(create_line(  multe_path,multe_charcter_prefab_group));

                vb->add_child(conver_multe_button);

            }

            


        }

    }

}
void CharacterPrefabProcessPanel::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_preview_mesh_path", "path"), &CharacterPrefabProcessPanel::set_preview_mesh_path);
    ClassDB::bind_method(D_METHOD("get_preview_mesh_path"), &CharacterPrefabProcessPanel::get_preview_mesh_path);

    ClassDB::bind_method(D_METHOD("get_charcter_prefab_groups"), &CharacterPrefabProcessPanel::get_charcter_prefab_groups);


    ClassDB::bind_method(D_METHOD("set_single_charcter_prefab_group", "group"), &CharacterPrefabProcessPanel::set_single_charcter_prefab_group);
    ClassDB::bind_method(D_METHOD("get_single_charcter_prefab_group"), &CharacterPrefabProcessPanel::get_single_charcter_prefab_group);

    ClassDB::bind_method(D_METHOD("set_multe_charcter_prefab_file_path", "path"), &CharacterPrefabProcessPanel::set_multe_charcter_prefab_file_path);
    ClassDB::bind_method(D_METHOD("get_multe_charcter_prefab_file_path"), &CharacterPrefabProcessPanel::get_multe_charcter_prefab_file_path);
    ClassDB::bind_method(D_METHOD("set_multe_charcter_prefab_group", "group"), &CharacterPrefabProcessPanel::set_multe_charcter_prefab_group);
    ClassDB::bind_method(D_METHOD("get_multe_charcter_prefab_group"), &CharacterPrefabProcessPanel::get_multe_charcter_prefab_group);

    ADD_PROPERTY(PropertyInfo(Variant::STRING, "preview_mesh_path"), "set_preview_mesh_path", "get_preview_mesh_path");

    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "single_charcter_prefab_group"), "set_single_charcter_prefab_group", "get_single_charcter_prefab_group");

    ADD_PROPERTY(PropertyInfo(Variant::STRING, "multe_charcter_prefab_file_path"), "set_multe_charcter_prefab_file_path", "get_multe_charcter_prefab_file_path");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "multe_charcter_prefab_group"), "set_multe_charcter_prefab_group", "get_multe_charcter_prefab_group");


}

void CharacterPrefabProcessPanel::set_preview_mesh_path(const String& path) {
	Ref<PackedScene> scene = ResourceLoader::load(path);
	if (scene.is_null())
	{
		print_line(L"SceneViewPanel: 路径不存在 :" + path);
        return;
	}
	Node* p_node = scene->instantiate(PackedScene::GEN_EDIT_STATE_DISABLED);
    Node3D* p_mesh = Object::cast_to<Node3D>(p_node);
    if(preview != nullptr && p_mesh != nullptr){
        preview->edit(p_mesh);
    }
    else {
		print_line(L"SceneViewPanel: 请选择一个模型 :" + path);
        p_node->queue_free();
        return;
    }
    preview_mesh_path = path;
    property_preview_mesh_path->update_property();

    save_charcter_prefab_config();
}
String CharacterPrefabProcessPanel::get_preview_mesh_path() {
    return preview_mesh_path;
}

void CharacterPrefabProcessPanel::set_single_charcter_prefab_group(const String& group) {
    single_charcter_prefab_group_name = group;     
    save_charcter_prefab_config();   
    single_charcter_prefab_group->update_property();
}
StringName CharacterPrefabProcessPanel::get_single_charcter_prefab_group() {
    return single_charcter_prefab_group_name;
}


void CharacterPrefabProcessPanel::set_multe_charcter_prefab_file_path(const String& path) {
    multe_charcter_prefab_file_path = path;
    save_charcter_prefab_config();
    multe_path->update_property();
}
String CharacterPrefabProcessPanel::get_multe_charcter_prefab_file_path() {
    return multe_charcter_prefab_file_path;
}
void CharacterPrefabProcessPanel::set_multe_charcter_prefab_group(const String& group) {
    multe_charcter_prefab_group_name = group;
    save_charcter_prefab_config();
    multe_charcter_prefab_group->update_property();
}
StringName CharacterPrefabProcessPanel::get_multe_charcter_prefab_group() {
    return multe_charcter_prefab_group_name;
}

Array CharacterPrefabProcessPanel::get_charcter_prefab_groups() {
    Array arr;
    CharacterManager::get_singleton()->get_animation_groups(&arr);
    return arr;
}
void CharacterPrefabProcessPanel::save_charcter_prefab_config() {
    String path = "res://.godot/charcter_prefab_process_panel_config.json";

    Dictionary dict;
    dict["preview_mesh_path"] = preview_mesh_path;

    dict["single_charcter_prefab_group"] = single_charcter_prefab_group_name;

    dict["multe_charcter_prefab_file_path"] = multe_charcter_prefab_file_path;
    dict["multe_charcter_prefab_group"] = multe_charcter_prefab_group_name;

    Ref<FileAccess> file = FileAccess::open(path, FileAccess::WRITE);
    file->store_string(JSON::stringify(dict));
    file->close();
}

void CharacterPrefabProcessPanel::load_charcter_prefab_config() {
    String path = "res://.godot/charcter_prefab_process_panel_config.json";
    Ref<FileAccess> file = FileAccess::open(path, FileAccess::READ);
    if (file == nullptr) {
        return;
    }
    String json = file->get_as_text();
    file->close();
    Dictionary dict = JSON::parse_string(json);

    preview_mesh_path = dict["preview_mesh_path"];

    single_charcter_prefab_group_name = dict["single_charcter_prefab_group"];

    multe_charcter_prefab_file_path = dict["multe_charcter_prefab_file_path"];
    multe_charcter_prefab_group_name = dict["multe_charcter_prefab_group"];

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
	print_line(L"CharacterPrefabProcessPanel.save_fbx_res: 存储资源 :" + save_path);
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
	print_line(L"CharacterPrefabProcessPanel.save_fbx_res: 存储资源 :" + save_path);
	save_path = sub_path.path_join(p_resource->get_name() + (is_resource ? ".tres" : ".tscn"));
}


static void get_fbx_meshs(Node* p_node, HashMap<String, MeshInstance3D* >& meshs)
{

	for (int i = 0; i < p_node->get_child_count(); i++)
	{
		Node* child = p_node->get_child(i);
		if (child->get_class() == "MeshInstance3D")
		{
			MeshInstance3D* mesh = Object::cast_to<MeshInstance3D>(child);
			if (!meshs.has(mesh->get_name())) {
				meshs[mesh->get_name()] = mesh;
			}
			else {
				String name = mesh->get_name();
				int index = 1;
				while (meshs.has(name + "_" + itos(index))) {
					name = mesh->get_name().str() + "_" + itos(index);
					index++;
				}
				meshs[name] = mesh;
			}
		}
		get_fbx_meshs(child, meshs);
	}
}

static void reset_owenr(Node* node, Node* owenr)
{
	for (int i = 0; i < node->get_child_count(); ++i)
	{
		Node* c = node->get_child(i);
		c->set_owner(nullptr);
		reset_owenr(c, owenr);
	}
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


void CharacterPrefabProcessPanel::_on_conver_single_pressed() {
    if( !FileAccess::exists(preview_mesh_path) ) {
		WARN_PRINT("_on_conver_single_pressed : 文件路徑不存在！");
        return;
    }
    if(single_charcter_prefab_group_name.str().is_empty()) {
        WARN_PRINT("请先设置动画组名");
        return;
    }
	String path = ResourceUID::ensure_path(preview_mesh_path);
	build_prefab(path,single_charcter_prefab_group_name,true);
}



void CharacterPrefabProcessPanel::editor_convert_prefab(String p_file_path, const StringName& charcter_prefab_group)
{

    PackedStringArray files = DirAccess::get_files_at(p_file_path);

    for (int i = 0; i < files.size(); ++i) {
        String file = files[i];
        String ext = file.get_extension().to_lower();
        if (ext == "fbx" || ext == "gltf" || ext == "glb") {
			build_prefab(p_file_path.path_join(file), charcter_prefab_group,true);
        }
    }
    PackedStringArray dirs = DirAccess::get_directories_at(p_file_path);
    for (int i = 0; i < dirs.size(); ++i) {
        String dir = p_file_path.path_join(dirs[i]);
        editor_convert_prefab(dir, charcter_prefab_group);
    }

}

void CharacterPrefabProcessPanel::_on_conver_multe_pressed() {
    if( !DirAccess::exists(multe_charcter_prefab_file_path) ) {
        return;
    }
    if(multe_charcter_prefab_group_name.str().is_empty()) {
        WARN_PRINT("请先设置动画组名");
        return;
    }
    editor_convert_prefab(multe_charcter_prefab_file_path,multe_charcter_prefab_group_name);
    
}
Ref<CharacterBodyPrefab> CharacterPrefabProcessPanel::build_prefab(const String& mesh_path, const StringName& animation_group,bool p_is_skeleton_human)
{
    if (!FileAccess::exists(mesh_path))
    {
        return Ref<CharacterBodyPrefab>();
    }

    // 加载模型
    Ref<PackedScene> scene = ResourceLoader::load(mesh_path);

    if (scene.is_null())
    {
        print_line(L"CharacterPrefabProcessPanel: 路径不存在 :" + mesh_path);
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
        
        Ref<HumanBoneConfig> config;

        skeleton->set_human_bone_mapping(bone_map);
        
        if(p_is_skeleton_human)
        {
            config.instantiate();
            HashMap<String,String> _bone_label = HumanAnim::HumanAnimmation::get_bone_label();
            HumanAnim::HumanAnimmation::build_virtual_pose(skeleton, *config.ptr(), _bone_label);
            skeleton->set_human_config(config);
            config = skeleton->get_human_config();
            config->set_name("human_config");
            save_fbx_res("human_config", p_group, config, ske_save_path, true);
        }

        // 存储骨骼映射
        Ref<CharacterBoneMap> bone_map_ref;
        bone_map_ref.instantiate();
        bone_map_ref->set_name("bone_map");
        bone_map_ref->set_bone_map(bone_map);
        bone_map_ref->set_bone_names(bone_names);
        bone_map_ref->set_human_config(config);
        bone_map_ref->set_skeleton_path(p_group.path_join("skeleton.scn" ));
        if(p_is_skeleton_human) {
            save_fbx_res("human_bone_map", p_group, bone_map_ref, bone_map_save_path, true);
        }
        else {
            save_fbx_res("bone_map", p_group, bone_map_ref, bone_map_save_path, true);            
        }


        skeleton->set_owner(nullptr);
        reset_owenr(skeleton, skeleton);

        // 存儲骨架信息
        Ref<PackedScene> packed_scene;
        packed_scene.instantiate();
        packed_scene->pack(skeleton);
        packed_scene->set_name("skeleton");
        if(p_is_skeleton_human) {
            save_fbx_res("human_skeleton", p_group, packed_scene, ske_save_path, false);
        }
        else {
            save_fbx_res("skeleton", p_group, packed_scene, ske_save_path, false);
        }

    }
    // 生成预制体
    Ref<CharacterBodyPrefab> _body_prefab;
    _body_prefab.instantiate();
    _body_prefab->set_name(p_group);
    _body_prefab->set_resource_group(animation_group);
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
        if(p_is_skeleton_human) {
            save_fbx_res("human_meshs", p_group, part, save_path, true);
        }
        else {
            save_fbx_res("meshs", p_group, part, save_path, true);
        }
        _body_prefab->parts[save_path] = true;
    }
    // 保存预制体
    _body_prefab->skeleton_path = ske_save_path;
    _body_prefab->set_is_human(p_is_skeleton_human);
    if(p_is_skeleton_human) {
        save_fbx_res("human_prefab", p_group, _body_prefab, bone_map_save_path, true);
    }
    else {
        save_fbx_res("prefab", p_group, _body_prefab, bone_map_save_path, true);
    }


    p_node->queue_free();
    return _body_prefab;
}
