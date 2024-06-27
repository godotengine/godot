#ifndef BODY_PART_H
#define BODY_PART_H

#include "scene/resources/packed_scene.h"
#include "scene/resources/3d/skin.h"
#include "scene/resources/mesh.h"
#include "scene/animation/animation_player.h"
#include "scene/animation/animation_tree.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "scene/3d/mesh_instance_3d.h"


// 身体的其他组件,不能播放动画，只能对骨骼进行设置
// 但是可以挂一些IK组件之类的,比如裙子,头发
class CharacterBodyPart : public Resource
{
    GDCLASS(CharacterBodyPart, Resource);
    static void _bind_methods();

    Ref<Skin>    skin;
    Ref<Mesh>    mesh;
    Ref<Material> material;


public:
    void set_skin(const Ref<Skin>& p_skin)
    {
        skin = p_skin;
	    emit_signal(SNAME("changed"));
    }
    Ref<Skin>    get_skin() const
    {
        return skin;
    }

    void set_mesh(const Ref<Mesh>& p_mesh)
    {
        mesh = p_mesh;
	    emit_signal(SNAME("changed"));
    }

    Ref<Mesh>    get_mesh() const
    {
        return mesh;
    }

    void set_material(const Ref<Material>& p_material)
    {
        material = p_material;
	    emit_signal(SNAME("changed"));
    }

    Ref<Material>    get_material() const
    {
        return material;
    }

    void init_form_mesh_instance(MeshInstance3D *mesh_instance,const Dictionary& p_bone_mapping = Dictionary())
    {
        ERR_FAIL_COND(mesh_instance == nullptr);
        skin = mesh_instance->get_skin();
        mesh = mesh_instance->get_mesh();
        material = mesh_instance->get_material_override();
        if(skin.is_valid())
        {
            skin = skin->duplicate();
            skin->set_human_bone_mapping(p_bone_mapping);
        }
        if(mesh.is_valid())
        {
            mesh = mesh->duplicate();
        }
        if(material.is_valid())
        {
            material = material->duplicate();
        }
    }
    bool isUsingRootBone = false;
};
// 身体骨骼的绑定
struct CharacterBodyBoneAttachment
{
    int attachBoneIdx = -1;

    void attachToBone(Skeleton3D *main_skeleton,int rootAttachBoneIdx, Skeleton3D *curr_skeleton)
    {
        auto rootBone = main_skeleton->get_bone_global_pose(rootAttachBoneIdx);
        curr_skeleton->set_bone_global_pose_override(attachBoneIdx, rootBone, 1, true);
    }

};
class CharacterBodyPartInstane : public Resource
{
    GDCLASS(CharacterBodyPartInstane, Resource);
    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("_on_part_changed"), &CharacterBodyPartInstane::_on_part_changed);
        ClassDB::bind_method(D_METHOD("set_part","p_part"), &CharacterBodyPartInstane::set_part);
        ClassDB::bind_method(D_METHOD("get_part"), &CharacterBodyPartInstane::get_part);

        ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "part", PROPERTY_HINT_RESOURCE_TYPE, "CharacterBodyPart"), "set_part", "get_part");

        
    }
public:
    Ref<CharacterBodyPart> part;
    Ref<Skin>    skin;
    MeshInstance3D *mesh_instance = nullptr;
    Skeleton3D *skeleton = nullptr;

    void set_part(Ref<CharacterBodyPart> p_part)
    {
        if(part == p_part)
        {
            return;
        }
        if(part.is_valid())
        {
            part->disconnect("part_changed",Callable(this,"_on_part_changed"));
        }
        part = p_part;
        if(part.is_valid())
        {
            part->connect("part_changed",Callable(this,"_on_part_changed"));
        }
        init();
    }
    Ref<CharacterBodyPart> get_part() 
    {
        return part;
    }
    void set_skeleton(Skeleton3D *p_skeleton)
    {
        if(skeleton == p_skeleton)
        {
            return;
        }
        skeleton = p_skeleton;
        init();
    }
    void init()
    {
        if(skeleton == nullptr)
        {
            clear();
            return;
        }
        if(part.is_valid())
        {
            skin = part->get_skin();
            if(mesh_instance != nullptr)
            {
                mesh_instance->queue_free();
            }
            mesh_instance = memnew(MeshInstance3D);
            if(mesh_instance)
            {
                skeleton->add_child(mesh_instance, true);
                mesh_instance->set_owner(skeleton->get_owner());
                if(skin.is_valid())
                {
                    skin = skin->duplicate();
                    mesh_instance->set_skeleton_path(NodePath("../Skeleton3D"));
                }
                mesh_instance->set_mesh(part->get_mesh());
                mesh_instance->set_skin(skin);
                mesh_instance->set_material_override(part->get_material());
            }
        }

    }
    void _on_part_changed()
    {
        clear();
        init();
    }

    void clear()
    {
        part.unref();
        skin.unref();
        if(mesh_instance != nullptr)
        {
            mesh_instance->queue_free();
            mesh_instance = nullptr;
        }
        skeleton = nullptr;
    }
    

    
    // void on_bone_pose_update(int p_bone_index)
    // {
    //     if(boneAttachment.has(p_bone_index))
    //     {
    //         boneAttachment[p_bone_index].attachToBone(skeleton,p_bone_index,skeleton);
    //     }
    // }
};

// 角色預製體
class CharacterBodyPrefab : public Resource
{
    GDCLASS(CharacterBodyPrefab, Resource);
    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_parts","p_parts"), &CharacterBodyPrefab::set_parts);
        ClassDB::bind_method(D_METHOD("get_parts"), &CharacterBodyPrefab::get_parts);
        ClassDB::bind_method(D_METHOD("set_skeleton_path","p_skeleton_path"), &CharacterBodyPrefab::set_skeleton_path);
        ClassDB::bind_method(D_METHOD("get_skeleton_path"), &CharacterBodyPrefab::get_skeleton_path);

        ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "parts", PROPERTY_HINT_ARRAY_TYPE,MAKE_RESOURCE_TYPE_HINT("CharacterBodyPart")), "set_parts", "get_parts");
        ADD_PROPERTY(PropertyInfo(Variant::STRING, "skeleton_path", PROPERTY_HINT_FILE, "*.tscn,*.scn"), "set_skeleton_path", "get_skeleton_path");
    }

public:
    void set_parts(TypedArray<CharacterBodyPart> p_parts)
    {
        parts = p_parts;
    }
    TypedArray<CharacterBodyPart> get_parts()
    {
        return parts;
    }

    void set_skeleton_path(String p_skeleton_path)
    {
        skeleton_path = p_skeleton_path;
    }
    String get_skeleton_path()
    {
        return skeleton_path;
    }

    TypedArray<CharacterBodyPart> parts;
    String skeleton_path;
};
#endif