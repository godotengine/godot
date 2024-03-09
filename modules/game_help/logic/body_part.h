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
    }
    Ref<Skin>    get_skin() const
    {
        return skin;
    }

    void set_mesh(const Ref<Mesh>& p_mesh)
    {
        mesh = p_mesh;
    }

    Ref<Mesh>    get_mesh() const
    {
        return mesh;
    }

    void set_material(const Ref<Material>& p_material)
    {
        material = p_material;
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
struct CharacterBodyPartInstane
{
    Ref<CharacterBodyPart> part;
    Ref<Skin> skin;
    MeshInstance3D *mesh_instance = nullptr;
    Skeleton3D *skeleton = nullptr;

    void init(Node* scene,Ref<CharacterBodyPart> p_part,Skeleton3D *p_skeleton)
    {
        part = p_part;
        if(part.is_valid())
        {
            skin = part->get_skin();
            mesh_instance = memnew(MeshInstance3D);
            if(mesh_instance)
            {
                mesh_instance->set_owner(scene);
                p_skeleton->add_child(mesh_instance, true);
                if(skin.is_valid())
                {
                    skin = skin->duplicate();
                }
                mesh_instance->set_mesh(part->get_mesh());
                mesh_instance->set_skin(skin);
                mesh_instance->set_material_override(part->get_material());
            }
            skeleton = p_skeleton;
        }

    }
    void clear()
    {
        part.unref();
        skin.unref();
        if(mesh_instance != nullptr)
        {
            memdelete(mesh_instance);
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

#endif