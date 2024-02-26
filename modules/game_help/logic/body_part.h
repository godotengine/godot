

#include "scene/resources/packed_scene.h"
#include "scene/animation/animation_player.h"
#include "scene/animation/animation_tree.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"


// 身体的其他组件,不能播放动画，只能对骨骼进行设置
// 但是可以挂一些IK组件之类的,比如裙子,头发
class BodyPart : public Resource
{
    GDCLASS(BodyPart, Resource);

    // 
    String meshName;
    bool isUsingRootBone = false;
};
// 身体骨骼的绑定
struct BodyBoneAttachment
{
    int attachBoneIdx = -1;

    void attachToBone(Skeleton3D *main_skeleton,int rootAttachBoneIdx, Skeleton3D *curr_skeleton)
    {
        auto rootBone = main_skeleton->get_bone_global_pose(rootAttachBoneIdx);
        curr_skeleton->set_bone_global_pose_override(attachBoneIdx, rootBone, 1, true);
    }

};
class BodyPartInstane : public Node3D
{
    GDCLASS(BodyPartInstane, Node3D);
    Skeleton3D *skeleton;    
    HashMap<int,BodyBoneAttachment> boneAttachment;
    void on_bone_pose_update(int p_bone_index)
    {
        if(boneAttachment.has(p_bone_index))
        {
            boneAttachment[p_bone_index].attachToBone(skeleton,p_bone_index,skeleton);
        }
    }
};