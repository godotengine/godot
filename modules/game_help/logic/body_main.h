

#include "scene/resources/packed_scene.h"
#include "scene/animation/animation_player.h"
#include "scene/animation/animation_tree.h"
#include "scene/3d/node_3d.h"
#include "scene/3d/skeleton_3d.h"
#include "body_part.h"
#include "animation_help.h"

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

class BodyMain : public Node3D {
    GDCLASS(BodyMain, Node3D);
    static void _bind_methods();

public:
    // 初始化身體
    void init_main_body(String p_mesh_file_path,StringName p_animation_group);

protected:
    Node3D *root = nullptr;
    Skeleton3D *skeleton = nullptr;
    AnimationPlayer *player = nullptr;
    AnimationTree *tree = nullptr;
    // 插槽信息
    HashMap<StringName,BodySocket> socket;
    // 身体部件信息
    HashMap<StringName,Ref<BodyPart>> bodyPart;


};