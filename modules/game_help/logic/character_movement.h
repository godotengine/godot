#ifndef _CHARACTER_MOVEMENT_H_
#define _CHARACTER_MOVEMENT_H_

#include "core/object/ref_counted.h"
#include "scene/3d/node_3d.h"

// 移动控制器,主要處理吸附移动物体(电梯,巨型boss,坐骑,飞行物体,汽车)
class CharacterMovement : public RefCounted
{
    GDCLASS(CharacterMovement,RefCounted);
    static void _bind_methods()
    {
        
    }
public:
    void update();
    const Transform3D& get_global_transform();
    void set_world_transform(const Transform3D & p_trans);
    void set_attach_target(ObjectID id);
    void detach_target();
    void moveing(const Vector3& pos);
    void moveing_forward(float dis,bool is_ground = false);
    void looking(const Vector3& pos,bool is_ground = false);
    void moveing_right(float dis,bool is_ground = false);
    void moveing_up(float dis);
    

    void on_attach_target_exit();
    void on_attack_transform_change();
protected:
    // 電梯，移動物件
    ObjectID attach_target;
    ObjectID target;
    Transform3D attach_world_pos;
    Transform3D world_pos;
    Transform3D local_rot;
    bool is_move = true;
};

#endif