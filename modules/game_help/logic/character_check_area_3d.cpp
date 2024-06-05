#include "character_check_area_3d.h"
#include "body_main.h"


void CharacterCheckArea3D::set_body_main(class CharacterBodyMain* p_mainBody)
{
    if(mainBody == p_mainBody)
    {
        return;
    }
    if(mainBody)
    {
        areaCollision->disconnect(SceneStringName(body_entered),callable_mp(this,&CharacterCheckArea3D::on_body_enter_area));
        areaCollision->disconnect(SceneStringName(body_exited),callable_mp(this,&CharacterCheckArea3D::on_body_exit_area));
        areaCollision->queue_free();
        areaCollision = nullptr;

    }
    mainBody = p_mainBody;
    if(mainBody)
    {            
        areaCollision = memnew(Area3D);
        mainBody->add_child(areaCollision);
        areaCollision->set_owner(mainBody);
        areaCollision->connect(SceneStringName(body_entered),callable_mp(this,&CharacterCheckArea3D::on_body_enter_area));
        areaCollision->connect(SceneStringName(body_exited),callable_mp(this,&CharacterCheckArea3D::on_body_exit_area));
        areaCollision->set_collision_layer(mainBody->get_collision_layer());
        areaCollision->set_collision_mask(collision_check_mask);
    }
    if(area_shape.is_valid())
    {
        area_shape->set_link_target(areaCollision);
    }

}