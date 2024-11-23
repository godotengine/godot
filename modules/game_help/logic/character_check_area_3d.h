#ifndef _CHARACTER_CHECK_AREA_3D_H_
#define _CHARACTER_CHECK_AREA_3D_H_
#include "scene/3d/node_3d.h"
#include "scene/3d/physics/area_3d.h"
#include "scene/3d/physics/character_body_3d.h"
#include "scene/3d/physics/collision_shape_3d.h"


struct CellPos
{
    enum class  SizeType
    {
        S1 = 1,
        S2 = 2,
        S4 = 4,
        S8 = 8,
        S16 = 16,
        S32 = 32,
        S64 = 64,
        S128 = 128,
        S256 = 256,
    };
    public:
    union body_main
    {
        struct
        {
            short x, z;
        };
        uint32_t key;
    } value;
    
    CellPos(){
        value.x = 0;
        value.z = 0;
    }
    CellPos(int _x, int _z)
    {
        value.x = _x;
        value.z = _z;
    }
    void Offset(Vector2i& _delta, int _cellSize = 2)
    {
        value.x += Math::fast_ftoi(_delta.x / _cellSize);
        value.z += Math::fast_ftoi(_delta.y / _cellSize);
    }
    void Offset(Vector3& _delta, int _cellSize = 2)
    {
        value.x += Math::fast_ftoi(_delta.x / _cellSize);
        value.z += Math::fast_ftoi(_delta.z / _cellSize);
    }
    friend bool operator == (const CellPos& lhs, const CellPos& rhs)
    {
        return (lhs.value.x == rhs.value.x && lhs.value.z == rhs.value.z);
    }
    friend bool operator != (const CellPos& lhs,const CellPos& rhs)
    {
        return !(lhs == rhs);
    }

    /// <summary>
    /// 把一个long pack 到2个uint里面
    /// </summary>
    /// <param name="uid"></param>
    /// <param name="x 高位"></param>
    /// <param name="y 低位"></param>
    static void EncodeInt(uint32_t uid,CellPos& pos)
    {
        pos.value.x = (short)(uid >> 16);
        pos.value.z = (short)((uid << 16) >> 16);
    }
    static _FORCE_INLINE_ uint32_t hash(const CellPos &pos) {
        return pos.value.key;
    }

    Vector3 worldPosition(int _cellSize = 2)const
    {
        return Vector3(_cellSize * value.x, 0.0f, _cellSize * value.z);
    }

    Vector3 centerPosition(int _cellSize = 2)
    {
        return Vector3(_cellSize * (value.x + 0.5f), 0.0f, _cellSize * (value.z + 0.5f));
    } 
    static bool compare(const CellPos &p_lhs, const CellPos &p_rhs) {
        return (p_lhs.value.key == p_rhs.value.key);
    }
};


class CharacterCheckArea3DResult : public RefCounted
{
    GDCLASS(CharacterCheckArea3DResult, RefCounted);
    static void _bind_methods()
    {

    }
public:

    void update(const Vector3& pos,const Vector3& forward)
    {
        Node3D* node = Object::cast_to<Node3D>(ObjectDB::get_instance(character));
        if(node)
        {
            squareDistance = (node->get_global_transform().origin - pos).length_squared();
            angle = forward.angle_to(node->get_global_transform().origin - pos);
        }
        else
        {
            character = ObjectID();
        }
    }
    

public:
    CellPos cellPos;
    ObjectID character;
    float squareDistance;
    float angle;
};
// 角色的检测范围
class CharacterCheckArea3D : public RefCounted
{
    GDCLASS(CharacterCheckArea3D, RefCounted);
    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_name", "name"), &CharacterCheckArea3D::set_name);
        ClassDB::bind_method(D_METHOD("get_name"), &CharacterCheckArea3D::get_name);

        ClassDB::bind_method(D_METHOD("set_cell_size", "cell_size"), &CharacterCheckArea3D::set_cell_size);
        ClassDB::bind_method(D_METHOD("get_cell_size"), &CharacterCheckArea3D::get_cell_size);

        ClassDB::bind_method(D_METHOD("set_area_shape", "shape"), &CharacterCheckArea3D::set_area_shape);
        ClassDB::bind_method(D_METHOD("get_area_shape"), &CharacterCheckArea3D::get_area_shape);

        ClassDB::bind_method(D_METHOD("set_collision_check_mask", "mask"), &CharacterCheckArea3D::set_collision_check_mask);
        ClassDB::bind_method(D_METHOD("get_collision_check_mask"), &CharacterCheckArea3D::get_collision_check_mask);

        ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME, "name"), "set_name", "get_name");
        ADD_PROPERTY(PropertyInfo(Variant::INT, "cell_size"), "set_cell_size", "get_cell_size");
        ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "area_shape", PROPERTY_HINT_ARRAY_TYPE, MAKE_RESOURCE_TYPE_HINT("CollisionObject3DConnectionShape")), "set_area_shape", "get_area_shape");
        ADD_PROPERTY(PropertyInfo(Variant::INT, "collision_check_mask",PROPERTY_HINT_LAYERS_3D_PHYSICS), "set_collision_check_mask", "get_collision_check_mask");
    }
public:
    
    // 更新世界位移
    _FORCE_INLINE_ void update_world_move(const Vector3& pos)
    {
        s_world_move.value.x = pos.x / WorldPageSize;
        s_world_move.value.z = pos.z / WorldPageSize;
        is_update_coord = true;
    }
    _FORCE_INLINE_ Vector3 world_pos_to_local_pos(const Vector3& pos)
    {
        Vector3 ret;
        ret.x = pos.x - s_world_move.value.x * WorldPageSize;
        ret.z = pos.z - s_world_move.value.z * WorldPageSize;
        ret.y = pos.y;
        return ret;
    }
    CellPos world_pos_to_cell_pos(const Vector3& pos)
    {
        CellPos ret;
        Vector3 local_pos = world_pos_to_local_pos(pos);
        ret.value.x = Math::fast_ftoi(local_pos.x / cell_size);
        ret.value.z = Math::fast_ftoi(local_pos.z / cell_size);
        return ret;
    }
    void set_name(StringName p_name)
    {
        name = p_name;
    }
    StringName get_name()
    {
        return name;
    }

    void set_cell_size(int p_cell_size)
    {
        cell_size = p_cell_size;
    }
    int get_cell_size()
    {
        return cell_size;
    }

    void set_collision_check_mask(int p_mask)
    {
        collision_check_mask = p_mask;
        Area3D* areaCollision = Object::cast_to<Area3D>(ObjectDB::get_instance(areaCollisionID));
        if(areaCollision != nullptr) {
            areaCollision->set_collision_layer(collision_check_mask);
        }
    }
    int get_collision_check_mask()
    {
        return collision_check_mask;
    }
    

    void on_body_enter_area(Node3D *p_area)
    {
        Ref<CharacterCheckArea3DResult> result ;
        result.instantiate();
        result->character = p_area->get_instance_id();
        boundOtherCharacter.insert(p_area,result);
        is_update_coord = true;
		print_line("on_body_enter_area:" + p_area->get_name());
    }
    void on_body_exit_area(Node3D *p_area)
    {
        boundOtherCharacter.erase(p_area);
        is_update_coord = true;
		print_line("*on_body_exit_area:" + p_area->get_name());
    }
    void set_body_main(class CharacterBodyMain* p_mainBody);
    void update_coord();
    void get_bound_other_character_by_angle(TypedArray<CharacterCheckArea3DResult>& _array,float angle);

    void set_area_shape(TypedArray<CollisionObject3DConnectionShape> p_shape)
    {

		Area3D* areaCollision = Object::cast_to<Area3D>(ObjectDB::get_instance(areaCollisionID));
        for(int i = 0;i < p_shape.size();i++ ) {
            Ref<CollisionObject3DConnectionShape> shape = p_shape[i];
            shape->set_link_target(areaCollision);
        }
        area_shape = p_shape;
    }
    TypedArray<CollisionObject3DConnectionShape> get_area_shape()
    {
        return area_shape;
    }
    void on_owenr_chaanged_collision_layer() ;

    void init();
    CharacterCheckArea3D()
    {

    }
    ~CharacterCheckArea3D()
    {
        set_body_main(nullptr);
    }
 protected:
 protected:
    CellPos s_world_move;
    int cell_size = 2;
    CellPos world_page;
    const int WorldPageSize = 1024;
    StringName name;
    HashMap<Node3D*,Ref<CharacterCheckArea3DResult>> boundOtherCharacter;
    HashMap<CellPos,LocalVector<Ref<CharacterCheckArea3DResult>>,CellPos,CellPos> boundOtherCharacterByCoord;
    class CharacterBodyMain* mainBody = nullptr;
    ObjectID areaCollisionID;
	TypedArray<CollisionObject3DConnectionShape> area_shape;
    uint32_t collision_check_mask = 0xFFFFFFFF;
    bool is_update_coord = true;

};


#endif
