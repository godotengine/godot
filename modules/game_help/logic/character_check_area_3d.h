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



// 角色的检测范围
class CharacterCheckArea3D : public RefCounted
{
    GDCLASS(CharacterCheckArea3D, RefCounted);
    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_area_shape", "shape"), &CharacterCheckArea3D::set_area_shape);
        ClassDB::bind_method(D_METHOD("get_area_shape"), &CharacterCheckArea3D::get_area_shape);

        ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "area_shape", PROPERTY_HINT_RESOURCE_TYPE, "CollisionObject3DConnection"), "set_area_shape", "get_area_shape");
    }
public:
    
    // 更新世界位移
    _FORCE_INLINE_ void update_world_move(const Vector3& pos)
    {
        s_world_move.value.x = pos.x / WorldPageSize;
        s_world_move.value.z = pos.z / WorldPageSize;
    }
    _FORCE_INLINE_ Vector3 world_pos_to_local_pos(const Vector3& pos)
    {
        Vector3 ret;
        ret.x = pos.x - s_world_move.value.x * WorldPageSize;
        ret.z = pos.z - s_world_move.value.z * WorldPageSize;
        ret.y = pos.y;
        return ret;
    }
    
    void set_area_shape(Ref<CollisionObject3DConnection> p_shape)
    {
        if(area_shape == p_shape)
        {
            return;
        }
        if(area_shape.is_valid())
        {
            area_shape->set_link_target(nullptr);
        }
        area_shape = p_shape;
        if(area_shape.is_valid())
        {
            area_shape->set_link_target(areaCollision);
        }
    }
    Ref<CollisionObject3DConnection> get_area_shape()
    {
        return area_shape;
    }

    void on_body_enter_area(Node3D *p_area)
    {
        boundOtherCharacter.insert(p_area);
    }
    void on_body_exit_area(Node3D *p_area)
    {
        boundOtherCharacter.erase(p_area);
    }
    void set_body_main(class CharacterBodyMain* p_mainBody);
    CharacterCheckArea3D()
    {

    }
    ~CharacterCheckArea3D()
    {
        set_body_main(nullptr);
    }

    CellPos s_world_move;
    int cell_size = 2;
    CellPos world_page;
    const int WorldPageSize = 1024;
    StringName name;
    HashSet<Node3D*> boundOtherCharacter;
    HashMap<CellPos,Node3D*,CellPos,CellPos> boundOtherCharacterByCoord;
    class CharacterBodyMain* mainBody = nullptr;
    Area3D * areaCollision = nullptr;
    Ref<CollisionObject3DConnection> area_shape;
    uint32_t collision_check_mask = 0;

};


#endif