#ifndef CHARACTER_SHAPE_CHARACTER_BODY_PREFAB_H
#define CHARACTER_SHAPE_CHARACTER_BODY_PREFAB_H
#include "core/io/resource.h"
#include "character_body_part.h"

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

        ADD_PROPERTY(PropertyInfo(Variant::ARRAY, "parts", PROPERTY_HINT_ARRAY_TYPE,"String"), "set_parts", "get_parts");
        ADD_PROPERTY(PropertyInfo(Variant::STRING, "skeleton_path", PROPERTY_HINT_FILE, "*.tscn,*.scn"), "set_skeleton_path", "get_skeleton_path");
    }

public:
    void set_parts(TypedArray<String> p_parts)
    {
        parts = p_parts;
    }
    TypedArray<String> get_parts()
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
    TypedArray<CharacterBodyPart> load_part()
    {
        TypedArray<CharacterBodyPart> rs;
        for(int i=0;i<parts.size();i++)
        {
            Ref<CharacterBodyPart> part = ResourceLoader::load(parts[i]);
            if(part.is_valid())
            {
                rs.push_back(part);
            }
        }
        return rs;
    }

    TypedArray<String> parts;
    String skeleton_path;
};


#endif