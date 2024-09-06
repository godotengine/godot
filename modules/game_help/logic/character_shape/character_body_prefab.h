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

        ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "parts", PROPERTY_HINT_ARRAY_TYPE,"String"), "set_parts", "get_parts");
        ADD_PROPERTY(PropertyInfo(Variant::STRING, "skeleton_path", PROPERTY_HINT_FILE, "*.tscn,*.scn"), "set_skeleton_path", "get_skeleton_path");
    }

public:
    void set_parts(Dictionary p_parts)
    {
        parts.clear();
        auto keys = p_parts.keys();
        for(int i = 0;i < keys.size();i++)
        {
            parts[keys[i]] = true;
        }
        emit_changed();
    }
    Dictionary get_parts()
    {
        Dictionary rs;
        for(const KeyValue<String,bool> &E : parts)
        {
            rs[E.key] = E.value;
        }
        return rs;
    }

    void set_skeleton_path(String p_skeleton_path)
    {
        skeleton_path = p_skeleton_path;
        emit_changed();
    }
    String get_skeleton_path()
    {
        return skeleton_path;
    }
    TypedArray<CharacterBodyPart> load_part()
    {
        TypedArray<CharacterBodyPart> rs;
        for(const KeyValue<String,bool> &E : parts)
        {
            if(!E.value)
            {
                continue;
            }
            if(!FileAccess::exists(E.key))
            {
                continue;
            }
            Ref<CharacterBodyPart> part = ResourceLoader::load(E.key);
            if(part.is_valid())
            {
                rs.push_back(part);
            }
        }
        return rs;
    }

    HashMap<String,bool> parts;
    String skeleton_path;
};


#endif