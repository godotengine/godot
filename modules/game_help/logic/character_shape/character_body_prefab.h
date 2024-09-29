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

        ClassDB::bind_method(D_METHOD("set_is_human","p_is_human"), &CharacterBodyPrefab::set_is_human);
        ClassDB::bind_method(D_METHOD("get_is_human"), &CharacterBodyPrefab::get_is_human);

        ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "parts", PROPERTY_HINT_ARRAY_TYPE,"String"), "set_parts", "get_parts");
        ADD_PROPERTY(PropertyInfo(Variant::STRING, "skeleton_path", PROPERTY_HINT_FILE, "*.tscn,*.scn"), "set_skeleton_path", "get_skeleton_path");
        ADD_PROPERTY(PropertyInfo(Variant::BOOL, "is_human"), "set_is_human", "get_is_human");
        
    }

public:
    void set_parts(Dictionary p_parts)
    {
        is_loading = false;
        parts.clear();
        auto keys = p_parts.keys();
        for(int i = 0;i < keys.size();i++)
        {
            parts[keys[i]] = p_parts[keys[i]];
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
		is_loading = false;
        skeleton_path = p_skeleton_path;
        emit_changed();
    }
    String get_skeleton_path()
    {
        return skeleton_path;
    }


    void set_is_human(bool p_is_human) { is_human = p_is_human; }
    bool get_is_human() { return is_human; }

    TypedArray<CharacterBodyPart> load_part();

    HashMap<String,bool> parts;
    String skeleton_path;
    TypedArray<CharacterBodyPart> part_cache;
    bool is_human = false;
    bool is_loading = false;
};


#endif
