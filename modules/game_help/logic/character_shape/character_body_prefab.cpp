#include "character_body_prefab.h"
#include "../character_manager.h"
TypedArray<CharacterBodyPart> CharacterBodyPrefab::load_part()
{
    if(is_loading)
    {
        return part_cache;
    }
	is_loading = true;
    part_cache.clear();
	String path;
    for(const KeyValue<String,bool> &E : parts)
    {
        if(!E.value)
        {
            continue;
        }
		if (E.key.begins_with("res://"))
			path = E.key;
		else
			path = CharacterManager::get_singleton()->get_mesh_root_path(is_human).path_join(E.key);
        if(!FileAccess::exists(path))
        {
            continue;
        }
        Ref<CharacterBodyPart> part;
		part = ResourceLoader::load(path);
        if(part.is_valid())
        {
            part_cache.push_back(part);
        }
    }
    return part_cache;
}
