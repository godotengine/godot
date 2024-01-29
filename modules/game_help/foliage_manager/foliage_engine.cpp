#include "foliage_engine.h"
#include "foliage_render_manager.h"

namespace Foliage
{


    void FoliageMapChunkConfig::_bind_methods()
    {
        ClassDB ::bind_method(D_METHOD("set_chunk_asset_name", "map_name"), &FoliageMapChunkConfig::set_chunk_asset_name);
        ClassDB ::bind_method(D_METHOD("get_chunk_asset_name"), &FoliageMapChunkConfig::get_chunk_asset_name);

        ADD_PROPERTY(PropertyInfo(Variant::STRING, "chunk_asset_name", PROPERTY_HINT_NONE), "set_chunk_asset_name", "get_chunk_asset_name");
    }






    void FoliageMapConfig::_bind_methods()
    {
        ClassDB ::bind_method(D_METHOD("set_config", "_config"), &FoliageMapConfig::set_config);
        ClassDB ::bind_method(D_METHOD("get_config"), &FoliageMapConfig::get_config);

        
        ADD_PROPERTY(PropertyInfo(Variant::STRING, "config", PROPERTY_HINT_RESOURCE_TYPE,MAKE_RESOURCE_TYPE_HINT("FoliageMapConfig")), "set_config", "get_config");

    }


    void FoliageEngine::add_prototype(FoliagePrototypeAsset* _proto)
    {
        auto r = _proto->get_region();
        FoliageCellPos pos;
        pos.x = r.position.x / FoliageGlobals::PAGE_SIZE;
        pos.z = r.position.y / FoliageGlobals::PAGE_SIZE;
        prototypes.insert(pos.DecodeInt(),_proto);
    }
    void FoliageEngine::remove_prototype(FoliagePrototypeAsset* _proto)
    {
        auto r = _proto->get_region();
        FoliageCellPos pos;
        pos.x = r.position.x / FoliageGlobals::PAGE_SIZE;
        pos.z = r.position.y / FoliageGlobals::PAGE_SIZE;
        prototypes.erase(pos.DecodeInt());

    }

    void FoliageEngine::init(Ref<FoliageMapConfig> _map_config)
    {
        print_line("foliage engine init");
        clear();
        map_config = _map_config;
    }
    void FoliageEngine::update(const Vector3& camera_pos)
    {
        // 檢測地块是否需要加载
        if(map_config.is_valid())
        {
            map_config->update(Vector2(camera_pos.x, camera_pos.z), load_range);
        }
        
    }
    void FoliageEngine::update_foliage_asset_load(const Vector3& camera_pos)
    {
        int map_index_x = camera_pos.x / (FoliageGlobals::CELL_SIZE * FoliageGlobals::PAGE_SIZE * map_page_count);
        int map_index_z = camera_pos.z / (FoliageGlobals::CELL_SIZE * FoliageGlobals::PAGE_SIZE * map_page_count);

        // 检测卸载
        auto map_it = load_map_index.begin();
        while(map_it)
        {
            if(Math::abs(map_it->x - map_index_x) > 1 || Math::abs(map_it->z - map_index_z) > 1)
            {
                upload_map(*map_it);
                map_it = load_map_index.erase(map_it);

                continue;
            }

            ++map_it;
        }
        // 检测是否需要加载

    }
    void FoliageEngine::clear()
    {
        FoliageRenderManager::get_instance()->clear();
        for(auto& it : prototypes)
        {
            it.value->clear();
        }
        prototypes.clear();
        load_cell_datas.clear();
        total_cell_datas.clear();
        load_map_index.clear();
        is_init = false;
    }
}