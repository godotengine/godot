#include "foliage_manager.h"

namespace Foliage
{
    void FoliageManager::update(const Vector3& camera_pos)
    {

        update_foliage_asset_load(camera_pos);




    }
    void FoliageManager::update_foliage_asset_load(const Vector3& camera_pos)
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
    
    void FoliageManager::upload_map(FoliageCellPos& _pos)
    {
        int index = _pos.DecodeInt();
        auto it = prototypes.find(index);
        if(it == prototypes.end())
        {
            return;
        }
        // 卸载地图
        it->value->unload(this);    
        prototypes.erase(index);
    }
    FoliageManager::FoliageManager()
    {

    }
    FoliageManager::~FoliageManager()
    {
    }

}