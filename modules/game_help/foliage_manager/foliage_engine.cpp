#include "foliage_engine.h"
#include "foliage_render_manager.h"

namespace Foliage
{
    void FoliageEngine::init(TypedArray<FoliageMapChunkConfig> map_config)
    {
        clear();
        for (int i = 0; i < map_config.size(); i++)
        {
            Ref<FoliageMapChunkConfig> config = map_config[i];
            // 加載每个地块
            FoliagePrototypeAsset* proto_type = memnew(FoliagePrototypeAsset);
            proto_type->set_file_name(config->foliage_asset_file_name);
            FoliageCellPos map_pos;
            map_pos.x = config->page_offset.x / FoliageGlobals::PAGE_SIZE;
            map_pos.z = config->page_offset.y / FoliageGlobals::PAGE_SIZE;
            prototypes.insert(map_pos.DecodeInt(),proto_type);
            FoliageCellPos page_pos;
            // 下面添加所有的页面
            for(int i = 0; i < config->pages_index.size(); ++i)
            {
                page_pos.x = config->pages_index[i].x;
                page_pos.z = config->pages_index[i].y;
                page_pos.Offset(Vector2i(config->page_offset.x,config->page_offset.y));
                // 初始化頁面信息
                FoliageCellAsset* cell = memnew(FoliageCellAsset);
                cell->set_file_name(config->pages_files[i]);
                cells.insert(page_pos.DecodeInt(),cell);
            }


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
            it.value->unload();
        }
        prototypes.clear();

        for(auto & it : cells)
        {
            it.value->clear();
        }
        cells.clear();
        load_cell_datas.clear();
        total_cell_datas.clear();
        load_map_index.clear();
        is_init = false;
    }
}