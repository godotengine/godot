#include "foliage_manager.h"
#include "foliage_render_manager.h"

namespace Foliage
{
    void FoliageManager::init(TypedArray<FoliageMapChunkConfig> map_config)
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
    
    void FoliageManager::upload_map(const FoliageCellPos& _pos)
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
    
    void FoliageManager::_notification(int p_what) 
    {
        switch (p_what) {
            case NOTIFICATION_READY: {
                //LOG(INFO, "NOTIFICATION_READY");
                //__ready();
                break;
            }

            case NOTIFICATION_PROCESS: {
                update();
                break;
            }

            case NOTIFICATION_PREDELETE: {
                //(INFO, "NOTIFICATION_PREDELETE");
                // _clear();
                break;
            }

            case NOTIFICATION_ENTER_TREE: {
                //LOG(INFO, "NOTIFICATION_ENTER_TREE");
                //_initialize();
                break;
            }

            case NOTIFICATION_EXIT_TREE: {
               // LOG(INFO, "NOTIFICATION_EXIT_TREE");
                //_clear();
                break;
            }

            case NOTIFICATION_ENTER_WORLD: {
             //   LOG(INFO, "NOTIFICATION_ENTER_WORLD");
                //_is_inside_world = true;
                //_update_instances();
                break;
            }

            case NOTIFICATION_TRANSFORM_CHANGED: {
                //LOG(INFO, "NOTIFICATION_TRANSFORM_CHANGED");
                break;
            }

            case NOTIFICATION_EXIT_WORLD: {
               // LOG(INFO, "NOTIFICATION_EXIT_WORLD");
                //_is_inside_world = false;
                break;
            }

            case NOTIFICATION_VISIBILITY_CHANGED: {
                //LOG(INFO, "NOTIFICATION_VISIBILITY_CHANGED");
                //_update_instances();
                break;
            }

            case NOTIFICATION_EDITOR_PRE_SAVE: {
                //LOG(INFO, "NOTIFICATION_EDITOR_PRE_SAVE");
                // if (!_storage.is_valid()) {
                //     LOG(DEBUG, "Save requested, but no valid storage. Skipping");
                // } else {
                //     _storage->save();
                // }
                // if (!_material.is_valid()) {
                //     LOG(DEBUG, "Save requested, but no valid material. Skipping");
                // } else {
                //     _material->save();
                // }
                // if (!_texture_list.is_valid()) {
                //     LOG(DEBUG, "Save requested, but no valid texture list. Skipping");
                // } else {
                //     _texture_list->save();
                // }
                break;
            }

            case NOTIFICATION_EDITOR_POST_SAVE: {
                //LOG(INFO, "NOTIFICATION_EDITOR_POST_SAVE");
                break;
            }
        }

    }
        
    void FoliageManager::clear()
    {
        FoliageRenderManager::get_instance()->clear();
        for(auto& it : prototypes)
        {
            it.value->unload(this);
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
    FoliageManager::FoliageManager()
    {

    }
    FoliageManager::~FoliageManager()
    {
    }

}