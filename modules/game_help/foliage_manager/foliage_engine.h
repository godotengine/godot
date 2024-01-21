#ifndef FOLIAGE_ENGINE_H
#define FOLIAGE_ENGINE_H

#include "foliage_cell_asset.h"
#include "foliage_proto_type.h"

#include "scene/3d/camera_3d.h"
#include "scene/3d/node_3d.h"

namespace Foliage
{

    class FoliageMapChunkConfig : public Resource
    {
        GDCLASS(FoliageMapChunkConfig, Resource);
        static void _bind_methods();
    public:
        void set_chunk_asset_name(String _map_name)
        {
            foliage_asset_file_name = _map_name;
        }
        String get_chunk_asset_name()
        {
            return foliage_asset_file_name;
        }
        void set_page_offset(Vector2 _offset)
        {
            page_offset = _offset;
        }
        Vector2 get_page_offset()
        {
            return page_offset;
        }
        String foliage_asset_file_name;
        Vector2 page_offset;

    };
    class FoliageMapConfig : public Resource
    {
        public:
        GDCLASS(FoliageMapConfig, Resource);
        static void _bind_methods();
        void set_config(TypedArray<FoliageMapChunkConfig> _config)
        {
            map_config = _config;
        }
        TypedArray<FoliageMapChunkConfig> get_config()
        {
            return map_config;
        }

        TypedArray<FoliageMapChunkConfig> map_config;
    };
   

 // 单体植被管理器类
    class FoliageEngine
    {
        public:
        static FoliageEngine & get_singleton()
        {
            static FoliageEngine engine;
            return engine;
        }
        void on_cell_load(Vector2i& _page_world_pos, const FoliageCellAsset::CellData * _cell)
        {
            FoliageCellPos  pos = _cell->position;
            pos.Offset(_page_world_pos);
            int index = pos.DecodeInt();
            total_cell_datas[index] = (FoliageCellAsset::CellData *)_cell;
        }
        void init(TypedArray<FoliageMapChunkConfig> map_config);
        void clear();
        void set_camera(Camera3D* p_camera)
        {
            mainCamera = p_camera;
        }
        void update()
        {
            update(mainCamera->get_global_transform().origin);
        }
        void update(const Vector3& camera_pos);
    private:
        void update_foliage_asset_load(const Vector3& camera_pos);
        void upload_map(const FoliageCellPos& _pos)
        {

        }
        String map_name = "foliage_map";
        // 設置主相機
        Camera3D* mainCamera;
        // 当前加载的格子列表
        HashMap<int,FoliageCellAsset::CellData*> load_cell_datas;
        // 全部的原型格子列表
        HashMap<int,FoliageCellAsset::CellData*> total_cell_datas;
        // 原型资源
        HashMap<int,Ref<FoliagePrototypeAsset>> prototypes;
        // 单元格资源
        HashMap<int,Ref<FoliageCellAsset>> cells;
        List<FoliageCellPos> load_map_index;
        class FoliageManager* curr_manager = nullptr;
        
        int map_page_count = 16;
        bool is_init = false;
    };
}

#endif