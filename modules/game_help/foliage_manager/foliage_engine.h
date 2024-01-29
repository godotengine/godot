#ifndef FOLIAGE_ENGINE_H
#define FOLIAGE_ENGINE_H

#include "foliage_cell_asset.h"
#include "foliage_proto_type.h"

#include "scene/3d/camera_3d.h"
#include "scene/3d/node_3d.h"
#include "core/math/geometry_2d.h"

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
        void load()
        {
            if(foliage_prototype_asset.is_null())
            {
                foliage_prototype_asset = Ref(memnew(FoliagePrototypeAsset));
            }
            foliage_prototype_asset->load_file(foliage_asset_file_name);
        }
        void unload()
        {
            if(foliage_prototype_asset.is_valid())
            {
                foliage_prototype_asset->clear();
            }
        }
        bool is_need_load(const Vector2& camera_circle,float radius)
        {            
            return Geometry2D::is_rect_intersecting_circle(chunk_area,camera_circle, radius);
        }
        void update(const Vector2& camera_circle,float radius)
        {
            if(is_need_load(camera_circle,radius))
            {
                foliage_prototype_asset->tick();
                load();
            }
            else
            {
                unload();
            }
        }
        String foliage_asset_file_name;
        Ref<FoliagePrototypeAsset> foliage_prototype_asset;
        Rect2 chunk_area;

    };
    class FoliageMapConfig : public Resource
    {
        public:
        GDCLASS(FoliageMapConfig, Resource);
        static void _bind_methods();
    public:
        void set_config(TypedArray<FoliageMapChunkConfig> _config)
        {
            map_config = _config;
        }
        TypedArray<FoliageMapChunkConfig> get_config()
        {
            return map_config;
        }
        void update(const Vector2& camera_circle,float radius)
        {
            for(int i = 0; i < map_config.size(); i++)
            {
                FoliageMapChunkConfig* config = (FoliageMapChunkConfig*)(Object*)map_config[i];
                config->update(camera_circle,radius);
            }
        }
        TypedArray<FoliageMapChunkConfig> map_config;
        Vector2 page_size;


    };
   

 // 单体植被管理器类
    class FoliageEngine
    {
        friend class FoliagePrototypeAsset;
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
        void init(Ref<FoliageMapConfig> _map_config);
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
        void add_prototype(FoliagePrototypeAsset* _proto);
        void remove_prototype(FoliagePrototypeAsset* _proto);
        void upload_map(const FoliageCellPos& _pos)
        {

        }
        String map_name = "foliage_map";
        Ref<FoliageMapConfig> map_config;
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
        float load_range = 4096;
        
        int map_page_count = 16;
        bool is_init = false;
    };
}

#endif