#ifndef FOLIAGE_MANAGER_H
#define FOLIAGE_MANAGER_H

#include "foliage_cell_asset.h"
#include "foliage_proto_type.h"

#include "scene/3d/camera_3d.h"
#include "scene/3d/node_3d.h"

namespace Foliage
{
    class FoliageMapChunkConfig : public Resource
    {
        GDCLASS(FoliageMapChunkConfig, Resource);
        static void _bind_methods()
        {
            ClassDB ::bind_method(D_METHOD("set_chunk_asset_name", "map_name"), &FoliageMapChunkConfig::set_chunk_asset_name);
            ClassDB ::bind_method(D_METHOD("get_chunk_asset_name"), &FoliageMapChunkConfig::get_chunk_asset_name);
            ClassDB ::bind_method(D_METHOD("set_page_offset", "offset"), &FoliageMapChunkConfig::set_page_offset);
            ClassDB ::bind_method(D_METHOD("get_page_offset"), &FoliageMapChunkConfig::get_page_offset);
            ClassDB ::bind_method(D_METHOD("set_page_index", "index"), &FoliageMapChunkConfig::set_page_index);
            ClassDB ::bind_method(D_METHOD("get_page_index"), &FoliageMapChunkConfig::get_page_index);
            ClassDB ::bind_method(D_METHOD("set_page_files", "files"), &FoliageMapChunkConfig::set_page_files);
            ClassDB ::bind_method(D_METHOD("get_page_files"), &FoliageMapChunkConfig::get_page_files);

            ADD_PROPERTY(PropertyInfo(Variant::STRING, "chunk_asset_name", PROPERTY_HINT_NONE), "set_chunk_asset_name", "get_chunk_asset_name");
            ADD_PROPERTY(PropertyInfo(Variant::VECTOR2, "page_offset"), "set_page_offset", "get_page_offset");
            ADD_PROPERTY(PropertyInfo(Variant::PACKED_VECTOR2_ARRAY, "page_index"), "set_page_index", "get_page_index");
            ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "page_files"), "set_page_files", "get_page_files");
        }
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
        void set_page_index(Vector<Vector2> _index)
        {
            pages_index = _index;
        }
        Vector<Vector2> get_page_index()
        {
            return pages_index;
        }
        void set_page_files(Vector<String> _files)
        {
            pages_files = _files;
        }
        Vector<String> get_page_files()
        {
            return pages_files;
        }
        String foliage_asset_file_name;
        Vector2 page_offset;
        Vector<Vector2> pages_index;
        Vector<String> pages_files;

    };
    class FoliageManager : public Node3D
    {
        GDCLASS(FoliageManager, Node3D);
        static void _bind_methods()
        {

        }
    public:
        FoliageManager();
        ~FoliageManager();
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
        void _notification(int p_what);
        void update()
        {
            update(mainCamera->get_global_transform().origin);
        }
        void update(const Vector3& camera_pos);
    private:
        void update_foliage_asset_load(const Vector3& camera_pos);
        void upload_map(const FoliageCellPos& _pos);
    private:
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
        
        int map_page_count = 16;
        bool is_init = false;
    };
}

#endif