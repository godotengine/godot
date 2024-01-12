#ifndef FOLIAGE_MANAGER_H
#define FOLIAGE_MANAGER_H

#include "foliage_cell_asset.h"
#include "foliage_proto_type.h"

namespace Foliage
{
    class FoliageManager
    {
    public:
        FoliageManager();
        ~FoliageManager();
        void on_cell_load(Vector2i& _pos, const FoliageCellAsset::CellData * _cell)
        {
            FoliageCellPos  pos = _cell->position;
            pos.Offset(_pos);
            int index = pos.DecodeInt();
            total_cell_datas[index] = (FoliageCellAsset::CellData *)_cell;
        }
        void update(const Vector3& camera_pos);
    private:
        void update_foliage_asset_load(const Vector3& camera_pos);
        void upload_map(FoliageCellPos& _pos);
    private:
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
    };
}

#endif