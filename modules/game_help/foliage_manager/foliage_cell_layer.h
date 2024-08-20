#ifndef FOLIAGE_CELL_LAYER_H
#define FOLIAGE_CELL_LAYER_H

#include "core/typedefs.h"

#include "foliage_cell_asset.h"
#include "foliage_proto_type.h"
#include "native_list.h"
#include "foliage_render_buffer.h"
#include "foliage_render.h"
#include "memory_pool.h"

namespace Foliage
{
    struct PendingProtoBox
    {
        MemoryPool::Block* instancePoint = nullptr;
        AABB box;
        int cellIndex;
        FoliageCellPos cellWorldPos;
        float distance;
        short protoID;
        uint16_t instanceStart = 0;
        uint16_t instanceCount = 0;
        // 2: Not loaded or unloaded, can be loaded.
        // 0: Preparing to unload.
        // 1: Loaded.
        uint8_t loadTag;
        uint8_t isRemove;
        uint8_t layerId;// 对应的层的ID
    };
    class FoliageCellLayer
    {
    public:
        FoliageCellLayer()
        {
        }
        ~FoliageCellLayer()
        {
        }
        HashMap<int,FoliageCellAsset::CellData> cell_datas; // <cell_pos,cell>
        // 当前加载的格子信息
        HashMap<int,MemoryPool::Block*> m_pendingDictLoadCells;
        // 
        int32_t layerIndex;
        bool is_remove = false;
        MemoryPool::Block* add_cell(Vector3& map_offset_pos, FoliageCellAsset::CellData * _cell
        ,MemoryPoolData<PendingProtoBox>& memoryPoolData,HashMap<String,FoliagePrototype>& prototypes)
        {
            FoliageCellPos pos = _cell->position;
            pos.Offset(map_offset_pos);
            auto it = cell_datas.find(pos.DecodeInt());
            if(it != cell_datas.end())
            {
                return nullptr;
            }

            int key = _cell->position.DecodeInt();
            FoliageCellPos worldpos = _cell->position;
            worldpos.Offset(map_offset_pos);
            _cell->is_load = false;
            _cell->block = memoryPoolData.allocal(_cell->prototypes.size());
            cell_datas[key] = *_cell;
            m_pendingDictLoadCells[key] = _cell->block;
            auto buf_ptr = memoryPoolData.get_buffer(_cell->block);
            for(int i = 0; i < _cell->prototypes.size(); ++i)
            {
                auto& protype = prototypes[_cell->prototypes[i].guid];
                // 计算原型的最大加载距离
                auto dis = protype.lodEndDistance();
                dis.y = protype.lod1Enabled ? dis.y : 0;
                dis.z = protype.lod2Enabled ? dis.z : 0;
                dis.w = protype.lod3Enabled ? dis.w : 0;
                float _maxLoadDistSq = MAX(dis.x, MAX(dis.y, MAX(dis.z, dis.w)));
                _maxLoadDistSq *= _maxLoadDistSq;

                auto& pd = _cell->prototypes[i];

                buf_ptr[i].protoID = protype.protypeId;
                auto box = protype.boxOS;
                box.position += worldpos.worldPosition();
                buf_ptr[i].box = protype.boxOS;
                buf_ptr[i].distance = _maxLoadDistSq;
                buf_ptr[i].loadTag = FoliageGlobals::LOAD_TAG_NONE;
                buf_ptr[i].layerId = layerIndex;
                buf_ptr[i].cellWorldPos = worldpos;
                buf_ptr[i].instanceStart = pd.instanceRange.x;
                buf_ptr[i].instanceCount = pd.instanceRange.y;
            }


            return _cell->block;
        }
        FoliageCellAsset::CellData * get_cell(FoliageCellPos& world_pos)
        {        
            auto it = cell_datas.find(world_pos.DecodeInt());
            if(it != cell_datas.end())
            {
                return &it->value;
            }
            return nullptr;
        }
        MemoryPool::Block* get_cell_box_point(FoliageCellPos& world_pos)
        {
            auto it = cell_datas.find(world_pos.DecodeInt());
            if(it != cell_datas.end())
            {
                return it->value.block;
            }
            return nullptr;
        }
        void remove_cell(FoliageCellPos & world_pos, MemoryPoolData<PendingProtoBox>& memoryPoolData)
        {
            auto it = cell_datas.find(world_pos.DecodeInt());
            if(it != cell_datas.end())
            {
                memoryPoolData.free(it->value.block);
                cell_datas.erase(it->key);
                m_pendingDictLoadCells.erase(world_pos.DecodeInt());
            }

        }
        void clear(MemoryPoolData<PendingProtoBox>& memoryPoolData)
        {
            for(auto & it : cell_datas)
            {
                memoryPoolData.free(it.value.block);
            }
            cell_datas.clear();
            m_pendingDictLoadCells.clear();
        }
       
    };
}

#endif
