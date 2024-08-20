#include "foliage_render_buffer.h"

#include "foliage_cell_asset.h"


namespace Foliage
{
    CellBlockItem * CellBlockItem::unuseNodeRoot = nullptr;
    void FoliageCellGpuInstanceLoadBuffer::AddLoadInfo(FoliageCellAsset::CellData & info,MemoryPool::Block* dest_point
        ,const Vector3& mapOffset,const AABB& boxOS, const Vector2i& instanceRange)
    {
        int copy_index = tempLoadCellData.size();
        tempLoadCellData.resize(tempLoadCellData.size() + (info.instances.size() * sizeof(FoliageCellAsset::InstanceData)) );
        memcpy(tempLoadCellData.ptrw() + copy_index
        ,info.instances.ptrw() + instanceRange.x 
        ,(instanceRange.y - instanceRange.x) * sizeof(FoliageCellAsset::InstanceData));

        int arg_index = computeShaderArg.size();
        computeShaderArg.resize(computeShaderArg.size() + sizeof(CellLoadComputeArg));
        CellLoadComputeArg* loadInfo = ((CellLoadComputeArg*)computeShaderArg.ptrw()) - arg_index;
        loadInfo->_cellDataOffset = (tempLoadCellData.size() - copy_index) / sizeof(FoliageCellAsset::InstanceData);
        loadInfo->_numInstances = (instanceRange.y - instanceRange.x);
        loadInfo->_mapOffset = mapOffset;
        loadInfo->_instanceStart = dest_point->Start();
        loadInfo->_boxMax = boxOS.position;
        loadInfo->_boxMin = boxOS.get_end();        
        loadInfo->_cellOffset.x = info.position.x;
        loadInfo->_cellOffset.z = info.position.z;
        loadInfo->_cellOffset.y = FoliageGlobals::CELL_SIZE;

    }
}
