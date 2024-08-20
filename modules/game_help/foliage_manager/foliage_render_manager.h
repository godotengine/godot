#ifndef FOLIAGE_RENDER_MANAGER_H
#define FOLIAGE_RENDER_MANAGER_H

#include "foliage_cell_asset.h"
#include "foliage_proto_type.h"
#include "native_list.h"
#include "foliage_render_buffer.h"
#include "foliage_render.h"
#include "memory_pool.h"
#include "foliage_cell_layer.h"

namespace Foliage
{

    // 植被渲染管理器
    class FoliageRenderManager
    {

    public:
        FoliageRenderManager();
        ~FoliageRenderManager();
        static FoliageRenderManager* get_instance()
        {
            static FoliageRenderManager instance;
            return &instance;
        }

        void pre_update()
        {
            wait_cull_task();

            
        }
        void update(Vector3 camera_pos)
        {
            currCameraPos = camera_pos;
            // 处理格子的加载卸载信息
            procell_gpu_cell_load_unload();
            UpdateRenderBlockBuffer();

        }
        void post_update()
        {

            process_load_cell_job();
        }

        // 格子的操作相關函數
    public:
        // 增加一个格子信息
        void add_cell(int layer_id_index,Vector3& map_offset_pos, FoliageCellAsset::CellData * _cell)
        {
            wait_cull_task();
            cellLayer.write[layer_id_index].add_cell(map_offset_pos,_cell,cellBoxes,prototypes);
            renderBlockUpdate = true;
        }
        // 移除一个格子信息
        void remove_cell(int layer_id,FoliageCellPos& pos)
        {
            wait_cull_task();
            auto cell = cellLayer.write[layer_id].get_cell(pos);
            auto cell_box_point = cellLayer.write[layer_id].get_cell_box_point(pos);
            if(cell == nullptr || cell_box_point == nullptr)
            {
                return;
            }
            auto box_ptr = cellBoxes.get_buffer(cell_box_point);
            for(int i = 0; i < cellBoxes.size(); ++i)
            {
                auto ptr = box_ptr + i;
                // 標記已經卸載了
                ptr->isRemove = true;
                unload_once_cell_box(ptr);
            }
            // 卸載格子信息
            cellLayer.write[layer_id].remove_cell(pos,cellBoxes);
            renderBlockUpdate = true;

        }



        // 原型操作相關函數
    public:
        void add_protype(FoliagePrototype * _prototype)
        {
            auto it = prototypes.find(_prototype->guid);
            if(it == prototypes.end())
            {
                ++it->value.refCount;
                return;
            }
            _prototype->protypeId = get_protypeid(_prototype->guid);
            _prototype->refCount = 0;
            prototypes[_prototype->guid] = *_prototype;
            renderArgBufferUpdate = true;
        }
        void remove_prototype(String _guid)
        {
           auto it = guidToRender.find(_guid);
           if(it == guidToRender.end())
           {
               return;
           }
           --it->value->prototype->refCount;
           // 還在被使用，直接返回
           if(it->value->prototype->refCount > 0)
           {
                return;
           }
           it->value->prototype = nullptr;
           removeRenderIndex.push_back(it->value->renderIndex);
           guidToRender.erase(it->key);
           renderArgBufferUpdate = true;
        }
    public:
        // 更新原型加载状态
        void update_protype_load_state()
        {
            if(!isUpdateLoad)
            {
                return;
            }
            isUpdateLoad = false;
            reset_protype_use_state();
            for(auto& layer : cellLayer)
            {
                for(auto & it : layer.cell_datas)
                {
                    for(int i = 0; i <  it.value.prototypes.size(); ++i)
                    {
                        init_render(it.value.prototypes[i].guid);
                    }

                }
            }
            // 移除沒有使用的Render
            remove_unuse_render();
        }
        int get_protypeid(String _guid)
        {
            auto it = prototypesIndexID.find(_guid);
            if(it != prototypesIndexID.end())
            {
                return it->value;
            }
            int id = prototypesIndexID.size();
            prototypesIndexID[_guid] = id;
            return id;
        }
        void reset_protype_use_state()
        {
            for(int i = 0; i < foliageRenderList.size(); ++i)
            {
                if(foliageRenderList[i].prototype)
                {
                    foliageRenderList[i].prototype->reset_use();
                }   
            }
        }
        // 初始化渲染信息
        bool init_render(String _guid)
        {
            auto it = guidToRender.find(_guid);
            if(it != guidToRender.end())
            {
                // 設置是否使用
                it->value->prototype->set_use();
                return false;
            }
            auto pit = prototypes.find(_guid);
            if(pit == prototypes.end())
            {
                // 沒有找到原型，直接返回
                return false;
            }
            FiliageRender * render = nullptr;
            if(removeRenderIndex.size()>0)
            {
                render = &foliageRenderList.write[removeRenderIndex[removeRenderIndex.size()-1]];
                removeRenderIndex.resize(removeRenderIndex.size()-1);
            }
            else
            {
                foliageRenderList.push_back(FiliageRender());
                render = &foliageRenderList.write[foliageRenderList.size()-1];
                // 設置渲染索引
                render->renderIndex = foliageRenderList.size() - 1;
            }
            render->renderIndex = foliageRenderList.size();
            render->prototype = &prototypes[_guid];
            render->prototype->set_use();
            guidToRender[_guid] = render;
            return true;
        }
        
        
        void remove_unuse_render()
        {
            for(int i = 0; i < foliageRenderList.size(); ++i)
            {
                if(foliageRenderList[i].prototype)
                {
                    if(!foliageRenderList.write[i].prototype->is_use())
                    {
                        removeRenderIndex.push_back(i);
                        foliageRenderList.write[i].prototype = nullptr;
                    }
                }
            }
        }

    private:
        // 更新每个渲染器的compute buffer
        void _update_render_compute_buffer()
        {
			uint32_t argoffset = 0;
            auto buf_ptr = foliageRenderBuffer.instanceBufferPerRenderer.get_ptr();
            foliageRenderBuffer.drawArgBuffer.clear();
            DrawArgData _argsPerMesh;
            for(int _rIter = 0; _rIter < foliageRenderList.size(); ++_rIter)
            {

                if(foliageRenderList[_rIter].prototype)
                {
                    auto& _renderer = foliageRenderList.write[_rIter];
                    auto protype = foliageRenderList[_rIter].prototype;
                    buf_ptr[_rIter].drawArgOffset = argoffset;
                    buf_ptr[_rIter].lodAndsubMesh = 0;
                    buf_ptr[_rIter].meshType = (uint32_t)2;
                    if (_renderer.heightType > 0)
                    {
                        //m_renderInfo.data[_rIter].meshType |=
                        //	_renderer.prototype.runtimeData.castShadow ? (uint)(1 << 5 | 1 << 6) : 0;
                    }
                    else
                    {
                        buf_ptr[_rIter].meshType |= _renderer.gpuResource.castShadow ? (uint32_t)(1 << 5 | 1 << 6 | 1 << 7 | 1 << 8) : 0;
                        int shadowLod = _renderer.prototype->pcShadowRenderLevel;
                        for (int i = 0; i < shadowLod; ++i)
                        {
                            buf_ptr[_rIter].meshType |= (uint32_t)(1 << (i + 5));
                        }
                        buf_ptr[_rIter].lodDistance = _renderer.prototype->lodEndDistance();
                        buf_ptr[_rIter].lodEnable = Vector4(_renderer.lodEnabled.x ? 1 : 0,
                            _renderer.lodEnabled.y ? 1 : 0,
                            _renderer.lodEnabled.z ? 1 : 0,
                            _renderer.lodEnabled.w ? 1 : 0);
                    }
                    uint32_t lodCount = 0;
                    uint32_t subMask = 0;
                    uint32_t subargOffset = 0;

                    for (int _lod = 0; _lod < s_maxMeshLodLv; _lod++)
                    {
                        if (!IsSupperRender(_renderer, _lod))
                        {
                            continue;
                        }

                        lodCount++;
                        LODRenderInfo& _lodRenderInfo = _renderer.gpuResource.lodRenderInfos.write[_lod];
                        uint32_t _numSubset = (uint32_t)_lodRenderInfo.meshRenderInfos.size();

                        subMask |= (uint32_t)_numSubset << (3 * _lod + 3);
                        buf_ptr[_rIter].subMeshArg |= subargOffset << (_lod * 4);
                        subargOffset += _numSubset;
                        argoffset += _numSubset;
                        for (uint32_t _subset = 0; _subset < _numSubset; _subset++)
                        {
                            MeshRenderInfo& _renderInfo =  _lodRenderInfo.meshRenderInfos.write[_subset];
                            _renderInfo.drawArgsBufOffset = foliageRenderBuffer.drawArgBuffer.buffer_size();
                            Ref<Mesh> _mesh = _renderInfo.mesh;
                           // _argsPerMesh.index_cout = _mesh.surface_get_array_index_len(_renderInfo.meshSubset);
                            _argsPerMesh.index_start = 0;
                            _argsPerMesh.vertex_offset = 0;
                            _argsPerMesh.first_instance = 0;
                            foliageRenderBuffer.drawArgBuffer.add_data(_argsPerMesh);
                        }

				    }

				    buf_ptr[_rIter].lodAndsubMesh = lodCount | subMask;
                    buf_ptr[_rIter].visbleIDOffset = foliageRenderBuffer.instanceCount;
                    buf_ptr[_rIter].numInstances = _renderer.gpuResource.numInstances;
                    foliageRenderBuffer.instanceCount += _renderer.gpuResource.numInstances;
                    renderArgBufferUpdate = true;
                }


            }
        }
        bool IsSupperRender(FiliageRender& _render,int lod)
        {
            return true;
        }
        void wait_cull_task()
        {
            if(cellCullJobHandle.is_valid())
            {
                cellCullJobHandle->wait_completion();
                cellCullJobHandle = Ref<TaskJobHandle>();
            }
            
            // 用于裁剪結果需要加載的格子列表
            cull_load_list.clear();
            // 用于裁剪結果需要卸載的格子
            cull_unload_list.clear();
        }
        void UpdateRenderBlockBuffer()
        {
            if(renderArgBufferUpdate)
            {
                renderArgBufferUpdate = false;
                foliageRenderBuffer.drawArgBuffer.update_buffer();
            }
            if(renderBlockUpdate)
            {
                renderBlockUpdate = false;
                foliageRenderBuffer.blockGPUData.update_buffer(renderBlock);
            }
        }
        void Cull()
        {
            UpdateRenderBlockBuffer();
        }
    public:
        void procell_gpu_cell_load_unload()
        {
            wait_cull_task();

            // 處理卸載
            for(int i = 0; i < cull_unload_list.Length(); ++i)
            {
                PendingProtoBox** _cellBox = cull_unload_list.GetUnsafePtr(i);
                unload_once_cell_box(*_cellBox);
            }
            // 處理加載
            for(int i = 0; i < cull_load_list.Length(); ++i)
            {
                PendingProtoBox** _cellBox = cull_load_list.GetUnsafePtr(i);
                load_once_cell_box(*_cellBox);
            }
            cull_unload_list.clear();
            cull_load_list.clear();
        }
        void load_once_cell_box(PendingProtoBox* _cellBox)
        {
            _cellBox->loadTag = FoliageGlobals::LOAD_TAG_LOAD;
            int key = GetSpawnKey(_cellBox->protoID, _cellBox->cellWorldPos);
            auto cell = cellLayer.write[_cellBox->layerId].get_cell(_cellBox->cellWorldPos);
            if(!renderBlock.has(key) && cell != nullptr)
            {
                CellBlockItem* item = CellBlockItem::Allocal();
                item->block = foliageRenderBuffer.instanceRenderBuffer.allocal(_cellBox->instanceCount,foliageRenderBuffer.instanceBufferChangeINfo);
                item->protoTypeID = _cellBox->protoID;

                item->data.start = item->block->Start();
                item->data.size = item->block->size;
                // 增加到加載列表
                foliageRenderBuffer.cellInstanceLoadBuffer.AddLoadInfo(*cell,item->block,_cellBox->cellWorldPos.worldPosition()
                ,_cellBox->box ,Vector2i(_cellBox->instanceStart,_cellBox->instanceStart + _cellBox->instanceCount));

                renderBlock.insert(key, item);
                renderBlockUpdate = true;
            }

        }
        void unload_once_cell_box(PendingProtoBox* _cellBox)
        {
            _cellBox->loadTag = FoliageGlobals::LOAD_TAG_NONE;
            int key = GetSpawnKey(_cellBox->protoID, _cellBox->cellWorldPos);
            auto rs = renderBlock.find(key);
            if(rs != renderBlock.end())
            {
                if(rs->value->block != nullptr)
                {
                    foliageRenderBuffer.instanceRenderBuffer.free_buffer(rs->value->block);
                    rs->value->block = nullptr;
                    renderBlockUpdate = true;
                }   
                renderBlock.remove(rs);
            }

        }
        static void job_cull_func(void* data,uint32_t index)
        {
            FoliageRenderManager* _this = (FoliageRenderManager*)data;
            _this->thread_procell_cull(index);
        }
        void process_load_cell_job()
        {
            
            wait_cull_task();
            cull_load_list.clear();
            cull_unload_list.clear();
            if(cellDataCullingData)
            {
                cellCullJobHandle = Ref<TaskJobHandle>();
                cellDataCullingData = false;
                if(cellBoxes.size() > 0)
                {
                    cellCullJobHandle = WorkerTaskPool::get_singleton()->add_native_group_task(&job_cull_func,this, 1,1,nullptr);
                }
            }
        }
        void thread_procell_cull(int index)
        {
            PendingProtoBox* _cellBox = cellBoxes.GetPtr(index);
            if (_cellBox->isRemove == 1) return;
            if (renderQualitySetting.GetUnsafePtr(_cellBox->protoID) )
            {
                // 品质设置关闭
                if(_cellBox->loadTag == 1)
                {
                    cull_unload_list.Thread_Add(_cellBox);

                }
                return;
            }
            Vector3 _closestPt = _cellBox->box.get_closest_point(currCameraPos);
            _closestPt.y = currCameraPos.y;
            float _distanceToCameraSq = _closestPt.distance_squared_to( currCameraPos);
            if ((_cellBox->loadTag == FoliageGlobals::LOAD_TAG_PRE_UNLOAD || _cellBox->loadTag == FoliageGlobals::LOAD_TAG_NONE) 
            && _distanceToCameraSq < _cellBox->distance)
            {
                cull_load_list.Thread_Add(_cellBox);
            }
            else if (_cellBox->loadTag == 1 && _distanceToCameraSq > _cellBox->distance + 8)
            {
                cull_unload_list.Thread_Add(_cellBox);
            }

        }
    public:
		static int GetSpawnKey(int prototypeID,  FoliageCellPos& cellWorldPos)
		{
			return prototypeID << 22 | cellWorldPos.x << 11 | cellWorldPos.z;
		}
        void clear()
        {

            wait_cull_task();
            for(int i = 0; i < cellLayer.size(); ++i)
            {
                cellLayer.write[i].clear(cellBoxes);
            }
            auto box_ptr = cellBoxes.list.GetUnsafePtr();
            int length = cellBoxes.list.Length();
            for(int i = 0; i < length; ++i)
            {
                unload_once_cell_box(box_ptr + i);
            }
            cellBoxes.clear();
            prototypesIndexID.clear();
            prototypes.clear();
            for(int i = 0; i < foliageRenderList.size(); ++i)
            {
                foliageRenderList.write[i].clear();
            }
            foliageRenderList.clear();
            guidToRender.clear();
            for(auto& it : renderBlock )
            {
                foliageRenderBuffer.instanceRenderBuffer.free_buffer(it.value->block);

                CellBlockItem::Release(it.value);
            }
            renderBlock.clear();
            for(int i = 0; i < foliageRenderBuffer.instanceBufferChangeINfo.size(); ++i)
            {
                RD::get_singleton()->free(foliageRenderBuffer.instanceBufferChangeINfo[i].preGpuTreeInstance);
                RD::get_singleton()->free(foliageRenderBuffer.instanceBufferChangeINfo[i].preGgpuMatrix);
            }
            foliageRenderBuffer.instanceBufferChangeINfo.clear();
            foliageRenderBuffer.drawArgBuffer.clear();
            renderBlockUpdate = true;
        }
        static const int s_maxMeshLodLv = 4;
        // 植被渲染信息
        struct FoliageRenderBuffer {
            int instanceCount = 0;
            SRenderInstanceInfoBuffer instanceBufferPerRenderer;
            DrawArgDataBuffer drawArgBuffer;
            // 缓冲改变信息
            Vector<FoliageInstanceRenderDataChangeInfo> instanceBufferChangeINfo;
            // 渲染用的每个实例的缓冲
            FoliageInstanceRenderData instanceRenderBuffer;

            // 格子实例的加载信息
            FoliageCellGpuInstanceLoadBuffer cellInstanceLoadBuffer;

            BlockGPUData blockGPUData;
            void clear()
            {
                for(int i = 0; i < instanceBufferChangeINfo.size(); ++i)
                {
                    RD::get_singleton()->free(instanceBufferChangeINfo[i].preGpuTreeInstance);
                    RD::get_singleton()->free(instanceBufferChangeINfo[i].preGgpuMatrix);
                }
                instanceBufferChangeINfo.clear();
            }
        };
        FoliageRenderBuffer foliageRenderBuffer;
        // 各自的层信息
        Vector<FoliageCellLayer> cellLayer;
        HashMap<int,int> cellLayerIDMap;

        // 记载的原型信息
        HashMap<String,FoliagePrototype> prototypes;
        // 原型對應的索引ID
        HashMap<String,int> prototypesIndexID;
        // 加载的原型渲染信息
        Vector<FiliageRender>   foliageRenderList;
        // 原型GUID 映射的渲染信息
        HashMap<String,FiliageRender*> guidToRender;
        // 当前加载的快渲染格子信息
        HashMap<int, CellBlockItem*> renderBlock;
        // 移除的渲染器信息
        Vector<int> removeRenderIndex;

        // 用于裁剪結果需要加載的格子列表
        NativeList<PendingProtoBox*> cull_load_list;
        // 用于裁剪結果需要卸載的格子
        NativeList<PendingProtoBox*> cull_unload_list;
        // 裁剪剔除的句柄
        Ref<TaskJobHandle> cellCullJobHandle;
        // 當前攝像機的位置
        Vector3 currCameraPos;
        // 品質設定
        NativeList<bool> renderQualitySetting;
        // 當前加載的Cell 的Box信息
        MemoryPoolData<PendingProtoBox> cellBoxes;
        bool cellDataCullingData = false;
        bool renderBlockUpdate= false;
        bool renderArgBufferUpdate = false;
        
        bool isUpdateLoad = false;
    };
}



#endif
