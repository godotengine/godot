#ifndef FOLIAGE_RENDER_BUFFER_H
#define FOLIAGE_RENDER_BUFFER_H
#include "servers/rendering/rendering_device.h"
#include "foliage_cell_asset.h"
#include "memory_pool.h"
#include "native_list.h"


namespace Foliage
{
    // 渲染的arg buffer
    struct DrawArgData
    {
        uint32_t index_count = 0;
        uint32_t instance_count = 0;
        uint32_t index_start = 0;
        uint32_t vertex_offset = 0;
        uint32_t first_instance = 0;
    };
    struct DrawArgDataBuffer
    {
        String name;
        // 渲染缓冲区
        Vector<uint8_t> buffer;
        RID bufferID;
        uint32_t renderBufferSize = 0;

        DrawArgData* get_ptr(uint32_t index = 0)
        {
            if(buffer.size() == 0 || index >= buffer.size()/sizeof(DrawArgData))
            {
                return nullptr;
            }
            return ((DrawArgData*)(buffer.ptrw())) + index;
        }
        uint32_t buffer_size()
        {
            return buffer.size();
        }
        void clear()
        {
            buffer.clear();
        }
        void release()
        {
            if(bufferID.is_valid())
            {
                RD::get_singleton()->free(bufferID);
                bufferID = RID();
            }
        }
        void add_data(DrawArgData & data)
        {
            int index = buffer.size() / sizeof(DrawArgData);
            buffer.resize(buffer.size() + sizeof(DrawArgData));
            memcpy(buffer.ptrw() + index, &data, sizeof(DrawArgData));
            renderBufferSize = buffer.size();
        }
        void update_buffer()
        {
            if(buffer.size() > (int)renderBufferSize || bufferID.is_null())
            {
                if(bufferID.is_valid())
                {
                    RD::get_singleton()->free(bufferID);
                    bufferID = RID();
                }
                if(buffer.size() > 0)
                {
                    bufferID = RD::get_singleton()->storage_buffer_create(buffer.size(), buffer,RD::STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT);
                }
            }
            if(buffer.size() <= 0)
            {
                return;
            }
            RD::get_singleton()->buffer_update(bufferID, 0, buffer.size(), buffer.ptr());
        } 

    };

    struct SRenderInstanceInfo
    {
		Vector4 lodDistance;
		Vector4 lodEnable;
		uint32_t lodAndsubMesh;
		uint32_t drawArgOffset;
		uint32_t subMeshArg;
		uint32_t meshType;
        uint32_t visbleIDOffset;
        uint32_t numInstances;
        uint32_t pad,pad1;
    };

    struct SRenderInstanceInfoBuffer
    {   
        String name;
        // 渲染缓冲区
        Vector<uint8_t> buffer;
        RID bufferID;

        SRenderInstanceInfo* get_ptr(uint32_t index = 0)
        {
            if(buffer.size() == 0 || index >= buffer.size()/sizeof(SRenderInstanceInfo))
            {
                return nullptr;
            }
            return reinterpret_cast<SRenderInstanceInfo*>(buffer.ptrw()) + index;
        }

        void resize(int size)
        {
            if(bufferID.is_valid())
            {
                RD::get_singleton()->free(bufferID);
                bufferID = RID();
            }
            buffer.resize(sizeof(SRenderInstanceInfo) * size);

            bufferID = RD::get_singleton()->storage_buffer_create(sizeof(SRenderInstanceInfo) * size, buffer);
            
//#ifdef DEV_ENABLED
            if(name.size() > 0)
            {
                RD::get_singleton()->set_resource_name(bufferID, name);
            }
            else
            {
	            RD::get_singleton()->set_resource_name(bufferID, "SRenderInstanceInfoBuffer:" + itos(bufferID.get_id()));
            }
//#endif
        }  
        void update_buffer()
        {
            if(bufferID.is_valid())
            {
                RD::get_singleton()->buffer_update(bufferID, 0, buffer.size(), buffer.ptr());
            }
        } 
        void clear()
        {

        }

    };

    /// 该结构体表示GPU中一棵树的实例，内存布局必须与Shader中同名结构体一�?
    /// </summary>
    struct GPUTreeInstance
    {
        /// <summary>
        /// 包围球，xyz为球心坐标，w为半�?
        /// </summary>
        Vector4 sphere;
        /// <summary>
        /// x--指向全局PrototypeLodInfo缓存的索�?
        /// y--是否支持渐变处理(0�?1，如LOD切换、树叶档住人物、出生、销�?)
        /// z--是否投影阴影(0�?1)
        /// w--是否是树(0�?1)
        /// </summary>
        Vector4i extraInfo;

        /// <summary>
        /// 灌木破碎信息
        /// x--是否砍碎(0�?1)
        /// y--砍碎时间
        /// zw--暂时无用
        /// </summary>
        Vector4 crackInfo;

        /// <summary>
        /// 树id
        /// </summary>
        //public float treeId => boxMin.w;

        /// <summary>
        /// xy -- guid
        /// z -- renderGroupID
        /// w -- renderLodID
        /// </summary>
        Vector4i extraInfo2;
    };
    struct FoliageInstanceRenderDataChangeInfo
    {
        
        RID preGpuTreeInstance;
        RID preGgpuMatrix;

        
        RID gpuTreeInstance;
        RID gpuMatrix;
        int oldCount = 0;
    };
    struct FoliageInstanceRenderData
    {
        RID gpuTreeInstance;
        RID gpuMatrix;
        int lastCount = 0;
        MemoryPool memoryPool;
        MemoryPool::Block* allocal(int count,Vector<FoliageInstanceRenderDataChangeInfo> & data_change_list)
        {
            if(lastCount == 0)
            {
                lastCount = 50000;
                gpuTreeInstance = RD::get_singleton()->storage_buffer_create(sizeof(GPUTreeInstance) * lastCount);
                gpuMatrix = RD::get_singleton()->storage_buffer_create(sizeof(float) * lastCount * 16);
            }
            auto block = memoryPool.Allocate(count,20000);
            if(block->End() > lastCount)
            {
                FoliageInstanceRenderDataChangeInfo info;
                info.preGpuTreeInstance = gpuTreeInstance;
                info.preGgpuMatrix = gpuMatrix;
                info.oldCount = lastCount;

                lastCount = block->End() + 20000;
                gpuTreeInstance = RD::get_singleton()->storage_buffer_create(sizeof(GPUTreeInstance) * lastCount);
                gpuMatrix = RD::get_singleton()->storage_buffer_create(sizeof(float) * lastCount * 16);
                info.gpuTreeInstance = gpuTreeInstance;
                info.gpuMatrix = gpuMatrix;
                data_change_list.push_back(info);
            }
            return block;

        }
        void free_buffer(MemoryPool::Block * point)
        {
            memoryPool.Free(point);
        }
    };
    struct CellLoadComputeArg
    {
        int _cellDataOffset;
        int _numInstances;
        int _instanceStart;
        Vector3 _mapOffset;
        Vector3 _cellOffset;
        Vector3 _boxMax;
        Vector3 _boxMin;
        int pad;
    };
    struct FoliageCellGpuInstanceLoadBuffer
    {
        
        Vector<uint8_t> tempLoadCellData;
        Vector<uint8_t> computeShaderArg;
        RID bufferID;
        RID argID;
        int instanceCount = 0;
        void clear_buffer()
        {
            tempLoadCellData.clear();
            computeShaderArg.clear();
        }
        void AddLoadInfo(FoliageCellAsset::CellData & info,MemoryPool::Block* dest_point
            , const Vector3& mapOffset, const AABB& boxOS, const Vector2i& instanceRange);
        void load_cell_gpu_instance(Vector<FoliageInstanceRenderDataChangeInfo>& instanceBufferChangeINfo)
        {
            // 處理緩衝區改變消息
            for(int i = 0; i < instanceBufferChangeINfo.size(); ++i)
            {
                // 拷貝緩衝區
            }
            // 執行Compute shader 加載盒子的实例数据
            
        }
        
    };

    struct BlockCPUData
    {
        int start;
        int size;
        int renderinfoID;
        int renderProtoID;
        Vector4 blockBoxMin;
        Vector4 blockBoxMax;            
    };
    struct CellBlockItem
    {
        MemoryPool::Block* block;
        BlockCPUData data;
        int protoTypeID;
        CellBlockItem* next;

        static CellBlockItem* unuseNodeRoot;
        static CellBlockItem* Allocal()
        {
            if (unuseNodeRoot != nullptr)
            {
                CellBlockItem* ret = unuseNodeRoot;
                unuseNodeRoot = unuseNodeRoot->next;
                return ret;
            }
            return memnew(CellBlockItem);
        }
        static void Release(CellBlockItem* node)
        {
            node->next = unuseNodeRoot;
            unuseNodeRoot = node;
        }
    };
    struct BlockGPUData
    {
        Vector<uint8_t> data;
        RID bufferID;
        uint32_t lastBufferCount = 0;
        void update_buffer(HashMap<int, CellBlockItem*>& renderBlock)
        {
            data.resize(data.size() + sizeof(BlockCPUData));
            int index = 0;
            for(auto& rb : renderBlock )
            {
                memcpy(data.ptrw() + (index * sizeof(BlockCPUData)), &rb.value->data, sizeof(BlockCPUData));
                ++index;
            }
            if(lastBufferCount == 0)
            {
                lastBufferCount = renderBlock.size() * sizeof(BlockCPUData);
                bufferID = RD::get_singleton()->storage_buffer_create(data.size());
            }
            else if(lastBufferCount < renderBlock.size())
            {
                RD::get_singleton()->free(bufferID);
                bufferID = RD::get_singleton()->storage_buffer_create(data.size());
            }
            RD::get_singleton()->buffer_update(bufferID, 0, data.size(), data.ptr());
        }
    };



    template<class T>
    class MemoryPoolData
    {
    public:
        MemoryPool pool;
        NativeList<T> list;
        MemoryPoolData()
        {

        }
        ~MemoryPoolData()
        {
        }
        MemoryPool::Block * allocal(int count)
        {
            MemoryPool::Block * r = pool.Allocate(count);
            list.AutoResize(r->End(),r->End() + 500);
            if(r->End() > list.size())
            {
                list.ResizeUninitialized(r->End(),5000);
            }
            return r;
        }
        T* GetPtr(int index)
        {
            return list.GetUnsafePtr(index);
        }
        void free(MemoryPool::Block* _block)
        {
            if(_block->End() == list.size())
            {
                list.ResizeUninitialized(_block->Start(),0);
            }
            pool.Free(_block);
        }
        T* get_buffer(MemoryPool::Block* _block)
        {
            return list.GetUnsafePtr(_block->Start());
        }
        int64_t size()
        {
            return list.Length();
        }
        void clear()
        {
            list.clear();
        }
    };
    

}

#endif
