#ifndef FOLIAGE_RENDER_MANAGER_H
#define FOLIAGE_RENDER_MANAGER_H

#include "foliage_cell_asset.h"
#include "foliage_proto_type.h"
#include "native_list.h"
namespace Foliage
{
    // 植被渲染管理器
    class FoliageRenderManager
    {

    public:
        FoliageRenderManager();
        ~FoliageRenderManager();

        // 原型渲染信息
        class FiliageRender
        {
            FiliageRender();
            ~FiliageRender();
            // 對應的原型信息
            FoliagePrototype * prototype = nullptr;
            int renderIndex = 0;

        };
        // 格子的操作相關函數
    public:
        // 增加一个格子信息
        void add_cell(int layer_id,FoliageCellPos& pos, FoliageCellAsset::CellData * _cell)
        {

            isUpdateLoad = true;
        }
        // 移除一个格子信息
        void remove_cell(int layer_id,FoliageCellPos& pos)
        {
            isUpdateLoad = true;

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
            isUpdateLoad = true;
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
           guidToRender.erase(it);
            isUpdateLoad = true;
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
            for(auto& it : cell_datas)
            {
                for(int i = 0; i <  it.value.prototypes.size(); ++i)
                {
                    init_render(it.value.prototypes[i].guid);
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
                it->value.set_use();
                return false;
            }
            auto pit = prototypes.find(_guid);
            if(pit == prototypes.end())
            {
                // 沒有找到原型，直接返回
                return;
            }
            FiliageRender * render = nullptr;
            if(removeRenderIndex.size()>0)
            {
                render = &foliageRenderList[removeRenderIndex[removeRenderIndex.size()-1]];
                removeRenderIndex.resize(removeRenderIndex.size()-1);
            }
            else
            {
                foliageRenderList.push_back(FiliageRender());
                render = *foliageRenderList.write[foliageRenderList.size()-1];
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
                    if(!foliageRenderList[i].prototype->is_use())
                    {
                        removeRenderIndex.push_back(i);
                        foliageRenderList[i].prototype = nullptr;
                    }
                }
            }
        }

    private:
        // 加载的格子層信息
        struct CellLayer
        {            
            // 
            int32_t layerIndex;
            HashMap<FoliageCellPos,FoliageCellAsset::CellData> cell_datas; // <cell_pos,cell>
            void add_cell(int layer_id,FoliageCellPos& pos, FoliageCellAsset::CellData * _cell)
            {

                auto it = cell_datas.find(pos.EncodeInt());
                if(it != cell_datas.end())
                {
                    return;
                }
                _cell->is_load = false;
                cell_datas[pos.EncodeInt()] = *_cell;
                isUpdateLoad = true;
            }
        };

		struct BlockCPUData
		{
			Vector4 blockBox;
			int start;
			int size;
			int renderProtoID;
            int pad;
		};
        struct DrawArgData
        {

        };
        struct FoliageRenderBuffer {
            // 每一個各自的实例缓冲区
            RID instanceBuffer;
            // 所有原型的渲染缓冲区
            RID protoBuffer;
            // 格子的每一个原型的信息
            RID blockBuffer;
            NativeList<>
            RID drawBuffer;
        };
        // 各自的层信息
        Vector<CellLayer> cellLayer;

        // 记载的原型信息
        HashMap<String,FoliagePrototype> prototypes;
        // 原型對應的索引ID
        HashMap<String,int> prototypesIndexID;
        // 加载的原型渲染信息
        Vector<FiliageRender>   foliageRenderList;
        // 原型GUID 映射的渲染信息
        HashMap<String,FiliageRender*> guidToRender;
        // 移除的渲染器信息
        Vector<int> removeRenderIndex;
        
        bool isUpdateLoad = false;
    };
}



#endif