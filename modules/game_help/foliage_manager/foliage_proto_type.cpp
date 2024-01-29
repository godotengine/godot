#include "foliage_proto_type.h"
#include "foliage_engine.h"

namespace Foliage
{
	void FoliagePrototypeAsset::load_imp(Ref<FileAccess> & file,uint32_t version,bool is_big_endian)
    {
        region.position.x = file->get_float();
        region.position.y = file->get_float();
        region.size.x = file->get_float();
        region.size.y = file->get_float();


        uint32_t foliage_count = file->get_32();
        prototypes.resize(foliage_count);
        // 加載原型信息
        for(uint32_t i = 0; i < foliage_count;i++)
        {
            prototypes.write[i].load(file,is_big_endian);
        }        

        uint32_t cell_count = file->get_32();
        String cell_file_name;
        FoliageCellPos pos;
        for(uint32_t i = 0; i < cell_count;i++)
        {
            pos.x = file->get_32();
            pos.z = file->get_32();
            FoliageCellAsset* cell_asset = memnew(FoliageCellAsset);
            cell_asset->set_region_offset(pos.x,pos.z);
            cell_file_name = file->get_as_utf8_string();
            cell_asset->load_file(cell_file_name);
            cellAssetConfig.insert(pos.DecodeInt(),cell_asset);
        }
    }
    void FoliagePrototypeAsset::unload_imp()
    {
        FoliageEngine::get_singleton().remove_prototype(this);
        for(auto it : cellAssetConfig)
        {
            it.value->clear();
        }
        cellAssetConfig.clear();    


        prototypes.clear();
    }
    void FoliagePrototypeAsset::tick_imp()
    {
        if(loadState == LoadFinish)
        {
            for(auto it : cellAssetConfig)
            {
                it.value->tick();
            }
        }
    }
    void FoliagePrototypeAsset::on_load_finish()
    {
        FoliageEngine::get_singleton().add_prototype(this);
    }
}