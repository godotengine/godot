#include "foliage_proto_type.h"

namespace Foliage
{
	void FoliagePrototypeAsset::load_imp(Ref<FileAccess> & file,uint32_t version,bool is_big_endian)
    {
        uint32_t foliage_count = file->get_32();
        prototypes.resize(foliage_count);
        // 加載原型信息
        for(uint32_t i = 0; i < foliage_count;i++)
        {
            prototypes.write[i].load(file,is_big_endian);
        }

        uint32_t cell_count = file->get_32();
        String file_name;
        FoliageCellPos pos;
        for(uint32_t i = 0; i < cell_count;i++)
        {
            pos.x = file->get_32();
            pos.z = file->get_32();
            FoliageCellAsset* cell_asset = memnew(FoliageCellAsset);
            cell_asset->set_region_offset(pos.x,pos.z);
            file_name = file->get_as_utf8_string();
            cell_asset->load_file(file_name);
            cellAssetConfig.insert(pos.DecodeInt(),cell_asset);
        }
    }
    void FoliagePrototypeAsset::unload_imp()
    {
        for(auto it : cellAssetConfig)
        {
            it.value->clear();
        }
        cellAssetConfig.clear();    


        prototypes.clear();
    }
}