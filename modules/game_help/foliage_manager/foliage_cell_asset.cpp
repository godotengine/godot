#include "foliage_cell_asset.h"
#include "foliage_engine.h"

namespace Foliage
{
    void FoliageCellAsset::_bind_methods()
    {
    }

    void FoliageCellAsset::load_imp(Ref<FileAccess> & file,uint32_t _version,bool is_big_endian)
    {

        region_offset.x = file->get_32();
        region_offset.y = file->get_32();

        x = file->get_32();
        z = file->get_32();

        int32_t count = file->get_32();
        datas.resize(count);
        for(int i = 0;i < count;i++)
        {
            datas.write[i].load(file,is_big_endian);
        }
    }

    /// <summary>
    /// 清除数据
    /// </summary>
    void FoliageCellAsset::unload_imp()
    {
        datas.clear();
    }










}