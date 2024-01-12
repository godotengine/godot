#include "foliage_cell_asset.h"
#include "foliage_manager.h"

namespace Foliage
{
    void FoliageCellAsset::_bind_methods()
    {
        ClassDB::bind_method(D_METHOD("load_file", "path"), &FoliageCellAsset::load_file);
        ClassDB::bind_method(D_METHOD("clear"), &FoliageCellAsset::clear);
    }

    void FoliageCellAsset::job_load_func(void* data,uint32_t index)
    {
        FoliageCellAsset::FileLoadData* d = (FoliageCellAsset::FileLoadData*)data;
        Ref<FoliageCellAsset> obj = d->dest;
        obj->load(d->file);
        d->file.unref();
        d->dest.unref();
    }
    void FoliageCellAsset::load_file(String _file)
    {
        configFile = _file;
        load_data.dest.reference_ptr(this);
        load_data.file = FileAccess::open(configFile, FileAccess::READ);
        handle_load = WorkerTaskPool::get_singleton()->add_native_group_task(&job_load_func,&load_data, 1,1,nullptr);
        load(load_data.file);
    }
    void FoliageCellAsset::load(Ref<FileAccess> & file)
    {
        clear();
        int64_t pos = file->get_position();
        file->get_buffer((uint8_t*)&big_endian, 1);
        file->seek(pos);
        // 如果是windows的小端模式，读取掩码的第一个字节肯定是一个有效数字
        file->set_big_endian(big_endian == 0);
        version = file->get_32();

        region_offset.x = file->get_32();
        region_offset.y = file->get_32();

        x = file->get_32();
        z = file->get_32();

        int32_t count = file->get_32();
        datas.resize(count);
        for(int i = 0;i < count;i++)
        {
            datas.write[i].load(file,big_endian);
        }
    }

    bool FoliageCellAsset::is_load_finish()
    {
        if(handle_load.is_valid())
        {
            return handle_load->is_completed();
        }
        return true;

    }
    void FoliageCellAsset::update_load(class FoliageManager * manager)
    {
        if(is_attach_to_manager)
        {
            return;
        }
        if(is_load_finish())
        {
            for(int i = 0;i < datas.size();i++)
            {
                FoliageCellPos pos = datas[i].position;
                pos.Offset(region_offset);
                manager->on_cell_load(pos,&datas[i]);
            }
            is_attach_to_manager = true;
            handle_load.unref();
        }
    }

    void FoliageCellAsset::wait_load_finish()
    {
        if(handle_load.is_valid())
        {
            handle_load->wait_completion();
            handle_load.unref();
        }
    }
    /// <summary>
    /// 清除数据
    /// </summary>
    void FoliageCellAsset::clear()
    {
        datas.clear();
    }










}