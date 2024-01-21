#include "foliage_resource.h"
namespace Foliage
{
    

    void FoliageResource::load_file(String _file)
    {
        configFile = _file;
        load_data.dest.reference_ptr(this);
        load_data.file = FileAccess::open(configFile, FileAccess::READ);
        handle_load = WorkerTaskPool::get_singleton()->add_native_group_task(&job_load_func,&load_data, 1,1,nullptr);
        load(load_data.file);
    }
    void FoliageResource::wait_load_finish()
    {
        if(handle_load.is_valid())
        {
            handle_load->wait_completion();
            handle_load.unref();
        }
    }
    bool FoliageResource::is_load_finish()
    {
        if(handle_load.is_valid())
        {
            return handle_load->is_completed();
        }
        return true;

    }
    void FoliageResource::job_load_func(void* data,uint32_t index)
    {
        FoliageResource::FileLoadData* d = (FoliageResource::FileLoadData*)data;
        Ref<FoliageResource> obj = d->dest;
        obj->load(d->file);
        d->file.unref();
        d->dest.unref();
    }
    void FoliageResource::load(Ref<FileAccess> & file)
    {
        clear();
        int64_t pos = file->get_position();
        uint32_t big_endian = 0;
        file->get_buffer((uint8_t*)&big_endian, 1);
        file->seek(pos);
        // 如果是windows的小端模式，读取掩码的第一个字节肯定是一个有效数字
        file->set_big_endian(big_endian == 0);
        uint32_t version = file->get_32();
        load_imp(file,version,big_endian);

    }
    
    void FoliageResource::_bind_methods()
    {
        ClassDB::bind_method(D_METHOD("load_file", "path"), &FoliageResource::load_file);
        ClassDB::bind_method(D_METHOD("clear"), &FoliageResource::clear);
        
    }
}