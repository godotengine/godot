#ifndef FOLIAGE_RESOURCE_H
#define FOLIAGE_RESOURCE_H
#include "core/math/vector2.h"
#include "core/math/vector3.h"
#include "core/math/aabb.h"
#include "core/math/vector2i.h"
#include "core/math/color.h"
#include "core/math/math_funcs.h"
#include "core/templates/vector.h"
#include "core/io/file_access.h"
#include "core/object/worker_thread_pool.h"

#include "memory_pool.h"


namespace Foliage
{
    class FoliageResource: public RefCounted
    {
        GDCLASS(FoliageResource,RefCounted)
        static void _bind_methods();
    public:
		// 加載文件
		void load_file(String _file);
        void load(Ref<FileAccess> & file);
        void clear()
        {
            wait_load_finish();
            unload_imp();
        }
		void wait_load_finish();
		bool is_load_finish();
    private:
    
    	static void job_load_func(void* data,uint32_t index);
    public:
        virtual void load_imp(Ref<FileAccess> & file,uint32_t version,bool is_big_endian){}
        virtual void unload_imp(){}
    private:
		struct FileLoadData
		{
			Ref<FoliageResource>	dest;
			Ref<FileAccess> file;
		};
    private:
		Ref<TaskJobHandle> handle_load;
        FileLoadData load_data;
        String configFile;
        bool is_load = false;

    };
}
#endif