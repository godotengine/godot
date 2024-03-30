#include "path_manager.h"
#include "data_table_manager.h"
static PathManager* singleton = nullptr;

PathManager::PathManager()
{
    singleton = this;
}
PathManager::~PathManager()
{
    if(singleton == this)
    {
        singleton = nullptr;
    }
}
PathManager *PathManager::get_singleton()
{
    static PathManager singleton;
    return &singleton;
}
void PathManager::init()
{
    if(is_init)
        return;
    Ref<DataTableItem> table = DataTableManager::get_singleton()->get_data_table(DataTableManager::get_singleton()->get_path_table_name());
    if(!table.is_valid())
    {
        ERR_FAIL_MSG("path table not found:" + DataTableManager::get_singleton()->get_path_table_name().str());
    }
    data_version = DataTableManager::get_singleton()->get_data_table_version();
    Array data = table->data.values();
    
    for (int i = 0; i < data.size(); i++) {
        Dictionary d = data[i];
        if(d.has("group") && d.has("path") && d.has("enable")){
            String name = d["name"];
            String path = d["path"];
            bool enable = d["enable"];
            add_path(name,path,enable);
        }
    }

    is_init = true;
}