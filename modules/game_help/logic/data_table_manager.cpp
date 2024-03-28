#include "data_table_manager.h"
#include "../csv/CSV_EditorImportPlugin.h"

static DataTableManager* singleton = nullptr;
DataTableManager::DataTableManager() {
    singleton = this;
}

DataTableManager::~DataTableManager() {
    if(singleton == this)
    {
        singleton = nullptr;
    }
}
DataTableManager *DataTableManager::get_singleton() {
    return singleton;
}
void DataTableManager::init() {
    Ref<CSVData> db = ResourceLoader::load(data_table_path);
    if(!db.is_valid())
    {
        ERR_FAIL_MSG("data table not found:" + data_table_path);
    }
    Array data = db->get_data().values();


    for (int i = 0; i < data.size(); i++) {
        Dictionary d = data[i];
        if(d.has("name") && d.has("path")){
            String name = d["name"];
            String path = d["path"];
            Ref<CSVData> table = ResourceLoader::load(path);
            ERR_CONTINUE_MSG(!table.is_valid(),"data table not found:" + path);
            data_table[name] = table->get_data();
        }
    }
    is_init = true;
}

void DataTableManager::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("set_data_atble_path","path"),&DataTableManager::set_data_atble_path);
    ClassDB::bind_method(D_METHOD("reload"),&DataTableManager::reload);
    ClassDB::bind_method(D_METHOD("get_data_table","name"),&DataTableManager::get_data_table);

}