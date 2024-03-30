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
    if(is_init)
    {
        return;
    }
    ++version;
    Ref<CSVData> db = ResourceLoader::load(data_table_path);
    if(!db.is_valid())
    {
        ERR_FAIL_MSG("data table not found:" + data_table_path);
    }
    Array data = db->get_data().values();

    HashMap<StringName,Ref<DataTableItem>> old_table = data_table;
    data_table.clear();
    for (int i = 0; i < data.size(); i++) {
        Dictionary d = data[i];
        if(d.has("name") && d.has("path")){
            String name = d["name"];
            String path = d["path"];
            Ref<CSVData> table = ResourceLoader::load(path);
            ERR_CONTINUE_MSG(!table.is_valid(),"data table not found:" + path);

            Ref<DataTableItem> item;
            if(old_table.has(name))
            {
                item = old_table[name];
            } else {
                item.instantiate();
            }
            item->version = version;
            item->data = table->get_data();
            data_table[name] = item;
        }
    }
    is_init = true;
}

void DataTableManager::_bind_methods()
{
    ClassDB::bind_method(D_METHOD("set_data_table_path","path"),&DataTableManager::set_data_table_path);
    ClassDB::bind_method(D_METHOD("reload"),&DataTableManager::reload);
    ClassDB::bind_method(D_METHOD("get_data_table","name"),&DataTableManager::_get_data_table);

    ClassDB::bind_method(D_METHOD("set_animation_table_name","name"),&DataTableManager::set_animation_table_name);
    ClassDB::bind_method(D_METHOD("get_animation_table_name"),&DataTableManager::get_animation_table_name);

    ClassDB::bind_method(D_METHOD("set_body_table_name","name"),&DataTableManager::set_body_table_name);
    ClassDB::bind_method(D_METHOD("get_body_table_name"),&DataTableManager::get_body_table_name);

    ClassDB::bind_method(D_METHOD("set_path_table_name","name"),&DataTableManager::set_path_table_name);
    ClassDB::bind_method(D_METHOD("get_path_table_name"),&DataTableManager::get_path_table_name);

    ClassDB::bind_method(D_METHOD("set_mesh_part_table_name","name"),&DataTableManager::set_mesh_part_table_name);
    ClassDB::bind_method(D_METHOD("get_mesh_part_table_name"),&DataTableManager::get_mesh_part_table_name);

    ClassDB::bind_method(D_METHOD("set_charecter_table_name","name"),&DataTableManager::set_charecter_table_name);
    ClassDB::bind_method(D_METHOD("get_charecter_table_name"),&DataTableManager::get_charecter_table_name);

    ClassDB::bind_method(D_METHOD("set_scene_table_name","name"),&DataTableManager::set_scene_table_name);
    ClassDB::bind_method(D_METHOD("get_scene_table_name"),&DataTableManager::get_scene_table_name);

    ClassDB::bind_method(D_METHOD("set_item_table_name","name"),&DataTableManager::set_item_table_name);
    ClassDB::bind_method(D_METHOD("get_item_table_name"),&DataTableManager::get_item_table_name);

    
    ClassDB::bind_method(D_METHOD("set_skill_table_name","name"),&DataTableManager::set_skill_table_name);
    ClassDB::bind_method(D_METHOD("get_skill_table_name"),&DataTableManager::get_skill_table_name);


    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME,"animation_table_name"), "set_animation_table_name","get_animation_table_name");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME,"body_table_name"), "set_body_table_name","get_body_table_name");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME,"path_table_name"), "set_path_table_name","get_path_table_name");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME,"mesh_part_table_name"), "set_mesh_part_table_name","get_mesh_part_table_name");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME,"charecter_table_name"), "set_charecter_table_name","get_charecter_table_name");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME,"scene_table_name"), "set_scene_table_name","get_scene_table_name");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME,"item_table_name"), "set_item_table_name","get_item_table_name");
    ADD_PROPERTY(PropertyInfo(Variant::STRING_NAME,"skill_table_name"), "set_skill_table_name","get_skill_table_name");

}