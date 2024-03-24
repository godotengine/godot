#pragma once
#include "core/io/resource_loader.h"

class DataTableManager : public Object
{
    GDCLASS(DataTableManager, Object);
    static void _bind_methods();
public:
    DataTableManager();
    ~DataTableManager();

    static DataTableManager* get_singleton();

    void set_data_atble_path(String path)
    {
        data_table_path = path;
        is_init = false;
    }
    void init();
    void reload()
    {
        data_table.clear();
        is_init = false;
    }
    // 获取一个表
    Dictionary get_data_table(String name)
    {
        if(!is_init)
        {
            init();
        }
        if(!data_table.has(name))
        {
            return Dictionary();
        }
        return data_table[name];
    }

    bool is_init = false;
    HashMap<String,Dictionary> data_table;
    String data_table_path = "res://data_table.csv";
};