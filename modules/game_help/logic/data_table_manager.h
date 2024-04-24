#pragma once
#include "core/io/resource_loader.h"
#include "core/io/json.h"


class DataTableItem : public RefCounted
{
    public:
    int version = 0;
    Dictionary data;
};

class DataTableManager : public Object
{
    GDCLASS(DataTableManager, Object);
    static void _bind_methods();
public:
    // 一些表名
    StringName animation_table_name = StringName("animation_table");
    StringName path_table_name = StringName("path_table");
    StringName body_table_name = StringName("body_table");
    StringName mesh_part_table_name = StringName("mesh_part_table");
    StringName charecter_table_name = StringName("charecter_table");
    StringName scene_table_name = StringName("scene_table");
    StringName item_table_name = StringName("item_table_name");
    StringName skill_table_name = StringName("skill_table_name");

    void set_animation_table_name(const StringName& name)
    {
        animation_table_name = name;
        reload();
    }
    StringName get_animation_table_name()
    {
        return animation_table_name;
    }

    void set_path_table_name(const StringName& name)
    {
        path_table_name = name;
        reload();
    }
    StringName get_path_table_name()
    {
        return path_table_name;
    }

    void set_body_table_name(const StringName& name)
    {
        body_table_name = name;
        reload();
    }
    StringName get_body_table_name()
    {
        return body_table_name;
    }

    void set_mesh_part_table_name(const StringName& name)
    {
        mesh_part_table_name = name;
        reload();
    }
    StringName get_mesh_part_table_name()
    {
        return mesh_part_table_name;
    }

    void set_charecter_table_name(const StringName& name)
    {
        charecter_table_name = name;
        reload();
    }
    StringName get_charecter_table_name()
    {
        return charecter_table_name;
    }

    void set_scene_table_name(const StringName& name)
    {
        scene_table_name = name;
        reload();
    }
    StringName get_scene_table_name()
    {
        return scene_table_name;
    }

    void set_item_table_name(const StringName& name)
    {
        item_table_name = name;
        reload();
    }
    StringName get_item_table_name()
    {
        return item_table_name;
    }

    void set_skill_table_name(const StringName& name)
    {
        skill_table_name = name;
        reload();
    }
    StringName get_skill_table_name()
    {
        return skill_table_name;
    }


public:
    DataTableManager();
    ~DataTableManager();

    void on_table_load()
    {
        is_init = false;
    }

    static DataTableManager* get_singleton();

    void set_data_table_path(String path)
    {
        data_table_path = path;
        is_init = false;
    }
    void init();
    void reload()
    {
        is_init = false;
    }
    // 获取一个表
    Ref<DataTableItem> get_data_table(const StringName& name)
    {
        if(!is_init)
        {
            init();
        }
        if(!data_table.has(name))
        {
            return Ref<DataTableItem>();
        }
        return data_table[name];
    }
    Dictionary _get_data_table(const StringName& name)
    {
        Ref<DataTableItem> item = get_data_table(name);
        if(!item.is_valid())
        {
            return Dictionary();
        }
        return item->data;
    }
    Dictionary _get_data_item(const StringName& name,int id)
    {
        Ref<DataTableItem> item = get_data_table(name);
        if(!item.is_valid())
        {
            return Dictionary();
        }
        if(!item->data.has(id))
        {
            return Dictionary();
        }
        return item->data[id];
    }
    int get_data_table_version()
    {
        return version;
    }

    Ref<JSON> parse_yaml(const String& text);
    Ref<JSON> parse_yaml_file(const String& file_path);
    void set_animation_load_cb(const Callable& cb );
    Callable get_animation_load_cb()
    {    
        return on_load_animation;
    }

    bool is_init = false;
    int version = 0;
	Callable on_load_animation;
    HashMap<StringName,Ref<DataTableItem>> data_table;
    String data_table_path = "res://data_table.csv";
};