#pragma once

#include "core/io/resource_importer.h"
#include "core/variant/typed_array.h"
#include "core/variant/variant_utility.h"

struct CSVDataArray
{
    Vector<Vector<Variant>> data;
    void insert_row(int p_idx, const Vector<Variant>& p_array )
    {
        if( p_idx < 0)
            data.append(p_array);
        else
            data.insert(p_idx, p_array);

    }
    void append_row(const Vector<Variant>& p_array )
    {
        insert_row(-1, p_array);

    }
};

class CSVData : public Resource
{
    GDCLASS(CSVData, Resource);
    static void _bind_methods()
    {
        ClassDB::bind_method(D_METHOD("set_headers", "p_headers"), &CSVData::set_headers);
        ClassDB::bind_method(D_METHOD("get_headers"), &CSVData::get_headers);
        ClassDB::bind_method(D_METHOD("set_data", "p_data"), &CSVData::set_data);
        ClassDB::bind_method(D_METHOD("get_data"), &CSVData::get_data);


        ADD_PROPERTY(PropertyInfo(Variant::PACKED_STRING_ARRAY, "headers"), "set_headers", "get_headers");
        ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY, "data"), "set_data", "get_data");

    }

    Vector<String> headers;
    Dictionary data;
    public:
    void set_headers(const Vector<String>& p_headers)
    {
        headers = p_headers;
    }
    Vector<String> get_headers() const
    {
        return headers;
    }
    void set_data(const Dictionary& p_data)
    {
        data = p_data;
    }
    Dictionary get_data() const
    {
        return data;
    }
    void setup(const Vector<String>& p_headers, const Vector<Vector<Variant>>& records )
    {
        headers = p_headers;
        Dictionary field_indexs;
        for(int i = 0; i < p_headers.size(); ++i)
        {
		    field_indexs[headers[i]] = i;
            for(auto& row : records){
                int primary_key = row[0];
                Dictionary row_data;
                for(int j = 0; j < p_headers.size(); ++j)
                {
                    row_data[p_headers[j]] = row[j];
                }
                data[primary_key] = row_data;
            }
        }
    }

};


class CSV_EditorImportPlugin : public ResourceImporter
{
	GDCLASS(CSV_EditorImportPlugin, ResourceImporter);
public:
    String get_importer_name() const override
    {
        return "csv";
    }
    String get_visible_name() const override
    {
        return "CSV";
    }
    float get_priority() const override
    {
        return 1.0f;
    }
    int get_import_order() const override
    {
        return 0;
    }
    void get_recognized_extensions(List<String> *p_extensions) const override
    {
        p_extensions->push_back("csv");
        p_extensions->push_back("tsv");
    }
    String get_save_extension() const override
    {
        return "tres";
    }

    String get_resource_type() const override
    {
        return "Resource";
    }
    int get_preset_count() const override
    {
        return 0;
    }
    String get_preset_name(int p_idx) const override
    {
        return "CSV";
    }

    void get_import_options(const String &p_path, List<ImportOption> *r_options, int p_preset) const override
    {
        r_options->push_back(ImportOption(PropertyInfo(Variant::INT, "delimiter", PROPERTY_HINT_ENUM, "Comma,Tab, Semicolon"), 0));
        r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL, "describe_headers"), true));

    }
    bool get_option_visibility(const String &p_path, const String &p_option, const HashMap<StringName, Variant> &p_options) const override
    {
        return true;
    }
    void on_table_loaded();

    Error import(const String &p_source_file, const String &p_save_path, const HashMap<StringName, Variant> &p_options, List<String> *r_platform_variants, List<String> *r_gen_files, Variant *r_metadata) override
    {
        String delim;
        if(p_options.has("delimiter")){
            int del = p_options["delimiter"];

            if(del == 0){
                delim = ",";
            }
            if(del == 1){
                delim = "\t";
            }
            if(del == 2){
                delim = ";";
            }
        }
        
        Ref<FileAccess> file = FileAccess::open(p_source_file, FileAccess::READ);

	    ERR_FAIL_COND_V_MSG(file.is_null(), ERR_CANT_OPEN, "Cannot open file from path :'" + p_source_file + "'.");

        CSVDataArray lines ;
        Dictionary meta = parse_headers(file, p_options,delim,p_source_file);
        if(meta.size()==0){
            return ERR_FILE_UNRECOGNIZED;
        }
        
        PackedStringArray headers = meta["headers"];
        Dictionary field_indexs = meta["field_indexs"];
        Dictionary field_types = meta["field_types"];
        int line_index = 3;
        while(file->eof_reached()==false){
		    PackedStringArray line = file->get_csv_line(delim);
            Vector<Variant> row = parse_type(line, headers, field_indexs,field_types,line_index,p_source_file);
            if(row.size()>0)
                lines.append_row(row);
            ++line_index;
        }
        file->close();

        Ref<CSVData> csv = Ref<CSVData>(memnew(CSVData));
        csv->set_headers(meta["headers"]);
        csv->setup(headers,lines.data);

        String file_name = p_save_path + ".tres";
        Error err =  ResourceSaver::save(csv,file_name);
        ERR_FAIL_COND_V_MSG(err != OK, ERR_CANT_CREATE, "Cannot create file from path '" + file_name + "'.");
        on_table_loaded();
        return err;

    }

    Dictionary parse_headers(const Ref<FileAccess> & f,const HashMap<StringName, Variant> &p_options,const String& delim,const String& file_name)
    {
        String model_name = "";
        if (p_options["describe_headers"])
        {
            PackedStringArray _desc = f->get_csv_line(delim);
            model_name= _desc[0];

        }
        
        PackedStringArray headers  = f->get_csv_line(delim);
        ERR_FAIL_COND_V_MSG(headers[0] != "id", Dictionary(),file_name + " : First column must be 'id'");
        PackedStringArray types  = f->get_csv_line(delim);
        ERR_FAIL_COND_V_MSG(headers.size() != types.size(), Dictionary(),file_name + " : Headers and types must be the same size");

        Dictionary field_indexs;
        Dictionary field_types;

        for(int i = 0; i  < headers.size(); ++i)
        {
            field_indexs[headers[i]] = i;

        }
        for(int i = 0; i  < types.size(); ++i)
        {
            field_types[headers[i]] = types[i];
        }
        Dictionary ret;
        ret["field_indexs"] = field_indexs;
        ret["field_types"] = field_types;
        ret["model_name"] = model_name;
        ret["headers"] = headers;

        return ret;
    }
    Vector<Variant> parse_type(const PackedStringArray & csv_row,const PackedStringArray& headers,const Dictionary& field_indexs,const Dictionary& field_types,int line_index,const String& file_name)
    {
        int column_count = headers.size();
        if(column_count!=csv_row.size()){
            //WARN_PRINT("[csv-importer]:csv row data not enough file:" + file_name + " line:\n" + itos(line_index) + itos(column_count) + " - > " + itos(csv_row.size()) +" = " + VariantUtilityFunctions::var_to_str( csv_row));
            return  Vector<Variant>();
        }

         Vector<Variant> ret;
        for(int i = 0; i  < headers.size(); ++i)
        {
            ret.push_back(parse_type_value(field_types[headers[i]],csv_row[field_indexs[headers[i]]]));
        }
        return ret;
    }

    Variant parse_type_value(const String &type,const String &value)
    {
        if(type=="int"){
            return value.to_int();
        }
        else if(type=="float"){
            return value.to_float();
        }
        else if(type=="string"){
            return value;
        }
        else if(type=="bool"){
            return VariantUtilityFunctions::str_to_var(value);
        }
        else if(type=="color"){
            return VariantUtilityFunctions::str_to_var(value);
        }
        else if(type=="rect2"){
            return VariantUtilityFunctions::str_to_var(value);
        }
        else if(type=="vector2"){
            return VariantUtilityFunctions::str_to_var(value);
        }
        else if(type=="vector3"){
            return VariantUtilityFunctions::str_to_var(value);
        }
        else if(type == "json")
        {
            if(value.is_empty()){
                return "[]";
            }
            String v = value.replace("`", "\"");
            return VariantUtilityFunctions::str_to_var(v);
        }
        return value;
    }

};
