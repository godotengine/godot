#include "mterrain_material.h"

#include "core/io/resource_loader.h"
#include "core/io/resource_loader.h"
#include "core/variant/variant.h"


#include "mgrid.h"
#include "mimage.h"

#define RSS RenderingServer::get_singleton()



void MTerrainMaterial::_bind_methods() {
    ClassDB::bind_method(D_METHOD("set_shader","input"), &MTerrainMaterial::set_shader);
    ClassDB::bind_method(D_METHOD("get_shader"), &MTerrainMaterial::get_shader);
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"shader",PROPERTY_HINT_RESOURCE_TYPE,"Shader"),"set_shader","get_shader");
    ClassDB::bind_method(D_METHOD("set_show_region","input"), &MTerrainMaterial::set_show_region);
    ClassDB::bind_method(D_METHOD("get_show_region"), &MTerrainMaterial::get_show_region);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL,"show_region",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_EDITOR),"set_show_region","get_show_region");
    ClassDB::bind_method(D_METHOD("set_active_region","input"), &MTerrainMaterial::set_active_region);
    ClassDB::bind_method(D_METHOD("get_active_region"), &MTerrainMaterial::get_active_region);
    ADD_PROPERTY(PropertyInfo(Variant::INT,"active_region"),"set_active_region","get_active_region");
    ClassDB::bind_method(D_METHOD("set_clear_all","input"),&MTerrainMaterial::set_clear_all);
    ClassDB::bind_method(D_METHOD("get_clear_all"),&MTerrainMaterial::get_clear_all);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL,"clear_all",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NONE),"set_clear_all","get_clear_all");

    ClassDB::bind_method(D_METHOD("_set_uniforms"), &MTerrainMaterial::set_uniforms);
    ClassDB::bind_method(D_METHOD("_get_uniforms"), &MTerrainMaterial::get_uniforms);
    ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY,"_uniforms",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE),"_set_uniforms","_get_uniforms");
    ClassDB::bind_method(D_METHOD("_set_next_passes"), &MTerrainMaterial::set_next_passes);
    ClassDB::bind_method(D_METHOD("_get_next_passes"), &MTerrainMaterial::get_next_passes);
    ADD_PROPERTY(PropertyInfo(Variant::DICTIONARY,"_next_passes",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE),"_set_next_passes","_get_next_passes");
    ClassDB::bind_method(D_METHOD("_shader_code_changed"), &MTerrainMaterial::_shader_code_changed);
    ClassDB::bind_method(D_METHOD("get_material"), &MTerrainMaterial::get_material);
    ClassDB::bind_method(D_METHOD("get_reserved_uniforms"), &MTerrainMaterial::get_reserved_uniforms);
}


void MTerrainMaterial::set_shader(Ref<Shader> input) {
    if(show_region){
        set_show_region(false);
    }
    if(input.is_valid()){
        ERR_FAIL_COND(input->get_class() == "VisualShader");
    }
    if(shader.is_valid()){
        shader->disconnect("changed",Callable(this,"_shader_code_changed"));
    }
    shader = input;
    if(shader.is_valid()){
        shader->connect("changed",Callable(this,"_shader_code_changed"));
        if(shader->get_code().is_empty()){
            shader->set_code(get_default_shader()->get_code());
        }
    }
    if(is_loaded){
        for(HashMap<int,RID>::Iterator it=materials.begin();it!=materials.end();++it){
            RSS->material_set_shader(it->value,get_currect_shader()->get_rid());
        }
    }
    update_uniforms_list();
}

Ref<Shader> MTerrainMaterial::get_shader() {
    return shader;
}

Ref<Shader> MTerrainMaterial::get_default_shader(){
    Ref<Shader> s;
    if(grid->is_opengl()){
        s = ResourceLoader::load(M_DEAFAULT_SHADER_OPENGL_PATH);
    } else {
        s = ResourceLoader::load(M_DEAFAULT_SHADER_PATH);
    }
   ERR_FAIL_COND_V_EDMSG(!s.is_valid(),s,"Default shader is not valid");
   return s;
}

Ref<Shader> MTerrainMaterial::get_currect_shader(){
    if(shader.is_valid()){
        return shader;
    }
    if(default_shader.is_valid()){
        return default_shader;
    }
    default_shader = get_default_shader();
    ERR_FAIL_COND_V(!default_shader.is_valid(),default_shader);
    return default_shader;
}

void MTerrainMaterial::set_uniforms(Dictionary input){
    uniforms = input;
}

Dictionary MTerrainMaterial::get_uniforms(){
    return uniforms;
}

void MTerrainMaterial::set_next_passes(Dictionary input){
    next_passes = input;
}

Dictionary MTerrainMaterial::get_next_passes(){
    return next_passes;
}

void MTerrainMaterial::update_uniforms_list(){
    if(!is_loaded){
        active_region = -1;
    }
    PackedStringArray reserved = get_reserved_uniforms();
    Vector<StringName> new_uniforms_names;
    PackedStringArray new_terrain_textures_names;
    if(get_currect_shader().is_valid()){
        List<PropertyInfo> param_list;
        shader->get_shader_uniform_list(& param_list);
        List<PropertyInfo>::Iterator bit= param_list.begin();
        while (bit)	
        {
            Dictionary u = *bit;
            String n = String(u["name"]);
            if(n.begins_with("mterrain_") && String(u["hint_string"]) == "Texture2D"){
                PackedStringArray parts = n.split("_");
                if(parts.size()>0){
                    String _n = parts[1];
                    for(int i=2; i<parts.size(); i++){
                        _n += "_" + parts[i];
                    }
                    new_terrain_textures_names.push_back(_n);
                    continue;
                }
            }
            if(reserved.has(n)){
                continue;
            }
            new_uniforms_names.push_back(StringName(n));
        }
        // Check if we a uniform is removed we remove its key and value Variant too
        if(param_list.size()!=0){ // if the size is zero maybe the shader code can not be compiled, For now there is no way to check if it compiled or not here
            Array reg_ids = uniforms.keys();
            for(int r=0;r<reg_ids.size();r++){
                Dictionary ureg = uniforms[reg_ids[r]];
                Array unames = ureg.keys();
                for(int n=0;n<unames.size();n++){
                    if(!new_uniforms_names.has(unames[n])){
                        ureg.erase(unames[n]);
                    }
                }
                uniforms[reg_ids[r]] = ureg;
            }
        }
    }
    uniforms_names = new_uniforms_names;
    terrain_textures_names = new_terrain_textures_names;
    notify_property_list_changed();
}

void MTerrainMaterial::_get_property_list(List<PropertyInfo> *p_list) const {
    PackedStringArray reserved = get_reserved_uniforms();
    if(active_region!=-1){
        p_list->push_back(PropertyInfo(Variant::BOOL,"Override Next Pass",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_EDITOR));
    }
    p_list->push_back(PropertyInfo(Variant::OBJECT,"Next Pass",PROPERTY_HINT_RESOURCE_TYPE,"BaseMaterial3D,ShaderMaterial",PROPERTY_USAGE_EDITOR));
    //Adding shader properties
    if(shader.is_valid()){
        p_list->push_back(PropertyInfo(Variant::INT,"Shader Parameters",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_SUBGROUP));
        // Clear ALL
        p_list->push_back(PropertyInfo(Variant::BOOL,"clear_all",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_EDITOR));
	    List<PropertyInfo> param_list;
        shader->get_shader_uniform_list(& param_list);
        List<PropertyInfo>::Iterator bit= param_list.begin();
        while (bit)	
        {
            Dictionary u = *bit;
            //for(int i=0;i<uniforms_props.size();i++){
            String n = String(u["name"]);
            if(reserved.has(n) || n.begins_with("mterrain_")){
                continue;
            }
            Variant::Type type = static_cast<Variant::Type>((int)u["type"]);
            PropertyHint hint = static_cast<PropertyHint>((int)u["hint"]);
            PropertyInfo p(type,n,hint,String(u["hint_string"]),PROPERTY_USAGE_EDITOR);
            p_list->push_back(p);
        }
    }
}

bool MTerrainMaterial::_get(const StringName &p_name, Variant &r_ret) const {
    if(p_name==StringName("Next Pass") && next_passes.has(active_region)){
        Dictionary npm = next_passes[active_region];
        if(npm.has("next_pass")){
            r_ret = npm["next_pass"];
            return true;
        }
        return false;
    }
    if(p_name==StringName("Override Next Pass") && next_passes.has(active_region)){
        Dictionary npm = next_passes[active_region];
        if(npm.has("override_next_pass")){
            r_ret = npm["override_next_pass"];
            return true;
        }
        return false;
    }
    if(uniforms.has(active_region) && uniforms_names.find(p_name)!=-1){
        Dictionary ureg = uniforms[active_region];
        r_ret = ureg[p_name];
        return true;
    }
    return false;
}

bool MTerrainMaterial::_set(const StringName &p_name, const Variant &p_value) {
    if(p_name==StringName("Next Pass")){
        Dictionary npm;
        if(next_passes.has(active_region)){
            npm = next_passes[active_region];
        }
        npm["next_pass"] = p_value;
        Ref<ShaderMaterial> shmat = p_value;
        if(shmat.is_valid()){
            if(!shmat->get_shader().is_valid()){
                Ref<Shader> s;
                s.instantiate();
                String c = "shader_type spatial;\n\nvoid vertex(){\n\n}\n\nvoid fragment(){\n\n}";
                s->set_code(c);
                shmat->set_shader(s);
            }
        }
        next_passes[active_region] = npm;
        set_next_pass(active_region);
        return true;
    }
    if(p_name==StringName("Override Next Pass")){
        Dictionary npm;
        if(next_passes.has(active_region)){
            npm = next_passes[active_region];
        }
        npm["override_next_pass"] = p_value;
        next_passes[active_region] = npm;
        set_next_pass(active_region);
        return true;
    }
    if(uniforms_names.find(p_name)!=-1){
        if(show_region){
            return false;
        }
        Dictionary ureg;
        if(uniforms.has(active_region)){
            ureg = uniforms[active_region];
        }
        ureg[p_name] = p_value;
        uniforms[active_region] = ureg;
        if(active_region==-1){
            set_default_uniform(p_name,p_value);
        } else {
            if(materials.has(active_region)){
                set_uniform(materials[active_region],p_name,p_value);
            }
        }
        return true;
    }
    return false;
}

void MTerrainMaterial::_shader_code_changed(){
    update_uniforms_list();
}

void MTerrainMaterial::set_active_region(int input){
    ERR_FAIL_COND_EDMSG(input!=-1 && !is_loaded,"You need to create the terrain to change this!");
    ERR_FAIL_COND(!grid);
    ERR_FAIL_COND(!grid->is_created());
    ERR_FAIL_COND(input<-1);
    ERR_FAIL_COND_EDMSG(input>grid->get_regions_count()-1,"Active region can not bigger than the number of regions");
    active_region = input;
    notify_property_list_changed();
}
int MTerrainMaterial::get_active_region(){
    return active_region;
}

void MTerrainMaterial::set_clear_all(bool input){
    if(active_region==-1){
        return;
    }
    uniforms.erase(active_region);
    back_all_to_default_uniform(active_region);
}

bool MTerrainMaterial::get_clear_all() {
    return true;
}

void MTerrainMaterial::set_show_region(bool input){
    if(!is_loaded){
        show_region = false;
        return;
    }
    show_region = input;
    if(!show_region_shader.is_valid()){
        show_region_shader = ResourceLoader::load(M_SHOW_REGION_SHADER_PATH);
    }
    if(show_region){
        for(HashMap<int,RID>::Iterator it=materials.begin();it!=materials.end();++it){
            RSS->material_set_shader(it->value,show_region_shader->get_rid());
        }
        grid->refresh_all_regions_uniforms();
    } else {
        Ref<Shader> s = get_currect_shader();
        for(HashMap<int,RID>::Iterator it=materials.begin();it!=materials.end();++it){
            RSS->material_set_shader(it->value,s->get_rid());
        }
        refresh_all_uniform();
    }
}
bool MTerrainMaterial::get_show_region(){
    return show_region;
}

void MTerrainMaterial::set_grid(MGrid* g) {
    grid = g;
    if(!g){
        return; // Maybe later do something when grid has been destroyed
    }
}

RID MTerrainMaterial::get_material(int region_id){
    if(materials.has(region_id)){
        return materials[region_id];
    }
    RID m = RSS->material_create();
    RSS->material_set_shader(m,get_currect_shader()->get_rid());
    materials.insert(region_id,m);
    //Setting uniforms
    Dictionary region_uniforms;
    Dictionary default_uniforms;
    if(uniforms.has(region_id)){
        region_uniforms = uniforms[region_id];
    }
    if(uniforms.has(-1)){
        default_uniforms = uniforms[-1];
    }
    for(int u=0;u<uniforms_names.size();u++){
        StringName uname = uniforms_names[u];
        if(region_uniforms.has(uname)){
            Variant val = region_uniforms[uname];
            set_uniform(m,uname,val);
            continue;
        }
        if(default_uniforms.has(uname)){
            Variant val = default_uniforms[uname];
            set_uniform(m,uname,val);
            continue;
        }
    }
    return m;
}

void MTerrainMaterial::remove_material(int region_id){
    if(!materials.has(region_id)){
        return;
    }
    RSS->free(materials[region_id]);
    materials.erase(region_id);
}

void MTerrainMaterial::load_images(Array images_names,Ref<MResource> first_res){
    /*
        terrain_textures_names come from shader uniform which has mterrain_ prefix
        images_names come from data directory
        if an image exist only in images_names not in terrain_textures_names, that image is flaged as RAM image which
        exist only in RAM side not VRAM side
    */
    ERR_FAIL_COND(!grid);
    ERR_FAIL_COND(!grid->is_created());
    if(is_loaded){
        clear();
    }
    update_uniforms_list();
    //Adding textures
    //Making sure images names are string not stringName
    PackedStringArray pimages_names;
    for(int i=0; i < images_names.size(); i++){
        pimages_names.push_back(String(images_names[i]));
    }
    //Combine
    for(int i=0;i<terrain_textures_names.size();i++){
        if(!pimages_names.has(terrain_textures_names[i])){
            pimages_names.push_back(terrain_textures_names[i]);
        }
    }
    for(int i=0;i<pimages_names.size();i++){
        Image::Format _f = Image::Format::FORMAT_MAX;
        if (pimages_names[i] == NORMALS_NAME){
            _f = Image::Format::FORMAT_RGB8;
        } else if(pimages_names[i] == HEIGHTMAP_NAME){
            _f = Image::Format::FORMAT_RF;
        } else if(first_res.is_valid()){
            _f = first_res->get_data_format(pimages_names[i]);
        }
        _f = _f == Image::Format::FORMAT_MAX ? Image::Format::FORMAT_L8 : _f;
        add_terrain_image(pimages_names[i], !terrain_textures_names.has(pimages_names[i]), _f);
    }
    set_all_next_passes();
    show_region = false;
    is_loaded = true;
}

void MTerrainMaterial::clear(){
    show_region = false;
	for(int i=0;i<all_images.size();i++){
		memdelete(all_images[i]);
	}
    all_heightmap_images.clear();
    all_images.clear();
    terrain_textures_added.clear();
    terrain_textures_ids.clear();
    for(HashMap<int,RID>::Iterator it=materials.begin();it!=materials.end();++it){
        RSS->free(it->value);
    }
    materials.clear();
    is_loaded = false;
    active_region = -1;
}

void MTerrainMaterial::add_terrain_image(StringName name, bool is_ram_image, Image::Format _f) {
    String uniform_name = "mterrain_" + name;
    MGridPos region_grid_size = grid->get_region_grid_size();
    for(int z=0; z<region_grid_size.z;z++){
        for(int x=0; x<region_grid_size.x;x++){
            MRegion* region = grid->get_region(x,z);
            MGridPos rpos(x,0,z);
            MImage* i = memnew(MImage(name,uniform_name,rpos,region));
            i->is_ram_image = is_ram_image;
            i->format = _f;
            region->add_image(i);
            all_images.push_back(i);
            if(name==HEIGHTMAP_NAME){
                all_heightmap_images.push_back(i);
            }
        }
    }
    terrain_textures_added.push_back(name);
    terrain_textures_ids.insert(name,terrain_textures_added.size()-1);
}

void MTerrainMaterial::create_empty_terrain_image(StringName name,Image::Format format){
    String uniform_name = "mterrain_" + name;
    MGridPos region_grid_size = grid->get_region_grid_size();
    for(int z=0; z<region_grid_size.z;z++){
        for(int x=0; x<region_grid_size.x;x++){
            MRegion* region = grid->get_region(x,z);
            MGridPos rpos(x,0,z);
            MImage* i = memnew(MImage(name,uniform_name,rpos,region));
            i->format = format;
            region->add_image(i);
            all_images.push_back(i);
            if(name==HEIGHTMAP_NAME){
                all_heightmap_images.push_back(i);
            }
        }
    }
    terrain_textures_added.push_back(name);
    terrain_textures_ids.insert(name,terrain_textures_added.size()-1);
}

int MTerrainMaterial::get_texture_id(const String& name){
    if(!terrain_textures_ids.has(name)){
        if(name.is_empty()){
            //WARN_PRINT("Texture name is empty");
            return -1;
        }
        WARN_PRINT("Texture "+name+" does not exist");
        return -1;
    }
    return terrain_textures_ids[name];
}

PackedStringArray MTerrainMaterial::get_textures_list(){
    return terrain_textures_added;
}

void MTerrainMaterial::set_uniform(RID mat,StringName uname,Variant value){
    if(value.get_type() == Variant::OBJECT){
        RID tex_rid = value;
        RSS->material_set_param(mat,uname,tex_rid);
        return;
    }
    RSS->material_set_param(mat,uname,value);
}

void MTerrainMaterial::set_default_uniform(StringName uname,Variant value){
    for(HashMap<int,RID>::Iterator it=materials.begin();it!=materials.end();++it){
        if(uniforms.has(it->key)){
            Dictionary ureg = uniforms[it->key];
            if(ureg.has(uname)){
                continue;
            }
        }
        set_uniform(it->value,uname,value);
    }
}

void MTerrainMaterial::back_to_default_uniform(int region_id,StringName uname){
    ERR_FAIL_COND(region_id==-1);
    ERR_FAIL_COND(!materials.has(region_id));
    ERR_FAIL_COND(!uniforms.has(-1));
    Dictionary ureg = uniforms[-1];
    if(ureg.has(uname)){
        set_uniform(materials[region_id],uname,ureg[uname]);
    } else {
        set_uniform(materials[region_id],uname,Variant());
    }
}

void MTerrainMaterial::back_all_to_default_uniform(int region_id){
    ERR_FAIL_COND(region_id==-1);
    ERR_FAIL_COND(!materials.has(region_id));
    ERR_FAIL_COND(!uniforms.has(-1));
    Dictionary ureg = uniforms[-1];
    for(int i=0;i<uniforms_names.size();i++){
        StringName uname = uniforms_names[i];
        if(ureg.has(uname)){
            set_uniform(materials[region_id],uname,ureg[uname]);
        } else {
            set_uniform(materials[region_id],uname,Variant());
        }
    }

}

void MTerrainMaterial::refresh_all_uniform(){
    for(HashMap<int,RID>::Iterator it=materials.begin();it!=materials.end();++it){
        RID m = it->value;
        int region_id = it->key;
        //Setting uniforms
        Dictionary region_uniforms;
        Dictionary default_uniforms;
        if(uniforms.has(region_id)){
            region_uniforms = uniforms[region_id];
        }
        if(uniforms.has(-1)){
            default_uniforms = uniforms[-1];
        }
        for(int u=0;u<uniforms_names.size();u++){
            StringName uname = uniforms_names[u];
            if(region_uniforms.has(uname)){
                Variant val = region_uniforms[uname];
                set_uniform(m,uname,val);
                continue;
            }
            if(default_uniforms.has(uname)){
                Variant val = default_uniforms[uname];
                set_uniform(m,uname,val);
                continue;
            }
        }
    }
    grid->refresh_all_regions_uniforms();
}

void MTerrainMaterial::clear_all_uniform(){
    for(HashMap<int,RID>::Iterator it=materials.begin();it!=materials.end();++it){
        RID m = it->value;
        int region_id = it->key;
        //Setting uniforms
        Dictionary region_uniforms;
        Dictionary default_uniforms;
        for(int u=0;u<uniforms_names.size();u++){
            StringName uname = uniforms_names[u];
            set_uniform(m,uname,Variant());
        }
    }
}

PackedStringArray MTerrainMaterial::get_reserved_uniforms() const{
    return String(M_SHADER_RESERVE_UNIFORMS).split(",");
}

void MTerrainMaterial::set_next_pass(int region_id){
    if(region_id==-1){
        for(HashMap<int,RID>::Iterator it=materials.begin();it!=materials.end();++it){
            if(next_passes.has(it->key)){
                Dictionary npm = next_passes[it->key];
                if(npm.has("override_next_pass")){
                    bool ov = npm["override_next_pass"];
                    if(ov){
                        continue;
                    }
                }
            }
            RID next_pass_rid;
            if(next_passes.has(-1)){
                Dictionary npm = next_passes[-1];
                if(npm.has("next_pass")){
                    next_pass_rid = npm["next_pass"];
                }
            }
            RSS->material_set_next_pass(it->value,next_pass_rid);
        }
        return;
    }
    RID next_pass_rid;
    if(next_passes.has(region_id)){
        Dictionary npm = next_passes[region_id];
        if(npm.has("override_next_pass")){
            bool ov = npm["override_next_pass"];
            if(ov){
                next_pass_rid = npm["next_pass"];
                RSS->material_set_next_pass(materials[region_id],next_pass_rid);
                return;
            }
        }
    }
    if(next_passes.has(-1)){
        Dictionary npm = next_passes[-1];
        if(npm.has("next_pass")){
            next_pass_rid = npm["next_pass"];
        }
    }
    RSS->material_set_next_pass(materials[region_id],next_pass_rid);
}

void MTerrainMaterial::set_all_next_passes() {
    for(HashMap<int,RID>::Iterator it=materials.begin();it!=materials.end();++it){
        if(next_passes.has(it->key)){
            Dictionary npm = next_passes[it->key];
            if(npm.has("override_next_pass")){
                bool ov = npm["override_next_pass"];
                if(ov){
                    RID next_pass_rid;
                    if(npm.has("next_pass")){
                        next_pass_rid = npm["next_pass"];
                    }
                    RSS->material_set_next_pass(it->value,next_pass_rid);
                    continue;
                }
            }
        }
        if(next_passes.has(-1)){
            Dictionary npm = next_passes[-1];
            Ref<Material> mat = npm["next_pass"];
            if(mat.is_valid()){
                RSS->material_set_next_pass(it->value,mat->get_rid());
                continue;
            }
        }
        RSS->material_set_next_pass(it->value,RID());
    }
}