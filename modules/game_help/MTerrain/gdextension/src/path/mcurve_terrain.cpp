#include "mcurve_terrain.h"



void MCurveTerrain::_bind_methods(){
    ClassDB::bind_method(D_METHOD("set_curve","input"), &MCurveTerrain::set_curve);
    ClassDB::bind_method(D_METHOD("get_curve"), &MCurveTerrain::get_curve);
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"curve",PROPERTY_HINT_NONE,"MCurve"),"set_curve","get_curve");

    ClassDB::bind_method(D_METHOD("set_terrain","m_terrain"), &MCurveTerrain::set_terrain);
    ClassDB::bind_method(D_METHOD("get_terrain"), &MCurveTerrain::get_terrain);
    ADD_PROPERTY(PropertyInfo(Variant::OBJECT,"terrain",PROPERTY_HINT_NONE,"MTerrain"),"set_terrain","get_terrain");

    ClassDB::bind_method(D_METHOD("set_terrain_layer_name","input"), &MCurveTerrain::set_terrain_layer_name);
    ClassDB::bind_method(D_METHOD("get_terrain_layer_name"), &MCurveTerrain::get_terrain_layer_name);
    ADD_PROPERTY(PropertyInfo(Variant::STRING,"terrain_layer_name"),"set_terrain_layer_name","get_terrain_layer_name");

    ClassDB::bind_method(D_METHOD("set_apply_tilt","input"), &MCurveTerrain::set_apply_tilt);
    ClassDB::bind_method(D_METHOD("get_apply_tilt"), &MCurveTerrain::get_apply_tilt);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL,"apply_tilt"),"set_apply_tilt","get_apply_tilt");

    ClassDB::bind_method(D_METHOD("set_apply_scale","input"), &MCurveTerrain::set_apply_scale);
    ClassDB::bind_method(D_METHOD("get_apply_scale"), &MCurveTerrain::get_apply_scale);
    ADD_PROPERTY(PropertyInfo(Variant::BOOL,"apply_scale"),"set_apply_scale","get_apply_scale");

    ClassDB::bind_method(D_METHOD("set_deform_offest","input"), &MCurveTerrain::set_deform_offest);
    ClassDB::bind_method(D_METHOD("get_deform_offest"), &MCurveTerrain::get_deform_offest);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"deform_offest"),"set_deform_offest","get_deform_offest");

    ClassDB::bind_method(D_METHOD("set_deform_radius","input"), &MCurveTerrain::set_deform_radius);
    ClassDB::bind_method(D_METHOD("get_deform_radius"), &MCurveTerrain::get_deform_radius);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"deform_radius"),"set_deform_radius","get_deform_radius");

    ClassDB::bind_method(D_METHOD("set_deform_falloff","input"), &MCurveTerrain::set_deform_falloff);
    ClassDB::bind_method(D_METHOD("get_deform_falloff"), &MCurveTerrain::get_deform_falloff);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"deform_falloff"),"set_deform_falloff","get_deform_falloff");

    ClassDB::bind_method(D_METHOD("set_terrain_image_name","input"), &MCurveTerrain::set_terrain_image_name);
    ClassDB::bind_method(D_METHOD("get_terrain_image_name"), &MCurveTerrain::get_terrain_image_name);
    ADD_PROPERTY(PropertyInfo(Variant::STRING,"terrain_image_name"),"set_terrain_image_name","get_terrain_image_name");

    ClassDB::bind_method(D_METHOD("set_paint_color","input"), &MCurveTerrain::set_paint_color);
    ClassDB::bind_method(D_METHOD("get_paint_color"), &MCurveTerrain::get_paint_color);
    ADD_PROPERTY(PropertyInfo(Variant::STRING,"paint_color"),"set_paint_color","get_paint_color");

    ClassDB::bind_method(D_METHOD("set_bg_color","input"), &MCurveTerrain::set_bg_color);
    ClassDB::bind_method(D_METHOD("get_bg_color"), &MCurveTerrain::get_bg_color);
    ADD_PROPERTY(PropertyInfo(Variant::STRING,"bg_color"),"set_bg_color","get_bg_color");

    ClassDB::bind_method(D_METHOD("set_paint_radius","input"), &MCurveTerrain::set_paint_radius);
    ClassDB::bind_method(D_METHOD("get_paint_radius"), &MCurveTerrain::get_paint_radius);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"paint_radius"),"set_paint_radius","get_paint_radius");

    ClassDB::bind_method(D_METHOD("set_paint_falloff","input"), &MCurveTerrain::set_paint_falloff);
    ClassDB::bind_method(D_METHOD("get_paint_falloff"), &MCurveTerrain::get_paint_falloff);
    ADD_PROPERTY(PropertyInfo(Variant::FLOAT,"paint_falloff"),"set_paint_falloff","get_paint_falloff");

    ClassDB::bind_method(D_METHOD("deform_on_conns","conn_ids"), &MCurveTerrain::deform_on_conns);
    ClassDB::bind_method(D_METHOD("clear_deform_aabb","aabb"), &MCurveTerrain::clear_deform_aabb);
    ClassDB::bind_method(D_METHOD("clear_deform","conn_ids"), &MCurveTerrain::clear_deform);

    ClassDB::bind_method(D_METHOD("paint_on_conns","conn_ids"), &MCurveTerrain::paint_on_conns);
    ClassDB::bind_method(D_METHOD("clear_paint_aabb","aabb"), &MCurveTerrain::clear_paint_aabb);
    ClassDB::bind_method(D_METHOD("clear_paint","conn_ids"), &MCurveTerrain::clear_paint);

    ClassDB::bind_method(D_METHOD("clear_grass_aabb","grass","aabb","radius_plus_offset"), &MCurveTerrain::clear_grass_aabb);
    ClassDB::bind_method(D_METHOD("clear_grass","conn_ids","grass","radius_plus_offset"), &MCurveTerrain::clear_grass);
    ClassDB::bind_method(D_METHOD("modify_grass","conn_ids","grass","start_offset","radius","add"), &MCurveTerrain::modify_grass);
}

void MCurveTerrain::set_curve(MCurve* input){
    curve = input;
}

MCurve* MCurveTerrain::get_curve(){
    return curve;
}

void MCurveTerrain::set_terrain(MTerrain* m_terrain){
    terrain = m_terrain;
    grid = m_terrain->grid;
}

MTerrain* MCurveTerrain::get_terrain(){
    return terrain;
}

void MCurveTerrain::set_terrain_layer_name(const String& input){
    terrain_layer_name = input;
}

String MCurveTerrain::get_terrain_layer_name(){
    return terrain_layer_name;
}

void MCurveTerrain::set_apply_tilt(bool input){
    apply_tilt = input;
}

bool MCurveTerrain::get_apply_tilt(){
    return apply_tilt;
}

void MCurveTerrain::set_apply_scale(bool input){
    apply_scale = input;
}

bool MCurveTerrain::get_apply_scale(){
    return apply_scale;
}

void MCurveTerrain::set_deform_offest(float input){
    deform_offest = input;
}

float MCurveTerrain::get_deform_offest(){
    return deform_offest;
}

void MCurveTerrain::set_deform_radius(float input){
    ERR_FAIL_COND(input < 0.0f);
    deform_radius = input;
}

float MCurveTerrain::get_deform_radius(){
    return deform_radius;
}

void MCurveTerrain::set_deform_falloff(float input){
    ERR_FAIL_COND(input < 0.0f);
    deform_falloff = input;
}

float MCurveTerrain::get_deform_falloff(){
    return deform_falloff;
}

void MCurveTerrain::set_terrain_image_name(const String& input){
    terrain_image_name = input;
}

String MCurveTerrain::get_terrain_image_name() const{
    return terrain_image_name;
}

void MCurveTerrain::set_paint_color(const Color& input){
    paint_color = input;
}

Color MCurveTerrain::get_paint_color() const{
    return paint_color;
}

void MCurveTerrain::set_bg_color(const Color& input){
    bg_color = input;
}

Color MCurveTerrain::get_bg_color() const{
    return bg_color;
}

void MCurveTerrain::set_paint_radius(float input){
    paint_radius = input;
}

float MCurveTerrain::get_paint_radius(){
    return paint_radius;
}

void MCurveTerrain::set_paint_falloff(float input){
    paint_falloff = input;
}

float MCurveTerrain::get_paint_falloff(){
    return paint_falloff;
}

void MCurveTerrain::clear_deform_aabb(AABB aabb){
    ERR_FAIL_COND(curve==nullptr);
    ERR_FAIL_COND(grid==nullptr);
    ERR_FAIL_COND(terrain==nullptr);
    ERR_FAIL_COND(!grid->is_created());
    String old_layer = terrain->get_active_layer_name();
    ERR_FAIL_COND_MSG(!terrain->set_active_layer_by_name(terrain_layer_name),"Terrain Layer "+terrain_layer_name+" does not exist");
    aabb = aabb.grow(deform_radius+deform_falloff+grid->get_h_scale());
    Vector2i start = grid->get_closest_pixel(aabb.position);
    Vector2i end = grid->get_closest_pixel(aabb.position + aabb.size);
    for(int j=start.y; j <= end.y; j++){
        for(int i=start.x; i <= end.x; i++){
            if(!grid->has_pixel(i,j)){
                continue;
            }
            float h = grid->get_height_by_pixel(i,j) - grid->get_height_by_pixel_in_layer(i,j);
            grid->set_height_by_pixel(i,j,h);
        }
    }
    grid->update_normals(start.x,end.x,start.y,end.y);
    grid->update_all_dirty_image_texture();
    terrain->set_active_layer_by_name(old_layer);
}

void MCurveTerrain::clear_deform(const PackedInt64Array& conn_ids){
    ERR_FAIL_COND(curve==nullptr);
    ERR_FAIL_COND(grid==nullptr);
    ERR_FAIL_COND(terrain==nullptr);
    ERR_FAIL_COND(!grid->is_created());
    //ERR_FAIL_COND(!curve->has_conn(conn_id));
    String old_layer = terrain->get_active_layer_name();
    ERR_FAIL_COND_MSG(!terrain->set_active_layer_by_name(terrain_layer_name),"Terrain Layer "+terrain_layer_name+" does not exist");
    float interval_meter = grid->get_h_scale() * 0.4;
    for(int c=0; c < conn_ids.size(); c++){
        int64_t conn_id = conn_ids[c];
        ERR_CONTINUE(!curve->has_conn(conn_id));
        int interval_count = curve->get_conn_lenght(conn_id)/interval_meter;
        float ratio_interval = 1.0f/interval_count;
        interval_count++;
        Vector<float> ratios;
        ratios.resize(interval_count);
        for(int i=0; i < interval_count; i++){
            float current_ratio = ratio_interval*i;
            ratios.set(i,current_ratio);
        }
        Vector<Transform3D> transforms;
        curve->get_conn_transforms(conn_id,ratios,transforms);
        HashSet<Vector2i> processed_px;
        for(int i=0; i < interval_count; i++){
            Transform3D t = transforms[i];
            Vector3 origin = t.origin;
            Vector3 z_dir = t.basis.get_column(2);
            float _paint_radius = deform_radius;
            float _paint_falloff = deform_falloff;
            if(apply_scale){
                float current_scale = z_dir.length();
                _paint_radius = deform_radius*current_scale;
                _paint_falloff = deform_falloff*current_scale;
            }
            z_dir.normalize();
            float total_dis = _paint_radius + _paint_falloff;
            float side_dis = -total_dis;
            while (side_dis <= total_dis)
            {
                Vector3 ppp = origin + z_dir*side_dis;
                Vector2i t_px = grid->get_closest_pixel(ppp);
                if(processed_px.has(t_px)){
                    side_dis += interval_meter;
                    continue;
                }
                processed_px.insert(t_px);
                float h = grid->get_height_by_pixel(t_px.x,t_px.y) - grid->get_height_by_pixel_in_layer(t_px.x,t_px.y);
                grid->set_height_by_pixel(t_px.x,t_px.y,h);
                side_dis += interval_meter;
            }
        }
    }
    AABB caabb = curve->get_conns_aabb(conn_ids);
    caabb = caabb.grow(grid->get_h_scale() + deform_radius + deform_falloff);
    Vector2i start = grid->get_closest_pixel(caabb.position);
    Vector2i end = grid->get_closest_pixel(caabb.position + caabb.size);
    grid->update_normals(start.x,end.x,start.y,end.y);
}

void MCurveTerrain::deform_on_conns(const PackedInt64Array& conn_ids){
    ERR_FAIL_COND(curve==nullptr);
    ERR_FAIL_COND(grid==nullptr);
    ERR_FAIL_COND(terrain==nullptr);
    ERR_FAIL_COND(!grid->is_created());
    String old_layer = terrain->get_active_layer_name();
    ERR_FAIL_COND_MSG(!terrain->set_active_layer_by_name(terrain_layer_name),"Terrain Layer "+terrain_layer_name+" does not exist");
    float interval_meter = grid->get_h_scale() * 0.4;
    for(int c=0; c < conn_ids.size(); c++){
        int64_t conn_id = conn_ids[c];
        ERR_CONTINUE(!curve->has_conn(conn_id));
        int interval_count = curve->get_conn_lenght(conn_id)/interval_meter;
        float ratio_interval = 1.0f/interval_count;
        interval_count++;
        Vector<float> ratios;
        ratios.resize(interval_count);
        for(int i=0; i < interval_count; i++){
            float current_ratio = ratio_interval*i;
            ratios.set(i,current_ratio);
        }
        Vector<Transform3D> transforms;
        curve->get_conn_transforms(conn_id,ratios,transforms,apply_tilt,apply_scale);
        HashSet<Vector2i> processed_px;
        for(int i=0; i < interval_count; i++){
            Transform3D t = transforms[i];
            Vector3 origin = t.origin;
            Vector3 z_dir = t.basis.get_column(2);
            Vector3 y_dir = t.basis.get_column(1);
            float _paint_radius = deform_radius;
            float _paint_falloff = deform_falloff;
            if(apply_scale){
                float current_scale = z_dir.length();
                _paint_radius = deform_radius*current_scale;
                _paint_falloff = deform_falloff*current_scale;
            }
            z_dir.normalize();
            y_dir.normalize();
            float total_dis = _paint_radius + _paint_falloff;
            float side_dis = -total_dis;
            while (side_dis <= total_dis)
            {
                Vector3 ppp = origin + z_dir*side_dis;
                ppp = ppp + y_dir*deform_offest;
                Vector2i t_px = grid->get_closest_pixel(ppp);
                if(processed_px.has(t_px)){
                    side_dis += interval_meter;
                    continue;
                }
                processed_px.insert(t_px);
                float h = grid->get_height_by_pixel(t_px.x,t_px.y);
                float t = VariantUtilityFunctions::smoothstep(total_dis,_paint_radius,std::abs(side_dis));
                h = (ppp.y - h)*t + h;
                grid->set_height_by_pixel(t_px.x,t_px.y,h);
                side_dis += interval_meter;
            }
        }
    }
    AABB caabb = curve->get_conns_aabb(conn_ids);
    caabb = caabb.grow(grid->get_h_scale() + deform_radius + deform_falloff);
    Vector2i start = grid->get_closest_pixel(caabb.position);
    Vector2i end = grid->get_closest_pixel(caabb.position + caabb.size);
    grid->update_normals(start.x,end.x,start.y,end.y);
}

void MCurveTerrain::clear_paint_aabb(AABB aabb){
    ERR_FAIL_COND(curve==nullptr);
    ERR_FAIL_COND(grid==nullptr);
    ERR_FAIL_COND(terrain==nullptr);
    ERR_FAIL_COND(!grid->is_created());
    int img_id = terrain->get_image_id(terrain_image_name);
    ERR_FAIL_COND_MSG(img_id==-1,"Image "+terrain_image_name+" is not found!");
    aabb = aabb.grow(deform_radius+deform_falloff+grid->get_h_scale());
    Vector2i start = grid->get_closest_pixel(aabb.position);
    Vector2i end = grid->get_closest_pixel(aabb.position + aabb.size);
    for(int j=start.y; j <= end.y; j++){
        for(int i=start.x; i <= end.x; i++){
            if(!grid->has_pixel(i,j)){
                continue;
            }
            grid->set_pixel(i,j,bg_color,img_id);
        }
    }
}

void MCurveTerrain::clear_paint(const PackedInt64Array& conn_ids){
    ERR_FAIL_COND(curve==nullptr);
    ERR_FAIL_COND(grid==nullptr);
    ERR_FAIL_COND(terrain==nullptr);
    ERR_FAIL_COND(!grid->is_created());
    int img_id = terrain->get_image_id(terrain_image_name);
    ERR_FAIL_COND_MSG(img_id==-1,"Image "+terrain_image_name+" is not found!");
    float interval_meter = grid->get_h_scale() * 0.4;
    for(int c=0; c < conn_ids.size(); c++){
        int64_t conn_id = conn_ids[c];
        ERR_CONTINUE(!curve->has_conn(conn_id));
        int interval_count = curve->get_conn_lenght(conn_id)/interval_meter;
        float ratio_interval = 1.0f/interval_count;
        interval_count++;
        Vector<float> ratios;
        ratios.resize(interval_count);
        for(int i=0; i < interval_count; i++){
            float current_ratio = ratio_interval*i;
            ratios.set(i,current_ratio);
        }
        Vector<Transform3D> transforms;
        curve->get_conn_transforms(conn_id,ratios,transforms,apply_tilt,apply_scale);
        HashSet<Vector2i> processed_px;
        for(int i=0; i < interval_count; i++){
            Transform3D t = transforms[i];
            Vector3 origin = t.origin;
            Vector3 z_dir = t.basis.get_column(2);
            Vector3 y_dir = t.basis.get_column(1);
            float _paint_radius = paint_radius;
            float _paint_falloff = paint_falloff;
            if(apply_scale){
                float current_scale = z_dir.length();
                _paint_radius = deform_radius*current_scale;
                _paint_falloff = deform_falloff*current_scale;
            }
            z_dir.normalize();
            y_dir.normalize();
            float total_dis = _paint_radius + _paint_falloff;
            float side_dis = -total_dis;
            while (side_dis <= total_dis)
            {
                Vector3 ppp = origin + z_dir*side_dis;
                ppp = ppp + y_dir*deform_offest;
                Vector2i t_px = grid->get_closest_pixel(ppp);
                if(processed_px.has(t_px)){
                    side_dis += interval_meter;
                    continue;
                }
                processed_px.insert(t_px);
                grid->set_pixel(t_px.x,t_px.y,bg_color,img_id);
                side_dis += interval_meter;
            }
        }
    }
}

void MCurveTerrain::paint_on_conns(const PackedInt64Array& conn_ids){
    ERR_FAIL_COND(curve==nullptr);
    ERR_FAIL_COND(grid==nullptr);
    ERR_FAIL_COND(terrain==nullptr);
    ERR_FAIL_COND(!grid->is_created());
    int img_id = terrain->get_image_id(terrain_image_name);
    ERR_FAIL_COND_MSG(img_id==-1,"Image "+terrain_image_name+" is not found!");
    float interval_meter = grid->get_h_scale() * 0.4;
    for(int c=0; c < conn_ids.size(); c++){
        int64_t conn_id = conn_ids[c];
        ERR_CONTINUE(!curve->has_conn(conn_id));
        int interval_count = curve->get_conn_lenght(conn_id)/interval_meter;
        float ratio_interval = 1.0f/interval_count;
        interval_count++;
        Vector<float> ratios;
        ratios.resize(interval_count);
        for(int i=0; i < interval_count; i++){
            float current_ratio = ratio_interval*i;
            ratios.set(i,current_ratio);
        }
        Vector<Transform3D> transforms;
        curve->get_conn_transforms(conn_id,ratios,transforms,apply_tilt,apply_scale);
        HashSet<Vector2i> processed_px;
        for(int i=0; i < interval_count; i++){
            Transform3D t = transforms[i];
            Vector3 origin = t.origin;
            Vector3 z_dir = t.basis.get_column(2);
            Vector3 y_dir = t.basis.get_column(1);
            float _paint_radius = paint_radius;
            float _paint_falloff = paint_falloff;
            if(apply_scale){
                float current_scale = z_dir.length();
                _paint_radius = paint_radius*current_scale;
                _paint_falloff = paint_falloff*current_scale;
            }
            z_dir.normalize();
            y_dir.normalize();
            float total_dis = _paint_radius + _paint_falloff;
            float side_dis = -total_dis;
            while (side_dis <= total_dis)
            {
                Vector3 ppp = origin + z_dir*side_dis;
                ppp = ppp + y_dir*deform_offest;
                Vector2i t_px = grid->get_closest_pixel(ppp);
                if(processed_px.has(t_px)){
                    side_dis += interval_meter;
                    continue;
                }
                processed_px.insert(t_px);
                float t = VariantUtilityFunctions::smoothstep(total_dis,_paint_falloff,std::abs(side_dis));
                Color tbg = grid->get_pixel(t_px.x,t_px.y,img_id);
                Color fcol = tbg.lerp(paint_color,t);
                grid->set_pixel(t_px.x,t_px.y,fcol,img_id);
                side_dis += interval_meter;
            }
        }
    }
}

void MCurveTerrain::clear_grass_aabb(MGrass* grass,AABB aabb,float radius_plus_offset){
    ERR_FAIL_COND(curve==nullptr);
    ERR_FAIL_COND(grid==nullptr);
    ERR_FAIL_COND(terrain==nullptr);
    ERR_FAIL_COND(!grid->is_created());
    ERR_FAIL_COND(grass==nullptr);
    ERR_FAIL_COND(!grass->is_init());

    aabb = aabb.grow(radius_plus_offset + grass->grass_data->density);
    grass->clear_grass_sublayer_aabb(aabb);
}


void MCurveTerrain::clear_grass(const PackedInt64Array& conn_ids,MGrass* grass,float radius_plus_offset){
    ERR_FAIL_COND(curve==nullptr);
    ERR_FAIL_COND(grid==nullptr);
    ERR_FAIL_COND(terrain==nullptr);
    ERR_FAIL_COND(!grid->is_created());
    ERR_FAIL_COND(grass==nullptr);
    ERR_FAIL_COND(!grass->is_init());

    if(!grass->has_sublayer()){
        grass->create_sublayer();
    }
    
    float interval_meter = grass->get_grass_data()->density * 0.4;
    radius_plus_offset += interval_meter;
    for(int c=0; c < conn_ids.size(); c++){
        int64_t conn_id = conn_ids[c];
        ERR_CONTINUE(!curve->has_conn(conn_id));
        int interval_count = curve->get_conn_lenght(conn_id)/interval_meter;
        float ratio_interval = 1.0f/interval_count;
        interval_count++;
        Vector<float> ratios;
        ratios.resize(interval_count);
        for(int i=0; i < interval_count; i++){
            float current_ratio = ratio_interval*i;
            ratios.set(i,current_ratio);
        }
        Vector<Transform3D> transforms;
        curve->get_conn_transforms(conn_id,ratios,transforms,apply_tilt,apply_scale);
        HashSet<Vector2i> processed_px;
        for(int i=0; i < interval_count; i++){
            Transform3D t = transforms[i];
            Vector3 origin = t.origin;
            Vector3 z_dir = t.basis.get_column(2);
            Vector3 y_dir = t.basis.get_column(1);
            // not use g_start_offset and g_radius in loop
            // they will be change for others too
            float _r = radius_plus_offset;
            if(apply_scale){
                float current_scale = z_dir.length();
                _r = radius_plus_offset*current_scale;
            }
            z_dir.normalize();
            y_dir.normalize();
            float total_dis = _r;
            float side_dis = -total_dis;
            while (side_dis <= total_dis)
            {
                Vector3 ppp = origin + z_dir*side_dis;
                Vector2i g_px = grass->get_closest_pixel(ppp);
                if(processed_px.has(g_px)){
                    side_dis += interval_meter;
                    continue;
                }
                processed_px.insert(g_px);
                grass->clear_grass_sublayer_by_pixel(g_px.x,g_px.y);
                side_dis += interval_meter;
            }
        }
    }
}

void MCurveTerrain::modify_grass(const PackedInt64Array& conn_ids,MGrass* grass,float g_start_offset,float g_radius,bool add){
    ERR_FAIL_COND(curve==nullptr);
    ERR_FAIL_COND(grid==nullptr);
    ERR_FAIL_COND(terrain==nullptr);
    ERR_FAIL_COND(!grid->is_created());
    ERR_FAIL_COND(grass==nullptr);
    ERR_FAIL_COND(!grass->is_init());

    if(!grass->has_sublayer()){
        grass->create_sublayer();
    }
    
    float interval_meter = grass->get_grass_data()->density * 0.4;
    for(int c=0; c < conn_ids.size(); c++){
        int64_t conn_id = conn_ids[c];
        ERR_CONTINUE(!curve->has_conn(conn_id));
        int interval_count = curve->get_conn_lenght(conn_id)/interval_meter;
        float ratio_interval = 1.0f/interval_count;
        interval_count++;
        Vector<float> ratios;
        ratios.resize(interval_count);
        for(int i=0; i < interval_count; i++){
            float current_ratio = ratio_interval*i;
            ratios.set(i,current_ratio);
        }
        Vector<Transform3D> transforms;
        curve->get_conn_transforms(conn_id,ratios,transforms,apply_tilt,apply_scale);
        HashSet<Vector2i> processed_px;
        for(int i=0; i < interval_count; i++){
            Transform3D t = transforms[i];
            Vector3 origin = t.origin;
            Vector3 z_dir = t.basis.get_column(2);
            Vector3 y_dir = t.basis.get_column(1);
            // not use g_start_offset and g_radius in loop
            // they will be change for others too
            float _gs = g_start_offset;
            float _gr = g_radius;
            if(apply_scale){
                float current_scale = z_dir.length();
                _gs = g_start_offset*current_scale;
                _gr = g_radius*current_scale;
            }
            z_dir.normalize();
            y_dir.normalize();
            float total_dis = _gs + _gr;
            float side_dis = -total_dis;
            while (side_dis <= total_dis)
            {
                Vector3 ppp = origin + z_dir*side_dis;
                Vector2i g_px = grass->get_closest_pixel(ppp);
                if(processed_px.has(g_px)){
                    side_dis += interval_meter;
                    continue;
                }
                processed_px.insert(g_px);
                if(std::abs(side_dis) >= _gs){
                    grass->set_grass_sublayer_by_pixel(g_px.x,g_px.y,add);
                } else {
                    grass->set_grass_sublayer_by_pixel(g_px.x,g_px.y,!add);
                }
                side_dis += interval_meter;
            }
        }
    }
}