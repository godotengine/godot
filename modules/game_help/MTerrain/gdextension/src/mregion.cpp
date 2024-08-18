#include "mregion.h"

#include "core/math/color.h"
#include "servers/rendering_server.h"
#include "core/variant/variant.h"
#include "servers/rendering_server.h"

#define RSS RenderingServer::get_singleton()

#include "mgrid.h"

Vector<Vector3> MRegion::nvecs;

MRegion::MRegion()
	:is_data_loaded(false) {
    lods = memnew(VSet<int8_t>);
}

MRegion::~MRegion(){
    memdelete(lods);
	remove_physics();
	images.clear();
}

void MRegion::set_material(RID input) {
    if(!input.is_valid()){
        return;
    }
	_material_rid = input;
	RSS->material_set_param(_material_rid,"region_size",grid->region_size_meter);
	RSS->material_set_param(_material_rid,"region_world_position",world_pos);
}

RID MRegion::get_material_rid() {
	return _material_rid;
}

void MRegion::add_image(MImage* input) {
    images.append(input);
	_images_init_status.push_back(false);
	if(input->name==HEIGHTMAP_NAME){
		heightmap = input;
	}
	else if(input->name==NORMALS_NAME){
		normals = input;
	}
}

void MRegion::configure() {
	ERR_FAIL_COND_MSG(!heightmap,"Heightmap is not loaded check MTerrain Material");
	ERR_FAIL_COND_MSG(!normals,"Normals is not loaded check MTerrain Material");
	for(int i=0; i < images.size(); i++){
		images[i]->region = this;
		images[i]->index = i;
		if(images[i]->name != NORMALS_NAME){
			images[i]->active_undo = true;
		}
		if(images[i]->is_null_image){
			if(images[i]->name == HEIGHTMAP_NAME){
				min_height = -0.01;
				max_height = 0.01;
			}
		}
	}
	uint32_t ss = grid->region_pixel_size - 1;
	normals_pixel_region.left = pos.x*ss;
	normals_pixel_region.right = (pos.x + 1)*ss;
	normals_pixel_region.top = pos.z*ss;
	normals_pixel_region.bottom = (pos.z + 1)*ss;
	//normals_pixel_region.grow_all_side(grid->grid_pixel_region);
}

void MRegion::load(){
	set_material(grid->_terrain_material->get_material(id));
	String res_path = get_res_path();
	Ref<MResource> mres;
	if(ResourceLoader::exists(res_path)){
		
		mres = ResourceLoader::load(res_path);
	}
	for(int i=0; i < images.size(); i++){
		images[i]->load(mres);
	}
	if(!mres.is_valid() || !mres->has_data(HEIGHTMAP_NAME)){
		min_height=0.0;
		max_height=0.0001;
	} else if(!is_min_max_height_calculated){
		// This value might change while correcting edge
		min_height = mres->get_min_height();
		max_height = mres->get_max_height();
		is_min_max_height_calculated = true;
		//_calculate_min_max_height();
	}
	//is_data_loaded will be set by update region
}

void MRegion::unload(){
	is_data_loaded.store(false, std::memory_order_release);
	Ref<MResource> mres;
	String res_path = get_res_path();
	if(ResourceLoader::exists(res_path)){
		mres = ResourceLoader::load(get_res_path());
	}
	for(int i=0; i < images.size(); i++){
		images[i]->unload(mres);
	}
}

String MRegion::get_res_path(){
	String res_name = String("x") + itos(pos.x) + String("_y") + itos(pos.z)+String(".res");
	return grid->dataDir.path_join(res_name);
}

void MRegion::update_region() {
	if(!is_data_loaded.load(std::memory_order_acquire)){
		return;
	}
	if(!_material_rid.is_valid()){
		return;
	}
    int8_t curren_lod = (lods->is_empty()) ? -1 : (*lods)[0];
	current_scale = pow(2, (int32_t)curren_lod);
	for(int i=0; i < images.size(); i++){
		MImage* img = images[i];
			if(last_lod != curren_lod || img->is_dirty){
			// if out of bound then we just add an empty texture
			if(curren_lod == -1){
				img->update_texture(0,false);
			} else {
				img->update_texture(current_scale,false);
			}
		}
	}
    last_lod = curren_lod;
    memdelete(lods);
    lods = memnew(VSet<int8_t>);
}

void MRegion::insert_lod(const int8_t input) {
    lods->insert(input);
}

void MRegion::apply_update() {
	if(!is_data_loaded.load(std::memory_order_acquire)){
		return;
	}
	if(!_material_rid.is_valid()){
		return;
	}
	for(int i=0; i < images.size(); i++){
		MImage* img = images[i];
		img->apply_update();
	}
	current_image_size = ((double)heightmap->current_size);
	RSS->material_set_param(_material_rid,"region_a",(current_image_size-1)/current_image_size);
	RSS->material_set_param(_material_rid,"region_b",0.5/current_image_size);
	RSS->material_set_param(_material_rid,"min_lod",last_lod);
}
	static PackedFloat32Array _PackedByteArray_decode_float_array(PackedByteArray *p_instance) {
		uint64_t size = p_instance->size();
		PackedFloat32Array dest;
		if (size == 0) {
			return dest;
		}
		ERR_FAIL_COND_V_MSG(size % sizeof(float), dest, "PackedByteArray size must be a multiple of 4 (size of 32-bit float) to convert to PackedFloat32Array.");
		const uint8_t *r = p_instance->ptr();
		dest.resize(size / sizeof(float));
		ERR_FAIL_COND_V(dest.is_empty(), dest); // Avoid UB in case resize failed.
		memcpy(dest.ptrw(), r, dest.size() * sizeof(float));
		return dest;
	}
void MRegion::create_physics() {
	std::lock_guard<std::mutex> lock(physics_mutex);
	ERR_FAIL_COND(heightmap == nullptr);
	if(has_physic || heightmap->is_corrupt_file || heightmap->is_null_image || to_be_remove || !get_data_load_status()){
		return;
	}
	physic_body = PhysicsServer3D::get_singleton()->body_create();
	PhysicsServer3D::get_singleton()->body_set_mode(physic_body, PhysicsServer3D::BodyMode::BODY_MODE_STATIC);
	heightmap_shape = PhysicsServer3D::get_singleton()->heightmap_shape_create();
	Dictionary d;
	d["width"] = heightmap->width;
	d["depth"] = heightmap->height;
	#ifdef REAL_T_IS_DOUBLE
	const float* hdata = (float*)heightmap->data.ptr();
	PackedFloat64Array hdata64;
	int size = heightmap->data.size()/4;
	hdata64.resize(size);
	for(int i=0;i<size;i++){
		hdata64.set(i,hdata[i]);
	}
	d["heights"] = hdata64;
	#else
	d["heights"] = _PackedByteArray_decode_float_array(&heightmap->data);// heightmap->data.to_float32_array();
	#endif
	d["min_height"] = min_height;
	d["max_height"] = max_height;
	Vector3 pos = world_pos + Vector3(grid->region_size_meter,0,grid->region_size_meter)/2;
	Basis basis(Vector3(grid->_chunks->h_scale,0,0), Vector3(0,1,0), Vector3(0,0,grid->_chunks->h_scale) );
	Transform3D transform(basis, pos);
	PhysicsServer3D::get_singleton()->shape_set_data(heightmap_shape, d);
	PhysicsServer3D::get_singleton()->body_add_shape(physic_body, heightmap_shape);
	PhysicsServer3D::get_singleton()->body_set_space(physic_body, grid->space);
	PhysicsServer3D::get_singleton()->body_set_state(physic_body, PhysicsServer3D::BodyState::BODY_STATE_TRANSFORM,transform);
	PhysicsServer3D::get_singleton()->body_set_collision_layer(physic_body,grid->collision_layer);
	PhysicsServer3D::get_singleton()->body_set_collision_mask(physic_body,grid->collision_mask);
	if(grid->physics_material.is_valid()){
		float friction = grid->physics_material->is_rough() ? - grid->physics_material->get_friction() : grid->physics_material->get_friction();
		float bounce = grid->physics_material->is_absorbent() ? - grid->physics_material->get_bounce() : grid->physics_material->get_bounce();
		PhysicsServer3D::get_singleton()->body_set_param(physic_body,PhysicsServer3D::BODY_PARAM_BOUNCE,bounce);
		PhysicsServer3D::get_singleton()->body_set_param(physic_body,PhysicsServer3D::BODY_PARAM_FRICTION,friction);
	}
	has_physic = true;
}

void MRegion::update_physics(){
	std::lock_guard<std::mutex> lock(physics_mutex);
	if(!has_physic){
		return;
	}
	Dictionary d;
	d["width"] = heightmap->width;
	d["depth"] = heightmap->height;
	#ifdef REAL_T_IS_DOUBLE
	const float* hdata = (float*)heightmap->data.ptr();
	PackedFloat64Array hdata64;
	int size = heightmap->data.size()/4;
	hdata64.resize(size);
	for(int i=0;i<size;i++){
		hdata64.set(i,hdata[i]);
	}
	d["heights"] = hdata64;
	#else
	d["heights"] = _PackedByteArray_decode_float_array(&heightmap->data);
	#endif
	d["min_height"] = min_height;
	d["max_height"] = max_height;
	PhysicsServer3D::get_singleton()->shape_set_data(heightmap_shape, d);
}

void MRegion::remove_physics(){
	std::lock_guard<std::mutex> lock(physics_mutex);
	if(!has_physic){
		return;
	}
	PhysicsServer3D::get_singleton()->free(physic_body);
	PhysicsServer3D::get_singleton()->free(heightmap_shape);
	physic_body = RID();
	heightmap_shape = RID();
	has_physic = false;
}

Color MRegion::get_pixel(const uint32_t x, const uint32_t y, const int32_t& index) const {
	return images[index]->get_pixel(x,y);
}

void MRegion::set_pixel(const uint32_t x, const uint32_t y,const Color& color,const int32_t& index){
	if(to_be_remove){
		return;
	}
	images[index]->set_pixel(x,y,color);
}

Color MRegion::get_normal_by_pixel(const uint32_t x, const uint32_t y) const{
	return normals->get_pixel(x,y);
}

void MRegion::set_normal_by_pixel(const uint32_t x, const uint32_t y,const Color& value){
	normals->set_pixel(x,y,value);
}

real_t MRegion::get_height_by_pixel(const uint32_t x, const uint32_t y) const {
	return heightmap->get_pixel_RF(x,y);
}

void MRegion::set_height_by_pixel(const uint32_t x, const uint32_t y,const real_t& value){
	if(to_be_remove){
		return;
	}
	heightmap->set_pixel_RF(x,y,value);
}

real_t MRegion::get_closest_height(Vector3 pos){
	pos.x -= world_pos.x;
	pos.z -= world_pos.z;
	pos /= grid->_chunks->h_scale;
	uint32_t x = (uint32_t)round(pos.x);
	uint32_t y = (uint32_t)round(pos.z);
	return heightmap->get_pixel_RF(x,y);
}

real_t MRegion::get_height_by_pixel_in_layer(const uint32_t x, const uint32_t y) const{
	return heightmap->get_pixel_RF_in_layer(x,y);
}

void MRegion::update_all_dirty_image_texture(){
	for(int i=0; i < images.size(); i++){
		if(images[i]->is_dirty){
			images[i]->update_texture(images[i]->current_scale, true);
		}
	}
}


void MRegion::save_image(Ref<MResource> mres,int index,bool force_save) {
	images[index]->save(mres,force_save);
}

void MRegion::recalculate_normals(bool use_thread,bool use_extra_margin){
	if(grid){
		MPixelRegion calculate_region = normals_pixel_region;
		if(use_extra_margin){
			calculate_region.grow_all_side(grid->grid_pixel_region);
		}
		if(use_thread){
			grid->generate_normals_thread(calculate_region);
		} else {
			grid->generate_normals(calculate_region);
		}
	}
}

void MRegion::refresh_all_uniforms(){
	if(_material_rid.is_valid()){
		RSS->material_set_param(_material_rid,"region_size",grid->region_size_meter);
		RSS->material_set_param(_material_rid,"region_world_position",world_pos);

		RSS->material_set_param(_material_rid,"region_a",(current_image_size-1)/current_image_size);
		RSS->material_set_param(_material_rid,"region_b",0.5/current_image_size);
		RSS->material_set_param(_material_rid,"min_lod",last_lod);
	}
}

void MRegion::make_normals_dirty(){
	if(normals){
		normals->is_dirty = true;
	}
}

void MRegion::make_neighbors_normals_dirty(){
	if(top){
		top->make_normals_dirty();
	}
	if(bottom){
		bottom->make_normals_dirty();
	}
	if(left){
		left->make_normals_dirty();
	}
	if(right){
		right->make_normals_dirty();
	}
}


_FORCE_INLINE_ void MRegion::_calculate_min_max_height(){
	int64_t index = 0;
	int64_t s = heightmap->data.size()/4;
	float* ptr = ((float *)heightmap->data.ptr());
	while (index < s)
	{
		float val = ptr[index];
		if(val > max_height){
			max_height = val;
		}
		if (val < min_height)
		{
			min_height = val;
		}
		index++;
	}
	is_min_max_height_calculated = true;
}

void MRegion::set_data_load_status(bool input){
	is_data_loaded.store(input, std::memory_order_release);
}

bool MRegion::get_data_load_status(){
	return is_data_loaded.load(std::memory_order_acquire);
}

bool MRegion::get_data_load_status_relax(){
	return is_data_loaded.load(std::memory_order_relaxed);
}

void MRegion::correct_edges(){
	correct_left_edge();
	correct_right_edge();
	correct_top_edge();
	correct_bottom_edge();
	correct_bottom_right_corner();
	correct_top_left_corner();
	is_edge_corrected = true;
}

void MRegion::correct_left_edge(){
	if(!left || !left->is_data_loaded_reg_thread || left->is_edge_corrected){
		return;
	}
	for(uint32_t ii=0;ii<images.size();ii++){
		MImage* img = images[ii];
		if(img->name==NORMALS_NAME){
			continue;
		}
		MImage* left_img = left->images[ii];
		if(!img->is_init || !left_img->is_init){
			return;
		}
		left_img->is_dirty = true;
		uint32_t wi = img->width - 1;
		for(uint32_t y=0;y<img->width;y++){
			// index in byte array
			uint32_t left_index = (wi + y*img->width)*img->pixel_size;
			uint32_t index = (y*img->width)*img->pixel_size;
			memcpy(left_img->data.ptrw()+left_index,img->data.ptr()+index,img->pixel_size);
		}
	}
	if(!left->is_min_max_right_considered){
		uint32_t wi = heightmap->width - 1;
		const float* ptr = (const float*)heightmap->data.ptr();
		for(uint32_t y=0;y<wi;y++){
			// index in byte array
			uint32_t index = (y*heightmap->width);
			if(left->min_height > ptr[index]){
				left->min_height = ptr[index];
				
			}
			if(left->max_height < ptr[index]){
				left->max_height = ptr[index];
			}
		}
		left->is_min_max_right_considered = true;
	}
}

void MRegion::correct_right_edge(){
	if(!right || !right->is_data_loaded_reg_thread || right->is_edge_corrected){
		return;
	}
	for(uint32_t ii=0;ii<images.size();ii++){
		MImage* img = images[ii];
		if(img->name==NORMALS_NAME){
			continue;
		}
		MImage* right_img = right->images[ii];
		if(!img->is_init || !right_img->is_init){
			return;
		}
		img->is_dirty = true;
		uint32_t wi = img->width - 1;
		for(uint32_t y=0;y<img->width;y++){
			// index in byte array
			uint32_t index = (wi + y*img->width)*img->pixel_size;
			uint32_t right_index = (y*img->width)*img->pixel_size;
			memcpy(img->data.ptrw()+index,right_img->data.ptr()+right_index,img->pixel_size);
		}
	}
	if(!is_min_max_right_considered){
		uint32_t wi = heightmap->width - 1;
		const float* ptr = (const float*)right->heightmap->data.ptr();
		for(uint32_t y=0;y<wi;y++){ // one last pixel is corner pixel
			// index in byte array
			uint32_t index = (y*heightmap->width);
			if(min_height > ptr[index]){
				min_height = ptr[index];
			}
			if(max_height < ptr[index]){
				max_height = ptr[index];
			}
		}
		is_min_max_right_considered = true;
	}
}

void MRegion::correct_top_edge(){
	if(!top || !top->is_data_loaded_reg_thread || top->is_edge_corrected){
		return;
	}
	for(uint32_t ii=0;ii<images.size();ii++){
		MImage* img = images[ii];
		if(img->name==NORMALS_NAME){
			continue;
		}
		MImage* top_img = top->images[ii];
		if(!img->is_init || !top_img->is_init){
			return;
		}
		top_img->is_dirty = true;
		uint32_t row_size = img->width*img->pixel_size;
		uint32_t top_index = (img->width -1)*img->width*img->pixel_size;
		memcpy(top_img->data.ptrw()+top_index,img->data.ptr(),row_size);
	}
	if(!top->is_min_max_bottom_considered){
		uint32_t wi = heightmap->width - 1;
		const float* ptr = (const float*)heightmap->data.ptr();
		for(uint32_t x=0;x<wi;x++){
			if(top->min_height > ptr[x]){
				top->min_height = ptr[x];
			}
			if(top->max_height < ptr[x]){
				top->max_height = ptr[x];
			}
		}
		top->is_min_max_bottom_considered = true;
	}
}

void MRegion::correct_bottom_edge(){
	if(!bottom || !bottom->is_data_loaded_reg_thread || bottom->is_edge_corrected){
		return;
	}
	for(uint32_t ii=0;ii<images.size();ii++){
		MImage* img = images[ii];
		if(img->name==NORMALS_NAME){
			continue;
		}
		MImage* bottom_img = bottom->images[ii];
		if(!img->is_init || !bottom_img->is_init){
			return;
		}
		img->is_dirty = true;
		uint32_t row_size = (img->width - 1)*img->pixel_size;
		uint32_t index = img->width*(img->width -1)*img->pixel_size;
		memcpy(img->data.ptrw()+index,bottom_img->data.ptr(),row_size);
	}
	if(!is_min_max_bottom_considered){
		uint32_t wi = heightmap->width - 1;
		const float* ptr = (const float*)bottom->heightmap->data.ptr();
		for(uint32_t x=0;x<wi;x++){
			if(min_height > ptr[x]){
				min_height = ptr[x];
			}
			if(max_height < ptr[x]){
				max_height = ptr[x];
			}
		}
		is_min_max_bottom_considered = true;
	}
}

void MRegion::correct_bottom_right_corner(){
	MRegion* br_reg = grid->get_region(pos.x+1,pos.z+1);
	if(!br_reg || !br_reg->is_data_loaded_reg_thread || br_reg->is_edge_corrected){
		return;
	}
	for(uint32_t ii=0;ii<images.size();ii++){
		MImage* img = images[ii];
		if(img->name==NORMALS_NAME){
			continue;
		}
		MImage* bt_img = br_reg->images[ii];
		if(!img->is_init || !bt_img->is_init){
			return;
		}
		uint32_t wi = img->width - 1;
		uint32_t index = (wi + (wi)*img->width)*img->pixel_size;
		memcpy(img->data.ptrw()+index,bt_img->data.ptr(),img->pixel_size);
	}
	{
		float val = ((const float*)br_reg->heightmap->data.ptr())[0];
		if (min_height > val){
			min_height = val;
		}
		if(max_height < val){
			max_height = val;
		}
	}
}

void MRegion::correct_top_left_corner(){
	MRegion* tl_reg = grid->get_region(pos.x-1,pos.z-1);
	if(!tl_reg || !tl_reg->is_data_loaded_reg_thread || tl_reg->is_edge_corrected){
		return;
	}
	for(uint32_t ii=0;ii<images.size();ii++){
		MImage* img = images[ii];
		if(img->name==NORMALS_NAME){
			continue;
		}
		MImage* tl_img = tl_reg->images[ii];
		if(!img->is_init || !tl_img->is_init){
			return;
		}
		uint32_t wi = img->width - 1;
		uint32_t index = (wi + (wi)*img->width)*img->pixel_size;
		memcpy(tl_img->data.ptrw()+index,img->data.ptr(),img->pixel_size);
	}
	{
		float val = ((const float*)heightmap->data.ptr())[0];
		if (tl_reg->min_height > val){
			tl_reg->min_height = val;
		}
		if(tl_reg->max_height < val){
			tl_reg->max_height = val;
		}
	}
}