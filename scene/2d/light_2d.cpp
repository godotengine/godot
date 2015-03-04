#include "light_2d.h"
#include "servers/visual_server.h"

void Light2D::edit_set_pivot(const Point2& p_pivot) {

	set_texture_offset(p_pivot);

}

Point2 Light2D::edit_get_pivot() const {

	return get_texture_offset();
}
bool Light2D::edit_has_pivot() const {

	return true;
}

Rect2 Light2D::get_item_rect() const {

	if (texture.is_null())
		return Rect2(0,0,1,1);

	Size2i s;

	s = texture->get_size();
	Point2i ofs=texture_offset;
	ofs-=s/2;

	if (s==Size2(0,0))
		s=Size2(1,1);

	return Rect2(ofs,s);
}


void Light2D::set_enabled( bool p_enabled) {

	VS::get_singleton()->canvas_light_set_enabled(canvas_light,p_enabled);
	enabled=p_enabled;
}

bool Light2D::is_enabled() const {

	return enabled;
}

void Light2D::set_texture( const Ref<Texture>& p_texture) {

	texture=p_texture;
	if (texture.is_valid())
		VS::get_singleton()->canvas_light_set_texture(canvas_light,texture->get_rid());
	else
		VS::get_singleton()->canvas_light_set_texture(canvas_light,RID());
}

Ref<Texture> Light2D::get_texture() const {

	return texture;
}

void Light2D::set_texture_offset( const Vector2& p_offset) {

	texture_offset=p_offset;
	VS::get_singleton()->canvas_light_set_texture_offset(canvas_light,texture_offset);
}

Vector2 Light2D::get_texture_offset() const {

	return texture_offset;
}

void Light2D::set_color( const Color& p_color) {

	color=p_color;
	VS::get_singleton()->canvas_light_set_color(canvas_light,color);

}
Color Light2D::get_color() const {

	return color;
}

void Light2D::set_height( float p_height) {

	height=p_height;
	VS::get_singleton()->canvas_light_set_height(canvas_light,height);

}
float Light2D::get_height() const {

	return height;
}

void Light2D::set_z_range_min( int p_min_z) {

	z_min=p_min_z;
	VS::get_singleton()->canvas_light_set_z_range(canvas_light,z_min,z_max);

}
int Light2D::get_z_range_min() const {

	return z_min;
}

void Light2D::set_z_range_max( int p_max_z) {

	z_max=p_max_z;
	VS::get_singleton()->canvas_light_set_z_range(canvas_light,z_min,z_max);

}
int Light2D::get_z_range_max() const {

	return z_max;
}

void Light2D::set_layer_range_min( int p_min_layer) {

	layer_min=p_min_layer;
	VS::get_singleton()->canvas_light_set_layer_range(canvas_light,layer_min,layer_max);

}
int Light2D::get_layer_range_min() const {

	return layer_min;
}

void Light2D::set_layer_range_max( int p_max_layer) {

	layer_max=p_max_layer;
	VS::get_singleton()->canvas_light_set_layer_range(canvas_light,layer_min,layer_max);

}
int Light2D::get_layer_range_max() const {

	return layer_max;
}

void Light2D::set_item_mask( int p_mask) {

	item_mask=p_mask;
	VS::get_singleton()->canvas_light_set_item_mask(canvas_light,item_mask);

}

int Light2D::get_item_mask() const {

	return item_mask;
}

void Light2D::set_subtract_mode( bool p_enable ) {

	subtract_mode=p_enable;
	VS::get_singleton()->canvas_light_set_subtract_mode(canvas_light,p_enable);
}

bool Light2D::get_subtract_mode() const {

	return subtract_mode;
}

void Light2D::set_shadow_enabled( bool p_enabled) {

	shadow=p_enabled;
	VS::get_singleton()->canvas_light_set_shadow_enabled(canvas_light,shadow);

}
bool Light2D::is_shadow_enabled() const {

	return shadow;
}

void Light2D::set_shadow_buffer_size( int p_size ) {

	shadow_buffer_size=p_size;
	VS::get_singleton()->canvas_light_set_shadow_buffer_size(canvas_light,shadow_buffer_size);
}

int Light2D::get_shadow_buffer_size() const {

	return shadow_buffer_size;
}

void Light2D::set_shadow_esm_multiplier( float p_multiplier) {

	shadow_esm_multiplier=p_multiplier;
	VS::get_singleton()->canvas_light_set_shadow_esm_multiplier(canvas_light,p_multiplier);
}

float Light2D::get_shadow_esm_multiplier() const{

	return shadow_esm_multiplier;
}


void Light2D::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE) {

		VS::get_singleton()->canvas_light_attach_to_canvas( canvas_light, get_canvas() );
	}

	if (p_what==NOTIFICATION_TRANSFORM_CHANGED) {

		VS::get_singleton()->canvas_light_set_transform( canvas_light, get_global_transform());
	}

	if (p_what==NOTIFICATION_EXIT_TREE) {

		VS::get_singleton()->canvas_light_attach_to_canvas( canvas_light, RID() );
	}

}

void Light2D::_bind_methods() {


	ObjectTypeDB::bind_method(_MD("set_enabled","enabled"),&Light2D::set_enabled);
	ObjectTypeDB::bind_method(_MD("is_enabled"),&Light2D::is_enabled);

	ObjectTypeDB::bind_method(_MD("set_texture","texture"),&Light2D::set_texture);
	ObjectTypeDB::bind_method(_MD("get_texture"),&Light2D::get_texture);

	ObjectTypeDB::bind_method(_MD("set_texture_offset","texture_offset"),&Light2D::set_texture_offset);
	ObjectTypeDB::bind_method(_MD("get_texture_offset"),&Light2D::get_texture_offset);

	ObjectTypeDB::bind_method(_MD("set_color","color"),&Light2D::set_color);
	ObjectTypeDB::bind_method(_MD("get_color"),&Light2D::get_color);

	ObjectTypeDB::bind_method(_MD("set_height","height"),&Light2D::set_height);
	ObjectTypeDB::bind_method(_MD("get_height"),&Light2D::get_height);

	ObjectTypeDB::bind_method(_MD("set_z_range_min","z"),&Light2D::set_z_range_min);
	ObjectTypeDB::bind_method(_MD("get_z_range_min"),&Light2D::get_z_range_min);

	ObjectTypeDB::bind_method(_MD("set_z_range_max","z"),&Light2D::set_z_range_max);
	ObjectTypeDB::bind_method(_MD("get_z_range_max"),&Light2D::get_z_range_max);

	ObjectTypeDB::bind_method(_MD("set_layer_range_min","layer"),&Light2D::set_layer_range_min);
	ObjectTypeDB::bind_method(_MD("get_layer_range_min"),&Light2D::get_layer_range_min);

	ObjectTypeDB::bind_method(_MD("set_layer_range_max","layer"),&Light2D::set_layer_range_max);
	ObjectTypeDB::bind_method(_MD("get_layer_range_max"),&Light2D::get_layer_range_max);


	ObjectTypeDB::bind_method(_MD("set_item_mask","item_mask"),&Light2D::set_item_mask);
	ObjectTypeDB::bind_method(_MD("get_item_mask"),&Light2D::get_item_mask);

	ObjectTypeDB::bind_method(_MD("set_subtract_mode","enable"),&Light2D::set_subtract_mode);
	ObjectTypeDB::bind_method(_MD("get_subtract_mode"),&Light2D::get_subtract_mode);

	ObjectTypeDB::bind_method(_MD("set_shadow_enabled","enabled"),&Light2D::set_shadow_enabled);
	ObjectTypeDB::bind_method(_MD("is_shadow_enabled"),&Light2D::is_shadow_enabled);

	ObjectTypeDB::bind_method(_MD("set_shadow_buffer_size","size"),&Light2D::set_shadow_buffer_size);
	ObjectTypeDB::bind_method(_MD("get_shadow_buffer_size"),&Light2D::get_shadow_buffer_size);

	ObjectTypeDB::bind_method(_MD("set_shadow_esm_multiplier","multiplier"),&Light2D::set_shadow_esm_multiplier);
	ObjectTypeDB::bind_method(_MD("get_shadow_esm_multiplier"),&Light2D::get_shadow_esm_multiplier);

	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"enabled"),_SCS("set_enabled"),_SCS("is_enabled"));
	ADD_PROPERTY( PropertyInfo(Variant::OBJECT,"texture",PROPERTY_HINT_RESOURCE_TYPE,"Texture"),_SCS("set_texture"),_SCS("get_texture"));
	ADD_PROPERTY( PropertyInfo(Variant::VECTOR2,"offset"),_SCS("set_texture_offset"),_SCS("get_texture_offset"));
	ADD_PROPERTY( PropertyInfo(Variant::COLOR,"color"),_SCS("set_color"),_SCS("get_color"));
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"subtract"),_SCS("set_subtract_mode"),_SCS("get_subtract_mode"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"range/height"),_SCS("set_height"),_SCS("get_height"));
	ADD_PROPERTY( PropertyInfo(Variant::INT,"range/z_min",PROPERTY_HINT_RANGE,itos(VS::CANVAS_ITEM_Z_MIN)+","+itos(VS::CANVAS_ITEM_Z_MAX)+",1"),_SCS("set_z_range_min"),_SCS("get_z_range_min"));
	ADD_PROPERTY( PropertyInfo(Variant::INT,"range/z_max",PROPERTY_HINT_RANGE,itos(VS::CANVAS_ITEM_Z_MIN)+","+itos(VS::CANVAS_ITEM_Z_MAX)+",1"),_SCS("set_z_range_max"),_SCS("get_z_range_max"));
	ADD_PROPERTY( PropertyInfo(Variant::INT,"range/layer_min",PROPERTY_HINT_RANGE,"-512,512,1"),_SCS("set_layer_range_min"),_SCS("get_layer_range_min"));
	ADD_PROPERTY( PropertyInfo(Variant::INT,"range/layer_max",PROPERTY_HINT_RANGE,"-512,512,1"),_SCS("set_layer_range_max"),_SCS("get_layer_range_max"));
	ADD_PROPERTY( PropertyInfo(Variant::INT,"range/item_mask",PROPERTY_HINT_ALL_FLAGS),_SCS("set_item_mask"),_SCS("get_item_mask"));
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"shadow/enabled"),_SCS("set_shadow_enabled"),_SCS("is_shadow_enabled"));
	ADD_PROPERTY( PropertyInfo(Variant::INT,"shadow/buffer_size",PROPERTY_HINT_RANGE,"32,16384,1"),_SCS("set_shadow_buffer_size"),_SCS("get_shadow_buffer_size"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"shadow/esm_multiplier",PROPERTY_HINT_RANGE,"1,4096,0.1"),_SCS("set_shadow_esm_multiplier"),_SCS("get_shadow_esm_multiplier"));


}

Light2D::Light2D() {

	canvas_light=VisualServer::get_singleton()->canvas_light_create();
	enabled=true;
	shadow=false;
	color=Color(1,1,1);
	height=0;
	z_min=-1024;
	z_max=1024;
	layer_min=0;
	layer_max=0;
	item_mask=1;
	subtract_mode=false;
	shadow_buffer_size=2048;
	shadow_esm_multiplier=80;

}

Light2D::~Light2D() {

	VisualServer::get_singleton()->free(canvas_light);
}
