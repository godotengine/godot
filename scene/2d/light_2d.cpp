#include "light_2d.h"
#include "servers/visual_server.h"

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

void Light2D::set_item_mask( int p_mask) {

	item_mask=p_mask;
	VS::get_singleton()->canvas_light_set_item_mask(canvas_light,item_mask);

}

int Light2D::get_item_mask() const {

	return item_mask;
}

void Light2D::set_blend_mode( LightBlendMode p_blend_mode ) {

	blend_mode=p_blend_mode;
	VS::get_singleton()->canvas_light_set_blend_mode(canvas_light,VS::CanvasLightBlendMode(blend_mode));
}

Light2D::LightBlendMode Light2D::get_blend_mode() const {

	return blend_mode;
}

void Light2D::set_shadow_enabled( bool p_enabled) {

	shadow=p_enabled;
	VS::get_singleton()->canvas_light_set_shadow_enabled(canvas_light,shadow);

}
bool Light2D::is_shadow_enabled() const {

	return shadow;
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

	ObjectTypeDB::bind_method(_MD("set_item_mask","item_mask"),&Light2D::set_item_mask);
	ObjectTypeDB::bind_method(_MD("get_item_mask"),&Light2D::get_item_mask);

	ObjectTypeDB::bind_method(_MD("set_blend_mode","blend_mode"),&Light2D::set_blend_mode);
	ObjectTypeDB::bind_method(_MD("get_blend_mode"),&Light2D::get_blend_mode);

	ObjectTypeDB::bind_method(_MD("set_shadow_enabled","enabled"),&Light2D::set_shadow_enabled);
	ObjectTypeDB::bind_method(_MD("is_shadow_enabled"),&Light2D::is_shadow_enabled);

	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"enabled"),_SCS("set_enabled"),_SCS("is_enabled"));
	ADD_PROPERTY( PropertyInfo(Variant::OBJECT,"texture",PROPERTY_HINT_RESOURCE_TYPE,"Texture"),_SCS("set_texture"),_SCS("get_texture"));
	ADD_PROPERTY( PropertyInfo(Variant::VECTOR2,"texture_offset"),_SCS("set_texture_offset"),_SCS("get_texture_offset"));
	ADD_PROPERTY( PropertyInfo(Variant::COLOR,"color"),_SCS("set_color"),_SCS("get_color"));
	ADD_PROPERTY( PropertyInfo(Variant::REAL,"height"),_SCS("set_height"),_SCS("get_height"));
	ADD_PROPERTY( PropertyInfo(Variant::INT,"z_range_min",PROPERTY_HINT_RANGE,itos(VS::CANVAS_ITEM_Z_MIN)+","+itos(VS::CANVAS_ITEM_Z_MAX)+",1"),_SCS("set_z_range_min"),_SCS("get_z_range_min"));
	ADD_PROPERTY( PropertyInfo(Variant::INT,"z_range_max",PROPERTY_HINT_RANGE,itos(VS::CANVAS_ITEM_Z_MIN)+","+itos(VS::CANVAS_ITEM_Z_MAX)+",1"),_SCS("set_z_range_max"),_SCS("get_z_range_max"));
	ADD_PROPERTY( PropertyInfo(Variant::INT,"item_mask",PROPERTY_HINT_ALL_FLAGS),_SCS("set_item_mask"),_SCS("get_item_mask"));
	ADD_PROPERTY( PropertyInfo(Variant::INT,"blend_mode",PROPERTY_HINT_ENUM,"Add,Sub,Mul,Dodge,Burn,Lighten,Darken,Overlay,Screen"),_SCS("set_blend_mode"),_SCS("get_blend_mode"));
	ADD_PROPERTY( PropertyInfo(Variant::BOOL,"shadow_enabled"),_SCS("set_shadow_enabled"),_SCS("is_shadow_enabled"));


}

Light2D::Light2D() {

	canvas_light=VisualServer::get_singleton()->canvas_light_create();
	enabled=true;
	shadow=false;
	color=Color(1,1,1);
	height=0;
	z_min=-1024;
	z_max=1024;
	item_mask=1;
	blend_mode=LIGHT_BLEND_ADD;

}

Light2D::~Light2D() {

	VisualServer::get_singleton()->free(canvas_light);
}
