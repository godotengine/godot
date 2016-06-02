#include "patch_9_frame.h"

#include "servers/visual_server.h"

void Patch9Frame::_notification(int p_what) {

	if (p_what==NOTIFICATION_DRAW) {

		if (texture.is_null())
			return;


		Size2 s=get_size();
		RID ci = get_canvas_item();
		VS::get_singleton()->canvas_item_add_style_box(ci,Rect2(Point2(),s),texture->get_rid(),Vector2(margin[MARGIN_LEFT],margin[MARGIN_TOP]),Vector2(margin[MARGIN_RIGHT],margin[MARGIN_BOTTOM]),draw_center,modulate);
//		draw_texture_rect(texture,Rect2(Point2(),s),false,modulate);

/*
		Vector<Point2> points;
		points.resize(4);
		points[0]=Point2(0,0);
		points[1]=Point2(s.x,0);
		points[2]=Point2(s.x,s.y);
		points[3]=Point2(0,s.y);
		Vector<Point2> uvs;
		uvs.resize(4);
		uvs[0]=Point2(0,0);
		uvs[1]=Point2(1,0);
		uvs[2]=Point2(1,1);
		uvs[3]=Point2(0,1);

		VisualServer::get_singleton()->canvas_item_add_primitive(ci,points,Vector<Color>(),uvs,texture->get_rid());
*/
	}
}

Size2 Patch9Frame::get_minimum_size() const {

	return Size2(margin[MARGIN_LEFT]+margin[MARGIN_RIGHT],margin[MARGIN_TOP]+margin[MARGIN_BOTTOM]);
}
void Patch9Frame::_bind_methods() {


	ObjectTypeDB::bind_method(_MD("set_texture","texture"), & Patch9Frame::set_texture );
	ObjectTypeDB::bind_method(_MD("get_texture"), & Patch9Frame::get_texture );
	ObjectTypeDB::bind_method(_MD("set_modulate","modulate"), & Patch9Frame::set_modulate );
	ObjectTypeDB::bind_method(_MD("get_modulate"), & Patch9Frame::get_modulate );
	ObjectTypeDB::bind_method(_MD("set_patch_margin","margin","value"), & Patch9Frame::set_patch_margin );
	ObjectTypeDB::bind_method(_MD("get_patch_margin","margin"), & Patch9Frame::get_patch_margin );
	ObjectTypeDB::bind_method(_MD("set_draw_center","draw_center"), & Patch9Frame::set_draw_center );
	ObjectTypeDB::bind_method(_MD("get_draw_center"), & Patch9Frame::get_draw_center );

	ADD_PROPERTYNZ( PropertyInfo( Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE, "Texture"), _SCS("set_texture"),_SCS("get_texture") );
	ADD_PROPERTYNO( PropertyInfo( Variant::COLOR, "modulate"), _SCS("set_modulate"),_SCS("get_modulate") );
	ADD_PROPERTYNO( PropertyInfo( Variant::BOOL, "draw_center"), _SCS("set_draw_center"),_SCS("get_draw_center") );
	ADD_PROPERTYINZ( PropertyInfo( Variant::INT, "patch_margin/left",PROPERTY_HINT_RANGE,"0,16384,1"), _SCS("set_patch_margin"),_SCS("get_patch_margin"),MARGIN_LEFT );
	ADD_PROPERTYINZ( PropertyInfo( Variant::INT, "patch_margin/top",PROPERTY_HINT_RANGE,"0,16384,1"), _SCS("set_patch_margin"),_SCS("get_patch_margin"),MARGIN_TOP );
	ADD_PROPERTYINZ( PropertyInfo( Variant::INT, "patch_margin/right",PROPERTY_HINT_RANGE,"0,16384,1"), _SCS("set_patch_margin"),_SCS("get_patch_margin"),MARGIN_RIGHT );
	ADD_PROPERTYINZ( PropertyInfo( Variant::INT, "patch_margin/bottom",PROPERTY_HINT_RANGE,"0,16384,1"), _SCS("set_patch_margin"),_SCS("get_patch_margin"),MARGIN_BOTTOM );

}


void Patch9Frame::set_texture(const Ref<Texture>& p_tex) {

	texture=p_tex;
	update();
	//if (texture.is_valid())
	//	texture->set_flags(texture->get_flags()&(~Texture::FLAG_REPEAT)); //remove repeat from texture, it looks bad in sprites
	minimum_size_changed();
}

Ref<Texture> Patch9Frame::get_texture() const {

	return texture;
}

void Patch9Frame::set_modulate(const Color& p_tex) {

	modulate=p_tex;
	update();
}

Color Patch9Frame::get_modulate() const{

	return modulate;
}


void Patch9Frame::set_patch_margin(Margin p_margin,int p_size) {

	ERR_FAIL_INDEX(p_margin,4);
	margin[p_margin]=p_size;
	update();
	minimum_size_changed();
}

int Patch9Frame::get_patch_margin(Margin p_margin) const{

	ERR_FAIL_INDEX_V(p_margin,4,0);
	return margin[p_margin];
}

void Patch9Frame::set_draw_center(bool p_draw) {

	draw_center=p_draw;
	update();
}

bool Patch9Frame::get_draw_center() const{

	return draw_center;
}

Patch9Frame::Patch9Frame() {


	margin[MARGIN_LEFT]=0;
	margin[MARGIN_RIGHT]=0;
	margin[MARGIN_BOTTOM]=0;
	margin[MARGIN_TOP]=0;
	modulate=Color(1,1,1,1);
	set_ignore_mouse(true);
	draw_center=true;
}


Patch9Frame::~Patch9Frame()
{
}


