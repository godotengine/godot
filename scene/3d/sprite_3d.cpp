#include "sprite_3d.h"
#include "scene/scene_string_names.h"
#include "core_string_names.h"


Color SpriteBase3D::_get_color_accum() {

	if (!color_dirty)
		return color_accum;

	if (parent_sprite)
		color_accum=parent_sprite->_get_color_accum();
	else
		color_accum=Color(1,1,1,1);

	color_accum.r*=modulate.r;
	color_accum.g*=modulate.g;
	color_accum.b*=modulate.b;
	color_accum.a*=modulate.a;
	color_dirty=false;
	return color_accum;
}

void SpriteBase3D::_propagate_color_changed() {

	if (color_dirty)
		return;

	color_dirty=true;
	_queue_update();

	for (List<SpriteBase3D*>::Element *E=children.front();E;E=E->next()) {

		E->get()->_propagate_color_changed();
	}
}

void SpriteBase3D::_notification(int p_what) {

	if (p_what==NOTIFICATION_ENTER_TREE) {

		if (!pending_update)
			_im_update();

		Node *parent=get_parent();
		if (parent) {

			parent_sprite=parent->cast_to<SpriteBase3D>();
			if (parent_sprite) {
				pI=parent_sprite->children.push_back(this);
			}
		}
	}

	if (p_what==NOTIFICATION_EXIT_TREE) {


		if (parent_sprite) {

			parent_sprite->children.erase(pI);
			pI=NULL;
			parent_sprite=NULL;
		}
	}

}


void SpriteBase3D::set_centered(bool p_center) {

	centered=p_center;
	_queue_update();

}

bool SpriteBase3D::is_centered() const {

	return centered;
}

void SpriteBase3D::set_offset(const Point2& p_offset) {

	offset=p_offset;
	_queue_update();

}
Point2 SpriteBase3D::get_offset() const {

	return offset;
}

void SpriteBase3D::set_flip_h(bool p_flip) {

	hflip=p_flip;
	_queue_update();
}
bool SpriteBase3D::is_flipped_h() const {

	return hflip;
}

void SpriteBase3D::set_flip_v(bool p_flip) {

	vflip=p_flip;
	_queue_update();
}
bool SpriteBase3D::is_flipped_v() const {

	return vflip;
}



void SpriteBase3D::set_modulate(const Color& p_color) {

	modulate=p_color;
	_propagate_color_changed();
	_queue_update();
}

Color SpriteBase3D::get_modulate() const{

	return modulate;
}


void SpriteBase3D::set_pixel_size(float p_amount) {

	pixel_size=p_amount;
	_queue_update();
}
float SpriteBase3D::get_pixel_size() const {

	return pixel_size;
}

void SpriteBase3D::set_opacity(float p_amount) {

	opacity=p_amount;
	_queue_update();
}
float SpriteBase3D::get_opacity() const {

	return opacity;
}


void SpriteBase3D::set_axis(Vector3::Axis p_axis) {

	axis=p_axis;
	_queue_update();
}
Vector3::Axis SpriteBase3D::get_axis() const {

	return axis;
}



void SpriteBase3D::_im_update() {


	_draw();


	pending_update=false;

	//texture->draw_rect_region(ci,dst_rect,src_rect,modulate);

}

void SpriteBase3D::_queue_update(){

	if  (pending_update)
		return;

	pending_update=true;
	call_deferred(SceneStringNames::get_singleton()->_im_update);
}


AABB SpriteBase3D::get_aabb() const {

	return aabb;
}
DVector<Face3> SpriteBase3D::get_faces(uint32_t p_usage_flags) const {

	return DVector<Face3>();

}

void SpriteBase3D::set_draw_flag(DrawFlags p_flag,bool p_enable) {

	ERR_FAIL_INDEX(p_flag,FLAG_MAX);
	flags[p_flag]=p_enable;
	_queue_update();
}

bool SpriteBase3D::get_draw_flag(DrawFlags p_flag) const{
	ERR_FAIL_INDEX_V(p_flag,FLAG_MAX,false);
	return flags[p_flag];
}

void SpriteBase3D::set_alpha_cut_mode(AlphaCutMode p_mode){

	ERR_FAIL_INDEX(p_mode,3);
	alpha_cut=p_mode;
	_queue_update();

}

SpriteBase3D::AlphaCutMode SpriteBase3D::get_alpha_cut_mode() const{

	return alpha_cut;
}


void SpriteBase3D::_bind_methods() {


	ObjectTypeDB::bind_method(_MD("set_centered","centered"),&SpriteBase3D::set_centered);
	ObjectTypeDB::bind_method(_MD("is_centered"),&SpriteBase3D::is_centered);

	ObjectTypeDB::bind_method(_MD("set_offset","offset"),&SpriteBase3D::set_offset);
	ObjectTypeDB::bind_method(_MD("get_offset"),&SpriteBase3D::get_offset);

	ObjectTypeDB::bind_method(_MD("set_flip_h","flip_h"),&SpriteBase3D::set_flip_h);
	ObjectTypeDB::bind_method(_MD("is_flipped_h"),&SpriteBase3D::is_flipped_h);

	ObjectTypeDB::bind_method(_MD("set_flip_v","flip_v"),&SpriteBase3D::set_flip_v);
	ObjectTypeDB::bind_method(_MD("is_flipped_v"),&SpriteBase3D::is_flipped_v);


	ObjectTypeDB::bind_method(_MD("set_modulate","modulate"),&SpriteBase3D::set_modulate);
	ObjectTypeDB::bind_method(_MD("get_modulate"),&SpriteBase3D::get_modulate);

	ObjectTypeDB::bind_method(_MD("set_opacity","opacity"),&SpriteBase3D::set_opacity);
	ObjectTypeDB::bind_method(_MD("get_opacity"),&SpriteBase3D::get_opacity);

	ObjectTypeDB::bind_method(_MD("set_pixel_size","pixel_size"),&SpriteBase3D::set_pixel_size);
	ObjectTypeDB::bind_method(_MD("get_pixel_size"),&SpriteBase3D::get_pixel_size);

	ObjectTypeDB::bind_method(_MD("set_axis","axis"),&SpriteBase3D::set_axis);
	ObjectTypeDB::bind_method(_MD("get_axis"),&SpriteBase3D::get_axis);

	ObjectTypeDB::bind_method(_MD("set_draw_flag","flag","enabled"),&SpriteBase3D::set_draw_flag);
	ObjectTypeDB::bind_method(_MD("get_draw_flag","flag"),&SpriteBase3D::get_draw_flag);

	ObjectTypeDB::bind_method(_MD("set_alpha_cut_mode","mode"),&SpriteBase3D::set_alpha_cut_mode);
	ObjectTypeDB::bind_method(_MD("get_alpha_cut_mode"),&SpriteBase3D::get_alpha_cut_mode);

	ObjectTypeDB::bind_method(_MD("get_item_rect"),&SpriteBase3D::get_item_rect);

	ObjectTypeDB::bind_method(_MD("_queue_update"),&SpriteBase3D::_queue_update);
	ObjectTypeDB::bind_method(_MD("_im_update"),&SpriteBase3D::_im_update);


	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "centered"), _SCS("set_centered"),_SCS("is_centered"));
	ADD_PROPERTY( PropertyInfo( Variant::VECTOR2, "offset"), _SCS("set_offset"),_SCS("get_offset"));
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "flip_h"), _SCS("set_flip_h"),_SCS("is_flipped_h"));
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "flip_v"), _SCS("set_flip_v"),_SCS("is_flipped_v"));
	ADD_PROPERTY( PropertyInfo( Variant::COLOR, "modulate"), _SCS("set_modulate"),_SCS("get_modulate"));
	ADD_PROPERTY( PropertyInfo( Variant::REAL, "opacity",PROPERTY_HINT_RANGE,"0,1,0.01"), _SCS("set_opacity"),_SCS("get_opacity"));
	ADD_PROPERTY( PropertyInfo( Variant::REAL, "pixel_size",PROPERTY_HINT_RANGE,"0.0001,128,0.0001"), _SCS("set_pixel_size"),_SCS("get_pixel_size"));
	ADD_PROPERTY( PropertyInfo( Variant::INT, "axis",PROPERTY_HINT_ENUM,"X-Axis,Y-Axis,Z-Axis"), _SCS("set_axis"),_SCS("get_axis"));
	ADD_PROPERTYI( PropertyInfo( Variant::BOOL, "flags/transparent"), _SCS("set_draw_flag"),_SCS("get_draw_flag"),FLAG_TRANSPARENT);
	ADD_PROPERTYI( PropertyInfo( Variant::BOOL, "flags/shaded"), _SCS("set_draw_flag"),_SCS("get_draw_flag"),FLAG_SHADED);
	ADD_PROPERTY( PropertyInfo( Variant::INT, "flags/alpha_cut",PROPERTY_HINT_ENUM,"Disabled,Discard,Opaque Pre-Pass"), _SCS("set_alpha_cut_mode"),_SCS("get_alpha_cut_mode"));


	BIND_CONSTANT( FLAG_TRANSPARENT );
	BIND_CONSTANT( FLAG_SHADED );
	BIND_CONSTANT( FLAG_MAX );

	BIND_CONSTANT( ALPHA_CUT_DISABLED );
	BIND_CONSTANT( ALPHA_CUT_DISCARD );
	BIND_CONSTANT( ALPHA_CUT_OPAQUE_PREPASS );



}




SpriteBase3D::SpriteBase3D() {

	color_dirty=true;
	centered=true;
	hflip=false;
	vflip=false;
	parent_sprite=NULL;
	pI=NULL;

	for(int i=0;i<FLAG_MAX;i++)
		flags[i]=i==FLAG_TRANSPARENT;

	axis=Vector3::AXIS_Z;
	pixel_size=0.01;
	modulate=Color(1,1,1,1);
	pending_update=false;
	opacity=1.0;
	immediate = VisualServer::get_singleton()->immediate_create();
	set_base(immediate);
}


SpriteBase3D::~SpriteBase3D() {

	VisualServer::get_singleton()->free(immediate);
}


///////////////////////////////////////////


void Sprite3D::_draw() {

	RID immediate = get_immediate();

	VS::get_singleton()->immediate_clear(immediate);
	if (!texture.is_valid())
		return; //no texuture no life
	Vector2 tsize = texture->get_size();
	if (tsize.x==0 || tsize.y==0)
		return;

	Size2i s;
	Rect2i src_rect;

	if (region) {

		s=region_rect.size;
		src_rect=region_rect;
	} else {
		s = texture->get_size();
		s=s/Size2i(hframes,vframes);

		src_rect.size=s;
		src_rect.pos.x+=(frame%hframes)*s.x;
		src_rect.pos.y+=(frame/hframes)*s.y;

	}

	Point2i ofs=get_offset();
	if (is_centered())
		ofs-=s/2;

	Rect2i dst_rect(ofs,s);


	Rect2 final_rect;
	Rect2 final_src_rect;
	if (!texture->get_rect_region(dst_rect,src_rect,final_rect,final_src_rect))
		return;


	if (final_rect.size.x==0 || final_rect.size.y==0)
		return;

	Color color=_get_color_accum();
	color.a*=get_opacity();

	float pixel_size=get_pixel_size();

	Vector2 vertices[4]={

		(final_rect.pos+Vector2(0,final_rect.size.y)) * pixel_size,
		(final_rect.pos+final_rect.size) * pixel_size,
		(final_rect.pos+Vector2(final_rect.size.x,0)) * pixel_size,
		final_rect.pos * pixel_size,


	};
	Vector2 uvs[4]={
		final_src_rect.pos / tsize,
		(final_src_rect.pos+Vector2(final_src_rect.size.x,0)) / tsize,
		(final_src_rect.pos+final_src_rect.size) / tsize,
		(final_src_rect.pos+Vector2(0,final_src_rect.size.y)) / tsize,
	};

	if (is_flipped_h()) {
		SWAP(uvs[0],uvs[1]);
		SWAP(uvs[2],uvs[3]);
	}
	if (is_flipped_v()) {

		SWAP(uvs[0],uvs[3]);
		SWAP(uvs[1],uvs[2]);
	}


	Vector3 normal;
	int axis = get_axis();
	normal[axis]=1.0;

	RID mat = VS::get_singleton()->material_2d_get(get_draw_flag(FLAG_SHADED),get_draw_flag(FLAG_TRANSPARENT),get_alpha_cut_mode()==ALPHA_CUT_DISCARD,get_alpha_cut_mode()==ALPHA_CUT_OPAQUE_PREPASS);
	VS::get_singleton()->immediate_set_material(immediate,mat);

	VS::get_singleton()->immediate_begin(immediate,VS::PRIMITIVE_TRIANGLE_FAN,texture->get_rid());

	int x_axis = ((axis + 1) % 3);
	int y_axis = ((axis + 2) % 3);

	if (axis!=Vector3::AXIS_Z) {
		SWAP(x_axis,y_axis);

		for(int i=0;i<4;i++) {
			//uvs[i] = Vector2(1.0,1.0)-uvs[i];
			//SWAP(vertices[i].x,vertices[i].y);
			if (axis==Vector3::AXIS_Y) {
				vertices[i].y = - vertices[i].y;
			} else if (axis==Vector3::AXIS_X) {
				vertices[i].x = - vertices[i].x;
			}
		}
	}

	AABB aabb;

	for(int i=0;i<4;i++) {
		VS::get_singleton()->immediate_normal(immediate,normal);
		VS::get_singleton()->immediate_color(immediate,color);
		VS::get_singleton()->immediate_uv(immediate,uvs[i]);

		Vector3 vtx;
		vtx[x_axis]=vertices[i][0];
		vtx[y_axis]=vertices[i][1];
		VS::get_singleton()->immediate_vertex(immediate,vtx);
		if (i==0) {
			aabb.pos=vtx;
			aabb.size=Vector3();
		} else {
			aabb.expand_to(vtx);
		}
	}
	set_aabb(aabb);
	VS::get_singleton()->immediate_end(immediate);


}

void Sprite3D::set_texture(const Ref<Texture>& p_texture) {

	if (p_texture==texture)
		return;
	if (texture.is_valid()) {
		texture->disconnect(CoreStringNames::get_singleton()->changed,this,SceneStringNames::get_singleton()->_queue_update);
	}
	texture=p_texture;
	if (texture.is_valid()) {
		texture->set_flags(texture->get_flags()); //remove repeat from texture, it looks bad in sprites
		texture->connect(CoreStringNames::get_singleton()->changed,this,SceneStringNames::get_singleton()->_queue_update);
	}
	_queue_update();

}

Ref<Texture> Sprite3D::get_texture() const {

	return texture;
}

void Sprite3D::set_region(bool p_region) {

	if (p_region==region)
		return;

	region=p_region;
	_queue_update();
}

bool Sprite3D::is_region() const{

	return region;
}

void Sprite3D::set_region_rect(const Rect2& p_region_rect) {

	bool changed=region_rect!=p_region_rect;
	region_rect=p_region_rect;
	if (region && changed) {
		_queue_update();
	}
}

Rect2 Sprite3D::get_region_rect() const {

	return region_rect;
}

void Sprite3D::set_frame(int p_frame) {

	ERR_FAIL_INDEX(p_frame,vframes*hframes);

	if (frame != p_frame)

	frame=p_frame;
	_queue_update();
	ADD_SIGNAL(MethodInfo("frame_changed"));

}

int Sprite3D::get_frame() const {

	return frame;
}

void Sprite3D::set_vframes(int p_amount) {

	ERR_FAIL_COND(p_amount<1);
	vframes=p_amount;
	_queue_update();
	_change_notify("frame");
}
int Sprite3D::get_vframes() const {

	return vframes;
}

void Sprite3D::set_hframes(int p_amount) {

	ERR_FAIL_COND(p_amount<1);
	hframes=p_amount;
	_queue_update();
	_change_notify("frame");
}
int Sprite3D::get_hframes() const {

	return hframes;
}

Rect2 Sprite3D::get_item_rect() const {

	if (texture.is_null())
		return Rect2(0,0,1,1);
	//if (texture.is_null())
	//	return CanvasItem::get_item_rect();

	Size2i s;

	if (region) {

		s=region_rect.size;
	} else {
		s = texture->get_size();
		s=s/Point2(hframes,vframes);
	}

	Point2i ofs=get_offset();
	if (is_centered())
		ofs-=s/2;

	if (s==Size2(0,0))
		s=Size2(1,1);

	return Rect2(ofs,s);
}

void Sprite3D::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_texture","texture:Texture"),&Sprite3D::set_texture);
	ObjectTypeDB::bind_method(_MD("get_texture:Texture"),&Sprite3D::get_texture);

	ObjectTypeDB::bind_method(_MD("set_region","enabled"),&Sprite3D::set_region);
	ObjectTypeDB::bind_method(_MD("is_region"),&Sprite3D::is_region);

	ObjectTypeDB::bind_method(_MD("set_region_rect","rect"),&Sprite3D::set_region_rect);
	ObjectTypeDB::bind_method(_MD("get_region_rect"),&Sprite3D::get_region_rect);

	ObjectTypeDB::bind_method(_MD("set_frame","frame"),&Sprite3D::set_frame);
	ObjectTypeDB::bind_method(_MD("get_frame"),&Sprite3D::get_frame);

	ObjectTypeDB::bind_method(_MD("set_vframes","vframes"),&Sprite3D::set_vframes);
	ObjectTypeDB::bind_method(_MD("get_vframes"),&Sprite3D::get_vframes);

	ObjectTypeDB::bind_method(_MD("set_hframes","hframes"),&Sprite3D::set_hframes);
	ObjectTypeDB::bind_method(_MD("get_hframes"),&Sprite3D::get_hframes);

	ADD_PROPERTY( PropertyInfo( Variant::OBJECT, "texture", PROPERTY_HINT_RESOURCE_TYPE,"Texture"), _SCS("set_texture"),_SCS("get_texture"));
	ADD_PROPERTY( PropertyInfo( Variant::INT, "vframes"), _SCS("set_vframes"),_SCS("get_vframes"));
	ADD_PROPERTY( PropertyInfo( Variant::INT, "hframes"), _SCS("set_hframes"),_SCS("get_hframes"));
	ADD_PROPERTY( PropertyInfo( Variant::INT, "frame"), _SCS("set_frame"),_SCS("get_frame"));
	ADD_PROPERTY( PropertyInfo( Variant::BOOL, "region"), _SCS("set_region"),_SCS("is_region"));
	ADD_PROPERTY( PropertyInfo( Variant::RECT2, "region_rect"), _SCS("set_region_rect"),_SCS("get_region_rect"));

	ADD_SIGNAL(MethodInfo("frame_changed"));

}

Sprite3D::Sprite3D() {


	region=false;
	frame=0;
	vframes=1;
	hframes=1;

}

////////////////////////////////////////


void AnimatedSprite3D::_draw() {

	RID immediate = get_immediate();
	VS::get_singleton()->immediate_clear(immediate);

	if (!frames.is_valid() || !frames->get_frame_count() || frame<0 || frame>=frames->get_frame_count()) {
		return;
	}

	Ref<Texture> texture = frames->get_frame(frame);
	if (!texture.is_valid())
		return; //no texuture no life
	Vector2 tsize = texture->get_size();
	if (tsize.x==0 || tsize.y==0)
		return;

	Size2i s=tsize;
	Rect2i src_rect;

	src_rect.size=s;

	Point2i ofs=get_offset();
	if (is_centered())
		ofs-=s/2;

	Rect2i dst_rect(ofs,s);


	Rect2 final_rect;
	Rect2 final_src_rect;
	if (!texture->get_rect_region(dst_rect,src_rect,final_rect,final_src_rect))
		return;


	if (final_rect.size.x==0 || final_rect.size.y==0)
		return;

	Color color=_get_color_accum();
	color.a*=get_opacity();

	float pixel_size=get_pixel_size();

	Vector2 vertices[4]={

		(final_rect.pos+Vector2(0,final_rect.size.y)) * pixel_size,
		(final_rect.pos+final_rect.size) * pixel_size,
		(final_rect.pos+Vector2(final_rect.size.x,0)) * pixel_size,
		final_rect.pos * pixel_size,


	};
	Vector2 uvs[4]={
		final_src_rect.pos / tsize,
		(final_src_rect.pos+Vector2(final_src_rect.size.x,0)) / tsize,
		(final_src_rect.pos+final_src_rect.size) / tsize,
		(final_src_rect.pos+Vector2(0,final_src_rect.size.y)) / tsize,
	};

	if (is_flipped_h()) {
		SWAP(uvs[0],uvs[1]);
		SWAP(uvs[2],uvs[3]);
	}
	if (is_flipped_v()) {

		SWAP(uvs[0],uvs[3]);
		SWAP(uvs[1],uvs[2]);
	}


	Vector3 normal;
	int axis = get_axis();
	normal[axis]=1.0;

	RID mat = VS::get_singleton()->material_2d_get(get_draw_flag(FLAG_SHADED),get_draw_flag(FLAG_TRANSPARENT),get_alpha_cut_mode()==ALPHA_CUT_DISCARD,get_alpha_cut_mode()==ALPHA_CUT_OPAQUE_PREPASS);
	VS::get_singleton()->immediate_set_material(immediate,mat);

	VS::get_singleton()->immediate_begin(immediate,VS::PRIMITIVE_TRIANGLE_FAN,texture->get_rid());

	int x_axis = ((axis + 1) % 3);
	int y_axis = ((axis + 2) % 3);

	if (axis!=Vector3::AXIS_Z) {
		SWAP(x_axis,y_axis);

		for(int i=0;i<4;i++) {
			//uvs[i] = Vector2(1.0,1.0)-uvs[i];
			//SWAP(vertices[i].x,vertices[i].y);
			if (axis==Vector3::AXIS_Y) {
				vertices[i].y = - vertices[i].y;
			} else if (axis==Vector3::AXIS_X) {
				vertices[i].x = - vertices[i].x;
			}
		}
	}

	AABB aabb;

	for(int i=0;i<4;i++) {
		VS::get_singleton()->immediate_normal(immediate,normal);
		VS::get_singleton()->immediate_color(immediate,color);
		VS::get_singleton()->immediate_uv(immediate,uvs[i]);

		Vector3 vtx;
		vtx[x_axis]=vertices[i][0];
		vtx[y_axis]=vertices[i][1];
		VS::get_singleton()->immediate_vertex(immediate,vtx);
		if (i==0) {
			aabb.pos=vtx;
			aabb.size=Vector3();
		} else {
			aabb.expand_to(vtx);
		}
	}
	set_aabb(aabb);
	VS::get_singleton()->immediate_end(immediate);

}

void AnimatedSprite3D::_bind_methods(){

	ObjectTypeDB::bind_method(_MD("set_sprite_frames","sprite_frames:SpriteFrames"),&AnimatedSprite3D::set_sprite_frames);
	ObjectTypeDB::bind_method(_MD("get_sprite_frames:Texture"),&AnimatedSprite3D::get_sprite_frames);
	ObjectTypeDB::bind_method(_MD("set_frame","frame"),&AnimatedSprite3D::set_frame);
	ObjectTypeDB::bind_method(_MD("get_frame"),&AnimatedSprite3D::get_frame);

	ADD_PROPERTY( PropertyInfo( Variant::OBJECT, "frames", PROPERTY_HINT_RESOURCE_TYPE,"SpriteFrames"), _SCS("set_sprite_frames"),_SCS("get_sprite_frames"));
	ADD_PROPERTY( PropertyInfo( Variant::INT, "frame"), _SCS("set_frame"),_SCS("get_frame"));

	ADD_SIGNAL(MethodInfo("frame_changed"));

}




void AnimatedSprite3D::set_sprite_frames(const Ref<SpriteFrames>& p_sprite_frames) {


	if (frames==p_sprite_frames)
		return;

	if (frames.is_valid())
		frames->disconnect("changed",this,"_queue_update");
	frames=p_sprite_frames;
	if (frames.is_valid())
		frames->connect("changed",this,"_queue_update");

	if (!frames.is_valid() || frame >=frames->get_frame_count()) {
		frame=0;

	}
	_queue_update();

}

Ref<SpriteFrames> AnimatedSprite3D::get_sprite_frames() const{

	return frames;
}

void AnimatedSprite3D::set_frame(int p_frame){

	if (frames.is_null())
		return;

	ERR_FAIL_INDEX(p_frame,frames->get_frame_count());

	if (frame==p_frame)
		return;

	frame=p_frame;
	_queue_update();
	emit_signal(SceneStringNames::get_singleton()->frame_changed);

}
int AnimatedSprite3D::get_frame() const{

	return frame;
}

Rect2 AnimatedSprite3D::get_item_rect() const {

	if (!frames.is_valid() || !frames->get_frame_count() || frame<0 || frame>=frames->get_frame_count()) {
		return Rect2(0,0,1,1);
	}

	Ref<Texture> t = frames->get_frame(frame);
	if (t.is_null())
		return Rect2(0,0,1,1);
	Size2i s = t->get_size();

	Point2i ofs=get_offset();
	if (is_centered())
		ofs-=s/2;

	if (s==Size2(0,0))
		s=Size2(1,1);

	return Rect2(ofs,s);
}



AnimatedSprite3D::AnimatedSprite3D() {

	frame=0;
}

