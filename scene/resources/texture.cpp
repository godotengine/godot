/*************************************************************************/
/*  texture.cpp                                                          */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/*                                                                       */
/* Permission is hereby granted, free of charge, to any person obtaining */
/* a copy of this software and associated documentation files (the       */
/* "Software"), to deal in the Software without restriction, including   */
/* without limitation the rights to use, copy, modify, merge, publish,   */
/* distribute, sublicense, and/or sell copies of the Software, and to    */
/* permit persons to whom the Software is furnished to do so, subject to */
/* the following conditions:                                             */
/*                                                                       */
/* The above copyright notice and this permission notice shall be        */
/* included in all copies or substantial portions of the Software.       */
/*                                                                       */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,       */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF    */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.*/
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY  */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,  */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE     */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                */
/*************************************************************************/
#include "texture.h"
#include "io/image_loader.h"
#include "core/os/os.h"



Size2 Texture::get_size() const {

	return Size2(get_width(),get_height());
}


void Texture::draw(RID p_canvas_item, const Point2& p_pos, const Color& p_modulate, bool p_transpose) const {

	VisualServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item,Rect2( p_pos, get_size()),get_rid(),false,p_modulate,p_transpose);

}
void Texture::draw_rect(RID p_canvas_item,const Rect2& p_rect, bool p_tile,const Color& p_modulate, bool p_transpose) const {

	VisualServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item,p_rect,get_rid(),p_tile,p_modulate,p_transpose);

}
void Texture::draw_rect_region(RID p_canvas_item,const Rect2& p_rect, const Rect2& p_src_rect,const Color& p_modulate, bool p_transpose) const{

	VisualServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item,p_rect,get_rid(),p_src_rect,p_modulate,p_transpose);
}

bool Texture::get_rect_region(const Rect2& p_rect, const Rect2& p_src_rect,Rect2& r_rect,Rect2& r_src_rect) const {

	r_rect=p_rect;
	r_src_rect=p_src_rect;

	return true;
}

void Texture::_bind_methods() {

	ClassDB::bind_method(_MD("get_width"),&Texture::get_width);
	ClassDB::bind_method(_MD("get_height"),&Texture::get_height);
	ClassDB::bind_method(_MD("get_size"),&Texture::get_size);
	ClassDB::bind_method(_MD("has_alpha"),&Texture::has_alpha);
	ClassDB::bind_method(_MD("set_flags","flags"),&Texture::set_flags);
	ClassDB::bind_method(_MD("get_flags"),&Texture::get_flags);
	ClassDB::bind_method(_MD("draw","canvas_item","pos","modulate","transpose"),&Texture::draw,DEFVAL(Color(1,1,1)),DEFVAL(false));
	ClassDB::bind_method(_MD("draw_rect","canvas_item","rect","tile","modulate","transpose"),&Texture::draw_rect,DEFVAL(Color(1,1,1)),DEFVAL(false));
	ClassDB::bind_method(_MD("draw_rect_region","canvas_item","rect","src_rect","modulate","transpose"),&Texture::draw_rect_region,DEFVAL(Color(1,1,1)),DEFVAL(false));

	BIND_CONSTANT( FLAG_MIPMAPS );
	BIND_CONSTANT( FLAG_REPEAT );
	BIND_CONSTANT( 	FLAG_FILTER );
	BIND_CONSTANT( FLAG_VIDEO_SURFACE );
	BIND_CONSTANT( FLAGS_DEFAULT );
	BIND_CONSTANT( FLAG_ANISOTROPIC_FILTER );
	BIND_CONSTANT( FLAG_CONVERT_TO_LINEAR );
	BIND_CONSTANT( FLAG_MIRRORED_REPEAT );

}

Texture::Texture() {



}



/////////////////////





void ImageTexture::reload_from_file() {

	String path=get_path();
	if (!path.is_resource_file())
		return;

	uint32_t flags = get_flags();
	Image img;

	Error err = ImageLoader::load_image(path,&img);
	ERR_FAIL_COND(err!=OK);

	create_from_image(img,flags);

}

bool ImageTexture::_set(const StringName& p_name, const Variant& p_value) {

	if (p_name=="image" && p_value.get_type()==Variant::IMAGE)
		create_from_image( p_value,flags );
	else if (p_name=="flags")
		if (w*h==0)
			flags=p_value;
		else
			set_flags(p_value);
	else if (p_name=="size") {
		Size2 s = p_value;
		w=s.width;
		h=s.height;
		VisualServer::get_singleton()->texture_set_size_override(texture,w,h);
	} else if (p_name=="storage") {
		storage=Storage(p_value.operator int());
	} else if (p_name=="lossy_quality") {
		lossy_storage_quality=p_value;
	} else if (p_name=="_data") {
		_set_data(p_value);
	} else
		return false;

	return true;

}

bool ImageTexture::_get(const StringName& p_name,Variant &r_ret) const {


	if (p_name=="image_data") {

	} else if (p_name=="image")
		r_ret= get_data();
	else if (p_name=="flags")
		r_ret= flags;
	else if (p_name=="size")
		r_ret=Size2(w,h);
	else if (p_name=="storage")
		r_ret= storage;
	else if (p_name=="lossy_quality")
		r_ret= lossy_storage_quality;
	else
		return false;

	return true;
}




void ImageTexture::_get_property_list( List<PropertyInfo> *p_list) const {


	PropertyHint img_hint=PROPERTY_HINT_NONE;
	if (storage==STORAGE_COMPRESS_LOSSY) {
		img_hint=PROPERTY_HINT_IMAGE_COMPRESS_LOSSY;
	} else if (storage==STORAGE_COMPRESS_LOSSLESS) {
		img_hint=PROPERTY_HINT_IMAGE_COMPRESS_LOSSLESS;
	}



	p_list->push_back( PropertyInfo( Variant::INT, "flags", PROPERTY_HINT_FLAGS,"Mipmaps,Repeat,Filter,Anisotropic,sRGB,Mirrored Repeat") );
	p_list->push_back( PropertyInfo( Variant::IMAGE, "image", img_hint,String::num(lossy_storage_quality)) );
	p_list->push_back( PropertyInfo( Variant::VECTOR2, "size",PROPERTY_HINT_NONE, ""));
	p_list->push_back( PropertyInfo( Variant::INT, "storage", PROPERTY_HINT_ENUM,"Uncompressed,Compress Lossy,Compress Lossless"));
	p_list->push_back( PropertyInfo( Variant::REAL, "lossy_quality", PROPERTY_HINT_RANGE,"0.0,1.0,0.01" ));
}

void ImageTexture::_reload_hook(const RID& p_hook) {

	String path=get_path();
	if (!path.is_resource_file())
		return;

	Image img;
	Error err = ImageLoader::load_image(path,&img);

	ERR_FAIL_COND(err!=OK);


	VisualServer::get_singleton()->texture_set_data(texture,img);

	_change_notify();
}

void ImageTexture::create(int p_width, int p_height,Image::Format p_format,uint32_t p_flags) {

	flags=p_flags;
	VisualServer::get_singleton()->texture_allocate(texture,p_width, p_height, p_format, p_flags);
	format=p_format;
	w=p_width;
	h=p_height;

}
void ImageTexture::create_from_image(const Image& p_image,  uint32_t p_flags) {

	flags=p_flags;
	w=p_image.get_width();
	h=p_image.get_height();
	format=p_image.get_format();

	VisualServer::get_singleton()->texture_allocate(texture,p_image.get_width(),p_image.get_height(), p_image.get_format(), p_flags);
	VisualServer::get_singleton()->texture_set_data(texture,p_image);
	_change_notify();
}

void ImageTexture::set_flags(uint32_t p_flags) {



/*	uint32_t cube = flags & FLAG_CUBEMAP;
	if (flags == p_flags&cube)
		return;

	flags=p_flags|cube;	*/
	flags=p_flags;
	VisualServer::get_singleton()->texture_set_flags(texture,p_flags);

}

uint32_t ImageTexture::get_flags() const {

	return ImageTexture::flags;
}

Image::Format ImageTexture::get_format() const {

	return format;
}

void ImageTexture::load(const String& p_path) {

    Image img;
    img.load(p_path);
    create_from_image(img);

}

void ImageTexture::set_data(const Image& p_image) {

	VisualServer::get_singleton()->texture_set_data(texture,p_image);

	_change_notify();
}

void ImageTexture::_resource_path_changed() {

	String path=get_path();
}

Image ImageTexture::get_data() const {

	return VisualServer::get_singleton()->texture_get_data(texture);
}

int ImageTexture::get_width() const {

	return w;
}

int ImageTexture::get_height() const {

	return h;
}


RID ImageTexture::get_rid() const {

	return texture;
}

void ImageTexture::fix_alpha_edges() {

	if (format==Image::FORMAT_RGBA8 /*&& !(flags&FLAG_CUBEMAP)*/) {

		Image img = get_data();
		img.fix_alpha_edges();
		set_data(img);
	}
}

void ImageTexture::premultiply_alpha() {

	if (format==Image::FORMAT_RGBA8 /*&& !(flags&FLAG_CUBEMAP)*/) {

		Image img = get_data();
		img.premultiply_alpha();
		set_data(img);
	}
}

void ImageTexture::normal_to_xy() {

	Image img = get_data();
	img.normalmap_to_xy();
	create_from_image(img,flags);
}

void ImageTexture::shrink_x2_and_keep_size() {

	Size2 sizeov=get_size();
	Image img = get_data();
	img.resize(img.get_width()/2,img.get_height()/2,Image::INTERPOLATE_BILINEAR);
	create_from_image(img,flags);
	set_size_override(sizeov);

}

bool ImageTexture::has_alpha() const {

	return ( format==Image::FORMAT_LA8 || format==Image::FORMAT_RGBA8 );
}


void ImageTexture::draw(RID p_canvas_item, const Point2& p_pos, const Color& p_modulate, bool p_transpose) const {

	if ((w|h)==0)
		return;
	VisualServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item,Rect2( p_pos, Size2(w,h)),texture,false,p_modulate,p_transpose);

}
void ImageTexture::draw_rect(RID p_canvas_item,const Rect2& p_rect, bool p_tile,const Color& p_modulate, bool p_transpose) const {

	if ((w|h)==0)
		return;
	VisualServer::get_singleton()->canvas_item_add_texture_rect(p_canvas_item,p_rect,texture,p_tile,p_modulate,p_transpose);

}
void ImageTexture::draw_rect_region(RID p_canvas_item,const Rect2& p_rect, const Rect2& p_src_rect,const Color& p_modulate, bool p_transpose) const{

	if ((w|h)==0)
		return;
	VisualServer::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item,p_rect,texture,p_src_rect,p_modulate,p_transpose);
}

void ImageTexture::set_size_override(const Size2& p_size) {

	Size2 s=p_size;
	if (s.x!=0)
		w=s.x;
	if (s.y!=0)
		h=s.y;
	VisualServer::get_singleton()->texture_set_size_override(texture,w,h);
}

void ImageTexture::set_path(const String& p_path,bool p_take_over) {

	if (texture.is_valid()) {
		VisualServer::get_singleton()->texture_set_path(texture,p_path);
	}

	Resource::set_path(p_path,p_take_over);
}


void ImageTexture::set_storage(Storage p_storage) {

	storage=p_storage;
}

ImageTexture::Storage ImageTexture::get_storage() const {

	return storage;
}

void ImageTexture::set_lossy_storage_quality(float p_lossy_storage_quality) {

	lossy_storage_quality=p_lossy_storage_quality;
}

float ImageTexture::get_lossy_storage_quality() const {

	return lossy_storage_quality;
}

void ImageTexture::_set_data(Dictionary p_data) {

	Image img = p_data["image"];
	uint32_t flags = p_data["flags"];

	create_from_image(img, flags);

	set_storage(Storage(p_data["storage"].operator int()));
	set_lossy_storage_quality(p_data["lossy_quality"]);

	set_size_override(p_data["size"]);
};

void ImageTexture::_bind_methods() {

	ClassDB::bind_method(_MD("create","width","height","format","flags"),&ImageTexture::create,DEFVAL(FLAGS_DEFAULT));
	ClassDB::bind_method(_MD("create_from_image","image","flags"),&ImageTexture::create_from_image,DEFVAL(FLAGS_DEFAULT));
	ClassDB::bind_method(_MD("get_format"),&ImageTexture::get_format);
	ClassDB::bind_method(_MD("load","path"),&ImageTexture::load);
	ClassDB::bind_method(_MD("set_data","image"),&ImageTexture::set_data);
	ClassDB::bind_method(_MD("get_data","cube_side"),&ImageTexture::get_data);
	ClassDB::bind_method(_MD("set_storage","mode"),&ImageTexture::set_storage);
	ClassDB::bind_method(_MD("get_storage"),&ImageTexture::get_storage);
	ClassDB::bind_method(_MD("set_lossy_storage_quality","quality"),&ImageTexture::set_lossy_storage_quality);
	ClassDB::bind_method(_MD("get_lossy_storage_quality"),&ImageTexture::get_lossy_storage_quality);
	ClassDB::bind_method(_MD("fix_alpha_edges"),&ImageTexture::fix_alpha_edges);
	ClassDB::bind_method(_MD("premultiply_alpha"),&ImageTexture::premultiply_alpha);
	ClassDB::bind_method(_MD("normal_to_xy"),&ImageTexture::normal_to_xy);
	ClassDB::bind_method(_MD("shrink_x2_and_keep_size"),&ImageTexture::shrink_x2_and_keep_size);

	ClassDB::bind_method(_MD("set_size_override","size"),&ImageTexture::set_size_override);
	ClassDB::set_method_flags(get_class_static(),_SCS("fix_alpha_edges"),METHOD_FLAGS_DEFAULT|METHOD_FLAG_EDITOR);
	ClassDB::set_method_flags(get_class_static(),_SCS("premultiply_alpha"),METHOD_FLAGS_DEFAULT|METHOD_FLAG_EDITOR);
	ClassDB::set_method_flags(get_class_static(),_SCS("normal_to_xy"),METHOD_FLAGS_DEFAULT|METHOD_FLAG_EDITOR);
	ClassDB::set_method_flags(get_class_static(),_SCS("shrink_x2_and_keep_size"),METHOD_FLAGS_DEFAULT|METHOD_FLAG_EDITOR);
	ClassDB::bind_method(_MD("_reload_hook","rid"),&ImageTexture::_reload_hook);


	BIND_CONSTANT( STORAGE_RAW );
	BIND_CONSTANT( STORAGE_COMPRESS_LOSSY  );
	BIND_CONSTANT( STORAGE_COMPRESS_LOSSLESS );


}

ImageTexture::ImageTexture() {

	w=h=0;
	flags=FLAGS_DEFAULT;
	texture = VisualServer::get_singleton()->texture_create();
	storage = STORAGE_RAW;
	lossy_storage_quality=0.7;


}


ImageTexture::~ImageTexture() {

	VisualServer::get_singleton()->free( texture );
}


//////////////////////////////////////////


int AtlasTexture::get_width() const {

	if (region.size.width==0) {
		if (atlas.is_valid())
			return atlas->get_width();
		return 1;
	} else {
		return region.size.width+margin.size.width;
	}
}
int AtlasTexture::get_height() const {

	if (region.size.height==0) {
		if (atlas.is_valid())
			return atlas->get_height();
		return 1;
	} else {
		return region.size.height+margin.size.height;
	}
}
RID AtlasTexture::get_rid() const {

	if (atlas.is_valid())
		return atlas->get_rid();

	return RID();
}

bool AtlasTexture::has_alpha() const {

	if (atlas.is_valid())
		return atlas->has_alpha();

	return false;
}

void AtlasTexture::set_flags(uint32_t p_flags) {

	if (atlas.is_valid())
		atlas->set_flags(p_flags);

}


uint32_t AtlasTexture::get_flags() const{

	if (atlas.is_valid())
		return atlas->get_flags();

	return 0;
}

void AtlasTexture::set_atlas(const Ref<Texture>& p_atlas){

	if (atlas==p_atlas)
		return;
	atlas=p_atlas;
	emit_changed();
	emit_signal("atlas_changed");
}
Ref<Texture> AtlasTexture::get_atlas() const{

	return atlas;
}

void AtlasTexture::set_region(const Rect2& p_region) {

	region=p_region;
	emit_changed();

}

Rect2 AtlasTexture::get_region() const {

	return region;
}

void AtlasTexture::set_margin(const Rect2& p_margin) {

	margin=p_margin;
	emit_changed();

}

Rect2 AtlasTexture::get_margin() const {

	return margin;
}

void AtlasTexture::_bind_methods() {

	ClassDB::bind_method(_MD("set_atlas","atlas:Texture"),&AtlasTexture::set_atlas);
	ClassDB::bind_method(_MD("get_atlas:Texture"),&AtlasTexture::get_atlas);

	ClassDB::bind_method(_MD("set_region","region"),&AtlasTexture::set_region);
	ClassDB::bind_method(_MD("get_region"),&AtlasTexture::get_region);

	ClassDB::bind_method(_MD("set_margin","margin"),&AtlasTexture::set_margin);
	ClassDB::bind_method(_MD("get_margin"),&AtlasTexture::get_margin);

	ADD_SIGNAL(MethodInfo("atlas_changed"));

	ADD_PROPERTY( PropertyInfo( Variant::OBJECT, "atlas", PROPERTY_HINT_RESOURCE_TYPE,"Texture"), _SCS("set_atlas"),_SCS("get_atlas") );
	ADD_PROPERTY( PropertyInfo( Variant::RECT2, "region"), _SCS("set_region"),_SCS("get_region") );
	ADD_PROPERTY( PropertyInfo( Variant::RECT2, "margin"), _SCS("set_margin"),_SCS("get_margin") );

}




void AtlasTexture::draw(RID p_canvas_item, const Point2& p_pos, const Color& p_modulate, bool p_transpose) const {

	Rect2 rc=region;

	if (!atlas.is_valid())
		return;

	if (rc.size.width==0) {
		rc.size.width=atlas->get_width();
	}

	if (rc.size.height==0) {
		rc.size.height=atlas->get_height();
	}

	VS::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item,Rect2(p_pos+margin.pos,rc.size),atlas->get_rid(),rc,p_modulate,p_transpose);
}

void AtlasTexture::draw_rect(RID p_canvas_item,const Rect2& p_rect, bool p_tile,const Color& p_modulate, bool p_transpose) const {

	Rect2 rc=region;

	if (!atlas.is_valid())
		return;

	if (rc.size.width==0) {
		rc.size.width=atlas->get_width();
	}

	if (rc.size.height==0) {
		rc.size.height=atlas->get_height();
	}

	Vector2 scale = p_rect.size / (region.size+margin.size);
	Rect2 dr( p_rect.pos+margin.pos*scale,rc.size*scale  );

	VS::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item,dr,atlas->get_rid(),rc,p_modulate,p_transpose);

}
void AtlasTexture::draw_rect_region(RID p_canvas_item,const Rect2& p_rect, const Rect2& p_src_rect,const Color& p_modulate, bool p_transpose) const {

	//this might not necesarily work well if using a rect, needs to be fixed properly
	Rect2 rc=region;

	if (!atlas.is_valid())
		return;

	Rect2 src=p_src_rect;
	src.pos+=(rc.pos-margin.pos);
	Rect2 src_c = rc.clip(src);
	if (src_c.size==Size2())
		return;
	Vector2 ofs = (src_c.pos-src.pos);

	Vector2 scale = p_rect.size / p_src_rect.size;
    if(scale.x < 0)
    {
        float mx = (margin.size.width - margin.pos.x);
        mx -= margin.pos.x;
        ofs.x = -(ofs.x + mx);
    }
    if(scale.y < 0)
    {
        float my = margin.size.height - margin.pos.y;
        my -= margin.pos.y;
        ofs.y = -(ofs.y + my);
    }
	Rect2 dr( p_rect.pos+ofs*scale,src_c.size*scale );

	VS::get_singleton()->canvas_item_add_texture_rect_region(p_canvas_item,dr,atlas->get_rid(),src_c,p_modulate,p_transpose);
}

bool AtlasTexture::get_rect_region(const Rect2& p_rect, const Rect2& p_src_rect,Rect2& r_rect,Rect2& r_src_rect) const {

	Rect2 rc=region;

	if (!atlas.is_valid())
		return false;

	Rect2 src=p_src_rect;
	src.pos+=(rc.pos-margin.pos);
	Rect2 src_c = rc.clip(src);
	if (src_c.size==Size2())
		return false;
	Vector2 ofs = (src_c.pos-src.pos);

	Vector2 scale = p_rect.size / p_src_rect.size;
    if(scale.x < 0)
    {
	float mx = (margin.size.width - margin.pos.x);
	mx -= margin.pos.x;
	ofs.x = -(ofs.x + mx);
    }
    if(scale.y < 0)
    {
	float my = margin.size.height - margin.pos.y;
	my -= margin.pos.y;
	ofs.y = -(ofs.y + my);
    }
	Rect2 dr( p_rect.pos+ofs*scale,src_c.size*scale );



	r_rect=dr;
	r_src_rect=src_c;
	return true;
}


AtlasTexture::AtlasTexture() {


}


//////////////////////////////////////////


int LargeTexture::get_width() const {

	return size.width;
}
int LargeTexture::get_height() const {

	return size.height;
}
RID LargeTexture::get_rid() const {

	return RID();
}

bool LargeTexture::has_alpha() const {

	for(int i=0;i<pieces.size();i++) {
		if (pieces[i].texture->has_alpha())
			return true;
	}

	return false;
}

void LargeTexture::set_flags(uint32_t p_flags) {

	for(int i=0;i<pieces.size();i++) {
		pieces[i].texture->set_flags(p_flags);
	}

}


uint32_t LargeTexture::get_flags() const{

	if (pieces.size())
		return pieces[0].texture->get_flags();

	return 0;
}


int LargeTexture::add_piece(const Point2& p_offset,const Ref<Texture>& p_texture) {

	ERR_FAIL_COND_V(p_texture.is_null(), -1);
	Piece p;
	p.offset=p_offset;
	p.texture=p_texture;
	pieces.push_back(p);

	return pieces.size() - 1;
}

void LargeTexture::set_piece_offset(int p_idx, const Point2& p_offset) {

	ERR_FAIL_INDEX(p_idx, pieces.size());
	pieces[p_idx].offset = p_offset;
};

void LargeTexture::set_piece_texture(int p_idx, const Ref<Texture>& p_texture) {

	ERR_FAIL_INDEX(p_idx, pieces.size());
	pieces[p_idx].texture = p_texture;
};



void LargeTexture::set_size(const Size2& p_size){

	size=p_size;
}
void LargeTexture::clear(){

	pieces.clear();
	size=Size2i();
}

Array LargeTexture::_get_data() const {

	Array arr;
	for(int i=0;i<pieces.size();i++) {
		arr.push_back(pieces[i].offset);
		arr.push_back(pieces[i].texture);
	}
	arr.push_back(Size2(size));
	return arr;

}
void LargeTexture::_set_data(const Array& p_array) {

	ERR_FAIL_COND(p_array.size()<1);
	ERR_FAIL_COND(!(p_array.size()&1));
	clear();
	for(int i=0;i<p_array.size()-1;i+=2) {
		add_piece(p_array[i],p_array[i+1]);
	}
	size=Size2(p_array[p_array.size()-1]);
}

int LargeTexture::get_piece_count() const {

	return pieces.size();
}
Vector2 LargeTexture::get_piece_offset(int p_idx) const{

	ERR_FAIL_INDEX_V(p_idx,pieces.size(),Vector2());
	return pieces[p_idx].offset;
}
Ref<Texture> LargeTexture::get_piece_texture(int p_idx) const{

	ERR_FAIL_INDEX_V(p_idx,pieces.size(),Ref<Texture>());
	return pieces[p_idx].texture;

}


void LargeTexture::_bind_methods() {

	ClassDB::bind_method(_MD("add_piece","ofs","texture:Texture"),&LargeTexture::add_piece);
	ClassDB::bind_method(_MD("set_piece_offset", "idx", "ofs"),&LargeTexture::set_piece_offset);
	ClassDB::bind_method(_MD("set_piece_texture","idx", "texture:Texture"),&LargeTexture::set_piece_texture);
	ClassDB::bind_method(_MD("set_size","size"),&LargeTexture::set_size);
	ClassDB::bind_method(_MD("clear"),&LargeTexture::clear);

	ClassDB::bind_method(_MD("get_piece_count"),&LargeTexture::get_piece_count);
	ClassDB::bind_method(_MD("get_piece_offset","idx"),&LargeTexture::get_piece_offset);
	ClassDB::bind_method(_MD("get_piece_texture:Texture","idx"),&LargeTexture::get_piece_texture);

	ClassDB::bind_method(_MD("_set_data","data"),&LargeTexture::_set_data);
	ClassDB::bind_method(_MD("_get_data"),&LargeTexture::_get_data);

	ADD_PROPERTY( PropertyInfo( Variant::ARRAY, "_data",PROPERTY_HINT_NONE,"",PROPERTY_USAGE_NOEDITOR), _SCS("_set_data"),_SCS("_get_data") );

}




void LargeTexture::draw(RID p_canvas_item, const Point2& p_pos, const Color& p_modulate, bool p_transpose) const {

	for(int i=0;i<pieces.size();i++) {

		// TODO
		pieces[i].texture->draw(p_canvas_item,pieces[i].offset+p_pos,p_modulate,p_transpose);
	}
}

void LargeTexture::draw_rect(RID p_canvas_item,const Rect2& p_rect, bool p_tile,const Color& p_modulate, bool p_transpose) const {

	//tiling not supported for this
	if (size.x==0 || size.y==0)
		return;

	Size2 scale = p_rect.size/size;

	for(int i=0;i<pieces.size();i++) {

		// TODO
		pieces[i].texture->draw_rect(p_canvas_item,Rect2(pieces[i].offset*scale+p_rect.pos,pieces[i].texture->get_size()*scale),false,p_modulate,p_transpose);
	}
}
void LargeTexture::draw_rect_region(RID p_canvas_item,const Rect2& p_rect, const Rect2& p_src_rect,const Color& p_modulate, bool p_transpose) const {


	//tiling not supported for this
	if (p_src_rect.size.x==0 || p_src_rect.size.y==0)
		return;

	Size2 scale = p_rect.size/p_src_rect.size;

	for(int i=0;i<pieces.size();i++) {

		// TODO
		Rect2 rect( pieces[i].offset, pieces[i].texture->get_size());
		if (!p_src_rect.intersects(rect))
			continue;
		Rect2 local = p_src_rect.clip(rect);
		Rect2 target = local;
		target.size*=scale;
		target.pos=p_rect.pos+(p_src_rect.pos+rect.pos)*scale;
		local.pos-=rect.pos;
		pieces[i].texture->draw_rect_region(p_canvas_item,target,local,p_modulate,p_transpose);
	}

}


LargeTexture::LargeTexture() {


}


///////////////////////////////////////////////




void CubeMap::set_flags(uint32_t p_flags) {

	flags=p_flags;
	if (_is_valid())
		VS::get_singleton()->texture_set_flags(cubemap,flags|VS::TEXTURE_FLAG_CUBEMAP);
}

uint32_t CubeMap::get_flags() const {

	return flags;
}

void CubeMap::set_side(Side p_side,const Image& p_image) {

	ERR_FAIL_COND(p_image.empty());
	ERR_FAIL_INDEX(p_side,6);
	if (!_is_valid()) {
		format = p_image.get_format();
		w=p_image.get_width();
		h=p_image.get_height();
		VS::get_singleton()->texture_allocate(cubemap,w,h,p_image.get_format(),flags|VS::TEXTURE_FLAG_CUBEMAP);
	}

	VS::get_singleton()->texture_set_data(cubemap,p_image,VS::CubeMapSide(p_side));
	valid[p_side]=true;
}

Image CubeMap::get_side(Side p_side) const {

	if (!valid[p_side])
		return Image();
	return VS::get_singleton()->texture_get_data(cubemap,VS::CubeMapSide(p_side));

}

Image::Format CubeMap::get_format() const {

	return format;
}
int CubeMap::get_width() const {

	return w;
}
int CubeMap::get_height() const {

	return h;
}

RID CubeMap::get_rid() const {

	return cubemap;
}


void CubeMap::set_storage(Storage p_storage) {

	storage=p_storage;
}

CubeMap::Storage CubeMap::get_storage() const {

	return storage;
}

void CubeMap::set_lossy_storage_quality(float p_lossy_storage_quality) {

	lossy_storage_quality=p_lossy_storage_quality;
}

float CubeMap::get_lossy_storage_quality() const {

	return lossy_storage_quality;
}

void CubeMap::set_path(const String& p_path,bool p_take_over) {

	if (cubemap.is_valid()) {
		VisualServer::get_singleton()->texture_set_path(cubemap,p_path);
	}

	Resource::set_path(p_path,p_take_over);
}


bool CubeMap::_set(const StringName& p_name, const Variant& p_value) {

	if (p_name=="side/left") {
		set_side(SIDE_LEFT,p_value);
	} else if (p_name=="side/right") {
		set_side(SIDE_RIGHT,p_value);
	} else if (p_name=="side/bottom") {
		set_side(SIDE_BOTTOM,p_value);
	} else if (p_name=="side/top") {
		set_side(SIDE_TOP,p_value);
	} else if (p_name=="side/front") {
		set_side(SIDE_FRONT,p_value);
	} else if (p_name=="side/back") {
		set_side(SIDE_BACK,p_value);
	} else if (p_name=="flags") {
		set_flags(p_value);
	} else if (p_name=="storage") {
		storage=Storage(p_value.operator int());
	} else if (p_name=="lossy_quality") {
		lossy_storage_quality=p_value;
	} else
		return false;

	return true;

}

bool CubeMap::_get(const StringName& p_name,Variant &r_ret) const {

	if (p_name=="side/left") {
		r_ret=get_side(SIDE_LEFT);
	} else if (p_name=="side/right") {
		r_ret=get_side(SIDE_RIGHT);
	} else if (p_name=="side/bottom") {
		r_ret=get_side(SIDE_BOTTOM);
	} else if (p_name=="side/top") {
		r_ret=get_side(SIDE_TOP);
	} else if (p_name=="side/front") {
		r_ret=get_side(SIDE_FRONT);
	} else if (p_name=="side/back") {
		r_ret=get_side(SIDE_BACK);
	} else if (p_name=="flags") {
		r_ret= flags;
	} else if (p_name=="storage") {
		r_ret= storage;
	} else if (p_name=="lossy_quality") {
		r_ret= lossy_storage_quality;
	} else
		return false;

	return true;
}




void CubeMap::_get_property_list( List<PropertyInfo> *p_list) const {


	PropertyHint img_hint=PROPERTY_HINT_NONE;
	if (storage==STORAGE_COMPRESS_LOSSY) {
		img_hint=PROPERTY_HINT_IMAGE_COMPRESS_LOSSY;
	} else if (storage==STORAGE_COMPRESS_LOSSLESS) {
		img_hint=PROPERTY_HINT_IMAGE_COMPRESS_LOSSLESS;
	}


	p_list->push_back( PropertyInfo( Variant::INT, "flags", PROPERTY_HINT_FLAGS,"Mipmaps,Repeat,Filter" ) );
	p_list->push_back( PropertyInfo( Variant::IMAGE, "side/left", img_hint,String::num(lossy_storage_quality)) );
	p_list->push_back( PropertyInfo( Variant::IMAGE, "side/right", img_hint,String::num(lossy_storage_quality)) );
	p_list->push_back( PropertyInfo( Variant::IMAGE, "side/bottom", img_hint,String::num(lossy_storage_quality)) );
	p_list->push_back( PropertyInfo( Variant::IMAGE, "side/top", img_hint,String::num(lossy_storage_quality)) );
	p_list->push_back( PropertyInfo( Variant::IMAGE, "side/front", img_hint,String::num(lossy_storage_quality)) );
	p_list->push_back( PropertyInfo( Variant::IMAGE, "side/back", img_hint,String::num(lossy_storage_quality)) );
	p_list->push_back( PropertyInfo( Variant::INT, "storage", PROPERTY_HINT_ENUM,"Uncompressed,Compress Lossy,Compress Lossless",PROPERTY_USAGE_EDITOR));
	p_list->push_back( PropertyInfo( Variant::REAL, "lossy_quality", PROPERTY_HINT_RANGE,"0.0,1.0,0.01" ) );
}

void CubeMap::_bind_methods() {

	ClassDB::bind_method(_MD("get_width"),&CubeMap::get_width);
	ClassDB::bind_method(_MD("get_height"),&CubeMap::get_height);
	//ClassDB::bind_method(_MD("get_rid"),&CubeMap::get_rid);
	ClassDB::bind_method(_MD("set_flags","flags"),&CubeMap::set_flags);
	ClassDB::bind_method(_MD("get_flags"),&CubeMap::get_flags);

	ClassDB::bind_method(_MD("set_side","side","image"),&CubeMap::set_side);
	ClassDB::bind_method(_MD("get_side","side"),&CubeMap::get_side);
	ClassDB::bind_method(_MD("set_storage","mode"),&CubeMap::set_storage);
	ClassDB::bind_method(_MD("get_storage"),&CubeMap::get_storage);
	ClassDB::bind_method(_MD("set_lossy_storage_quality","quality"),&CubeMap::set_lossy_storage_quality);
	ClassDB::bind_method(_MD("get_lossy_storage_quality"),&CubeMap::get_lossy_storage_quality);


	BIND_CONSTANT( STORAGE_RAW );
	BIND_CONSTANT( STORAGE_COMPRESS_LOSSY  );
	BIND_CONSTANT( STORAGE_COMPRESS_LOSSLESS );
	BIND_CONSTANT( SIDE_LEFT );
	BIND_CONSTANT( SIDE_RIGHT );
	BIND_CONSTANT( SIDE_BOTTOM );
	BIND_CONSTANT( SIDE_TOP );
	BIND_CONSTANT( SIDE_FRONT );
	BIND_CONSTANT( SIDE_BACK );
	BIND_CONSTANT( FLAG_MIPMAPS );
	BIND_CONSTANT( FLAG_REPEAT );
	BIND_CONSTANT( FLAG_FILTER );
	BIND_CONSTANT( FLAGS_DEFAULT );


}

CubeMap::CubeMap() {

	w=h=0;
	flags=FLAGS_DEFAULT;
	for(int i=0;i<6;i++)
		valid[i]=false;
	cubemap = VisualServer::get_singleton()->texture_create();
	storage = STORAGE_RAW;
	lossy_storage_quality=0.7;


}


CubeMap::~CubeMap() {

	VisualServer::get_singleton()->free( cubemap );
}




/*	BIND_CONSTANT( FLAG_CUBEMAP );
	BIND_CONSTANT( CUBEMAP_LEFT );
	BIND_CONSTANT( CUBEMAP_RIGHT );
	BIND_CONSTANT( CUBEMAP_BOTTOM );
	BIND_CONSTANT( CUBEMAP_TOP );
	BIND_CONSTANT( CUBEMAP_FRONT );
	BIND_CONSTANT( CUBEMAP_BACK );
*/
