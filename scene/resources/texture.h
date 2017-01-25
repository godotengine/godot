/*************************************************************************/
/*  texture.h                                                            */
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
#ifndef TEXTURE_H
#define TEXTURE_H

#include "resource.h"
#include "servers/visual_server.h"
#include "math_2d.h"

/**
	@author Juan Linietsky <reduzio@gmail.com>
*/



class Texture : public Resource {

	GDCLASS( Texture, Resource );
	OBJ_SAVE_TYPE( Texture ); //children are all saved as Texture, so they can be exchanged
protected:

	static void _bind_methods();
public:

	enum Flags {
		FLAG_MIPMAPS=VisualServer::TEXTURE_FLAG_MIPMAPS,
		FLAG_REPEAT=VisualServer::TEXTURE_FLAG_REPEAT,
		FLAG_FILTER=VisualServer::TEXTURE_FLAG_FILTER,
		FLAG_ANISOTROPIC_FILTER=VisualServer::TEXTURE_FLAG_ANISOTROPIC_FILTER,
		FLAG_CONVERT_TO_LINEAR=VisualServer::TEXTURE_FLAG_CONVERT_TO_LINEAR,
		FLAG_VIDEO_SURFACE=VisualServer::TEXTURE_FLAG_USED_FOR_STREAMING,
		FLAGS_DEFAULT=FLAG_MIPMAPS|FLAG_REPEAT|FLAG_FILTER,
		FLAG_MIRRORED_REPEAT=VisualServer::TEXTURE_FLAG_MIRRORED_REPEAT
	};


	virtual int get_width() const=0;
	virtual int get_height() const=0;
	virtual Size2 get_size() const;
	virtual RID get_rid() const=0;

	virtual bool has_alpha() const=0;

	virtual void set_flags(uint32_t p_flags)=0;
	virtual uint32_t get_flags() const=0;

	virtual void draw(RID p_canvas_item, const Point2& p_pos, const Color& p_modulate=Color(1,1,1), bool p_transpose=false) const;
	virtual void draw_rect(RID p_canvas_item,const Rect2& p_rect, bool p_tile=false,const Color& p_modulate=Color(1,1,1), bool p_transpose=false) const;
	virtual void draw_rect_region(RID p_canvas_item,const Rect2& p_rect, const Rect2& p_src_rect,const Color& p_modulate=Color(1,1,1), bool p_transpose=false) const;
	virtual bool get_rect_region(const Rect2& p_rect, const Rect2& p_src_rect,Rect2& r_rect,Rect2& r_src_rect) const;



	Texture();
};

VARIANT_ENUM_CAST( Texture::Flags );


class ImageTexture : public Texture {

	GDCLASS( ImageTexture, Texture );
	RES_BASE_EXTENSION("tex");
public:
	enum Storage {
		STORAGE_RAW,
		STORAGE_COMPRESS_LOSSY,
		STORAGE_COMPRESS_LOSSLESS
	};
private:
	RID texture;
	Image::Format format;
	uint32_t flags;
	int w,h;
	Storage storage;
	Size2 size_override;
	float lossy_storage_quality;

protected:
	virtual void reload_from_file();

	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _get_property_list( List<PropertyInfo> *p_list) const;

	void _reload_hook(const RID& p_hook);
	virtual void _resource_path_changed();
	static void _bind_methods();

	void _set_data(Dictionary p_data);

public:


	void create(int p_width, int p_height,Image::Format p_format,uint32_t p_flags=FLAGS_DEFAULT);
	void create_from_image(const Image& p_image,  uint32_t p_flags=FLAGS_DEFAULT);


	void set_flags(uint32_t p_flags);
	uint32_t get_flags() const;
	Image::Format get_format() const;
	void load(const String& p_path);
	void set_data(const Image& p_image);
	Image get_data() const;

	int get_width() const;
	int get_height() const;

	virtual RID get_rid() const;

	bool has_alpha() const;
	virtual void draw(RID p_canvas_item, const Point2& p_pos, const Color& p_modulate=Color(1,1,1), bool p_transpose=false) const;
	virtual void draw_rect(RID p_canvas_item,const Rect2& p_rect, bool p_tile=false,const Color& p_modulate=Color(1,1,1), bool p_transpose=false) const;
	virtual void draw_rect_region(RID p_canvas_item,const Rect2& p_rect, const Rect2& p_src_rect,const Color& p_modulate=Color(1,1,1), bool p_transpose=false) const;
	void set_storage(Storage p_storage);
	Storage get_storage() const;

	void set_lossy_storage_quality(float p_lossy_storage_quality);
	float get_lossy_storage_quality() const;

	void fix_alpha_edges();
	void premultiply_alpha();
	void normal_to_xy();
	void shrink_x2_and_keep_size();


	void set_size_override(const Size2& p_size);

	virtual void set_path(const String& p_path,bool p_take_over=false);

	ImageTexture();
	~ImageTexture();

};


VARIANT_ENUM_CAST( ImageTexture::Storage );

class AtlasTexture : public Texture {

	GDCLASS( AtlasTexture, Texture );
	RES_BASE_EXTENSION("atex");
protected:


	Ref<Texture> atlas;
	Rect2 region;
	Rect2 margin;

	static void _bind_methods();
public:

	virtual int get_width() const;
	virtual int get_height() const;
	virtual RID get_rid() const;

	virtual bool has_alpha() const;

	virtual void set_flags(uint32_t p_flags);
	virtual uint32_t get_flags() const;

	void set_atlas(const Ref<Texture>& p_atlas);
	Ref<Texture> get_atlas() const;

	void set_region(const Rect2& p_region);
	Rect2 get_region() const ;

	void set_margin(const Rect2& p_margin);
	Rect2 get_margin() const ;

	virtual void draw(RID p_canvas_item, const Point2& p_pos, const Color& p_modulate=Color(1,1,1), bool p_transpose=false) const;
	virtual void draw_rect(RID p_canvas_item,const Rect2& p_rect, bool p_tile=false,const Color& p_modulate=Color(1,1,1), bool p_transpose=false) const;
	virtual void draw_rect_region(RID p_canvas_item,const Rect2& p_rect, const Rect2& p_src_rect,const Color& p_modulate=Color(1,1,1), bool p_transpose=false) const;
	virtual bool get_rect_region(const Rect2& p_rect, const Rect2& p_src_rect,Rect2& r_rect,Rect2& r_src_rect) const;


	AtlasTexture();
};

class LargeTexture : public Texture {

	GDCLASS( LargeTexture, Texture );
	RES_BASE_EXTENSION("ltex");
protected:

	struct Piece {

		Point2 offset;
		Ref<Texture> texture;
	};

	Vector<Piece> pieces;
	Size2i size;


	Array _get_data() const;
	void _set_data(const Array& p_array);
	static void _bind_methods();
public:

	virtual int get_width() const;
	virtual int get_height() const;
	virtual RID get_rid() const;

	virtual bool has_alpha() const;

	virtual void set_flags(uint32_t p_flags);
	virtual uint32_t get_flags() const;

	int add_piece(const Point2& p_offset,const Ref<Texture>& p_texture);
	void set_piece_offset(int p_idx, const Point2& p_offset);
	void set_piece_texture(int p_idx, const Ref<Texture>& p_texture);

	void set_size(const Size2& p_size);
	void clear();

	int get_piece_count() const;
	Vector2 get_piece_offset(int p_idx) const;
	Ref<Texture> get_piece_texture(int p_idx) const;

	virtual void draw(RID p_canvas_item, const Point2& p_pos, const Color& p_modulate=Color(1,1,1), bool p_transpose=false) const;
	virtual void draw_rect(RID p_canvas_item,const Rect2& p_rect, bool p_tile=false,const Color& p_modulate=Color(1,1,1), bool p_transpose=false) const;
	virtual void draw_rect_region(RID p_canvas_item,const Rect2& p_rect, const Rect2& p_src_rect,const Color& p_modulate=Color(1,1,1), bool p_transpose=false) const;


	LargeTexture();
};



class CubeMap : public Resource {

	GDCLASS( CubeMap, Resource );
	RES_BASE_EXTENSION("cbm");
public:
	enum Storage {
		STORAGE_RAW,
		STORAGE_COMPRESS_LOSSY,
		STORAGE_COMPRESS_LOSSLESS
	};

	enum Side {

		SIDE_LEFT,
		SIDE_RIGHT,
		SIDE_BOTTOM,
		SIDE_TOP,
		SIDE_FRONT,
		SIDE_BACK
	};

	enum Flags {
		FLAG_MIPMAPS=VisualServer::TEXTURE_FLAG_MIPMAPS,
		FLAG_REPEAT=VisualServer::TEXTURE_FLAG_REPEAT,
		FLAG_FILTER=VisualServer::TEXTURE_FLAG_FILTER,
		FLAGS_DEFAULT=FLAG_MIPMAPS|FLAG_REPEAT|FLAG_FILTER,
	};

private:

	bool valid[6];
	RID cubemap;
	Image::Format format;
	uint32_t flags;
	int w,h;
	Storage storage;
	Size2 size_override;
	float lossy_storage_quality;

	_FORCE_INLINE_ bool _is_valid() const { for(int i=0;i<6;i++) { if (valid[i]) return true; }  return false; }

protected:

	bool _set(const StringName& p_name, const Variant& p_value);
	bool _get(const StringName& p_name,Variant &r_ret) const;
	void _get_property_list( List<PropertyInfo> *p_list) const;

	static void _bind_methods();
public:

	void set_flags(uint32_t p_flags);
	uint32_t get_flags() const;
	void set_side(Side p_side,const Image& p_image);
	Image get_side(Side p_side) const;

	Image::Format get_format() const;
	int get_width() const;
	int get_height() const;

	virtual RID get_rid() const;

	void set_storage(Storage p_storage);
	Storage get_storage() const;

	void set_lossy_storage_quality(float p_lossy_storage_quality);
	float get_lossy_storage_quality() const;

	virtual void set_path(const String& p_path,bool p_take_over=false);

	CubeMap();
	~CubeMap();

};

VARIANT_ENUM_CAST( CubeMap::Flags );
VARIANT_ENUM_CAST( CubeMap::Side );
VARIANT_ENUM_CAST( CubeMap::Storage );


/*
	enum CubeMapSide {

		CUBEMAP_LEFT,
		CUBEMAP_RIGHT,
		CUBEMAP_BOTTOM,
		CUBEMAP_TOP,
		CUBEMAP_FRONT,
		CUBEMAP_BACK,
	};

*/
//VARIANT_ENUM_CAST( Texture::CubeMapSide );


#endif
