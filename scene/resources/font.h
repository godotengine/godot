/*************************************************************************/
/*  font.h                                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2015 Juan Linietsky, Ariel Manzur.                 */
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
#ifndef FONT_H
#define FONT_H

#include "resource.h"
#include "scene/resources/texture.h"
#include "map.h"
#include "ttf_font.h"
/**
	@author Juan Linietsky <reduzio@gmail.com>
*/
class Font : public Resource {
	
	OBJ_TYPE( Font, Resource );
	RES_BASE_EXTENSION("fnt");
	
	mutable Vector< Ref<Texture> > textures;

    friend class TtfFont;
    Vector<Image *> atlas_images;
    Ref<TtfFont> ttf_font;
    Dictionary ttf_options;
    int atlas_x, atlas_y, atlas_height;
    int atlas_dirty_index;
    mutable bool atlas_dirty;

    void update_atlas() const;

public:
	struct Character {
		
		int texture_idx;
		Rect2 rect;
		float v_align;
		float h_align;
		float advance;
		
		Character() { texture_idx=0; v_align=0; }
	};

	struct KerningPairKey {

		union {
			struct {
				uint32_t A,B;
			};

			uint64_t pair;
		};

		_FORCE_INLINE_ bool operator<(const KerningPairKey& p_r) const { return pair<p_r.pair; }
	};

private:

	
	HashMap< CharType, Character > char_map;
	Map<KerningPairKey,int> kerning_map;
	
	float height;
	float ascent;
	bool distance_field_hint;

	void _set_chars(const DVector<int>& p_chars);
	DVector<int> _get_chars() const;
	void _set_kernings(const DVector<int>& p_kernings);
	DVector<int> _get_kernings() const;
	void _set_textures(const Vector<Variant> & p_textures);
	Vector<Variant> _get_textures() const;
	void _reload_hook(const RID& p_hook);

protected:
	
	static void _bind_methods();

public:

	Error create_from_fnt(const String& p_file);
	
	void set_height(float p_height);
	float get_height() const;
	
	void set_ascent(float p_ascent);
	float get_ascent() const;	
	float get_descent() const;
	
	void add_texture(const Ref<Texture>& p_texture);
	void add_char(CharType p_char, int p_texture_idx, const Rect2& p_rect, const Size2& p_align, float p_advance=-1);

	int get_character_count() const;
	Vector<CharType> get_char_keys() const;
	Character get_character(CharType p_char) const;
	const Character *get_character_p(CharType p_char) const;
    bool create_character(CharType p_char);

	int get_texture_count() const;
	Ref<Texture> get_texture(int p_idx) const;

	void add_kerning_pair(CharType p_A,CharType p_B,int p_kerning);
	int get_kerning_pair(CharType p_A,CharType p_B) const;
	Vector<KerningPairKey> get_kerning_pair_keys() const;

	_FORCE_INLINE_ Size2 get_char_size(CharType p_char,CharType p_next=0) const;
	Size2 get_string_size(const String& p_string) const;

    bool set_ttf_path(const String& p_path, int p_size);
    void set_ttf_font(const Ref<TtfFont>& p_font);
    Ref<TtfFont> get_ttf_font() const;
    void set_ttf_options(const Dictionary& p_options);
    const Dictionary& get_ttf_options() const;
	
	void clear();

	void set_distance_field_hint(bool p_distance_field);
	bool is_distance_field_hint() const;
	
	void draw(RID p_canvas_item, const Point2& p_pos, const String& p_text,const Color& p_modulate=Color(1,1,1),int p_clip_w=-1) const;
	void draw_halign(RID p_canvas_item, const Point2& p_pos, HAlign p_align,float p_width,const String& p_text,const Color& p_modulate=Color(1,1,1)) const;
	float draw_char(RID p_canvas_item, const Point2& p_pos, const CharType& p_char,const CharType& p_next=0,const Color& p_modulate=Color(1,1,1)) const;
	
	Font();
	~Font();
};


Size2 Font::get_char_size(CharType p_char,CharType p_next) const {

	const Character * c = get_character_p(p_char);

	if (!c)
		return Size2();

	Size2 ret(c->advance,c->rect.size.y);

	if (p_next) {

		KerningPairKey kpk;
		kpk.A=p_char;
		kpk.B=p_next;

		const Map<KerningPairKey,int>::Element *E=kerning_map.find(kpk);
		if (E) {

			ret.width-=E->get();
		}
	}

	return ret;
}



#endif
