/*************************************************************************/
/*  font.cpp                                                             */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                 */
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
#include "font.h"

#include "core/os/file_access.h"
#include "core/io/resource_loader.h"

void Font::_set_chars(const DVector<int>& p_chars) {

	int len = p_chars.size();
	//char 1 charsize 1 texture, 4 rect, 2 align, advance 1
	ERR_FAIL_COND(len%9);
	if (!len)
		return; //none to do
	int chars = len/9;


	DVector<int>::Read r=p_chars.read();
	for(int i=0;i<chars;i++) {

		const int* data = &r[i*9];
		add_char(data[0],data[1],Rect2(data[2],data[3],data[4],data[5]), Size2(data[6],data[7]),data[8]);
	}

}

DVector<int> Font::_get_chars() const {

	DVector<int> chars;

	const CharType* key=NULL;

	while((key=char_map.next(key))) {

		const Character *c=char_map.getptr(*key);
		chars.push_back(*key);
		chars.push_back(c->texture_idx);
		chars.push_back(c->rect.pos.x);
		chars.push_back(c->rect.pos.y);

		chars.push_back(c->rect.size.x);
		chars.push_back(c->rect.size.y);
		chars.push_back(c->h_align);
		chars.push_back(c->v_align);
		chars.push_back(c->advance);
	}

	return chars;
}

void Font::_set_kernings(const DVector<int>& p_kernings) {

	int len=p_kernings.size();
	ERR_FAIL_COND(len%3);
	if (!len)
		return;
	DVector<int>::Read r=p_kernings.read();

	for(int i=0;i<len/3;i++) {

		const int* data = &r[i*3];
		add_kerning_pair(data[0],data[1],data[2]);
	}
}

DVector<int> Font::_get_kernings() const {

	DVector<int> kernings;

	for(Map<KerningPairKey,int>::Element *E=kerning_map.front();E;E=E->next()) {

		kernings.push_back(E->key().A);
		kernings.push_back(E->key().B);
		kernings.push_back(E->get());
	}

	return kernings;
}


void Font::_set_textures(const Vector<Variant> & p_textures) {

	for(int i=0;i<p_textures.size();i++) {
		Ref<Texture> tex = p_textures[i];
		ERR_CONTINUE(!tex.is_valid());
		add_texture(tex);
	}

}

Vector<Variant> Font::_get_textures() const {

	Vector<Variant> rtextures;
	for(int i=0;i<textures.size();i++)
		rtextures.push_back(textures[i].get_ref_ptr());
	return rtextures;
}

Error Font::create_from_fnt(const String& p_string) {
	//fnt format used by angelcode bmfont
	//http://www.angelcode.com/products/bmfont/

	FileAccess *f = FileAccess::open(p_string,FileAccess::READ);

	if (!f) {
		ERR_EXPLAIN("Can't open font: "+p_string);
		ERR_FAIL_V(ERR_FILE_NOT_FOUND);
	}

	clear();

	while(true) {

		String line=f->get_line();

		int delimiter=line.find(" ");
		String type=line.substr(0,delimiter);
		int pos = delimiter+1;
		Map<String,String> keys;

		while (pos < line.size() && line[pos]==' ')
			pos++;


		while(pos<line.size()) {

			int eq = line.find("=",pos);
			if (eq==-1)
				break;
			String key=line.substr(pos,eq-pos);
			int end=-1;
			String value;
			if (line[eq+1]=='"') {
				end=line.find("\"",eq+2);
				if (end==-1)
					break;
				value=line.substr(eq+2,end-1-eq-1);
				pos=end+1;
			} else {
				end=line.find(" ",eq+1);
				if (end==-1)
					end=line.size();

				value=line.substr(eq+1,end-eq);

				pos=end;

			}

			while (pos<line.size() && line[pos]==' ')
				pos++;


			keys[key]=value;

		}


		if (type=="info") {

			if (keys.has("face"))
				set_name(keys["face"]);
			//if (keys.has("size"))
			//	font->set_height(keys["size"].to_int());

		} else if (type=="common") {

			if (keys.has("lineHeight"))
				set_height(keys["lineHeight"].to_int());
			if (keys.has("base"))
				set_ascent(keys["base"].to_int());

		} else if (type=="page") {

			if (keys.has("file")) {

				String file = keys["file"];
				file=p_string.get_base_dir()+"/"+file;
				Ref<Texture> tex = ResourceLoader::load(file);
				if (tex.is_null()) {
					ERR_PRINT("Can't load font texture!");
				} else {
					add_texture(tex);
				}
			}
		} else if (type=="char") {

			CharType idx=0;
			if (keys.has("id"))
				idx=keys["id"].to_int();

			Rect2 rect;

			if (keys.has("x"))
				rect.pos.x=keys["x"].to_int();
			if (keys.has("y"))
				rect.pos.y=keys["y"].to_int();
			if (keys.has("width"))
				rect.size.width=keys["width"].to_int();
			if (keys.has("height"))
				rect.size.height=keys["height"].to_int();

			Point2 ofs;

			if (keys.has("xoffset"))
				ofs.x=keys["xoffset"].to_int();
			if (keys.has("yoffset"))
				ofs.y=keys["yoffset"].to_int();

			int texture=0;
			if (keys.has("page"))
				texture=keys["page"].to_int();
			int advance=-1;
			if (keys.has("xadvance"))
				advance=keys["xadvance"].to_int();

			add_char(idx,texture,rect,ofs,advance);

		}  else if (type=="kerning") {

			CharType first=0,second=0;
			int k=0;

			if (keys.has("first"))
				first=keys["first"].to_int();
			if (keys.has("second"))
				second=keys["second"].to_int();
			if (keys.has("amount"))
				k=keys["amount"].to_int();

			add_kerning_pair(first,second,-k);

		}

		if (f->eof_reached())
			break;
	}



	memdelete(f);

	return OK;
}



void Font::set_height(float p_height) {
	
	height=p_height;
}
float Font::get_height() const{
	
	return height;
}

void Font::set_ascent(float p_ascent){
	
	ascent=p_ascent;
}
float Font::get_ascent() const {
	
	return ascent;
}
float Font::get_descent() const {
	
	return height-ascent;
}

void Font::add_texture(const Ref<Texture>& p_texture) {

	ERR_FAIL_COND( p_texture.is_null());
	textures.push_back( p_texture );
}

int Font::get_texture_count() const {

	return textures.size();
};

Ref<Texture> Font::get_texture(int p_idx) const {

	ERR_FAIL_INDEX_V(p_idx, textures.size(), Ref<Texture>());
	return textures[p_idx];
};

int Font::get_character_count() const {

	return char_map.size();
};

Vector<CharType> Font::get_char_keys() const {

	Vector<CharType> chars;
	chars.resize(char_map.size());
	const CharType* ct = NULL;
	int count = 0;
	while ( (ct = char_map.next(ct)) ) {

		chars[count++] = *ct;
	};

	return chars;
};

Font::Character Font::get_character(CharType p_char) const {

	if (!char_map.has(p_char)) {
		ERR_FAIL_COND_V(!(const_cast<Font *>(this))->create_character(p_char),Character());
	};

	return char_map[p_char];
};

const Font::Character *Font::get_character_p(CharType p_char) const {

	if (!char_map.has(p_char)) {
		ERR_FAIL_COND_V(!(const_cast<Font *>(this))->create_character(p_char),NULL);
	};

	return &char_map[p_char];
}

bool Font::create_character(CharType p_char) {

    if(!ttf_font.is_valid())
        return false;

    if(ttf_font->render_char(p_char, *this)) {
        atlas_dirty=true;
        return true;
    }
	char_map[p_char]=Character();
    return true;
}

void Font::add_char(CharType p_char, int p_texture_idx, const Rect2& p_rect, const Size2& p_align, float p_advance) {

	if (p_advance<0)
		p_advance=p_rect.size.width;

	Character c;
	c.rect=p_rect;
	c.texture_idx=p_texture_idx;
	c.v_align=p_align.y;
	c.advance=p_advance;
	c.h_align=p_align.x;
	
	char_map[p_char]=c;
}

void Font::add_kerning_pair(CharType p_A,CharType p_B,int p_kerning) {


	KerningPairKey kpk;
	kpk.A=p_A;
	kpk.B=p_B;

	if (p_kerning==0 && kerning_map.has(kpk)) {

		kerning_map.erase(kpk);
	} else {

		kerning_map[kpk]=p_kerning;
	}
}

Vector<Font::KerningPairKey> Font::get_kerning_pair_keys() const {


	Vector<Font::KerningPairKey> ret;
	ret.resize(kerning_map.size());
	int i=0;

	for (Map<KerningPairKey,int>::Element *E=kerning_map.front();E;E=E->next()) {
		ret[i++]=E->key();

	}

	return ret;

}

int Font::get_kerning_pair(CharType p_A,CharType p_B) const {

	KerningPairKey kpk;
	kpk.A=p_A;
	kpk.B=p_B;

	const Map<KerningPairKey,int>::Element *E=kerning_map.find(kpk);
	if (E)
		return E->get();

	return 0;
}


void Font::clear() {
	
	height=1;
	ascent=0;
	char_map.clear();
	textures.clear();
	kerning_map.clear();

    atlas_x=0;
    atlas_y=0;
    atlas_height=0;
    atlas_dirty_index=0;
    atlas_dirty=false;
    for(int i=0;i<atlas_images.size();i++) {
        memdelete(atlas_images[i]);
    }
    atlas_images.clear();
}

Size2 Font::get_string_size(const String& p_string) const {

	float w=0;
	
	int l = p_string.length();
	if (l==0)
		return Size2(0,height);
	const CharType *sptr = &p_string[0];

	for (int i=0;i<l;i++) {
			
		w+=get_char_size(sptr[i],sptr[i+1]).width;
	}

	return Size2(w,height);
}

void Font::draw_halign(RID p_canvas_item, const Point2& p_pos, HAlign p_align,float p_width,const String& p_text,const Color& p_modulate) const {

	float length=get_string_size(p_text).width;
	if (length>=p_width) {
		draw(p_canvas_item,p_pos,p_text,p_modulate,p_width);
		return;
	}

	float ofs;
	switch(p_align) {
		case HALIGN_LEFT: {
			ofs=0;
		} break;
		case HALIGN_CENTER: {
			 ofs = Math::floor( (p_width-length) / 2.0 );
		} break;
		case HALIGN_RIGHT: {
			ofs=p_width-length;
		} break;
	}
	draw(p_canvas_item,p_pos+Point2(ofs,0),p_text,p_modulate,p_width);
}

void Font::draw(RID p_canvas_item, const Point2& p_pos, const String& p_text, const Color& p_modulate,int p_clip_w) const {
		

    update_atlas();

	Point2 pos=p_pos;
	float ofs=0;
	VisualServer *vs = VisualServer::get_singleton();
	
	for (int i=0;i<p_text.length();i++) {

		const Character * c = get_character_p(p_text[i]);

		if (!c)
			continue;
			
//		if (p_clip_w>=0 && (ofs+c->rect.size.width)>(p_clip_w))
//			break; //width exceeded

		if (p_clip_w>=0 && (ofs+c->rect.size.width)>p_clip_w)
			break; //clip
		Point2 cpos=pos;
		cpos.x+=ofs+c->h_align;
		cpos.y-=ascent;
		cpos.y+=c->v_align;
		if( c->texture_idx<-1 || c->texture_idx>=textures.size())
        {
            if (p_text[i]==' '||p_text[i]=='\u3000')
                continue;
            else
		        ERR_CONTINUE( c->texture_idx<-1 || c->texture_idx>=textures.size());
        }
		if (c->texture_idx!=-1)
			textures[c->texture_idx]->draw_rect_region( p_canvas_item, Rect2( cpos, c->rect.size ), c->rect, p_modulate );
		
		ofs+=get_char_size(p_text[i],p_text[i+1]).width;
	}
}

float Font::draw_char(RID p_canvas_item, const Point2& p_pos, const CharType& p_char,const CharType& p_next,const Color& p_modulate) const {
	
	const Character * c = get_character_p(p_char);
	
	if (!c)
		return 0;


    update_atlas();

	Point2 cpos=p_pos;
	cpos.x+=c->h_align;
	cpos.y-=ascent;
	cpos.y+=c->v_align;
	if (c->texture_idx<-1 || c->texture_idx>=textures.size())
    {
        if (p_char==' '||p_char=='\u3000')
            return 0;
        else
    	    ERR_FAIL_COND_V( c->texture_idx<-1 || c->texture_idx>=textures.size(),0)
    }
	if (c->texture_idx!=-1)
		VisualServer::get_singleton()->canvas_item_add_texture_rect_region( p_canvas_item, Rect2( cpos, c->rect.size ), textures[c->texture_idx]->get_rid(),c->rect, p_modulate );
	
	return get_char_size(p_char,p_next).width;
}

void Font::update_atlas() const {

    if ((!atlas_dirty)||ttf_font.is_null())
        return;
    atlas_dirty=false;

    for (int i=atlas_dirty_index;i<atlas_images.size();i++) {
        Image& img = *atlas_images[i];
        Ref<ImageTexture> tex;
        if (textures.size()==i) {
            tex=Ref<Texture>(memnew( ImageTexture ));

            bool filter_enabled = ttf_options["filter/enabled"];
            tex->create_from_image( img );
            if (!filter_enabled)
                tex->set_flags(Texture::FLAG_MIPMAPS | Texture::FLAG_REPEAT);
            tex->set_storage( ImageTexture::STORAGE_COMPRESS_LOSSLESS );

            textures.push_back(tex);
        } else {
            tex=textures[i];
            if (tex.is_valid()) {
                tex->set_data( img );
            }
        }
    }
}

bool Font::set_ttf_path(const String& p_path, int p_size) {
    RES res=ResourceLoader::load(p_path);
    Ref<TtfFont> ttf_font=res;
    if(!ttf_font.is_valid())
        return false;

    int height=0;
    int ascent=0;
    int max_up,max_down;
    ERR_EXPLAIN("Error calc font height/ascent.");
    ERR_FAIL_COND_V( !ttf_font->calc_size(p_size, height, ascent, max_up, max_down), false );

    Dictionary options;
    options["font/size"]=p_size;
    options["meta/height"]=height;
    options["meta/ascent"]=ascent;
    options["meta/max_up"]=max_up;
    options["meta/max_down"]=max_down;

    clear();
    set_ttf_font(ttf_font);
    set_ttf_options(options);
    set_height(height);
    set_ascent(ascent);

    return true;
}

void Font::set_ttf_font(const Ref<TtfFont>& p_font) {

	if (p_font==ttf_font)
		return;
    //clear();
	char_map.clear();
	textures.clear();

    atlas_x=0;
    atlas_y=0;
    atlas_height=0;
    atlas_dirty_index=0;
    atlas_dirty=false;
    for(int i=0;i<atlas_images.size();i++) {
        memdelete(atlas_images[i]);
    }
    atlas_images.clear();

	ttf_font=p_font;
    if (!ttf_options.empty()) {
        kerning_map.clear();
        ttf_font->gen_kerning(this);
    }
}

Ref<TtfFont> Font::get_ttf_font() const {

	return ttf_font;
}

void Font::set_ttf_options(const Dictionary& p_options) {

    ttf_options=p_options;
    if (kerning_map.empty()&&ttf_font.is_valid()) {
        ttf_font->gen_kerning(this);
    }
}

const Dictionary& Font::get_ttf_options() const {

    return ttf_options;
}

void Font::_bind_methods() {

	ObjectTypeDB::bind_method(_MD("set_height","px"),&Font::set_height);
	ObjectTypeDB::bind_method(_MD("get_height"),&Font::get_height);

	ObjectTypeDB::bind_method(_MD("set_ascent","px"),&Font::set_ascent);
	ObjectTypeDB::bind_method(_MD("get_ascent"),&Font::get_ascent);
	ObjectTypeDB::bind_method(_MD("get_descent"),&Font::get_descent);

	ObjectTypeDB::bind_method(_MD("add_kerning_pair","char_a","char_b","kerning"),&Font::add_kerning_pair);
	ObjectTypeDB::bind_method(_MD("get_kerning_pair"),&Font::get_kerning_pair);

	ObjectTypeDB::bind_method(_MD("add_texture","texture:Texture"),&Font::add_texture);
	ObjectTypeDB::bind_method(_MD("add_char","character","texture","rect","align","advance"),&Font::add_char,DEFVAL(Point2()),DEFVAL(-1));

	ObjectTypeDB::bind_method(_MD("get_char_size","char","next"),&Font::get_char_size,DEFVAL(0));
	ObjectTypeDB::bind_method(_MD("get_string_size","string"),&Font::get_string_size);

	ObjectTypeDB::bind_method(_MD("clear"),&Font::clear);

	ObjectTypeDB::bind_method(_MD("draw","canvas_item","pos","string","modulate","clip_w"),&Font::draw,DEFVAL(Color(1,1,1)),DEFVAL(-1));
	ObjectTypeDB::bind_method(_MD("draw_char","canvas_item","pos","char","next","modulate"),&Font::draw_char,DEFVAL(-1),DEFVAL(Color(1,1,1)));

	ObjectTypeDB::bind_method(_MD("_set_chars"),&Font::_set_chars);
	ObjectTypeDB::bind_method(_MD("_get_chars"),&Font::_get_chars);

	ObjectTypeDB::bind_method(_MD("_set_kernings"),&Font::_set_kernings);
	ObjectTypeDB::bind_method(_MD("_get_kernings"),&Font::_get_kernings);

	ObjectTypeDB::bind_method(_MD("_set_textures"),&Font::_set_textures);
	ObjectTypeDB::bind_method(_MD("_get_textures"),&Font::_get_textures);


	ADD_PROPERTY( PropertyInfo( Variant::ARRAY, "textures", PROPERTY_HINT_NONE,"", PROPERTY_USAGE_NOEDITOR ), _SCS("_set_textures"), _SCS("_get_textures") );
	ADD_PROPERTY( PropertyInfo( Variant::INT_ARRAY, "chars", PROPERTY_HINT_NONE,"", PROPERTY_USAGE_NOEDITOR ), _SCS("_set_chars"), _SCS("_get_chars") );
	ADD_PROPERTY( PropertyInfo( Variant::INT_ARRAY, "kernings", PROPERTY_HINT_NONE,"", PROPERTY_USAGE_NOEDITOR ), _SCS("_set_kernings"), _SCS("_get_kernings") );

	ADD_PROPERTY( PropertyInfo( Variant::REAL, "height", PROPERTY_HINT_RANGE,"-1024,1024,1" ), _SCS("set_height"), _SCS("get_height") );
	ADD_PROPERTY( PropertyInfo( Variant::REAL, "ascent", PROPERTY_HINT_RANGE,"-1024,1024,1" ), _SCS("set_ascent"), _SCS("get_ascent") );

    ObjectTypeDB::bind_method(_MD("set_ttf_options"),&Font::set_ttf_options);
	ObjectTypeDB::bind_method(_MD("get_ttf_options"),&Font::get_ttf_options);
    ADD_PROPERTY( PropertyInfo( Variant::DICTIONARY, "data", PROPERTY_HINT_NONE,"",PROPERTY_USAGE_STORAGE), _SCS("set_ttf_options"),_SCS("get_ttf_options"));

	ObjectTypeDB::bind_method(_MD("set_ttf_font","font:TtfFont"),&Font::set_ttf_font);
	ObjectTypeDB::bind_method(_MD("get_ttf_font:TtfFont"),&Font::get_ttf_font);
	ADD_PROPERTY( PropertyInfo( Variant::OBJECT, "font", PROPERTY_HINT_RESOURCE_TYPE,"TtfFont"), _SCS("set_ttf_font"),_SCS("get_ttf_font"));
}

Font::Font() {
	
	clear();
	

}


Font::~Font() {
	
	clear();
}


