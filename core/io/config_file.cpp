/*************************************************************************/
/*  config_file.cpp                                                      */
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
#include "config_file.h"
#include "os/keyboard.h"
#include "os/file_access.h"

StringArray ConfigFile::_get_sections() const {

	List<String> s;
	get_sections(&s);
	StringArray arr;
	arr.resize(s.size());
	int idx=0;
	for(const List<String>::Element *E=s.front();E;E=E->next()) {

		arr.set(idx++,E->get());
	}

	return arr;
}

StringArray ConfigFile::_get_section_keys(const String& p_section) const{

	List<String> s;
	get_section_keys(p_section,&s);
	StringArray arr;
	arr.resize(s.size());
	int idx=0;
	for(const List<String>::Element *E=s.front();E;E=E->next()) {

		arr.set(idx++,E->get());
	}

	return arr;

}


void ConfigFile::set_value(const String& p_section, const String& p_key, const Variant& p_value){

	if (p_value.get_type()==Variant::NIL) {
		//erase
		if (!values.has(p_section))
			return; // ?
		values[p_section].erase(p_key);
		if (values[p_section].empty()) {
			values.erase(p_section);
		}

	} else {
		if (!values.has(p_section)) {
			values[p_section]=Map<String, Variant>();
		}

		values[p_section][p_key]=p_value;

	}

}
Variant ConfigFile::get_value(const String& p_section, const String& p_key) const{

	ERR_FAIL_COND_V(!values.has(p_section),Variant());
	ERR_FAIL_COND_V(!values[p_section].has(p_key),Variant());
	return values[p_section][p_key];

}

bool ConfigFile::has_section(const String& p_section) const {

	return values.has(p_section);
}
bool ConfigFile::has_section_key(const String& p_section,const String& p_key) const {

	if (!values.has(p_section))
		return false;
	return values[p_section].has(p_key);
}

void ConfigFile::get_sections(List<String> *r_sections) const{

	for(const Map< String, Map<String, Variant> >::Element *E=values.front();E;E=E->next()) {
		r_sections->push_back(E->key());
	}
}
void ConfigFile::get_section_keys(const String& p_section,List<String> *r_keys) const{

	ERR_FAIL_COND(!values.has(p_section));

	for(const Map<String, Variant> ::Element *E=values[p_section].front();E;E=E->next()) {
		r_keys->push_back(E->key());
	}

}

static String _encode_variant(const Variant& p_variant) {

	switch(p_variant.get_type()) {

		case Variant::BOOL: {
			bool val = p_variant;
			return (val?"true":"false");
		} break;
		case Variant::INT: {
			int val = p_variant;
			return itos(val);
		} break;
		case Variant::REAL: {
			float val = p_variant;
			return rtos(val)+(val==int(val)?".0":"");
		} break;
		case Variant::STRING: {
			String val = p_variant;
			return "\""+val.xml_escape()+"\"";
		} break;
		case Variant::COLOR: {

			Color val = p_variant;
			return "#"+val.to_html();
		} break;
		case Variant::STRING_ARRAY:
		case Variant::INT_ARRAY:
		case Variant::REAL_ARRAY:
		case Variant::ARRAY: {
			Array arr = p_variant;
			String str="[";
			for(int i=0;i<arr.size();i++) {

				if (i>0)
					str+=", ";
				str+=_encode_variant(arr[i]);
			}
			str+="]";
			return str;
		} break;
		case Variant::DICTIONARY: {
			Dictionary d = p_variant;
			String str="{";
			List<Variant> keys;
			d.get_key_list(&keys);
			for(List<Variant>::Element *E=keys.front();E;E=E->next()) {

				if (E!=keys.front())
					str+=", ";
				str+=_encode_variant(E->get());
				str+=":";
				str+=_encode_variant(d[E->get()]);

			}
			str+="}";
			return str;
		} break;
		case Variant::IMAGE: {
			String str="img(";

			Image img=p_variant;
			if (!img.empty()) {

				String format;
				switch(img.get_format()) {

					case Image::FORMAT_GRAYSCALE: format="grayscale"; break;
					case Image::FORMAT_INTENSITY: format="intensity"; break;
					case Image::FORMAT_GRAYSCALE_ALPHA: format="grayscale_alpha"; break;
					case Image::FORMAT_RGB: format="rgb"; break;
					case Image::FORMAT_RGBA: format="rgba"; break;
					case Image::FORMAT_INDEXED : format="indexed"; break;
					case Image::FORMAT_INDEXED_ALPHA: format="indexed_alpha"; break;
					case Image::FORMAT_BC1: format="bc1"; break;
					case Image::FORMAT_BC2: format="bc2"; break;
					case Image::FORMAT_BC3: format="bc3"; break;
					case Image::FORMAT_BC4: format="bc4"; break;
					case Image::FORMAT_BC5: format="bc5"; break;
					case Image::FORMAT_CUSTOM: format="custom custom_size="+itos(img.get_data().size())+""; break;
					default: {}
				}

				str+=format+", ";
				str+=itos(img.get_mipmaps())+", ";
				str+=itos(img.get_width())+", ";
				str+=itos(img.get_height())+", ";
				DVector<uint8_t> data = img.get_data();
				int ds=data.size();
				DVector<uint8_t>::Read r = data.read();
				for(int i=0;i<ds;i++) {
					uint8_t byte = r[i];
					const char  hex[16]={'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'};
					char bstr[3]={ hex[byte>>4], hex[byte&0xF], 0};
					str+=bstr;
				}
			}
			str+=")";
			return str;
		} break;
		case Variant::INPUT_EVENT: {

			InputEvent ev = p_variant;

			switch(ev.type) {

				case InputEvent::KEY: {

					String mods;
					if (ev.key.mod.control)
						mods+="C";
					if (ev.key.mod.shift)
						mods+="S";
					if (ev.key.mod.alt)
						mods+="A";
					if (ev.key.mod.meta)
						mods+="M";
					if (mods!="")
						mods=", "+mods;

					return "key("+keycode_get_string(ev.key.scancode)+mods+")";
				} break;
				case InputEvent::MOUSE_BUTTON: {

					return "mbutton("+itos(ev.device)+", "+itos(ev.mouse_button.button_index)+")";
				} break;
				case InputEvent::JOYSTICK_BUTTON: {

					return "jbutton("+itos(ev.device)+", "+itos(ev.joy_button.button_index)+")";
				} break;
				case InputEvent::JOYSTICK_MOTION: {

					return "jaxis("+itos(ev.device)+", "+itos(ev.joy_motion.axis)+")";
				} break;
				default: {

					return "nil";
				} break;

			}
		} break;
		default: {}
	}

	return "nil"; //don't know wha to do with this
}


Error ConfigFile::save(const String& p_path){

	Error err;
	FileAccess *file = FileAccess::open(p_path,FileAccess::WRITE,&err);

	if (err) {
		return err;
	}


	for(Map< String, Map<String, Variant> >::Element *E=values.front();E;E=E->next()) {

		if (E!=values.front())
			file->store_string("\n");
		file->store_string("["+E->key()+"]\n\n");

		for(Map<String, Variant>::Element *F=E->get().front();F;F=F->next()) {

			file->store_string(F->key()+"="+_encode_variant(F->get())+"\n");
		}
	}

	memdelete(file);

	return OK;
}

static Vector<String> _decode_params(const String& p_string) {

	int begin=p_string.find("(");
	ERR_FAIL_COND_V(begin==-1,Vector<String>());
	begin++;
	int end=p_string.find(")");
	ERR_FAIL_COND_V(end<begin,Vector<String>());
	return p_string.substr(begin,end-begin).split(",");
}

static String _get_chunk(const String& str,int &pos, int close_pos) {


	enum {
		MIN_COMMA,
		MIN_COLON,
		MIN_CLOSE,
		MIN_QUOTE,
		MIN_PARENTHESIS,
		MIN_CURLY_OPEN,
		MIN_OPEN
	};

	int min_pos=close_pos;
	int min_what=MIN_CLOSE;

#define TEST_MIN(m_how,m_what) \
{\
int res = str.find(m_how,pos);\
if (res!=-1 && res < min_pos) {\
	min_pos=res;\
	min_what=m_what;\
}\
}\


	TEST_MIN(",",MIN_COMMA);
	TEST_MIN("[",MIN_OPEN);
	TEST_MIN("{",MIN_CURLY_OPEN);
	TEST_MIN("(",MIN_PARENTHESIS);
	TEST_MIN("\"",MIN_QUOTE);

	int end=min_pos;


	switch(min_what) {

		case MIN_COMMA: {
		} break;
		case MIN_CLOSE: {
			//end because it's done
		} break;
		case MIN_QUOTE: {
			end=str.find("\"",min_pos+1)+1;
			ERR_FAIL_COND_V(end==-1,Variant());

		} break;
		case MIN_PARENTHESIS: {

			end=str.find(")",min_pos+1)+1;
			ERR_FAIL_COND_V(end==-1,Variant());

		} break;
		case MIN_OPEN: {
			int level=1;
			while(end<close_pos) {

				if (str[end]=='[')
					level++;
				if (str[end]==']') {
					level--;
					if (level==0)
						break;
				}
				end++;
			}
			ERR_FAIL_COND_V(level!=0,Variant());
			end++;
		} break;
		case MIN_CURLY_OPEN: {
			int level=1;
			while(end<close_pos) {

				if (str[end]=='{')
					level++;
				if (str[end]=='}') {
					level--;
					if (level==0)
						break;
				}
				end++;
			}
			ERR_FAIL_COND_V(level!=0,Variant());
			end++;
		} break;

	}

	String ret = str.substr(pos,end-pos);

	pos=end;
	while(pos<close_pos) {
		if (str[pos]!=',' && str[pos]!=' ' && str[pos]!=':')
			break;
		pos++;
	}

	return ret;

}


static Variant _decode_variant(const String& p_string) {


	String str = p_string.strip_edges();

	if (str.nocasecmp_to("true")==0)
		return Variant(true);
	if (str.nocasecmp_to("false")==0)
		return Variant(false);
	if (str.nocasecmp_to("nil")==0)
		return Variant();
	if (str.is_valid_float()) {
		if (str.find(".")==-1)
			return str.to_int();
		else
			return str.to_double();

	}
	if (str.begins_with("#")) { //string
		return Color::html(str);
	}
	if (str.begins_with("\"")) { //string
		int end = str.find_last("\"");
		ERR_FAIL_COND_V(end==0,Variant());
		return str.substr(1,end-1).xml_unescape();

	}

	if (str.begins_with("[")) { //array

		int close_pos = str.find_last("]");
		ERR_FAIL_COND_V(close_pos==-1,Variant());
		Array array;

		int pos=1;

		while(pos<close_pos) {

			String s = _get_chunk(str,pos,close_pos);
			array.push_back(_decode_variant(s));
		}
		return array;

	}

	if (str.begins_with("{")) { //array

		int close_pos = str.find_last("}");
		ERR_FAIL_COND_V(close_pos==-1,Variant());
		Dictionary d;

		int pos=1;

		while(pos<close_pos) {

			String key = _get_chunk(str,pos,close_pos);
			String data = _get_chunk(str,pos,close_pos);
			d[_decode_variant(key)]=_decode_variant(data);
		}
		return d;

	}
	if (str.begins_with("key")) {
		Vector<String> params = _decode_params(p_string);
		ERR_FAIL_COND_V(params.size()!=1 && params.size()!=2,Variant());
		int scode=0;

		if (params[0].is_numeric()) {
			scode=params[0].to_int();
			if (scode < 10) {
				scode=KEY_0+scode;
			}
		} else
			scode=find_keycode(params[0]);

		InputEvent ie;
		ie.type=InputEvent::KEY;
		ie.key.scancode=scode;

		if (params.size()==2) {
			String mods=params[1];
			if (mods.findn("C")!=-1)
				ie.key.mod.control=true;
			if (mods.findn("A")!=-1)
				ie.key.mod.alt=true;
			if (mods.findn("S")!=-1)
				ie.key.mod.shift=true;
			if (mods.findn("M")!=-1)
				ie.key.mod.meta=true;
		}
		return ie;

	}

	if (str.begins_with("mbutton")) {
		Vector<String> params = _decode_params(p_string);
		ERR_FAIL_COND_V(params.size()!=2,Variant());

		InputEvent ie;
		ie.type=InputEvent::MOUSE_BUTTON;
		ie.device=params[0].to_int();
		ie.mouse_button.button_index=params[1].to_int();

		return ie;
	}

	if (str.begins_with("jbutton")) {
		Vector<String> params = _decode_params(p_string);
		ERR_FAIL_COND_V(params.size()!=2,Variant());

		InputEvent ie;
		ie.type=InputEvent::JOYSTICK_BUTTON;
		ie.device=params[0].to_int();
		ie.joy_button.button_index=params[1].to_int();

		return ie;
	}

	if (str.begins_with("jaxis")) {
		Vector<String> params = _decode_params(p_string);
		ERR_FAIL_COND_V(params.size()!=2,Variant());

		InputEvent ie;
		ie.type=InputEvent::JOYSTICK_MOTION;
		ie.device=params[0].to_int();
		ie.joy_motion.axis=params[1].to_int();

		return ie;
	}
	if (str.begins_with("img")) {
		Vector<String> params = _decode_params(p_string);
		if (params.size()==0) {
			return Image();
		}

		ERR_FAIL_COND_V(params.size()!=5,Image());

		String format=params[0].strip_edges();

		Image::Format imgformat;

		if (format=="grayscale") {
			imgformat=Image::FORMAT_GRAYSCALE;
		} else if (format=="intensity") {
			imgformat=Image::FORMAT_INTENSITY;
		} else if (format=="grayscale_alpha") {
			imgformat=Image::FORMAT_GRAYSCALE_ALPHA;
		} else if (format=="rgb") {
			imgformat=Image::FORMAT_RGB;
		} else if (format=="rgba") {
			imgformat=Image::FORMAT_RGBA;
		} else if (format=="indexed") {
			imgformat=Image::FORMAT_INDEXED;
		} else if (format=="indexed_alpha") {
			imgformat=Image::FORMAT_INDEXED_ALPHA;
		} else if (format=="bc1") {
			imgformat=Image::FORMAT_BC1;
		} else if (format=="bc2") {
			imgformat=Image::FORMAT_BC2;
		} else if (format=="bc3") {
			imgformat=Image::FORMAT_BC3;
		} else if (format=="bc4") {
			imgformat=Image::FORMAT_BC4;
		} else if (format=="bc5") {
			imgformat=Image::FORMAT_BC5;
		} else if (format=="custom") {
			imgformat=Image::FORMAT_CUSTOM;
		} else {

			ERR_FAIL_V( Image() );
		}

		int mipmaps=params[1].to_int();
		int w=params[2].to_int();
		int h=params[3].to_int();

		if (w == 0 && h == 0) {
			//r_v = Image(w, h, imgformat);
			return Image();
		};


		String data=params[4];
		int datasize=data.length()/2;
		DVector<uint8_t> pixels;
		pixels.resize(datasize);
		DVector<uint8_t>::Write wb = pixels.write();
		const CharType *cptr=data.c_str();

		int idx=0;
		uint8_t byte;
		while( idx<datasize*2) {

			CharType c=*(cptr++);

			ERR_FAIL_COND_V(c=='<',ERR_FILE_CORRUPT);

			if ( (c>='0' && c<='9') || (c>='A' && c<='F') || (c>='a' && c<='f') ) {

				if (idx&1) {

					byte|=HEX2CHR(c);
					wb[idx>>1]=byte;
				} else {

					byte=HEX2CHR(c)<<4;
				}

				idx++;
			}

		}

		wb = DVector<uint8_t>::Write();

		return Image(w,h,mipmaps,imgformat,pixels);
	}

	if (str.find(",")!=-1) { //vector2 or vector3
		Vector<float> farr = str.split_floats(",",true);
		if (farr.size()==2) {
			return Point2(farr[0],farr[1]);
		}
		if (farr.size()==3) {
			return Vector3(farr[0],farr[1],farr[2]);
		}
		ERR_FAIL_V(Variant());
	}


	return Variant();
}

Error ConfigFile::load(const String& p_path) {

	Error err;
	FileAccess *f= FileAccess::open(p_path,FileAccess::READ,&err);

	if (err!=OK) {

		return err;
	}


	String line;
	String section;
	String subpath;

	int line_count = 0;

	while(!f->eof_reached()) {

		String line = f->get_line().strip_edges();
		line_count++;

		if (line=="")
			continue;

		// find comments

		 {

			int pos=0;
			while (true) {
				int ret = line.find(";",pos);
				if (ret==-1)
					break;

				int qc=0;
				for(int i=0;i<ret;i++) {

					if (line[i]=='"')
						qc++;
				}

				if ( !(qc&1) ) {
					//not inside string, real comment
					line=line.substr(0,ret);
					break;

				}

				pos=ret+1;


			}
		}

		if (line.begins_with("[")) {

			int end = line.find_last("]");
			ERR_CONTINUE(end!=line.length()-1);

			section=line.substr(1,line.length()-2);

		} else if (line.find("=")!=-1) {


			int eqpos = line.find("=");
			String var=line.substr(0,eqpos).strip_edges();
			String value=line.substr(eqpos+1,line.length()).strip_edges();

			Variant val = _decode_variant(value);

			set_value(section,var,val);

		} else {

			if (line.length() > 0) {
				ERR_PRINT(String("Syntax error on line "+itos(line_count)+" of file "+p_path).ascii().get_data());
			};
		};
	}

	memdelete(f);

	return OK;
}



void ConfigFile::_bind_methods(){

	ObjectTypeDB::bind_method(_MD("set_value","section","key","value"),&ConfigFile::set_value);
	ObjectTypeDB::bind_method(_MD("get_value","section","key"),&ConfigFile::get_value);

	ObjectTypeDB::bind_method(_MD("has_section","section"),&ConfigFile::has_section);
	ObjectTypeDB::bind_method(_MD("has_section_key","section","key"),&ConfigFile::has_section_key);

	ObjectTypeDB::bind_method(_MD("get_sections"),&ConfigFile::_get_sections);
	ObjectTypeDB::bind_method(_MD("get_section_keys"),&ConfigFile::_get_section_keys);

	ObjectTypeDB::bind_method(_MD("load:Error","path"),&ConfigFile::load);
	ObjectTypeDB::bind_method(_MD("save:Error","path"),&ConfigFile::save);

}


ConfigFile::ConfigFile()
{
}
