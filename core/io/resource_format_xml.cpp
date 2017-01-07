/*************************************************************************/
/*  resource_format_xml.cpp                                              */
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
#include "resource_format_xml.h"
#include "globals.h"
#include "version.h"
#include "os/dir_access.h"


ResourceInteractiveLoaderXML::Tag* ResourceInteractiveLoaderXML::parse_tag(bool *r_exit, bool p_printerr, List<String> *r_order) {


	while(get_char()!='<' && !f->eof_reached()) {}
	if (f->eof_reached()) {
		return NULL;
	}

	Tag tag;
	bool exit=false;
	if (r_exit)
		*r_exit=false;

	bool complete=false;
	while(!f->eof_reached()) {

		CharType c=get_char();
		if (c<33 && tag.name.length() && !exit) {
			break;
		} else if (c=='>') {
			complete=true;
			break;
		} else if (c=='/') {
			exit=true;
		} else {
			tag.name+=c;
		}
	}

	if (f->eof_reached()) {

		return NULL;
	}

	if (exit) {
		if (!tag_stack.size()) {
			if (!p_printerr)
				return NULL;
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Unmatched exit tag </"+tag.name+">");
			ERR_FAIL_COND_V(!tag_stack.size(),NULL);
		}

		if (tag_stack.back()->get().name!=tag.name) {
			if (!p_printerr)
				return NULL;
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Mismatched exit tag. Got </"+tag.name+">, expected </"+tag_stack.back()->get().name+">");
			ERR_FAIL_COND_V(tag_stack.back()->get().name!=tag.name,NULL);
		}

		if (!complete) {
			while(get_char()!='>' && !f->eof_reached()) {}
			if (f->eof_reached())
				return NULL;
		}

		if (r_exit)
			*r_exit=true;

		tag_stack.pop_back();
		return NULL;

	}

	if (!complete) {
		String name;
		CharString r_value;
		bool reading_value=false;

		while(!f->eof_reached()) {

			CharType c=get_char();
			if (c=='>') {
				if (r_value.size()) {

					r_value.push_back(0);
					String str;
					str.parse_utf8(r_value.get_data());
					tag.args[name]=str;
					if (r_order)
						r_order->push_back(name);
				}
				break;

			} else if ( ((!reading_value && (c<33)) || c=='=' || c=='"' || c=='\'') && tag.name.length()) {

				if (!reading_value && name.length()) {

					reading_value=true;
				} else if (reading_value && r_value.size()) {

					r_value.push_back(0);
					String str;
					str.parse_utf8(r_value.get_data());
					tag.args[name]=str;
					if (r_order)
						r_order->push_back(name);
					name="";
					r_value.clear();
					reading_value=false;
				}

			} else if (reading_value) {

				r_value.push_back(c);
			} else {

				name+=c;
			}
		}

		if (f->eof_reached())
			return NULL;
	}

	tag_stack.push_back(tag);

	return &tag_stack.back()->get();
}


Error ResourceInteractiveLoaderXML::close_tag(const String& p_name) {

	int level=0;
	bool inside_tag=false;

	while(true) {

		if (f->eof_reached()) {

			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": EOF found while attempting to find  </"+p_name+">");
			ERR_FAIL_COND_V( f->eof_reached(), ERR_FILE_CORRUPT );
		}

		uint8_t c = get_char();

		if (c == '<') {

			if (inside_tag) {
				ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Malformed XML. Already inside Tag.");
				ERR_FAIL_COND_V(inside_tag,ERR_FILE_CORRUPT);
			}
			inside_tag=true;
			c = get_char();
			if (c == '/') {

				--level;
			} else {

				++level;
			};
		} else if (c == '>') {

			if (!inside_tag) {
				ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Malformed XML. Already outside Tag");
				ERR_FAIL_COND_V(!inside_tag,ERR_FILE_CORRUPT);
			}
			inside_tag=false;
			if (level == -1) {
				tag_stack.pop_back();
				return OK;
			};
		};
	}

	return OK;
}

void ResourceInteractiveLoaderXML::unquote(String& p_str) {


	p_str=p_str.strip_edges().replace("\"","").xml_unescape();

	/*p_str=p_str.strip_edges();
	p_str=p_str.replace("\"","");
	p_str=p_str.replace("&gt;","<");
	p_str=p_str.replace("&lt;",">");
	p_str=p_str.replace("&apos;","'");
	p_str=p_str.replace("&quot;","\"");
	for (int i=1;i<32;i++) {

		char chr[2]={i,0};
		p_str=p_str.replace("&#"+String::num(i)+";",chr);
	}
	p_str=p_str.replace("&amp;","&");
*/
	//p_str.parse_utf8( p_str.ascii(true).get_data() );

}

Error ResourceInteractiveLoaderXML::goto_end_of_tag() {

	uint8_t c;
	while(true) {

		c=get_char();
		if (c=='>') //closetag
			break;
		if (f->eof_reached()) {

			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": EOF found while attempting to find close tag.");
			ERR_FAIL_COND_V( f->eof_reached(), ERR_FILE_CORRUPT );
		}

	}
	tag_stack.pop_back();

	return OK;
}


Error ResourceInteractiveLoaderXML::parse_property_data(String &r_data) {

	r_data="";
	CharString cs;
	while(true) {

		CharType c=get_char();
		if (c=='<')
			break;
		ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);
		cs.push_back(c);
	}

	cs.push_back(0);

	r_data.parse_utf8(cs.get_data());

	while(get_char()!='>' && !f->eof_reached()) {}
	if (f->eof_reached()) {

		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Malformed XML.");
		ERR_FAIL_COND_V( f->eof_reached(), ERR_FILE_CORRUPT );
	}

	r_data=r_data.strip_edges();
	tag_stack.pop_back();

	return OK;
}


Error ResourceInteractiveLoaderXML::_parse_array_element(Vector<char> &buff,bool p_number_only,FileAccess *f,bool *end) {

	if (buff.empty())
		buff.resize(32); // optimi

	int buff_max=buff.size();
	int buff_size=0;
	*end=false;
	char *buffptr=&buff[0];
	bool found=false;
	bool quoted=false;

	while(true) {

		char c=get_char();

		if (c==0) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": File corrupt (zero found).");
			ERR_FAIL_V(ERR_FILE_CORRUPT);
		} else if (c=='"') {
			quoted=!quoted;
		} else if ((!quoted && ((p_number_only && c<33) || c==',')) || c=='<') {


			if (c=='<') {
				*end=true;
				break;
			}
			if (c<32 && f->eof_reached()) {
				*end=true;
				ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": File corrupt (unexpected EOF).");
				ERR_FAIL_V(ERR_FILE_CORRUPT);
			}

			if (found)
				break;

		} else {

			found=true;
			if (buff_size>=buff_max) {

				buff_max++;
				buff.resize(buff_max);
				buffptr=buff.ptr();

			}

			buffptr[buff_size]=c;
			buff_size++;
		}
	}

	if (buff_size>=buff_max) {

		buff_max++;
		buff.resize(buff_max);

	}

	buff[buff_size]=0;
	buff_size++;

	return OK;
}

Error ResourceInteractiveLoaderXML::parse_property(Variant& r_v, String &r_name)  {

	bool exit;
	Tag *tag = parse_tag(&exit);

	if (!tag) {
		if (exit) // shouldn't have exited
			return ERR_FILE_EOF;
		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": File corrupt (No Property Tag).");
		ERR_FAIL_V(ERR_FILE_CORRUPT);
	}

	r_v=Variant();
	r_name="";


	//ERR_FAIL_COND_V(tag->name!="property",ERR_FILE_CORRUPT);
	//ERR_FAIL_COND_V(!tag->args.has("name"),ERR_FILE_CORRUPT);
//	ERR_FAIL_COND_V(!tag->args.has("type"),ERR_FILE_CORRUPT);

	//String name=tag->args["name"];
	//ERR_FAIL_COND_V(name=="",ERR_FILE_CORRUPT);
	String type=tag->name;
	String name=tag->args["name"];

	if (type=="") {
		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": 'type' field is empty.");
		ERR_FAIL_COND_V(type=="",ERR_FILE_CORRUPT);
	}

	if (type=="dictionary") {

        Dictionary d( tag->args.has("shared") && (String(tag->args["shared"])=="true" || String(tag->args["shared"])=="1"));

		while(true) {

			Error err;
			String tagname;
			Variant key;

			int dictline = get_current_line();


			err=parse_property(key,tagname);

			if (err && err!=ERR_FILE_EOF) {
				ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Error parsing dictionary: "+name+" (from line "+itos(dictline)+")");
				ERR_FAIL_COND_V(err && err!=ERR_FILE_EOF,err);
			}
			//ERR_FAIL_COND_V(tagname!="key",ERR_FILE_CORRUPT);
			if (err)
				break;
			Variant value;
			err=parse_property(value,tagname);
			if (err) {
				ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Error parsing dictionary: "+name+" (from line "+itos(dictline)+")");
			}

			ERR_FAIL_COND_V(err,err);
			//ERR_FAIL_COND_V(tagname!="value",ERR_FILE_CORRUPT);

			d[key]=value;
		}


		//err=parse_property_data(name); // skip the rest
		//ERR_FAIL_COND_V(err,err);

		r_name=name;
		r_v=d;
		return OK;

	} else if (type=="array") {

		if (!tag->args.has("len")) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Array missing 'len' field: "+name);
			ERR_FAIL_COND_V(!tag->args.has("len"),ERR_FILE_CORRUPT);
		}


		int len=tag->args["len"].to_int();
		bool shared = tag->args.has("shared") && (String(tag->args["shared"])=="true" || String(tag->args["shared"])=="1");

		Array array(shared);
		array.resize(len);

		Error err;
		Variant v;
		String tagname;
		int idx=0;
		while( (err=parse_property(v,tagname))==OK ) {

			ERR_CONTINUE( idx <0 || idx >=len );

			array.set(idx,v);
			idx++;
		}

		if (idx!=len) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Error loading array (size mismatch): "+name);
			ERR_FAIL_COND_V(idx!=len,err);
		}

		if (err!=ERR_FILE_EOF) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Error loading array: "+name);
			ERR_FAIL_COND_V(err!=ERR_FILE_EOF,err);
		}

		//err=parse_property_data(name); // skip the rest
		//ERR_FAIL_COND_V(err,err);

		r_name=name;
		r_v=array;
		return OK;

	} else if (type=="resource") {

		if (tag->args.has("path")) {

			String path=tag->args["path"];
			String hint;
			if (tag->args.has("resource_type"))
				hint=tag->args["resource_type"];

			if (path.begins_with("local://"))
				path=path.replace("local://",local_path+"::");
			else if (path.find("://")==-1 && path.is_rel_path()) {
				// path is relative to file being loaded, so convert to a resource path
				path=GlobalConfig::get_singleton()->localize_path(local_path.get_base_dir().plus_file(path));

			}

			if (remaps.has(path)) {
				path=remaps[path];
			}

			//take advantage of the resource loader cache. The resource is cached on it, even if
			RES res=ResourceLoader::load(path,hint);


			if (res.is_null()) {

				WARN_PRINT(String("Couldn't load resource: "+path).ascii().get_data());
			}

			r_v=res.get_ref_ptr();
		} else if (tag->args.has("external")) {

			int index = tag->args["external"].to_int();
			if (ext_resources.has(index)) {
				String path=ext_resources[index].path;
				String type=ext_resources[index].type;

				//take advantage of the resource loader cache. The resource is cached on it, even if
				RES res=ResourceLoader::load(path,type);

				if (res.is_null()) {

					WARN_PRINT(String("Couldn't load externalresource: "+path).ascii().get_data());
				}

				r_v=res.get_ref_ptr();
			} else {
				WARN_PRINT(String("Invalid external resource index: "+itos(index)).ascii().get_data());

			}
		}




		Error err=goto_end_of_tag();
		if (err) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Error closing <resource> tag.");
			ERR_FAIL_COND_V(err,err);
		}


		r_name=name;

		return OK;

	} else if (type=="image") {

		if (!tag->args.has("encoding")) {
			//empty image
			r_v=Image();
			String sdfsdfg;
			Error err=parse_property_data(sdfsdfg);
			return OK;
		}

		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Image missing 'encoding' field.");
		ERR_FAIL_COND_V( !tag->args.has("encoding"), ERR_FILE_CORRUPT );
		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Image missing 'width' field.");
		ERR_FAIL_COND_V( !tag->args.has("width"), ERR_FILE_CORRUPT );
		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Image missing 'height' field.");
		ERR_FAIL_COND_V( !tag->args.has("height"), ERR_FILE_CORRUPT );
		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Image missing 'format' field.");
		ERR_FAIL_COND_V( !tag->args.has("format"), ERR_FILE_CORRUPT );

		String encoding=tag->args["encoding"];

		if (encoding=="raw") {
			String width=tag->args["width"];
			String height=tag->args["height"];
			String format=tag->args["format"];
			int mipmaps=tag->args.has("mipmaps")?int(tag->args["mipmaps"].to_int()):int(0);
			int custom_size = tag->args.has("custom_size")?int(tag->args["custom_size"].to_int()):int(0);

			r_name=name;

			Image::Format imgformat;

/*
			if (format=="grayscale") {
				imgformat=Image::FORMAT_L8;
			} else if (format=="intensity") {
				imgformat=Image::FORMAT_INTENSITY;
			} else if (format=="grayscale_alpha") {
				imgformat=Image::FORMAT_LA8;
			} else if (format=="rgb") {
				imgformat=Image::FORMAT_RGB8;
			} else if (format=="rgba") {
				imgformat=Image::FORMAT_RGBA8;
			} else if (format=="indexed") {
				imgformat=Image::FORMAT_INDEXED;
			} else if (format=="indexed_alpha") {
				imgformat=Image::FORMAT_INDEXED_ALPHA;
			} else if (format=="bc1") {
				imgformat=Image::FORMAT_DXT1;
			} else if (format=="bc2") {
				imgformat=Image::FORMAT_DXT3;
			} else if (format=="bc3") {
				imgformat=Image::FORMAT_DXT5;
			} else if (format=="bc4") {
				imgformat=Image::FORMAT_ATI1;
			} else if (format=="bc5") {
				imgformat=Image::FORMAT_ATI2;
			} else if (format=="pvrtc2") {
				imgformat=Image::FORMAT_PVRTC2;
			} else if (format=="pvrtc2a") {
				imgformat=Image::FORMAT_PVRTC2A;
			} else if (format=="pvrtc4") {
				imgformat=Image::FORMAT_PVRTC4;
			} else if (format=="pvrtc4a") {
				imgformat=Image::FORMAT_PVRTC4A;
			} else if (format=="etc") {
				imgformat=Image::FORMAT_ETC;
			} else if (format=="atc") {
				imgformat=Image::FORMAT_ATC;
			} else if (format=="atcai") {
				imgformat=Image::FORMAT_ATC_ALPHA_INTERPOLATED;
			} else if (format=="atcae") {
				imgformat=Image::FORMAT_ATC_ALPHA_EXPLICIT;
			} else if (format=="custom") {
				imgformat=Image::FORMAT_CUSTOM;
			} else {

				ERR_FAIL_V( ERR_FILE_CORRUPT );
			}*/


			int datasize;
			int w=width.to_int();
			int h=height.to_int();

			if (w == 0 && h == 0) {
				//r_v = Image(w, h, imgformat);
				r_v=Image();
				String sdfsdfg;
				Error err=parse_property_data(sdfsdfg);
				return OK;
			};


			if (datasize==0) {
				//r_v = Image(w, h, imgformat);
				r_v=Image();
				String sdfsdfg;
				Error err=parse_property_data(sdfsdfg);
				return OK;
			};

			PoolVector<uint8_t> pixels;
			pixels.resize(datasize);
			PoolVector<uint8_t>::Write wb = pixels.write();

			int idx=0;
			uint8_t byte;
			while( idx<datasize*2) {

				CharType c=get_char();

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
			ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);

			wb=PoolVector<uint8_t>::Write();

			r_v=Image(w,h,mipmaps,imgformat,pixels);
			String sdfsdfg;
			Error err=parse_property_data(sdfsdfg);
			ERR_FAIL_COND_V(err,err);

			return OK;
		}

		ERR_FAIL_V(ERR_FILE_CORRUPT);

	} else if (type=="raw_array") {

		if (!tag->args.has("len")) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": RawArray missing 'len' field: "+name);
			ERR_FAIL_COND_V(!tag->args.has("len"),ERR_FILE_CORRUPT);
		}
		int len=tag->args["len"].to_int();

		PoolVector<uint8_t> bytes;
		bytes.resize(len);
		PoolVector<uint8_t>::Write w=bytes.write();
		uint8_t *bytesptr=w.ptr();
		int idx=0;
		uint8_t byte;

		while( idx<len*2) {

			CharType c=get_char();
			if (c<=32)
				continue;

			if (idx&1) {

				byte|=HEX2CHR(c);
				bytesptr[idx>>1]=byte;
				//printf("%x\n",int(byte));
			} else {

				byte=HEX2CHR(c)<<4;
			}

			idx++;
		}

		ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);

		w=PoolVector<uint8_t>::Write();
		r_v=bytes;
		String sdfsdfg;
		Error err=parse_property_data(sdfsdfg);
		ERR_FAIL_COND_V(err,err);
		r_name=name;

		return OK;

	} else if (type=="int_array") {

		if (!tag->args.has("len")) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Array missing 'len' field: "+name);
			ERR_FAIL_COND_V(!tag->args.has("len"),ERR_FILE_CORRUPT);
		}
		int len=tag->args["len"].to_int();

		PoolVector<int> ints;
		ints.resize(len);
		PoolVector<int>::Write w=ints.write();
		int *intsptr=w.ptr();
		int idx=0;
		String str;
#if 0
		while( idx<len ) {


			CharType c=get_char();
			ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);

			if (c<33 || c==',' || c=='<') {

				if (str.length()) {

					intsptr[idx]=str.to_int();
					str="";
					idx++;
				}

				if (c=='<') {

					while(get_char()!='>' && !f->eof_reached()) {}
					ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);
					break;
				}

			} else {

				str+=c;
			}
		}

#else

		Vector<char> tmpdata;

		while( idx<len ) {

			bool end=false;
			Error err = _parse_array_element(tmpdata,true,f,&end);
			ERR_FAIL_COND_V(err,err);

			intsptr[idx]=String::to_int(&tmpdata[0]);
			idx++;
			if (end)
				break;

		}

#endif
		w=PoolVector<int>::Write();

		r_v=ints;
		Error err=goto_end_of_tag();
		ERR_FAIL_COND_V(err,err);
		r_name=name;

		return OK;
	} else if (type=="real_array") {

		if (!tag->args.has("len")) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Array missing 'len' field: "+name);
			ERR_FAIL_COND_V(!tag->args.has("len"),ERR_FILE_CORRUPT);
		}
		int len=tag->args["len"].to_int();;

		PoolVector<real_t> reals;
		reals.resize(len);
		PoolVector<real_t>::Write w=reals.write();
		real_t *realsptr=w.ptr();
		int idx=0;
		String str;


#if 0
		while( idx<len ) {


			CharType c=get_char();
			ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);


			if (c<33 || c==',' || c=='<') {

				if (str.length()) {

					realsptr[idx]=str.to_double();
					str="";
					idx++;
				}

				if (c=='<') {

					while(get_char()!='>' && !f->eof_reached()) {}
					ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);
					break;
				}

			} else {

				str+=c;
			}
		}

#else



		Vector<char> tmpdata;

		while( idx<len ) {

			bool end=false;
			Error err = _parse_array_element(tmpdata,true,f,&end);
			ERR_FAIL_COND_V(err,err);

			realsptr[idx]=String::to_double(&tmpdata[0]);
			idx++;

			if (end)
				break;
		}

#endif

		w=PoolVector<real_t>::Write();
		r_v=reals;

		Error err=goto_end_of_tag();
		ERR_FAIL_COND_V(err,err);
		r_name=name;

		return OK;
	} else if (type=="string_array") {
#if 0
		if (!tag->args.has("len")) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Array missing 'len' field: "+name);
			ERR_FAIL_COND_V(!tag->args.has("len"),ERR_FILE_CORRUPT);
		}
		int len=tag->args["len"].to_int();

		PoolVector<String> strings;
		strings.resize(len);
		PoolVector<String>::Write w=strings.write();
		String *stringsptr=w.ptr();
		int idx=0;
		String str;

		bool inside_str=false;
		CharString cs;
		while( idx<len ) {


			CharType c=get_char();
			ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);


			if (c=='"') {
				if (inside_str) {

					cs.push_back(0);
					String str;
					str.parse_utf8(cs.get_data());
					unquote(str);
					stringsptr[idx]=str;
					cs.clear();
					idx++;
					inside_str=false;
				} else {
					inside_str=true;
				}
			} else if (c=='<') {

				while(get_char()!='>' && !f->eof_reached()) {}
				ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);
				break;


			} else if (inside_str){

				cs.push_back(c);
			}
		}
		w=PoolVector<String>::Write();
		r_v=strings;
		String sdfsdfg;
		Error err=parse_property_data(sdfsdfg);
		ERR_FAIL_COND_V(err,err);

		r_name=name;

		return OK;
#endif
		if (!tag->args.has("len")) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": String Array missing 'len' field: "+name);
			ERR_FAIL_COND_V(!tag->args.has("len"),ERR_FILE_CORRUPT);
		}


		int len=tag->args["len"].to_int();

		StringArray array;
		array.resize(len);
		PoolVector<String>::Write w = array.write();

		Error err;
		Variant v;
		String tagname;
		int idx=0;


		while( (err=parse_property(v,tagname))==OK ) {

			ERR_CONTINUE( idx <0 || idx >=len );
			String str = v; //convert back to string
			w[idx]=str;
			idx++;
		}

		if (idx!=len) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Error loading array (size mismatch): "+name);
			ERR_FAIL_COND_V(idx!=len,err);
		}

		if (err!=ERR_FILE_EOF) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Error loading array: "+name);
			ERR_FAIL_COND_V(err!=ERR_FILE_EOF,err);
		}

		//err=parse_property_data(name); // skip the rest
		//ERR_FAIL_COND_V(err,err);

		r_name=name;
		r_v=array;
		return OK;

	} else if (type=="vector3_array") {

		if (!tag->args.has("len")) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Array missing 'len' field: "+name);
			ERR_FAIL_COND_V(!tag->args.has("len"),ERR_FILE_CORRUPT);
		}
		int len=tag->args["len"].to_int();;

		PoolVector<Vector3> vectors;
		vectors.resize(len);
		PoolVector<Vector3>::Write w=vectors.write();
		Vector3 *vectorsptr=w.ptr();
		int idx=0;
		int subidx=0;
		Vector3 auxvec;
		String str;

//		uint64_t tbegin = OS::get_singleton()->get_ticks_usec();
#if 0
		while( idx<len ) {


			CharType c=get_char();
			ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);


			if (c<33 || c==',' || c=='<') {

				if (str.length()) {

					auxvec[subidx]=str.to_double();
					subidx++;
					str="";
					if (subidx==3) {
						vectorsptr[idx]=auxvec;

						idx++;
						subidx=0;
					}
				}

				if (c=='<') {

					while(get_char()!='>' && !f->eof_reached()) {}
					ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);
					break;
				}

			} else {

				str+=c;
			}
		}
#else

		Vector<char> tmpdata;

		while( idx<len ) {

			bool end=false;
			Error err = _parse_array_element(tmpdata,true,f,&end);
			ERR_FAIL_COND_V(err,err);


			auxvec[subidx]=String::to_double(&tmpdata[0]);
			subidx++;
			if (subidx==3) {
				vectorsptr[idx]=auxvec;

				idx++;
				subidx=0;
			}

			if (end)
				break;
		}



#endif
		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Premature end of vector3 array");
		ERR_FAIL_COND_V(idx<len,ERR_FILE_CORRUPT);
//		double time_taken = (OS::get_singleton()->get_ticks_usec() - tbegin)/1000000.0;


		w=PoolVector<Vector3>::Write();
		r_v=vectors;
		String sdfsdfg;
		Error err=goto_end_of_tag();
		ERR_FAIL_COND_V(err,err);
		r_name=name;

		return OK;

	} else if (type=="vector2_array") {

		if (!tag->args.has("len")) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Array missing 'len' field: "+name);
			ERR_FAIL_COND_V(!tag->args.has("len"),ERR_FILE_CORRUPT);
		}
		int len=tag->args["len"].to_int();;

		PoolVector<Vector2> vectors;
		vectors.resize(len);
		PoolVector<Vector2>::Write w=vectors.write();
		Vector2 *vectorsptr=w.ptr();
		int idx=0;
		int subidx=0;
		Vector2 auxvec;
		String str;

//		uint64_t tbegin = OS::get_singleton()->get_ticks_usec();
#if 0
		while( idx<len ) {


			CharType c=get_char();
			ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);


			if (c<22 || c==',' || c=='<') {

				if (str.length()) {

					auxvec[subidx]=str.to_double();
					subidx++;
					str="";
					if (subidx==2) {
						vectorsptr[idx]=auxvec;

						idx++;
						subidx=0;
					}
				}

				if (c=='<') {

					while(get_char()!='>' && !f->eof_reached()) {}
					ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);
					break;
				}

			} else {

				str+=c;
			}
		}
#else

		Vector<char> tmpdata;

		while( idx<len ) {

			bool end=false;
			Error err = _parse_array_element(tmpdata,true,f,&end);
			ERR_FAIL_COND_V(err,err);


			auxvec[subidx]=String::to_double(&tmpdata[0]);
			subidx++;
			if (subidx==2) {
				vectorsptr[idx]=auxvec;

				idx++;
				subidx=0;
			}

			if (end)
				break;
		}



#endif
		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Premature end of vector2 array");
		ERR_FAIL_COND_V(idx<len,ERR_FILE_CORRUPT);
//		double time_taken = (OS::get_singleton()->get_ticks_usec() - tbegin)/1000000.0;


		w=PoolVector<Vector2>::Write();
		r_v=vectors;
		String sdfsdfg;
		Error err=goto_end_of_tag();
		ERR_FAIL_COND_V(err,err);
		r_name=name;

		return OK;

	} else if (type=="color_array") {

		if (!tag->args.has("len")) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Array missing 'len' field: "+name);
			ERR_FAIL_COND_V(!tag->args.has("len"),ERR_FILE_CORRUPT);
		}
		int len=tag->args["len"].to_int();;

		PoolVector<Color> colors;
		colors.resize(len);
		PoolVector<Color>::Write w=colors.write();
		Color *colorsptr=w.ptr();
		int idx=0;
		int subidx=0;
		Color auxcol;
		String str;

		while( idx<len ) {


			CharType c=get_char();
			ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);


			if (c<33 || c==',' || c=='<') {

				if (str.length()) {

					auxcol[subidx]=str.to_double();
					subidx++;
					str="";
					if (subidx==4) {
						colorsptr[idx]=auxcol;
						idx++;
						subidx=0;
					}
				}

				if (c=='<') {

					while(get_char()!='>' && !f->eof_reached()) {}
					ERR_FAIL_COND_V(f->eof_reached(),ERR_FILE_CORRUPT);
					break;
				}

			} else {

				str+=c;
			}
		}
		w=PoolVector<Color>::Write();
		r_v=colors;
		String sdfsdfg;
		Error err=parse_property_data(sdfsdfg);
		ERR_FAIL_COND_V(err,err);
		r_name=name;

		return OK;
	}


	String data;
	Error err = parse_property_data(data);
	ERR_FAIL_COND_V(err!=OK,err);

	if (type=="nil") {
		// uh do nothing

	} else if (type=="bool") {
		// uh do nothing
		if (data.nocasecmp_to("true")==0 || data.to_int()!=0)
			r_v=true;
		else
			r_v=false;
	} else if (type=="int") {

		r_v=data.to_int();
	} else if (type=="real") {

		r_v=data.to_double();
	} else if (type=="string") {

		String str=data;
		unquote(str);
		r_v=str;
	} else if (type=="vector3") {


		r_v=Vector3(
				data.get_slicec(',',0).to_double(),
				data.get_slicec(',',1).to_double(),
				data.get_slicec(',',2).to_double()
			   );

	} else if (type=="vector2") {


		r_v=Vector2(
				data.get_slicec(',',0).to_double(),
				data.get_slicec(',',1).to_double()
			   );

	} else if (type=="plane") {

		r_v=Plane(
				data.get_slicec(',',0).to_double(),
				data.get_slicec(',',1).to_double(),
				data.get_slicec(',',2).to_double(),
				data.get_slicec(',',3).to_double()
			 );

	} else if (type=="quaternion") {

		r_v=Quat(
				data.get_slicec(',',0).to_double(),
				data.get_slicec(',',1).to_double(),
				data.get_slicec(',',2).to_double(),
				data.get_slicec(',',3).to_double()
			 );

	} else if (type=="rect2") {

		r_v=Rect2(
			Vector2(
				data.get_slicec(',',0).to_double(),
				data.get_slicec(',',1).to_double()
			),
			Vector2(
				data.get_slicec(',',2).to_double(),
				data.get_slicec(',',3).to_double()
			)
		);


	} else if (type=="aabb") {

		r_v=AABB(
			Vector3(
				data.get_slicec(',',0).to_double(),
				data.get_slicec(',',1).to_double(),
				data.get_slicec(',',2).to_double()
			),
			Vector3(
				data.get_slicec(',',3).to_double(),
				data.get_slicec(',',4).to_double(),
				data.get_slicec(',',5).to_double()
			)
		);

	} else if (type=="matrix32") {

		Matrix32 m3;
		for (int i=0;i<3;i++) {
			for (int j=0;j<2;j++) {
				m3.elements[i][j]=data.get_slicec(',',i*2+j).to_double();
			}
		}
		r_v=m3;

	} else if (type=="matrix3") {

		Matrix3 m3;
		for (int i=0;i<3;i++) {
			for (int j=0;j<3;j++) {
				m3.elements[i][j]=data.get_slicec(',',i*3+j).to_double();
			}
		}
		r_v=m3;

	} else if (type=="transform") {

		Transform tr;
		for (int i=0;i<3;i++) {
			for (int j=0;j<3;j++) {
				tr.basis.elements[i][j]=data.get_slicec(',',i*3+j).to_double();
			}

		}
		tr.origin=Vector3(
			     data.get_slicec(',',9).to_double(),
			     data.get_slicec(',',10).to_double(),
			     data.get_slicec(',',11).to_double()
			   );
		r_v=tr;

	} else if (type=="color") {

		r_v=Color(
			   data.get_slicec(',',0).to_double(),
			   data.get_slicec(',',1).to_double(),
			   data.get_slicec(',',2).to_double(),
			   data.get_slicec(',',3).to_double()
			 );

	} else if (type=="node_path") {

		String str=data;
		unquote(str);
		r_v=NodePath( str );
	} else if (type=="input_event") {

		// ?
	} else {
		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Unrecognized tag in file: "+type);
		ERR_FAIL_V(ERR_FILE_CORRUPT);
	}
	r_name=name;
	return OK;
}



int ResourceInteractiveLoaderXML::get_current_line() const {

	return lines;
}


uint8_t ResourceInteractiveLoaderXML::get_char() const {

	uint8_t c = f->get_8();
	if (c=='\n')
		lines++;
	return c;

}




///

void ResourceInteractiveLoaderXML::set_local_path(const String& p_local_path) {

	res_path=p_local_path;
}

Ref<Resource> ResourceInteractiveLoaderXML::get_resource() {

	return resource;
}
Error ResourceInteractiveLoaderXML::poll() {

	if (error!=OK)
		return error;

	bool exit;
	Tag *tag = parse_tag(&exit);


	if (!tag) {
		error=ERR_FILE_CORRUPT;
		if (!exit) // shouldn't have exited
			ERR_FAIL_V(error);
		error=ERR_FILE_EOF;
		return error;
	}

	RES res;
	//Object *obj=NULL;

	bool main;

	if (tag->name=="ext_resource") {

		error=ERR_FILE_CORRUPT;
		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": <ext_resource> missing 'path' field.");
		ERR_FAIL_COND_V(!tag->args.has("path"),ERR_FILE_CORRUPT);

		String type="Resource";
		if (tag->args.has("type"))
			type=tag->args["type"];

		String path = tag->args["path"];


		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": <ext_resource> can't use a local path, this is a bug?.");
		ERR_FAIL_COND_V(path.begins_with("local://"),ERR_FILE_CORRUPT);

		if (path.find("://")==-1 && path.is_rel_path()) {
			// path is relative to file being loaded, so convert to a resource path
			path=GlobalConfig::get_singleton()->localize_path(local_path.get_base_dir().plus_file(path));
		}

		if (remaps.has(path)) {
			path=remaps[path];
		}

		RES res = ResourceLoader::load(path,type);

		if (res.is_null()) {

			if (ResourceLoader::get_abort_on_missing_resources()) {
				ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": <ext_resource> referenced nonexistent resource at: "+path);
				ERR_FAIL_V(error);
			} else {
				ResourceLoader::notify_dependency_error(local_path,path,type);
			}
		} else {

			resource_cache.push_back(res);
		}

		if (tag->args.has("index")) {
			ExtResource er;
			er.path=path;
			er.type=type;
			ext_resources[tag->args["index"].to_int()]=er;
		}


		Error err = close_tag("ext_resource");
		if (err)
			return error;


		error=OK;
		resource_current++;
		return error;

	} else if (tag->name=="resource") {

		main=false;
	} else if (tag->name=="main_resource") {
		main=true;
	} else {
		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": unexpected main tag: "+tag->name);
		error=ERR_FILE_CORRUPT;
		ERR_FAIL_V(error);
	}


	String type;
	String path;
	int subres=0;

	if (!main) {
		//loading resource

		error=ERR_FILE_CORRUPT;
		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": <resource> missing 'len' field.");
		ERR_FAIL_COND_V(!tag->args.has("path"),ERR_FILE_CORRUPT);
		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": <resource> missing 'type' field.");
		ERR_FAIL_COND_V(!tag->args.has("type"),ERR_FILE_CORRUPT);
		path=tag->args["path"];

		error=OK;

		if (path.begins_with("local://")) {
			//built-in resource (but really external)

			path=path.replace("local://","");
			subres=path.to_int();
			path=local_path+"::"+path;
		}


		if (ResourceCache::has(path)) {
			Error err = close_tag(tag->name);
			if (err) {
				error=ERR_FILE_CORRUPT;
			}
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Unable to close <resource> tag.");
			ERR_FAIL_COND_V( err, err );
			resource_current++;
			error=OK;
			return OK;
		}

		type = tag->args["type"];
	} else {
		type=resource_type;
	}

	Object *obj = ClassDB::instance(type);
	if (!obj) {
		error=ERR_FILE_CORRUPT;
		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Object of unrecognized type in file: "+type);
	}
	ERR_FAIL_COND_V(!obj,ERR_FILE_CORRUPT);

	Resource *r = obj->cast_to<Resource>();
	if (!r) {
		error=ERR_FILE_CORRUPT;
		memdelete(obj); //bye
		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": Object type in resource field not a resource, type is: "+obj->get_class());
		ERR_FAIL_COND_V(!r,ERR_FILE_CORRUPT);
	}

	res = RES( r );
	if (path!="")
		r->set_path(path);
	r->set_subindex(subres);

	//load properties

	while(true) {

		String name;
		Variant v;
		Error err;
		err = parse_property(v,name);
		if (err==ERR_FILE_EOF) //tag closed
			break;
		if (err!=OK) {
			ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": XML Parsing aborted.");
			ERR_FAIL_COND_V(err!=OK,ERR_FILE_CORRUPT);
		}

		obj->set(name,v);
	}
#ifdef TOOLS_ENABLED
	res->set_edited(false);
#endif
	resource_cache.push_back(res); //keep it in mem until finished loading
	resource_current++;
	if (main) {
		f->close();
		resource=res;
		if (!ResourceCache::has(res_path)) {
			resource->set_path(res_path);
		}
		error=ERR_FILE_EOF;
		return error;

	}
	error=OK;
	return OK;
}

int ResourceInteractiveLoaderXML::get_stage() const {

	return resource_current;
}
int ResourceInteractiveLoaderXML::get_stage_count() const {

	return resources_total;//+ext_resources;
}

ResourceInteractiveLoaderXML::~ResourceInteractiveLoaderXML() {

	memdelete(f);
}

void ResourceInteractiveLoaderXML::get_dependencies(FileAccess *f,List<String> *p_dependencies,bool p_add_types) {


	open(f);
	ERR_FAIL_COND(error!=OK);

	while(true) {
		bool exit;
		Tag *tag = parse_tag(&exit);


		if (!tag) {
			error=ERR_FILE_CORRUPT;
			ERR_FAIL_COND(!exit);
			error=ERR_FILE_EOF;
			return;
		}

		if (tag->name!="ext_resource") {

			return;
		}

		error=ERR_FILE_CORRUPT;
		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": <ext_resource> missing 'path' field.");
		ERR_FAIL_COND(!tag->args.has("path"));

		String path = tag->args["path"];

		ERR_EXPLAIN(local_path+":"+itos(get_current_line())+": <ext_resource> can't use a local path, this is a bug?.");
		ERR_FAIL_COND(path.begins_with("local://"));

		if (path.find("://")==-1 && path.is_rel_path()) {
			// path is relative to file being loaded, so convert to a resource path
			path=GlobalConfig::get_singleton()->localize_path(local_path.get_base_dir().plus_file(path));
		}

		if (path.ends_with("*")) {
			ERR_FAIL_COND(!tag->args.has("type"));
			String type = tag->args["type"];
			path = ResourceLoader::guess_full_filename(path,type);
		}

		if (p_add_types && tag->args.has("type")) {
			path+="::"+tag->args["type"];
		}

		p_dependencies->push_back(path);

		Error err = close_tag("ext_resource");
		if (err)
			return;

		error=OK;
	}

}

Error ResourceInteractiveLoaderXML::rename_dependencies(FileAccess *p_f, const String &p_path,const Map<String,String>& p_map) {

	open(p_f);
	ERR_FAIL_COND_V(error!=OK,error);

	//FileAccess

	bool old_format=false;

	FileAccess *fw = NULL;

	String base_path=local_path.get_base_dir();

	while(true) {
		bool exit;
		List<String> order;

		Tag *tag = parse_tag(&exit,true,&order);

		bool done=false;

		if (!tag) {
			if (fw) {
				memdelete(fw);
			}
			error=ERR_FILE_CORRUPT;
			ERR_FAIL_COND_V(!exit,error);
			error=ERR_FILE_EOF;

			return error;
		}

		if (tag->name=="ext_resource") {

			if (!tag->args.has("index") || !tag->args.has("path") || !tag->args.has("type")) {
				old_format=true;
				break;
			}

			if (!fw) {

				fw=FileAccess::open(p_path+".depren",FileAccess::WRITE);
				fw->store_line("<?xml version=\"1.0\" encoding=\"UTF-8\" ?>"); //no escape
				fw->store_line("<resource_file type=\""+resource_type+"\" subresource_count=\""+itos(resources_total)+"\" version=\""+itos(VERSION_MAJOR)+"."+itos(VERSION_MINOR)+"\" version_name=\""+VERSION_FULL_NAME+"\">");

			}

			String path = tag->args["path"];
			String index = tag->args["index"];
			String type = tag->args["type"];


			bool relative=false;
			if (!path.begins_with("res://")) {
				path=base_path.plus_file(path).simplify_path();
				relative=true;
			}


			if (p_map.has(path)) {
				String np=p_map[path];
				path=np;
			}

			if (relative) {
				//restore relative
				path=base_path.path_to_file(path);
			}

			tag->args["path"]=path;
			tag->args["index"]=index;
			tag->args["type"]=type;

		} else {

			done=true;
		}

		String tagt="\t<";
		if (exit)
			tagt+="/";
		tagt+=tag->name;

		for(List<String>::Element *E=order.front();E;E=E->next()) {
			tagt+=" "+E->get()+"=\""+tag->args[E->get()]+"\"";
		}
		tagt+=">";
		fw->store_line(tagt);
		if (done)
			break;
		close_tag("ext_resource");
		fw->store_line("\t</ext_resource>");

	}


	if (old_format) {
		if (fw)
			memdelete(fw);

		DirAccess *da = DirAccess::create(DirAccess::ACCESS_FILESYSTEM);
		da->remove(p_path+".depren");
		memdelete(da);
		//fuck it, use the old approach;

		WARN_PRINT(("This file is old, so it can't refactor dependencies, opening and resaving: "+p_path).utf8().get_data());

		Error err;
		FileAccess *f2 = FileAccess::open(p_path,FileAccess::READ,&err);
		if (err!=OK) {
			ERR_FAIL_COND_V(err!=OK,ERR_FILE_CANT_OPEN);
		}

		Ref<ResourceInteractiveLoaderXML> ria = memnew( ResourceInteractiveLoaderXML );
		ria->local_path=GlobalConfig::get_singleton()->localize_path(p_path);
		ria->res_path=ria->local_path;
		ria->remaps=p_map;
	//	ria->set_local_path( Globals::get_singleton()->localize_path(p_path) );
		ria->open(f2);

		err = ria->poll();

		while(err==OK) {
			err=ria->poll();
		}

		ERR_FAIL_COND_V(err!=ERR_FILE_EOF,ERR_FILE_CORRUPT);
		RES res = ria->get_resource();
		ERR_FAIL_COND_V(!res.is_valid(),ERR_FILE_CORRUPT);

		return ResourceFormatSaverXML::singleton->save(p_path,res);
	}

	if (!fw) {

		return OK; //nothing to rename, do nothing
	}

	uint8_t c=f->get_8();
	while(!f->eof_reached()) {
		fw->store_8(c);
		c=f->get_8();
	}
	f->close();

	bool all_ok = fw->get_error()==OK;

	memdelete(fw);

	if (!all_ok) {
		return ERR_CANT_CREATE;
	}

	DirAccess *da = DirAccess::create(DirAccess::ACCESS_RESOURCES);
	da->remove(p_path);
	da->rename(p_path+".depren",p_path);
	memdelete(da);

	return OK;

}


void ResourceInteractiveLoaderXML::open(FileAccess *p_f) {

	error=OK;

	lines=1;
	f=p_f;


	ResourceInteractiveLoaderXML::Tag *tag = parse_tag();
	if (!tag || tag->name!="?xml" || !tag->args.has("version") || !tag->args.has("encoding") || tag->args["encoding"]!="UTF-8") {

		error=ERR_FILE_CORRUPT;
		ResourceLoader::notify_load_error("XML is invalid (missing header tags)");
		ERR_EXPLAIN("Not a XML:UTF-8 File: "+local_path);
		ERR_FAIL();
	}

	tag_stack.clear();

	tag = parse_tag();


	if (!tag || tag->name!="resource_file" || !tag->args.has("type") || !tag->args.has("version")) {

		ResourceLoader::notify_load_error(local_path+": XML is not a valid resource file.");
		error=ERR_FILE_CORRUPT;
		ERR_EXPLAIN("Unrecognized XML File: "+local_path);
		ERR_FAIL();
	}


	if (tag->args.has("subresource_count"))
		resources_total=tag->args["subresource_count"].to_int();
	resource_current=0;
	resource_type=tag->args["type"];

	String version = tag->args["version"];
	if (version.get_slice_count(".")!=2) {

		error=ERR_FILE_CORRUPT;
		ResourceLoader::notify_load_error(local_path+":XML version string is invalid: "+version);
		ERR_EXPLAIN("Invalid Version String '"+version+"'' in file: "+local_path);
		ERR_FAIL();
	}

	int major = version.get_slicec('.',0).to_int();
	if (major>VERSION_MAJOR) {

		error=ERR_FILE_UNRECOGNIZED;
		ResourceLoader::notify_load_error(local_path+": File Format '"+version+"' is too new. Please upgrade to a newer engine version.");
		ERR_EXPLAIN("File Format '"+version+"' is too new! Please upgrade to a a new engine version: "+local_path);
		ERR_FAIL();

	}

	/*
	String preload_depts = "deps/"+local_path.md5_text();
	if (Globals::get_singleton()->has(preload_depts)) {
		ext_resources.clear();
		//ignore external resources and use these
		NodePath depts=Globals::get_singleton()->get(preload_depts);

		for(int i=0;i<depts.get_name_count();i++) {
			ext_resources.push_back(depts.get_name(i));
		}
		print_line(local_path+" - EXTERNAL RESOURCES: "+itos(ext_resources.size()));
	}
*/

}

String ResourceInteractiveLoaderXML::recognize(FileAccess *p_f) {

	error=OK;

	lines=1;
	f=p_f;

	ResourceInteractiveLoaderXML::Tag *tag = parse_tag();
	if (!tag || tag->name!="?xml" || !tag->args.has("version") || !tag->args.has("encoding") || tag->args["encoding"]!="UTF-8") {


		return ""; //unrecognized
	}

	tag_stack.clear();

	tag = parse_tag();

	if (!tag || tag->name!="resource_file" || !tag->args.has("type") || !tag->args.has("version")) {

		return ""; //unrecognized
	}

	return tag->args["type"];

}

/////////////////////

Ref<ResourceInteractiveLoader> ResourceFormatLoaderXML::load_interactive(const String &p_path, Error *r_error) {

	if (r_error)
		*r_error=ERR_CANT_OPEN;

	Error err;
	FileAccess *f = FileAccess::open(p_path,FileAccess::READ,&err);


	if (err!=OK) {

		ERR_FAIL_COND_V(err!=OK,Ref<ResourceInteractiveLoader>());
	}

	Ref<ResourceInteractiveLoaderXML> ria = memnew( ResourceInteractiveLoaderXML );
	ria->local_path=GlobalConfig::get_singleton()->localize_path(p_path);
	ria->res_path=ria->local_path;
//	ria->set_local_path( Globals::get_singleton()->localize_path(p_path) );
	ria->open(f);

	return ria;
}

void ResourceFormatLoaderXML::get_recognized_extensions_for_type(const String& p_type,List<String> *p_extensions) const {

	if (p_type=="") {
		get_recognized_extensions(p_extensions);
		return;
	}

	List<String> extensions;
	ClassDB::get_extensions_for_type(p_type,&extensions);

	extensions.sort();

	for(List<String>::Element *E=extensions.front();E;E=E->next()) {
		String ext = E->get().to_lower();
		if (ext=="res")
			continue;
		p_extensions->push_back("x"+ext);
	}

	p_extensions->push_back("xml");


}
void ResourceFormatLoaderXML::get_recognized_extensions(List<String> *p_extensions) const{

	List<String> extensions;
	ClassDB::get_resource_base_extensions(&extensions);
	extensions.sort();

	for(List<String>::Element *E=extensions.front();E;E=E->next()) {
		String ext = E->get().to_lower();
		if (ext=="res")
			continue;
		p_extensions->push_back("x"+ext);
	}

	p_extensions->push_back("xml");
}

bool ResourceFormatLoaderXML::handles_type(const String& p_type) const{

	return true;
}
String ResourceFormatLoaderXML::get_resource_type(const String &p_path) const{


	String ext=p_path.extension().to_lower();
	if (!ext.begins_with("x")) //a lie but..
		return "";

	FileAccess *f = FileAccess::open(p_path,FileAccess::READ);
	if (!f) {

		return ""; //could not rwead
	}

	Ref<ResourceInteractiveLoaderXML> ria = memnew( ResourceInteractiveLoaderXML );
	ria->local_path=GlobalConfig::get_singleton()->localize_path(p_path);
	ria->res_path=ria->local_path;
//	ria->set_local_path( Globals::get_singleton()->localize_path(p_path) );
	String r = ria->recognize(f);
	return r;
}


void ResourceFormatLoaderXML::get_dependencies(const String& p_path,List<String> *p_dependencies,bool p_add_types) {

	FileAccess *f = FileAccess::open(p_path,FileAccess::READ);
	if (!f) {

		ERR_FAIL();
	}

	Ref<ResourceInteractiveLoaderXML> ria = memnew( ResourceInteractiveLoaderXML );
	ria->local_path=GlobalConfig::get_singleton()->localize_path(p_path);
	ria->res_path=ria->local_path;
//	ria->set_local_path( Globals::get_singleton()->localize_path(p_path) );
	ria->get_dependencies(f,p_dependencies,p_add_types);


}

Error ResourceFormatLoaderXML::rename_dependencies(const String &p_path,const Map<String,String>& p_map) {

	FileAccess *f = FileAccess::open(p_path,FileAccess::READ);
	if (!f) {

		ERR_FAIL_V(ERR_CANT_OPEN);
	}

	Ref<ResourceInteractiveLoaderXML> ria = memnew( ResourceInteractiveLoaderXML );
	ria->local_path=GlobalConfig::get_singleton()->localize_path(p_path);
	ria->res_path=ria->local_path;
//	ria->set_local_path( Globals::get_singleton()->localize_path(p_path) );
	return ria->rename_dependencies(f,p_path,p_map);
}


/****************************************************************************************/
/****************************************************************************************/
/****************************************************************************************/
/****************************************************************************************/
/****************************************************************************************/
/****************************************************************************************/
/****************************************************************************************/
/****************************************************************************************/
/****************************************************************************************/
/****************************************************************************************/
/****************************************************************************************/
/****************************************************************************************/
/****************************************************************************************/



void ResourceFormatSaverXMLInstance::escape(String& p_str) {

	p_str=p_str.replace("&","&amp;");
	p_str=p_str.replace("<","&lt;");
	p_str=p_str.replace(">","&gt;");
	p_str=p_str.replace("'","&apos;");
	p_str=p_str.replace("\"","&quot;");
	for (char i=1;i<32;i++) {

		char chr[2]={i,0};
		const char hexn[16]={'0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f'};
		const char hex[8]={'&','#','0','0',hexn[i>>4],hexn[i&0xf],';',0};

		p_str=p_str.replace(chr,hex);
	}


}
void ResourceFormatSaverXMLInstance::write_tabs(int p_diff) {

	for (int i=0;i<depth+p_diff;i++) {

		f->store_8('\t');
	}
}

void ResourceFormatSaverXMLInstance::write_string(String p_str,bool p_escape) {

	/* write an UTF8 string */
	if (p_escape)
		escape(p_str);

	f->store_string(p_str);;
	/*
	CharString cs=p_str.utf8();
	const char *data=cs.get_data();

	while (*data) {
		f->store_8(*data);
		data++;
	}*/


}

void ResourceFormatSaverXMLInstance::enter_tag(const char* p_tag,const String& p_args) {

	f->store_8('<');
	int cc = 0;
	const char *c=p_tag;
	while(*c) {
		cc++;
		c++;
	}
	f->store_buffer((const uint8_t*)p_tag,cc);
	if (p_args.length()) {
		f->store_8(' ');
		f->store_string(p_args);
	}
	f->store_8('>');
	depth++;

}
void ResourceFormatSaverXMLInstance::exit_tag(const char* p_tag) {

	depth--;
	f->store_8('<');
	f->store_8('/');
	int cc = 0;
	const char *c=p_tag;
	while(*c) {
		cc++;
		c++;
	}
	f->store_buffer((const uint8_t*)p_tag,cc);
	f->store_8('>');

}

/*
static bool _check_type(const Variant& p_property) {

	if (p_property.get_type()==Variant::_RID)
		return false;
	if (p_property.get_type()==Variant::OBJECT) {
		RES res = p_property;
		if (res.is_null())
			return false;
	}

	return true;
}*/

void ResourceFormatSaverXMLInstance::write_property(const String& p_name,const Variant& p_property,bool *r_ok) {

	if (r_ok)
		*r_ok=false;

	const char* type;
	String params;
	bool oneliner=true;

	switch( p_property.get_type() ) {

		case Variant::NIL: 		type="nil"; break;
		case Variant::BOOL:		type="bool"; break;
		case Variant::INT: 		type="int"; break;
		case Variant::REAL:		type="real"; break;
		case Variant::STRING:		type="string"; break;
		case Variant::VECTOR2:		type="vector2"; break;
		case Variant::RECT2:		type="rect2"; break;
		case Variant::VECTOR3:		type="vector3"; break;
		case Variant::PLANE:		type="plane"; break;
		case Variant::_AABB:		type="aabb"; break;
		case Variant::QUAT:		type="quaternion"; break;
		case Variant::MATRIX32:		type="matrix32"; break;
		case Variant::MATRIX3:		type="matrix3"; break;
		case Variant::TRANSFORM:		type="transform"; break;
		case Variant::COLOR:		type="color"; break;
		case Variant::IMAGE: {
			type="image";
			Image img=p_property;
			if (img.empty()) {
				write_tabs();
				enter_tag(type,"name=\""+p_name+"\"");
				exit_tag(type);
				if (r_ok)
					*r_ok=true;
				return;
			}
			params+="encoding=\"raw\"";
			params+=" width=\""+itos(img.get_width())+"\"";
			params+=" height=\""+itos(img.get_height())+"\"";
			params+=" mipmaps=\""+itos(img.has_mipmaps())+"\"";
/*
			switch(img.get_format()) {

				case Image::FORMAT_L8: params+=" format=\"grayscale\""; break;
				case Image::FORMAT_INTENSITY: params+=" format=\"intensity\""; break;
				case Image::FORMAT_LA8: params+=" format=\"grayscale_alpha\""; break;
				case Image::FORMAT_RGB8: params+=" format=\"rgb\""; break;
				case Image::FORMAT_RGBA8: params+=" format=\"rgba\""; break;
				case Image::FORMAT_INDEXED : params+=" format=\"indexed\""; break;
				case Image::FORMAT_INDEXED_ALPHA: params+=" format=\"indexed_alpha\""; break;
				case Image::FORMAT_DXT1: params+=" format=\"bc1\""; break;
				case Image::FORMAT_DXT3: params+=" format=\"bc2\""; break;
				case Image::FORMAT_DXT5: params+=" format=\"bc3\""; break;
				case Image::FORMAT_ATI1: params+=" format=\"bc4\""; break;
				case Image::FORMAT_ATI2: params+=" format=\"bc5\""; break;
				case Image::FORMAT_PVRTC2: params+=" format=\"pvrtc2\""; break;
				case Image::FORMAT_PVRTC2A: params+=" format=\"pvrtc2a\""; break;
				case Image::FORMAT_PVRTC4: params+=" format=\"pvrtc4\""; break;
				case Image::FORMAT_PVRTC4A: params+=" format=\"pvrtc4a\""; break;
				case Image::FORMAT_ETC: params+=" format=\"etc\""; break;
				case Image::FORMAT_ATC: params+=" format=\"atc\""; break;
				case Image::FORMAT_ATC_ALPHA_EXPLICIT: params+=" format=\"atcae\""; break;
				case Image::FORMAT_ATC_ALPHA_INTERPOLATED: params+=" format=\"atcai\""; break;
				case Image::FORMAT_CUSTOM: params+=" format=\"custom\" custom_size=\""+itos(img.get_data().size())+"\""; break;
				default: {}
			}*/
		} break;
		case Variant::NODE_PATH:		type="node_path"; break;
		case Variant::OBJECT:	{
			type="resource";
			RES res = p_property;
			if (res.is_null()) {
				write_tabs();
				enter_tag(type,"name=\""+p_name+"\"");
				exit_tag(type);
				if (r_ok)
					*r_ok=true;

				return; // don't save it
			}

			if (external_resources.has(res)) {

				params="external=\""+itos(external_resources[res])+"\"";
			} else {
				params="resource_type=\""+res->get_save_class()+"\"";


				if (res->get_path().length() && res->get_path().find("::")==-1) {
					//external resource
					String path=relative_paths?local_path.path_to_file(res->get_path()):res->get_path();
					escape(path);
					params+=" path=\""+path+"\"";
				} else {

					//internal resource
					ERR_EXPLAIN("Resource was not pre cached for the resource section, bug?");
					ERR_FAIL_COND(!resource_set.has(res));

					params+=" path=\"local://"+itos(res->get_subindex())+"\"";
				}
			}

		} break;
		case Variant::INPUT_EVENT:	type="input_event"; break;
        case Variant::DICTIONARY:	type="dictionary"; params="shared=\""+String(p_property.is_shared()?"true":"false")+"\""; oneliner=false; break;
        case Variant::ARRAY:		type="array"; params="len=\""+itos(p_property.operator Array().size())+"\" shared=\""+String(p_property.is_shared()?"true":"false")+"\""; oneliner=false; break;

		case Variant::RAW_ARRAY:		type="raw_array"; params="len=\""+itos(p_property.operator PoolVector < uint8_t >().size())+"\""; break;
		case Variant::INT_ARRAY:		type="int_array"; params="len=\""+itos(p_property.operator PoolVector < int >().size())+"\""; break;
		case Variant::REAL_ARRAY:	type="real_array"; params="len=\""+itos(p_property.operator PoolVector < real_t >().size())+"\""; break;
		case Variant::STRING_ARRAY:	oneliner=false; type="string_array"; params="len=\""+itos(p_property.operator PoolVector < String >().size())+"\""; break;
		case Variant::VECTOR2_ARRAY:	type="vector2_array"; params="len=\""+itos(p_property.operator PoolVector < Vector2 >().size())+"\""; break;
		case Variant::VECTOR3_ARRAY:	type="vector3_array"; params="len=\""+itos(p_property.operator PoolVector < Vector3 >().size())+"\""; break;
		case Variant::COLOR_ARRAY:	type="color_array"; params="len=\""+itos(p_property.operator PoolVector < Color >().size())+"\""; break;
		default: {

			ERR_PRINT("Unknown Variant type.");
			ERR_FAIL();
		}

	}

	write_tabs();

	if (p_name!="") {
		if (params.length())
			enter_tag(type,"name=\""+p_name+"\" "+params);
		else
			enter_tag(type,"name=\""+p_name+"\"");
	} else {
		if (params.length())
			enter_tag(type," "+params);
		else
			enter_tag(type,String());
	}

	if (!oneliner)
		f->store_8('\n');
	else
		f->store_8(' ');


	switch( p_property.get_type() ) {

		case Variant::NIL: {

		} break;
		case Variant::BOOL: {

			write_string( p_property.operator bool() ? "True":"False" );
		} break;
		case Variant::INT: {

			write_string( itos(p_property.operator int()) );
		} break;
		case Variant::REAL: {

			write_string( rtos(p_property.operator real_t()) );
		} break;
		case Variant::STRING: {

			String str=p_property;
			escape(str);
			str="\""+str+"\"";
			write_string( str,false );
		} break;
		case Variant::VECTOR2: {

			Vector2 v = p_property;
			write_string( rtoss(v.x) +", "+rtoss(v.y) );
		} break;
		case Variant::RECT2: {

			Rect2 aabb = p_property;
			write_string( rtoss(aabb.pos.x) +", "+rtoss(aabb.pos.y) +", "+rtoss(aabb.size.x) +", "+rtoss(aabb.size.y) );

		} break;
		case Variant::VECTOR3: {

			Vector3 v = p_property;
			write_string( rtoss(v.x) +", "+rtoss(v.y)+", "+rtoss(v.z) );
		} break;
		case Variant::PLANE: {

			Plane p = p_property;
			write_string( rtoss(p.normal.x) +", "+rtoss(p.normal.y)+", "+rtoss(p.normal.z)+", "+rtoss(p.d) );

		} break;
		case Variant::_AABB: {

			AABB aabb = p_property;
			write_string( rtoss(aabb.pos.x) +", "+rtoss(aabb.pos.y) +", "+rtoss(aabb.pos.z) +", "+rtoss(aabb.size.x) +", "+rtoss(aabb.size.y) +", "+rtoss(aabb.size.z)  );

		} break;
		case Variant::QUAT: {

			Quat quat = p_property;
			write_string( rtoss(quat.x)+", "+rtoss(quat.y)+", "+rtoss(quat.z)+", "+rtoss(quat.w)+", ");

		} break;
		case Variant::MATRIX32: {

			String s;
			Matrix32 m3 = p_property;
			for (int i=0;i<3;i++) {
				for (int j=0;j<2;j++) {

					if (i!=0 || j!=0)
						s+=", ";
					s+=rtoss( m3.elements[i][j] );
				}
			}

			write_string(s);

		} break;
		case Variant::MATRIX3: {

			String s;
			Matrix3 m3 = p_property;
			for (int i=0;i<3;i++) {
				for (int j=0;j<3;j++) {

					if (i!=0 || j!=0)
						s+=", ";
					s+=rtoss( m3.elements[i][j] );
				}
			}

			write_string(s);

		} break;
		case Variant::TRANSFORM: {

			String s;
			Transform t = p_property;
			Matrix3 &m3 = t.basis;
			for (int i=0;i<3;i++) {
				for (int j=0;j<3;j++) {

					if (i!=0 || j!=0)
						s+=", ";
					s+=rtoss( m3.elements[i][j] );
				}
			}

			s=s+", "+rtoss(t.origin.x) +", "+rtoss(t.origin.y)+", "+rtoss(t.origin.z);

			write_string(s);
		} break;

			// misc types
		case Variant::COLOR: {

			Color c = p_property;
			write_string( rtoss(c.r) +", "+rtoss(c.g)+", "+rtoss(c.b)+", "+rtoss(c.a) );

		} break;
		case Variant::IMAGE: {

			String s;
			Image img = p_property;
			PoolVector<uint8_t> data = img.get_data();
			int len = data.size();
			PoolVector<uint8_t>::Read r = data.read();
			const uint8_t *ptr=r.ptr();;
			for (int i=0;i<len;i++) {

				uint8_t byte = ptr[i];
				const char  hex[16]={'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'};
				char str[3]={ hex[byte>>4], hex[byte&0xF], 0};
				s+=str;
			}

			write_string(s);
		} break;
		case Variant::NODE_PATH: {

			String str=p_property;
			escape(str);
			str="\""+str+"\"";
			write_string( str,false);

		} break;

		case Variant::OBJECT: {
			/* this saver does not save resources in here
			RES res = p_property;

			if (!res.is_null()) {

				String path=res->get_path();
				if (!res->is_shared() || !path.length()) {
					// if no path, or path is from inside a scene
					write_object( *res );
				}

			}
			*/

		} break;
		case Variant::INPUT_EVENT: {

			write_string( p_property.operator String() );
		} break;
		case Variant::DICTIONARY: {

			Dictionary dict = p_property;


			List<Variant> keys;
			dict.get_key_list(&keys);
			keys.sort();

			for(List<Variant>::Element *E=keys.front();E;E=E->next()) {

				//if (!_check_type(dict[E->get()]))
				//	continue;
				bool ok;
				write_property("",E->get(),&ok);
				ERR_CONTINUE(!ok);

				write_property("",dict[E->get()],&ok);
				if (!ok)
					write_property("",Variant()); //at least make the file consistent..
			}




		} break;
		case Variant::ARRAY: {

			Array array = p_property;
			int len=array.size();
			for (int i=0;i<len;i++) {

				write_property("",array[i]);

			}

		} break;

		case Variant::RAW_ARRAY: {

			String s;
			PoolVector<uint8_t> data = p_property;
			int len = data.size();
			PoolVector<uint8_t>::Read r = data.read();
			const uint8_t *ptr=r.ptr();;
			for (int i=0;i<len;i++) {

				uint8_t byte = ptr[i];
				const char  hex[16]={'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'};
				char str[3]={ hex[byte>>4], hex[byte&0xF], 0};
				s+=str;
			}

			write_string(s,false);

		} break;
		case Variant::INT_ARRAY: {

			PoolVector<int> data = p_property;
			int len = data.size();
			PoolVector<int>::Read r = data.read();
			const int *ptr=r.ptr();;
			write_tabs();

			for (int i=0;i<len;i++) {

				if (i>0)
					write_string(", ",false);

				write_string(itos(ptr[i]),false);
			}



		} break;
		case Variant::REAL_ARRAY: {

			PoolVector<real_t> data = p_property;
			int len = data.size();
			PoolVector<real_t>::Read r = data.read();
			const real_t *ptr=r.ptr();;
			write_tabs();
			String cm=", " ;

			for (int i=0;i<len;i++) {

				if (i>0)
					write_string(cm,false);
				write_string(rtoss(ptr[i]),false);
			}


		} break;
		case Variant::STRING_ARRAY: {

			PoolVector<String> data = p_property;
			int len = data.size();
			PoolVector<String>::Read r = data.read();
			const String *ptr=r.ptr();;
			String s;
			//write_string("\n");



			for (int i=0;i<len;i++) {

				write_tabs(0);
				String str=ptr[i];
				escape(str);
				write_string("<string> \""+str+"\" </string>\n",false);
			}
		} break;
		case Variant::VECTOR2_ARRAY: {

			PoolVector<Vector2> data = p_property;
			int len = data.size();
			PoolVector<Vector2>::Read r = data.read();
			const Vector2 *ptr=r.ptr();;
			write_tabs();

			for (int i=0;i<len;i++) {

				if (i>0)
					write_string(", ",false);
				write_string(rtoss(ptr[i].x),false);
				write_string(", "+rtoss(ptr[i].y),false);

			}


		} break;
		case Variant::VECTOR3_ARRAY: {

			PoolVector<Vector3> data = p_property;
			int len = data.size();
			PoolVector<Vector3>::Read r = data.read();
			const Vector3 *ptr=r.ptr();;
			write_tabs();

			for (int i=0;i<len;i++) {

				if (i>0)
					write_string(", ",false);
				write_string(rtoss(ptr[i].x),false);
				write_string(", "+rtoss(ptr[i].y),false);
				write_string(", "+rtoss(ptr[i].z),false);

			}


		} break;
		case Variant::COLOR_ARRAY: {

			PoolVector<Color> data = p_property;
			int len = data.size();
			PoolVector<Color>::Read r = data.read();
			const Color *ptr=r.ptr();;
			write_tabs();

			for (int i=0;i<len;i++) {

				if (i>0)
					write_string(", ",false);

				write_string(rtoss(ptr[i].r),false);
				write_string(", "+rtoss(ptr[i].g),false);
				write_string(", "+rtoss(ptr[i].b),false);
				write_string(", "+rtoss(ptr[i].a),false);

			}

		} break;
		default: {}

	}
	if (oneliner)
		f->store_8(' ');
	else
		write_tabs(-1);
	exit_tag(type);

	f->store_8('\n');

	if (r_ok)
		*r_ok=true;

}


void ResourceFormatSaverXMLInstance::_find_resources(const Variant& p_variant,bool p_main) {


	switch(p_variant.get_type()) {
		case Variant::OBJECT: {


			RES res = p_variant.operator RefPtr();

			if (res.is_null() || external_resources.has(res))
				return;

			if (!p_main && (!bundle_resources ) && res->get_path().length() && res->get_path().find("::") == -1 ) {
				int index = external_resources.size();
				external_resources[res]=index;
				return;
			}

			if (resource_set.has(res))
				return;

			List<PropertyInfo> property_list;

			res->get_property_list( &property_list );
			property_list.sort();

			List<PropertyInfo>::Element *I=property_list.front();

			while(I) {

				PropertyInfo pi=I->get();

				if (pi.usage&PROPERTY_USAGE_STORAGE) {

					Variant v=res->get(I->get().name);
					_find_resources(v);
				}

				I=I->next();
			}

			resource_set.insert( res ); //saved after, so the childs it needs are available when loaded
			saved_resources.push_back(res);

		} break;
		case Variant::ARRAY: {

			Array varray=p_variant;
			int len=varray.size();
			for(int i=0;i<len;i++) {

				Variant v=varray.get(i);
				_find_resources(v);
			}

		} break;
		case Variant::DICTIONARY: {

			Dictionary d=p_variant;
			List<Variant> keys;
			d.get_key_list(&keys);
			for(List<Variant>::Element *E=keys.front();E;E=E->next()) {

				Variant v = d[E->get()];
				_find_resources(v);
			}
		} break;
		default: {}
	}

}



Error ResourceFormatSaverXMLInstance::save(const String &p_path,const RES& p_resource,uint32_t p_flags) {

	Error err;
	f = FileAccess::open(p_path, FileAccess::WRITE,&err);
	ERR_FAIL_COND_V( err, ERR_CANT_OPEN );
	FileAccessRef _fref(f);

	local_path = GlobalConfig::get_singleton()->localize_path(p_path);

	relative_paths=p_flags&ResourceSaver::FLAG_RELATIVE_PATHS;
	skip_editor=p_flags&ResourceSaver::FLAG_OMIT_EDITOR_PROPERTIES;
	bundle_resources=p_flags&ResourceSaver::FLAG_BUNDLE_RESOURCES;
	takeover_paths=p_flags&ResourceSaver::FLAG_REPLACE_SUBRESOURCE_PATHS;
	if (!p_path.begins_with("res://")) {
		takeover_paths=false;
	}
	depth=0;

	// save resources
	_find_resources(p_resource,true);

	ERR_FAIL_COND_V(err!=OK,err);

	write_string("<?xml version=\"1.0\" encoding=\"UTF-8\" ?>",false); //no escape
	write_string("\n",false);
	enter_tag("resource_file","type=\""+p_resource->get_class()+"\" subresource_count=\""+itos(saved_resources.size()+external_resources.size())+"\" version=\""+itos(VERSION_MAJOR)+"."+itos(VERSION_MINOR)+"\" version_name=\""+VERSION_FULL_NAME+"\"");
	write_string("\n",false);

	for(Map<RES,int>::Element *E=external_resources.front();E;E=E->next()) {

		write_tabs();
		String p = E->key()->get_path();

		enter_tag("ext_resource","path=\""+p+"\" type=\""+E->key()->get_save_class()+"\" index=\""+itos(E->get())+"\""); //bundled
		exit_tag("ext_resource"); //bundled
		write_string("\n",false);
	}

	Set<int> used_indices;

	for(List<RES>::Element *E=saved_resources.front();E;E=E->next()) {

		RES res = E->get();
		if (E->next() && (res->get_path()=="" || res->get_path().find("::") != -1 )) {

			if (res->get_subindex()!=0) {
				if (used_indices.has(res->get_subindex())) {
					res->set_subindex(0); //repeated
				} else {
					used_indices.insert(res->get_subindex());
				}
			}
		}
	}

	for(List<RES>::Element *E=saved_resources.front();E;E=E->next()) {

		RES res = E->get();
		ERR_CONTINUE(!resource_set.has(res));
		bool main = (E->next()==NULL);

		write_tabs();

		if (main)
			enter_tag("main_resource",""); //bundled
		else if (res->get_path().length() && res->get_path().find("::") == -1 )
			enter_tag("resource","type=\""+res->get_class()+"\" path=\""+res->get_path()+"\""); //bundled
		else {

			if (res->get_subindex()==0) {
				int new_subindex=1;
				if (used_indices.size()) {
					new_subindex=used_indices.back()->get()+1;
				}

				res->set_subindex(new_subindex);
				used_indices.insert(new_subindex);
			}

			int idx = res->get_subindex();
			enter_tag("resource","type=\""+res->get_class()+"\" path=\"local://"+itos(idx)+"\"");
			if (takeover_paths) {
				res->set_path(p_path+"::"+itos(idx),true);
			}
#ifdef TOOLS_ENABLED
			res->set_edited(false);
#endif


		}
		write_string("\n",false);


		List<PropertyInfo> property_list;
		res->get_property_list(&property_list);
//		property_list.sort();
		for(List<PropertyInfo>::Element *PE = property_list.front();PE;PE=PE->next()) {


			if (skip_editor && PE->get().name.begins_with("__editor"))
				continue;

			if (PE->get().usage&PROPERTY_USAGE_STORAGE ) {

				String name = PE->get().name;
				Variant value = res->get(name);


				if ((PE->get().usage&PROPERTY_USAGE_STORE_IF_NONZERO && value.is_zero())||(PE->get().usage&PROPERTY_USAGE_STORE_IF_NONONE && value.is_one()) )
					continue;


				write_property(name,value);
			}


		}

		write_string("\n",false);
		write_tabs(-1);
		if (main)
			exit_tag("main_resource");
		else
			exit_tag("resource");

		write_string("\n",false);
	}

	exit_tag("resource_file");
	if (f->get_error()!=OK && f->get_error()!=ERR_FILE_EOF) {
		f->close();
		return ERR_CANT_CREATE;
	}

	f->close();
	//memdelete(f);

	return OK;
}



Error ResourceFormatSaverXML::save(const String &p_path,const RES& p_resource,uint32_t p_flags) {

	ResourceFormatSaverXMLInstance saver;
	return saver.save(p_path,p_resource,p_flags);

}

bool ResourceFormatSaverXML::recognize(const RES& p_resource) const {


	return true; // all recognized!
}
void ResourceFormatSaverXML::get_recognized_extensions(const RES& p_resource,List<String> *p_extensions) const {


	//here comes the sun, lalalala
	String base = p_resource->get_base_extension().to_lower();
	p_extensions->push_back("xml");
	if (base!="res") {

		p_extensions->push_back("x"+base);
	}

}

ResourceFormatSaverXML* ResourceFormatSaverXML::singleton=NULL;
ResourceFormatSaverXML::ResourceFormatSaverXML() {
	singleton=this;
}
