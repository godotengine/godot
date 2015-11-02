/*************************************************************************/
/*  marshalls.cpp                                                        */
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
#include "marshalls.h"
#include "print_string.h"
#include "os/keyboard.h"
#include <stdio.h>

Error decode_variant(Variant& r_variant,const uint8_t *p_buffer, int p_len,int *r_len) {

	const uint8_t * buf=p_buffer;
	int len=p_len;

	if (len<4) {

		ERR_FAIL_COND_V(len<4,ERR_INVALID_DATA);
	}


	uint32_t type=decode_uint32(buf);

	ERR_FAIL_COND_V(type>=Variant::VARIANT_MAX,ERR_INVALID_DATA);

	buf+=4;
	len-=4;
	if (r_len)
		*r_len=4;

	switch(type) {

		case Variant::NIL: {

			r_variant=Variant();
		} break;
		case Variant::BOOL: {

			ERR_FAIL_COND_V(len<4,ERR_INVALID_DATA);
			bool val = decode_uint32(buf);
			r_variant=val;
			if (r_len)
				(*r_len)+=4;
		} break;
		case Variant::INT: {

			ERR_FAIL_COND_V(len<4,ERR_INVALID_DATA);
			int val = decode_uint32(buf);
			r_variant=val;
			if (r_len)
				(*r_len)+=4;

		} break;
		case Variant::REAL: {

			ERR_FAIL_COND_V(len<(int)4,ERR_INVALID_DATA);
			float val = decode_float(buf);
			r_variant=val;
			if (r_len)
				(*r_len)+=4;

		} break;
		case Variant::STRING: {

			ERR_FAIL_COND_V(len<4,ERR_INVALID_DATA);
			uint32_t strlen = decode_uint32(buf);
			buf+=4;
			len-=4;
			ERR_FAIL_COND_V((int)strlen>len,ERR_INVALID_DATA);

			String str;
			str.parse_utf8((const char*)buf,strlen);
			r_variant=str;

			if (r_len) {				
				if (strlen%4)
					(*r_len)+=4-strlen%4;
				(*r_len)+=4+strlen;

			}

		} break;
		// math types

		case Variant::VECTOR2: {

			ERR_FAIL_COND_V(len<(int)4*2,ERR_INVALID_DATA);
			Vector2 val;
			val.x=decode_float(&buf[0]);
			val.y=decode_float(&buf[4]);
			r_variant=val;

			if (r_len)
				(*r_len)+=4*2;

		} break;		// 5
		case Variant::RECT2: {

			ERR_FAIL_COND_V(len<(int)4*4,ERR_INVALID_DATA);
			Rect2 val;
			val.pos.x=decode_float(&buf[0]);
			val.pos.y=decode_float(&buf[4]);
			val.size.x=decode_float(&buf[8]);
			val.size.y=decode_float(&buf[12]);
			r_variant=val;

			if (r_len)
				(*r_len)+=4*4;

		} break;
		case Variant::VECTOR3: {

			ERR_FAIL_COND_V(len<(int)4*3,ERR_INVALID_DATA);
			Vector3 val;
			val.x=decode_float(&buf[0]);
			val.y=decode_float(&buf[4]);
			val.z=decode_float(&buf[8]);
			r_variant=val;

			if (r_len)
				(*r_len)+=4*3;

		} break;
		case Variant::MATRIX32: {

			ERR_FAIL_COND_V(len<(int)4*6,ERR_INVALID_DATA);
			Matrix32 val;
			for(int i=0;i<3;i++) {
				for(int j=0;j<2;j++) {

					val.elements[i][j]=decode_float(&buf[(i*2+j)*4]);
				}
			}

			r_variant=val;

			if (r_len)
				(*r_len)+=4*6;

		} break;
		case Variant::PLANE: {

			ERR_FAIL_COND_V(len<(int)4*4,ERR_INVALID_DATA);
			Plane val;
			val.normal.x=decode_float(&buf[0]);
			val.normal.y=decode_float(&buf[4]);
			val.normal.z=decode_float(&buf[8]);
			val.d=decode_float(&buf[12]);
			r_variant=val;

			if (r_len)
				(*r_len)+=4*4;

		} break;
		case Variant::QUAT: {

			ERR_FAIL_COND_V(len<(int)4*4,ERR_INVALID_DATA);
			Quat val;
			val.x=decode_float(&buf[0]);
			val.y=decode_float(&buf[4]);
			val.z=decode_float(&buf[8]);
			val.w=decode_float(&buf[12]);
			r_variant=val;

			if (r_len)
				(*r_len)+=4*4;

		} break;
		case Variant::_AABB: {

			ERR_FAIL_COND_V(len<(int)4*6,ERR_INVALID_DATA);
			AABB val;
			val.pos.x=decode_float(&buf[0]);
			val.pos.y=decode_float(&buf[4]);
			val.pos.z=decode_float(&buf[8]);
			val.size.x=decode_float(&buf[12]);
			val.size.y=decode_float(&buf[16]);
			val.size.z=decode_float(&buf[20]);
			r_variant=val;

			if (r_len)
				(*r_len)+=4*6;

		} break;
		case Variant::MATRIX3: {

			ERR_FAIL_COND_V(len<(int)4*9,ERR_INVALID_DATA);
			Matrix3 val;
			for(int i=0;i<3;i++) {
				for(int j=0;j<3;j++) {

					val.elements[i][j]=decode_float(&buf[(i*3+j)*4]);
				}
			}

			r_variant=val;

			if (r_len)
				(*r_len)+=4*9;

		} break;
		case Variant::TRANSFORM: {

			ERR_FAIL_COND_V(len<(int)4*12,ERR_INVALID_DATA);
			Transform val;
			for(int i=0;i<3;i++) {
				for(int j=0;j<3;j++) {

					val.basis.elements[i][j]=decode_float(&buf[(i*3+j)*4]);
				}
			}
			val.origin[0]=decode_float(&buf[36]);
			val.origin[1]=decode_float(&buf[40]);
			val.origin[2]=decode_float(&buf[44]);

			r_variant=val;

			if (r_len)
				(*r_len)+=4*12;

		} break;

		// misc types
		case Variant::COLOR: {

			ERR_FAIL_COND_V(len<(int)4*4,ERR_INVALID_DATA);
			Color val;
			val.r=decode_float(&buf[0]);
			val.g=decode_float(&buf[4]);
			val.b=decode_float(&buf[8]);
			val.a=decode_float(&buf[12]);
			r_variant=val;

			if (r_len)
				(*r_len)+=4*4;

		} break;
		case Variant::IMAGE: {

			ERR_FAIL_COND_V(len<(int)5*4,ERR_INVALID_DATA);
			Image::Format fmt = (Image::Format)decode_uint32(&buf[0]);
			ERR_FAIL_INDEX_V( fmt, Image::FORMAT_MAX, ERR_INVALID_DATA);
			uint32_t mipmaps = decode_uint32(&buf[4]);
			uint32_t w = decode_uint32(&buf[8]);
			uint32_t h = decode_uint32(&buf[12]);
			uint32_t datalen = decode_uint32(&buf[16]);

			Image img;
			if (datalen>0) {
				len-=5*4;
				ERR_FAIL_COND_V( len < datalen, ERR_INVALID_DATA );
				DVector<uint8_t> data;
				data.resize(datalen);
				DVector<uint8_t>::Write wr = data.write();
				copymem(&wr[0],&buf[20],datalen);
				wr = DVector<uint8_t>::Write();



				img=Image(w,h,mipmaps,fmt,data);
			}

			r_variant=img;
			if (r_len) {
				if (datalen%4)
					(*r_len)+=4-datalen%4;

				(*r_len)+=4*5+datalen;
			}

		} break;
		case Variant::NODE_PATH: {

			ERR_FAIL_COND_V(len<4,ERR_INVALID_DATA);			
			uint32_t strlen = decode_uint32(buf);

			if (strlen&0x80000000) {
				//new format
				ERR_FAIL_COND_V(len<12,ERR_INVALID_DATA);
				Vector<StringName> names;
				Vector<StringName> subnames;
				StringName prop;

				uint32_t namecount=strlen&=0x7FFFFFFF;
				uint32_t subnamecount = decode_uint32(buf+4);
				uint32_t flags = decode_uint32(buf+8);

				len-=12;
				buf+=12;

				int total=namecount+subnamecount;
				if (flags&2)
					total++;

				if (r_len)
					(*r_len)+=12;


				for(int i=0;i<total;i++) {

					ERR_FAIL_COND_V((int)len<4,ERR_INVALID_DATA);
					strlen = decode_uint32(buf);

					int pad=0;

					if (strlen%4)
						pad+=4-strlen%4;

					buf+=4;
					len-=4;
					ERR_FAIL_COND_V((int)strlen+pad>len,ERR_INVALID_DATA);

					String str;
					str.parse_utf8((const char*)buf,strlen);


					if (i<namecount)
						names.push_back(str);
					else if (i<namecount+subnamecount)
						subnames.push_back(str);
					else
						prop=str;

					buf+=strlen+pad;
					len-=strlen+pad;

					if (r_len)
						(*r_len)+=4+strlen+pad;

				}

				r_variant=NodePath(names,subnames,flags&1,prop);

			} else {
				//old format, just a string

				buf+=4;
				len-=4;
				ERR_FAIL_COND_V((int)strlen>len,ERR_INVALID_DATA);


				String str;
				str.parse_utf8((const char*)buf,strlen);

				r_variant=NodePath(str);

				if (r_len)
					(*r_len)+=4+strlen;
			}

		} break;
		/*case Variant::RESOURCE: {

			ERR_EXPLAIN("Can't marshallize resources");
			ERR_FAIL_V(ERR_INVALID_DATA); //no, i'm sorry, no go
		} break;*/
		case Variant::_RID: {

			r_variant = RID();
		} break;
		case Variant::OBJECT: {


			r_variant = (Object*)NULL;
		} break;
		case Variant::INPUT_EVENT: {

			InputEvent ie;

			ie.type=decode_uint32(&buf[0]);
			ie.device=decode_uint32(&buf[4]);

			if (r_len)
				(*r_len)+=12;

			switch(ie.type) {

				case InputEvent::KEY: {

					uint32_t mods=decode_uint32(&buf[12]);
					if (mods&KEY_MASK_SHIFT)
						ie.key.mod.shift=true;
					if (mods&KEY_MASK_CTRL)
						ie.key.mod.control=true;
					if (mods&KEY_MASK_ALT)
						ie.key.mod.alt=true;
					if (mods&KEY_MASK_META)
						ie.key.mod.meta=true;
					ie.key.scancode=decode_uint32(&buf[16]);

					if (r_len)
						(*r_len)+=8;


				} break;
				case InputEvent::MOUSE_BUTTON: {

					ie.mouse_button.button_index=decode_uint32(&buf[12]);
					if (r_len)
						(*r_len)+=4;

				} break;
				case InputEvent::JOYSTICK_BUTTON: {

					ie.joy_button.button_index=decode_uint32(&buf[12]);
					if (r_len)
						(*r_len)+=4;
				} break;
				case InputEvent::SCREEN_TOUCH: {

					ie.screen_touch.index=decode_uint32(&buf[12]);
					if (r_len)
						(*r_len)+=4;
				} break;
				case InputEvent::JOYSTICK_MOTION: {

					ie.joy_motion.axis=decode_uint32(&buf[12]);
					if (r_len)
						(*r_len)+=4;
				} break;
			}

			r_variant = ie;

		} break;
		case Variant::DICTIONARY: {

			ERR_FAIL_COND_V(len<4,ERR_INVALID_DATA);
            uint32_t count = decode_uint32(buf);
            bool shared = count&0x80000000;
            count&=0x7FFFFFFF;

			buf+=4;
			len-=4;

			if (r_len) {
				(*r_len)+=4;
			}

            Dictionary d(shared);

            for(uint32_t i=0;i<count;i++) {

				Variant key,value;

				int used;
				Error err = decode_variant(key,buf,len,&used);
				ERR_FAIL_COND_V(err,err);

				buf+=used;
				len-=used;
				if (r_len) {
					(*r_len)+=used;
				}

				err = decode_variant(value,buf,len,&used);
				ERR_FAIL_COND_V(err,err);

				buf+=used;
				len-=used;
				if (r_len) {
					(*r_len)+=used;
				}

				d[key]=value;
			}

			r_variant=d;

		} break;
		case Variant::ARRAY: {

			ERR_FAIL_COND_V(len<4,ERR_INVALID_DATA);
			uint32_t count = decode_uint32(buf);
            bool shared = count&0x80000000;
            count&=0x7FFFFFFF;

			buf+=4;
			len-=4;

			if (r_len) {
				(*r_len)+=4;
			}

            Array varr(shared);

            for(uint32_t i=0;i<count;i++) {

				int used=0;
				Variant v;
				Error err = decode_variant(v,buf,len,&used);
				ERR_FAIL_COND_V(err,err);
				buf+=used;
				len-=used;
				varr.push_back(v);
				if (r_len) {
					(*r_len)+=used;
				}
			}

			r_variant=varr;


		} break;

		// arrays
		case Variant::RAW_ARRAY: {

			ERR_FAIL_COND_V(len<4,ERR_INVALID_DATA);
			uint32_t count = decode_uint32(buf);
			buf+=4;
			len-=4;
			ERR_FAIL_COND_V((int)count>len,ERR_INVALID_DATA);


			DVector<uint8_t> data;

			if (count) {
				data.resize(count);
				DVector<uint8_t>::Write w = data.write();
				for(int i=0;i<count;i++) {

					w[i]=buf[i];
				}

				w = DVector<uint8_t>::Write();
			}

			r_variant=data;

			if (r_len) {
				if (count%4)
					(*r_len)+=4-count%4;
				(*r_len)+=4+count;
			}



		} break;
		case Variant::INT_ARRAY: {

			ERR_FAIL_COND_V(len<4,ERR_INVALID_DATA);
			uint32_t count = decode_uint32(buf);
			buf+=4;
			len-=4;
			ERR_FAIL_COND_V((int)count*4>len,ERR_INVALID_DATA);

			DVector<int> data;

			if (count) {
				//const int*rbuf=(const int*)buf;
				data.resize(count);
				DVector<int>::Write w = data.write();
				for(int i=0;i<count;i++) {

					w[i]=decode_uint32(&buf[i*4]);
				}

				w = DVector<int>::Write();
			}
			r_variant=Variant(data);
			if (r_len) {
				(*r_len)+=4+count*sizeof(int);
			}

		} break;
		case Variant::REAL_ARRAY: {

			ERR_FAIL_COND_V(len<4,ERR_INVALID_DATA);
			uint32_t count = decode_uint32(buf);
			buf+=4;
			len-=4;
			ERR_FAIL_COND_V((int)count*4>len,ERR_INVALID_DATA);

			DVector<float> data;

			if (count) {
				//const float*rbuf=(const float*)buf;
				data.resize(count);
				DVector<float>::Write w = data.write();
				for(int i=0;i<count;i++) {

					w[i]=decode_float(&buf[i*4]);
				}

				w = DVector<float>::Write();
			}
			r_variant=data;

			if (r_len) {
				(*r_len)+=4+count*sizeof(float);
			}


		} break;
		case Variant::STRING_ARRAY: {

			ERR_FAIL_COND_V(len<4,ERR_INVALID_DATA);
			uint32_t count = decode_uint32(buf);
			ERR_FAIL_COND_V(count<0,ERR_INVALID_DATA);

			DVector<String> strings;
			buf+=4;
			len-=4;

			if (r_len)
				(*r_len)+=4;
			//printf("string count: %i\n",count);

			for(int i=0;i<(int)count;i++) {

				ERR_FAIL_COND_V(len<4,ERR_INVALID_DATA);
				uint32_t strlen = decode_uint32(buf);

				buf+=4;
				len-=4;
				ERR_FAIL_COND_V((int)strlen>len,ERR_INVALID_DATA);

				//printf("loaded string: %s\n",(const char*)buf);
				String str;
				str.parse_utf8((const char*)buf,strlen);

				strings.push_back(str);

				buf+=strlen;
				len-=strlen;

				if (r_len)
					(*r_len)+=4+strlen;

				if (strlen%4) {
					int pad = 4-(strlen%4);
					buf+=pad;
					len-=pad;
					if (r_len) {
						(*r_len)+=pad;
					}
				}

			}

			r_variant=strings;


		} break;
		case Variant::VECTOR2_ARRAY: {

			ERR_FAIL_COND_V(len<4,ERR_INVALID_DATA);
			uint32_t count = decode_uint32(buf);
			ERR_FAIL_COND_V(count<0,ERR_INVALID_DATA);
			buf+=4;
			len-=4;

			ERR_FAIL_COND_V((int)count*4*2>len,ERR_INVALID_DATA);
			DVector<Vector2> varray;

			if (r_len) {
				(*r_len)+=4;
			}

			if (count) {
				varray.resize(count);
				DVector<Vector2>::Write w = varray.write();
				const float *r = (const float*)buf;

				for(int i=0;i<(int)count;i++) {

					w[i].x=decode_float(buf+i*4*2+4*0);
					w[i].y=decode_float(buf+i*4*2+4*1);

				}

				int adv = 4*2*count;

				if (r_len)
					(*r_len)+=adv;
				len-=adv;
				buf+=adv;

			}

			r_variant=varray;

		} break;
		case Variant::VECTOR3_ARRAY: {

			ERR_FAIL_COND_V(len<4,ERR_INVALID_DATA);
			uint32_t count = decode_uint32(buf);
			ERR_FAIL_COND_V(count<0,ERR_INVALID_DATA);
			buf+=4;
			len-=4;

			ERR_FAIL_COND_V((int)count*4*3>len,ERR_INVALID_DATA);
			DVector<Vector3> varray;

			if (r_len) {
				(*r_len)+=4;
			}

			if (count) {
				varray.resize(count);
				DVector<Vector3>::Write w = varray.write();
				const float *r = (const float*)buf;

				for(int i=0;i<(int)count;i++) {

					w[i].x=decode_float(buf+i*4*3+4*0);
					w[i].y=decode_float(buf+i*4*3+4*1);
					w[i].z=decode_float(buf+i*4*3+4*2);

				}

				int adv = 4*3*count;

				if (r_len)
					(*r_len)+=adv;
				len-=adv;
				buf+=adv;

			}

			r_variant=varray;

		} break;
		case Variant::COLOR_ARRAY: {

			ERR_FAIL_COND_V(len<4,ERR_INVALID_DATA);
			uint32_t count = decode_uint32(buf);
			ERR_FAIL_COND_V(count<0,ERR_INVALID_DATA);
			buf+=4;
			len-=4;

			ERR_FAIL_COND_V((int)count*4*4>len,ERR_INVALID_DATA);
			DVector<Color> carray;

			if (r_len) {
				(*r_len)+=4;
			}

			if (count) {
				carray.resize(count);
				DVector<Color>::Write w = carray.write();
				const float *r = (const float*)buf;

				for(int i=0;i<(int)count;i++) {

					w[i].r=decode_float(buf+i*4*4+4*0);
					w[i].g=decode_float(buf+i*4*4+4*1);
					w[i].b=decode_float(buf+i*4*4+4*2);
					w[i].a=decode_float(buf+i*4*4+4*3);

				}

				int adv = 4*4*count;

				if (r_len)
					(*r_len)+=adv;
				len-=adv;
				buf+=adv;

			}

			r_variant=carray;

		} break;
		default: { ERR_FAIL_V(ERR_BUG); }
	}

	return OK;
}

Error encode_variant(const Variant& p_variant, uint8_t *r_buffer, int &r_len) {

	uint8_t * buf=r_buffer;

	r_len=0;

	if (buf) {
		encode_uint32(p_variant.get_type(),buf);
		buf+=4;
	}
	r_len+=4;

	switch(p_variant.get_type()) {

		case Variant::NIL: {

			//nothing to do
		} break;
		case Variant::BOOL: {

			if (buf) {
				encode_uint32(p_variant.operator bool(),buf);
			}

			r_len+=4;

		} break;
		case Variant::INT: {

			if (buf) {
				encode_uint32(p_variant.operator int(),buf);
			}

			r_len+=4;

		} break;
		case Variant::REAL: {

			if (buf) {
				encode_float(p_variant.operator float(),buf);
			}

			r_len+=4;

		} break;
		case Variant::NODE_PATH: {

			NodePath np=p_variant;
			if (buf) {
				encode_uint32(uint32_t(np.get_name_count())|0x80000000,buf);	//for compatibility with the old format
				encode_uint32(np.get_subname_count(),buf+4);
				uint32_t flags=0;
				if (np.is_absolute())
					flags|=1;
				if (np.get_property()!=StringName())
					flags|=2;

				encode_uint32(flags,buf+8);

				buf+=12;
			}

			r_len+=12;

			int total = np.get_name_count()+np.get_subname_count();
			if (np.get_property()!=StringName())
				total++;

			for(int i=0;i<total;i++) {

				String str;

				if (i<np.get_name_count())
					str=np.get_name(i);
				else if (i<np.get_name_count()+np.get_subname_count())
					str=np.get_subname(i-np.get_subname_count());
				else
					str=np.get_property();

				CharString utf8 = str.utf8();

				int pad = 0;

				if (utf8.length()%4)
					pad=4-utf8.length()%4;

				if (buf) {
					encode_uint32(utf8.length(),buf);
					buf+=4;
					copymem(buf,utf8.get_data(),utf8.length());
					buf+=pad+utf8.length();
				}


				r_len+=4+utf8.length()+pad;
			}

		} break;
		case Variant::STRING: {


			CharString utf8 = p_variant.operator String().utf8();

			if (buf) {
				encode_uint32(utf8.length(),buf);
				buf+=4;
				copymem(buf,utf8.get_data(),utf8.length());
			}

			r_len+=4+utf8.length();
			while (r_len%4)
				r_len++; //pad

		} break;
		// math types

		case Variant::VECTOR2: {

			if (buf) {
				Vector2 v2=p_variant;
				encode_float(v2.x,&buf[0]);
				encode_float(v2.y,&buf[4]);

			}

			r_len+=2*4;

		} break;		// 5
		case Variant::RECT2: {

			if (buf) {
				Rect2 r2=p_variant;
				encode_float(r2.pos.x,&buf[0]);
				encode_float(r2.pos.y,&buf[4]);
				encode_float(r2.size.x,&buf[8]);
				encode_float(r2.size.y,&buf[12]);
			}
			r_len+=4*4;

		} break;
		case Variant::VECTOR3: {

			if (buf) {
				Vector3 v3=p_variant;
				encode_float(v3.x,&buf[0]);
				encode_float(v3.y,&buf[4]);
				encode_float(v3.z,&buf[8]);
			}

			r_len+=3*4;

		} break;
		case Variant::MATRIX32: {

			if (buf) {
				Matrix32 val=p_variant;
				for(int i=0;i<3;i++) {
					for(int j=0;j<2;j++) {

						copymem(&buf[(i*2+j)*4],&val.elements[i][j],sizeof(float));
					}
				}
			}


			r_len+=6*4;

		} break;
		case Variant::PLANE: {

			if (buf) {
				Plane p=p_variant;
				encode_float(p.normal.x,&buf[0]);
				encode_float(p.normal.y,&buf[4]);
				encode_float(p.normal.z,&buf[8]);
				encode_float(p.d,&buf[12]);
			}

			r_len+=4*4;

		} break;
		case Variant::QUAT: {

			if (buf) {
				Quat q=p_variant;
				encode_float(q.x,&buf[0]);
				encode_float(q.y,&buf[4]);
				encode_float(q.z,&buf[8]);
				encode_float(q.w,&buf[12]);
			}

			r_len+=4*4;

		} break;
		case Variant::_AABB: {

			if (buf) {
				AABB aabb=p_variant;
				encode_float(aabb.pos.x,&buf[0]);
				encode_float(aabb.pos.y,&buf[4]);
				encode_float(aabb.pos.z,&buf[8]);
				encode_float(aabb.size.x,&buf[12]);
				encode_float(aabb.size.y,&buf[16]);
				encode_float(aabb.size.z,&buf[20]);
			}

			r_len+=6*4;


		} break;
		case Variant::MATRIX3: {

			if (buf) {
				Matrix3 val=p_variant;
				for(int i=0;i<3;i++) {
					for(int j=0;j<3;j++) {

						copymem(&buf[(i*3+j)*4],&val.elements[i][j],sizeof(float));
					}
				}
			}


			r_len+=9*4;

		} break;
		case Variant::TRANSFORM: {

			if (buf) {
				Transform val=p_variant;
				for(int i=0;i<3;i++) {
					for(int j=0;j<3;j++) {

						copymem(&buf[(i*3+j)*4],&val.basis.elements[i][j],sizeof(float));
					}
				}

				encode_float(val.origin.x,&buf[36]);
				encode_float(val.origin.y,&buf[40]);
				encode_float(val.origin.z,&buf[44]);


			}

			r_len+=12*4;

		} break;

		// misc types
		case Variant::COLOR: {

			if (buf) {
				Color c=p_variant;
				encode_float(c.r,&buf[0]);
				encode_float(c.g,&buf[4]);
				encode_float(c.b,&buf[8]);
				encode_float(c.a,&buf[12]);
			}

			r_len+=4*4;

		} break;
		case Variant::IMAGE: {

			Image image = p_variant;
			DVector<uint8_t> data=image.get_data();

			if (buf) {

				encode_uint32(image.get_format(),&buf[0]);
				encode_uint32(image.get_mipmaps(),&buf[4]);
				encode_uint32(image.get_width(),&buf[8]);
				encode_uint32(image.get_height(),&buf[12]);
				int ds=data.size();
				encode_uint32(ds,&buf[16]);
				DVector<uint8_t>::Read r = data.read();
				copymem(&buf[20],&r[0],ds);
			}

			int pad=0;
			if (data.size()%4)
				pad=4-data.size()%4;

			r_len+=data.size()+5*4+pad;

		} break;		
		/*case Variant::RESOURCE: {

			ERR_EXPLAIN("Can't marshallize resources");
			ERR_FAIL_V(ERR_INVALID_DATA); //no, i'm sorry, no go
		} break;*/
		case Variant::_RID:
		case Variant::OBJECT: {


		} break;
		case Variant::INPUT_EVENT: {


			InputEvent ie=p_variant;

			if (buf) {

				encode_uint32(ie.type,&buf[0]);
				encode_uint32(ie.device,&buf[4]);
				encode_uint32(0,&buf[8]);
			}

			int llen=12;

			switch(ie.type) {

				case InputEvent::KEY: {

					if (buf) {

						uint32_t mods=0;
						if (ie.key.mod.shift)
							mods|=KEY_MASK_SHIFT;
						if (ie.key.mod.control)
							mods|=KEY_MASK_CTRL;
						if (ie.key.mod.alt)
							mods|=KEY_MASK_ALT;
						if (ie.key.mod.meta)
							mods|=KEY_MASK_META;

						encode_uint32(mods,&buf[llen]);
						encode_uint32(ie.key.scancode,&buf[llen+4]);
					}
					llen+=8;

				} break;
				case InputEvent::MOUSE_BUTTON: {

					if (buf) {

						encode_uint32(ie.mouse_button.button_index,&buf[llen]);
					}
					llen+=4;
				} break;
				case InputEvent::JOYSTICK_BUTTON: {

					if (buf) {

						encode_uint32(ie.joy_button.button_index,&buf[llen]);
					}
					llen+=4;
				} break;
				case InputEvent::SCREEN_TOUCH: {

					if (buf) {

						encode_uint32(ie.screen_touch.index,&buf[llen]);
					}
					llen+=4;
				} break;
				case InputEvent::JOYSTICK_MOTION: {

					if (buf) {

						int axis = ie.joy_motion.axis;
						encode_uint32(axis,&buf[llen]);
					}
					llen+=4;
				} break;
			}

			if (buf)
				encode_uint32(llen,&buf[8]);
			r_len+=llen;


			// not supported
		} break;
		case Variant::DICTIONARY: {

			Dictionary d = p_variant;

			if (buf) {
                encode_uint32(uint32_t(d.size())|(d.is_shared()?0x80000000:0),buf);
				buf+=4;
			}
			r_len+=4;

			List<Variant> keys;
			d.get_key_list(&keys);


			for(List<Variant>::Element *E=keys.front();E;E=E->next()) {

				/*
				CharString utf8 = E->->utf8();

				if (buf) {
					encode_uint32(utf8.length()+1,buf);
					buf+=4;
					copymem(buf,utf8.get_data(),utf8.length()+1);
				}

				r_len+=4+utf8.length()+1;
				while (r_len%4)
					r_len++; //pad
				*/
				int len;
				encode_variant(E->get(),buf,len);
				ERR_FAIL_COND_V(len%4,ERR_BUG);
				r_len+=len;
				if (buf)
					buf += len;
				encode_variant(d[E->get()],buf,len);
				ERR_FAIL_COND_V(len%4,ERR_BUG);
				r_len+=len;
				if (buf)
					buf += len;
			}

		} break;
		case Variant::ARRAY: {

			Array v = p_variant;

			if (buf) {
                encode_uint32(uint32_t(v.size())|(v.is_shared()?0x80000000:0),buf);
				buf+=4;
			}

			r_len+=4;

			for(int i=0;i<v.size();i++) {

				int len;
				encode_variant(v.get(i),buf,len);
				ERR_FAIL_COND_V(len%4,ERR_BUG);
				r_len+=len;
				if (buf)
					buf+=len;
			}


		} break;
		// arrays
		case Variant::RAW_ARRAY: {

			DVector<uint8_t> data = p_variant;
			int datalen=data.size();
			int datasize=sizeof(uint8_t);
			
			if (buf) {
				encode_uint32(datalen,buf);
				buf+=4;
				DVector<uint8_t>::Read r = data.read();
				copymem(buf,&r[0],datalen*datasize);

			}

			r_len+=4+datalen*datasize;
			while(r_len%4)
				r_len++;

		} break;
		case Variant::INT_ARRAY: {

			DVector<int> data = p_variant;
			int datalen=data.size();
			int datasize=sizeof(int32_t);

			if (buf) {
				encode_uint32(datalen,buf);
				buf+=4;
				DVector<int>::Read r = data.read();
				for(int i=0;i<datalen;i++)
					encode_uint32(r[i],&buf[i*datasize]);

			}

			r_len+=4+datalen*datasize;

		} break;
		case Variant::REAL_ARRAY: {

			DVector<real_t> data = p_variant;
			int datalen=data.size();
			int datasize=sizeof(real_t);

			if (buf) {
				encode_uint32(datalen,buf);
				buf+=4;
				DVector<real_t>::Read r = data.read();
				for(int i=0;i<datalen;i++)
					encode_float(r[i],&buf[i*datasize]);

			}

			r_len+=4+datalen*datasize;

		} break;
		case Variant::STRING_ARRAY: {


			DVector<String> data = p_variant;
			int len=data.size();

			if (buf) {
				encode_uint32(len,buf);
				buf+=4;
			}

			r_len+=4;

			for(int i=0;i<len;i++) {


				CharString utf8 = data.get(i).utf8();

				if (buf) {
					encode_uint32(utf8.length()+1,buf);
					buf+=4;
					copymem(buf,utf8.get_data(),utf8.length()+1);
					buf+=utf8.length()+1;
				}

				r_len+=4+utf8.length()+1;
				while (r_len%4) {
					r_len++; //pad
					if (buf)
						buf++;
				}
			}

		} break;
		case Variant::VECTOR2_ARRAY: {

			DVector<Vector2> data = p_variant;
			int len=data.size();

			if (buf) {
				encode_uint32(len,buf);
				buf+=4;
			}

			r_len+=4;

			if (buf) {

				for(int i=0;i<len;i++) {

					Vector2 v = data.get(i);

					encode_float(v.x,&buf[0]);
					encode_float(v.y,&buf[4]);
					buf+=4*2;

				}
			}

			r_len+=4*2*len;

		} break;
		case Variant::VECTOR3_ARRAY: {

			DVector<Vector3> data = p_variant;
			int len=data.size();

			if (buf) {
				encode_uint32(len,buf);
				buf+=4;
			}

			r_len+=4;

			if (buf) {

				for(int i=0;i<len;i++) {

					Vector3 v = data.get(i);

					encode_float(v.x,&buf[0]);
					encode_float(v.y,&buf[4]);
					encode_float(v.z,&buf[8]);
					buf+=4*3;

				}
			}

			r_len+=4*3*len;

		} break;
		case Variant::COLOR_ARRAY: {

			DVector<Color> data = p_variant;
			int len=data.size();

			if (buf) {
				encode_uint32(len,buf);
				buf+=4;
			}

			r_len+=4;

			if (buf) {

				for(int i=0;i<len;i++) {

					Color c = data.get(i);


					encode_float(c.r,&buf[0]);
					encode_float(c.g,&buf[4]);
					encode_float(c.b,&buf[8]);
					encode_float(c.a,&buf[12]);
					buf+=4*4;
				}
			}

			r_len+=4*4*len;

		} break;
		default: { ERR_FAIL_V(ERR_BUG); }
	}

	return OK;

}


