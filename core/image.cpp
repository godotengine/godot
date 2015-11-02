/*************************************************************************/
/*  image.cpp                                                            */
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
#include "image.h"
#include "hash_map.h"
#include "core/io/image_loader.h"
#include "core/os/copymem.h"

#include "print_string.h"
#include <stdio.h>


const char* Image::format_names[Image::FORMAT_MAX]={
	"Grayscale",
	"Intensity",
	"GrayscaleAlpha",
	"RGB",
	"RGBA",
	"Indexed",
	"IndexedAlpha",
	"YUV422",
	"YUV444",
	"BC1",
	"BC2",
	"BC3",
	"BC4",
	"BC5",
	"PVRTC2",
	"PVRTC2Alpha",
	"PVRTC4",
	"PVRTC4Alpha",
	"ETC",
	"ATC",
	"ATCAlphaExp",
	"ATCAlphaInterp",

};

SavePNGFunc Image::save_png_func = NULL;

void Image::_put_pixel(int p_x,int p_y, const BColor& p_color, unsigned char *p_data) {

	_put_pixelw(p_x,p_y,width,p_color,p_data);

}

void Image::_put_pixelw(int p_x,int p_y, int p_width, const BColor& p_color, unsigned char *p_data) {


	int ofs=p_y*p_width+p_x;

	switch(format) {
		case FORMAT_GRAYSCALE: {

			p_data[ofs]=p_color.gray();
		} break;
		case FORMAT_INTENSITY: {

			p_data[ofs]=p_color.a;
		} break;
		case FORMAT_GRAYSCALE_ALPHA: {

			p_data[ofs*2]=p_color.gray();
			p_data[ofs*2+1]=p_color.a;

		} break;
		case FORMAT_RGB: {

			p_data[ofs*3+0]=p_color.r;
			p_data[ofs*3+1]=p_color.g;
			p_data[ofs*3+2]=p_color.b;

		} break;
		case FORMAT_RGBA: {

			p_data[ofs*4+0]=p_color.r;
			p_data[ofs*4+1]=p_color.g;
			p_data[ofs*4+2]=p_color.b;
			p_data[ofs*4+3]=p_color.a;

		} break;
		case FORMAT_INDEXED:
		case FORMAT_INDEXED_ALPHA: {

			ERR_FAIL();
		} break;
		default: {};

	}

}



void Image::_get_mipmap_offset_and_size(int p_mipmap,int &r_offset, int &r_width,int &r_height) const {

	int w=width;
	int h=height;
	int ofs=0;

	int pixel_size = get_format_pixel_size(format);
	int pixel_rshift = get_format_pixel_rshift(format);
	int minw,minh;
	_get_format_min_data_size(format,minw,minh);

	for(int i=0;i<p_mipmap;i++) {
		int s = w*h;
		s*=pixel_size;
		s>>=pixel_rshift;
		ofs+=s;
		w=MAX(minw,w>>1);
		h=MAX(minh,h>>1);
	}

	r_offset=ofs;
	r_width=w;
	r_height=h;
}
int Image::get_mipmap_offset(int p_mipmap) const {

	ERR_FAIL_INDEX_V(p_mipmap,(mipmaps+1),-1);

	int ofs,w,h;
	_get_mipmap_offset_and_size(p_mipmap,ofs,w,h);
	return ofs;
}

void Image::get_mipmap_offset_and_size(int p_mipmap,int &r_ofs, int &r_size) const {

	int ofs,w,h;
	_get_mipmap_offset_and_size(p_mipmap,ofs,w,h);
	int ofs2;
	_get_mipmap_offset_and_size(p_mipmap+1,ofs2,w,h);
	r_ofs=ofs;
	r_size=ofs2-ofs;

}

void Image::get_mipmap_offset_size_and_dimensions(int p_mipmap,int &r_ofs, int &r_size,int &w, int& h) const {


	int ofs;
	_get_mipmap_offset_and_size(p_mipmap,ofs,w,h);
	int ofs2,w2,h2;
	_get_mipmap_offset_and_size(p_mipmap+1,ofs2,w2,h2);
	r_ofs=ofs;
	r_size=ofs2-ofs;

}

void Image::put_pixel(int p_x,int p_y, const Color& p_color,int p_mipmap){

	ERR_FAIL_INDEX(p_mipmap,mipmaps+1);
	int ofs,w,h;
	_get_mipmap_offset_and_size(p_mipmap,ofs,w,h);
	ERR_FAIL_INDEX(p_x,w);
	ERR_FAIL_INDEX(p_y,h);

	DVector<uint8_t>::Write wp = data.write();
	unsigned char *data_ptr=wp.ptr();

	_put_pixelw(p_x,p_y,w,BColor(p_color.r*255,p_color.g*255,p_color.b*255,p_color.a*255),&data_ptr[ofs]);

}


Image::BColor Image::_get_pixel(int p_x,int p_y,const unsigned char *p_data,int p_data_size) const{

	return _get_pixelw(p_x,p_y,width,p_data,p_data_size);
}
Image::BColor Image::_get_pixelw(int p_x,int p_y,int p_width,const unsigned char *p_data,int p_data_size) const{

	int ofs=p_y*p_width+p_x;
	BColor result(0,0,0,0);
	switch(format) {

		case FORMAT_GRAYSCALE: {

			result=BColor(p_data[ofs],p_data[ofs],p_data[ofs],255.0);
		} break;
		case FORMAT_INTENSITY: {

			result=BColor(255,255,255,p_data[ofs]);
		} break;
		case FORMAT_GRAYSCALE_ALPHA: {

			result=BColor(p_data[ofs*2],p_data[ofs*2],p_data[ofs*2],p_data[ofs*2+1]);

		} break;
		case FORMAT_RGB: {

			result=BColor(p_data[ofs*3],p_data[ofs*3+1],p_data[ofs*3+2]);

		} break;
		case FORMAT_RGBA: {

			result=BColor(p_data[ofs*4],p_data[ofs*4+1],p_data[ofs*4+2],p_data[ofs*4+3]);
		} break;
		case FORMAT_INDEXED_ALPHA: {

			int pitch = 4;
			const uint8_t* pal = &p_data[ p_data_size - pitch * 256 ];
			int idx = p_data[ofs];
			result=BColor(pal[idx * pitch + 0] , pal[idx * pitch + 1] , pal[idx * pitch + 2] , pal[idx * pitch + 3] );

		} break;
		case FORMAT_INDEXED: {

			int pitch = 3;
			const uint8_t* pal = &p_data[ p_data_size - pitch * 256 ];
			int idx = p_data[ofs];
			result=BColor(pal[idx * pitch + 0] , pal[idx * pitch + 1] , pal[idx * pitch + 2] ,255);
		} break;
		case FORMAT_YUV_422: {

			int y, u, v;
			if (p_x % 2) {
				const uint8_t* yp = &p_data[p_width * 2 * p_y + p_x * 2];
				u = *(yp-1);
				y = yp[0];
				v = yp[1];
			} else {

				const uint8_t* yp = &p_data[p_width * 2 * p_y + p_x * 2];
				y = yp[0];
				u = yp[1];
				v = yp[3];
			};

			int32_t r = 1.164 * (y - 16) + 1.596 * (v - 128);
			int32_t g = 1.164 * (y - 16) - 0.813 * (v - 128) - 0.391 * (u - 128);
			int32_t b = 1.164 * (y - 16) + 2.018 * (u - 128);
			result = BColor(CLAMP(r, 0, 255), CLAMP(g, 0, 255), CLAMP(b, 0, 255));
		} break;
		case FORMAT_YUV_444: {

			uint8_t y, u, v;
			const uint8_t* yp = &p_data[p_width * 3 * p_y + p_x * 3];
			y = yp[0];
			u = yp[1];
			v = yp[2];

			int32_t r = 1.164 * (y - 16) + 1.596 * (v - 128);
			int32_t g = 1.164 * (y - 16) - 0.813 * (v - 128) - 0.391 * (u - 128);
			int32_t b = 1.164 * (y - 16) + 2.018 * (u - 128);
			result = BColor(CLAMP(r, 0, 255), CLAMP(g, 0, 255), CLAMP(b, 0, 255));
		} break;
		default:{}

	}

	return result;

}

void Image::put_indexed_pixel(int p_x, int p_y, uint8_t p_idx,int p_mipmap) {

	ERR_FAIL_COND(format != FORMAT_INDEXED && format != FORMAT_INDEXED_ALPHA);
	ERR_FAIL_INDEX(p_mipmap,mipmaps+1);
	int ofs,w,h;
	_get_mipmap_offset_and_size(p_mipmap,ofs,w,h);
	ERR_FAIL_INDEX(p_x,w);
	ERR_FAIL_INDEX(p_y,h);

	data.set(ofs + p_y * w + p_x, p_idx);
};

uint8_t Image::get_indexed_pixel(int p_x, int p_y,int p_mipmap) const {

	ERR_FAIL_COND_V(format != FORMAT_INDEXED && format != FORMAT_INDEXED_ALPHA, 0);

	ERR_FAIL_INDEX_V(p_mipmap,mipmaps+1,0);
	int ofs,w,h;
	_get_mipmap_offset_and_size(p_mipmap,ofs,w,h);
	ERR_FAIL_INDEX_V(p_x,w,0);
	ERR_FAIL_INDEX_V(p_y,h,0);


	return data[ofs + p_y * w + p_x];
};

void Image::set_pallete(const DVector<uint8_t>& p_data) {


	int len = p_data.size();

	ERR_FAIL_COND(format != FORMAT_INDEXED && format != FORMAT_INDEXED_ALPHA);
	ERR_FAIL_COND(format == FORMAT_INDEXED && len!=(256*3));
	ERR_FAIL_COND(format == FORMAT_INDEXED_ALPHA && len!=(256*4));

	int ofs,w,h;
	_get_mipmap_offset_and_size(mipmaps+1,ofs,w,h);

	int pal_ofs = ofs;
	data.resize(pal_ofs + p_data.size());

	DVector<uint8_t>::Write wp = data.write();
	unsigned char *dst=wp.ptr() + pal_ofs;

	DVector<uint8_t>::Read r = data.read();
	const unsigned char *src=r.ptr();

	copymem(dst, src, len);
};

int Image::get_width() const {

	return width;
}
int Image::get_height() const{

	return height;
}

int Image::get_mipmaps() const {


	return mipmaps;
}

Color Image::get_pixel(int p_x,int p_y,int p_mipmap) const {


	ERR_FAIL_INDEX_V(p_mipmap,mipmaps+1,Color());
	int ofs,w,h;
	_get_mipmap_offset_and_size(p_mipmap,ofs,w,h);
	ERR_FAIL_INDEX_V(p_x,w,Color());
	ERR_FAIL_INDEX_V(p_y,h,Color());


	int len = data.size();
	DVector<uint8_t>::Read r = data.read();
	const unsigned char*data_ptr=r.ptr();
	BColor c = _get_pixelw(p_x,p_y,w,&data_ptr[ofs],len);
	return Color( c.r/255.0,c.g/255.0,c.b/255.0,c.a/255.0 );
}

void Image::convert( Format p_new_format ){

	if (data.size()==0)
		return;

	if (p_new_format==format)
		return;

	if (format>=FORMAT_BC1 || p_new_format>=FORMAT_BC1) {

		ERR_EXPLAIN("Cannot convert to <-> from compressed/custom image formats (for now).");
		ERR_FAIL();
	}

	if (p_new_format==FORMAT_INDEXED || p_new_format==FORMAT_INDEXED_ALPHA) {


		return;
	}


	Image new_img(width,height,0,p_new_format);
	
	int len=data.size();

	DVector<uint8_t>::Read r = data.read();
	DVector<uint8_t>::Write w = new_img.data.write();

	const uint8_t *rptr = r.ptr();
	uint8_t *wptr = w.ptr();

	if (p_new_format==FORMAT_RGBA && format==FORMAT_INDEXED_ALPHA) {

		//optimized unquantized form
		int dataend = len-256*4;
		const uint32_t *palpos = (const uint32_t*)&rptr[dataend];
		uint32_t *dst32 = (uint32_t *)wptr;

		for(int i=0;i<dataend;i++)
			dst32[i]=palpos[rptr[i]]; //since this is read/write, endianness is not a problem

	} else {

		//this is temporary, must find a faster way to do it.
		for(int i=0;i<width;i++)
			for(int j=0;j<height;j++)
				new_img._put_pixel(i,j,_get_pixel(i,j,rptr,len),wptr);
	}

	r = DVector<uint8_t>::Read();
	w = DVector<uint8_t>::Write();

	bool gen_mipmaps=mipmaps>0;
			
	*this=new_img;

	if (gen_mipmaps)
		generate_mipmaps();


}

Image::Format Image::get_format() const{

	return format;
}

static double _bicubic_interp_kernel( double x ) {

	x = ABS(x);

	double bc = 0;

	if ( x <= 1 )
		bc = ( 1.5 * x - 2.5 ) * x * x + 1;
	else if ( x < 2 )
		bc = ( ( -0.5 * x + 2.5 ) * x - 4 ) * x + 2;


	return bc;
}

template<int CC>
static void _scale_cubic(const uint8_t* p_src, uint8_t* p_dst, uint32_t p_src_width, uint32_t p_src_height, uint32_t p_dst_width, uint32_t p_dst_height) {


	// get source image size
	int width   = p_src_width;
	int height  = p_src_height;
	double xfac = (double) width / p_dst_width;
	double yfac = (double) height / p_dst_height;
	// coordinates of source points and cooefficiens
	double  ox, oy, dx, dy, k1, k2;
	int     ox1, oy1, ox2, oy2;
	// destination pixel values
	// width and height decreased by 1
	int ymax = height - 1;
	int xmax = width - 1;
	// temporary pointer

	for ( int y = 0; y < p_dst_height; y++ ) {
		// Y coordinates
		oy  = (double) y * yfac - 0.5f;
		oy1 = (int) oy;
		dy  = oy - (double) oy1;

		for ( int x = 0; x < p_dst_width; x++ )	{
			// X coordinates
			ox  = (double) x * xfac - 0.5f;
			ox1 = (int) ox;
			dx  = ox - (double) ox1;

			// initial pixel value

			uint8_t *dst=p_dst + (y*p_dst_width+x)*CC;

			double color[CC];
			for(int i=0;i<CC;i++) {
				color[i]=0;
			}



			for ( int n = -1; n < 3; n++ ) {
				// get Y cooefficient
				k1 = _bicubic_interp_kernel( dy - (double) n );

				oy2 = oy1 + n;
				if ( oy2 < 0 )
					oy2 = 0;
				if ( oy2 > ymax )
					oy2 = ymax;

				for ( int m = -1; m < 3; m++ ) {
					// get X cooefficient
					k2 = k1 * _bicubic_interp_kernel( (double) m - dx );

					ox2 = ox1 + m;
					if ( ox2 < 0 )
						ox2 = 0;
					if ( ox2 > xmax )
						ox2 = xmax;

					// get pixel of original image
					const uint8_t *p = p_src + (oy2 * p_src_width + ox2)*CC;

					for(int i=0;i<CC;i++) {

						color[i]+=p[i]*k2;
					}
				}
			}

			for(int i=0;i<CC;i++) {
				dst[i]=CLAMP(Math::fast_ftoi(color[i]),0,255);
			}
		}
	}
}




template<int CC>
static void _scale_bilinear(const uint8_t* p_src, uint8_t* p_dst, uint32_t p_src_width, uint32_t p_src_height, uint32_t p_dst_width, uint32_t p_dst_height) {

	enum {
		FRAC_BITS=8,
		FRAC_LEN=(1<<FRAC_BITS),
		FRAC_MASK=FRAC_LEN-1

	};

	for(uint32_t i=0;i<p_dst_height;i++) {

		uint32_t src_yofs_up_fp = (i*p_src_height*FRAC_LEN/p_dst_height);
		uint32_t src_yofs_frac = src_yofs_up_fp & FRAC_MASK;
		uint32_t src_yofs_up = src_yofs_up_fp >> FRAC_BITS;


		uint32_t src_yofs_down = (i+1)*p_src_height/p_dst_height;
		if (src_yofs_down>=p_src_height)
			src_yofs_down=p_src_height-1;

		//src_yofs_up*=CC;
		//src_yofs_down*=CC;

		uint32_t y_ofs_up = src_yofs_up * p_src_width * CC;
		uint32_t y_ofs_down = src_yofs_down * p_src_width * CC;

		for(uint32_t j=0;j<p_dst_width;j++) {

			uint32_t src_xofs_left_fp = (j*p_src_width*FRAC_LEN/p_dst_width);
			uint32_t src_xofs_frac = src_xofs_left_fp & FRAC_MASK;
			uint32_t src_xofs_left = src_xofs_left_fp >> FRAC_BITS;
			uint32_t src_xofs_right = (j+1)*p_src_width/p_dst_width;
			if (src_xofs_right>=p_src_width)
				src_xofs_right=p_src_width-1;

			src_xofs_left*=CC;
			src_xofs_right*=CC;

			for(uint32_t l=0;l<CC;l++) {

				uint32_t p00=p_src[y_ofs_up+src_xofs_left+l]<<FRAC_BITS;
				uint32_t p10=p_src[y_ofs_up+src_xofs_right+l]<<FRAC_BITS;
				uint32_t p01=p_src[y_ofs_down+src_xofs_left+l]<<FRAC_BITS;
				uint32_t p11=p_src[y_ofs_down+src_xofs_right+l]<<FRAC_BITS;

				uint32_t interp_up = p00+(((p10-p00)*src_xofs_frac)>>FRAC_BITS);
				uint32_t interp_down = p01+(((p11-p01)*src_xofs_frac)>>FRAC_BITS);
				uint32_t interp = interp_up+(((interp_down-interp_up)*src_yofs_frac)>>FRAC_BITS);
				interp>>=FRAC_BITS;
				p_dst[i*p_dst_width*CC+j*CC+l]=interp;
			}
		}
	}
}


template<int CC>
static void _scale_nearest(const uint8_t* p_src, uint8_t* p_dst, uint32_t p_src_width, uint32_t p_src_height, uint32_t p_dst_width, uint32_t p_dst_height) {


	for(uint32_t i=0;i<p_dst_height;i++) {

		uint32_t src_yofs = i*p_src_height/p_dst_height;
		uint32_t y_ofs = src_yofs * p_src_width * CC;

		for(uint32_t j=0;j<p_dst_width;j++) {

			uint32_t src_xofs = j*p_src_width/p_dst_width;
			src_xofs*=CC;

			for(uint32_t l=0;l<CC;l++) {

				uint32_t p=p_src[y_ofs+src_xofs+l];
				p_dst[i*p_dst_width*CC+j*CC+l]=p;
			}
		}
	}
}


void Image::resize_to_po2(bool p_square) {

	if (!_can_modify(format)) {
		ERR_EXPLAIN("Cannot resize in indexed, compressed or custom image formats.");
		ERR_FAIL();
	}

	int w = nearest_power_of_2(width);
	int h = nearest_power_of_2(height);

	if (w==width && h==height) {

		if (!p_square || w==h)
			return; //nothing to do
	}

	resize(w,h);
}

Image Image::resized( int p_width, int p_height, int p_interpolation ) {

	Image ret = *this;
	ret.resize(p_width, p_height, (Interpolation)p_interpolation);

	return ret;
};


void Image::resize( int p_width, int p_height, Interpolation p_interpolation ) {

	if (!_can_modify(format)) {
		ERR_EXPLAIN("Cannot resize in indexed, compressed or custom image formats.");
		ERR_FAIL();
	}

	ERR_FAIL_COND(p_width<=0);
	ERR_FAIL_COND(p_height<=0);
	ERR_FAIL_COND(p_width>MAX_WIDTH);
	ERR_FAIL_COND(p_height>MAX_HEIGHT);
	

	if (p_width==width && p_height==height)
		return;
		
	Image dst( p_width, p_height, 0, format );
	 
	if (format==FORMAT_INDEXED)
		p_interpolation=INTERPOLATE_NEAREST;


	DVector<uint8_t>::Read r = data.read();
	const unsigned char*r_ptr=r.ptr();

	DVector<uint8_t>::Write w = dst.data.write();
	unsigned char*w_ptr=w.ptr();


	switch(p_interpolation) {

		case INTERPOLATE_NEAREST: {

			switch(get_format_pixel_size(format)) {
				case 1: _scale_nearest<1>(r_ptr,w_ptr,width,height,p_width,p_height); break;
				case 2: _scale_nearest<2>(r_ptr,w_ptr,width,height,p_width,p_height); break;
				case 3: _scale_nearest<3>(r_ptr,w_ptr,width,height,p_width,p_height); break;
				case 4: _scale_nearest<4>(r_ptr,w_ptr,width,height,p_width,p_height); break;
			}
		} break;
		case INTERPOLATE_BILINEAR: {

			switch(get_format_pixel_size(format)) {
				case 1: _scale_bilinear<1>(r_ptr,w_ptr,width,height,p_width,p_height); break;
				case 2: _scale_bilinear<2>(r_ptr,w_ptr,width,height,p_width,p_height); break;
				case 3: _scale_bilinear<3>(r_ptr,w_ptr,width,height,p_width,p_height); break;
				case 4: _scale_bilinear<4>(r_ptr,w_ptr,width,height,p_width,p_height); break;
			}

		} break;
		case INTERPOLATE_CUBIC: {

			switch(get_format_pixel_size(format)) {
				case 1: _scale_cubic<1>(r_ptr,w_ptr,width,height,p_width,p_height); break;
				case 2: _scale_cubic<2>(r_ptr,w_ptr,width,height,p_width,p_height); break;
				case 3: _scale_cubic<3>(r_ptr,w_ptr,width,height,p_width,p_height); break;
				case 4: _scale_cubic<4>(r_ptr,w_ptr,width,height,p_width,p_height); break;
			}

		} break;


	}

	r = DVector<uint8_t>::Read();
	w = DVector<uint8_t>::Write();

	if (mipmaps>0)
		dst.generate_mipmaps();

	*this=dst;
}
void Image::crop( int p_width, int p_height ) {

	if (!_can_modify(format)) {
		ERR_EXPLAIN("Cannot crop in indexed, compressed or custom image formats.");
		ERR_FAIL();
	}
	ERR_FAIL_COND(p_width<=0);
	ERR_FAIL_COND(p_height<=0);
	ERR_FAIL_COND(p_width>MAX_WIDTH);
	ERR_FAIL_COND(p_height>MAX_HEIGHT);
	
	/* to save memory, cropping should be done in-place, however, since this function
	   will most likely either not be used much, or in critical areas, for now it wont, because
	   it's a waste of time. */

	if (p_width==width && p_height==height)
		return;
		
	Image dst( p_width, p_height,0, format );

	
	for (int y=0;y<p_height;y++) {
	
		for (int x=0;x<p_width;x++) {

			Color col = (x>=width || y>=height)? Color() : get_pixel(x,y);
			dst.put_pixel(x,y,col);
		}
	}
	
	if (mipmaps>0)
		dst.generate_mipmaps();
	*this=dst;
	
}

void Image::flip_y() {
	
	if (!_can_modify(format)) {
		ERR_EXPLAIN("Cannot flip_y in indexed, compressed or custom image formats.");
		ERR_FAIL();
	}

	bool gm=mipmaps;

	if (gm)
		clear_mipmaps();;




	for (int y=0;y<(height/2);y++) {
	
		for (int x=0;x<width;x++) {

			Color up = get_pixel(x,y);
			Color down = get_pixel(x,height-y-1);
			
			put_pixel(x,y,down);
			put_pixel(x,height-y-1,up);
		}
	}
	if (gm)
		generate_mipmaps();;

}

void Image::flip_x() {

	if (!_can_modify(format)) {
		ERR_EXPLAIN("Cannot flip_x in indexed, compressed or custom image formats.");
		ERR_FAIL();
	}

	bool gm=mipmaps;
	if (gm)
		clear_mipmaps();;

	for (int y=0;y<(height/2);y++) {
		
		for (int x=0;x<width;x++) {
			
			Color up = get_pixel(x,y);
			Color down = get_pixel(width-x-1,y);
			
			put_pixel(x,y,down);
			put_pixel(width-x-1,y,up);
		}
	}

	if (gm)
		generate_mipmaps();;

}

int Image::_get_dst_image_size(int p_width, int p_height, Format p_format,int &r_mipmaps,int p_mipmaps) {

	int size=0;
	int w=p_width;
	int h=p_height;
	int mm=0;

	int pixsize=get_format_pixel_size(p_format);
	int pixshift=get_format_pixel_rshift(p_format);
	int minw,minh;
	_get_format_min_data_size(p_format,minw,minh);


	switch(p_format) {

		case FORMAT_INDEXED: pixsize=1; size=256*3; break;
		case FORMAT_INDEXED_ALPHA: pixsize=1; size=256*4;break;
		default: {}
	} ;

	while(true) {

		int s = w*h;
		s*=pixsize;
		s>>=pixshift;

		size+=s;

		if (p_mipmaps>=0 && mm==p_mipmaps)
			break;

		if (p_mipmaps>=0) {

			w=MAX(minw,w>>1);
			h=MAX(minh,h>>1);
		} else {
			if (w==minw && h==minh)
				break;
			w=MAX(minw,w>>1);
			h=MAX(minh,h>>1);
		}
		mm++;
	};

	r_mipmaps=mm;
	return size;
}

bool Image::_can_modify(Format p_format) const {

	switch(p_format) {

		//these are OK
		case FORMAT_GRAYSCALE:
		case FORMAT_INTENSITY:
		case FORMAT_GRAYSCALE_ALPHA:
		case FORMAT_RGB:
		case FORMAT_RGBA:
			return true;
		default:
			return false;
	}

	return false;
}

template<int CC>
static void _generate_po2_mipmap(const uint8_t* p_src, uint8_t* p_dst, uint32_t p_width, uint32_t p_height) {

	//fast power of 2 mipmap generation
	uint32_t dst_w = p_width >> 1;
	uint32_t dst_h = p_height >> 1;

	for(uint32_t i=0;i<dst_h;i++) {

		const uint8_t *rup_ptr = &p_src[i*2*p_width*CC];
		const uint8_t *rdown_ptr = rup_ptr + p_width * CC;
		uint8_t *dst_ptr = &p_dst[i*dst_w*CC];
		uint32_t count=dst_w;


		while(count--) {

			for(int j=0;j<CC;j++) {

				uint16_t val=0;
				val+=rup_ptr[j];
				val+=rup_ptr[j+CC];
				val+=rdown_ptr[j];
				val+=rdown_ptr[j+CC];
				dst_ptr[j]=val>>2;

			}

			dst_ptr+=CC;
			rup_ptr+=CC*2;
			rdown_ptr+=CC*2;
		}
	}
}


Error Image::generate_mipmaps(int p_mipmaps,bool p_keep_existing)  {

	if (!_can_modify(format)) {
		ERR_EXPLAIN("Cannot generate mipmaps in indexed, compressed or custom image formats.");
		ERR_FAIL_V(ERR_UNAVAILABLE);

	}

	int from_mm=1;
	if (p_keep_existing) {
		from_mm=mipmaps+1;
	}
	int size = _get_dst_image_size(width,height,format,mipmaps,p_mipmaps);

	data.resize(size);

	DVector<uint8_t>::Write wp=data.write();

	if (nearest_power_of_2(width)==uint32_t(width) && nearest_power_of_2(height)==uint32_t(height)) {
		//use fast code for powers of 2
		int prev_ofs=0;
		int prev_h=height;
		int prev_w=width;

		for(int i=1;i<mipmaps;i++) {


			int ofs,w,h;
			_get_mipmap_offset_and_size(i,ofs, w,h);

			if (i>=from_mm) {

				switch(format) {

					case FORMAT_GRAYSCALE:
					case FORMAT_INTENSITY: _generate_po2_mipmap<1>(&wp[prev_ofs], &wp[ofs], prev_w,prev_h); break;
					case FORMAT_GRAYSCALE_ALPHA: _generate_po2_mipmap<2>(&wp[prev_ofs], &wp[ofs], prev_w,prev_h); break;
					case FORMAT_RGB: _generate_po2_mipmap<3>(&wp[prev_ofs], &wp[ofs], prev_w,prev_h); break;
					case FORMAT_RGBA: _generate_po2_mipmap<4>(&wp[prev_ofs], &wp[ofs], prev_w,prev_h); break;
					default: {}
				}
			}

			prev_ofs=ofs;
			prev_w=w;
			prev_h=h;
		}


	} else {
		//use slow code..

		//use bilinear filtered code for non powers of 2
		int prev_ofs=0;
		int prev_h=height;
		int prev_w=width;

		for(int i=1;i<mipmaps;i++) {


			int ofs,w,h;
			_get_mipmap_offset_and_size(i,ofs, w,h);

			if (i>=from_mm) {

				switch(format) {

					case FORMAT_GRAYSCALE:
					case FORMAT_INTENSITY: _scale_bilinear<1>(&wp[prev_ofs], &wp[ofs], prev_w,prev_h,w,h); break;
					case FORMAT_GRAYSCALE_ALPHA: _scale_bilinear<2>(&wp[prev_ofs], &wp[ofs], prev_w,prev_h,w,h); break;
					case FORMAT_RGB: _scale_bilinear<3>(&wp[prev_ofs], &wp[ofs], prev_w,prev_h,w,h); break;
					case FORMAT_RGBA: _scale_bilinear<4>(&wp[prev_ofs], &wp[ofs], prev_w,prev_h,w,h); break;
					default: {}
				}
			}

			prev_ofs=ofs;
			prev_w=w;
			prev_h=h;
		}

	}




	return OK;
}

void Image::clear_mipmaps() {

	if (mipmaps==0)
		return;

	if (format==FORMAT_CUSTOM) {
		ERR_EXPLAIN("Cannot clear mipmaps in indexed, compressed or custom image formats.");
		ERR_FAIL();

	}

	if (empty())
		return;

	int ofs,w,h;
	_get_mipmap_offset_and_size(1,ofs,w,h);
	int palsize = get_format_pallete_size(format);
	DVector<uint8_t> pallete;
	ERR_FAIL_COND(ofs+palsize > data.size()); //bug?
	if (palsize) {

		pallete.resize(palsize);
		DVector<uint8_t>::Read r = data.read();
		DVector<uint8_t>::Write w = pallete.write();

		copymem(&w[0],&r[data.size()-palsize],palsize);
	}

	data.resize(ofs+palsize);

	if (palsize) {

		DVector<uint8_t>::Read r = pallete.read();
		DVector<uint8_t>::Write w = data.write();

		copymem(&w[ofs],&r[0],palsize);
	}

	mipmaps=0;

}

void Image::make_normalmap(float p_height_scale) {

	if (!_can_modify(format)) {
		ERR_EXPLAIN("Cannot crop in indexed, compressed or custom image formats.");
		ERR_FAIL();
	}

	ERR_FAIL_COND( empty() );
	
	Image normalmap(width,height,0, FORMAT_RGB);
	/*
	for (int y=0;y<height;y++) {
		for (int x=0;x<width;x++) {
		
			float center=get_pixel(x,y).gray()/255.0;
			float up=(y>0)?get_pixel(x,y-1).gray()/255.0:center;
			float down=(y<(height-1))?get_pixel(x,y+1).gray()/255.0:center;
			float left=(x>0)?get_pixel(x-1,y).gray()/255.0:center;
			float right=(x<(width-1))?get_pixel(x+1,y).gray()/255.0:center;
		
		
			// uhm, how do i do this? ....
			
			Color result( (uint8_t)((normal.x+1.0)*127.0), (uint8_t)((normal.y+1.0)*127.0), (uint8_t)((normal.z+1.0)*127.0) );
			
			normalmap.put_pixel( x, y, result );
		}
		
	}
	*/
	*this=normalmap;
}

bool Image::empty() const {

	return (data.size()==0);
}

DVector<uint8_t> Image::get_data() const {
	
	return data;
}

void Image::create(int p_width, int p_height, bool p_use_mipmaps,Format p_format) {


	int mm=0;	
	int size = _get_dst_image_size(p_width,p_height,p_format,mm,p_use_mipmaps?-1:0);
	data.resize( size );
	{
		DVector<uint8_t>::Write w= data.write();
		zeromem(w.ptr(),size);
	}

	width=p_width;
	height=p_height;
	mipmaps=mm;
	format=p_format;


}

void Image::create(int p_width, int p_height, int p_mipmaps, Format p_format, const DVector<uint8_t>& p_data) {
	
	ERR_FAIL_INDEX(p_width-1,MAX_WIDTH);
	ERR_FAIL_INDEX(p_height-1,MAX_HEIGHT);

	if (p_format < FORMAT_CUSTOM) {
		int mm;
		int size = _get_dst_image_size(p_width,p_height,p_format,mm,p_mipmaps);

		if (size!=p_data.size()) {
			ERR_EXPLAIN("Expected data size of "+itos(size)+" in Image::create()");
			ERR_FAIL_COND(p_data.size()!=size);
		}
	};
	
	height=p_height;
	width=p_width;
	format=p_format;
	data=p_data;	
	mipmaps=p_mipmaps;
}


void Image::create( const char ** p_xpm ) {
	

	int size_width,size_height;
	int pixelchars=0;
	mipmaps=0;
	bool has_alpha=false;
	
	enum Status {
		READING_HEADER,
		READING_COLORS,
		READING_PIXELS,
		DONE
	};
	
	Status status = READING_HEADER;
	int line=0;
	
	HashMap<String,Color> colormap;
	int colormap_size;
	
	while (status!=DONE) {
		
		const char * line_ptr = p_xpm[line];
		
		
		switch (status) {
			
		case READING_HEADER: {
			
			String line_str=line_ptr;
			line_str.replace("\t"," ");
			
			size_width=line_str.get_slicec(' ',0).to_int();
			size_height=line_str.get_slicec(' ',1).to_int();
			colormap_size=line_str.get_slicec(' ',2).to_int();
			pixelchars=line_str.get_slicec(' ',3).to_int();
			ERR_FAIL_COND(colormap_size > 32766);
			ERR_FAIL_COND(pixelchars > 5);
			ERR_FAIL_COND(size_width > 32767);
			ERR_FAIL_COND(size_height > 32767);
			status=READING_COLORS;
		} break;
		case READING_COLORS: {
			
			String colorstring;
			for (int i=0;i<pixelchars;i++) {
				
				colorstring+=*line_ptr;
				line_ptr++;
			}
				//skip spaces
			while (*line_ptr==' ' ||  *line_ptr=='\t' || *line_ptr==0) {
				if (*line_ptr==0)
					break;
				line_ptr++;
			}
			if (*line_ptr=='c') {
				
				line_ptr++;
				while (*line_ptr==' ' ||  *line_ptr=='\t' || *line_ptr==0) {
					if (*line_ptr==0)
						break;
					line_ptr++;
				}
				
				if (*line_ptr=='#') {
					line_ptr++;
					uint8_t col_r;
					uint8_t col_g;
					uint8_t col_b;
//					uint8_t col_a=255;
					
					for (int i=0;i<6;i++) {
						
						char v = line_ptr[i];
						
						if (v>='0' && v<='9')
							v-='0';
						else if (v>='A' && v<='F')
							v=(v-'A')+10;
						else if (v>='a' && v<='f')
							v=(v-'a')+10;
						else
							break;
						
						switch(i) {
						case 0: col_r=v<<4; break;
						case 1: col_r|=v; break;
						case 2: col_g=v<<4; break;
						case 3: col_g|=v; break;
						case 4: col_b=v<<4; break;
						case 5: col_b|=v; break;
						};
						
					}
					
							// magenta mask
					if (col_r==255 && col_g==0 && col_b==255) {
						
						colormap[colorstring]=Color(0,0,0,0);
						has_alpha=true;
					} else {

						colormap[colorstring]=Color(col_r/255.0,col_g/255.0,col_b/255.0,1.0);
					}
					
				}
			}
			if (line==colormap_size) {
				
				status=READING_PIXELS;
				create(size_width,size_height,0,has_alpha?FORMAT_RGBA:FORMAT_RGB);
			}
		} break;
		case READING_PIXELS: {
			
			int y=line-colormap_size-1;
			for (int x=0;x<size_width;x++) {
				
				char pixelstr[6]={0,0,0,0,0,0};
				for (int i=0;i<pixelchars;i++)
					pixelstr[i]=line_ptr[x*pixelchars+i];
				
				Color *colorptr = colormap.getptr(pixelstr);
				ERR_FAIL_COND(!colorptr);
				put_pixel(x,y,*colorptr);
				
			}
			
			if (y==(size_height-1))
				status=DONE;
		} break;
		default:{}
		}
		
		line++;
	}
}
#define DETECT_ALPHA_MAX_TRESHOLD 254
#define DETECT_ALPHA_MIN_TRESHOLD 2

#define DETECT_ALPHA( m_value )\
{ \
	uint8_t value=m_value;\
	if (value<DETECT_ALPHA_MIN_TRESHOLD)\
		bit=true;\
	else if (value<DETECT_ALPHA_MAX_TRESHOLD) {\
		\
		detected=true;\
		break;\
	}\
}

#define DETECT_NON_ALPHA( m_value )\
{ \
	uint8_t value=m_value;\
	if (value>0) {\
		\
		detected=true;\
		break;\
	}\
}


bool Image::is_invisible() const {

	if (format==FORMAT_GRAYSCALE ||
	    format==FORMAT_RGB ||
	    format==FORMAT_INDEXED)
		return false;

	int len = data.size();

	if (len==0)
		return true;

	if (format >= FORMAT_YUV_422 && format <= FORMAT_YUV_444)
		return false;

	int w,h;
	_get_mipmap_offset_and_size(1,len,w,h);

	DVector<uint8_t>::Read r = data.read();
	const unsigned char *data_ptr=r.ptr();

	bool detected=false;

	switch(format) {
		case FORMAT_INTENSITY: {

			for(int i=0;i<len;i++) {
				DETECT_NON_ALPHA(data_ptr[i]);
			}
		} break;
		case FORMAT_GRAYSCALE_ALPHA: {


			for(int i=0;i<(len>>1);i++) {
				DETECT_NON_ALPHA(data_ptr[(i<<1)+1]);
			}

		} break;
		case FORMAT_RGBA: {

			for(int i=0;i<(len>>2);i++) {
				DETECT_NON_ALPHA(data_ptr[(i<<2)+3])
			}

		} break;
		case FORMAT_INDEXED: {

			return false;
		} break;
		case FORMAT_INDEXED_ALPHA: {

			return false;
		} break;
		case FORMAT_PVRTC2_ALPHA:
		case FORMAT_PVRTC4_ALPHA:
		case FORMAT_BC2:
		case FORMAT_BC3: {
			detected=true;
		} break;
		default: {}
	}

	return !detected;
}

Image::AlphaMode Image::detect_alpha() const {

	if (format==FORMAT_GRAYSCALE ||
	    format==FORMAT_RGB ||
	    format==FORMAT_INDEXED)
		return ALPHA_NONE;

	int len = data.size();

	if (len==0)
		return ALPHA_NONE;

	if (format >= FORMAT_YUV_422 && format <= FORMAT_YUV_444)
		return ALPHA_NONE;

	int w,h;
	_get_mipmap_offset_and_size(1,len,w,h);

	DVector<uint8_t>::Read r = data.read();
	const unsigned char *data_ptr=r.ptr();

	bool bit=false;
	bool detected=false;

	switch(format) {
		case FORMAT_INTENSITY: {

			for(int i=0;i<len;i++) {
				DETECT_ALPHA(data_ptr[i]);
			}
		} break;
		case FORMAT_GRAYSCALE_ALPHA: {


			for(int i=0;i<(len>>1);i++) {
				DETECT_ALPHA(data_ptr[(i<<1)+1]);
			}

		} break;
		case FORMAT_RGBA: {

			for(int i=0;i<(len>>2);i++) {
				DETECT_ALPHA(data_ptr[(i<<2)+3])
			}

		} break;
		case FORMAT_INDEXED: {

			return ALPHA_NONE;
		} break;
		case FORMAT_INDEXED_ALPHA: {

			return ALPHA_BLEND;
		} break;
		case FORMAT_PVRTC2_ALPHA:
		case FORMAT_PVRTC4_ALPHA:
		case FORMAT_BC2:
		case FORMAT_BC3: {
			detected=true;
		} break;
		default: {}
	}

	if (detected)
		return ALPHA_BLEND;
	else if (bit)
		return ALPHA_BIT;
	else
		return ALPHA_NONE;

}

Error Image::load(const String& p_path) {

	return ImageLoader::load_image(p_path, this);
}

Error Image::save_png(const String& p_path) {

	if (save_png_func == NULL)
		return ERR_UNAVAILABLE;

	return save_png_func(p_path, *this);
};

bool Image::operator==(const Image& p_image) const {

	if (data.size() == 0 && p_image.data.size() == 0)
		return true;
	DVector<uint8_t>::Read r = data.read();
	DVector<uint8_t>::Read pr = p_image.data.read();

	return r.ptr() == pr.ptr();
}


int Image::get_format_pixel_size(Format p_format) {

	switch(p_format) {
		case FORMAT_GRAYSCALE: {

			return 1;
		} break;
		case FORMAT_INTENSITY: {

			return 1;
		} break;
		case FORMAT_GRAYSCALE_ALPHA: {

			return 2;
		} break;
		case FORMAT_RGB: {

			return 3;
		} break;
		case FORMAT_RGBA: {

			return 4;
		} break;
		case FORMAT_INDEXED: {

			return 1;
		} break;
		case FORMAT_INDEXED_ALPHA: {

			return 1;
		} break;
		case FORMAT_BC1:
		case FORMAT_BC2:
		case FORMAT_BC3:
		case FORMAT_BC4:
		case FORMAT_BC5: {

			return 1;
		} break;
		case FORMAT_PVRTC2:
		case FORMAT_PVRTC2_ALPHA: {

			return 1;
		} break;
		case FORMAT_PVRTC4:
		case FORMAT_PVRTC4_ALPHA: {

			return 1;
		} break;
		case FORMAT_ATC:
		case FORMAT_ATC_ALPHA_EXPLICIT:
		case FORMAT_ATC_ALPHA_INTERPOLATED: {

			return 1;
		} break;
		case FORMAT_ETC: {

			return 1;
		} break;
		case FORMAT_YUV_422: {
			return 2;
		};
		case FORMAT_YUV_444: {
			return 3;
		} break;
		case FORMAT_CUSTOM: {

			ERR_EXPLAIN("pixel size requested for custom image format, and it's unknown obviously");
			ERR_FAIL_V(1);
		} break;
		default:{
			ERR_EXPLAIN("Cannot obtain pixel size from this format");
			ERR_FAIL_V(1);

		}
	}
	return 0;
}

int Image::get_image_data_size(int p_width, int p_height, Format p_format,int p_mipmaps)  {

	int mm;
	return _get_dst_image_size(p_width,p_height,p_format,mm,p_mipmaps);

}

int Image::get_image_required_mipmaps(int p_width, int p_height, Format p_format) {

	int mm;
	_get_dst_image_size(p_width,p_height,p_format,mm,-1);
	return mm;

}

void Image::_get_format_min_data_size(Format p_format,int &r_w, int &r_h) {


	switch(p_format) {
		case FORMAT_BC1:
		case FORMAT_BC2:
		case FORMAT_BC3:
		case FORMAT_BC4:
		case FORMAT_BC5: {
			r_w=4;
			r_h=4;
		} break;
		case FORMAT_PVRTC2:
		case FORMAT_PVRTC2_ALPHA: {

			r_w=16;
			r_h=8;
		} break;
		case FORMAT_PVRTC4_ALPHA:
		case FORMAT_PVRTC4: {

			r_w=8;
			r_h=8;
		} break;
		case FORMAT_ATC:
		case FORMAT_ATC_ALPHA_EXPLICIT:
		case FORMAT_ATC_ALPHA_INTERPOLATED: {

			r_w=8;
			r_h=8;

		} break;

		case FORMAT_ETC: {

			r_w=4;
			r_h=4;
		} break;
		default: {
			r_w=1;
			r_h=1;
		} break;
	}

}


int Image::get_format_pixel_rshift(Format p_format) {

	if (p_format==FORMAT_BC1 || p_format==FORMAT_BC4 || p_format==FORMAT_ATC || p_format==FORMAT_PVRTC4 || p_format==FORMAT_PVRTC4_ALPHA || p_format==FORMAT_ETC)
		return 1;
	else if (p_format==FORMAT_PVRTC2 || p_format==FORMAT_PVRTC2_ALPHA)
		return 2;
	else
		return 0;
}

int Image::get_format_pallete_size(Format p_format) {

	switch(p_format) {
		case FORMAT_GRAYSCALE: {

			return 0;
		} break;
		case FORMAT_INTENSITY: {

			return 0;
		} break;
		case FORMAT_GRAYSCALE_ALPHA: {

			return 0;
		} break;
		case FORMAT_RGB: {

			return 0;
		} break;
		case FORMAT_RGBA: {

			return 0;
		} break;
		case FORMAT_INDEXED: {

			return 3*256;
		} break;
		case FORMAT_INDEXED_ALPHA: {

			return 4*256;
		} break;
		default:{}
	}
	return 0;
}




Error Image::_decompress_bc() {

	print_line("decompressing bc");

	int mm;
	int size = _get_dst_image_size(width,height,FORMAT_RGBA,mm,mipmaps);

	DVector<uint8_t> newdata;
	newdata.resize(size);

	DVector<uint8_t>::Write w = newdata.write();
	DVector<uint8_t>::Read r = data.read();

	int rofs=0;
	int wofs=0;
	int wd=width,ht=height;

	for(int i=0;i<=mm;i++) {

		switch(format) {

			case FORMAT_BC1: {

				int len = (wd*ht)/16;
				uint8_t* dst=&w[wofs];

				uint32_t ofs_table[16];
				for(int x=0;x<4;x++) {

					for(int y=0;y<4;y++) {

						ofs_table[15-(y*4+(3-x))]=(x+y*wd)*4;
					}
				}


				for(int j=0;j<len;j++) {

					const uint8_t* src=&r[rofs+j*8];
					uint16_t col_a=src[1];
					col_a<<=8;
					col_a|=src[0];
					uint16_t col_b=src[3];
					col_b<<=8;
					col_b|=src[2];

					uint8_t table[4][4]={
						{ (col_a>>11)<<3, ((col_a>>5)&0x3f)<<2,	((col_a)&0x1f)<<3, 255 },
						{ (col_b>>11)<<3, ((col_b>>5)&0x3f)<<2,	((col_b)&0x1f)<<3, 255 },
						{0,0,0,255},
						{0,0,0,255}
					};

					if (col_a<col_b) {
						//punchrough
						table[2][0]=(int(table[0][0])+int(table[1][0]))>>1;
						table[2][1]=(int(table[0][1])+int(table[1][1]))>>1;
						table[2][2]=(int(table[0][2])+int(table[1][2]))>>1;
						table[3][3]=0; //premul alpha black
					} else {
						//gradient
						table[2][0]=(int(table[0][0])*2+int(table[1][0]))/3;
						table[2][1]=(int(table[0][1])*2+int(table[1][1]))/3;
						table[2][2]=(int(table[0][2])*2+int(table[1][2]))/3;
						table[3][0]=(int(table[0][0])+int(table[1][0])*2)/3;
						table[3][1]=(int(table[0][1])+int(table[1][1])*2)/3;
						table[3][2]=(int(table[0][2])+int(table[1][2])*2)/3;
					}

					uint32_t block=src[4];
					block<<=8;
					block|=src[5];
					block<<=8;
					block|=src[6];
					block<<=8;
					block|=src[7];

					int y = (j/(wd/4))*4;
					int x = (j%(wd/4))*4;
					int pixofs = (y*wd+x)*4;

					for(int k=0;k<16;k++) {
						int idx = pixofs+ofs_table[k];
						dst[idx+0]=table[block&0x3][0];
						dst[idx+1]=table[block&0x3][1];
						dst[idx+2]=table[block&0x3][2];
						dst[idx+3]=table[block&0x3][3];
						block>>=2;
					}

				}

				rofs+=len*8;
				wofs+=wd*ht*4;


				wd/=2;
				ht/=2;

			} break;
			case FORMAT_BC2: {

				int len = (wd*ht)/16;
				uint8_t* dst=&w[wofs];

				uint32_t ofs_table[16];
				for(int x=0;x<4;x++) {

					for(int y=0;y<4;y++) {

						ofs_table[15-(y*4+(3-x))]=(x+y*wd)*4;
					}
				}


				for(int j=0;j<len;j++) {

					const uint8_t* src=&r[rofs+j*16];

					uint64_t ablock=src[1];
					ablock<<=8;
					ablock|=src[0];
					ablock<<=8;
					ablock|=src[3];
					ablock<<=8;
					ablock|=src[2];
					ablock<<=8;
					ablock|=src[5];
					ablock<<=8;
					ablock|=src[4];
					ablock<<=8;
					ablock|=src[7];
					ablock<<=8;
					ablock|=src[6];


					uint16_t col_a=src[8+1];
					col_a<<=8;
					col_a|=src[8+0];
					uint16_t col_b=src[8+3];
					col_b<<=8;
					col_b|=src[8+2];

					uint8_t table[4][4]={
						{ (col_a>>11)<<3, ((col_a>>5)&0x3f)<<2,	((col_a)&0x1f)<<3, 255 },
						{ (col_b>>11)<<3, ((col_b>>5)&0x3f)<<2,	((col_b)&0x1f)<<3, 255 },
						{0,0,0,255},
						{0,0,0,255}
					};

					//always gradient
					table[2][0]=(int(table[0][0])*2+int(table[1][0]))/3;
					table[2][1]=(int(table[0][1])*2+int(table[1][1]))/3;
					table[2][2]=(int(table[0][2])*2+int(table[1][2]))/3;
					table[3][0]=(int(table[0][0])+int(table[1][0])*2)/3;
					table[3][1]=(int(table[0][1])+int(table[1][1])*2)/3;
					table[3][2]=(int(table[0][2])+int(table[1][2])*2)/3;

					uint32_t block=src[4+8];
					block<<=8;
					block|=src[5+8];
					block<<=8;
					block|=src[6+8];
					block<<=8;
					block|=src[7+8];

					int y = (j/(wd/4))*4;
					int x = (j%(wd/4))*4;
					int pixofs = (y*wd+x)*4;

					for(int k=0;k<16;k++) {
						uint8_t alpha = ablock&0xf;
						alpha=int(alpha)*255/15; //right way for alpha
						int idx = pixofs+ofs_table[k];
						dst[idx+0]=table[block&0x3][0];
						dst[idx+1]=table[block&0x3][1];
						dst[idx+2]=table[block&0x3][2];
						dst[idx+3]=alpha;
						block>>=2;
						ablock>>=4;
					}

				}

				rofs+=len*16;
				wofs+=wd*ht*4;


				wd/=2;
				ht/=2;

			} break;
			case FORMAT_BC3: {

				int len = (wd*ht)/16;
				uint8_t* dst=&w[wofs];

				uint32_t ofs_table[16];
				for(int x=0;x<4;x++) {

					for(int y=0;y<4;y++) {

						ofs_table[15-(y*4+(3-x))]=(x+y*wd)*4;
					}
				}



				for(int j=0;j<len;j++) {

					const uint8_t* src=&r[rofs+j*16];

					uint8_t a_start=src[1];
					uint8_t a_end=src[0];

					uint64_t ablock=src[3];
					ablock<<=8;
					ablock|=src[2];
					ablock<<=8;
					ablock|=src[5];
					ablock<<=8;
					ablock|=src[4];
					ablock<<=8;
					ablock|=src[7];
					ablock<<=8;
					ablock|=src[6];

					uint8_t atable[8];

					if (a_start>a_end) {

						atable[0]=(int(a_start)*7+int(a_end)*0)/7;
						atable[1]=(int(a_start)*6+int(a_end)*1)/7;
						atable[2]=(int(a_start)*5+int(a_end)*2)/7;
						atable[3]=(int(a_start)*4+int(a_end)*3)/7;
						atable[4]=(int(a_start)*3+int(a_end)*4)/7;
						atable[5]=(int(a_start)*2+int(a_end)*5)/7;
						atable[6]=(int(a_start)*1+int(a_end)*6)/7;
						atable[7]=(int(a_start)*0+int(a_end)*7)/7;
					} else {

						atable[0]=(int(a_start)*5+int(a_end)*0)/5;
						atable[1]=(int(a_start)*4+int(a_end)*1)/5;
						atable[2]=(int(a_start)*3+int(a_end)*2)/5;
						atable[3]=(int(a_start)*2+int(a_end)*3)/5;
						atable[4]=(int(a_start)*1+int(a_end)*4)/5;
						atable[5]=(int(a_start)*0+int(a_end)*5)/5;
						atable[6]=0;
						atable[7]=255;

					}


					uint16_t col_a=src[8+1];
					col_a<<=8;
					col_a|=src[8+0];
					uint16_t col_b=src[8+3];
					col_b<<=8;
					col_b|=src[8+2];

					uint8_t table[4][4]={
						{ (col_a>>11)<<3, ((col_a>>5)&0x3f)<<2,	((col_a)&0x1f)<<3, 255 },
						{ (col_b>>11)<<3, ((col_b>>5)&0x3f)<<2,	((col_b)&0x1f)<<3, 255 },
						{0,0,0,255},
						{0,0,0,255}
					};

					//always gradient
					table[2][0]=(int(table[0][0])*2+int(table[1][0]))/3;
					table[2][1]=(int(table[0][1])*2+int(table[1][1]))/3;
					table[2][2]=(int(table[0][2])*2+int(table[1][2]))/3;
					table[3][0]=(int(table[0][0])+int(table[1][0])*2)/3;
					table[3][1]=(int(table[0][1])+int(table[1][1])*2)/3;
					table[3][2]=(int(table[0][2])+int(table[1][2])*2)/3;


					uint32_t block=src[4+8];
					block<<=8;
					block|=src[5+8];
					block<<=8;
					block|=src[6+8];
					block<<=8;
					block|=src[7+8];

					int y = (j/(wd/4))*4;
					int x = (j%(wd/4))*4;
					int pixofs = (y*wd+x)*4;



					for(int k=0;k<16;k++) {
						uint8_t alpha = ablock&0x7;
						int idx = pixofs+ofs_table[k];
						dst[idx+0]=table[block&0x3][0];
						dst[idx+1]=table[block&0x3][1];
						dst[idx+2]=table[block&0x3][2];
						dst[idx+3]=atable[alpha];
						block>>=2;
						ablock>>=3;
					}

				}

				rofs+=len*16;
				wofs+=wd*ht*4;


				wd/=2;
				ht/=2;

			} break;
		}

	}

	w=DVector<uint8_t>::Write();
	r=DVector<uint8_t>::Read();

	data=newdata;
	format=FORMAT_RGBA;

	return OK;
}

bool Image::is_compressed() const {
	return format>=FORMAT_BC1;
}


Image Image::decompressed() const {

	Image img=*this;
	img.decompress();
	return img;
}

Error Image::decompress() {

	if (format>=FORMAT_BC1 && format<=FORMAT_BC5 )
		_decompress_bc();//_image_decompress_bc(this);
	else if (format>=FORMAT_PVRTC2 && format<=FORMAT_PVRTC4_ALPHA && _image_decompress_pvrtc)
		_image_decompress_pvrtc(this);
	else if (format==FORMAT_ETC && _image_decompress_etc)
		_image_decompress_etc(this);
	else
		return ERR_UNAVAILABLE;
	return OK;
}


Error Image::compress(CompressMode p_mode) {

	switch(p_mode) {

		case COMPRESS_BC: {

			ERR_FAIL_COND_V(!_image_compress_bc_func, ERR_UNAVAILABLE);
			_image_compress_bc_func(this);
		} break;
		case COMPRESS_PVRTC2: {

			ERR_FAIL_COND_V(!_image_compress_pvrtc2_func, ERR_UNAVAILABLE);
			_image_compress_pvrtc2_func(this);
		} break;
		case COMPRESS_PVRTC4: {

			ERR_FAIL_COND_V(!_image_compress_pvrtc4_func, ERR_UNAVAILABLE);
			_image_compress_pvrtc4_func(this);
		} break;
		case COMPRESS_ETC: {

			ERR_FAIL_COND_V(!_image_compress_etc_func, ERR_UNAVAILABLE);
			_image_compress_etc_func(this);
		} break;
	}


	return OK;
}

Image Image::compressed(int p_mode) {

	Image ret = *this;
	ret.compress((Image::CompressMode)p_mode);

	return ret;
};

Image::Image(const char **p_xpm) {
	
	width=0;
	height=0;
	mipmaps=0;
	format=FORMAT_GRAYSCALE;

	create(p_xpm);
}


Image::Image(int p_width, int p_height,bool p_use_mipmaps, Format p_format) {

	width=0;
	height=0;
	mipmaps=0;
	format=FORMAT_GRAYSCALE;

	create(p_width,p_height,p_use_mipmaps,p_format);

}

Image::Image(int p_width, int p_height, int p_mipmaps, Format p_format, const DVector<uint8_t>& p_data) {

	width=0;
	height=0;
	mipmaps=0;
	format=FORMAT_GRAYSCALE;

	create(p_width,p_height,p_mipmaps,p_format,p_data);

}


Image Image::brushed(const Image& p_src, const Image& p_brush, const Point2& p_dest) const {

	Image img = *this;
	img.brush_transfer(p_src,p_brush,p_dest);
	return img;
}

Rect2 Image::get_used_rect() const {

	if (format==FORMAT_GRAYSCALE ||
	    format==FORMAT_RGB ||
	    format==FORMAT_INDEXED || format>FORMAT_INDEXED_ALPHA)
		return Rect2(Point2(),Size2(width,height));

	int len = data.size();

	if (len==0)
		return Rect2();

	int data_size = len;
	DVector<uint8_t>::Read r = data.read();
	const unsigned char *rptr=r.ptr();

	int minx=0xFFFFFF,miny=0xFFFFFFF;
	int maxx=-1,maxy=-1;
	for(int i=0;i<width;i++) {
		for(int j=0;j<height;j++) {

			bool opaque = _get_pixel(i,j,rptr,data_size).a>2;
			if (!opaque)
				continue;
			if (i>maxx)
				maxx=i;
			if (j>maxy)
				maxy=j;
			if (i<minx)
				minx=i;
			if (j<miny)
				miny=j;
		}
	}

	if (maxx==-1)
		return Rect2();
	else
		return Rect2(minx,miny,maxx-minx+1,maxy-miny+1);

}


Image Image::get_rect(const Rect2& p_area) const {

	Image img(p_area.size.x, p_area.size.y, mipmaps, format);
	img.blit_rect(*this, p_area, Point2(0, 0));

	return img;
};

void Image::brush_transfer(const Image& p_src, const Image& p_brush, const Point2& p_dest) {


	ERR_FAIL_COND( width != p_src.width || height !=p_src.height);

	int dst_data_size = data.size();
	DVector<uint8_t>::Write wp = data.write();
	unsigned char *dst_data_ptr=wp.ptr();


	int src_data_size = p_src.data.size();
	DVector<uint8_t>::Read rp = p_src.data.read();
	const unsigned char *src_data_ptr=rp.ptr();

	int brush_data_size = p_brush.data.size();
	DVector<uint8_t>::Read bp = p_brush.data.read();
	const unsigned char *src_brush_ptr=bp.ptr();

	int bw = p_brush.get_width();
	int bh = p_brush.get_height();
	int dx=p_dest.x;
	int dy=p_dest.y;

	for(int i=dy;i<dy+bh;i++) {

		if (i<0 || i >= height)
			continue;
		for(int j=dx;j<dx+bw;j++) {

			if (j<0 || j>=width)
				continue;

			BColor src = p_src._get_pixel(j,i,src_data_ptr,src_data_size);
			BColor dst = _get_pixel(j,i,dst_data_ptr,dst_data_size);
			BColor brush = p_brush._get_pixel(j-dx,i-dy,src_brush_ptr,brush_data_size);
			uint32_t mult = brush.r;
			dst.r = dst.r + (((int32_t(src.r)-int32_t(dst.r))*mult)>>8);
			dst.g = dst.g + (((int32_t(src.g)-int32_t(dst.g))*mult)>>8);
			dst.b = dst.b + (((int32_t(src.b)-int32_t(dst.b))*mult)>>8);
			dst.a = dst.a + (((int32_t(src.a)-int32_t(dst.a))*mult)>>8);
			_put_pixel(j,i,dst,dst_data_ptr);
		}
	}
}


void Image::blit_rect(const Image& p_src, const Rect2& p_src_rect,const Point2& p_dest) {

	int dsize=data.size();
	int srcdsize=p_src.data.size();
	ERR_FAIL_COND( dsize==0 );
	ERR_FAIL_COND( srcdsize==0 );



	Rect2 rrect = Rect2(0,0,p_src.width,p_src.height).clip(p_src_rect);

	DVector<uint8_t>::Write wp = data.write();
	unsigned char *dst_data_ptr=wp.ptr();

	DVector<uint8_t>::Read rp = p_src.data.read();
	const unsigned char *src_data_ptr=rp.ptr();

	if ((format==FORMAT_INDEXED || format == FORMAT_INDEXED_ALPHA) && (p_src.format==FORMAT_INDEXED || p_src.format == FORMAT_INDEXED_ALPHA)) {

		Point2i desti(p_dest.x, p_dest.y);
		Point2i srci(rrect.pos.x, rrect.pos.y);

		for(int i=0;i<rrect.size.y;i++) {

			if (i<0 || i >= height)
				continue;
			for(int j=0;j<rrect.size.x;j++) {

				if (j<0 || j>=width)
					continue;

				dst_data_ptr[width * (desti.y + i) + desti.x + j] = src_data_ptr[p_src.width * (srci.y+i) + srci.x+j];
			}
		}

	} else {

		for(int i=0;i<rrect.size.y;i++) {

			if (i<0 || i >= height)
				continue;
			for(int j=0;j<rrect.size.x;j++) {

				if (j<0 || j>=width)
					continue;

				_put_pixel(p_dest.x+j,p_dest.y+i,p_src._get_pixel(rrect.pos.x+j,rrect.pos.y+i,src_data_ptr,srcdsize),dst_data_ptr);
			}
		}
	}

}


Image (*Image::_png_mem_loader_func)(const uint8_t*,int)=NULL;
void (*Image::_image_compress_bc_func)(Image *)=NULL;
void (*Image::_image_compress_pvrtc2_func)(Image *)=NULL;
void (*Image::_image_compress_pvrtc4_func)(Image *)=NULL;
void (*Image::_image_compress_etc_func)(Image *)=NULL;
void (*Image::_image_decompress_pvrtc)(Image *)=NULL;
void (*Image::_image_decompress_bc)(Image *)=NULL;
void (*Image::_image_decompress_etc)(Image *)=NULL;

DVector<uint8_t> (*Image::lossy_packer)(const Image& ,float )=NULL;
Image (*Image::lossy_unpacker)(const DVector<uint8_t>& )=NULL;
DVector<uint8_t> (*Image::lossless_packer)(const Image& )=NULL;
Image (*Image::lossless_unpacker)(const DVector<uint8_t>& )=NULL;

void Image::set_compress_bc_func(void (*p_compress_func)(Image *)) {

	_image_compress_bc_func=p_compress_func;
}



void Image::normalmap_to_xy() {

	convert(Image::FORMAT_RGBA);

	{
		int len = data.size()/4;
		DVector<uint8_t>::Write wp = data.write();
		unsigned char *data_ptr=wp.ptr();

		for(int i=0;i<len;i++) {

			data_ptr[(i<<2)+3]=data_ptr[(i<<2)+0]; //x to w
			data_ptr[(i<<2)+0]=data_ptr[(i<<2)+1]; //y to xz
			data_ptr[(i<<2)+2]=data_ptr[(i<<2)+1];
		}
	}

	convert(Image::FORMAT_GRAYSCALE_ALPHA);
}

void Image::srgb_to_linear() {

	if (data.size()==0)
		return;

	static const uint8_t srgb2lin[256]={0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 13, 13, 13, 14, 14, 15, 15, 16, 16, 16, 17, 17, 18, 18, 19, 19, 20, 20, 21, 22, 22, 23, 23, 24, 24, 25, 26, 26, 27, 27, 28, 29, 29, 30, 31, 31, 32, 33, 33, 34, 35, 36, 36, 37, 38, 38, 39, 40, 41, 42, 42, 43, 44, 45, 46, 47, 47, 48, 49, 50, 51, 52, 53, 54, 55, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70, 71, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 87, 88, 89, 90, 92, 93, 94, 95, 97, 98, 99, 101, 102, 103, 105, 106, 107, 109, 110, 112, 113, 114, 116, 117, 119, 120, 122, 123, 125, 126, 128, 129, 131, 132, 134, 135, 137, 139, 140, 142, 144, 145, 147, 148, 150, 152, 153, 155, 157, 159, 160, 162, 164, 166, 167, 169, 171, 173, 175, 176, 178, 180, 182, 184, 186, 188, 190, 192, 193, 195, 197, 199, 201, 203, 205, 207, 209, 211, 213, 215, 218, 220, 222, 224, 226, 228, 230, 232, 235, 237, 239, 241, 243, 245, 248, 250, 252};


	ERR_FAIL_COND( format!=FORMAT_RGB && format!=FORMAT_RGBA  );

	if (format==FORMAT_RGBA) {

		int len = data.size()/4;
		DVector<uint8_t>::Write wp = data.write();
		unsigned char *data_ptr=wp.ptr();

		for(int i=0;i<len;i++) {

			data_ptr[(i<<2)+0]=srgb2lin[ data_ptr[(i<<2)+0] ];
			data_ptr[(i<<2)+1]=srgb2lin[ data_ptr[(i<<2)+1] ];
			data_ptr[(i<<2)+2]=srgb2lin[ data_ptr[(i<<2)+2] ];
		}

	} else if (format==FORMAT_RGB) {

		int len = data.size()/3;
		DVector<uint8_t>::Write wp = data.write();
		unsigned char *data_ptr=wp.ptr();

		for(int i=0;i<len;i++) {

			data_ptr[(i*3)+0]=srgb2lin[ data_ptr[(i*3)+0] ];
			data_ptr[(i*3)+1]=srgb2lin[ data_ptr[(i*3)+1] ];
			data_ptr[(i*3)+2]=srgb2lin[ data_ptr[(i*3)+2] ];
		}
	}

}

void Image::premultiply_alpha() {

	if (data.size()==0)
		return;

	if (format!=FORMAT_RGBA)
		return; //not needed

	DVector<uint8_t>::Write wp = data.write();
	unsigned char *data_ptr=wp.ptr();


	for(int i=0;i<height;i++) {
		for(int j=0;j<width;j++) {

			BColor bc = _get_pixel(j,i,data_ptr,0);
			bc.r=(int(bc.r)*int(bc.a))>>8;
			bc.g=(int(bc.g)*int(bc.a))>>8;
			bc.b=(int(bc.b)*int(bc.a))>>8;
			_put_pixel(j,i,bc,data_ptr);
		}
	}
}

void Image::fix_alpha_edges() {

	if (data.size()==0)
		return;

	if (format!=FORMAT_RGBA)
		return; //not needed

	DVector<uint8_t> dcopy = data;
	DVector<uint8_t>::Read rp = data.read();
	const uint8_t *rptr=rp.ptr();

	DVector<uint8_t>::Write wp = data.write();
	unsigned char *data_ptr=wp.ptr();

	const int max_radius=4;
	const int alpha_treshold=20;
	const int max_dist=0x7FFFFFFF;

	for(int i=0;i<height;i++) {
		for(int j=0;j<width;j++) {

			BColor bc = _get_pixel(j,i,rptr,0);
			if (bc.a>=alpha_treshold)
				continue;

			int closest_dist=max_dist;
			BColor closest_color;
			closest_color.a=bc.a;
			int from_x = MAX(0,j-max_radius);
			int to_x = MIN(width-1,j+max_radius);
			int from_y = MAX(0,i-max_radius);
			int to_y = MIN(height-1,i+max_radius);

			for(int k=from_y;k<=to_y;k++) {
				for(int l=from_x;l<=to_x;l++) {

					int dy = i-k;
					int dx = j-l;
					int dist = dy*dy+dx*dx;
					if (dist>=closest_dist)
						continue;

					const uint8_t * rp = &rptr[(k*width+l)<<2];

					if (rp[3]<alpha_treshold)
						continue;

					closest_dist=dist;
					closest_color.r=rp[0];
					closest_color.g=rp[1];
					closest_color.b=rp[2];

				}
			}


			if (closest_dist!=max_dist)
				_put_pixel(j,i,closest_color,data_ptr);

		}
	}

}

String Image::get_format_name(Format p_format) {

	ERR_FAIL_INDEX_V(p_format,FORMAT_MAX,String());
	return format_names[p_format];
}

Image::Image(const uint8_t* p_png,int p_len) {

	width=0;
	height=0;
	mipmaps=0;
	format=FORMAT_GRAYSCALE;

	if (_png_mem_loader_func) {
		*this = _png_mem_loader_func(p_png,p_len);
	}
}

Image::Image() {

	width=0;
	height=0;
	mipmaps=0;
	format = FORMAT_GRAYSCALE;
}

Image::~Image() {

}


