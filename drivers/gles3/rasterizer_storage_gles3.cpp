#include "rasterizer_storage_gles3.h"
#include "globals.h"

/* TEXTURE API */

#define _EXT_COMPRESSED_RGB_PVRTC_4BPPV1_IMG                   0x8C00
#define _EXT_COMPRESSED_RGB_PVRTC_2BPPV1_IMG                   0x8C01
#define _EXT_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG                  0x8C02
#define _EXT_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG                  0x8C03

#define _EXT_COMPRESSED_SRGB_PVRTC_2BPPV1_EXT               0x8A54
#define _EXT_COMPRESSED_SRGB_PVRTC_4BPPV1_EXT               0x8A55
#define _EXT_COMPRESSED_SRGB_ALPHA_PVRTC_2BPPV1_EXT         0x8A56
#define _EXT_COMPRESSED_SRGB_ALPHA_PVRTC_4BPPV1_EXT         0x8A57


#define _EXT_COMPRESSED_RGBA_S3TC_DXT1_EXT 0x83F1
#define _EXT_COMPRESSED_RGBA_S3TC_DXT3_EXT 0x83F2
#define _EXT_COMPRESSED_RGBA_S3TC_DXT5_EXT 0x83F3

#define _EXT_COMPRESSED_LUMINANCE_LATC1_EXT                 0x8C70
#define _EXT_COMPRESSED_SIGNED_LUMINANCE_LATC1_EXT          0x8C71
#define _EXT_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT           0x8C72
#define _EXT_COMPRESSED_SIGNED_LUMINANCE_ALPHA_LATC2_EXT    0x8C73


#define _EXT_COMPRESSED_RED_RGTC1_EXT 0x8DBB
#define _EXT_COMPRESSED_RED_RGTC1 0x8DBB
#define _EXT_COMPRESSED_SIGNED_RED_RGTC1 0x8DBC
#define _EXT_COMPRESSED_RG_RGTC2 0x8DBD
#define _EXT_COMPRESSED_SIGNED_RG_RGTC2 0x8DBE
#define _EXT_COMPRESSED_SIGNED_RED_RGTC1_EXT 0x8DBC
#define _EXT_COMPRESSED_RED_GREEN_RGTC2_EXT 0x8DBD
#define _EXT_COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT 0x8DBE
#define _EXT_ETC1_RGB8_OES           0x8D64



#define _EXT_SLUMINANCE_NV                                  0x8C46
#define _EXT_SLUMINANCE_ALPHA_NV                            0x8C44
#define _EXT_SRGB8_NV                                       0x8C41
#define _EXT_SLUMINANCE8_NV                                 0x8C47
#define _EXT_SLUMINANCE8_ALPHA8_NV                          0x8C45


#define _EXT_COMPRESSED_SRGB_S3TC_DXT1_NV                   0x8C4C
#define _EXT_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_NV             0x8C4D
#define _EXT_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_NV             0x8C4E
#define _EXT_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_NV             0x8C4F



#define _EXT_ATC_RGB_AMD                        0x8C92
#define _EXT_ATC_RGBA_EXPLICIT_ALPHA_AMD        0x8C93
#define _EXT_ATC_RGBA_INTERPOLATED_ALPHA_AMD    0x87EE


#define _TEXTURE_SRGB_DECODE_EXT        0x8A48
#define _DECODE_EXT             0x8A49
#define _SKIP_DECODE_EXT        0x8A4A


#define _GL_TEXTURE_MAX_ANISOTROPY_EXT          0x84FE
#define _GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT      0x84FF



Image RasterizerStorageGLES3::_get_gl_image_and_format(const Image& p_image, Image::Format p_format, uint32_t p_flags,GLenum& r_gl_format,GLenum& r_gl_internal_format,GLenum &r_gl_type,bool &r_compressed,bool &srgb) {


	r_compressed=false;
	r_gl_format=0;
	Image image=p_image;
	srgb=false;

	bool need_decompress=false;

	switch(p_format) {

		case Image::FORMAT_L8: {
			r_gl_internal_format=GL_LUMINANCE;
			r_gl_format=GL_LUMINANCE;
			r_gl_type=GL_UNSIGNED_BYTE;

		} break;
		case Image::FORMAT_LA8: {

			r_gl_internal_format=GL_LUMINANCE_ALPHA;
			r_gl_format=GL_LUMINANCE_ALPHA;
			r_gl_type=GL_UNSIGNED_BYTE;

		} break;
		case Image::FORMAT_R8: {

			r_gl_internal_format=GL_R8;
			r_gl_format=GL_RED;
			r_gl_type=GL_UNSIGNED_BYTE;

		} break;
		case Image::FORMAT_RG8: {

			r_gl_internal_format=GL_RG8;
			r_gl_format=GL_RG;
			r_gl_type=GL_UNSIGNED_BYTE;

		} break;
		case Image::FORMAT_RGB8: {

			r_gl_internal_format=(config.srgb_decode_supported || p_flags&VS::TEXTURE_FLAG_CONVERT_TO_LINEAR)?GL_SRGB8:GL_RGB8;
			r_gl_format=GL_RGB;
			r_gl_type=GL_UNSIGNED_BYTE;
			srgb=true;

		} break;
		case Image::FORMAT_RGBA8: {

			r_gl_format=GL_RGBA;
			r_gl_internal_format=(config.srgb_decode_supported || p_flags&VS::TEXTURE_FLAG_CONVERT_TO_LINEAR)?GL_SRGB8_ALPHA8:GL_RGBA8;
			r_gl_type=GL_UNSIGNED_BYTE;
			srgb=true;

		} break;
		case Image::FORMAT_RGB565: {

			r_gl_internal_format=GL_RGB565;
			r_gl_format=GL_RGB;
			r_gl_type=GL_UNSIGNED_SHORT_5_6_5;

		} break;
		case Image::FORMAT_RGBA4444: {

			r_gl_internal_format=GL_RGBA4;
			r_gl_format=GL_RGBA;
			r_gl_type=GL_UNSIGNED_SHORT_4_4_4_4;

		} break;
		case Image::FORMAT_RGBA5551: {

			r_gl_internal_format=GL_RGB5_A1;
			r_gl_format=GL_RGBA;
			r_gl_type=GL_UNSIGNED_SHORT_5_5_5_1;


		} break;
		case Image::FORMAT_RF: {


			r_gl_internal_format=GL_R32F;
			r_gl_format=GL_RED;
			r_gl_type=GL_FLOAT;

		} break;
		case Image::FORMAT_RGF: {

			r_gl_internal_format=GL_RG32F;
			r_gl_format=GL_RG;
			r_gl_type=GL_FLOAT;

		} break;
		case Image::FORMAT_RGBF: {

			r_gl_internal_format=GL_RGB32F;
			r_gl_format=GL_RGB;
			r_gl_type=GL_FLOAT;

		} break;
		case Image::FORMAT_RGBAF: {

			r_gl_internal_format=GL_RGBA32F;
			r_gl_format=GL_RGBA;
			r_gl_type=GL_FLOAT;

		} break;
		case Image::FORMAT_RH: {
			r_gl_internal_format=GL_R32F;
			r_gl_format=GL_RED;
			r_gl_type=GL_HALF_FLOAT;
		} break;
		case Image::FORMAT_RGH: {
			r_gl_internal_format=GL_RG32F;
			r_gl_format=GL_RG;
			r_gl_type=GL_HALF_FLOAT;

		} break;
		case Image::FORMAT_RGBH: {
			r_gl_internal_format=GL_RGB32F;
			r_gl_format=GL_RGB;
			r_gl_type=GL_HALF_FLOAT;

		} break;
		case Image::FORMAT_RGBAH: {
			r_gl_internal_format=GL_RGBA32F;
			r_gl_format=GL_RGBA;
			r_gl_type=GL_HALF_FLOAT;

		} break;
		case Image::FORMAT_DXT1: {

			if (config.s3tc_supported) {


				r_gl_internal_format=(config.srgb_decode_supported || p_flags&VS::TEXTURE_FLAG_CONVERT_TO_LINEAR)?_EXT_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_NV:_EXT_COMPRESSED_RGBA_S3TC_DXT1_EXT;
				r_gl_format=GL_RGBA;
				r_gl_type=GL_UNSIGNED_BYTE;
				r_compressed=true;
				srgb=true;

			} else {

				need_decompress=true;
			}


		} break;
		case Image::FORMAT_DXT3: {


			if (config.s3tc_supported) {


				r_gl_internal_format=(config.srgb_decode_supported || p_flags&VS::TEXTURE_FLAG_CONVERT_TO_LINEAR)?_EXT_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_NV:_EXT_COMPRESSED_RGBA_S3TC_DXT3_EXT;
				r_gl_format=GL_RGBA;
				r_gl_type=GL_UNSIGNED_BYTE;
				r_compressed=true;
				srgb=true;

			} else {

				need_decompress=true;
			}


		} break;
		case Image::FORMAT_DXT5: {

			if (config.s3tc_supported) {


				r_gl_internal_format=(config.srgb_decode_supported || p_flags&VS::TEXTURE_FLAG_CONVERT_TO_LINEAR)?_EXT_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_NV:_EXT_COMPRESSED_RGBA_S3TC_DXT5_EXT;
				r_gl_format=GL_RGBA;
				r_gl_type=GL_UNSIGNED_BYTE;
				r_compressed=true;
				srgb=true;

			} else {

				need_decompress=true;
			}


		} break;
		case Image::FORMAT_ATI1: {

			if (config.latc_supported) {


				r_gl_internal_format=_EXT_COMPRESSED_LUMINANCE_LATC1_EXT;
				r_gl_format=GL_RGBA;
				r_gl_type=GL_UNSIGNED_BYTE;
				r_compressed=true;
				srgb=true;

			} else {

				need_decompress=true;
			}



		} break;
		case Image::FORMAT_ATI2: {

			if (config.latc_supported) {


				r_gl_internal_format=_EXT_COMPRESSED_LUMINANCE_ALPHA_LATC2_EXT;
				r_gl_format=GL_RGBA;
				r_gl_type=GL_UNSIGNED_BYTE;
				r_compressed=true;
			} else {

				need_decompress=true;
			}

		} break;
		case Image::FORMAT_BPTC_RGBA: {

			if (config.bptc_supported) {


				r_gl_internal_format=(config.srgb_decode_supported || p_flags&VS::TEXTURE_FLAG_CONVERT_TO_LINEAR)?GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM:GL_COMPRESSED_RGBA_BPTC_UNORM;
				r_gl_format=GL_RGBA;
				r_gl_type=GL_UNSIGNED_BYTE;
				r_compressed=true;
				srgb=true;

			} else {

				need_decompress=true;
			}
		} break;
		case Image::FORMAT_BPTC_RGBF: {

			if (config.bptc_supported) {


				r_gl_internal_format=GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT;
				r_gl_format=GL_RGB;
				r_gl_type=GL_FLOAT;
				r_compressed=true;
			} else {

				need_decompress=true;
			}
		} break;
		case Image::FORMAT_BPTC_RGBFU: {
			if (config.bptc_supported) {


				r_gl_internal_format=GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT;
				r_gl_format=GL_RGB;
				r_gl_type=GL_FLOAT;
				r_compressed=true;
			} else {

				need_decompress=true;
			}
		} break;
		case Image::FORMAT_PVRTC2: {

			if (config.pvrtc_supported) {


				r_gl_internal_format=(config.srgb_decode_supported || p_flags&VS::TEXTURE_FLAG_CONVERT_TO_LINEAR)?_EXT_COMPRESSED_SRGB_PVRTC_2BPPV1_EXT:_EXT_COMPRESSED_RGB_PVRTC_2BPPV1_IMG;
				r_gl_format=GL_RGBA;
				r_gl_type=GL_UNSIGNED_BYTE;
				r_compressed=true;
				srgb=true;

			} else {

				need_decompress=true;
			}
		} break;
		case Image::FORMAT_PVRTC2A: {

			if (config.pvrtc_supported) {


				r_gl_internal_format=(config.srgb_decode_supported || p_flags&VS::TEXTURE_FLAG_CONVERT_TO_LINEAR)?_EXT_COMPRESSED_SRGB_ALPHA_PVRTC_2BPPV1_EXT:_EXT_COMPRESSED_RGBA_PVRTC_2BPPV1_IMG;
				r_gl_format=GL_RGBA;
				r_gl_type=GL_UNSIGNED_BYTE;
				r_compressed=true;
				srgb=true;

			} else {

				need_decompress=true;
			}


		} break;
		case Image::FORMAT_PVRTC4: {

			if (config.pvrtc_supported) {


				r_gl_internal_format=(config.srgb_decode_supported || p_flags&VS::TEXTURE_FLAG_CONVERT_TO_LINEAR)?_EXT_COMPRESSED_SRGB_PVRTC_4BPPV1_EXT:_EXT_COMPRESSED_RGB_PVRTC_4BPPV1_IMG;
				r_gl_format=GL_RGBA;
				r_gl_type=GL_UNSIGNED_BYTE;
				r_compressed=true;
				srgb=true;

			} else {

				need_decompress=true;
			}

		} break;
		case Image::FORMAT_PVRTC4A: {

			if (config.pvrtc_supported) {


				r_gl_internal_format=(config.srgb_decode_supported || p_flags&VS::TEXTURE_FLAG_CONVERT_TO_LINEAR)?_EXT_COMPRESSED_SRGB_ALPHA_PVRTC_4BPPV1_EXT:_EXT_COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;
				r_gl_format=GL_RGBA;
				r_gl_type=GL_UNSIGNED_BYTE;
				r_compressed=true;
				srgb=true;

			} else {

				need_decompress=true;
			}


		} break;
		case Image::FORMAT_ETC: {

			if (config.etc_supported) {

				r_gl_internal_format=_EXT_ETC1_RGB8_OES;
				r_gl_format=GL_RGBA;
				r_gl_type=GL_UNSIGNED_BYTE;
				r_compressed=true;

			} else {

				need_decompress=true;
			}

		} break;
		case Image::FORMAT_ETC2_R11: {

			if (config.etc2_supported) {

				r_gl_internal_format=GL_COMPRESSED_R11_EAC;
				r_gl_format=GL_RED;
				r_gl_type=GL_UNSIGNED_BYTE;
				r_compressed=true;

			} else {

				need_decompress=true;
			}
		} break;
		case Image::FORMAT_ETC2_R11S: {

			if (config.etc2_supported) {

				r_gl_internal_format=GL_COMPRESSED_SIGNED_R11_EAC;
				r_gl_format=GL_RED;
				r_gl_type=GL_UNSIGNED_BYTE;
				r_compressed=true;

			} else {

				need_decompress=true;
			}
		} break;
		case Image::FORMAT_ETC2_RG11: {

			if (config.etc2_supported) {

				r_gl_internal_format=GL_COMPRESSED_RG11_EAC;
				r_gl_format=GL_RG;
				r_gl_type=GL_UNSIGNED_BYTE;
				r_compressed=true;

			} else {

				need_decompress=true;
			}
		} break;
		case Image::FORMAT_ETC2_RG11S: {
			if (config.etc2_supported) {

				r_gl_internal_format=GL_COMPRESSED_SIGNED_RG11_EAC;
				r_gl_format=GL_RG;
				r_gl_type=GL_UNSIGNED_BYTE;
				r_compressed=true;

			} else {
				need_decompress=true;
			}
		} break;
		case Image::FORMAT_ETC2_RGB8: {

			if (config.etc2_supported) {

				r_gl_internal_format=(config.srgb_decode_supported || p_flags&VS::TEXTURE_FLAG_CONVERT_TO_LINEAR)?GL_COMPRESSED_SRGB8_ETC2:GL_COMPRESSED_RGB8_ETC2;
				r_gl_format=GL_RGB;
				r_gl_type=GL_UNSIGNED_BYTE;
				r_compressed=true;
				srgb=true;


			} else {

				need_decompress=true;
			}
		} break;
		case Image::FORMAT_ETC2_RGBA8: {

			if (config.etc2_supported) {

				r_gl_internal_format=(config.srgb_decode_supported || p_flags&VS::TEXTURE_FLAG_CONVERT_TO_LINEAR)?GL_COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:GL_COMPRESSED_RGBA8_ETC2_EAC;
				r_gl_format=GL_RGBA;
				r_gl_type=GL_UNSIGNED_BYTE;
				r_compressed=true;
				srgb=true;


			} else {

				need_decompress=true;
			}
		} break;
		case Image::FORMAT_ETC2_RGB8A1: {

			if (config.etc2_supported) {

				r_gl_internal_format=(config.srgb_decode_supported || p_flags&VS::TEXTURE_FLAG_CONVERT_TO_LINEAR)?GL_COMPRESSED_SRGB8_PUNCHTHROUGH_ALPHA1_ETC2:GL_COMPRESSED_RGB8_PUNCHTHROUGH_ALPHA1_ETC2;
				r_gl_format=GL_RGBA;
				r_gl_type=GL_UNSIGNED_BYTE;
				r_compressed=true;
				srgb=true;


			} else {

				need_decompress=true;
			}
		} break;
		default: {

			ERR_FAIL_V(Image());
		}
	}

	if (need_decompress) {

		if (!image.empty()) {
			image.decompress();
			ERR_FAIL_COND_V(image.is_compressed(),image);
			image.convert(Image::FORMAT_RGBA8);
		}


		r_gl_format=GL_RGBA;
		r_gl_internal_format=(config.srgb_decode_supported || p_flags&VS::TEXTURE_FLAG_CONVERT_TO_LINEAR)?GL_SRGB8_ALPHA8:GL_RGBA8;
		r_gl_type=GL_UNSIGNED_BYTE;
		r_compressed=false;
		srgb=true;

		return image;

	}


	return image;
}

static const GLenum _cube_side_enum[6]={

	GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
	GL_TEXTURE_CUBE_MAP_POSITIVE_X,
	GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
	GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
	GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
	GL_TEXTURE_CUBE_MAP_POSITIVE_Z,

};

RID RasterizerStorageGLES3::texture_create() {

	Texture *texture = memnew(Texture);
	ERR_FAIL_COND_V(!texture,RID());
	glGenTextures(1, &texture->tex_id);
	texture->active=false;
	texture->total_data_size=0;

	return texture_owner.make_rid( texture );

}

void RasterizerStorageGLES3::texture_allocate(RID p_texture,int p_width, int p_height,Image::Format p_format,uint32_t p_flags) {

	int components;
	GLenum format;
	GLenum internal_format;
	GLenum type;

	bool compressed;
	bool srgb;

	if (p_flags&VS::TEXTURE_FLAG_USED_FOR_STREAMING) {
		p_flags&=~VS::TEXTURE_FLAG_MIPMAPS; // no mipies for video
	}


	Texture *texture = texture_owner.get( p_texture );
	ERR_FAIL_COND(!texture);
	texture->width=p_width;
	texture->height=p_height;
	texture->format=p_format;
	texture->flags=p_flags;
	texture->target = (p_flags & VS::TEXTURE_FLAG_CUBEMAP) ? GL_TEXTURE_CUBE_MAP : GL_TEXTURE_2D;

	_get_gl_image_and_format(Image(),texture->format,texture->flags,format,internal_format,type,compressed,srgb);

	texture->alloc_width = texture->width;
	texture->alloc_height = texture->height;


	texture->gl_format_cache=format;
	texture->gl_type_cache=type;
	texture->gl_internal_format_cache=internal_format;
	texture->compressed=compressed;
	texture->srgb=srgb;
	texture->data_size=0;
	texture->mipmaps=1;


	glActiveTexture(GL_TEXTURE0);
	glBindTexture(texture->target, texture->tex_id);


	if (p_flags&VS::TEXTURE_FLAG_USED_FOR_STREAMING) {
		//prealloc if video
		glTexImage2D(texture->target, 0, internal_format, p_width, p_height, 0, format, type,NULL);
	}

	texture->active=true;
}

void RasterizerStorageGLES3::texture_set_data(RID p_texture,const Image& p_image,VS::CubeMapSide p_cube_side) {

	Texture * texture = texture_owner.get(p_texture);

	ERR_FAIL_COND(!texture);
	ERR_FAIL_COND(!texture->active);
	ERR_FAIL_COND(texture->render_target);
	ERR_FAIL_COND(texture->format != p_image.get_format() );
	ERR_FAIL_COND( p_image.empty() );

	GLenum type;
	GLenum format;
	GLenum internal_format;
	bool compressed;
	bool srgb;


	Image img = _get_gl_image_and_format(p_image, p_image.get_format(),texture->flags,format,internal_format,type,compressed,srgb);

	if (config.shrink_textures_x2 && (p_image.has_mipmaps() || !p_image.is_compressed()) && !(texture->flags&VS::TEXTURE_FLAG_USED_FOR_STREAMING)) {

		texture->alloc_height = MAX(1,texture->alloc_height/2);
		texture->alloc_width = MAX(1,texture->alloc_width/2);

		if (texture->alloc_width == img.get_width()/2 && texture->alloc_height == img.get_height()/2) {

			img.shrink_x2();
		} else if (img.get_format() <= Image::FORMAT_RGB565) {

			img.resize(texture->alloc_width, texture->alloc_height, Image::INTERPOLATE_BILINEAR);

		}
	};


	GLenum blit_target = (texture->target == GL_TEXTURE_CUBE_MAP)?_cube_side_enum[p_cube_side]:GL_TEXTURE_2D;

	texture->data_size=img.get_data().size();
	DVector<uint8_t>::Read read = img.get_data().read();

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(texture->target, texture->tex_id);

	texture->ignore_mipmaps = compressed && !img.has_mipmaps();

	if (texture->flags&VS::TEXTURE_FLAG_MIPMAPS && !texture->ignore_mipmaps)
		glTexParameteri(texture->target,GL_TEXTURE_MIN_FILTER,config.use_fast_texture_filter?GL_LINEAR_MIPMAP_NEAREST:GL_LINEAR_MIPMAP_LINEAR);
	else {
		if (texture->flags&VS::TEXTURE_FLAG_FILTER) {
			glTexParameteri(texture->target,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
		} else {
			glTexParameteri(texture->target,GL_TEXTURE_MIN_FILTER,GL_NEAREST);

		}
	}


	if (config.srgb_decode_supported && srgb) {

		if (texture->flags&VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) {

			glTexParameteri(texture->target,_TEXTURE_SRGB_DECODE_EXT,_DECODE_EXT);
		} else {
			glTexParameteri(texture->target,_TEXTURE_SRGB_DECODE_EXT,_SKIP_DECODE_EXT);
		}
	}

	if (texture->flags&VS::TEXTURE_FLAG_FILTER) {

		glTexParameteri(texture->target,GL_TEXTURE_MAG_FILTER,GL_LINEAR);	// Linear Filtering

	} else {

		glTexParameteri(texture->target,GL_TEXTURE_MAG_FILTER,GL_NEAREST);	// raw Filtering
	}

	if ((texture->flags&VS::TEXTURE_FLAG_REPEAT || texture->flags&VS::TEXTURE_FLAG_MIRRORED_REPEAT) && texture->target != GL_TEXTURE_CUBE_MAP) {

		if (texture->flags&VS::TEXTURE_FLAG_MIRRORED_REPEAT){
			glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT );
			glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT );
		}
		else{
			glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
			glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
		}
	} else {

		//glTexParameterf( texture->target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );
		glTexParameterf( texture->target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
		glTexParameterf( texture->target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
	}

	if (config.use_anisotropic_filter) {

		if (texture->flags&VS::TEXTURE_FLAG_ANISOTROPIC_FILTER) {

			glTexParameterf(texture->target, _GL_TEXTURE_MAX_ANISOTROPY_EXT, config.anisotropic_level);
		} else {
			glTexParameterf(texture->target, _GL_TEXTURE_MAX_ANISOTROPY_EXT, 1);
		}
	}

	int mipmaps= (texture->flags&VS::TEXTURE_FLAG_MIPMAPS && img.has_mipmaps()) ? img.get_mipmap_count() +1: 1;


	int w=img.get_width();
	int h=img.get_height();

	int tsize=0;
	for(int i=0;i<mipmaps;i++) {

		int size,ofs;
		img.get_mipmap_offset_and_size(i,ofs,size);

		//print_line("mipmap: "+itos(i)+" size: "+itos(size)+" w: "+itos(mm_w)+", h: "+itos(mm_h));

		if (texture->compressed) {
			glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
			glCompressedTexImage2D( blit_target, i, format,w,h,0,size,&read[ofs] );

		} else {
			glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
			if (texture->flags&VS::TEXTURE_FLAG_USED_FOR_STREAMING) {
				glTexSubImage2D( blit_target, i, 0,0,w, h,format,type,&read[ofs] );
			} else {
				glTexImage2D(blit_target, i, internal_format, w, h, 0, format, type,&read[ofs]);
			}

		}
		tsize+=size;

		w = MAX(1,w>>1);
		h = MAX(1,h>>1);

	}

	info.texture_mem-=texture->total_data_size;
	texture->total_data_size=tsize;
	info.texture_mem+=texture->total_data_size;

	//printf("texture: %i x %i - size: %i - total: %i\n",texture->width,texture->height,tsize,_rinfo.texture_mem);


	if (texture->flags&VS::TEXTURE_FLAG_MIPMAPS && mipmaps==1 && !texture->ignore_mipmaps) {
		//generate mipmaps if they were requested and the image does not contain them
		glGenerateMipmap(texture->target);
	}

	texture->mipmaps=mipmaps;

	//texture_set_flags(p_texture,texture->flags);


}

Image RasterizerStorageGLES3::texture_get_data(RID p_texture,VS::CubeMapSide p_cube_side) const {

	Texture * texture = texture_owner.get(p_texture);

	ERR_FAIL_COND_V(!texture,Image());
	ERR_FAIL_COND_V(!texture->active,Image());
	ERR_FAIL_COND_V(texture->data_size==0,Image());
	ERR_FAIL_COND_V(texture->render_target,Image());

#ifdef GLEW_ENABLED

	DVector<uint8_t> data;

	int data_size = Image::get_image_data_size(texture->width,texture->height,texture->format,texture->mipmaps>1?-1:0);

	data.resize(data_size);
	DVector<uint8_t>::Write wb = data.write();

	glActiveTexture(GL_TEXTURE0);

	glBindTexture(texture->target,texture->tex_id);

	for(int i=0;i<texture->mipmaps;i++) {

		int ofs=0;
		if (i>0) {
			ofs=Image::get_image_data_size(texture->alloc_width,texture->alloc_height,texture->format,i-1);
		}

		if (texture->compressed) {

			glPixelStorei(GL_PACK_ALIGNMENT, 4);
			glGetCompressedTexImage(texture->target,i,&wb[ofs]);

		} else {
			glPixelStorei(GL_PACK_ALIGNMENT, 1);
			glGetTexImage(texture->target,i,texture->gl_format_cache,texture->gl_type_cache,&wb[ofs]);
		}
	}


	wb=DVector<uint8_t>::Write();

	Image img(texture->alloc_width,texture->alloc_height,texture->mipmaps>1?true:false,texture->format,data);

	return img;
#else

	ERR_EXPLAIN("Sorry, It's not posible to obtain images back in OpenGL ES");
#endif
}

void RasterizerStorageGLES3::texture_set_flags(RID p_texture,uint32_t p_flags) {

	Texture *texture = texture_owner.get( p_texture );
	ERR_FAIL_COND(!texture);
	if (texture->render_target) {

		p_flags&=VS::TEXTURE_FLAG_FILTER;//can change only filter
	}

	bool had_mipmaps = texture->flags&VS::TEXTURE_FLAG_MIPMAPS;

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(texture->target, texture->tex_id);
	uint32_t cube = texture->flags & VS::TEXTURE_FLAG_CUBEMAP;
	texture->flags=p_flags|cube; // can't remove a cube from being a cube


	if ((texture->flags&VS::TEXTURE_FLAG_REPEAT || texture->flags&VS::TEXTURE_FLAG_MIRRORED_REPEAT) && texture->target != GL_TEXTURE_CUBE_MAP) {

		if (texture->flags&VS::TEXTURE_FLAG_MIRRORED_REPEAT){
			glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT );
			glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT );
		}
		else {
			glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
			glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
		}
	} else {
		//glTexParameterf( texture->target, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );
		glTexParameterf( texture->target, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
		glTexParameterf( texture->target, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );

	}


	if (config.use_anisotropic_filter) {

		if (texture->flags&VS::TEXTURE_FLAG_ANISOTROPIC_FILTER) {

			glTexParameterf(texture->target, _GL_TEXTURE_MAX_ANISOTROPY_EXT, config.anisotropic_level);
		} else {
			glTexParameterf(texture->target, _GL_TEXTURE_MAX_ANISOTROPY_EXT, 1);
		}
	}

	if (texture->flags&VS::TEXTURE_FLAG_MIPMAPS && !texture->ignore_mipmaps) {
		if (!had_mipmaps && texture->mipmaps==1) {
			glGenerateMipmap(texture->target);
		}
		glTexParameteri(texture->target,GL_TEXTURE_MIN_FILTER,config.use_fast_texture_filter?GL_LINEAR_MIPMAP_NEAREST:GL_LINEAR_MIPMAP_LINEAR);

	} else{
		if (texture->flags&VS::TEXTURE_FLAG_FILTER) {
			glTexParameteri(texture->target,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
		} else {
			glTexParameteri(texture->target,GL_TEXTURE_MIN_FILTER,GL_NEAREST);

		}
	}


	if (config.srgb_decode_supported && texture->srgb) {

		if (texture->flags&VS::TEXTURE_FLAG_CONVERT_TO_LINEAR) {

			glTexParameteri(texture->target,_TEXTURE_SRGB_DECODE_EXT,_DECODE_EXT);
		} else {
			glTexParameteri(texture->target,_TEXTURE_SRGB_DECODE_EXT,_SKIP_DECODE_EXT);
		}
	}

	if (texture->flags&VS::TEXTURE_FLAG_FILTER) {

		glTexParameteri(texture->target,GL_TEXTURE_MAG_FILTER,GL_LINEAR);	// Linear Filtering

	} else {

		glTexParameteri(texture->target,GL_TEXTURE_MAG_FILTER,GL_NEAREST);	// raw Filtering
	}

}
uint32_t RasterizerStorageGLES3::texture_get_flags(RID p_texture) const {

	Texture * texture = texture_owner.get(p_texture);

	ERR_FAIL_COND_V(!texture,0);

	return texture->flags;

}
Image::Format RasterizerStorageGLES3::texture_get_format(RID p_texture) const {

	Texture * texture = texture_owner.get(p_texture);

	ERR_FAIL_COND_V(!texture,Image::FORMAT_L8);

	return texture->format;
}
uint32_t RasterizerStorageGLES3::texture_get_width(RID p_texture) const {

	Texture * texture = texture_owner.get(p_texture);

	ERR_FAIL_COND_V(!texture,0);

	return texture->width;
}
uint32_t RasterizerStorageGLES3::texture_get_height(RID p_texture) const {

	Texture * texture = texture_owner.get(p_texture);

	ERR_FAIL_COND_V(!texture,0);

	return texture->height;
}


void RasterizerStorageGLES3::texture_set_size_override(RID p_texture,int p_width, int p_height) {

	Texture * texture = texture_owner.get(p_texture);

	ERR_FAIL_COND(!texture);
	ERR_FAIL_COND(texture->render_target);

	ERR_FAIL_COND(p_width<=0 || p_width>16384);
	ERR_FAIL_COND(p_height<=0 || p_height>16384);
	//real texture size is in alloc width and height
	texture->width=p_width;
	texture->height=p_height;

}

void RasterizerStorageGLES3::texture_set_path(RID p_texture,const String& p_path) {
	Texture * texture = texture_owner.get(p_texture);
	ERR_FAIL_COND(!texture);

	texture->path=p_path;

}

String RasterizerStorageGLES3::texture_get_path(RID p_texture) const{

	Texture * texture = texture_owner.get(p_texture);
	ERR_FAIL_COND_V(!texture,String());
	return texture->path;
}
void RasterizerStorageGLES3::texture_debug_usage(List<VS::TextureInfo> *r_info){

	List<RID> textures;
	texture_owner.get_owned_list(&textures);

	for (List<RID>::Element *E=textures.front();E;E=E->next()) {

		Texture *t = texture_owner.get(E->get());
		if (!t)
			continue;
		VS::TextureInfo tinfo;
		tinfo.path=t->path;
		tinfo.format=t->format;
		tinfo.size.x=t->alloc_width;
		tinfo.size.y=t->alloc_height;
		tinfo.bytes=t->total_data_size;
		r_info->push_back(tinfo);
	}

}

void RasterizerStorageGLES3::texture_set_shrink_all_x2_on_set_data(bool p_enable) {

	config.shrink_textures_x2=p_enable;
}



/* SHADER API */


RID RasterizerStorageGLES3::shader_create(VS::ShaderMode p_mode){

	return RID();
}

void RasterizerStorageGLES3::shader_set_mode(RID p_shader,VS::ShaderMode p_mode){


}
VS::ShaderMode RasterizerStorageGLES3::shader_get_mode(RID p_shader) const {

	return VS::SHADER_SPATIAL;
}
void RasterizerStorageGLES3::shader_set_code(RID p_shader, const String& p_code){


}
String RasterizerStorageGLES3::shader_get_code(RID p_shader) const{

	return String();
}
void RasterizerStorageGLES3::shader_get_param_list(RID p_shader, List<PropertyInfo> *p_param_list) const{


}

void RasterizerStorageGLES3::shader_set_default_texture_param(RID p_shader, const StringName& p_name, RID p_texture){


}
RID RasterizerStorageGLES3::shader_get_default_texture_param(RID p_shader, const StringName& p_name) const{

	return RID();
}


/* COMMON MATERIAL API */

RID RasterizerStorageGLES3::material_create(){

	return RID();
}

void RasterizerStorageGLES3::material_set_shader(RID p_shader_material, RID p_shader){


}
RID RasterizerStorageGLES3::material_get_shader(RID p_shader_material) const{

	return RID();
}

void RasterizerStorageGLES3::material_set_param(RID p_material, const StringName& p_param, const Variant& p_value){


}
Variant RasterizerStorageGLES3::material_get_param(RID p_material, const StringName& p_param) const{

	return Variant();
}

/* MESH API */

RID RasterizerStorageGLES3::mesh_create(){

	return RID();
}

void RasterizerStorageGLES3::mesh_add_surface(RID p_mesh,uint32_t p_format,VS::PrimitiveType p_primitive,const DVector<uint8_t>& p_array,int p_vertex_count,const DVector<uint8_t>& p_index_array,int p_index_count,const Vector<DVector<uint8_t> >& p_blend_shapes){


}

void RasterizerStorageGLES3::mesh_set_morph_target_count(RID p_mesh,int p_amount){


}
int RasterizerStorageGLES3::mesh_get_morph_target_count(RID p_mesh) const{

	return 0;
}


void RasterizerStorageGLES3::mesh_set_morph_target_mode(RID p_mesh,VS::MorphTargetMode p_mode){


}
VS::MorphTargetMode RasterizerStorageGLES3::mesh_get_morph_target_mode(RID p_mesh) const{

	return VS::MORPH_MODE_NORMALIZED;
}

void RasterizerStorageGLES3::mesh_surface_set_material(RID p_mesh, int p_surface, RID p_material){


}
RID RasterizerStorageGLES3::mesh_surface_get_material(RID p_mesh, int p_surface) const{

	return RID();
}

int RasterizerStorageGLES3::mesh_surface_get_array_len(RID p_mesh, int p_surface) const{

	return 0;
}
int RasterizerStorageGLES3::mesh_surface_get_array_index_len(RID p_mesh, int p_surface) const{


	return 0;
}

DVector<uint8_t> RasterizerStorageGLES3::mesh_surface_get_array(RID p_mesh, int p_surface) const{

	return DVector<uint8_t>();
}
DVector<uint8_t> RasterizerStorageGLES3::mesh_surface_get_index_array(RID p_mesh, int p_surface) const{


	return DVector<uint8_t>();
}


uint32_t RasterizerStorageGLES3::mesh_surface_get_format(RID p_mesh, int p_surface) const{

	return 0;
}
VS::PrimitiveType RasterizerStorageGLES3::mesh_surface_get_primitive_type(RID p_mesh, int p_surface) const{

	return VS::PRIMITIVE_MAX;
}

void RasterizerStorageGLES3::mesh_remove_surface(RID p_mesh,int p_index){


}
int RasterizerStorageGLES3::mesh_get_surface_count(RID p_mesh) const{

	return 0;
}

void RasterizerStorageGLES3::mesh_set_custom_aabb(RID p_mesh,const AABB& p_aabb){


}
AABB RasterizerStorageGLES3::mesh_get_custom_aabb(RID p_mesh) const{

	return AABB();
}

AABB RasterizerStorageGLES3::mesh_get_aabb(RID p_mesh) const{

	return AABB();
}
void RasterizerStorageGLES3::mesh_clear(RID p_mesh){


}

/* MULTIMESH API */


RID RasterizerStorageGLES3::multimesh_create(){

	return RID();
}

void RasterizerStorageGLES3::multimesh_allocate(RID p_multimesh,int p_instances,VS::MultimeshTransformFormat p_transform_format,VS::MultimeshColorFormat p_color_format,bool p_gen_aabb){


}
int RasterizerStorageGLES3::multimesh_get_instance_count(RID p_multimesh) const{

	return 0;
}

void RasterizerStorageGLES3::multimesh_set_mesh(RID p_multimesh,RID p_mesh){


}
void RasterizerStorageGLES3::multimesh_set_custom_aabb(RID p_multimesh,const AABB& p_aabb){


}
void RasterizerStorageGLES3::multimesh_instance_set_transform(RID p_multimesh,int p_index,const Transform& p_transform){


}
void RasterizerStorageGLES3::multimesh_instance_set_transform_2d(RID p_multimesh,int p_index,const Matrix32& p_transform){


}
void RasterizerStorageGLES3::multimesh_instance_set_color(RID p_multimesh,int p_index,const Color& p_color){


}

RID RasterizerStorageGLES3::multimesh_get_mesh(RID p_multimesh) const{


	return RID();
}
AABB RasterizerStorageGLES3::multimesh_get_custom_aabb(RID p_multimesh) const{

	return AABB();
}

Transform RasterizerStorageGLES3::multimesh_instance_get_transform(RID p_multimesh,int p_index) const{

	return Transform();
}
Matrix32 RasterizerStorageGLES3::multimesh_instance_get_transform_2d(RID p_multimesh,int p_index) const{


	return Matrix32();
}
Color RasterizerStorageGLES3::multimesh_instance_get_color(RID p_multimesh,int p_index) const{

	return Color();
}

void RasterizerStorageGLES3::multimesh_set_visible_instances(RID p_multimesh,int p_visible){


}
int RasterizerStorageGLES3::multimesh_get_visible_instances(RID p_multimesh) const{

	return 0;
}

AABB RasterizerStorageGLES3::multimesh_get_aabb(RID p_mesh) const{

	return AABB();
}

/* IMMEDIATE API */

RID RasterizerStorageGLES3::immediate_create(){

	return RID();
}
void RasterizerStorageGLES3::immediate_begin(RID p_immediate,VS::PrimitiveType p_rimitive,RID p_texture){


}
void RasterizerStorageGLES3::immediate_vertex(RID p_immediate,const Vector3& p_vertex){


}
void RasterizerStorageGLES3::immediate_vertex_2d(RID p_immediate,const Vector3& p_vertex){


}
void RasterizerStorageGLES3::immediate_normal(RID p_immediate,const Vector3& p_normal){


}
void RasterizerStorageGLES3::immediate_tangent(RID p_immediate,const Plane& p_tangent){


}
void RasterizerStorageGLES3::immediate_color(RID p_immediate,const Color& p_color){


}
void RasterizerStorageGLES3::immediate_uv(RID p_immediate,const Vector2& tex_uv){


}
void RasterizerStorageGLES3::immediate_uv2(RID p_immediate,const Vector2& tex_uv){


}
void RasterizerStorageGLES3::immediate_end(RID p_immediate){


}
void RasterizerStorageGLES3::immediate_clear(RID p_immediate){


}
void RasterizerStorageGLES3::immediate_set_material(RID p_immediate,RID p_material){


}
RID RasterizerStorageGLES3::immediate_get_material(RID p_immediate) const{

	return RID();
}

/* SKELETON API */

RID RasterizerStorageGLES3::skeleton_create(){

	return RID();
}
void RasterizerStorageGLES3::skeleton_allocate(RID p_skeleton,int p_bones,bool p_2d_skeleton){


}
int RasterizerStorageGLES3::skeleton_get_bone_count(RID p_skeleton) const{

	return 0;
}
void RasterizerStorageGLES3::skeleton_bone_set_transform(RID p_skeleton,int p_bone, const Transform& p_transform){


}
Transform RasterizerStorageGLES3::skeleton_bone_get_transform(RID p_skeleton,int p_bone) const{

	return Transform();
}
void RasterizerStorageGLES3::skeleton_bone_set_transform_2d(RID p_skeleton,int p_bone, const Matrix32& p_transform){


}
Matrix32 RasterizerStorageGLES3::skeleton_bone_get_transform_2d(RID p_skeleton,int p_bone) const{

	return Matrix32();
}

/* Light API */

RID RasterizerStorageGLES3::light_create(VS::LightType p_type){

	return RID();
}

void RasterizerStorageGLES3::light_set_color(RID p_light,const Color& p_color){


}
void RasterizerStorageGLES3::light_set_param(RID p_light,VS::LightParam p_param,float p_value){


}
void RasterizerStorageGLES3::light_set_shadow(RID p_light,bool p_enabled){


}
void RasterizerStorageGLES3::light_set_projector(RID p_light,RID p_texture){


}
void RasterizerStorageGLES3::light_set_attenuation_texure(RID p_light,RID p_texture){


}
void RasterizerStorageGLES3::light_set_negative(RID p_light,bool p_enable){


}
void RasterizerStorageGLES3::light_set_cull_mask(RID p_light,uint32_t p_mask){


}
void RasterizerStorageGLES3::light_set_shader(RID p_light,RID p_shader){


}


void RasterizerStorageGLES3::light_directional_set_shadow_mode(RID p_light,VS::LightDirectionalShadowMode p_mode){


}

/* PROBE API */

RID RasterizerStorageGLES3::reflection_probe_create(){

	return RID();
}

void RasterizerStorageGLES3::reflection_probe_set_intensity(RID p_probe, float p_intensity){


}
void RasterizerStorageGLES3::reflection_probe_set_clip(RID p_probe, float p_near, float p_far){


}
void RasterizerStorageGLES3::reflection_probe_set_min_blend_distance(RID p_probe, float p_distance){


}
void RasterizerStorageGLES3::reflection_probe_set_extents(RID p_probe, const Vector3& p_extents){


}
void RasterizerStorageGLES3::reflection_probe_set_origin_offset(RID p_probe, const Vector3& p_offset){


}
void RasterizerStorageGLES3::reflection_probe_set_enable_parallax_correction(RID p_probe, bool p_enable){


}
void RasterizerStorageGLES3::reflection_probe_set_resolution(RID p_probe, int p_resolution){


}
void RasterizerStorageGLES3::reflection_probe_set_hide_skybox(RID p_probe, bool p_hide){


}
void RasterizerStorageGLES3::reflection_probe_set_cull_mask(RID p_probe, uint32_t p_layers){


}


/* ROOM API */

RID RasterizerStorageGLES3::room_create(){

	return RID();
}
void RasterizerStorageGLES3::room_add_bounds(RID p_room, const DVector<Vector2>& p_convex_polygon,float p_height,const Transform& p_transform){


}
void RasterizerStorageGLES3::room_clear_bounds(RID p_room){


}

/* PORTAL API */

// portals are only (x/y) points, forming a convex shape, which its clockwise
// order points outside. (z is 0);

RID RasterizerStorageGLES3::portal_create(){

	return RID();
}
void RasterizerStorageGLES3::portal_set_shape(RID p_portal, const Vector<Point2>& p_shape){


}
void RasterizerStorageGLES3::portal_set_enabled(RID p_portal, bool p_enabled){


}
void RasterizerStorageGLES3::portal_set_disable_distance(RID p_portal, float p_distance){


}
void RasterizerStorageGLES3::portal_set_disabled_color(RID p_portal, const Color& p_color){


}


/* RENDER TARGET */


void RasterizerStorageGLES3::_render_target_clear(RenderTarget *rt) {

	if (rt->front.fbo) {
		glDeleteFramebuffers(1,&rt->front.fbo);
		glDeleteTextures(1,&rt->front.color);
		rt->front.fbo=0;
	}

	if (rt->back.fbo) {
		glDeleteFramebuffers(1,&rt->back.fbo);
		glDeleteTextures(1,&rt->back.color);
		rt->back.fbo=0;
	}

	if (rt->deferred.fbo) {
		glDeleteFramebuffers(1,&rt->deferred.fbo);
		glDeleteFramebuffers(1,&rt->deferred.fbo_color);
		glDeleteTextures(1,&rt->deferred.albedo_ao);
		glDeleteTextures(1,&rt->deferred.normal_special);
		glDeleteTextures(1,&rt->deferred.metal_rough_motion);
		rt->deferred.fbo=0;
		rt->deferred.fbo_color=0;
	}

	if (rt->depth) {
		glDeleteRenderbuffers(1,&rt->depth);
		rt->depth=0;
	}

}

void RasterizerStorageGLES3::_render_target_allocate(RenderTarget *rt){

	if (rt->width<=0 || rt->height<=0)
		return;

	glGenFramebuffers(1, &rt->front.fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, rt->front.fbo);


	glGenRenderbuffers(1, &rt->depth);
	glBindRenderbuffer(GL_RENDERBUFFER, rt->depth );
	if (config.fbo_format==FBO_FORMAT_16_BITS) {
		glRenderbufferStorage(GL_RENDERBUFFER,GL_DEPTH_COMPONENT16, rt->width, rt->height);
	} else {
		glRenderbufferStorage(GL_RENDERBUFFER,GL_DEPTH24_STENCIL8, rt->width, rt->height);
	}
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rt->depth);
	glBindRenderbuffer(GL_RENDERBUFFER, 0 );


	glGenTextures(1, &rt->front.color);
	glBindTexture(GL_TEXTURE_2D, rt->front.color);


	GLuint color_internal_format;
	GLuint color_format;
	GLuint color_type;


	if (config.fbo_format==FBO_FORMAT_16_BITS) {

		if (rt->flags[RENDER_TARGET_TRANSPARENT]) {
			color_internal_format=GL_RGB5_A1;
			color_format=GL_RGBA;
			color_type=GL_UNSIGNED_SHORT_5_5_5_1;
		} else {
			color_internal_format=GL_RGB565;
			color_format=GL_RGB;
			color_type=GL_UNSIGNED_SHORT_5_6_5;
		}

	} else if (config.fbo_format==FBO_FORMAT_32_BITS) {

		if (rt->flags[RENDER_TARGET_TRANSPARENT]) {
			color_internal_format=GL_RGBA8;
			color_format=GL_RGBA;
			color_type=GL_UNSIGNED_BYTE;
		} else {
			color_internal_format=GL_RGB10_A2;
			color_format=GL_RGBA;
			color_type=GL_UNSIGNED_INT_2_10_10_10_REV;
		}
	} else if (config.fbo_format==FBO_FORMAT_FLOAT) {

		color_internal_format=GL_RGBA16F;
		color_format=GL_RGBA;
		color_type=GL_HALF_FLOAT;
	}

	glTexImage2D(GL_TEXTURE_2D, 0, color_internal_format,  rt->width, rt->height, 0, color_format, color_type, NULL);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt->front.color, 0);

	{
		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		glBindFramebuffer(GL_FRAMEBUFFER, config.system_fbo);

		ERR_FAIL_COND( status != GL_FRAMEBUFFER_COMPLETE );
	}


	if (!rt->flags[RENDER_TARGET_NO_SAMPLING]) {

		glGenFramebuffers(1, &rt->back.fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, rt->back.fbo);

		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rt->depth);

		glGenTextures(1, &rt->back.color);
		glBindTexture(GL_TEXTURE_2D, rt->back.color);

		glTexImage2D(GL_TEXTURE_2D, 0, color_internal_format,  rt->width, rt->height, 0, color_format, color_type, NULL);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt->back.color, 0);

		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		glBindFramebuffer(GL_FRAMEBUFFER, config.system_fbo);

		if (status != GL_FRAMEBUFFER_COMPLETE) {
			_render_target_clear(rt);
			ERR_FAIL_COND( status != GL_FRAMEBUFFER_COMPLETE );
		}
	}



	if (config.fbo_deferred && !rt->flags[RENDER_TARGET_NO_3D]) {


		//regular fbo
		glGenFramebuffers(1, &rt->deferred.fbo);
		glBindFramebuffer(GL_FRAMEBUFFER, rt->deferred.fbo);

		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rt->depth);

		glGenTextures(1, &rt->deferred.albedo_ao);
		glBindTexture(GL_TEXTURE_2D, rt->deferred.albedo_ao);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,  rt->width, rt->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt->deferred.albedo_ao, 0);

		glGenTextures(1, &rt->deferred.metal_rough_motion);
		glBindTexture(GL_TEXTURE_2D, rt->deferred.metal_rough_motion);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,  rt->width, rt->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, rt->deferred.metal_rough_motion, 0);

		glGenTextures(1, &rt->deferred.normal_special);
		glBindTexture(GL_TEXTURE_2D, rt->deferred.normal_special);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,  rt->width, rt->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, rt->deferred.normal_special, 0);


		GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		glBindFramebuffer(GL_FRAMEBUFFER, config.system_fbo);

		if (status != GL_FRAMEBUFFER_COMPLETE) {
			_render_target_clear(rt);
			ERR_FAIL_COND( status != GL_FRAMEBUFFER_COMPLETE );
		}

		//regular fbo with color attachment (needed for emission or objects rendered as forward)

		glGenFramebuffers(1, &rt->deferred.fbo_color);
		glBindFramebuffer(GL_FRAMEBUFFER, rt->deferred.fbo_color);

		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, rt->depth);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, rt->deferred.albedo_ao, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, rt->deferred.metal_rough_motion, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, rt->deferred.normal_special, 0);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT3, GL_TEXTURE_2D, rt->front.color, 0);


		status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
		glBindFramebuffer(GL_FRAMEBUFFER, config.system_fbo);

		if (status != GL_FRAMEBUFFER_COMPLETE) {
			_render_target_clear(rt);
			ERR_FAIL_COND( status != GL_FRAMEBUFFER_COMPLETE );
		}
	}


}


RID RasterizerStorageGLES3::render_target_create(){

	RenderTarget *rt = memnew( RenderTarget );
	return render_target_owner.make_rid(rt);
}

void RasterizerStorageGLES3::render_target_set_size(RID p_render_target,int p_width, int p_height){

	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND(!rt);

	if (rt->width==p_width && rt->height==p_height)
		return;

	_render_target_clear(rt);
	rt->width=p_width;
	rt->height=p_height;
	_render_target_allocate(rt);

}


RID RasterizerStorageGLES3::render_target_get_texture(RID p_render_target) const{

	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND_V(!rt,RID());


	return RID();
}
Image RasterizerStorageGLES3::render_target_get_image(RID p_render_target) const{

	return Image();
}
void RasterizerStorageGLES3::render_target_set_flag(RID p_render_target,RenderTargetFlags p_flag,bool p_value) {

	RenderTarget *rt = render_target_owner.getornull(p_render_target);
	ERR_FAIL_COND(!rt);

	rt->flags[p_flag]=p_value;

	switch(p_flag) {
		case RENDER_TARGET_NO_3D:
		case RENDER_TARGET_TRANSPARENT: {
			//must reset for these formats
			_render_target_clear(rt);
			_render_target_allocate(rt);

		} break;
		default: {}
	}
}

bool RasterizerStorageGLES3::render_target_renedered_in_frame(RID p_render_target){

	return false;
}

/* CANVAS SHADOW */


RID RasterizerStorageGLES3::canvas_light_shadow_buffer_create(int p_width) {

	CanvasLightShadow *cls = memnew( CanvasLightShadow );
	if (p_width>config.max_texture_size)
		p_width=config.max_texture_size;

	cls->size=p_width;
	cls->height=16;

	glActiveTexture(GL_TEXTURE0);

	glGenFramebuffers(1, &cls->fbo);
	glBindFramebuffer(GL_FRAMEBUFFER, cls->fbo);

	glGenRenderbuffers(1, &cls->depth);
	glBindRenderbuffer(GL_RENDERBUFFER, cls->depth );
	glRenderbufferStorage(GL_RENDERBUFFER,GL_DEPTH_COMPONENT24, cls->size, cls->height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, cls->depth);
	glBindRenderbuffer(GL_RENDERBUFFER, 0 );

	glGenTextures(1,&cls->distance);
	glBindTexture(GL_TEXTURE_2D, cls->distance);
	if (config.use_rgba_2d_shadows) {
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, cls->size, cls->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
	} else {
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, cls->size, cls->height, 0, GL_RED, GL_FLOAT, NULL);
	}




	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, cls->distance, 0);


	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	//printf("errnum: %x\n",status);
	glBindFramebuffer(GL_FRAMEBUFFER, config.system_fbo);

	ERR_FAIL_COND_V( status != GL_FRAMEBUFFER_COMPLETE, RID() );

	return canvas_light_shadow_owner.make_rid(cls);
}

/* LIGHT SHADOW MAPPING */


RID RasterizerStorageGLES3::canvas_light_occluder_create() {

	CanvasOccluder *co = memnew( CanvasOccluder );
	co->index_id=0;
	co->vertex_id=0;
	co->len=0;

	return canvas_occluder_owner.make_rid(co);
}

void RasterizerStorageGLES3::canvas_light_occluder_set_polylines(RID p_occluder, const DVector<Vector2>& p_lines) {

	CanvasOccluder *co = canvas_occluder_owner.get(p_occluder);
	ERR_FAIL_COND(!co);

	co->lines=p_lines;

	if (p_lines.size()!=co->len) {

		if (co->index_id)
			glDeleteBuffers(1,&co->index_id);
		if (co->vertex_id)
			glDeleteBuffers(1,&co->vertex_id);

		co->index_id=0;
		co->vertex_id=0;
		co->len=0;

	}

	if (p_lines.size()) {



		DVector<float> geometry;
		DVector<uint16_t> indices;
		int lc = p_lines.size();

		geometry.resize(lc*6);
		indices.resize(lc*3);

		DVector<float>::Write vw=geometry.write();
		DVector<uint16_t>::Write iw=indices.write();


		DVector<Vector2>::Read lr=p_lines.read();

		const int POLY_HEIGHT = 16384;

		for(int i=0;i<lc/2;i++) {

			vw[i*12+0]=lr[i*2+0].x;
			vw[i*12+1]=lr[i*2+0].y;
			vw[i*12+2]=POLY_HEIGHT;

			vw[i*12+3]=lr[i*2+1].x;
			vw[i*12+4]=lr[i*2+1].y;
			vw[i*12+5]=POLY_HEIGHT;

			vw[i*12+6]=lr[i*2+1].x;
			vw[i*12+7]=lr[i*2+1].y;
			vw[i*12+8]=-POLY_HEIGHT;

			vw[i*12+9]=lr[i*2+0].x;
			vw[i*12+10]=lr[i*2+0].y;
			vw[i*12+11]=-POLY_HEIGHT;

			iw[i*6+0]=i*4+0;
			iw[i*6+1]=i*4+1;
			iw[i*6+2]=i*4+2;

			iw[i*6+3]=i*4+2;
			iw[i*6+4]=i*4+3;
			iw[i*6+5]=i*4+0;

		}

		//if same buffer len is being set, just use BufferSubData to avoid a pipeline flush


		if (!co->vertex_id) {
			glGenBuffers(1,&co->vertex_id);
			glBindBuffer(GL_ARRAY_BUFFER,co->vertex_id);
			glBufferData(GL_ARRAY_BUFFER,lc*6*sizeof(real_t),vw.ptr(),GL_STATIC_DRAW);
		} else {

			glBindBuffer(GL_ARRAY_BUFFER,co->vertex_id);
			glBufferSubData(GL_ARRAY_BUFFER,0,lc*6*sizeof(real_t),vw.ptr());

		}

		glBindBuffer(GL_ARRAY_BUFFER,0); //unbind

		if (!co->index_id) {

			glGenBuffers(1,&co->index_id);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,co->index_id);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER,lc*3*sizeof(uint16_t),iw.ptr(),GL_STATIC_DRAW);
		} else {


			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,co->index_id);
			glBufferSubData(GL_ELEMENT_ARRAY_BUFFER,0,lc*3*sizeof(uint16_t),iw.ptr());
		}

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,0); //unbind

		co->len=lc;

	}

}

bool RasterizerStorageGLES3::free(RID p_rid){

	if (render_target_owner.owns(p_rid)) {

		RenderTarget *rt = render_target_owner.getornull(p_rid);
		_render_target_clear(rt);
		render_target_owner.free(p_rid);
		memdelete(rt);

	} else if (texture_owner.owns(p_rid)) {
		// delete the texture
		Texture *texture = texture_owner.get(p_rid);
		info.texture_mem-=texture->total_data_size;
		texture_owner.free(p_rid);
		memdelete(texture);
	} else if (canvas_occluder_owner.owns(p_rid)) {


		CanvasOccluder *co = canvas_occluder_owner.get(p_rid);
		if (co->index_id)
			glDeleteBuffers(1,&co->index_id);
		if (co->vertex_id)
			glDeleteBuffers(1,&co->vertex_id);

		canvas_occluder_owner.free(p_rid);
		memdelete(co);

	} else if (canvas_light_shadow_owner.owns(p_rid)) {

		CanvasLightShadow *cls = canvas_light_shadow_owner.get(p_rid);
		glDeleteFramebuffers(1,&cls->fbo);
		glDeleteRenderbuffers(1,&cls->depth);
		glDeleteTextures(1,&cls->distance);
		canvas_light_shadow_owner.free(p_rid);
		memdelete(cls);
	} else {
		return false;
	}

	return true;
}

////////////////////////////////////////////


void RasterizerStorageGLES3::initialize() {

	config.fbo_format=FBOFormat(int(Globals::get_singleton()->get("rendering/gles3/framebuffer_format")));
	config.fbo_deferred=int(Globals::get_singleton()->get("rendering/gles3/lighting_technique"));

	config.system_fbo=0;


	//// extensions config
	///

	{
		Vector<String> ext= String((const char*)glGetString( GL_EXTENSIONS )).split(" ",false);
		for(int i=0;i<ext.size();i++) {
			config.extensions.insert(ext[i]);
		}
	}

	config.shrink_textures_x2=false;
	config.use_fast_texture_filter=int(Globals::get_singleton()->get("rendering/gles3/use_nearest_mipmap_filter"));
	config.use_anisotropic_filter = config.extensions.has("GL_EXT_texture_filter_anisotropic");

	config.s3tc_supported=config.extensions.has("GL_EXT_texture_compression_dxt1") || config.extensions.has("GL_EXT_texture_compression_s3tc") || config.extensions.has("WEBGL_compressed_texture_s3tc");
	config.etc_supported=config.extensions.has("GL_OES_compressed_ETC1_RGB8_texture");
	config.latc_supported=config.extensions.has("GL_EXT_texture_compression_latc");
	config.bptc_supported=config.extensions.has("GL_ARB_texture_compression_bptc");
#ifdef GLEW_ENABLED
	config.etc2_supported=false;
#else
	config.etc2_supported=true;
#endif
	config.pvrtc_supported=config.extensions.has("GL_IMG_texture_compression_pvrtc");
	config.srgb_decode_supported=config.extensions.has("GL_EXT_texture_sRGB_decode");



	config.anisotropic_level=1.0;
	config.use_anisotropic_filter=config.extensions.has("GL_EXT_texture_filter_anisotropic");
	if (config.use_anisotropic_filter) {
		glGetFloatv(_GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT,&config.anisotropic_level);
		config.anisotropic_level=MIN(int(Globals::get_singleton()->get("rendering/gles3/anisotropic_filter_level")),config.anisotropic_level);
	}


	frame.clear_request=false;

	shaders.copy.init();

	{
		//default textures


		glGenTextures(1, &resources.white_tex);
		unsigned char whitetexdata[8*8*3];
		for(int i=0;i<8*8*3;i++) {
			whitetexdata[i]=255;
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D,resources.white_tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 8, 8, 0, GL_RGB, GL_UNSIGNED_BYTE,whitetexdata);
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D,0);

		glGenTextures(1, &resources.black_tex);
		unsigned char blacktexdata[8*8*3];
		for(int i=0;i<8*8;i++) {
			blacktexdata[i]=0;
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D,resources.black_tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 8, 8, 0, GL_RGB, GL_UNSIGNED_BYTE,blacktexdata);
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D,0);

		glGenTextures(1, &resources.normal_tex);
		unsigned char normaltexdata[8*8*3];
		for(int i=0;i<8*8*3;i+=3) {
			normaltexdata[i+0]=128;
			normaltexdata[i+1]=128;
			normaltexdata[i+2]=255;
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D,resources.normal_tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 8, 8, 0, GL_RGB, GL_UNSIGNED_BYTE,normaltexdata);
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D,0);

	}

	glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS,&config.max_texture_image_units);
	glGetIntegerv(GL_MAX_TEXTURE_SIZE,&config.max_texture_size);

#ifdef GLEW_ENABLED
	config.use_rgba_2d_shadows=false;
#else
	config.use_rgba_2d_shadows=true;
#endif
}

void RasterizerStorageGLES3::finalize() {

	glDeleteTextures(1, &resources.white_tex);
	glDeleteTextures(1, &resources.black_tex);
	glDeleteTextures(1, &resources.normal_tex);

}


RasterizerStorageGLES3::RasterizerStorageGLES3()
{

}
