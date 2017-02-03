#include "resource_importer_texture.h"
#include "io/image_loader.h"
#include "scene/resources/texture.h"

String ResourceImporterTexture::get_importer_name() const {

	return "texture";
}

String ResourceImporterTexture::get_visible_name() const{

	return "Texture";
}
void ResourceImporterTexture::get_recognized_extensions(List<String> *p_extensions) const{

	ImageLoader::get_recognized_extensions(p_extensions);
}
String ResourceImporterTexture::get_save_extension() const {
	return "stex";
}

String ResourceImporterTexture::get_resource_type() const{

	return "StreamTexture";
}

bool ResourceImporterTexture::get_option_visibility(const String& p_option,const Map<StringName,Variant>& p_options) const {

	if (p_option=="compress/lossy_quality" && int(p_options["compress/mode"])!=COMPRESS_LOSSY)
		return false;

	return true;
}

int ResourceImporterTexture::get_preset_count() const {
	return 4;
}
String ResourceImporterTexture::get_preset_name(int p_idx) const {

	static const char* preset_names[]={
		"2D, Detect 3D",
		"2D",
		"2D Pixel",
		"3D"
	};

	return preset_names[p_idx];
}


void ResourceImporterTexture::get_import_options(List<ImportOption> *r_options,int p_preset) const {


	r_options->push_back(ImportOption(PropertyInfo(Variant::INT,"compress/mode",PROPERTY_HINT_ENUM,"Lossless,Lossy,Video RAM,Uncompressed",PROPERTY_USAGE_DEFAULT|PROPERTY_USAGE_UPDATE_ALL_IF_MODIFIED),p_preset==PRESET_3D?2:0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::REAL,"compress/lossy_quality",PROPERTY_HINT_RANGE,"0,1,0.01"),0.7));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT,"flags/repeat",PROPERTY_HINT_ENUM,"Disabled,Enabled,Mirrored"),p_preset==PRESET_3D?1:0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL,"flags/filter"),p_preset==PRESET_2D_PIXEL?false:true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL,"flags/mipmaps"),p_preset==PRESET_3D?true:false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL,"flags/anisotropic"),false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL,"flags/srgb",PROPERTY_HINT_ENUM,"Disable,Enable,Detect"),2));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL,"process/fix_alpha_border"),p_preset!=PRESET_3D?true:false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL,"process/premult_alpha"),true));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT,"stream"),false));
	r_options->push_back(ImportOption(PropertyInfo(Variant::INT,"size_limit",PROPERTY_HINT_RANGE,"0,4096,1"),0));
	r_options->push_back(ImportOption(PropertyInfo(Variant::BOOL,"detect_3d"),p_preset==PRESET_DETECT));

}


void ResourceImporterTexture::_save_stex(const Image& p_image,const String& p_to_path,int p_compress_mode,float p_lossy_quality,Image::CompressMode p_vram_compression,bool p_mipmaps,int p_texture_flags,bool p_streamable) {


	FileAccess *f = FileAccess::open(p_to_path,FileAccess::WRITE);
	f->store_8('G');
	f->store_8('D');
	f->store_8('S');
	f->store_8('T'); //godot streamable texture

	f->store_32(p_image.get_width());
	f->store_32(p_image.get_height());
	f->store_32(p_texture_flags);

	uint32_t format=0;

	if (p_streamable)
		format|=StreamTexture::FORMAT_BIT_STREAM;
	if (p_mipmaps || p_compress_mode==COMPRESS_VIDEO_RAM) //VRAM always uses mipmaps
		format|=StreamTexture::FORMAT_BIT_HAS_MIPMAPS; //mipmaps bit

	switch (p_compress_mode) {
		case COMPRESS_LOSSLESS: {

			Image image = p_image;
			if (p_mipmaps) {
				image.generate_mipmaps();
			} else {
				image.clear_mipmaps();
			}

			int mmc = image.get_mipmap_count() + 1;

			format=StreamTexture::FORMAT_BIT_LOSSLESS;
			f->store_32(format);
			f->store_32(mmc);

			for(int i=0;i<mmc;i++) {

				if (i>0) {
					image.shrink_x2();
				}

				PoolVector<uint8_t> data = Image::lossless_packer(image);
				int data_len = data.size();
				f->store_32(data_len);

				PoolVector<uint8_t>::Read r= data.read();
				f->store_buffer(r.ptr(),data_len);

			}


		} break;
		case COMPRESS_LOSSY: {
			Image image = p_image;
			if (p_mipmaps) {
				image.generate_mipmaps();
			} else {
				image.clear_mipmaps();
			}

			int mmc = image.get_mipmap_count() + 1;

			format=StreamTexture::FORMAT_BIT_LOSSY;
			f->store_32(format);
			f->store_32(mmc);

			for(int i=0;i<mmc;i++) {

				if (i>0) {
					image.shrink_x2();
				}

				PoolVector<uint8_t> data = Image::lossy_packer(image,p_lossy_quality);
				int data_len = data.size();
				f->store_32(data_len);

				PoolVector<uint8_t>::Read r = data.read();
				f->store_buffer(r.ptr(),data_len);

			}
		} break;
		case COMPRESS_VIDEO_RAM: {

			Image image = p_image;
			image.generate_mipmaps();
			image.compress(p_vram_compression);

			format |= image.get_format();

			f->store_32(format);

			PoolVector<uint8_t> data=image.get_data();
			int dl = data.size();
			PoolVector<uint8_t>::Read r = data.read();

			f->store_buffer(r.ptr(),dl);

		} break;
		case COMPRESS_UNCOMPRESSED: {

			Image image = p_image;
			if (p_mipmaps) {
				image.generate_mipmaps();
			} else {
				image.clear_mipmaps();
			}

			format |= image.get_format();
			f->store_32(format);

			PoolVector<uint8_t> data=image.get_data();
			int dl = data.size();
			PoolVector<uint8_t>::Read r = data.read();

			f->store_buffer(r.ptr(),dl);

		} break;
	}

	memdelete(f);
}

Error ResourceImporterTexture::import(const String& p_source_file, const String& p_save_path, const Map<StringName,Variant>& p_options, List<String>* r_platform_variants, List<String> *r_gen_files) {

	int compress_mode = p_options["compress/mode"];
	float lossy= p_options["compress/lossy_quality"];
	int repeat= p_options["flags/repeat"];
	bool filter= p_options["flags/filter"];
	bool mipmaps= p_options["flags/mipmaps"];
	bool anisotropic= p_options["flags/anisotropic"];
	bool srgb= p_options["flags/srgb"];
	bool fix_alpha_border= p_options["process/fix_alpha_border"];
	bool premult_alpha= p_options["process/premult_alpha"];
	bool stream = p_options["stream"];
	int size_limit = p_options["size_limit"];


	Image image;
	Error err = ImageLoader::load_image(p_source_file,&image);
	if (err!=OK)
		return err;


	int tex_flags=0;
	if (repeat>0)
		tex_flags|=Texture::FLAG_REPEAT;
	if (repeat==2)
		tex_flags|=Texture::FLAG_MIRRORED_REPEAT;
	if (filter)
		tex_flags|=Texture::FLAG_FILTER;
	if (mipmaps || compress_mode==COMPRESS_VIDEO_RAM)
		tex_flags|=Texture::FLAG_MIPMAPS;
	if (anisotropic)
		tex_flags|=Texture::FLAG_ANISOTROPIC_FILTER;
	if (srgb)
		tex_flags|=Texture::FLAG_CONVERT_TO_LINEAR;

	if (size_limit >0 && (image.get_width()>size_limit || image.get_height()>size_limit )) {
		//limit size
		if (image.get_width() >= image.get_height()) {
			int new_width = size_limit;
			int new_height = image.get_height() * new_width / image.get_width();

			image.resize(new_width,new_height,Image::INTERPOLATE_CUBIC);
		} else {

			int new_height = size_limit;
			int new_width = image.get_width() * new_height / image.get_height();

			image.resize(new_width,new_height,Image::INTERPOLATE_CUBIC);
		}
	}

	if (fix_alpha_border) {
		image.fix_alpha_edges();
	}

	if (premult_alpha) {
		image.premultiply_alpha();
	}


	if (compress_mode==COMPRESS_VIDEO_RAM) {
		//must import in all formats
		//Android, GLES 2.x
		_save_stex(image,p_save_path+".etc.stex",compress_mode,lossy,Image::COMPRESS_ETC,mipmaps,tex_flags,stream);
		r_platform_variants->push_back("etc");
		//_save_stex(image,p_save_path+".etc2.stex",compress_mode,lossy,Image::COMPRESS_ETC2,mipmaps,tex_flags,stream);
		//r_platform_variants->push_back("etc2");
		_save_stex(image,p_save_path+".s3tc.stex",compress_mode,lossy,Image::COMPRESS_S3TC,mipmaps,tex_flags,stream);
		r_platform_variants->push_back("s3tc");

	} else {
		//import normally
		_save_stex(image,p_save_path+".stex",compress_mode,lossy,Image::COMPRESS_16BIT /*this is ignored */,mipmaps,tex_flags,stream);
	}

	return OK;
}

ResourceImporterTexture::ResourceImporterTexture()
{

}
