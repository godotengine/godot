#include "texture_bu.h"
#if 0
#include "core/os/os.h"

#ifdef TOOLS_ENABLED
#include "basisu_comp.h"
#endif

#include "transcoder/basisu.h"

void TextureBU::_bind_methods() {

	ClassDB::bind_method(D_METHOD("set_bu_data", "data"), &TextureBU::set_bu_data);
	ClassDB::bind_method(D_METHOD("get_bu_data"), &TextureBU::get_data);
	ClassDB::bind_method(D_METHOD("import"), &TextureBU::import);

	ADD_PROPERTY(PropertyInfo(Variant::POOL_BYTE_ARRAY, "bu_data"), "set_bu_data", "get_bu_data");

};

int TextureBU::get_width() const {

	return tex_size.x;
};

int TextureBU::get_height() const {

	return tex_size.y;
};

RID TextureBU::get_rid() const {

	return texture;
};


bool TextureBU::has_alpha() const {

	return false;
};

void TextureBU::set_flags(uint32_t p_flags) {

	flags = p_flags;
	VisualServer::get_singleton()->texture_set_flags(texture, p_flags);
};

uint32_t TextureBU::get_flags() const {

	return flags;
};


void TextureBU::set_bu_data(const PoolVector<uint8_t>& p_data) {

#ifdef TOOLS_ENABLED
	data = p_data;
#endif

	PoolVector<uint8_t>::Read r = p_data.read();
	const void* ptr = r.ptr();
	int size = p_data.size();

	basist::transcoder_texture_format format;
	Image::Format imgfmt;

	if (OS::get_singleton()->has_feature("s3tc")) {

		format = basist::cTFBC3; // get this from renderer
		imgfmt = Image::FORMAT_DXT5;

	} else if (OS::get_singleton()->has_feature("etc2")) {

		format = basist::cTFETC2;
		imgfmt = Image::FORMAT_ETC2_RGBA8;
	};

	basist::basisu_transcoder tr(NULL);

	ERR_FAIL_COND(!tr.validate_header(ptr, size));

	basist::basisu_image_info info;
	tr.get_image_info(ptr, size, info, 0);
	tex_size = Size2(info.m_width, info.m_height);

	int block_size = basist::basis_get_bytes_per_block(format);
	PoolVector<uint8_t> gpudata;
	gpudata.resize(info.m_total_blocks * block_size);

	{
		PoolVector<uint8_t>::Write w = gpudata.write();
		uint8_t* dst = w.ptr();
		for (int i=0; i<gpudata.size(); i++)
			dst[i] = 0x00;

		int ofs = 0;
		tr.start_transcoding(ptr, size);
		for (int i=0; i<info.m_total_levels; i++) {

			basist::basisu_image_level_info level;
			tr.get_image_level_info(ptr, size, level, 0, i);

			bool ret = tr.transcode_image_level(ptr, size, 0, i, dst + ofs, level.m_total_blocks - i, format);
			if (!ret) {
				printf("failed! on level %i\n", i);
				break;
			};

			ofs += level.m_total_blocks * block_size;
		};
	};

	Ref<Image> img;
	img.instance();
	img->create(info.m_width, info.m_height, info.m_total_levels > 1, imgfmt, gpudata);

	VisualServer::get_singleton()->texture_allocate(texture, tex_size.x, tex_size.y, 0, img->get_format(), VS::TEXTURE_TYPE_2D, flags);
	VisualServer::get_singleton()->texture_set_data(texture, img);
};

Error TextureBU::import(const Ref<Image>& p_img) {

#ifdef TOOLS_ENABLED

	PoolVector<uint8_t> budata;

	{
		Image::Format format = p_img->get_format();
		if (format != Image::FORMAT_RGB8 && format != Image::FORMAT_RGBA8) {
			ERR_FAIL_V(ERR_INVALID_PARAMETER);
			return ERR_INVALID_PARAMETER;
		};

		Ref<Image> copy = p_img->duplicate();
		if (format == Image::FORMAT_RGB8)
			copy->convert(Image::FORMAT_RGBA8);

		basisu::image buimg(p_img->get_width(), p_img->get_height());
		int size = p_img->get_width() * p_img->get_height() * 4;

		PoolVector<uint8_t> vec = copy->get_data();
		{
			PoolVector<uint8_t>::Read r = vec.read();
			memcpy(buimg.get_ptr(), r.ptr(), size);
		};

		basisu::basis_compressor_params params;
		params.m_max_endpoint_clusters = 512;
		params.m_max_selector_clusters = 512;
		params.m_multithreading = true;

		basisu::job_pool jpool(1);
		params.m_pJob_pool = &jpool;

		params.m_mip_gen = p_img->get_mipmap_count() > 0;
		params.m_source_images.push_back(buimg);

		basisu::basis_compressor c;
		c.init(params);

		int buerr = c.process();
		if (buerr != basisu::basis_compressor::cECSuccess) {
			ERR_FAIL_V(ERR_INVALID_PARAMETER);
			return ERR_INVALID_PARAMETER;
		};

		const basisu::uint8_vec& buvec = c.get_output_basis_file();
		budata.resize(buvec.size());

		{
			PoolVector<uint8_t>::Write w = budata.write();
			memcpy(w.ptr(), &buvec[0], budata.size());
		};
	};

	set_bu_data(budata);

	return OK;
#else

	return ERR_UNAVAILABLE;
#endif
};


PoolVector<uint8_t> TextureBU::get_bu_data() const {

	return data;
};

TextureBU::TextureBU() {

	flags = FLAGS_DEFAULT;
	texture = VisualServer::get_singleton()->texture_create();
};


TextureBU::~TextureBU() {

	VisualServer::get_singleton()->free(texture);
};

#endif
