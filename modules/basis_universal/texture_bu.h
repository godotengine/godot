#include "scene/resources/texture.h"

#ifdef TOOLS_ENABLED
#include "thirdparty/basis_universal/basisu_comp.h"
#endif

#include "thirdparty/basis_universal/transcoder/basisu.h"

#if 0
class TextureBU : public Texture {

	GDCLASS(TextureBU, Texture);
	RES_BASE_EXTENSION("butex");

	RID texture;
	Size2 tex_size;

	uint32_t flags;

	PoolVector<uint8_t> data;

	static void _bind_methods();

public:

	virtual int get_width() const;
	virtual int get_height() const;
	virtual RID get_rid() const;
	virtual bool has_alpha() const;

	virtual void set_flags(uint32_t p_flags);
	virtual uint32_t get_flags() const;


	Error import(const Ref<Image> &p_img);

	void set_bu_data(const PoolVector<uint8_t>& p_data);

	PoolVector<uint8_t> get_bu_data() const;
	String get_img_path() const;

	TextureBU();
	~TextureBU();

};

#endif
