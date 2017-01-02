#ifndef SKYBOX_H
#define SKYBOX_H

#include "scene/resources/texture.h"

class SkyBox : public Resource {
	OBJ_TYPE(SkyBox,Resource);

public:

	enum RadianceSize {
		RADIANCE_SIZE_256,
		RADIANCE_SIZE_512,
		RADIANCE_SIZE_1024,
		RADIANCE_SIZE_2048,
		RADIANCE_SIZE_MAX
	};
private:

	RadianceSize radiance_size;
protected:
	static void _bind_methods();
	virtual void _radiance_changed()=0;
public:

	void set_radiance_size(RadianceSize p_size);
	RadianceSize get_radiance_size() const;
	SkyBox();
};

VARIANT_ENUM_CAST(SkyBox::RadianceSize)


class ImageSkyBox : public SkyBox {
	OBJ_TYPE(ImageSkyBox,SkyBox);

public:

	enum ImagePath {
		IMAGE_PATH_NEGATIVE_X,
		IMAGE_PATH_POSITIVE_X,
		IMAGE_PATH_NEGATIVE_Y,
		IMAGE_PATH_POSITIVE_Y,
		IMAGE_PATH_NEGATIVE_Z,
		IMAGE_PATH_POSITIVE_Z,
		IMAGE_PATH_MAX
	};
private:
	RID cube_map;
	RID sky_box;
	bool cube_map_valid;

	String image_path[IMAGE_PATH_MAX];
protected:
	static void _bind_methods();
	virtual void _radiance_changed();
public:

	void set_image_path(ImagePath p_image, const String &p_path);
	String get_image_path(ImagePath p_image) const;

	virtual RID get_rid() const;

	ImageSkyBox();
	~ImageSkyBox();
};

VARIANT_ENUM_CAST(ImageSkyBox::ImagePath)


#endif // SKYBOX_H
