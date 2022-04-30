/*************************************************************************/
/*  resource_saver_jpege.h                                               */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2022 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2022 Godot Engine contributors (cf. AUTHORS.md).   */
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

#ifndef RESOURCE_SAVER_JPG_H
#define RESOURCE_SAVER_JPG_H

#include "core/io/image.h"
#include "core/io/resource_saver.h"

#include <jpge.h>

class ResourceSaverJPG : public ResourceFormatSaver {
	GDCLASS(ResourceSaverJPG, ResourceFormatSaver);

public:
	enum SubSamplingFactor {
		SUBSAPLING_Y_ONLY = 0,
		SUBSAPLING_H1V1,
		SUBSAPLING_H2V1,
		SUBSAPLING_H2V2,
		SUBSAPLING_MAX,
	};

protected:
	static void _bind_methods();

private:
	Dictionary jpge_options;

	static Error _configure_jpge_parameters(const Dictionary &p_config, jpge::params &p_params);

public:
	virtual Error save(const String &p_path, const RES &p_resource, uint32_t p_flags = 0) override;
	virtual bool recognize(const RES &p_resource) const override;
	virtual void get_recognized_extensions(const RES &p_resource, List<String> *p_extensions) const override;

	Error save_image(const String &p_path, const Ref<Image> &p_image);
	Vector<uint8_t> save_jpg_to_buffer(Ref<Image> p_image);
	void set_encode_options(const Dictionary &p_options);

	ResourceSaverJPG() {}
};

VARIANT_ENUM_CAST(ResourceSaverJPG::SubSamplingFactor);

#endif // RESOURCE_SAVER_JPG_H
