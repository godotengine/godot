/*************************************************************************/
/*  gltf_spec_gloss.h                                                    */
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

#ifndef GLTF_SPEC_GLOSS_H
#define GLTF_SPEC_GLOSS_H

#include "core/io/image.h"
#include "core/io/resource.h"

class GLTFSpecGloss : public Resource {
	GDCLASS(GLTFSpecGloss, Resource);
	friend class GLTFDocument;

private:
	Ref<Image> diffuse_img = nullptr;
	Color diffuse_factor = Color(1.0f, 1.0f, 1.0f);
	float gloss_factor = 1.0f;
	Color specular_factor = Color(1.0f, 1.0f, 1.0f);
	Ref<Image> spec_gloss_img = nullptr;

protected:
	static void _bind_methods();

public:
	Ref<Image> get_diffuse_img();
	void set_diffuse_img(Ref<Image> p_diffuse_img);

	Color get_diffuse_factor();
	void set_diffuse_factor(Color p_diffuse_factor);

	float get_gloss_factor();
	void set_gloss_factor(float p_gloss_factor);

	Color get_specular_factor();
	void set_specular_factor(Color p_specular_factor);

	Ref<Image> get_spec_gloss_img();
	void set_spec_gloss_img(Ref<Image> p_spec_gloss_img);
};
#endif // GLTF_SPEC_GLOSS_H
