/*************************************************************************/
/*  canvas_texture_storage.h                                             */
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

#ifndef CANVAS_TEXTURE_STORAGE_RD_H
#define CANVAS_TEXTURE_STORAGE_RD_H

#include "core/templates/rid_owner.h"
#include "servers/rendering/storage/canvas_texture_storage.h"

namespace RendererRD {

class CanvasTexture {
public:
	RID diffuse;
	RID normal_map;
	RID specular;
	Color specular_color = Color(1, 1, 1, 1);
	float shininess = 1.0;

	RS::CanvasItemTextureFilter texture_filter = RS::CANVAS_ITEM_TEXTURE_FILTER_DEFAULT;
	RS::CanvasItemTextureRepeat texture_repeat = RS::CANVAS_ITEM_TEXTURE_REPEAT_DEFAULT;
	RID uniform_sets[RS::CANVAS_ITEM_TEXTURE_FILTER_MAX][RS::CANVAS_ITEM_TEXTURE_REPEAT_MAX];

	Size2i size_cache = Size2i(1, 1);
	bool use_normal_cache = false;
	bool use_specular_cache = false;
	bool cleared_cache = true;

	void clear_sets();
	~CanvasTexture();
};

class CanvasTextureStorage : public RendererCanvasTextureStorage {
private:
	static CanvasTextureStorage *singleton;

	RID_Owner<RendererRD::CanvasTexture, true> canvas_texture_owner;

public:
	static CanvasTextureStorage *get_singleton();

	CanvasTextureStorage();
	virtual ~CanvasTextureStorage();

	CanvasTexture *get_canvas_texture(RID p_rid) { return canvas_texture_owner.get_or_null(p_rid); };
	bool owns_canvas_texture(RID p_rid) { return canvas_texture_owner.owns(p_rid); };

	virtual RID canvas_texture_allocate() override;
	virtual void canvas_texture_initialize(RID p_rid) override;
	virtual void canvas_texture_free(RID p_rid) override;

	virtual void canvas_texture_set_channel(RID p_canvas_texture, RS::CanvasTextureChannel p_channel, RID p_texture) override;
	virtual void canvas_texture_set_shading_parameters(RID p_canvas_texture, const Color &p_base_color, float p_shininess) override;

	virtual void canvas_texture_set_texture_filter(RID p_item, RS::CanvasItemTextureFilter p_filter) override;
	virtual void canvas_texture_set_texture_repeat(RID p_item, RS::CanvasItemTextureRepeat p_repeat) override;

	bool canvas_texture_get_uniform_set(RID p_texture, RS::CanvasItemTextureFilter p_base_filter, RS::CanvasItemTextureRepeat p_base_repeat, RID p_base_shader, int p_base_set, RID &r_uniform_set, Size2i &r_size, Color &r_specular_shininess, bool &r_use_normal, bool &r_use_specular);
};

} // namespace RendererRD

#endif // !CANVAS_TEXTURE_STORAGE_RD_H
