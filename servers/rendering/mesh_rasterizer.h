/**************************************************************************/
/*  mesh_rasterizer.h                                                     */
/**************************************************************************/
/*                         This file is part of:                          */
/*                             GODOT ENGINE                               */
/*                        https://godotengine.org                         */
/**************************************************************************/
/* Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md). */
/* Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.                  */
/*                                                                        */
/* Permission is hereby granted, free of charge, to any person obtaining  */
/* a copy of this software and associated documentation files (the        */
/* "Software"), to deal in the Software without restriction, including    */
/* without limitation the rights to use, copy, modify, merge, publish,    */
/* distribute, sublicense, and/or sell copies of the Software, and to     */
/* permit persons to whom the Software is furnished to do so, subject to  */
/* the following conditions:                                              */
/*                                                                        */
/* The above copyright notice and this permission notice shall be         */
/* included in all copies or substantial portions of the Software.        */
/*                                                                        */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,        */
/* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF     */
/* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. */
/* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY   */
/* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,   */
/* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE      */
/* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.                 */
/**************************************************************************/

#pragma once

#include "servers/rendering_server.h"

class MeshRasterizer {
private:
	static MeshRasterizer *singleton;

public:
	virtual RID mesh_rasterizer_allocate() = 0;
	virtual void mesh_rasterizer_initialize(RID p_mesh_rasterizer, RID p_mesh, int p_surface_index) = 0;
	virtual void mesh_rasterizer_draw(RID p_mesh_rasterizer, RID p_material, RID p_texture_drawable, Ref<RasterizerBlendState> p_blend_state, const Color &p_clear_color, RD::TextureSamples p_multisample = RD::TEXTURE_SAMPLES_1) = 0;

	virtual bool free(RID p_mesh_rasterizer) = 0;

	static MeshRasterizer *get_singleton() { return singleton; }
	MeshRasterizer();
	virtual ~MeshRasterizer();
};

class RasterizerBlendState : public RefCounted {
	GDCLASS(RasterizerBlendState, RefCounted)

	Color blend_constant;
	bool enable_blend = false;
	RD::BlendFactor src_color_blend_factor = RD::BLEND_FACTOR_ZERO;
	RD::BlendFactor dst_color_blend_factor = RD::BLEND_FACTOR_ZERO;
	RD::BlendOperation color_blend_op = RD::BLEND_OP_ADD;
	RD::BlendFactor src_alpha_blend_factor = RD::BLEND_FACTOR_ZERO;
	RD::BlendFactor dst_alpha_blend_factor = RD::BLEND_FACTOR_ZERO;
	RD::BlendOperation alpha_blend_op = RD::BLEND_OP_ADD;
	bool write_r = true;
	bool write_g = true;
	bool write_b = true;
	bool write_a = true;

protected:
	static void _bind_methods() {
#define RASTERIZER_BIND(m_variant_type, m_class, m_member)                                                  \
	ClassDB::bind_method(D_METHOD("set_" _MKSTR(m_member), "p_" _MKSTR(member)), &m_class::set_##m_member); \
	ClassDB::bind_method(D_METHOD("get_" _MKSTR(m_member)), &m_class::get_##m_member);                      \
	ADD_PROPERTY(PropertyInfo(m_variant_type, #m_member), "set_" _MKSTR(m_member), "get_" _MKSTR(m_member))

		RASTERIZER_BIND(Variant::BOOL, RasterizerBlendState, enable_blend);
		RASTERIZER_BIND(Variant::COLOR, RasterizerBlendState, blend_constant);

		RASTERIZER_BIND(Variant::INT, RasterizerBlendState, src_color_blend_factor);
		RASTERIZER_BIND(Variant::INT, RasterizerBlendState, dst_color_blend_factor);
		RASTERIZER_BIND(Variant::INT, RasterizerBlendState, color_blend_op);

		RASTERIZER_BIND(Variant::INT, RasterizerBlendState, src_alpha_blend_factor);
		RASTERIZER_BIND(Variant::INT, RasterizerBlendState, dst_alpha_blend_factor);
		RASTERIZER_BIND(Variant::INT, RasterizerBlendState, alpha_blend_op);

		RASTERIZER_BIND(Variant::BOOL, RasterizerBlendState, write_r);
		RASTERIZER_BIND(Variant::BOOL, RasterizerBlendState, write_g);
		RASTERIZER_BIND(Variant::BOOL, RasterizerBlendState, write_b);
		RASTERIZER_BIND(Variant::BOOL, RasterizerBlendState, write_a);

#undef RASTERIZER_BIND
	}

public:
	bool equal(const Ref<RasterizerBlendState> &b) {
		if (b.is_null()) {
			return false;
		}
		bool eq = true;
		eq = eq && get_blend_constant() == b->get_blend_constant();
		eq = eq && get_enable_blend() == b->get_enable_blend();
		eq = eq && get_src_color_blend_factor() == b->get_src_color_blend_factor();
		eq = eq && get_dst_color_blend_factor() == b->get_dst_color_blend_factor();
		eq = eq && get_color_blend_op() == b->get_color_blend_op();
		eq = eq && get_src_alpha_blend_factor() == b->get_src_alpha_blend_factor();
		eq = eq && get_dst_alpha_blend_factor() == b->get_dst_alpha_blend_factor();
		eq = eq && get_alpha_blend_op() == b->get_alpha_blend_op();
		eq = eq && get_write_r() == b->get_write_r();
		eq = eq && get_write_g() == b->get_write_g();
		eq = eq && get_write_b() == b->get_write_b();
		eq = eq && get_write_a() == b->get_write_a();
		return eq;
	}
	void get_rd_blend_state(RD::PipelineColorBlendState &r_blend_state) {
		r_blend_state.blend_constant = blend_constant;
		r_blend_state.attachments.write[0].enable_blend = enable_blend;
		r_blend_state.attachments.write[0].src_color_blend_factor = src_color_blend_factor;
		r_blend_state.attachments.write[0].dst_color_blend_factor = dst_color_blend_factor;
		r_blend_state.attachments.write[0].color_blend_op = color_blend_op;
		r_blend_state.attachments.write[0].src_alpha_blend_factor = src_alpha_blend_factor;
		r_blend_state.attachments.write[0].dst_alpha_blend_factor = dst_alpha_blend_factor;
		r_blend_state.attachments.write[0].alpha_blend_op = alpha_blend_op;
		r_blend_state.attachments.write[0].write_r = write_r;
		r_blend_state.attachments.write[0].write_g = write_g;
		r_blend_state.attachments.write[0].write_b = write_b;
		r_blend_state.attachments.write[0].write_a = write_a;
	}

#define RASTERIZER_SETGET(m_type, m_member)    \
	void set_##m_member(m_type p_##m_member) { \
		m_member = p_##m_member;               \
	}                                          \
	m_type get_##m_member() const {            \
		return m_member;                       \
	}

	RASTERIZER_SETGET(bool, enable_blend);
	RASTERIZER_SETGET(Color, blend_constant);

	RASTERIZER_SETGET(RD::BlendFactor, src_color_blend_factor);
	RASTERIZER_SETGET(RD::BlendFactor, dst_color_blend_factor);
	RASTERIZER_SETGET(RD::BlendOperation, color_blend_op);

	RASTERIZER_SETGET(RD::BlendFactor, src_alpha_blend_factor);
	RASTERIZER_SETGET(RD::BlendFactor, dst_alpha_blend_factor);
	RASTERIZER_SETGET(RD::BlendOperation, alpha_blend_op);

	RASTERIZER_SETGET(bool, write_r);
	RASTERIZER_SETGET(bool, write_g);
	RASTERIZER_SETGET(bool, write_b);
	RASTERIZER_SETGET(bool, write_a);

#undef RASTERIZER_SETGET
};
