/*************************************************************************/
/*  visual_server_viewport.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                    http://www.godotengine.org                         */
/*************************************************************************/
/* Copyright (c) 2007-2017 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2017 Godot Engine contributors (cf. AUTHORS.md)    */
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
#include "visual_server_viewport.h"
#include "global_config.h"
#include "visual_server_canvas.h"
#include "visual_server_global.h"
#include "visual_server_scene.h"

void VisualServerViewport::_draw_viewport(Viewport *p_viewport) {

/* Camera should always be BEFORE any other 3D */
#if 0
	bool scenario_draw_canvas_bg=false;
	int scenario_canvas_max_layer=0;

	if (!p_viewport->hide_canvas && !p_viewport->disable_environment && scenario_owner.owns(p_viewport->scenario)) {

		Scenario *scenario=scenario_owner.get(p_viewport->scenario);
		if (scenario->environment.is_valid()) {
			if (rasterizer->is_environment(scenario->environment)) {
				scenario_draw_canvas_bg=rasterizer->environment_get_background(scenario->environment)==VS::ENV_BG_CANVAS;
				scenario_canvas_max_layer=rasterizer->environment_get_background_param(scenario->environment,VS::ENV_BG_PARAM_CANVAS_MAX_LAYER);
			}
		}
	}

	bool can_draw_3d=!p_viewport->hide_scenario && camera_owner.owns(p_viewport->camera) && scenario_owner.owns(p_viewport->scenario);


	if (scenario_draw_canvas_bg) {

		rasterizer->begin_canvas_bg();
	}

	if (!scenario_draw_canvas_bg && can_draw_3d) {

		_draw_viewport_camera(p_viewport,false);

	} else if (true /*|| !p_viewport->canvas_list.empty()*/){

		//clear the viewport black because of no camera? i seriously should..
		if (p_viewport->render_target_clear_on_new_frame || p_viewport->render_target_clear) {
			if (p_viewport->transparent_bg) {
				rasterizer->clear_viewport(Color(0,0,0,0));
			}
			else {
				Color cc=clear_color;
				if (scenario_draw_canvas_bg)
					cc.a=0;
				rasterizer->clear_viewport(cc);
			}
			p_viewport->render_target_clear=false;
		}
	}
#endif

	if (p_viewport->clear_mode != VS::VIEWPORT_CLEAR_NEVER) {
		VSG::rasterizer->clear_render_target(clear_color);
		if (p_viewport->clear_mode == VS::VIEWPORT_CLEAR_ONLY_NEXT_FRAME) {
			p_viewport->clear_mode = VS::VIEWPORT_CLEAR_NEVER;
		}
	}

	if (!p_viewport->disable_3d && p_viewport->camera.is_valid()) {

		VSG::scene->render_camera(p_viewport->camera, p_viewport->scenario, p_viewport->size, p_viewport->shadow_atlas);
	}

	if (!p_viewport->hide_canvas) {
		int i = 0;

		Map<Viewport::CanvasKey, Viewport::CanvasData *> canvas_map;

		Rect2 clip_rect(0, 0, p_viewport->size.x, p_viewport->size.y);
		RasterizerCanvas::Light *lights = NULL;
		RasterizerCanvas::Light *lights_with_shadow = NULL;
		RasterizerCanvas::Light *lights_with_mask = NULL;
		Rect2 shadow_rect;

		int light_count = 0;

		for (Map<RID, Viewport::CanvasData>::Element *E = p_viewport->canvas_map.front(); E; E = E->next()) {

			Transform2D xf = p_viewport->global_transform * E->get().transform;

			VisualServerCanvas::Canvas *canvas = static_cast<VisualServerCanvas::Canvas *>(E->get().canvas);

			//find lights in canvas

			for (Set<RasterizerCanvas::Light *>::Element *F = canvas->lights.front(); F; F = F->next()) {

				RasterizerCanvas::Light *cl = F->get();
				if (cl->enabled && cl->texture.is_valid()) {
					//not super efficient..
					Size2 tsize(VSG::storage->texture_get_width(cl->texture), VSG::storage->texture_get_height(cl->texture));
					tsize *= cl->scale;

					Vector2 offset = tsize / 2.0;
					cl->rect_cache = Rect2(-offset + cl->texture_offset, tsize);
					cl->xform_cache = xf * cl->xform;

					if (clip_rect.intersects_transformed(cl->xform_cache, cl->rect_cache)) {

						cl->filter_next_ptr = lights;
						lights = cl;
						cl->texture_cache = NULL;
						Transform2D scale;
						scale.scale(cl->rect_cache.size);
						scale.elements[2] = cl->rect_cache.pos;
						cl->light_shader_xform = (cl->xform_cache * scale).affine_inverse();
						cl->light_shader_pos = cl->xform_cache[2];
						if (cl->shadow_buffer.is_valid()) {

							cl->shadows_next_ptr = lights_with_shadow;
							if (lights_with_shadow == NULL) {
								shadow_rect = cl->xform_cache.xform(cl->rect_cache);
							} else {
								shadow_rect = shadow_rect.merge(cl->xform_cache.xform(cl->rect_cache));
							}
							lights_with_shadow = cl;
							cl->radius_cache = cl->rect_cache.size.length();
						}
						if (cl->mode == VS::CANVAS_LIGHT_MODE_MASK) {
							cl->mask_next_ptr = lights_with_mask;
							lights_with_mask = cl;
						}

						light_count++;
					}

					VSG::canvas_render->light_internal_update(cl->light_internal, cl);
				}
			}

			//print_line("lights: "+itos(light_count));
			canvas_map[Viewport::CanvasKey(E->key(), E->get().layer)] = &E->get();
		}

		if (lights_with_shadow) {
			//update shadows if any

			RasterizerCanvas::LightOccluderInstance *occluders = NULL;

			//make list of occluders
			for (Map<RID, Viewport::CanvasData>::Element *E = p_viewport->canvas_map.front(); E; E = E->next()) {

				VisualServerCanvas::Canvas *canvas = static_cast<VisualServerCanvas::Canvas *>(E->get().canvas);
				Transform2D xf = p_viewport->global_transform * E->get().transform;

				for (Set<RasterizerCanvas::LightOccluderInstance *>::Element *F = canvas->occluders.front(); F; F = F->next()) {

					if (!F->get()->enabled)
						continue;
					F->get()->xform_cache = xf * F->get()->xform;
					if (shadow_rect.intersects_transformed(F->get()->xform_cache, F->get()->aabb_cache)) {

						F->get()->next = occluders;
						occluders = F->get();
					}
				}
			}
			//update the light shadowmaps with them
			RasterizerCanvas::Light *light = lights_with_shadow;
			while (light) {

				VSG::canvas_render->canvas_light_shadow_buffer_update(light->shadow_buffer, light->xform_cache.affine_inverse(), light->item_mask, light->radius_cache / 1000.0, light->radius_cache * 1.1, occluders, &light->shadow_matrix_cache);
				light = light->shadows_next_ptr;
			}

			//VSG::canvas_render->reset_canvas();
		}

		VSG::rasterizer->restore_render_target();

#if 0
		if (scenario_draw_canvas_bg && canvas_map.front() && canvas_map.front()->key().layer>scenario_canvas_max_layer) {

			_draw_viewport_camera(p_viewport,!can_draw_3d);
			scenario_draw_canvas_bg=false;

		}
#endif

		for (Map<Viewport::CanvasKey, Viewport::CanvasData *>::Element *E = canvas_map.front(); E; E = E->next()) {

			VisualServerCanvas::Canvas *canvas = static_cast<VisualServerCanvas::Canvas *>(E->get()->canvas);

			//print_line("canvas "+itos(i)+" size: "+itos(I->get()->canvas->child_items.size()));
			//print_line("GT "+p_viewport->global_transform+". CT: "+E->get()->transform);
			Transform2D xform = p_viewport->global_transform * E->get()->transform;

			RasterizerCanvas::Light *canvas_lights = NULL;

			RasterizerCanvas::Light *ptr = lights;
			while (ptr) {
				if (E->get()->layer >= ptr->layer_min && E->get()->layer <= ptr->layer_max) {
					ptr->next_ptr = canvas_lights;
					canvas_lights = ptr;
				}
				ptr = ptr->filter_next_ptr;
			}

			VSG::canvas->render_canvas(canvas, xform, canvas_lights, lights_with_mask, clip_rect);
			i++;
#if 0
			if (scenario_draw_canvas_bg && E->key().layer>=scenario_canvas_max_layer) {
				_draw_viewport_camera(p_viewport,!can_draw_3d);
				scenario_draw_canvas_bg=false;
			}
#endif
		}
#if 0
		if (scenario_draw_canvas_bg) {
			_draw_viewport_camera(p_viewport,!can_draw_3d);
			scenario_draw_canvas_bg=false;
		}
#endif

		//VSG::canvas_render->canvas_debug_viewport_shadows(lights_with_shadow);
	}
}

void VisualServerViewport::draw_viewports() {

	//sort viewports

	//draw viewports

	clear_color = GLOBAL_GET("rendering/viewport/default_clear_color");

	active_viewports.sort_custom<ViewportSort>();

	for (int i = 0; i < active_viewports.size(); i++) {

		Viewport *vp = active_viewports[i];

		if (vp->update_mode == VS::VIEWPORT_UPDATE_DISABLED)
			continue;

		ERR_CONTINUE(!vp->render_target.is_valid());

		bool visible = vp->viewport_to_screen_rect != Rect2() || vp->update_mode == VS::VIEWPORT_UPDATE_ALWAYS || vp->update_mode == VS::VIEWPORT_UPDATE_ONCE;

		if (!visible)
			continue;

		VSG::rasterizer->set_current_render_target(vp->render_target);
		_draw_viewport(vp);

		if (vp->viewport_to_screen_rect != Rect2()) {
			//copy to screen if set as such
			VSG::rasterizer->set_current_render_target(RID());
			VSG::rasterizer->blit_render_target_to_screen(vp->render_target, vp->viewport_to_screen_rect, vp->viewport_to_screen);
		}

		if (vp->update_mode == VS::VIEWPORT_UPDATE_ONCE) {
			vp->update_mode = VS::VIEWPORT_UPDATE_DISABLED;
		}
	}
}

RID VisualServerViewport::viewport_create() {

	Viewport *viewport = memnew(Viewport);

	RID rid = viewport_owner.make_rid(viewport);

	viewport->self = rid;
	viewport->hide_scenario = false;
	viewport->hide_canvas = false;
	viewport->render_target = VSG::storage->render_target_create();
	viewport->shadow_atlas = VSG::scene_render->shadow_atlas_create();

	return rid;
}

void VisualServerViewport::viewport_set_size(RID p_viewport, int p_width, int p_height) {

	ERR_FAIL_COND(p_width < 0 && p_height < 0);

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	viewport->size = Size2(p_width, p_height);
	VSG::storage->render_target_set_size(viewport->render_target, p_width, p_height);
}

void VisualServerViewport::viewport_set_active(RID p_viewport, bool p_active) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	if (p_active) {
		ERR_FAIL_COND(active_viewports.find(viewport) != -1); //already active
		active_viewports.push_back(viewport);
	} else {
		active_viewports.erase(viewport);
	}
}

void VisualServerViewport::viewport_set_parent_viewport(RID p_viewport, RID p_parent_viewport) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	viewport->parent = p_parent_viewport;
}

void VisualServerViewport::viewport_set_clear_mode(RID p_viewport, VS::ViewportClearMode p_clear_mode) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	viewport->clear_mode = p_clear_mode;
}

void VisualServerViewport::viewport_attach_to_screen(RID p_viewport, const Rect2 &p_rect, int p_screen) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	viewport->viewport_to_screen_rect = p_rect;
	viewport->viewport_to_screen = p_screen;
}
void VisualServerViewport::viewport_detach(RID p_viewport) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	viewport->viewport_to_screen_rect = Rect2();
	viewport->viewport_to_screen = 0;
}

void VisualServerViewport::viewport_set_update_mode(RID p_viewport, VS::ViewportUpdateMode p_mode) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	viewport->update_mode = p_mode;
}
void VisualServerViewport::viewport_set_vflip(RID p_viewport, bool p_enable) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	VSG::storage->render_target_set_flag(viewport->render_target, RasterizerStorage::RENDER_TARGET_VFLIP, p_enable);
}

RID VisualServerViewport::viewport_get_texture(RID p_viewport) const {

	const Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND_V(!viewport, RID());

	return VSG::storage->render_target_get_texture(viewport->render_target);
}

void VisualServerViewport::viewport_set_hide_scenario(RID p_viewport, bool p_hide) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	viewport->hide_scenario = p_hide;
}
void VisualServerViewport::viewport_set_hide_canvas(RID p_viewport, bool p_hide) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	viewport->hide_canvas = p_hide;
}
void VisualServerViewport::viewport_set_disable_environment(RID p_viewport, bool p_disable) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	viewport->disable_environment = p_disable;
}

void VisualServerViewport::viewport_set_disable_3d(RID p_viewport, bool p_disable) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	viewport->disable_3d = p_disable;
	VSG::storage->render_target_set_flag(viewport->render_target, RasterizerStorage::RENDER_TARGET_NO_3D, p_disable);
}

void VisualServerViewport::viewport_attach_camera(RID p_viewport, RID p_camera) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	viewport->camera = p_camera;
}
void VisualServerViewport::viewport_set_scenario(RID p_viewport, RID p_scenario) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	viewport->scenario = p_scenario;
}
void VisualServerViewport::viewport_attach_canvas(RID p_viewport, RID p_canvas) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	ERR_FAIL_COND(viewport->canvas_map.has(p_canvas));
	VisualServerCanvas::Canvas *canvas = VSG::canvas->canvas_owner.getornull(p_canvas);
	ERR_FAIL_COND(!canvas);

	canvas->viewports.insert(p_viewport);
	viewport->canvas_map[p_canvas] = Viewport::CanvasData();
	viewport->canvas_map[p_canvas].layer = 0;
	viewport->canvas_map[p_canvas].canvas = canvas;
}

void VisualServerViewport::viewport_remove_canvas(RID p_viewport, RID p_canvas) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	VisualServerCanvas::Canvas *canvas = VSG::canvas->canvas_owner.getornull(p_canvas);
	ERR_FAIL_COND(!canvas);

	viewport->canvas_map.erase(p_canvas);
	canvas->viewports.erase(p_viewport);
}
void VisualServerViewport::viewport_set_canvas_transform(RID p_viewport, RID p_canvas, const Transform2D &p_offset) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	ERR_FAIL_COND(!viewport->canvas_map.has(p_canvas));
	viewport->canvas_map[p_canvas].transform = p_offset;
}
void VisualServerViewport::viewport_set_transparent_background(RID p_viewport, bool p_enabled) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	VSG::storage->render_target_set_flag(viewport->render_target, RasterizerStorage::RENDER_TARGET_TRANSPARENT, p_enabled);
}

void VisualServerViewport::viewport_set_global_canvas_transform(RID p_viewport, const Transform2D &p_transform) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	viewport->global_transform = p_transform;
}
void VisualServerViewport::viewport_set_canvas_layer(RID p_viewport, RID p_canvas, int p_layer) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	ERR_FAIL_COND(!viewport->canvas_map.has(p_canvas));
	viewport->canvas_map[p_canvas].layer = p_layer;
}

void VisualServerViewport::viewport_set_shadow_atlas_size(RID p_viewport, int p_size) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	viewport->shadow_atlas_size = p_size;

	VSG::scene_render->shadow_atlas_set_size(viewport->shadow_atlas, viewport->shadow_atlas_size);
}

void VisualServerViewport::viewport_set_shadow_atlas_quadrant_subdivision(RID p_viewport, int p_quadrant, int p_subdiv) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	VSG::scene_render->shadow_atlas_set_quadrant_subdivision(viewport->shadow_atlas, p_quadrant, p_subdiv);
}

void VisualServerViewport::viewport_set_msaa(RID p_viewport, VS::ViewportMSAA p_msaa) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	VSG::storage->render_target_set_msaa(viewport->render_target, p_msaa);
}

void VisualServerViewport::viewport_set_hdr(RID p_viewport, bool p_enabled) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	VSG::storage->render_target_set_flag(viewport->render_target, RasterizerStorage::RENDER_TARGET_HDR, p_enabled);
}

bool VisualServerViewport::free(RID p_rid) {

	if (viewport_owner.owns(p_rid)) {

		Viewport *viewport = viewport_owner.getornull(p_rid);

		VSG::storage->free(viewport->render_target);
		VSG::scene_render->free(viewport->shadow_atlas);

		while (viewport->canvas_map.front()) {
			viewport_remove_canvas(p_rid, viewport->canvas_map.front()->key());
		}

		viewport_set_scenario(p_rid, RID());
		active_viewports.erase(viewport);

		viewport_owner.free(p_rid);
		memdelete(viewport);

		return true;
	}

	return false;
}

VisualServerViewport::VisualServerViewport() {
}
