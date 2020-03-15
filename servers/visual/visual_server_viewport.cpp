/*************************************************************************/
/*  visual_server_viewport.cpp                                           */
/*************************************************************************/
/*                       This file is part of:                           */
/*                           GODOT ENGINE                                */
/*                      https://godotengine.org                          */
/*************************************************************************/
/* Copyright (c) 2007-2020 Juan Linietsky, Ariel Manzur.                 */
/* Copyright (c) 2014-2020 Godot Engine contributors (cf. AUTHORS.md).   */
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

#include "core/project_settings.h"
#include "visual_server_canvas.h"
#include "visual_server_globals.h"
#include "visual_server_scene.h"

static Transform2D _canvas_get_transform(VisualServerViewport::Viewport *p_viewport, VisualServerCanvas::Canvas *p_canvas, VisualServerViewport::Viewport::CanvasData *p_canvas_data, const Vector2 &p_vp_size) {

	Transform2D xf = p_viewport->global_transform;

	float scale = 1.0;
	if (p_viewport->canvas_map.has(p_canvas->parent)) {
		xf = xf * p_viewport->canvas_map[p_canvas->parent].transform;
		scale = p_canvas->parent_scale;
	}

	xf = xf * p_canvas_data->transform;

	if (scale != 1.0 && !VSG::canvas->disable_scale) {
		Vector2 pivot = p_vp_size * 0.5;
		Transform2D xfpivot;
		xfpivot.set_origin(pivot);
		Transform2D xfscale;
		xfscale.scale(Vector2(scale, scale));

		xf = xfpivot.affine_inverse() * xf;
		xf = xfscale * xf;
		xf = xfpivot * xf;
	}

	return xf;
}

void VisualServerViewport::_draw_3d(Viewport *p_viewport, ARVRInterface::Eyes p_eye) {

	RENDER_TIMESTAMP(">Begin Rendering 3D Scene");

	Ref<ARVRInterface> arvr_interface;
	if (ARVRServer::get_singleton() != NULL) {
		arvr_interface = ARVRServer::get_singleton()->get_primary_interface();
	}

	if (p_viewport->use_arvr && arvr_interface.is_valid()) {
		VSG::scene->render_camera(p_viewport->render_buffers, arvr_interface, p_eye, p_viewport->camera, p_viewport->scenario, p_viewport->size, p_viewport->shadow_atlas);
	} else {
		VSG::scene->render_camera(p_viewport->render_buffers, p_viewport->camera, p_viewport->scenario, p_viewport->size, p_viewport->shadow_atlas);
	}
	RENDER_TIMESTAMP("<End Rendering 3D Scene");
}

void VisualServerViewport::_draw_viewport(Viewport *p_viewport, ARVRInterface::Eyes p_eye) {

	/* Camera should always be BEFORE any other 3D */

	bool scenario_draw_canvas_bg = false; //draw canvas, or some layer of it, as BG for 3D instead of in front
	int scenario_canvas_max_layer = 0;

	Color bgcolor = VSG::storage->get_default_clear_color();

	if (!p_viewport->hide_canvas && !p_viewport->disable_environment && VSG::scene->scenario_owner.owns(p_viewport->scenario)) {

		VisualServerScene::Scenario *scenario = VSG::scene->scenario_owner.getornull(p_viewport->scenario);
		ERR_FAIL_COND(!scenario);
		if (VSG::scene_render->is_environment(scenario->environment)) {
			scenario_draw_canvas_bg = VSG::scene_render->environment_get_background(scenario->environment) == VS::ENV_BG_CANVAS;

			scenario_canvas_max_layer = VSG::scene_render->environment_get_canvas_max_layer(scenario->environment);
		}
	}

	bool can_draw_3d = VSG::scene->camera_owner.owns(p_viewport->camera);

	if (p_viewport->clear_mode != VS::VIEWPORT_CLEAR_NEVER) {
		if (p_viewport->transparent_bg) {
			bgcolor = Color(0, 0, 0, 0);
		}
		if (p_viewport->clear_mode == VS::VIEWPORT_CLEAR_ONLY_NEXT_FRAME) {
			p_viewport->clear_mode = VS::VIEWPORT_CLEAR_NEVER;
		}
	}

	if ((scenario_draw_canvas_bg || can_draw_3d) && !p_viewport->render_buffers.is_valid()) {
		//wants to draw 3D but there is no render buffer, create
		p_viewport->render_buffers = VSG::scene_render->render_buffers_create();
		VSG::scene_render->render_buffers_configure(p_viewport->render_buffers, p_viewport->render_target, p_viewport->size.width, p_viewport->size.height, p_viewport->msaa);
	}

	VSG::storage->render_target_request_clear(p_viewport->render_target, bgcolor);

	if (!scenario_draw_canvas_bg && can_draw_3d) {
		_draw_3d(p_viewport, p_eye);
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

		RENDER_TIMESTAMP("Cull Canvas Lights");
		for (Map<RID, Viewport::CanvasData>::Element *E = p_viewport->canvas_map.front(); E; E = E->next()) {

			VisualServerCanvas::Canvas *canvas = static_cast<VisualServerCanvas::Canvas *>(E->get().canvas);

			Transform2D xf = _canvas_get_transform(p_viewport, canvas, &E->get(), clip_rect.size);

			//find lights in canvas

			for (Set<RasterizerCanvas::Light *>::Element *F = canvas->lights.front(); F; F = F->next()) {

				RasterizerCanvas::Light *cl = F->get();
				if (cl->enabled && cl->texture.is_valid()) {
					//not super efficient..
					Size2 tsize = VSG::storage->texture_size_with_proxy(cl->texture);
					tsize *= cl->scale;

					Vector2 offset = tsize / 2.0;
					cl->rect_cache = Rect2(-offset + cl->texture_offset, tsize);
					cl->xform_cache = xf * cl->xform;

					if (clip_rect.intersects_transformed(cl->xform_cache, cl->rect_cache)) {

						cl->filter_next_ptr = lights;
						lights = cl;
						//						cl->texture_cache = NULL;
						Transform2D scale;
						scale.scale(cl->rect_cache.size);
						scale.elements[2] = cl->rect_cache.position;
						cl->light_shader_xform = cl->xform * scale;
						//cl->light_shader_pos = cl->xform_cache[2];
						if (cl->use_shadow) {

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

					//guess this is not needed, but keeping because it may be
					//VSG::canvas_render->light_internal_update(cl->light_internal, cl);
				}
			}

			canvas_map[Viewport::CanvasKey(E->key(), E->get().layer, E->get().sublayer)] = &E->get();
		}

		if (lights_with_shadow) {
			//update shadows if any

			RasterizerCanvas::LightOccluderInstance *occluders = NULL;

			RENDER_TIMESTAMP(">Render 2D Shadows");
			RENDER_TIMESTAMP("Cull Occluders");

			//make list of occluders
			for (Map<RID, Viewport::CanvasData>::Element *E = p_viewport->canvas_map.front(); E; E = E->next()) {

				VisualServerCanvas::Canvas *canvas = static_cast<VisualServerCanvas::Canvas *>(E->get().canvas);
				Transform2D xf = _canvas_get_transform(p_viewport, canvas, &E->get(), clip_rect.size);

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

				RENDER_TIMESTAMP("Render Shadow");

				VSG::canvas_render->light_update_shadow(light->light_internal, light->xform_cache.affine_inverse(), light->item_shadow_mask, light->radius_cache / 1000.0, light->radius_cache * 1.1, occluders);
				light = light->shadows_next_ptr;
			}

			//VSG::canvas_render->reset_canvas();
			RENDER_TIMESTAMP("<End rendering 2D Shadows");
		}

		if (scenario_draw_canvas_bg && canvas_map.front() && canvas_map.front()->key().get_layer() > scenario_canvas_max_layer) {
			if (!can_draw_3d) {
				VSG::scene->render_empty_scene(p_viewport->render_buffers, p_viewport->scenario, p_viewport->shadow_atlas);
			} else {
				_draw_3d(p_viewport, p_eye);
			}
			scenario_draw_canvas_bg = false;
		}

		for (Map<Viewport::CanvasKey, Viewport::CanvasData *>::Element *E = canvas_map.front(); E; E = E->next()) {

			VisualServerCanvas::Canvas *canvas = static_cast<VisualServerCanvas::Canvas *>(E->get()->canvas);

			Transform2D xform = _canvas_get_transform(p_viewport, canvas, E->get(), clip_rect.size);

			RasterizerCanvas::Light *canvas_lights = NULL;

			RasterizerCanvas::Light *ptr = lights;
			while (ptr) {
				if (E->get()->layer >= ptr->layer_min && E->get()->layer <= ptr->layer_max) {
					ptr->next_ptr = canvas_lights;
					canvas_lights = ptr;
				}
				ptr = ptr->filter_next_ptr;
			}

			VSG::canvas->render_canvas(p_viewport->render_target, canvas, xform, canvas_lights, lights_with_mask, clip_rect);
			i++;

			if (scenario_draw_canvas_bg && E->key().get_layer() >= scenario_canvas_max_layer) {
				if (!can_draw_3d) {
					VSG::scene->render_empty_scene(p_viewport->render_buffers, p_viewport->scenario, p_viewport->shadow_atlas);
				} else {
					_draw_3d(p_viewport, p_eye);
				}

				scenario_draw_canvas_bg = false;
			}
		}

		if (scenario_draw_canvas_bg) {
			if (!can_draw_3d) {
				VSG::scene->render_empty_scene(p_viewport->render_buffers, p_viewport->scenario, p_viewport->shadow_atlas);
			} else {
				_draw_3d(p_viewport, p_eye);
			}
		}

		//VSG::canvas_render->canvas_debug_viewport_shadows(lights_with_shadow);
	}

	if (VSG::storage->render_target_is_clear_requested(p_viewport->render_target)) {
		//was never cleared in the end, force clear it
		VSG::storage->render_target_do_clear_request(p_viewport->render_target);
	}
}

void VisualServerViewport::draw_viewports() {

#if 0
	// get our arvr interface in case we need it
	Ref<ARVRInterface> arvr_interface;

	if (ARVRServer::get_singleton() != NULL) {
		arvr_interface = ARVRServer::get_singleton()->get_primary_interface();

		// process all our active interfaces
		ARVRServer::get_singleton()->_process();
	}
#endif

	if (Engine::get_singleton()->is_editor_hint()) {
		set_default_clear_color(GLOBAL_GET("rendering/environment/default_clear_color"));
	}

	//sort viewports
	active_viewports.sort_custom<ViewportSort>();

	Map<int, Vector<Rasterizer::BlitToScreen> > blit_to_screen_list;
	//draw viewports
	RENDER_TIMESTAMP(">Render Viewports");

	for (int i = 0; i < active_viewports.size(); i++) {

		Viewport *vp = active_viewports[i];

		if (vp->update_mode == VS::VIEWPORT_UPDATE_DISABLED)
			continue;

		if (!vp->render_target.is_valid()) {
			continue;
		}
		//ERR_CONTINUE(!vp->render_target.is_valid());

		bool visible = vp->viewport_to_screen_rect != Rect2() || vp->update_mode == VS::VIEWPORT_UPDATE_ALWAYS || vp->update_mode == VS::VIEWPORT_UPDATE_ONCE || (vp->update_mode == VS::VIEWPORT_UPDATE_WHEN_VISIBLE && VSG::storage->render_target_was_used(vp->render_target));
		visible = visible && vp->size.x > 1 && vp->size.y > 1;

		if (!visible)
			continue;

		RENDER_TIMESTAMP(">Rendering Viewport " + itos(i));

		VSG::storage->render_target_set_as_unused(vp->render_target);
#if 0
		if (vp->use_arvr && arvr_interface.is_valid()) {
			// override our size, make sure it matches our required size
			vp->size = arvr_interface->get_render_targetsize();
			VSG::storage->render_target_set_size(vp->render_target, vp->size.x, vp->size.y);

			// render mono or left eye first
			ARVRInterface::Eyes leftOrMono = arvr_interface->is_stereo() ? ARVRInterface::EYE_LEFT : ARVRInterface::EYE_MONO;

			// check for an external texture destination for our left eye/mono
			VSG::storage->render_target_set_external_texture(vp->render_target, arvr_interface->get_external_texture_for_eye(leftOrMono));

			// set our render target as current
			VSG::rasterizer->set_current_render_target(vp->render_target);

			// and draw left eye/mono
			_draw_viewport(vp, leftOrMono);
			arvr_interface->commit_for_eye(leftOrMono, vp->render_target, vp->viewport_to_screen_rect);

			// render right eye
			if (leftOrMono == ARVRInterface::EYE_LEFT) {
				// check for an external texture destination for our right eye
				VSG::storage->render_target_set_external_texture(vp->render_target, arvr_interface->get_external_texture_for_eye(ARVRInterface::EYE_RIGHT));

				// commit for eye may have changed the render target
				VSG::rasterizer->set_current_render_target(vp->render_target);

				_draw_viewport(vp, ARVRInterface::EYE_RIGHT);
				arvr_interface->commit_for_eye(ARVRInterface::EYE_RIGHT, vp->render_target, vp->viewport_to_screen_rect);
			}

			// and for our frame timing, mark when we've finished committing our eyes
			ARVRServer::get_singleton()->_mark_commit();
		} else {
#endif
		{
			VSG::storage->render_target_set_external_texture(vp->render_target, 0);

			VSG::scene_render->set_debug_draw_mode(vp->debug_draw);
			VSG::storage->render_info_begin_capture();

			// render standard mono camera
			_draw_viewport(vp);

			VSG::storage->render_info_end_capture();
			vp->render_info[VS::VIEWPORT_RENDER_INFO_OBJECTS_IN_FRAME] = VSG::storage->get_captured_render_info(VS::INFO_OBJECTS_IN_FRAME);
			vp->render_info[VS::VIEWPORT_RENDER_INFO_VERTICES_IN_FRAME] = VSG::storage->get_captured_render_info(VS::INFO_VERTICES_IN_FRAME);
			vp->render_info[VS::VIEWPORT_RENDER_INFO_MATERIAL_CHANGES_IN_FRAME] = VSG::storage->get_captured_render_info(VS::INFO_MATERIAL_CHANGES_IN_FRAME);
			vp->render_info[VS::VIEWPORT_RENDER_INFO_SHADER_CHANGES_IN_FRAME] = VSG::storage->get_captured_render_info(VS::INFO_SHADER_CHANGES_IN_FRAME);
			vp->render_info[VS::VIEWPORT_RENDER_INFO_SURFACE_CHANGES_IN_FRAME] = VSG::storage->get_captured_render_info(VS::INFO_SURFACE_CHANGES_IN_FRAME);
			vp->render_info[VS::VIEWPORT_RENDER_INFO_DRAW_CALLS_IN_FRAME] = VSG::storage->get_captured_render_info(VS::INFO_DRAW_CALLS_IN_FRAME);

			if (vp->viewport_to_screen_rect != Rect2() && (!vp->viewport_render_direct_to_screen || !VSG::rasterizer->is_low_end())) {
				//copy to screen if set as such
				Rasterizer::BlitToScreen blit;
				blit.render_target = vp->render_target;
				blit.rect = vp->viewport_to_screen_rect;

				if (!blit_to_screen_list.has(vp->viewport_to_screen)) {
					blit_to_screen_list[vp->viewport_to_screen] = Vector<Rasterizer::BlitToScreen>();
				}

				blit_to_screen_list[vp->viewport_to_screen].push_back(blit);
			}
		}

		if (vp->update_mode == VS::VIEWPORT_UPDATE_ONCE) {
			vp->update_mode = VS::VIEWPORT_UPDATE_DISABLED;
		}

		RENDER_TIMESTAMP("<Rendering Viewport " + itos(i));
	}
	VSG::scene_render->set_debug_draw_mode(VS::VIEWPORT_DEBUG_DRAW_DISABLED);

	RENDER_TIMESTAMP("<Render Viewports");
	//this needs to be called to make screen swapping more efficient
	VSG::rasterizer->prepare_for_blitting_render_targets();

	for (Map<int, Vector<Rasterizer::BlitToScreen> >::Element *E = blit_to_screen_list.front(); E; E = E->next()) {
		VSG::rasterizer->blit_render_targets_to_screen(E->key(), E->get().ptr(), E->get().size());
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
	viewport->viewport_render_direct_to_screen = false;

	return rid;
}

void VisualServerViewport::viewport_set_use_arvr(RID p_viewport, bool p_use_arvr) {
	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	viewport->use_arvr = p_use_arvr;
}

void VisualServerViewport::viewport_set_size(RID p_viewport, int p_width, int p_height) {

	ERR_FAIL_COND(p_width < 0 && p_height < 0);

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	//	if (viewport->size.width == p_width && viewport->size.height == p_height) {
	//		return; //nothing to do
	//	}
	viewport->size = Size2(p_width, p_height);
	VSG::storage->render_target_set_size(viewport->render_target, p_width, p_height);
	if (viewport->render_buffers.is_valid()) {
		VSG::scene_render->render_buffers_configure(viewport->render_buffers, viewport->render_target, viewport->size.width, viewport->size.height, viewport->msaa);
	}
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

	// If using GLES2 we can optimize this operation by rendering directly to system_fbo
	// instead of rendering to fbo and copying to system_fbo after
	if (VSG::rasterizer->is_low_end() && viewport->viewport_render_direct_to_screen) {

		VSG::storage->render_target_set_size(viewport->render_target, p_rect.size.x, p_rect.size.y);
		VSG::storage->render_target_set_position(viewport->render_target, p_rect.position.x, p_rect.position.y);
	}

	viewport->viewport_to_screen_rect = p_rect;
	viewport->viewport_to_screen = p_screen;
}

void VisualServerViewport::viewport_set_render_direct_to_screen(RID p_viewport, bool p_enable) {
	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	if (p_enable == viewport->viewport_render_direct_to_screen)
		return;

	// if disabled, reset render_target size and position
	if (!p_enable) {

		VSG::storage->render_target_set_position(viewport->render_target, 0, 0);
		VSG::storage->render_target_set_size(viewport->render_target, viewport->size.x, viewport->size.y);
	}

	VSG::storage->render_target_set_flag(viewport->render_target, RasterizerStorage::RENDER_TARGET_DIRECT_TO_SCREEN, p_enable);
	viewport->viewport_render_direct_to_screen = p_enable;

	// if attached to screen already, setup screen size and position, this needs to happen after setting flag to avoid an unnecessary buffer allocation
	if (VSG::rasterizer->is_low_end() && viewport->viewport_to_screen_rect != Rect2() && p_enable) {

		VSG::storage->render_target_set_size(viewport->render_target, viewport->viewport_to_screen_rect.size.x, viewport->viewport_to_screen_rect.size.y);
		VSG::storage->render_target_set_position(viewport->render_target, viewport->viewport_to_screen_rect.position.x, viewport->viewport_to_screen_rect.position.y);
	}
}

void VisualServerViewport::viewport_detach(RID p_viewport) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	// if render_direct_to_screen was used, reset size and position
	if (VSG::rasterizer->is_low_end() && viewport->viewport_render_direct_to_screen) {

		VSG::storage->render_target_set_position(viewport->render_target, 0, 0);
		VSG::storage->render_target_set_size(viewport->render_target, viewport->size.x, viewport->size.y);
	}

	viewport->viewport_to_screen_rect = Rect2();
	viewport->viewport_to_screen = 0;
}

void VisualServerViewport::viewport_set_update_mode(RID p_viewport, VS::ViewportUpdateMode p_mode) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	viewport->update_mode = p_mode;
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
	viewport->canvas_map[p_canvas].sublayer = 0;
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
	viewport->transparent_bg = p_enabled;
}

void VisualServerViewport::viewport_set_global_canvas_transform(RID p_viewport, const Transform2D &p_transform) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	viewport->global_transform = p_transform;
}
void VisualServerViewport::viewport_set_canvas_stacking(RID p_viewport, RID p_canvas, int p_layer, int p_sublayer) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	ERR_FAIL_COND(!viewport->canvas_map.has(p_canvas));
	viewport->canvas_map[p_canvas].layer = p_layer;
	viewport->canvas_map[p_canvas].sublayer = p_sublayer;
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

	if (viewport->msaa == p_msaa) {
		return;
	}
	viewport->msaa = p_msaa;
	if (viewport->render_buffers.is_valid()) {
		VSG::scene_render->render_buffers_configure(viewport->render_buffers, viewport->render_target, viewport->size.width, viewport->size.height, p_msaa);
	}
}

int VisualServerViewport::viewport_get_render_info(RID p_viewport, VS::ViewportRenderInfo p_info) {

	ERR_FAIL_INDEX_V(p_info, VS::VIEWPORT_RENDER_INFO_MAX, -1);

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	if (!viewport)
		return 0; //there should be a lock here..

	return viewport->render_info[p_info];
}

void VisualServerViewport::viewport_set_debug_draw(RID p_viewport, VS::ViewportDebugDraw p_draw) {

	Viewport *viewport = viewport_owner.getornull(p_viewport);
	ERR_FAIL_COND(!viewport);

	viewport->debug_draw = p_draw;
}

bool VisualServerViewport::free(RID p_rid) {

	if (viewport_owner.owns(p_rid)) {

		Viewport *viewport = viewport_owner.getornull(p_rid);

		VSG::storage->free(viewport->render_target);
		VSG::scene_render->free(viewport->shadow_atlas);
		if (viewport->render_buffers.is_valid()) {
			VSG::scene_render->free(viewport->render_buffers);
		}

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

void VisualServerViewport::set_default_clear_color(const Color &p_color) {
	VSG::storage->set_default_clear_color(p_color);
}

VisualServerViewport::VisualServerViewport() {
}
