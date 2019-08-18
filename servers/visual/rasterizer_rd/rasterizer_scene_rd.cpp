#include "rasterizer_scene_rd.h"

RID RasterizerSceneRD::render_buffers_create() {
	RenderBuffers rb;
	rb.data = _create_render_buffer_data();
	return render_buffers_owner.make_rid(rb);
}

void RasterizerSceneRD::render_buffers_configure(RID p_render_buffers, RID p_render_target, int p_width, int p_height, VS::ViewportMSAA p_msaa) {

	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	rb->width = p_width;
	rb->height = p_height;
	rb->render_target = p_render_target;
	rb->msaa = p_msaa;
	rb->data->configure(p_render_target, p_width, p_height, p_msaa);
}

void RasterizerSceneRD::render_scene(RID p_render_buffers, const Transform &p_cam_transform, const CameraMatrix &p_cam_projection, bool p_cam_ortogonal, InstanceBase **p_cull_result, int p_cull_count, RID *p_light_cull_result, int p_light_cull_count, RID *p_reflection_probe_cull_result, int p_reflection_probe_cull_count, RID p_environment, RID p_shadow_atlas, RID p_reflection_atlas, RID p_reflection_probe, int p_reflection_probe_pass) {

	RenderBuffers *rb = render_buffers_owner.getornull(p_render_buffers);
	ERR_FAIL_COND(!rb && p_render_buffers.is_valid());

	_render_scene(rb->data, p_cam_transform, p_cam_projection, p_cam_ortogonal, p_cull_result, p_cull_count, p_light_cull_result, p_light_cull_count, p_reflection_probe_cull_result, p_reflection_probe_cull_count, p_environment, p_shadow_atlas, p_reflection_atlas, p_reflection_probe, p_reflection_probe_pass);
}

bool RasterizerSceneRD::free(RID p_rid) {

	if (render_buffers_owner.owns(p_rid)) {
		RenderBuffers *rb = render_buffers_owner.getornull(p_rid);
		memdelete(rb->data);
		render_buffers_owner.free(p_rid);
	} else {
		return false;
	}

	return true;
}

RasterizerSceneRD::RasterizerSceneRD() {
}
