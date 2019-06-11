#ifndef RASTERIZER_RD_H
#define RASTERIZER_RD_H

#include "core/os/os.h"
#include "servers/visual/rasterizer/rasterizer.h"
#include "servers/visual/rasterizer/rasterizer_canvas_rd.h"
#include "servers/visual/rasterizer/rasterizer_scene_forward_rd.h"
#include "servers/visual/rasterizer/rasterizer_storage_rd.h"
class RasterizerRD : public Rasterizer {
protected:
	RasterizerCanvasRD *canvas;
	RasterizerStorageRD *storage;
	RasterizerSceneForwardRD *scene;

public:
	RasterizerStorage *get_storage() { return storage; }
	RasterizerCanvas *get_canvas() { return canvas; }
	RasterizerScene *get_scene() { return scene; }

	void set_boot_image(const Ref<Image> &p_image, const Color &p_color, bool p_scale, bool p_use_filter) {}

	void initialize() {}
	void begin_frame(double frame_step) {}
	void set_current_render_target(RID p_render_target) {}
	void restore_render_target(bool p_3d_was_drawn) {}
	void clear_render_target(const Color &p_color) {}
	void blit_render_target_to_screen(RID p_render_target, const Rect2 &p_screen_rect, int p_screen = 0) {}
	void output_lens_distorted_to_screen(RID p_render_target, const Rect2 &p_screen_rect, float p_k1, float p_k2, const Vector2 &p_eye_center, float p_oversample) {}
	void end_frame(bool p_swap_buffers) { OS::get_singleton()->swap_buffers(); }
	void finalize() {}

	static Error is_viable() {
		return OK;
	}

	static Rasterizer *_create_current() {
		return memnew(RasterizerRD);
	}

	static void make_current() {
		_create_func = _create_current;
	}

	virtual bool is_low_end() const { return true; }

	RasterizerRD();
	~RasterizerRD() {}
};
#endif // RASTERIZER_RD_H
