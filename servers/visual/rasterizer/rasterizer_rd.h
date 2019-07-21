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

	RID copy_viewports_rd_shader;
	RID copy_viewports_rd_pipeline;
	RID copy_viewports_rd_index_buffer;
	RID copy_viewports_rd_array;
	RID copy_viewports_sampler;

	Map<RID, RID> render_target_descriptors;

	double time;

public:
	RasterizerStorage *get_storage() { return storage; }
	RasterizerCanvas *get_canvas() { return canvas; }
	RasterizerScene *get_scene() { return scene; }

	void set_boot_image(const Ref<Image> &p_image, const Color &p_color, bool p_scale, bool p_use_filter) {}

	void initialize();
	void begin_frame(double frame_step);
	void prepare_for_blitting_render_targets();
	void blit_render_targets_to_screen(int p_screen, const BlitToScreen *p_render_targets, int p_amount);

	void end_frame(bool p_swap_buffers);
	void finalize();

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
