#ifndef RASTERIZERGLES3_H
#define RASTERIZERGLES3_H

#include "servers/visual/rasterizer.h"
#include "rasterizer_storage_gles3.h"
#include "rasterizer_canvas_gles3.h"
#include "rasterizer_scene_gles3.h"


class RasterizerGLES3 : public Rasterizer {

	static Rasterizer *_create_current();

	RasterizerStorageGLES3 *storage;
	RasterizerCanvasGLES3 *canvas;
	RasterizerSceneGLES3 *scene;

public:

	virtual RasterizerStorage *get_storage();
	virtual RasterizerCanvas *get_canvas();
	virtual RasterizerScene *get_scene();

	virtual void initialize();
	virtual void begin_frame();
	virtual void set_current_render_target(RID p_render_target);
	virtual void restore_render_target();
	virtual void clear_render_target(const Color& p_color);
	virtual void blit_render_target_to_screen(RID p_render_target,const Rect2& p_screen_rect,int p_screen=0);
	virtual void end_frame();
	virtual void finalize();

	static void make_current();


	static void register_config();
	RasterizerGLES3();
	~RasterizerGLES3();
};

#endif // RASTERIZERGLES3_H
