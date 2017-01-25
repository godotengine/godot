#ifndef VISUALSERVERGLOBAL_H
#define VISUALSERVERGLOBAL_H

#include "rasterizer.h"

class VisualServerCanvas;
class VisualServerViewport;
class VisualServerScene;

class VisualServerGlobals
{
public:

	static RasterizerStorage *storage;
	static RasterizerCanvas *canvas_render;
	static RasterizerScene *scene_render;
	static Rasterizer *rasterizer;

	static VisualServerCanvas *canvas;
	static VisualServerViewport *viewport;
	static VisualServerScene *scene;
};

#define VSG VisualServerGlobals

#endif // VISUALSERVERGLOBAL_H
