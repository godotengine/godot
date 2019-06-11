#include "rasterizer_rd.h"

RasterizerRD::RasterizerRD() {
	canvas = memnew(RasterizerCanvasRD);
	storage = memnew(RasterizerStorageRD);
	scene = memnew(RasterizerSceneForwardRD);
}
