#ifndef VISUALSERVERLIGHTBAKER_H
#define VISUALSERVERLIGHTBAKER_H

#include "servers/visual_server.h"

class VisualServerLightBaker {
public:

	struct BakeCell {

		uint32_t cells[8];
		uint32_t neighbours[7]; //one unused
		uint32_t albedo; //albedo in RGBE
		uint32_t emission; //emissive light in RGBE
		uint32_t light[4]; //accumulated light in 16:16 fixed point (needs to be integer for moving lights fast)
		float alpha; //used for upsampling
		uint32_t directional_pass; //used for baking directional

	};






	VisualServerLightBaker();
};

#endif // VISUALSERVERLIGHTBAKER_H
