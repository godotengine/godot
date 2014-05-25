#ifndef BAKED_LIGHT_H
#define BAKED_LIGHT_H

#include "scene/3d/spatial.h"
class BakedLightBaker;


class BakedLight : public Spatial {
	OBJ_TYPE(BakedLight,Spatial);

public:
	BakedLight();
};

#endif // BAKED_LIGHT_H
