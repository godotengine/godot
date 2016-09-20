#include "editor_scale.h"
#include "os/os.h"

static float scale = 1.0;

void editor_set_scale(float p_scale) {

	scale=p_scale;
}
float editor_get_scale() {

	return scale;
}
