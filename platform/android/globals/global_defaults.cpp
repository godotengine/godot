
#include "global_defaults.h"
#include "globals.h"


void register_android_global_defaults() {

	GLOBAL_DEF("rasterizer.Android/use_fragment_lighting",false);
	GLOBAL_DEF("rasterizer.Android/fp16_framebuffer",false);
	GLOBAL_DEF("display.Android/driver","GLES2");
//	GLOBAL_DEF("rasterizer.Android/trilinear_mipmap_filter",false);

	Globals::get_singleton()->set_custom_property_info("display.Android/driver",PropertyInfo(Variant::STRING,"display.Android/driver",PROPERTY_HINT_ENUM,"GLES2"));
}
