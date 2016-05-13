
#include "global_defaults.h"
#include "globals.h"


void register_iphone_global_defaults() {

	GLOBAL_DEF("rasterizer.iOS/use_fragment_lighting",false);
	GLOBAL_DEF("rasterizer.iOS/fp16_framebuffer",false);
	GLOBAL_DEF("display.iOS/driver","GLES2");
	Globals::get_singleton()->set_custom_property_info("display.iOS/driver",PropertyInfo(Variant::STRING,"display.iOS/driver",PROPERTY_HINT_ENUM,"GLES1,GLES2"));
	GLOBAL_DEF("display.iOS/use_cadisplaylink",true);
}
