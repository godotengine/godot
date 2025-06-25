#include "svg_utils.h"
#include "thirdparty/lunasvg/include/lunasvg.h"

#include <lunasvg.h>

void SVGUtils::set_default_font(const void *font_data, int length) {
	lunasvg_add_font_face_from_data("", false, false, font_data, length, nullptr, nullptr);
}
