#include "editor_scale.h"
#include "os/os.h"

bool editor_is_hidpi() {

	return OS::get_singleton()->get_screen_dpi(0) > 150;
}
