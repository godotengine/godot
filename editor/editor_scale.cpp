#include "editor_scale.h"
#include "os/os.h"

static bool editor_hidpi = false;

void editor_set_hidpi(bool p_hidpi) {

	editor_hidpi = p_hidpi;
}

bool editor_is_hidpi() {

	return editor_hidpi;
}
