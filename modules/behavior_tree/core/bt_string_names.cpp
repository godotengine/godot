#include "bt_string_names.h"

BTStringNames* BTStringNames::singleton=NULL;

BTStringNames::BTStringNames() {
	_continue = StaticCString::create("_bt_continue");
	_prepare = StaticCString::create("_bt_prepare");
	_self_update = StaticCString::create("_bt_self_update");
	_child_update = StaticCString::create("_bt_child_update");
	_update = StaticCString::create("_bt_update");
	_pre_update = StaticCString::create("_bt_pre_update");
	_post_update = StaticCString::create("_bt_post_update");
	_abort = StaticCString::create("_bt_abort");
}
