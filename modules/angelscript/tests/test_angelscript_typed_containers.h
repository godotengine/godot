/**************************************************************************/
/*  test_angelscript_typed_containers.h                                   */
/**************************************************************************/
#pragma once

#include "tests/test_macros.h"

void as_typed_containers_register_all();
void as_typed_containers_numeric_arrays();
void as_typed_containers_vector_arrays();
void as_typed_containers_maps();
void as_typed_containers_bounds_exception();
void as_typed_containers_object_ids();
void as_typed_containers_bulk_conversion();

#ifndef ANGELSCRIPT_TYPED_CONTAINERS_TESTS_IMPL
TEST_CASE("[AngelScript] typed containers register all fixed types") {
	as_typed_containers_register_all();
}
TEST_CASE("[AngelScript] typed numeric arrays support batches") {
	as_typed_containers_numeric_arrays();
}
TEST_CASE("[AngelScript] typed vector arrays support batches") {
	as_typed_containers_vector_arrays();
}
TEST_CASE("[AngelScript] typed maps store fixed key value types") {
	as_typed_containers_maps();
}
TEST_CASE("[AngelScript] typed arrays report bounds errors") {
	as_typed_containers_bounds_exception();
}
TEST_CASE("[AngelScript] typed object containers use safe ObjectIDs") {
	as_typed_containers_object_ids();
}
TEST_CASE("[AngelScript] typed containers bulk convert Godot containers") {
	as_typed_containers_bulk_conversion();
}
#endif
