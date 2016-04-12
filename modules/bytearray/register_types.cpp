/*************************************************/
/*  register_script_types.cpp                    */
/*************************************************/
/*            This file is part of:              */
/*                GODOT ENGINE                   */
/*************************************************/
/*       Source code within this file is:        */
/*  (c) 2007-2016 Juan Linietsky, Ariel Manzur   */
/*             All Rights Reserved.              */
/*************************************************/

/* register_types.cpp */

#include "register_types.h"
#include "object_type_db.h"
#include "_bytearray.h"

void register_bytearray_types() {

        ObjectTypeDB::register_type<_ByteArray>();
}

void unregister_bytearray_types() {
   //nothing to do here
}

