/**
 * register_types.h
 * =============================================================================
 * Copyright 2021-2024 Serhii Snitsaruk
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

#ifndef LIMBOAI_REGISTER_TYPES_H
#define LIMBOAI_REGISTER_TYPES_H

#ifdef LIMBOAI_MODULE
#include "modules/register_module_types.h"
#endif // LIMBOAI_MODULE

#ifdef LIMBOAI_GDEXTENSION
#include <godot_cpp/core/class_db.hpp>
using namespace godot;
#endif // LIMBOAI_GDEXTENSION

void initialize_limboai_module(ModuleInitializationLevel p_level);
void uninitialize_limboai_module(ModuleInitializationLevel p_level);

#endif // LIMBOAI_REGISTER_TYPES_H
