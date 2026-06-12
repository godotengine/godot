/*
imports Godot's definitions for module initialization*/
#pragma once

#include "modules/register_module_types.h"

//at startup
void initialize_cross_runtime_module(ModuleInitializationLevel p_level);
//at shutdown
void uninitialize_cross_runtime_module(ModuleInitializationLevel p_level);