#ifndef REGISTER_MODULE_TYPES_H
#define REGISTER_MODULE_TYPES_H


void register_module_types();
void unregister_module_types();

#ifdef TOOLS_ENABLED
#include "scene/resources/theme.h"
void register_module_icons(Ref<Theme> theme);
#endif

#endif
