//
//  register_types.h
//  std_io
//
//  Created by Oded Streigold on 4/18/20.
//

#ifndef register_types_h
#define register_types_h

#include "modules/register_module_types.h"
void initialize_std_io_module(ModuleInitializationLevel p_level);
//void initialize_std_io_types();
void uninitialize_std_io_module(ModuleInitializationLevel p_level);
//void uninitialize_std_io_types();
/* yes, the word in the middle must be the same as the module folder name */


#endif /* register_types_h */
