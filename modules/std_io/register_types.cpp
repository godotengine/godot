//
//  register_types.cpp
//  std_io
//
//  Created by Oded Streigold on 4/18/20.
//

#include "register_types.h"
#include "core/object/class_db.h"
#include "std_io.h"


void initialize_std_io_module(ModuleInitializationLevel p_level) {
   if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
      return;
   }
    //ClassDB::initialize_class<STD_IO>();
   GDREGISTER_CLASS(STD_IO);
}

void uninitialize_std_io_module(ModuleInitializationLevel p_level) {
   // Nothing to do here in this example.
   if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
      return;
   }
}
