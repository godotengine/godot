//
//  std_io.h
//  std_io
//
//  Created by Oded Streigold on 4/18/20.
//

#ifndef std_io_h
#define std_io_h

#include "core/object/class_db.h"
#include "core/variant/variant.h"
#include "core/object/callable_method_pointer.h"
#include "core/object/ref_counted.h"

#include "core/string/print_string.h"
#include "core/string/ustring.h"
#include "process.hpp"


using namespace std;
using namespace TinyProcessLib;

class STD_IO : public RefCounted{
    GDCLASS(STD_IO, RefCounted);

    int count;
    Process* process = nullptr;
    Callable godot_output_callback;
    Variant returnval;

protected:
    static void _bind_methods();
    
    

public:
    
    std::string godot_string_std_string(String godot_string);

    STD_IO();
    ~STD_IO();
    int start_process(String path);
    int stop_process();
    int send_command(String command);
    void set_stdout_callback(Callable callback );

};


#endif /* std_io_h */
