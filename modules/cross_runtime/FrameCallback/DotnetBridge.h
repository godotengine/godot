#pragma once
#include "core/object/object.h"
#include "scene/main/scene_tree.h"
#include "core/config/engine.h"
#include "core/object/message_queue.h"
#include "core/object/callable_mp.h"  
#include "core/os/os.h"
#include "scene/main/window.h"

class DotnetBridge : public Object {
    GDCLASS(DotnetBridge, Object);
protected:
    static void _bind_methods() {}
public:
    static DotnetBridge *singleton;
    DotnetBridge();
    void _try_connect();
    inline void _on_process_frame();
};