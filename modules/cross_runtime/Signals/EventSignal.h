// EventSignal.h
#pragma once
#include "core/object/object.h"
#include "core/object/object_id.h"
#include "core/string/string_name.h"
#include "core/variant/callable.h"
#include "core/variant/array.h"
#include "core/variant/variant.h"
#include <emscripten.h>
#include <vector>

class CrossRuntimeEventSignal : public Object {
    GDCLASS(CrossRuntimeEventSignal, Object);
    static CrossRuntimeEventSignal *singleton;
protected:
    static void _bind_methods();
public:
    CrossRuntimeEventSignal();
    ~CrossRuntimeEventSignal();
    static CrossRuntimeEventSignal *get_singleton() { return singleton; }
    void _on_any_signal(const Variant **p_args, int p_argcount, Callable::CallError &r_error);
    void connect_signal(int64_t object_id, const String &signal_name);
    void disconnect_signal(int64_t object_id, const String &signal_name);
};