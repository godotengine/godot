#pragma once
#include "core/io/resource.h"

class ResourceEditorToolItem : public RefCounted
{
    GDCLASS(ResourceEditorToolItem, RefCounted)
    static void _bind_methods() {}
public:
    virtual String get_name() const { return String(L"ResourceEditorToolItem"); }
    virtual Control *get_control() { return nullptr; }
    virtual ~ResourceEditorToolItem() {}
};
