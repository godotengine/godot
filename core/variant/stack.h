#ifndef STACK_H
#define STACK_H

#include "core/variant/array.h"
#include "core/object/ref_counted.h"

class Stack : public RefCounted {
    GDCLASS(Stack, RefCounted);

private:
    Array _data;

protected:
    static void _bind_methods();

public:
    Stack();
    ~Stack();

    void push(const Variant &p_value);
    Variant pop();
    Variant peek() const;
    bool is_empty() const;
    int size() const;
    void clear();
    Array to_array() const;

    String _to_string() const;
};

#endif // STACK_H