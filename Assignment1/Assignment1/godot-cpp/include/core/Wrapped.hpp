#ifndef WRAPPED_HPP
#define WRAPPED_HPP

#include <gdnative/gdnative.h>

namespace godot {

class _Wrapped {
public:
	godot_object *_owner;
	size_t _type_tag;
};

} // namespace godot

#endif // WRAPPED_HPP
