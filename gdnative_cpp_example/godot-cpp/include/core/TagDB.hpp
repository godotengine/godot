#ifndef TAGDB_HPP
#define TAGDB_HPP

#include <stddef.h>

namespace godot {

namespace _TagDB {

void register_type(size_t type_tag, size_t base_type_tag);
bool is_type_known(size_t type_tag);
void register_global_type(const char *name, size_t type_tag, size_t base_type_tag);
bool is_type_compatible(size_t type_tag, size_t base_type_tag);

} // namespace _TagDB

} // namespace godot

#endif // TAGDB_HPP
