
// need.h
// Aggregated include header for all generated C++ exports.
// Pulls in the full set of Godot core headers required across
// variant_accessors, variant_constructors, and every generated class file.
// Also declares the resource_cache (HashMap<ObjectID, Ref<Resource>>) used
// by RefCounted return paths, and the cache_loaded_resource() helper that
// registers a Resource with the cache to prevent premature collection.
// The resource_cache definition lives in the ResourceLoader class file;
// a release method must be called from the JS side to evict entries.

#include "core/object/object_id.h"
#include "core/string/node_path.h"
#include "core/string/string_name.h"
#include "core/string/ustring.h"
#include "core/templates/rid.h"
#include "core/variant/array.h"
#include "core/variant/callable.h"
#include "core/variant/dictionary.h"
#include "core/variant/variant.h"
#include "core/error/error_macros.h"

#include "core/object/class_db.h"
#include "core/object/method_bind.h"

#include "core/templates/hash_map.h"
#include "core/object/object_id.h"
#include "core/object/ref_counted.h"
#include "core/object/object.h"
#include "core/io/resource.h"
#include <emscripten.h>
#include <string>
#include <cstdint>

// Caches Refcounted resources
extern HashMap<ObjectID, Ref<Resource>> resource_cache;

static inline void cache_loaded_resource(Object *obj) {
    if (!obj) {
        return;
    }

    Resource *res = Object::cast_to<Resource>(obj);
    if (!res) {
        return;
    }

    ObjectID id = res->get_instance_id();
    resource_cache[id] = Ref<Resource>(res);
}
