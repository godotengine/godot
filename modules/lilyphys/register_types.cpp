#include "register_types.h"

#include "core/class_db.h"
#include "core/engine.h"

#include "lilyphys_server.h"
#include "_lilyphys_server.h"
#include "l_collision_solver.h"

#include "nodes/l_collision_object.h"
#include "nodes/l_physics_body.h"
#include "nodes/l_static_body.h"
#include "nodes/l_rigid_body.h"
#include "nodes/l_trigger.h"
#include "nodes/l_collision_shape.h"
#include "nodes/l_shape.h"
#include "lilyphys_editor_plugin.h"

static LilyphysServer *lilyphys_server = nullptr;
static _LilyphysServer *_lilyphys_sever = nullptr;

void register_lilyphys_types() {
    // Register the server.
    lilyphys_server = memnew(LilyphysServer);
    lilyphys_server->init();
    _lilyphys_sever = memnew(_LilyphysServer);
    ClassDB::register_class<_LilyphysServer>();
    Engine::get_singleton()->add_singleton(Engine::Singleton("LilyphysServer", _LilyphysServer::get_singleton()));

    // Register our nodes.
    ClassDB::register_virtual_class<LCollisionObject>();
    ClassDB::register_virtual_class<LPhysicsBody>();
    ClassDB::register_class<LRigidBody>();
    ClassDB::register_class<LStaticBody>();
    ClassDB::register_class<LTrigger>();
    ClassDB::register_class<LCollisionShape>();
    ClassDB::register_virtual_class<LShape>();
    ClassDB::register_class<LBoxShape>();
    ClassDB::register_class<LSphereShape>();
    ClassDB::register_class<LCollision>();

#ifdef TOOLS_ENABLED
    EditorPlugins::add_by_type<LilyphysEditorPlugin>();
#endif
}

void unregister_lilyphys_types() {
    if (lilyphys_server) {
        lilyphys_server->finish();
        memdelete(lilyphys_server);
    }

    if (_lilyphys_sever) {
        memdelete(_lilyphys_sever);
    }
}