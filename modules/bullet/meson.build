_module_bullet_srcs = files([
    'area_bullet.cpp',
    'btRayShape.cpp',
    'bullet_physics_server.cpp',
    'bullet_types_converter.cpp',
    'collision_object_bullet.cpp',
    'cone_twist_joint_bullet.cpp',
    'constraint_bullet.cpp',
    'generic_6dof_joint_bullet.cpp',
    'godot_collision_configuration.cpp',
    'godot_collision_dispatcher.cpp',
    'godot_ray_world_algorithm.cpp',
    'godot_result_callbacks.cpp',
    'hinge_joint_bullet.cpp',
    'joint_bullet.cpp',
    'pin_joint_bullet.cpp',
    'register_types.cpp',
    'rigid_body_bullet.cpp',
    'shape_bullet.cpp',
    'shape_owner_bullet.cpp',
    'slider_joint_bullet.cpp',
    'soft_body_bullet.cpp',
    'space_bullet.cpp',
])

if not DEP_BULLET.found()
    error('module bullet requires bullet as a dependency')
endif

_lib_module_bullet = library('module_bullet', _module_bullet_srcs, dependencies: [DEP_GODOT_ALL, DEP_BULLET])

DEP_MODULE_BULLET = declare_dependency(link_with: _lib_module_bullet)

MODULE_DEPENDENCIES += [DEP_MODULE_BULLET]
