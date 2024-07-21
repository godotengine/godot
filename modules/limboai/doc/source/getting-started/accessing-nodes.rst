.. _accessing_nodes:

Accessing nodes in the scene tree
=================================

There are several ways to access nodes in the agent's scene tree from a :ref:`BTTask<class_BTTask>`.

    **🛈 Note:** The root node of the agent's scene tree can be accessed with the
    :ref:`scene_root<class_BTTask_property_scene_root>` property.


With ``BBNode`` property
------------------------

.. code:: gdscript

   @export var cast_param: BBNode

   func _tick(delta) -> Status:
       var node: ShapeCast3D = cast_param.get_value(scene_root, blackboard)


With ``NodePath`` property
--------------------------

.. code:: gdscript

   @export var cast_path: NodePath

   func _tick(delta) -> Status:
       var node: ShapeCast3D = scene_root.get_node(cast_path)


Using blackboard plan
---------------------

You can :ref:`create a blackboard variable<editing_plan>` in the editor with the type ``NodePath``
and point it to the proper node in the :ref:`BTPlayer<class_BTPlayer>` blackboard plan. By default,
any ``NodePath`` variable will be replaced with the node instance when the blackboard is instantiated
at runtime (see :ref:`BlackboardPlan.prefetch_nodepath_vars<class_BlackboardPlan_property_prefetch_nodepath_vars>`).

.. code:: gdscript

   extends BTCondition

   @export var shape_var: StringName = &"shape_cast"

   func _tick(delta) -> Status:
       var shape_cast: ShapeCast3D = blackboard.get_var(shape_var)

The property :ref:`BTPlayer.prefetch_nodepath_vars<class_BTPlayer_property_prefetch_nodepath_vars>` should be set to ``true``.
