.. _custom_tasks:

Creating custom tasks in GDScript
=================================

By default, user tasks should be placed in the ``res://ai/tasks``
directory. You can set an alternative location for user tasks in the
``Project Settings → Limbo AI`` (To see those options,
``Advanced Settings`` should be enabled in the Project Settings).

Each subdirectory within the user tasks directory is treated as a category.
Therefore, if you create a subdirectory named “motion_and_physics,” your
custom tasks in that directory will automatically be categorized under
“Motion And Physics.”

When creating custom tasks, **extend one of the following** base classes:
:ref:`BTAction<class_BTAction>`, :ref:`BTCondition<class_BTCondition>`, :ref:`BTDecorator<class_BTDecorator>`, :ref:`BTComposite<class_BTComposite>`.
More on task types you can read in the :ref:`introduction`.

   **🛈 Note:** To help you write new tasks, you can add a script template to
   your project using “Misc → Create script template” menu option.

Using the :ref:`Blackboard<class_Blackboard>` is covered in :ref:`accessing_blackboard`.

Task anatomy
------------

.. code:: gdscript

   @tool
   extends BTAction

   # Task parameters.
   @export var parameter1: float
   @export var parameter2: Vector2

   ## Note: Each method declaration is optional.
   ## At minimum, you only need to define the "_tick" method.


   # Called to generate a display name for the task (requires @tool).
   func _generate_name() -> String:
       return "MyTask"


   # Called to initialize the task.
   func _setup() -> void:
       pass


   # Called when the task is entered.
   func _enter() -> void:
       pass


   # Called when the task is exited.
   func _exit() -> void:
       pass


   # Called each time this task is ticked (aka executed).
   func _tick(delta: float) -> Status:
       return SUCCESS


   # Strings returned from this method are displayed as warnings in the editor.
   func _get_configuration_warnings() -> PackedStringArray:
       var warnings := PackedStringArray()
       return warnings


Example 1: A simple action
--------------------------

.. code:: gdscript

   @tool
   extends BTAction

   ## Shows or hides a node and returns SUCCESS.
   ## Returns FAILURE if the node is not found.

   # Task parameters.
   @export var node_path: NodePath
   @export var visible := true


   # Called to generate a display name for the task (requires @tool).
   func _generate_name() -> String:
       return "SetVisible  %s  node_path: \"%s\"" % [visible, node_path]


   # Called each time this task is ticked (aka executed).
   func _tick(p_delta: float) -> Status:
       var n: CanvasItem = scene_root.get_node_or_null(node_path)
       if is_instance_valid(n):
           n.visible = visible
           return SUCCESS
       return FAILURE


.. _example_in_range:

Example 2: InRange condition
----------------------------

.. code:: gdscript

   @tool
   extends BTCondition

   ## InRange condition checks if the agent is within a range of target,
   ## defined by distance_min and distance_max.
   ## Returns SUCCESS if the agent is within the defined range;
   ## otherwise, returns FAILURE.

   @export var distance_min: float
   @export var distance_max: float
   @export var target_var: StringName = &"target"

   var _min_distance_squared: float
   var _max_distance_squared: float


   # Called to generate a display name for the task.
   func _generate_name() -> String:
       return "InRange (%d, %d) of %s" % [distance_min, distance_max,
           LimboUtility.decorate_var(target_var)]


   # Called to initialize the task.
   func _setup() -> void:
       _min_distance_squared = distance_min * distance_min
       _max_distance_squared = distance_max * distance_max


   # Called when the task is executed.
   func _tick(_delta: float) -> Status:
       var target: Node2D = blackboard.get_var(target_var, null)
       if not is_instance_valid(target):
           return FAILURE

       var dist_sq: float = agent.global_position.distance_squared_to(target.global_position)
       if dist_sq >= _min_distance_squared and dist_sq <= _max_distance_squared:
           return SUCCESS
       else:
           return FAILURE
