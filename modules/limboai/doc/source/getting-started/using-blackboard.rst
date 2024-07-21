.. _blackboard:

Sharing data using Blackboard
=============================

To share data between different tasks and states, we employ a feature known as the :ref:`Blackboard<class_Blackboard>`.
The :ref:`Blackboard<class_Blackboard>` serves as a central repository where tasks and states can store and retrieve named variables,
allowing for seamless data interchange. Each instance of a behavior tree or a state machine gets its own dedicated :ref:`Blackboard<class_Blackboard>`. It has the capability to store various data types,
including objects and resources.

Using the :ref:`Blackboard<class_Blackboard>`, you can easily share data in your behavior trees, making the tasks in the behavior tree more flexible.

.. _accessing_blackboard:

Accessing the Blackboard in a Task
----------------------------------

Every :ref:`BTTask<class_BTTask>` has access to the :ref:`Blackboard<class_Blackboard>`, providing a
straightforward mechanism for data exchange.
Here's an example of how you can interact with the :ref:`Blackboard<class_Blackboard>` in GDScript:

.. code:: gdscript

    @export var speed_var: StringName = &"speed"

    func _tick(delta: float) -> Status:
        # Set the value of the "speed" variable:
        blackboard.set_var(speed_var, 200.0)

        # Get the value of the "speed" variable, with a default value of 100.0 if not found:
        var speed: float = blackboard.get_var(speed_var, 100.0)

        # Check if the "speed" variable exists:
        if blackboard.has_var(speed_var):
            # ...

It is recommended to suffix variable name properties with ``_var``, like in the example above, which enables the
inspector to provide a more convenient property editor for the variable. This editor
allows you to select or add the variable to the blackboard plan, and provides a
warning icon if the variable does not exist in the blackboard plan.

    **🛈 Note:** The variable doesn't need to exist when you set it in code.

.. _editing_plan:

Editing the Blackboard Plan
---------------------------

The Blackboard Plan, associated with each :ref:`BehaviorTree<class_BehaviorTree>`
resource, dictates how the :ref:`Blackboard<class_Blackboard>` initializes for each
new instance of the :ref:`BehaviorTree<class_BehaviorTree>`.
BlackboardPlan resource stores default values, type information, and data bindings
necessary for :ref:`BehaviorTree<class_BehaviorTree>` initialization.

To add, modify, or remove variables from the Blackboard Plan, follow these steps:

1. Load the behavior tree in the LimboAI editor.
2. Click on the resource header to access :ref:`BehaviorTree<class_BehaviorTree>` data in the Inspector.
3. In the Inspector, select the "Blackboard Plan" property and click the "Manage..." button.
4. A popup window will appear, allowing you to edit behavior tree variables, reorder them, and modify property types and hints.

Overriding variables in BTPlayer
--------------------------------

Each :ref:`BTPlayer<class_BTPlayer>` node also has a "Blackboard Plan" property,
providing the ability to override values of the BehaviorTree's blackboard variables.
These overrides are specific to the BTPlayer's scene
and do not impact other scenes using the same :ref:`BehaviorTree<class_BehaviorTree>`.
To modify these values:

1. Select the BTPlayer node in the scene tree.
2. In the Inspector, locate the "Blackboard Plan" property.
3. Override the desired values to tailor the blackboard variables for the specific scene.

Task parameters
---------------

In some cases, it can be beneficial to allow behavior tree tasks to export parameters
that can either be **bound to a blackboard variable or specified directly** by the user.
For this purpose, LimboAI provides special parameter types that begin with "BB",
such as :ref:`BBInt<class_BBInt>`, :ref:`BBBool<class_BBBool>`, :ref:`BBString<class_BBString>`,
:ref:`BBFloat<class_BBFloat>`, :ref:`BBNode<class_BBNode>`, and more.
For a complete list, please refer to the :ref:`BBParam<class_BBParam>` class reference.

Usage example:

.. code:: gdscript

    extends BTAction

    @export var speed: BBFloat

    func _tick(delta: float) -> Status:
        var current_speed: float = speed.get_value(scene_root, blackboard, 0.0)
        ...

Advanced topic: Blackboard scopes
---------------------------------

The :ref:`Blackboard<class_Blackboard>` in LimboAI can act as a parent scope
for another :ref:`Blackboard<class_Blackboard>`.
This means that if a specific variable is not found in the active scope,
the system will look in the parent :ref:`Blackboard<class_Blackboard>` to find it.
This creates a "blackboard scope chain," where each :ref:`Blackboard<class_Blackboard>` can have its own parent scope,
and there is no limit to how many blackboards can be in this chain.
It's important to note that the :ref:`Blackboard<class_Blackboard>` doesn't modify values in the parent scopes.

Some scopes are created automatically. For instance, when using the :ref:`BTNewScope<class_BTNewScope>`
and :ref:`BTSubtree<class_BTSubtree>` decorators, or when a :ref:`LimboState<class_LimboState>`
has non-empty blackboard plan defined, or when a root-level :ref:`LimboHSM<class_LimboHSM>`
node is used. Such scopes prevent naming collisions between contextually separate environments.

Sharing data between several agents
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The blackboard scope mechanism can also be used for sharing data between several agents.
In the following example, we have a group of agents, and we want to share a common target between them:

.. code:: gdscript

    extends BTAction

    @export var group_target_var: StringName = &"group_target"

    func _tick(delta: float) -> Status:
        if not blackboard.has_var(group_target_var):
            var new_target: Node = acquire_target()
            # Set common target shared between agents in a group:
            blackboard.top().set_var(group_target_var, new_target)

        # Access common target shared between agents in a group:
        var target: Node = blackboard.get_var(group_target_var)


In this example, :ref:`blackboard.top()<class_Blackboard_method_top>` accesses the root scope of the
:ref:`Blackboard<class_Blackboard>` chain.
We assign that scope to each agent in a group through code:

.. code:: gdscript

    class_name AgentGroup
    extends Node2D
    ## AgentGroup node: Manages the shared Blackboard for agents in a group.
    ## Children of this node are assumed to be agents that belong to a common group.
    ## This implementation assumes that each agent has a "BTPlayer" node for AI.

    @export var blackboard_plan: BlackboardPlan

    var shared_scope: Blackboard

    func _ready() -> void:
        if blackboard_plan == null:
            shared_scope = Blackboard.new()
        else:
            shared_scope = blackboard_plan.create_blackboard()

        for child in get_children():
            var bt_player: BTPlayer = child.find_child("BTPlayer")
            if is_instance_valid(bt_player):
                bt_player.blackboard.set_parent(shared_scope)

In conclusion, the :ref:`Blackboard<class_Blackboard>` scope chain not only
prevents naming conflicts that can occur between state machines, behavior trees, and sub-trees,
but it can also be used to share data between several agents.
