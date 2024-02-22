.. _blackboard:

Sharing data using Blackboard
=============================

To share data between different tasks and states, we employ a feature known as the :ref:`Blackboard<class_Blackboard>`.
The :ref:`Blackboard<class_Blackboard>` serves as a central repository where tasks and states can store and retrieve named variables,
allowing for seamless data interchange. Each instance of a behavior tree or a state machine gets its own dedicated :ref:`Blackboard<class_Blackboard>`. It has the capability to store various data types,
including objects and resources.

Using the :ref:`Blackboard<class_Blackboard>`, you can easily share data in your behavior trees, making the tasks in the behavior tree more flexible.

Accessing the Blackboard in a Task
----------------------------------

Every :ref:`BTTask<class_BTTask>` has access to the :ref:`Blackboard<class_Blackboard>`, providing a
straightforward mechanism for data exchange.
Here's an example of how you can interact with the :ref:`Blackboard<class_Blackboard>` in GDScript:

.. code:: gdscript

    @export var speed_var: String = "speed"

    func _tick(delta: float) -> Status:
        # Set the value of the "speed" variable:
        blackboard.set_var(speed_var, 200.0)

        # Get the value of the "speed" variable, with a default value of 100.0 if not found:
        var speed: float = blackboard.get_var(speed_var, 100.0)

        # Check if the "speed" variable exists:
        if blackboard.has_var(speed_var):
            # ...
..

    **🛈 Note:** The variable doesn't need to exist when you set it.

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

Advanced topic: Blackboard scopes
---------------------------------

    **🛈 Note:** This section is not finished.

    **🛈 Note:** Blackboard scopes isolate variable namespaces and enable advanced techniques like sharing data between agents in a group.
