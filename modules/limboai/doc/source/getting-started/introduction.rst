.. _introduction:

Introduction to Behavior Trees
==============================


**Behavior Trees (BT)** are hierarchical structures used to model and
control the behavior of agents in a game (e.g., characters, enemies,
entities). They are designed to make it easier to create complex and
highly modular behaviors for your games.

Behavior Trees are composed of tasks that represent specific actions or
decision-making rules. Tasks can be broadly categorized into two main
types: control tasks and leaf tasks. Control tasks determine the
execution flow within the tree. They include :ref:`Sequence<class_BTSequence>`,
:ref:`Selector<class_BTSelector>`, and
:ref:`Invert<class_BTInvert>`. Leaf tasks represent specific actions
to perform, like moving or attacking, or conditions that need to be
checked. The :ref:`BTTask<class_BTTask>` class provides the foundation for various
building blocks of the Behavior Trees. BT tasks can share data with the
help of the :ref:`Blackboard<class_Blackboard>`.

   **🛈 Note:** To create your own actions, extend the :ref:`BTAction<class_BTAction>`
   class.

The Behavior Tree is executed from the root task and follows the rules
specified by the control tasks, all the way down to the leaf tasks,
which represent the actual actions that the agent should perform or
conditions that should be checked. Each task returns a status when it is
executed. It can be ``SUCCESS``, ``RUNNING``, or ``FAILURE``. These
statuses determine how the tree progresses. They are defined in
:ref:`BT.Status <enum_BT_Status>`.

Behavior Trees handle conditional logic using condition tasks. These
tasks check for specific conditions and return either ``SUCCESS`` or
``FAILURE`` based on the state of the agent or its environment (e.g.,
“IsLowOnHealth”, “IsTargetInSight”). Conditions can be used together
with :ref:`Sequence<class_BTSequence>` and :ref:`Selector<class_BTSelector>`
to craft your decision-making logic.

   **🛈 Note:** To create your own conditions, extend the :ref:`BTCondition<class_BTCondition>`
   class.

Check out the :ref:`BTTask<class_BTTask>` class documentation, which
provides the foundation for various building blocks of Behavior Trees.
