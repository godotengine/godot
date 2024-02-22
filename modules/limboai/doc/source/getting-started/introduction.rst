.. _introduction:

Introduction to Behavior Trees
==============================

   **🛈 Note:** Demo project includes a tutorial that provides an introduction to behavior trees through illustrative examples.

**Behavior Trees (BT)** are hierarchical structures used to model and
control the behavior of agents in a game (e.g., characters, enemies,
entities). They are designed to make it easier to create complex and
highly modular behaviors for your games.

Behavior Trees are composed of tasks that represent specific actions or
decision-making rules. Tasks can be broadly categorized into two main
types: control tasks and leaf tasks. **Control tasks** determine the
execution flow within the tree. They include :ref:`Sequence<class_BTSequence>`,
:ref:`Selector<class_BTSelector>`, and
:ref:`Invert<class_BTInvert>`. **Leaf tasks** represent specific actions
to perform, like moving or attacking, or conditions that need to be
checked. The :ref:`BTTask<class_BTTask>` class provides the foundation for various
building blocks of the Behavior Trees. Such tasks can :ref:`share data using the Blackboard<blackboard>`.

   **🛈 Note:** To :ref:`create your own actions<custom_tasks>`, extend the :ref:`BTAction<class_BTAction>`
   class.

A Behavior Tree is usually processed each frame. It is traversed from top to bottom,
with the control tasks determining the control flow. Each task has a :ref:`_tick<class_BTTask_private_method__tick>`
method, which performs the task's work and returns a status indicating its progress:
``SUCCESS``, ``FAILURE``, or ``RUNNING``. ``SUCCESS`` and ``FAILURE`` indicate the
outcome of finished work, while ``RUNNING`` status is returned when a task requires
more than one tick to complete its job. These statuses determine how the tree
progresses, with the ``RUNNING`` status usually meaning that the tree will
continue execution during the next frame.

There are *four types of tasks*:

* **Actions** are leaf tasks that perform the actual work.

  * Examples: :ref:`PlayAnimation<class_BTPlayAnimation>`, :ref:`Wait<class_BTWait>`.

* **Conditions** are leaf tasks that conduct various checks.

  * Examples: :ref:`CheckVar<class_BTCheckVar>`, :ref:`InRange<example_in_range>`.

* **Composites** can have one or more child tasks, and dictate the execution flow of their children.

  * Examples: :ref:`Sequence<class_BTSequence>`, :ref:`Selector<class_BTSelector>`, :ref:`Parallel<class_BTParallel>`.

* **Decorators** can only have a single child and they change how their child task operates.

  * Examples: :ref:`AlwaysSucceed<class_BTAlwaysSucceed>`, :ref:`Invert<class_BTInvert>`, :ref:`TimeLimit<class_BTTimeLimit>`.

:ref:`Sequence<class_BTSequence>` is one of the core composite tasks.
It executes its child tasks sequentially, from first to last, until one of them
returns ``FAILURE``, or all of them result in ``SUCCESS``. In other words,
if any child task results in ``FAILURE``, the :ref:`Sequence<class_BTSequence>`
execution will be aborted, and the :ref:`Sequence<class_BTSequence>` itself will
return ``FAILURE``.

:ref:`Selector<class_BTSelector>` is another essential composite task.
It executes its child tasks sequentially, from first to last, until one of them
returns ``SUCCESS`` or all of them result in ``FAILURE``. In other words, when
a child task results in ``FAILURE``, it moves on to the next one until it
finds the one that returns ``SUCCESS``. Once a child task results in ``SUCCESS``,
the :ref:`Selector<class_BTSelector>` stops and also returns ``SUCCESS``.
The purpose of the :ref:`Selector<class_BTSelector>` is to find a child that succeeds.

Behavior Trees handle conditional logic using **condition tasks**. These
tasks check for specific conditions and return either ``SUCCESS`` or
``FAILURE`` based on the state of the agent or its environment (e.g.,
“IsLowOnHealth”, “IsTargetInSight”). Conditions can be used together
with :ref:`Sequence<class_BTSequence>` and :ref:`Selector<class_BTSelector>`
to craft your decision-making logic.

   **🛈 Note:** To :ref:`create your own conditions<custom_tasks>`, extend the :ref:`BTCondition<class_BTCondition>`
   class.

Check out the :ref:`BTTask<class_BTTask>` class documentation, which
provides the foundation for various building blocks of Behavior Trees.
