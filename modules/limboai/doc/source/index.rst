LimboAI Documentation
=====================

About
-----

**LimboAI** is an open-source C++ module for **Godot Engine 4** providing a combination of
**Behavior Trees** and **State Machines** for crafting your game’s AI. It comes with a
behavior tree editor, built-in documentation, visual debugger, and more! While
it is implemented in C++, it fully supports GDScript for :ref:`creating your own tasks <custom_tasks>`
and states. The full list of features is available on the
`LimboAI GitHub <https://github.com/limbonaut/limboai#features>`_ page.

.. SCREENSHOT

**Behavior Trees** are powerful hierarchical structures used to model and control the behavior
of agents in a game (e.g., characters, enemies, entities). They are designed to
make it easier to create complex and highly modular behaviors for your games.
To learn more about behavior trees, check out :ref:`introduction`.


Getting LimboAI
---------------

Precompiled builds are available on the official
`LimboAI GitHub <https://github.com/limbonaut/limboai#getting-limboai>`_ page,
and in the Asset Library (coming soon!).

LimboAI can be used as either a C++ module or as a GDExtension shared library.
There are some differences between the two. In short, GDExtension version is more
convenient to use but somewhat limited in features. Whichever you choose to use,
your project will stay compatible with both and you can switch from one to
the other any time. For more information on this topic, see :ref:`gdextension`.

   **🛈 Note:** Class reference is available in the side bar.

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting started

   getting-started/introduction
   getting-started/custom-tasks
   getting-started/using-blackboard
   getting-started/gdextension
   getting-started/c-sharp
   getting-started/accessing-nodes
   getting-started/featured-classes

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Class reference
   :glob:

   classes/class_*
