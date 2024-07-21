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


First steps
-----------

Choose the version you'd like to use. The module version provides better editor
experience and performance, while the GDExtension version is more convenient to use.
If you're unsure, start with the GDExtension version.
You can change your decision at any time - both versions are fully compatible.
For more information, see :ref:`gdextension`.

With GDExtension version
~~~~~~~~~~~~~~~~~~~~~~~~

1. Make sure you're using the latest stable version of the Godot editor.
2. Create a new project for your experiments with LimboAI.
3. In Godot, click AssetLib tab at the top of the screen and search for LimboAI. Download it. LimboAI plugin will be downloaded with the demo project files. Don't mind the errors printed at this point, this is due to the extension library not being loaded just yet.
4. Reload your project with `Project -> Reload project`. There shouldn't be any errors printed now.
5. In the project files, locate a scene file called `showcase.tscn` and run it. It's the demo's entry point.

With module version
~~~~~~~~~~~~~~~~~~~

1. In `GitHub releases <https://github.com/limbonaut/limboai/releases/>`_, download the latest pre-compiled release build for your platform.
2. Download the demo project archive from the same release.
3. Extract the pre-compiled editor and the demo project files.
4. Launch the pre-compiled editor binary, import and open the demo project.
5. Run the project.

Creating your own behavior trees
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. Make a scene file for your agent, or open an existing scene.
2. Add a :ref:`BTPlayer<class_BTPlayer>` node to your scene.
3. Select :ref:`BTPlayer<class_BTPlayer>`, and create a new behavior tree in the inspector.
4. Optionally, you can save the behavior tree to a file using the property's context menu.
5. Click the behavior tree property to open it in the LimboAI editor.


.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting started

   getting-started/introduction
   getting-started/custom-tasks
   getting-started/using-blackboard
   getting-started/accessing-nodes
   getting-started/hsm
   getting-started/gdextension
   getting-started/c-sharp
   getting-started/featured-classes

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Class reference
   :glob:

   classes/class_*

Debugging behavior trees
~~~~~~~~~~~~~~~~~~~~~~~~

In Godot Engine, follow to "Bottom Panel > Debugger > LimboAI" tab. With the LimboAI debugger,
you can inspect any currently active behavior tree within the running project. The debugger can be detached
from the main editor window, which can be particularly useful if you have a HiDPI or a secondary display.
