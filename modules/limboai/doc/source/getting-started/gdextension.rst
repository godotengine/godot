.. _gdextension:

Using GDExtension
=================

    **🛈 See also:** `What is GDExtension? <https://docs.godotengine.org/en/stable/tutorials/scripting/gdextension/what_is_gdextension.html#what-is-gdextension>`_

LimboAI can be used as either a C++ module or as a GDExtension shared library.
The module version is the most feature-full and slightly more performant, but
it requires using custom engine builds including the export templates.

    **🛈 Note:** Precompiled builds are available on the official
    `LimboAI GitHub <https://github.com/limbonaut/limboai#getting-limboai>`_ page.

GDExtension version is more convenient to use, as you don't need a custom engine
build. You can simply download the extension and put it inside your project.
However, it has certain limitations, described in detail in the next section.

Whichever you choose to use, remember, your project will stay compatible with
both and you can transition from one to the other any time.


Limitations of the GDExtension version
--------------------------------------

GDExtension is the most convenient way of using the LimboAI plugin, but it comes
with certain limitations.

* Built-in documentation is not available. The plugin will open online documentation instead when requested.
* Documentation tooltips are not available.
* Handy :ref:`class_BBParam` property editor is not available in the extension due to dependencies on the engine classes that are not available in the Godot API.
