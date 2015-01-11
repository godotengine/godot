### x11-window-management branch

#### New GDScript Methods for the OS Class:
* int OS.get_screen_count()
* Vector2 OS.get_screen_size(int screen=0)
* Vector2 OS.get_window_position()
* void OS.set_window_position(Vector2 position)
* Vector2 OS.get_window_size()
* void OS.set_window_size(Vector2 size)
* void OS.set_fullscreen(bool enabled, int screen=0)
* bool OS.is_fullscreen()

#### Demo
A demo/test is available at "demos/misc/window-management"

#### Warning
Just only works for X11. It breaks other platforms at the moment.

![GODOT](/logo.png)

### The Engine

Godot is a fully featured, open source, MIT licensed, game engine. It focuses on having great tools, and a visual oriented workflow that can export to PC, Mobile and Web platforms with no hassle.
The editor, language and APIs are feature rich, yet simple to learn, allowing you to become productive in a matter of hours.

### About

Godot has been developed by Juan Linietsky and Ariel Manzur for several years, and was born as an in-house engine, used to publish several work-for-hire titles.
Development is sponsored by OKAM Studio (http://www.okamstudio.com).

### Documentation

Documentation has been moved to the [GitHub Wiki](https://github.com/okamstudio/godot/wiki).

### Binary Downloads, Community, etc.

Binary downloads, community, etc. can be found in Godot homepage:

http://www.godotengine.org

### Compiling from Source

Compilation instructions for every platform can be found in the Wiki:
https://github.com/okamstudio/godot/wiki/advanced
