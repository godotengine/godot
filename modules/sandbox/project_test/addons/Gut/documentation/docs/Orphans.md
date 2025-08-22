# Orphans
GUT can display when your program or test generates orphaned nodes.  This can be very helpful when tracking down memory leaks in your application.  Note that GUT has no way to know if it was your program or your test that created the orphans.

This option is enabled by default.  You can disable it in the scene by unchecking "show orphans".  From the command line specify the `-ghide_orphans` option.

## Printing stray nodes
### Command Line
If you run your tests with the Godot `--verbose` flag from the command line then Godot will print out all the stray nodes and references at the end.  You can also print out just the stray nodes yourself if you use a [post-run script](Hooks.md).

```
extends GutHookScript

func run():
    # Note, this will node will be included in the stray node list.
    var n = Node.new()
    n.print_stray_nodes()
    n.free()
```
