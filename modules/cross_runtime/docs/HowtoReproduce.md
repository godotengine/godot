# How to Reproduce

## API Generation

All C++ and C# bindings are generated from Godot’s ClassDB to avoid manual duplication.

### Steps

**1. Get Godot for Linux (.NET edition)**

**2. Create a minimal dump project**

```text
godot_api_dump/project.godot:
[application]
config/name="API Dump"
```

**3. Generate `api.json`**

```bash
path/to/Godot_v4.x-stable_mono_linux_x86_64 \
  --headless \
  --path path/to/godot_api_dump \
  --script godot/modules/cross_runtime/api_generator/FindBindings/dump.gd > api.json
```

**4. Run the generator**

```bash
cd godot/modules/cross_runtime/api_generator
python3 Main.py path/to/api.json <destination_folder>
```

**5. Generated output**

```text
<destination>/
  command_dispatcher.cpp
  cpp/headers/bridge_api.h
  cpp/headers/bridge_helpers.h
  cpp/*               # generated C++ API files
  cs/Commands.cs
  cs/GodotApi/*       # generated C# API files
```

**6. Integrate the generated files**

Update `SCsub` to include the generated sources:

```python
Import("env")

cpp_sources = Glob("cpp/*.cpp")

env.add_source_files(env.modules_sources, [
    "register_types.cpp",
    "command_dispatcher.cpp",
    "command_processor.cpp",
] + cpp_sources)
```

Build Godot,
then build the .NET project after writing your .NET scripts. It is similar to Godotsharp API so it should be able to be built easily. This is an example:
```csharp
using System;
using System.Threading;
using System.Threading.Tasks;
using GodotWeb;

public static class TwoDogRunner
{

    public static async Task Run()
    {
        Helpers.ResetCommandBuffer();

        ulong sceneTreeId = Engine.get_singleton();

        Console.WriteLine($"SceneTree ID: {sceneTreeId}");
        if (sceneTreeId == 0)
        {
            Console.WriteLine("Failed to get SceneTree");
            return;
        }

        SceneTree tree = new SceneTree(sceneTreeId);

        // Everything below is exactly the same generated API
        ulong rootId = tree.get_root();
        if (rootId == 0)
        {
            Console.WriteLine("get_root returned 0");
            return;
        }

        Node root = new Node(rootId);

        ulong labelId = root.find_child("TargetLabel", true, false);
        if (labelId == 0)
        {
            Console.WriteLine("TargetLabel not found");
            return;
        }

        Label label = new Label(labelId);
        Console.WriteLine($"Label found, id = {label.Id}");
        Console.WriteLine("Entering the loop");

        int tick = 0;
        while (true)
        {
            tick++;
            label.set_text($"2dog running - tick {tick}");

        }
    }

}
```

- For a complete implementation of the API and a working demo, refer to https://github.com/tommygrammar/godot-cross-runtime-demo.
