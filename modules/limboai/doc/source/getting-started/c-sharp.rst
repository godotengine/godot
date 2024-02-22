.. _csharp:

Using LimboAI with C#
=====================

Here's how you can use the module version with C# to write your tasks and states.

1. Locate LimboAI NuGet package files.

Each provided build comes with a GodotSharp folder, which has packages under
"GodotSharp/Tools/nupkgs/". Note down the directory path; you'll need it in the next step.

2. Add a local source for NuGet packages using the following command:

.. code:: shell

    dotnet nuget add source path/to/limboai/nupkgs --name LimboNugetSource

3. Your C# project should be able to see LimboAI classes and compile.

Regarding GDExtension, I can only confirm success with the module version and C#.
Unfortunately, I haven't had any luck with the GDExtension version yet.
If you've had success with GDExtension, please let me know via Discord or email.

Creating custom tasks in C#
---------------------------

   **🛈 Note:** For more information, check out :ref:`creating custom tasks in GDScript <custom_tasks>`.

You can use the following script template for custom tasks:

.. code:: csharp

    using Godot;
    using System;

    [Tool]
    public partial class _CLASS_ : _BASE_
    {
        public override string _GenerateName()
        {
            return "_CLASS_";
        }

        public override void _Setup()
        {
        }

        public override void _Enter()
        {
        }

        public override void _Exit()
        {
        }

        public override Status _Tick(double delta)
        {
            return Status.Success;
        }

        public override string[] _GetConfigurationWarnings()
        {
            return Array.Empty<string>();
        }
    }

