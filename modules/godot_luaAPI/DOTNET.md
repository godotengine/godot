Godot Lua API DotNet Notes
===============

<!-- TOC -->
* [Godot Lua API DotNet Notes](#godot-lua-api-dotnet-notes)
  * [Introduction](#introduction)
  * [Commands](#commands)
  * [Getting Started Example (In C#)](#getting-started-example-in-c)
<!-- TOC -->

Introduction
-------

When working with this version of the editor, or the extension addon, you will need to use specific versions of the 
nuget packages for your project. This has been tested with v2.1-beta6.

For this document, we will use *nix style paths. If you are a Windows user, please convert lines like `/path/to/` to 
`C:\path\to\` so that it will work with your system. MacOS users can use the *nix paths.

The editor builds have a file named `mono_instructions.txt` included in the zipped release.
The contents of this file are as follows:

```
If you have GodotSharp locally cached dotnet may use one of them instead. To clear local cache run the following command:
dotnet nuget locals all --clear

To add the local nuget source, please run the following command:
dotnet nuget add source /path/to/nuget_packages/ --name LuaAPINugetSource
```

What this is saying is that you will want to open up a command prompt and do the following steps to install the correct 
nuget packages into your project. If you have downloaded the editor zip file, it comes with a directory named `nuget_packages` 
This is the directory that you will want to include into the second step in the instructions file. So, if you have extracted 
the Editor to `C:\Users\username\Godot` the correct path to use would be `C:\Users\username\Godot\nuget_packages`. 
Likewise, if you extracted the editor to `/home/username/Godot/` you would want to use `/home/username/Godot/nuget_packages`.

Make sure to enclose directories with spaces with either a `"` or `'` so that the path resolves correctly.

Additionally, you will most likely want to execute a `dotnet restore` command to install the correct nuget packages. See 
Commands below for an example on how to do this.

Commands
-------

In your command prompt, execute these commands in the following order.

`dotnet nuget locals all --clear`

`dotnet nuget add source /path/to/nuget_packages --name LuaAPINugetSource`

`dotnet restore '/pathtoproject/example_project.csproj' -f -s  LuaAPINugetSource`

This will set up the proper packages to work with the Editor / Add-on. Note: you may have to select the correct nuget 
source within your IDE. If so, please use the `LuaAPINugetSource` option. Note that in the third command, we are using 
the specific location (`-s <source>`) and we are forcing (`-f`) the restore. This is done to specifically use the custom 
nuget packages.

Once you have done this, you will need to rebuild your project. You can do so either through your IDE or inside of the 
Godot Editor. I highly recommend keeping this section handy, as you will need to use these for each new project 
that you create. If you can, it's advisable to change the nuget sources so that the local source is first in the 
list. This will make life easier. And then, with new projects, run the clear command, then restore the project in 
your IDE, or simply build the project in Godot. (Godot will perform a restore in the build process.) By having the 
local sources first in the list, this will ensure that they are put in, and that your project will work.

A note on the LuaAPI specific nuget packages: They are included in the Mono (DotNet) builds from the `Releases` tab on 
the Github page.

Additional help with the Nuget Packages installation:
In some cases the packages will fail to restore, and if that happens to you, this is something that you can do to try 
to make it work. You will need to remove the existing `LuaAPINugetSource` that you made above, and then put this file 
in your project directory. As it uses an absolute path, others will need to change it to their location, if they are 
part of your team. (Like an open source project, or if they are compiling the code themselves.)
* Create a nuget.config file in the root of your project or solution (if it doesn't already exist).
* Open or create the nuget.config file, and add the following:

```xml
    <?xml version="1.0" encoding="utf-8"?>
    <configuration>
      <packageSources>
        <add key="LuaAPINugetSource" value="/path/to/editor/editor-mono/nuget_packages" />
        <!-- Add other package sources if needed -->
      </packageSources>
    </configuration>
```

* Replace `/path/to/editor/editor-mono/nuget_packages` with the correct path to your local NuGet source. Note the 
lack of a trailing slash. 
* Save the nuget.config file.

With this configuration, the NuGet restore process will automatically consider the sources listed in the nuget.config. 
Note that windows users may have other issues with dotnet, files being marked unsafe because they originated from other 
computers, not being done as an administrator, etc. Sadly, those are on the user to fix as it is beyond the scope of a 
getting started file.

Getting Started Example (In C#)
-------

As you can expect, the method names in the LuaAPI class are in Dotnet style (Camel Case). So, something like 
`LuaAPI.push_variant` would be `LuaAPI.PushVariant`

In this example, we recreate the main GDScript example from the README.md file. I have fully commented this
example, to explain what each part does. Specific differences and a special note on this example: In C# you
cannot assign a Method to a `Variant` for use with `LuaAPI.PushVariant`. So, to get around that, and make the
example work, we will first wrap our called function in a `Callable` variable (wrapper). Please note the line
`Callable print = new Callable(this, MethodName.LuaPrint);` in the example below.

This example also shows how to use the newly added `LuaFunctionRef` class to `invoke` a lua function. This allows 
for creating a call back function to respond to things like events. Note how we use `.As<LuaFunctionRef>()` to cast 
the Variant `val` into a `LuaFunctionRef` variable (`get_message`). Additionally, we do not need to use the nullable 
invoke style common to most event firing, as the `get_message` variable is not null. The `.Invoke()` method takes a 
`Godot.Collections.Array` as its only parameter. In this example, we make use of the shorthand to create a `new` object 
for the Godot Array class. This shorthand is only possible because of the way that Godot defines the `Array` class, and 
is not a normal C# `new Object()` creation style.

```csharp
using Godot;

public partial class Node2D : Godot.Node2D {
	private LuaApi lua = new LuaApi();

	public void LuaPrint(string message) {
		GD.Print(message);
	}

	public override void _Ready() {
		GD.Print("Starting Node2D.cs.");

		// All builtin libraries are available to bind with. Use OS and IO at your own risk.
		// BindLibraries requires a "Godot Array" so, let's build one.
		Godot.Collections.Array libraries = new() {
			"base",  // Base Lua commands
			"table", // Table functionality.
			"string" // String Specific functionality.
		};
		lua.BindLibraries(libraries); // Assign the specified libraries to the LuaAPI object.

		// In C#, .PushVariant does not work with Methods, so we use Callable to wrap our function.
		Callable print = new Callable(this, MethodName.LuaPrint);
		// Assign the Callable, so that the API can call our function.
		// Note, the lua function "cs_print" is now callable within Lua script.
		lua.PushVariant("cs_print", print);
		// Assign a Lua Variable named "message" and give it a value.
		lua.PushVariant("message", "Hello lua!");

		// Use .DoString() to execute our Lua code.
		LuaError error = lua.DoString("cs_print(message)");
		// Check for errors, and if there are any, Print them to the Godot Console.
		if (error != null && error.Message != "") {
			GD.Print("An error occurred calling DoString.");
			GD.Print("ERROR %d: %s", error.Type, error.Message);
		}

		error = lua.DoString(@"
                                  for i=1,10,1 do
                                  	cs_print(message)
                                  end
                                  function get_message()
                                  	return ""This message was sent from 'get_message()'""
                                  end
                                  ");

		// Check for errors, and if there are any, Print them to the Godot Console.
		if (error != null && error.Message != "") {
			GD.Print("An error occurred calling DoString.");
			GD.Print("ERROR %d: %s", error.Type, error.Message);
		}
		
		// Let's pull our lua function from the lua code.
		var val = lua.PullVariant("get_message");
		// Check to see if it returned an error, or a value.
		if (val.GetType() == typeof(LuaError)) {
			GD.Print("ERROR %d: %s", error.Type, error.Message);
			return;
		}

		// To use LuaFunctionRefs we need to change the system to use it. We do this by changing
		// the .UseCallables flag to 'false'. (If your LuaFunctionRef variable is null, you didn't
		// set this flag. 
		lua.UseCallables = false;

		// We create a LuaFunctionRef as our reference to the Lua code's function,
		// then we use .As<LuaFunctionRef>() to cast it as a LuaFunctionRef.
		LuaFunctionRef get_message = val.As<LuaFunctionRef>();
		if (get_message == null) {
			GD.Print("ERROR: get_message is null.");
			return;
		}

		// Calling Lua (code) functions requires a Godot.Collections.Array as the container
		// for the parameters passed in. 
		Godot.Collections.Array Params = new();
		// We use .Invoke to actually call the lua function within the Lua State. 
		// And, finally, we log the output of the function to Godot Output Console.
		GD.Print(get_message.Invoke(Params));

	}
}
```
Note that `lua.DoString()` is the dotnet version of `lua.do_string()`.

More examples will be included very soon (see Wiki).