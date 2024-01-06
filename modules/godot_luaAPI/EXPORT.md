Godot Lua API Installing Export Templates
===============

<!-- TOC -->
* [Godot Lua API Installing Export Templates](#godot-lua-api-installing-export-templates)
  * [Notes](#notes)
  * [How To Use](#how-to-use)
<!-- TOC -->

Notes
-------
You will need to download and install the templates from the releases, every time you update to a new release.
To do so, go to the [Releases](https://github.com/WeaselGames/godot_luaAPI/releases) section on the GitHub repository and
download the appropriate Export Templates package for your build. For example, if you are using the `v2.0.2-stable` 
release (current as of the writing of this document), you will want to download `export-templates.tpz` from the `Assets` 
section. This file, and its extracted contents will be discussed in this document.
If you are using the DotNet (C#) Mono build, then you will want to download the `export-templates-mono.zip`. Go ahead and 
download these now, if you have not done so already.

How To Use
-------

Now on how to use them, under releases we have 2 export template options. Use `export-templates` for the normal engine 
(GDScript only) builds without mono. These support all platforms. Then `export-templates-mono` is of course for Dotnet / 
Mono. Since Godot C# has low platform support right now, these templates only support Linux, Windows and MacOS.

Inside the zip files you will see executables for each platform and architecture that is supported. There will be 2 for 
each supported combination, Release and Debug. Debug templates are to allow you to debug the exported project using 
the Godot debugging tools. The Release version has that all of the debugging code stripped out to optimize the final 
Release build.

Here is where to find the Project Export profiles:
![projexpmenu.png](.github%2FDocumentationImages%2Fprojexpmenu.png)

You will want to either update your existing Project Export profiles, or make new ones. When creating or updating the 
export profile in the editor, You will see where it has you select the platform and architecture. To make use of our 
templates you need to set the file path for both the debug and release templates in the fields provided. Those fields 
are labeled Debug and Release in the Custom Templates section. (See image.)

![exportwin.png](.github%2FDocumentationImages%2Fexportwin.png)

Use the folder icon next to the two fields to select the template that matches with the `Architecture` and the 
`Custom Template` selection. In the case of what is shown in the image, we are using `Linux` and `x86_64`. So, we will 
select the location and file, based on where you extracted the the templates zip file that you downloaded above. 
(See Notes.) In this case, we will be using `linux_debug.x86_64` and `linux_release.x86_64`. 

You can now export your project or close the window. (The Project Export settings are saved automatically.) 
