![gut logo](images/gut_logo_256x256.png)


If you'd like to support me you can [buy me a coffee](https://buymeacoffee.com/bitwes).

# GUT
GUT (Godot Unit Test) is a unit testing framework for the [Godot Engine](https://godotengine.org/).  It allows you to write tests for your gdscript in gdscript.

GUT versions 9.x are for Godot 4.x.<br>
GUT versions 7.x (currently 7.4.2) are for Godot 3.x.




# Documentation
Documentation is hosted at https://gut.readthedocs.io/
* [Latest](https://gut.readthedocs.io/en/latest)
* [GUT 9.x for Godot 4](https://gut.readthedocs.io/en/v9.4.0/Quick-Start.html)
* [GUT 7.x for Godot 3](https://gut.readthedocs.io/en/v7.4.2/Quick-Start.html)




# Version Links
There are only two versions of GUT in the Asset Library (one for Godot 3 and one for Godot 4).  GUT will not appear in the Asset Library if your current version of Godot is less than GUT's required version.  Here's the highest version of GUT per Godot version.

|GUT|Required Godot<br>Version||||
|-|-|-|-|-|
|[godot_4_5 branch](https://github.com/bitwes/Gut/tree/godot_4_5) | 4.5| |
|Main Branch |4.4| | |
|[9.4.0](https://github.com/bitwes/Gut/releases/tag/v9.4.0)|4.3|[repo](https://github.com/bitwes/Gut/tree/v9.4.0)|[zip](https://github.com/bitwes/Gut/archive/refs/tags/v9.4.0.zip)|[Asset Library](https://godotengine.org/asset-library/asset/1709)|
|[9.3.0](https://github.com/bitwes/Gut/releases/tag/v9.3.0)|4.2|[repo](https://github.com/bitwes/Gut/tree/v9.3.0)|[zip](https://github.com/bitwes/Gut/archive/refs/tags/v9.3.0.zip)||
|[9.1.1](https://github.com/bitwes/Gut/releases/tag/v9.1.1)|4.1|[repo](https://github.com/bitwes/Gut/tree/v9.1.1)|[zip](https://github.com/bitwes/Gut/archive/refs/tags/v9.1.1.zip)||
|[9.0.1](https://github.com/bitwes/Gut/releases/tag/v9.0.1)|4.0|[repo](https://github.com/bitwes/Gut/tree/v9.0.1)|[zip](https://github.com/bitwes/Gut/archive/refs/tags/v9.0.1.zip)||
|[7.4.3](https://github.com/bitwes/Gut/releases/tag/v7.4.3)|3.5|[repo](https://github.com/bitwes/Gut/tree/v7.4.3)|[zip](https://github.com/bitwes/Gut/archive/refs/tags/v7.4.3.zip)|[Asset Library](https://godotengine.org/asset-library/asset/54)|

To install from the zip link:
* Download the zip and extract it
* Put the `addons/gut` directory into your project.
* Enable the GUT plugin.

You will need to relaunch Godot.




# Features
* [Simple install](https://gut.readthedocs.io/en/latest/Install.html) via the Asset Library
* A plethora of [asserts and utility methods](https://gut.readthedocs.io/en/latest/Asserts-and-Methods.html) to help make your tests simple and concise
* Support for [Inner Test Classes](https://gut.readthedocs.io/en/latest/Inner-Test-Classes.html) to give your tests some extra context and maintainability
* Doubling:  [Full](https://gut.readthedocs.io/en/latest/Doubles.html) and [Partial](https://gut.readthedocs.io/en/latest/Partial-Doubles.html), [Stubbing](https://gut.readthedocs.io/en/latest/Stubbing.html), [Spies](https://gut.readthedocs.io/en/latest/Spies.html)
* Command Line Interface [(CLI)](https://gut.readthedocs.io/en/latest/Command-Line.html)
* [Parameterized Tests](https://gut.readthedocs.io/en/latest/Parameterized-Tests.html)
* [Export results](https://gut.readthedocs.io/en/latest/Export-Test-Results.html) in standard JUnit XML format

![Panel](https://gut.readthedocs.io/en/latest/_images/gut_panel.png)




# Getting Started
* [Install](https://gut.readthedocs.io/en/latest/Install.html)
* [Quick Start](https://gut.readthedocs.io/en/latest/Quick-Start.html)
* [Creating Tests](https://gut.readthedocs.io/en/latest/Creating-Tests.html)
* [Asserts and Methods](https://gut.readthedocs.io/en/latest/Asserts-and-Methods.html)




# VSCode Extension
Run your tests directly from the VSCode Editor.  Search VSCode extensions for "gut-extension".  The plugin has commands to run your entire test suite, a single test script or a single test.
* [Visual Studio Marketplace](https://marketplace.visualstudio.com/items?itemName=bitwes.gut-extension)
* [Github repo](https://github.com/bitwes/gut-extension)
* [Quick tutorial on setup and use](https://youtu.be/pqcA8A52CMs)


Thanks for using GUT.  If you'd like to support me you can [buy me a coffee](https://buymeacoffee.com/bitwes).

# License
Gut is provided under the MIT license.  License is in `addons/gut/LICENSE.md`.
