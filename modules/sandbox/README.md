<p align="center">
<img src="https://github.com/libriscv/godot-sandbox/blob/main/banner.png?raw=true" width="312px"/>
</p>
<p align="center">

<p align="center">
        <img src="https://github.com/libriscv/godot-sandbox/actions/workflows/runner.yml/badge.svg?branch=main"
            alt="Godot Sandbox Build"></a>
        <img src="https://img.shields.io/badge/Godot-4.3-%23478cbf?logo=godot-engine&logoColor=white" />
</p>

<p align = "center">
    <strong>
        <a href="https://libriscv.no">Website</a> | <a href="https://github.com/libriscv/godot-sandbox-programs">Code Examples</a> | <a href="https://discord.gg/n4GcXr66X5">Discord</a> | <a href="https://gonzerelli.itch.io/demo">Web Demo</a>
    </strong>
</p>


-----

<p align = "center">
<b>Safe, low-latency and fast sandbox</b>
<i>for the Godot game engine.</i>
</p>

<p align = "center">
	<strong>
  		<a href="https://github.com/user-attachments/files/17729740/Introducing.the.New.Godot.Sandbox.pdf">GodotCon 2024 Presentation</a>
	</strong>
</p>

-----

Godot Sandbox allows Godot creators to implement safe modding support, such that they can pass around programs built by other players, knowing that _restricted_ programs cannot harm other players. All Godot platforms are supported.


## Installation

- Automatic (Recommended): Download the plugin from the official [Godot Asset Store](https://godotengine.org/asset-library/asset/3192) using the **AssetLib** tab in Godot by searching for **Godot Sandbox**.

- Manual: Download the [latest github release](https://github.com/libriscv/godot-sandbox/releases/latest) and move only the **addons** folder into your project **addons** folder.

### What can I do?

- Modding API
	- You can implement a modding API for your game. This API can then be used by other players to extend your game, in a safe manner.
	- Put restrictions on resources, classes and objects to say what is accessible inside the sandbox.
- Build once, run everywhere
	- You can publish your game for all mobile and console platforms, without paying a performance penalty. It's not going to be laggy on some platforms, which is a risk with solutions that are only fast when JIT is available.
- High-performance
	- You can use this extension as a way to write higher performance code than GDScript permits, without having to resort to writing and maintaining a GDExtension for all platforms.
	- Enable full binary translation to increase performance drastically. Also works on all platforms, but has to be [embedded in the project or loaded as a DLL](https://libriscv.no/docs/godot_docs/bintr).
	- Yields [2.5-10x performance boost by default](https://libriscv.no/docs/performance/#godot-sandbox-performance), 5-50x with binary translation
	- Experimental JIT builds are also available in the Releases section for Windows and Linux. The JIT will enhance performance automatically in the background for all programs.
- Publish and then make updates without re-publishing
	- You can distribute programs from a server to clients as part of the login sequence. You can use this to live-distribute changes like bugfixes or even new features to the game without having to re-publish the game itself. I do this in my game.

## Usage

- Write C++ or Rust in the Godot editor. An accompanying ELF resource is created. This resource can be loaded into any Sandbox instance on every platform without recompiling.

- Create a new `Sandbox` and [assign the ELF resource to it](https://libriscv.no/docs/godot/sandbox/#create-a-sandbox)
	- Lifetime as any other node
	- Auto-completion from other GDScripts using @export

- Or, [directly assign an ELF script resource to a node](https://libriscv.no/docs/godot/sandbox/#using-programs-directly-as-scripts)
	- Shared sandbox among all instances with that script
	- Maximum scalability
	- Call functions and attach signals like GDScript

## Examples

```C++
#include "api.hpp"
static int coins = 0;

static Variant reset_game() {
	coins = 0;
	return Nil;
}

static inline void add_coin(const Node& player) {
	coins ++;
	// In our demo project we can access the coin label from the player
	// using a node path: Player -> get_parent() -> Texts -> CoinLabel
	Label coinlabel = player.get_node<Label>("../Texts/CoinLabel");
	coinlabel.set_text("You have collected "
		+ std::to_string(coins) + ((coins == 1) ? " coinyboys" : " coins"));
}

static Variant _on_body_entered(CharacterBody2D body) {
	if (body.get_name() != "Player")
		return Nil;

	get_node().queue_free(); // Remove the current coin!
	add_coin(body);
	return Nil;
}

static Variant _ready() {
	if (is_editor_hint()) {
		get_node().set_process_input(false);
	}
	return Nil; //
}

static Variant _process(double delta) {
	if (is_editor_hint()) {
		AnimatedSprite2D sprite("AnimatedSprite2D");
		sprite.play("idle");
		sprite.set_speed_scale(1.0f);
	}
	return Nil;
}

static Variant _input(InputEvent event) {
	if (event.is_action_pressed("jump")) {
		get_node<Node2D>().set_modulate(0xFF6060FF);
	} else if (event.is_action_released("jump")) {
		get_node<Node2D>().set_modulate(0xFFFFFFFF);
	}
	return Nil;
}

int main() {
	ADD_API_FUNCTION(_on_body_entered, "void", "CharacterBody2D");
	ADD_API_FUNCTION(_ready, "void");
	ADD_API_FUNCTION(_process, "void", "double");
	ADD_API_FUNCTION(_input, "void", "InputEvent");
	ADD_API_FUNCTION(reset_game, "void");
}
```

Script of a simple Coin pickup, with a counter that updates a label in the outer tree. This script can be attached to the Coin in the editor just like GDScript.

You may also have a look at our [demo repository](https://github.com/libriscv/godot-sandbox-demo) for the Godot Sandbox. It's a tiny platformer that uses C++ and Rust. There is also [a 3D plane demo](https://github.com/libriscv/godot_bird_plane) that enables controlling the plane from C++ scripts.

## Module Build

In order to build module, add it to a godot repo:

```
git submodule add https://github.com/libriscv/godot-sandbox modules/sandbox
cd modules/sandbox
git submodule update --init --recursive
```

## Contributing

Requirements:
- [SCons](https://www.scons.org)
- [python3](https://www.python.org)

If you want to contribute to this repo, here are steps on how to build locally:

```sh
./build.sh
```

You can also use `scons` similar to how godot-cpp addons are built.

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/fwsGonzo"><img src="https://avatars.githubusercontent.com/u/3758947?v=4?s=100" width="100px;" alt="Alf-André Walla"/><br /><sub><b>Alf-André Walla</b></sub></a><br /><a href="https://github.com/libriscv/godot-sandbox/commits?author=fwsGonzo" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://chibifire.com"><img src="https://avatars.githubusercontent.com/u/32321?v=4?s=100" width="100px;" alt="K. S. Ernest (iFire) Lee"/><br /><sub><b>K. S. Ernest (iFire) Lee</b></sub></a><br /><a href="https://github.com/libriscv/godot-sandbox/commits?author=fire" title="Code">💻</a> <a href="#research-fire" title="Research">🔬</a> <a href="https://github.com/libriscv/godot-sandbox/commits?author=fire" title="Tests">⚠️</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://appsinacup.com"><img src="https://avatars.githubusercontent.com/u/2369380?v=4?s=100" width="100px;" alt="Dragos Daian"/><br /><sub><b>Dragos Daian</b></sub></a><br /><a href="https://github.com/libriscv/godot-sandbox/commits?author=Ughuuu" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Other Projects

The [Jenova Framework](https://github.com/Jenova-Framework/) has a built-in C++ compiler, and supports writing C++ in the Godot editor with hot-reloading support.
