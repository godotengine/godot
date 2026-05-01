# Reginleif Engine
Blah blah blah, insert boilerplate, you've seen this a ton of times. This fork was made because I was a little fed up with how absolutely
glacial Godot is at accepting PRs and how slow work was on GDScript. Now, don't get me wrong, I fucking love Godot, but I think GDScript 
is the absolute worst thing about it. But instead of adding a whole new language, this fork will focus on improving GDScript, because I'd
be fucking braindead if I said GDScript was completely useless. It is EXTREMELY good for rapid-fucking-iteration. So I don't want to give up
on this language yet.

## Who is this for?
certainly not for everyone! that's pretty intentional.

this fork is, first and foremost, for me, and my group of friends i hang out with and make games with.

but that's not enough to know if YOU, random stranger (perhaps) would be into this project. so let's talk about that.

this fork is for devs who LIKE gdscript's workflow, engine integration and rapid iteration cycles, but are at their limits (like i) by the limits of the static analysis and the type system. it is aimed at devs building larger, or perhaps longer term projects where compile-time guarantees matter more than maintaining beginner simplicity, who don't want to be told 'har har har har just use C#'

this fork will NOT aim to preserve GDScript as an extremely simplistic dynamic scripting language.

i strongly recommend turning on the static typing error in the project settings. you know which one, riiiight?

## Get started
Compile the engine by `git clone`ing this repository, use `pip install scons` to install `scons`, then compile by typing in just `scons`. Should take a little bit of time. 

If you don't want to go through the hassle of all that, I periodically throw a few prebuilt binaries in the releases section on the right. Ideally, there should be backwards compatibility with projects that use strongly typed GDScript, but back-compat is not the biggest prio for me, even if it *is* a priority.

## Shit I want to add
- type unifier
- traits (holy shit!!!)
- structs
- sum types
- errors as values
- exhaustive pattern matching

## Shit I added
- generics (currently only type-erased)
- nested types
- completely optional braces {} based scoping

## How to use the shit I added

### Generics

a feature that lets you specify the type of a variable with a placeholder, usually `T`, where that `T` is usually meant to be replaced later on with a 'concrete type'. 

currently, generics are TYPE-ERASED, which means they reduce to `Variant` in the actual runtime. all of the correctness heavywork is done by cranking static analysis up to the max. soon i intend to reify them, so correctness extends to runtime too.

#### Class-level Generics

declare a class level generic by stating it alongside the class_name identifier in your `class_name` declaration, wrapped in `[brackets]`

```gdscript
class_name Box[T]
```

and then use the generic parameter as a type anywhere in the class body:
```gdscript
class_name Box[T]

var val: T

func _init(arg: T) -> void:
	arg = val
```

multiple parameters are also allowed in the declaration, as such:

```gdscript
class_name Result[T, E]
```

now, you can instance a Box by using the constructor as such:

```gdscript
func _ready() -> void:
	var b := Box.new("waltuh...")
```
Static analysis magically infers that `b` is a `Box[String]`. Assigning `var x: Node = b.val` will give you a compile-time error. try it out!

you can set an "upper bound" on the generics, which are type constraints placed on that generic.
```gdscript
class_name Box[T: Node3D]
```
now you can no longer pass Node2Ds into Box, as T only accepts Node3Ds and its children:
```gdscript
var b := Box.new(AnimatedSprite2D.new())
#compile-time error!
```

to check if a generic is a certain concrete type, you can perform `if x is int:` style comparisons.
```gdscript
func balls(x: T) -> void:
	if x is int: print(x)
```

!!however!! because generics are type-erased as of now, `if x is Box[int]` gets erased to `if x is Box` during the runtime. Please match on the actual value like such: `if x is Box and x.val is int` for now. Yes, this is a problem, and I will address this soon.

#### Function-level generics

to declare function-level generics, you can list them after the function name identifier in `[brackets]`:
```gdscript
func balls[U, V](input: U) -> V:
```

these generics are usually infered via arguments that are passed in automagically.

however, in case you want to explicitly pass in the types for these generic parameters, you may use the
turbobrick `::[]` for this. some of you might be familiar with a similar construct from a certain language.

calling `balls()` with explicit types:
```gdscript
balls::[int, float](32)
```

in case your function only takes in only one function level generic, you may omit the `[brackets]`:
```gdscript
succ::int(32)
```

however, in 90% of cases, you will probably not need the turbobrick, as inference magics away the types for you.

### full nested typing

you remember how you could not make an `Array[Dictionary[StringName, Resource]]` in vanilla gdscript?
the reins are unlocked and you can go batshit crazy with your type tetris now:

```gdscript
var x: Array[Dictionary[Node2D, Array[Dictionary[Resource, StringName]]]]
#(please don't actually go that crazy for the sake of future you)
```

a caveat is that arrays/dicts that are typed to generics `[T]` don't have methods that take in `T`. they still want a `Variant` like in vanilla GDScript. of course, all these are planned to be addressed, but that's how it is for now.

### completely optional braces{} based scoping

This is now possible:
```gdscript
func _ready() -> void {
	#your code goes here
}
```

However, funnily enough, **this is not supported**:
```gdscript
func _ready() -> void
{
	#your code
}
```
this is because the `NEWLINE` token forces the parser into an ambiguous position that is very hard to solve without somehow breaking abstract functions. so just use KnR braces if you intend to use this feature lol

oh, yeah! you can skip the `pass` keyword when using braces for empty blocks. this is completely legal:
```gdscript
func _ready() -> void {}
```

A very unfortunate consequence of optional braces and then whitespace scoping being able to be mixed is that monstrous abominations straight out of Dante's Inferno are possible:
```gdscript
func _ready() -> void {
	var x := 3
	if x > 3 {
		print("6")
	} else:
		print("7")
}
```

have fuuun with that!


### Some more caveats
- Godot's `core` is rotten. Generics can LIE to you at runtime because static analysis is turned off for Variant-typed variables!!! (I didn't add this, this is Godot's default behaviour) Use static typing everywhere lest you want to run into undefined behaviour with generics.
- Do not inherit from a generic class, that behaviour is currently UNDEFINED because I haven't added it in yet lol sorry about that
- Static analysis for Dictionaries right now is not as good as Arrays, so that needs some work.
- I'm not fucking omniscient bro. There might be bugs, and I ask you to REPORT THEM!! catch my ass on discord at monarch_zero or open an issue here.


## Backwards compat breaks
Currently, this specific syntax is broken:
```gdscript
class InnerClass:
	extends Object
```
Instead, you should just 1-line this:
```gdscript
class InnerClass extends Object:
```

I didn't even know this was possible until making this project. But it looks so cursed I'm not fixing it just yet.


## Motivation

Why not contribute to Godot upstream itself? Well, I WANT to. But I have a few issues with that...

0. GDScript dev team and I will probably never see eye-to-eye. We are solar systems apart with our philosophy towards languages. In what way, you ask? Ah, let me ramble a bit. I generally believe that beginner-friendliness should not come at the cost of correctness or large-scale scalability. The language should *let you grow.* I also think that this difference in interests is fine, that's what the magic of open source is. 
1. I want traits, sum types, exhaustive matching, structs (and not in the "just make object leaner" way), PROPER ERROR HANDLING, tuples, stronger static guarantees, hell I'd even be delighted if GDScript dropped dynamic typing altogether, as I largely consider that to be a beginner trap. I am very glad that the Godot project is at least open-source, so that I may leech off of upstream and add my own changes. This project exists as a 'here you go!' for people who want the same rapid-iteration power GDScript is known for while allowing expression of stronger invariants, without bending knees to C#.
2. Godot PR review is GLACIAL. By glacial I do mean INSANELY GLACIAL. New features take YEARS to be accepted. The dev team actually just hates it when you touch `core`. I won't pretend they're evil and do it just because they're lazy or something (cough cough Redot), they have very real reasons to take as much time as they take, Godot being the backbone of millions of indie gamedevs around. but obviously I'm unhappy with the pace, so I'll go ahead implementing some of these myself.
3. I want the freedom to make mistakes with my PRs. I don't like C++ as a language and how much it relies to on me being completely fucking omniscient. Speaking of which...
4. If I had a Rust dependency (something the Godot team will never accept) later on like Cranelift to enable JITs, I want a platform to be able to do that.
5. I also just want to have fun adding silly little things! Professionalism is the antithesis of fun. I want the freedom to add a little life to the engine.
6. Lastly, I have an amazing group of gamedev friends who have a vested interest in this specific project. 

## Lmao why not c#
because i don't wanna. if you want to, good! go ahead.

## you should've waited for GDType/Big Core Rewrite/Godot 5/Weekly Steel Ball Run
i don't wanna. i prioritise usability now. i'm not saying GDType and shit are bad, i'm saying i'm an impatient kid.
