# Reginleif Engine
Blah blah blah, insert boilerplate, you've seen this a ton of times. This fork was made because I was a little fed up with how absolutely
glacial Godot is at accepting PRs and how slow work was on GDScript. Now, don't get me wrong, I fucking love Godot, but I think GDScript 
is the absolute worst thing about it. But instead of adding a whole new language, this fork will focus on improving GDScript, because I'd
be fucking braindead if I said GDScript was completely useless. It is EXTREMELY good for rapid-fucking-iteration. So I don't want to give up
on this language yet.

Why not contribute to Godot itself? Well, I WANT to. But I have a few issues with that...
1. Godot PR review is GLACIAL. By glacial I do mean INSANELY GLACIAL. New features take YEARS to be accepted. The dev team actually just hates it when you touch `core`. I won't pretend they're evil and do it just because they're lazy or something (cough cough Redot), but obviously I'm unhappy with the pace, so I'll go ahead implementing some of these myself.
2. I want the freedom to make mistakes with my PRs. I don't like C++ as a language and how much it relies to on me being completely fucking omniscient. Speaking of which...
3. If I had a Rust dependency later on (something the Godot team will never accept), I want a platform to be able to do that.
4. I also just want to have fun adding silly little things! Professionalism is the antithesis of fun. I want the freedom to add a little life to the engine.
5. This fork will never fucking take off, so it'll be a little retreat for me and my friends. Yes this is just point 4 restated. I can do whatever I want and that's fun.

## Shit I want to add

- type unifier
- traits (holy shit!!!)

## Shit I added
- generics
- nested types

## Caveats
- Godot's `core` is rotten. Generics can LIE to you at runtime because static analysis is turned off for Variant-typed variables!!! (I didn't add this, this is Godot's default behaviour) Use static typing everywhere lest you want to run into undefined behaviour with generics.
- Do not inherit from a generic class, that behaviour is currently UNDEFINED because I haven't added it in yet lol sorry about that
- Nested types can ALSO lie to you because the runtime is genuinely unaware of what goes inside a nested type. MAKE SURE TO USE STATIC TYPING EVERYWHERE, because at least STATIC ANALYSIS is aware of types!! (i added that in, heh)
- I'm not fucking omniscient bro. There might be bugs, and I ask you to REPORT THEM!! catch my ass on discord at monarch_zero or open an issue here.


## call for help
I hate c++ but my spite is greater. contribute and I'll be in your eternal debt 🙏

## clarification
i don't hate the godot dev team, i fucking love them!!!
i just don't like the whole 'approachability over correctness' philosophy that is used to explain away the absence of BASIC fucking language features. that's it. 
