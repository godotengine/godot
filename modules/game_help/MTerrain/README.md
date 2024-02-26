# godot M Terrain Plugin

![Screenshot_20230707_104154](https://github.com/mohsenph69/Godot-MTerrain-plugin/assets/52196206/7e3eb7da-af57-4ae5-8f55-f9fc1c8b26f8)


## Please read before using this plugin
Using this plugin require to learn some concept about terrain, This terrain plugin will not work out of the box, so I really suggest to read the [wiki](https://github.com/mohsenph69/Godot-MTerrain-plugin/wiki/) which I add recently added, I will add more stuff to wiki but for now I wrote the main concept that you need to know.

Also watching this video will be helpful:
https://www.youtube.com/watch?v=PcAkWClET4U

And then this video shows how to use use height brushes to modifying the terrain:
https://www.youtube.com/watch?v=e7nplXnemGo

Video about Texture painting:
https://www.youtube.com/watch?v=0zEYzKEMWR8

## patreon

You can support me with patreon [Click here](https://patreon.com/mohsenzare?utm_medium=clipboard_copy&utm_source=copyLink&utm_campaign=creatorshare_creator&utm_content=join_link)

## Features

* Tested with terrrain up to size 16km X 16 km
* Supporting grass system Even with collission for things like trees (Grass is paintable)
* Suppoting baking navigation system from terrain (navigation mesh is paintable)
* Terrain sculptur
* Color brush which support different algorithm (splatmapping, bitwise, index mapping ...)
  
![Screenshot_20230719_144752](https://github.com/mohsenph69/Godot-MTerrain-plugin/assets/52196206/704c51a8-7554-4345-907b-efc635a67dd0)


## download
To downalod the latest release use this link:
https://github.com/mohsenph69/Godot-MTerrain-plugin/releases

![Screenshot_20230719_144757](https://github.com/mohsenph69/Godot-MTerrain-plugin/assets/52196206/ef78652f-c4cc-4226-948e-9f4e44bb1af8)

## build by yourself
First clone this repo on your local machine, so you need godot-cpp to exist in GDExtension folder so you can build that, godot-cpp is added as a submodule in this project so to put that inside GDExtension folder only thing you need to do after cloning this repo is runing this code
```
git submodule update --init --recursive
```
This will automaticly pull godot-cpp into GDextension folder, After that go inside GDExtension folder and use scons to build this project
