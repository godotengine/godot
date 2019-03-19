### install SailfishSDK 
1. Download and install [SailfishSDK](https://sailfishos.org/wiki/Application_SDK)
2. (optional) Install docker image, use [this instcruction](https://github.com/CODeRUS/docker-sailfishos-sdk-local) from @CODeRUS

### login to SailfishSDK (by docker or ssh)
**docker**
```sh
docker exec -it <sailfish_container_name_or_id> bash
```
or by **ssh**
```sh
ssh mersdk@localhost -p2222
```

#### Install dependencies in SailfishSDK targets
check list of targets
```sh
sb2-config -l
```

should output something like this 

```sh
SailfishOS-3.0.1.11-armv7hl
SailfishOS-3.0.1.11-i486
```

**armv7hl**
```sh
sb2 -t SailfishOS-3.0.1.11-armv7hl -R zypper in -y SDL2-debugsource SDL2-devel libaudioresource-devel pulseaudio-devel openssl-devel libwebp-devel libvpx-devel wayland-devel libpng-devel scons
```

**i486**
```sh
sb2 -t SailfishOS-3.0.1.11-i486 -R zypper in -y SDL2-debugsource SDL2-devel libaudioresource-devel pulseaudio-devel openssl-devel libwebp-devel libvpx-devel wayland-devel libpng-devel scons
```

Аfter installing **SDL2-debugsource** need little hack, because SDL2-debugsource have not SDL_internal.h file.  
Make new empty files:
```sh
sb2 -t SailfishOS-3.0.1.11-armv7hl -R touch /usr/src/debug/SDL2-2.0.3-1.3.2.jolla.arm/src/SDL_internal.h
sb2 -t SailfishOS-3.0.1.11-i486 -R touch /usr/src/debug/SDL2-2.0.3-1.3.2.jolla.i386/src/SDL_internal.h
```

### Build Godot export template for Sailfish OS
login to your **mersdk** or just use ssh

```sh
# building for arm platfrom
# /home/src1/godot - is you godot git checout dir, its maounted inside build 
# engine, in host system it could be something like 
# /home/developer/SailfishSDK/Projects/godot
ssh mersdk@localhost -p2222 "cd /home/src1/godot && sb2 -t SailfishOS-3.0.1.11-armv7hl scons arch=arm platform=sailfish tools=no bits=32 target=release\""  
# than do same for i486 platfrom
ssh mersdk@localhost -p2222 "cd /home/src1/godot && sb2 -t SailfishOS-3.0.1.11-i486 scons arch=x86 platform=sailfish tools=no bits=32 target=release\""  
```