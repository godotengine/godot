## A Docker image to build Linux, Windows and Android godot binaries.

The main reason to write this, is to provide a simple way in all platforms to integrate external godot modules and build a custom version of godot.

## usage
1. Install docker on Linux or docker toolbox on Windows or Mac.
2. Open a terminal on linux or "Docker Quickstart Terminal" on Windows or Mac.
3. Run command:
	- Linux: `cd`
	- Windows: `cd /c/Users/YOUR_USERNAME`
	- Mac: `cd /Users/YOUR_USERNAME`
4. Get godot source code: `git clone https://github.com/godotengine/godot.git`
5. Run command: `cd godot/tools/docker`
6. Run command: `docker build -t godot .`(In Linux run Docker commands with `sudo` or add your user to docker group before run the Docker commands). The godot docker image will be build after a while.
7. Run command:
	- Linux: `docker run -it --name=godot-dev -v /home/YOUR_USERNAME/godot:/godot-dev/godot godot`
	- Windows: `docker run -it --name=godot-dev -v /c/Users/YOUR_USERNAME/godot:/godot-dev/godot godot`
	- Mac: `docker run -it --name=godot-dev -v /Users/YOUR_USERNAME/godot:/godot-dev/godot godot`
	You are in the godot-dev container and /godot-dev directory now.
8. Run `./install-android-tools` to download and install all android development tools.
9. Run command: `source ~/.bashrc`
10. Run command: `cd godot`
11. Run command: `scons p=android target=release` to test everything is ok. You can set platform to x11, windows, android, haiku and server.

After use and exit, you can use this environment again by open terminal and type commands: `docker start godot-dev && docker attach godot-dev`.

### Windows and Mac stuffs:

- Speed up compilation:
	- Exit from container.
	- Run command: `docker-machine stop`
	- Open "Oracle VM VirtualBox".
	- In settings of default VM increase CPU cores and RAM to suitable values.
	- Run command: `docker-machine start`
	- Run command: `docker start godot-dev && docker attach godot-dev`

- ssh to VM(can be useful sometimes):
	- `docker-machine ssh`

Check docker and boot2docker projects for more details.
