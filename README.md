# scons wayland-scanner

A wayland-scanner builder for SCons.

## Usage

First, install it to `site_scons`:

    git clone https://git.sr.ht/~sircmpwn/scons-wayland-scanner site_scons/site_tools/wayland-scanner

Then add `wayland-scanner` to your Environment:

    environment = Environment(
        tools = ['wayland-scanner'],
    )

And use it like so:

    environment.WaylandScanner("client-header",
      "xdg-shell-client-protocol.h", "xdg-shell.xml")

For a full example, see `example/SConstruct`.
