# Fish completion for the Godot editor
# To use it, install this file in `~/.config/fish/completions` then restart your shell.
# You can also `source` this file directly in your shell startup file.
#
# Copyright (c) 2014-present Godot Engine contributors (see AUTHORS.md).
# Copyright (c) 2007-2014 Juan Linietsky, Ariel Manzur.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

function godot_video_driver_args
    # Use a function instead of a fixed string to customize the argument descriptions.
    echo -e "GLES3\tOpenGL ES 2.0 renderer"
    echo -e "GLES2\tOpenGL ES 2.0 renderer"
end

# Erase existing completions for Godot.
complete -c godot -e

# General options:
complete -c godot -s h -l help -d "Display the full help message"
complete -c godot -l version -d "Display the version string"
complete -c godot -s v -l verbose -d "Use verbose stdout mode"
complete -c godot -l quiet -d "Quiet mode, silences stdout messages (errors are still displayed)"

# Run options:
complete -c godot -s e -l editor -d "Start the editor instead of running the scene"
complete -c godot -s p -l project-manager -d "Start the project manager, even if a project is auto-detected"
complete -c godot -s q -l quit -d "Quit after the first iteration"
complete -c godot -s l -l language -d "Use a specific locale (<locale> being a two-letter code)" -x
complete -c godot -l path -d "Path to a project (<directory> must contain a 'project.godot' file)" -r
complete -c godot -s u -l upwards -d "Scan folders upwards for project.godot file"
complete -c godot -l main-pack -d "Path to a pack (.pck) file to load" -r
complete -c godot -l render-thread -d "Set the render thread mode" -x -a "unsafe safe separate"
complete -c godot -l remote-fs -d "Use a remote filesystem (<host/IP>[:<port>] address)" -x
complete -c godot -l remote-fs-password -d "Password for remote filesystem" -x
complete -c godot -l audio-driver -d "Set the audio driver" -x
complete -c godot -l video-driver -d "Set the video driver" -x -a "(godot_video_driver_args)"

# Display options:
complete -c godot -s f -l fullscreen -d "Request fullscreen mode"
complete -c godot -s m -l maximized -d "Request a maximized window"
complete -c godot -s w -l windowed -d "Request windowed mode"
complete -c godot -s t -l always-on-top -d "Request an always-on-top window"
complete -c godot -l resolution -d "Request window resolution" -x
complete -c godot -l position -d "Request window position" -x
complete -c godot -l low-dpi -d "Force low-DPI mode (macOS and Windows only)"
complete -c godot -l no-window -d "Disable window creation (Windows only), useful together with --script"
complete -c godot -l enable-vsync-via-compositor -d "When Vsync is enabled, Vsync via the OS' window compositor (Windows only)"
complete -c godot -l disable-vsync-via-compositor -d "Disable Vsync via the OS' window compositor (Windows only)"

# Debug options:
complete -c godot -s d -l debug -d "Debug (local stdout debugger)"
complete -c godot -s b -l breakpoints -d "Specify the breakpoint list as source::line comma-separated pairs, no spaces (use %20 instead)" -x
complete -c godot -l profiling -d "Enable profiling in the script debugger"
complete -c godot -l remote-debug -d "Enable remote debugging"
complete -c godot -l debug-collisions -d "Show collision shapes when running the scene"
complete -c godot -l debug-navigation -d "Show navigation polygons when running the scene"
complete -c godot -l frame-delay -d "Simulate high CPU load (delay each frame by the given number of milliseconds)" -x
complete -c godot -l time-scale -d "Force time scale (higher values are faster, 1.0 is normal speed)" -x
complete -c godot -l disable-render-loop -d "Disable render loop so rendering only occurs when called explicitly from script"
complete -c godot -l disable-crash-handler -d "Disable crash handler when supported by the platform code"
complete -c godot -l fixed-fps -d "Force a fixed number of frames per second (this setting disables real-time synchronization)" -x
complete -c godot -l print-fps -d "Print the frames per second to the stdout"

# Standalone tools:
complete -c godot -s s -l script -d "Run a script" -r
complete -c godot -l check-only -d "Only parse for errors and quit (use with --script)"
complete -c godot -l export -d "Export the project using the given preset and matching release template" -x
complete -c godot -l export-debug -d "Same as --export, but using the debug template" -x
complete -c godot -l export-pack -d "Same as --export, but only export the game pack for the given preset" -x
complete -c godot -l doctool -d "Dump the engine API reference to the given path in XML format, merging if existing files are found" -r
complete -c godot -l no-docbase -d "Disallow dumping the base types (used with --doctool)"
complete -c godot -l build-solutions -d "Build the scripting solutions (e.g. for C# projects)"
complete -c godot -l gdnative-generate-json-api -d "Generate JSON dump of the Godot API for GDNative bindings"
complete -c godot -l test -d "Run a unit test" -x
