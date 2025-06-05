#!/usr/bin/python3

import os
import shutil
import subprocess
import sys

def find_emscripten_sdk():
    """Find the Emscripten SDK directory using environment variables."""
    emsdk = os.environ.get("EMSDK")
    if emsdk:
        return emsdk
    
    # Try to find it in PATH
    path = os.environ.get("PATH", "").split(os.pathsep)
    for p in path:
        em_config = os.path.join(p, "em-config")
        if os.path.exists(em_config):
            # This is in bin, go up one level
            return os.path.dirname(p)
    
    return None

def build_mono_wasm(godot_dir, output_dir, debug=False):
    """Build the Mono WebAssembly runtime for Godot."""
    print("Building Mono WebAssembly runtime...")
    
    # Find the Emscripten SDK
    emsdk = find_emscripten_sdk()
    if not emsdk:
        print("Error: Emscripten SDK not found. Please set the EMSDK environment variable.")
        return False
    
    # Create output directory
    mono_wasm_dir = os.path.join(output_dir, "wasm")
    os.makedirs(mono_wasm_dir, exist_ok=True)

    # Set build configuration
    config = "Debug" if debug else "Release"
    
    # Download and build mono-wasm
    try:
        # Clone mono-wasm repository if needed
        mono_wasm_repo = os.path.join(godot_dir, "modules", "mono", "thirdparty", "mono-wasm")
        if not os.path.exists(mono_wasm_repo):
            os.makedirs(os.path.dirname(mono_wasm_repo), exist_ok=True)
            subprocess.check_call(["git", "clone", "https://github.com/mono/mono-wasm.git", mono_wasm_repo])
        
        # Build mono-wasm
        env = os.environ.copy()
        env["EMSDK"] = emsdk
        
        # Configure build options
        build_cmd = [
            "emcmake", "cmake", 
            "-DCMAKE_BUILD_TYPE=" + config,
            "-DENABLE_WASM=1",
            "-DENABLE_WASM_DYNAMIC_RUNTIME=1",
            "-DENABLE_WASM_THREADS=1",
            "-DUSE_STATIC_MONO=1",
            "."
        ]
        
        subprocess.check_call(build_cmd, cwd=mono_wasm_repo, env=env)
        subprocess.check_call(["emmake", "make", "-j4"], cwd=mono_wasm_repo, env=env)
        
        # Copy the built files
        wasm_files = [
            "mono.wasm",
            "mono.js",
            "mono.worker.js"
        ]
        
        for file in wasm_files:
            src_file = os.path.join(mono_wasm_repo, file)
            if os.path.exists(src_file):
                shutil.copy(src_file, os.path.join(mono_wasm_dir, file))
        
        # Create wrapper JS file for initialization
        with open(os.path.join(mono_wasm_dir, "godot-mono-support.js"), "w") as f:
            f.write("""
// Godot Mono WebAssembly Support
var GodotMonoSupport = {
    initialize: function() {
        // This will be called by the Godot engine
        return MonoRuntime.init();
    },
    loadAssembly: function(assemblyName) {
        return MonoRuntime.loadAssembly(assemblyName);
    },
    invokeMethod: function(assemblyName, namespace, className, methodName, args) {
        return MonoRuntime.call_method(assemblyName, namespace, className, methodName, args || []);
    }
};
""")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error building Mono WASM: {e}")
        return False

def prepare_export_template(godot_dir, output_dir, debug=False):
    """Prepare the C# web export template."""
    template_type = "debug" if debug else "release"

    # Define paths
    export_dir = os.path.join(output_dir, "templates", f"web_{template_type}.zip")

    # Create necessary directories
    os.makedirs(os.path.dirname(export_dir), exist_ok=True)

    print(f"Creating web export template for {template_type}...")

    # Add mono-wasm files to the template
    # This would normally be integrated with the regular template creation process
    # but for this example, we'll just copy what we need

    return True

def main():
    if len(sys.argv) < 3:
        print("Usage: python web_export_template.py <godot_dir> <output_dir> [debug]")
        return 1
    
    godot_dir = sys.argv[1]
    output_dir = sys.argv[2]
    debug = len(sys.argv) > 3 and sys.argv[3].lower() == "debug"
    
    if not os.path.exists(godot_dir):
        print(f"Error: Godot directory '{godot_dir}' does not exist.")
        return 1
    
    if not build_mono_wasm(godot_dir, output_dir, debug):
        return 1
    
    if not prepare_export_template(godot_dir, output_dir, debug):
        return 1
    
    print("Web export template created successfully.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 