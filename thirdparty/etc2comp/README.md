# Etc2Comp - Texture to ETC2 compressor

Etc2Comp is a command line tool that converts textures (e.g. bitmaps)
into the [ETC2](https://en.wikipedia.org/wiki/Ericsson_Texture_Compression)
format. The tool is built with a focus on encoding performance
to reduce the amount of time required to compile asset heavy applications as
well as reduce overall application size.

This repo provides source code that can be compiled into a binary. The
binary can then be used to convert textures to the ETC2 format.

Important: This is not an official Google product. It is an experimental
library published as-is. Please see the CONTRIBUTORS.md file for information
about questions or issues.

## Setup
This project uses [CMake](https://cmake.org/) to generate platform-specific
build files:
 - Linux: make files
 - OS X: Xcode workspace files
 - Microsoft Windows: Visual Studio solution files
 - Note: CMake supports other formats, but this doc only provides steps for
 one of each platform for brevity.

Refer to each platform's setup section to setup your environment and build
an Etc2Comp binary. Then skip to the usage section of this page for examples
of how to use the library.

### Setup for OS X
 build tested on this config:
  OS X 10.9.5 i7 16GB RAM
  Xcode 5.1.1
  cmake 3.2.3
  
Start by downloading and installing the following components if they are not
already installed on your development machine.
 - *Xcode* version 5.1.1, or greater
 - [CMake](https://cmake.org/download/) version 3.2.3, or greater

To build the Etc2Comp binary:
 1. Open a *Terminal* window and navigate to the project directory.
 1. Run `mkdir build_xcode`
 1. Run `cd build_xcode`
 1. Run `cmake -G Xcode ../`
 1. Open *Xcode* and import the `build_xcode/EtcTest.xcodeproj` file.
 1. Open the Product menu and choose Build For -> Running.
 1. Once the build succeeds the binary located at `build_xcode/EtcTool/Debug/EtcTool`
can be executed.

Optional
Xcode EtcTool ‘Run’ preferences
note: if the build_xcode/EtcTest.xcodeproj is manually deleted then some Xcode preferences 
will need to be set by hand after cmake is run (these prefs are retained across 
cmake updates if the .xcodeproj is not deleted/removed)

1. Set the active scheme to ‘EtcTool’
1. Edit the scheme
1. Select option ‘Run EtcTool’, then tab ‘Arguments’. 
Add this launch argument: ‘-argfile ../../EtcTool/args.txt’
1. Select tab ‘Options’ and set a custom working directory to: ‘$(SRCROOT)/Build_Xcode/EtcTool’

### SetUp for Windows

1. Open a *Terminal* window and navigate to the project directory.
1. Run `mkdir build_vs`
1. Run `cd build_vs`
1. Run CMAKE, noting what build version you need, and pointing to the parent directory as the source root; 
  For VS 2013 : `cmake -G "Visual Studio 12 2013 Win64" ../`
  For VS 2015 : `cmake -G "Visual Studio 14 2015 Win64" ../`
  NOTE: To see what supported Visual Studio outputs there are, run `cmake -G`
1. open the 'EtcTest' solution
1. make the 'EtcTool' project the start up project 
1. (optional) in the project properties, under 'Debugging ->command arguments' 
add the argfile textfile thats included in the EtcTool directory. 
example: -argfile C:\etc2\EtcTool\Args.txt

### Setup For Linux
The Linux build was tested on this config:
  Ubuntu desktop 14.04
  gcc/g++ 4.8
  cmake 2.8.12.2

1. Verify linux has cmake and C++-11 capable g++ installed
1. Open shell
1. Run `mkdir build_linux`
1. Run `cd build_linux`
1. Run `cmake ../`
1. Run `make`
1. navigate to the newly created EtcTool directory `cd EtcTool`
1. run the executable: `./EtcTool -argfile ../../EtcTool/args.txt`

Skip to the <a href="#usage">Usage</a> section for more information about using the
tool.

## Usage

### Command Line Usage
EtcTool can be run from the command line with the following usage:
    etctool.exe source_image [options ...] -output encoded_image

The encoder will use an array of RGBA floats read from the source_image to create 
an ETC1 or ETC2 encoded image in encoded_image.  The RGBA floats should be in the 
range [0:1].

Options:

    -analyze <analysis_folder>
    -argfile <arg_file>           additional command line arguments read from a file
    -blockAtHV <H V>              encodes a single block that contains the
                                  pixel specified by the H V coordinates
    -compare <comparison_image>   compares source_image to comparison_image
    -effort <amount>              number between 0 and 100 to specify the encoding quality 
                                  (100 is the highest quality)
    -errormetric <error_metric>   specify the error metric, the options are
                                  rgba, rgbx, rec709, numeric and normalxyz
    -format <etc_format>          ETC1, RGB8, SRGB8, RGBA8, SRGB8, RGB8A1,
                                  SRGB8A1 or R11
    -help                         prints this message
    -jobs or -j <thread_count>    specifies the number of threads (default=1)
    -normalizexyz                 normalize RGB to have a length of 1
    -verbose or -v                shows status information during the encoding
                                  process
	-mipmaps or -m <mip_count>    sets the maximum number of mipaps to generate (default=1)
	-mipwrap or -w <x|y|xy>       sets the mipmap filter wrap mode (default=clamp)

* -analyze will run an analysis of the encoding and place it in folder 
"analysis_folder" (e.g. ../analysis/kodim05).  within the analysis_folder, a folder 
will be created with a name of the current date/time (e.g. 20151204_153306).  this 
date/time folder is used to compare encodings of the same texture over time.  
within the date/time folder is a text file with several encoding stats and a 2x png 
image showing the encoding mode for each 4x4 block.

* -argfile allows additional command line arguments to be placed in a text file

* -blockAtHV selects the 4x4 pixel subset of the source image at position (H,V).  
This is mainly used for debugging

* -compare compares the source image to the created encoded image. The encoding
will dictate what error analysis is used in the comparison.

* -effort uses an "amount" between 0 and 100 to determine how much additional effort 
to apply during the encoding.

* -errormetric selects the fitting algorithm used by the encoder.  "rgba" calculates 
RMS error using RGB components that are weighted by A.  "rgbx" calculates RMS error 
using RGBA components, where A is treated as an additional data channel, instead of 
as alpha.  "rec709" is similar to "rgba", except the RGB components are also weighted 
according to Rec709.  "numeric" calculates RMS error using unweighted RGBA components.  
"normalize" calculates error based on dot product and vector length for RGB and RMS 
error for A.

* -help prints out the usage message

* -jobs enables multi-threading to speed up image encoding

* -normalizexyz normalizes the source RGB to have a length of 1.

* -verbose shows information on the current encoding process. It will then display the 
PSNR and time time it took to encode the image.

* -mipmaps takes an argument that specifies how many mipmaps to generate from the 
source image.  The mipmaps are generated with a lanczos3 filter using edge clamping.
If the mipmaps option is not specified no mipmaps are created.

* -mipwrap takes an argument that specifies the mipmap filter wrap mode.  The options 
are "x", "y" and "xy" which specify wrapping in x only, y only or x and y respectively.
The default options are clamping in both x and y.

Note: Path names can use slashes or backslashes.  The tool will convert the 
slashes to the appropriate polarity for the current platform.


## API

The library supports two different APIs - a C-like API that is not heavily 
class-based and a class-based API.

main() in EtcTool.cpp contains an example of both APIs.

The Encode() method now returns an EncodingStatus that contains bit flags for
reporting various warnings and flags encountered when encoding.


## Copyright
Copyright 2015 Etc2Comp Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
