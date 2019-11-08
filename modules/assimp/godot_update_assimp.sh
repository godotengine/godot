rm -rf ../../thirdparty/assimp
cd ../../thirdparty/
git clone https://github.com/assimp/assimp.git
cd assimp
rm -rf code/3DS/
rm -rf code/3MF/
rm -rf code/AMF/
rm -rf code/ASE/
rm -rf code/BVH/
rm -rf code/Assbin/
rm -rf code/Assjson/
rm -rf code/Assxml/
rm -rf code/B3D/
rm -rf code/COB/
rm -rf code/M3D/
rm -rf code/MMD/
rm -rf code/Irr/
rm -rf code/Ply/
rm -rf code/Blender/
rm -rf code/Collada/
rm -rf code/C4D/
rm -rf code/AC/
rm -rf code/IRR/
rm -rf code/LWS/
rm -rf code/MD2/
rm -rf code/MD3/
rm -rf code/MD5/
rm -rf code/MDC/
rm -rf code/MDL/
rm -rf code/NFF/
rm -rf code/OFF/
rm -rf code/PLY/
rm -rf code/LWO/
rm -rf code/MD4/
rm -rf code/Terragen/
rm -rf code/Unreal/
rm -rf code/X/
rm -rf code/HMP/
rm -rf code/Importer/
rm -rf code/MS3D/
rm -rf code/X3D/
rm -rf code/Step/
rm -rf code/SMD/
rm -rf code/SIB/
rm -rf code/STL/
rm -rf code/XGL/
rm -rf code/OpenGEX/
rm -rf code/Q3BSP/
rm -rf code/Raw/
rm -rf code/Ogre/
rm -rf code/NDO/
rm -rf code/DXF/
rm -rf code/CSM/
rm -rf code/Q3D/
rm -rf .git
rm -rf cmake-modules
rm -rf doc
rm -rf packaging
rm -rf port
rm -rf samples
rm -rf scripts
rm -rf test
rm -rf tools
rm -rf contrib/zlib
rm -rf contrib/android-cmake
rm -rf contrib/gtest
rm -rf contrib/clipper
rm -rf contrib/irrXML
rm -rf contrib/Open3DGC
rm -rf contrib/openddlparser
rm -rf contrib/poly2tri
rm -rf contrib/unzip
rm code/Common/ZipArchiveIOSystem.cpp
rm -rf contrib/zip
rm -rf contrib/stb_image
rm -rf contrib/CMakeLists.txt
rm -rf contrib/irrXML_note.txt
rm -rf contrib/poly2tri_patch.txt
git checkout -- assimp/config.h
git checkout -- code/revision.h
rm .travis*
rm -rf .github/
rm include/assimp/.editorconfig
rm CodeConventions.md
rm INSTALL
rm README
rm Readme.md
rm CONTRIBUTING.md
rm revision.h.in
rm code/CMakeLists.txt
rm code/.editorconfig
rm cmake/HunterGate.cmake
rm cmake/assimp-hunter-config.cmake.in
rm CHANGES
rm BUILDBINARIES_EXAMPLE.bat
git checkout -- assimp/config.h
git checkout -- code/revision.h
rm *.cmake.in
rm appveyor.yml
rm .gitignore
rm .gitattributes
rm CMakeLists.txt
rm Build.md
rm .coveralls.yml
rm .editorconfig
rm assimp.pc.in
rm -rf cmake/
rm include/assimp/irrXMLWrapper.h 