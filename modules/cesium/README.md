**编译.NET**
1. 生成胶水
```
进入bin目录
[godot.windows.editor.double.x86_64.mono.exe] --headless --generate-mono-glue modules/mono/glue
```

2. 生成托管库
```
./modules/mono/build_scripts/build_assemblies.py --godot-output-dir=./bin
./modules/mono/build_scripts/build_assemblies.py --godot-output-dir ./bin --push-nupkgs-local ~/MyLocalNugetSource --precision=double
```
3. 插件编译
每次改动 需要首先执行这个
scons platform=windows precision=double dev_build=yes

4. .NET编译

````
scons platform=windows precision=double dev_build=yes module_mono_enabled=yes
<godot_binary> --headless --generate-mono-glue modules/mono/glue
python ./modules/mono/build_scripts/build_assemblies.py --godot-output-dir=./bin --precision=double

