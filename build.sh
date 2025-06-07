# build in archlinux x86
scons platform=linuxbsd
scons platform=linuxbsd use_llvm=yes

# 构建导出模板
scons platform=linuxbsd target=template_release arch=x86_64
scons platform=linuxbsd target=template_debug arch=x86_64
