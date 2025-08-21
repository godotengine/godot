TARGET = template_debug

BASE = scons target=$(TARGET) $(EXTRA_ARGS)
LINUX = $(BASE) platform=linux
WINDOWS = $(BASE) platform=windows
MACOS = $(BASE) platform=macos


.PHONY: usage
usage:
	@echo -e "Specify one of the available targets:\n"
        # https://stackoverflow.com/a/26339924
	@LC_ALL=C $(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/(^|\n)# Files(\n|$$)/,/(^|\n)# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | grep -E -v -e '^[^[:alnum:]]' -e '^$@$$'
	@echo -e "\nDefine the SCons target with TARGET, and pass extra SCons arguments with EXTRA_ARGS."


linux:
	make linux32
	make linux64

linux32: SConstruct
	$(LINUX) arch=x86_32

linux64: SConstruct
	$(LINUX) arch=x86_64


windows:
	make windows32
	make windows64

windows32: SConstruct
	$(WINDOWS) arch=x86_32

windows64: SConstruct
	$(WINDOWS) arch=x86_64


macos: SConstruct
	$(MACOS)
