all:
	scons platform=linuxbsd
analyze:
	# scan-build -o static_analysis scons platform=linuxbsd
	# scan-build -o static_analysis  -plist scons platform=linuxbsd
	# scan-build -o static_analysis scons 
	# scan-build --analyze scons platform=linuxbsd
	ANALYZE=1 scons platform=linuxbsd
