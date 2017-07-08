#!/bin/bash

#### Configuration:
# The name of the executable. It is assumed
# that it is in the current working directory.
EXE_NAME=hidapi-testgui
# Path to the executable directory inside the bundle.
# This must be an absolute path, so use $PWD.
EXEPATH=$PWD/TestGUI.app/Contents/MacOS
# Libraries to explicitly bundle, even though they
# may not be in /opt/local. One per line. These
# are used with grep, so only a portion of the name
# is required. eg: libFOX, libz, etc.
LIBS_TO_BUNDLE=libFOX


function copydeps {
	local file=$1
	# echo "Copying deps for $file...."
	local BASE_OF_EXE=`basename $file`

	# A will contain the dependencies of this library
	local A=`otool -LX $file |cut -f 1 -d " "`
	local i
	for i in $A; do
		local BASE=`basename $i`

		# See if it's a lib we specifically want to bundle
		local bundle_this_lib=0
		local j
		for j in $LIBS_TO_BUNDLE; do
			echo $i |grep -q $j
			if [ $? -eq 0 ]; then
				bundle_this_lib=1
				echo "bundling $i because it's in the list."
				break;
			fi
		done

		# See if it's in /opt/local. Bundle all in /opt/local
		local isOptLocal=0
		echo $i |grep -q /opt/local
		if [ $? -eq 0 ]; then
			isOptLocal=1
			echo "bundling $i because it's in /opt/local."
		fi
		
		# Bundle the library
		if [ $isOptLocal -ne 0 ] || [ $bundle_this_lib -ne 0 ]; then

			# Copy the file into the bundle if it exists.
			if [ -f $EXEPATH/$BASE ]; then
				z=0
			else
				cp $i $EXEPATH
				chmod 755 $EXEPATH/$BASE
			fi
			
			
			# echo "$BASE_OF_EXE depends on $BASE"
			
			# Fix the paths using install_name_tool and then
			# call this function recursively for each dependency
			# of this library. 
			if [ $BASE_OF_EXE != $BASE ]; then
			
				# Fix the paths
				install_name_tool -id @executable_path/$BASE $EXEPATH/$BASE
				install_name_tool -change $i @executable_path/$BASE $EXEPATH/$BASE_OF_EXE

				# Call this function (recursive) on
				# on each dependency of this library.
				copydeps $EXEPATH/$BASE
			fi
		fi
	done
}

rm -f $EXEPATH/*

# Copy the binary into the bundle. Use ../libtool to do this if it's
# available beacuse if $EXE_NAME was built with autotools, it will be
# necessary.  If ../libtool not available, just use cp to do the copy, but
# only if $EXE_NAME is a binary.
if [ -x ../libtool ]; then
	../libtool --mode=install cp $EXE_NAME $EXEPATH
else
	file -bI $EXE_NAME |grep binary
	if [ $? -ne 0 ]; then
		echo "There is no ../libtool and $EXE_NAME is not a binary."
		echo "I'm not sure what to do."
		exit 1
	else
		cp $EXE_NAME $EXEPATH
	fi
fi
copydeps $EXEPATH/$EXE_NAME
