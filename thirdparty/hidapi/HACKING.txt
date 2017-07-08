This file is mostly for the maintainer.

1. Build hidapi.dll
2. Build hidtest.exe in DEBUG and RELEASE
3. Commit all

4. Run the Following
	export VERSION=0.1.0
	export TAG_NAME=hidapi-$VERSION
	git tag $TAG_NAME
	git archive --format zip --prefix $TAG_NAME/ $TAG_NAME >../$TAG_NAME.zip
5. Test the zip file.
6. Run the following:
	git push origin $TAG_NAME

