
import glob

pixmaps = glob.glob("*.png")

f = open("../editor_icons.cpp","wb")


f.write("#include \"editor_icons.h\"\n\n")
f.write("#include \"scene/resources/theme.h\"\n\n")

for x in pixmaps:

	var_str=x[:-4]+"_png";

	f.write("static const unsigned char "+ var_str +"[]={\n");

	pngf=open(x,"rb");

	b=pngf.read(1);
	while(len(b)==1):
		f.write(hex(ord(b)))
		b=pngf.read(1);
		if (len(b)==1):
			f.write(",")

	f.write("\n};\n\n\n");
	pngf.close();

f.write("static Ref<ImageTexture> make_icon(const uint8_t* p_png) {\n")
f.write("\tRef<ImageTexture> texture( memnew( ImageTexture ) );\n")
f.write("\ttexture->create_from_image( Image(p_png),ImageTexture::FLAG_FILTER );\n")
f.write("\treturn texture;\n")
f.write("}\n\n")

f.write("void editor_register_icons(Ref<Theme> p_theme) {\n\n")


for x in pixmaps:

	type=x[5:-4].title().replace("_","");
	var_str=x[:-4]+"_png";
	f.write("\tp_theme->set_icon(\""+type+"\",\"EditorIcons\",make_icon("+var_str+"));\n");

f.write("\n\n}\n\n");
f.close()


