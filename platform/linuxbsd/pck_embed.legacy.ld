SECTIONS
{
	/* The exporter will patch this section to enclose the data appended to the executable (embedded PCK) */
	pck 0 (INFO) : AT ( ADDR (.rodata) + SIZEOF (.rodata) )
	{
		/* binutils < 2.30 need some actual content for the linker not to discard the section */
		BYTE(0);
	}
}
INSERT AFTER .rodata;
