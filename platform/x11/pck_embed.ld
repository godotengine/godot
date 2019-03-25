SECTIONS
{
	/* Add a zero-sized section; the exporter will patch it to enclose the data appended to the executable (embedded PCK) */
	pck 0 (NOLOAD) :
	{
		/* Just some content to avoid the linker discarding the section */
		. = ALIGN(8);
	}
}
INSERT AFTER .rodata;
