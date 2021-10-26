SECTIONS
{
	/* Add a zero-sized section; the exporter will patch it to enclose the data appended to the executable (embedded PCK) */
	pck 0 (INFO) :
	{
		/* binutils >= 2.30 allow it being zero-sized, but needs something between the braces to keep the section */
		. = ALIGN(8);
	}
}
INSERT AFTER .rodata;
