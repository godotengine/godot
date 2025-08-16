/* $Id: upnpdev.c,v 1.1 2015/08/28 12:14:19 nanard Exp $ */
/* Project : miniupnp
 * Web : http://miniupnp.free.fr/
 * Author : Thomas BERNARD
 * copyright (c) 2005-2015 Thomas Bernard
 * This software is subjet to the conditions detailed in the
 * provided LICENSE file. */
#include <stdlib.h>
#include "upnpdev.h"

/* freeUPNPDevlist() should be used to
 * free the chained list returned by upnpDiscover() */
void freeUPNPDevlist(struct UPNPDev * devlist)
{
	struct UPNPDev * next;
	while(devlist)
	{
		next = devlist->pNext;
		free(devlist);
		devlist = next;
	}
}

