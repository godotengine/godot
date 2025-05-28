/*
  Simple DirectMedia Layer
  Copyright (C) 1997-2025 Sam Lantinga <slouken@libsdl.org>

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
*/
#include "SDL_internal.h"

#include "./SDL_list.h"

// Push
bool SDL_ListAdd(SDL_ListNode **head, void *ent)
{
    SDL_ListNode *node = (SDL_ListNode *)SDL_malloc(sizeof(*node));

    if (!node) {
        return false;
    }

    node->entry = ent;
    node->next = *head;
    *head = node;
    return true;
}

// Pop from end as a FIFO (if add with SDL_ListAdd)
void SDL_ListPop(SDL_ListNode **head, void **ent)
{
    SDL_ListNode **ptr = head;

    // Invalid or empty
    if (!head || !*head) {
        return;
    }

    while ((*ptr)->next) {
        ptr = &(*ptr)->next;
    }

    if (ent) {
        *ent = (*ptr)->entry;
    }

    SDL_free(*ptr);
    *ptr = NULL;
}

void SDL_ListRemove(SDL_ListNode **head, void *ent)
{
    SDL_ListNode **ptr = head;

    while (*ptr) {
        if ((*ptr)->entry == ent) {
            SDL_ListNode *tmp = *ptr;
            *ptr = (*ptr)->next;
            SDL_free(tmp);
            return;
        }
        ptr = &(*ptr)->next;
    }
}

void SDL_ListClear(SDL_ListNode **head)
{
    SDL_ListNode *l = *head;
    *head = NULL;
    while (l) {
        SDL_ListNode *tmp = l;
        l = l->next;
        SDL_free(tmp);
    }
}
