// Copyright 2017 plutoo
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <lz4.h>
#include "sha256.h"
#include "elf64.h"

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint8_t u8;

typedef struct {
    u32 FileOff;
    u32 DstOff;
    u32 DecompSz;
    u32 AlignOrTotalSz;
} NsoSegment;

typedef u8 Sha2Hash[0x20];

typedef struct {
    u8  Magic[4];
    u32 Unk1;
    u32 Unk2;
    u32 Unk3;
    NsoSegment Segments[3];
    u8  BuildId[0x20];
    u32 CompSz[3];
    u8  Padding[0x24];
    u64 Unk4;
    u64 Unk5;
    Sha2Hash Hashes[3];    
} NsoHeader;

uint8_t* ReadEntireFile(const char* fn, size_t* len_out) {
    FILE* fd = fopen(fn, "rb");
    if (fd == NULL)
        return NULL;

    fseek(fd, 0, SEEK_END);
    size_t len = ftell(fd);
    fseek(fd, 0, SEEK_SET);

    uint8_t* buf = malloc(len);
    if (buf == NULL) {
        fclose(fd);
        return NULL;
    }

    size_t rc = fread(buf, 1, len, fd);
    if (rc != len) {
        fclose(fd);
        free(buf);
        return NULL;
    }

    *len_out = len;
    return buf;
}

// -- GODOT start --
int elf2nso_main(int argc, char* argv[]) {
// -- GODOT end --
    if (argc != 3) {
        fprintf(stderr, "%s <elf-file> <nso-file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    NsoHeader nso_hdr;
    memset(&nso_hdr, 0, sizeof(nso_hdr));
    memcpy(nso_hdr.Magic, "NSO0", 4);
    nso_hdr.Unk3 = 0x3f;

    if (sizeof(NsoHeader) != 0x100) {
        fprintf(stderr, "Bad compile environment!\n");
        return EXIT_FAILURE;
    }

    size_t elf_len;
    uint8_t* elf = ReadEntireFile(argv[1], &elf_len);
    if (elf == NULL) {
        fprintf(stderr, "Failed to open input!\n");
        return EXIT_FAILURE;
    }

    if (elf_len < sizeof(Elf64_Ehdr)) {
        fprintf(stderr, "Input file doesn't fit ELF header!\n");
        return EXIT_FAILURE;
    }

    Elf64_Ehdr* hdr = (Elf64_Ehdr*) elf;
    if (hdr->e_machine != EM_AARCH64) {
        fprintf(stderr, "Invalid ELF: expected AArch64!\n");
        return EXIT_FAILURE;
    }

    Elf64_Off ph_end = hdr->e_phoff + hdr->e_phnum * sizeof(Elf64_Phdr);

    if (ph_end < hdr->e_phoff || ph_end > elf_len) {
        fprintf(stderr, "Invalid ELF: phdrs outside file!\n");
        return EXIT_FAILURE;
    }

    Elf64_Phdr* phdrs = (Elf64_Phdr*) &elf[hdr->e_phoff];
    size_t i, j = 0;
    size_t file_off = sizeof(NsoHeader);

    uint8_t* comp_buf[3];
    int comp_sz[3];

    for (i=0; i<3; i++) {
        Elf64_Phdr* phdr = NULL;
        while (j < hdr->e_phnum) {
            Elf64_Phdr* cur = &phdrs[j++];
            if (cur->p_type == PT_LOAD) {
                phdr = cur;
                break;
            }
        }

        if (phdr == NULL) {
            fprintf(stderr, "Invalid ELF: expected 3 loadable phdrs!\n");
            return EXIT_FAILURE;
        }

        nso_hdr.Segments[i].FileOff = file_off;
        nso_hdr.Segments[i].DstOff = phdr->p_vaddr;
        nso_hdr.Segments[i].DecompSz = phdr->p_filesz;

        // for .data segment this field contains bss size
        if (i == 2)
            nso_hdr.Segments[i].AlignOrTotalSz = phdr->p_memsz - phdr->p_filesz;
        else
            nso_hdr.Segments[i].AlignOrTotalSz = 1;

        SHA256_CTX ctx;
        sha256_init(&ctx);
        sha256_update(&ctx, &elf[phdr->p_offset], phdr->p_filesz);
        sha256_final(&ctx, (u8*) &nso_hdr.Hashes[i]);

        size_t comp_max = LZ4_compressBound(phdr->p_filesz);
        comp_buf[i] = malloc(comp_max);

        if (comp_buf[i] == NULL) {
            fprintf(stderr, "Compressing: Out of memory!\n");
            return EXIT_FAILURE;
        }

        // TODO check p_offset
        comp_sz[i] = LZ4_compress_default(&elf[phdr->p_offset], comp_buf[i], phdr->p_filesz, comp_max);

        if (comp_sz[i] < 0) {
            fprintf(stderr, "Failed to compress!\n");
            return EXIT_FAILURE;
        }

        nso_hdr.CompSz[i] = comp_sz[i];
        file_off += comp_sz[i];
    }
    
    /* Iterate over sections to find build id. */
    size_t cur_sect_hdr_ofs = hdr->e_shoff;
    for (unsigned int i = 0; i < hdr->e_shnum; i++) {
        Elf64_Shdr *cur_shdr = (Elf64_Shdr *)(elf + cur_sect_hdr_ofs);
        if (cur_shdr->sh_type == SHT_NOTE) {
            Elf64_Nhdr *note_hdr = (Elf64_Nhdr *)(elf + cur_shdr->sh_offset);
            u8 *note_name = (u8 *)((uintptr_t)note_hdr + sizeof(Elf64_Nhdr));
            u8 *note_desc = note_name + note_hdr->n_namesz;
            if (note_hdr->n_type == NT_GNU_BUILD_ID && note_hdr->n_namesz == 4 && memcmp(note_name, "GNU\x00", 4) == 0) {
                size_t build_id_size = note_hdr->n_descsz;
                if (build_id_size > 0x20) {
                    build_id_size = 0x20;
                }
                memcpy(nso_hdr.BuildId, note_desc, build_id_size);
            }
        }
        cur_sect_hdr_ofs += hdr->e_shentsize;
    }

    FILE* out = fopen(argv[2], "wb");

    if (out == NULL) {
        fprintf(stderr, "Failed to open output file!\n");
        return EXIT_FAILURE;
    }

    // TODO check retvals
    fwrite(&nso_hdr, sizeof(nso_hdr), 1, out);

    for (i=0; i<3; i++)
        fwrite(comp_buf[i], comp_sz[i], 1, out);

    return EXIT_SUCCESS;
}
