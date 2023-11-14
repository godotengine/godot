#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <inttypes.h>

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint8_t u8;

typedef struct {
    char name[0x200];
    char author[0x100];
} NacpLanguageEntry;

typedef struct {
    NacpLanguageEntry lang[12];
    NacpLanguageEntry lang_unk[4];//?

    u8  x3000_unk[0x24];////Normally all-zero?
    u32 x3024_unk;
    u32 x3028_unk;
    u32 x302C_unk;
    u32 x3030_unk;
    u32 x3034_unk;
    u64 titleid0;

    u8 x3040_unk[0x20];
    char version[0x10];

    u64 titleid_dlcbase;
    u64 titleid1;

    u32 x3080_unk;
    u32 x3084_unk;
    u32 x3088_unk;
    u8 x308C_unk[0x24];//zeros?

    u64 titleid2;
    u64 titleids[7];//"Array of application titleIDs, normally the same as the above app-titleIDs. Only set for game-updates?"

    u32 x30F0_unk;
    u32 x30F4_unk;

    u64 titleid3;//"Application titleID. Only set for game-updates?"

    char bcat_passphrase[0x40];
    u8 x3140_unk[0xEC0];//Normally all-zero?
} NacpStruct;

// -- GODOT start --
int nacp_main(int argc, char* argv[]) {
// -- GODOT end --
    if (argc < 6 || strncmp(argv[1], "--create", 8)!=0) {
        fprintf(stderr, "%s --create <name> <author> <version> <outfile> [options]\n\n", argv[0]);
        fprintf(stderr, "FLAGS:\n");
        fprintf(stderr, "--create : Create control.nacp for use with Switch homebrew applications.\n");
        fprintf(stderr, "Options:\n");
        fprintf(stderr, "--titleid=<titleID> Set the application titleID.\n");
        return EXIT_FAILURE;
    }

    NacpStruct nacp;
    memset(&nacp, 0, sizeof(nacp));

    if (sizeof(NacpStruct) != 0x4000) {
        fprintf(stderr, "Bad compile environment!\n");
        return EXIT_FAILURE;
    }

    char *name = argv[2];
    char *author = argv[3];
    char *names[12];
    char *authors[12];

    int i;
    for (i=0; i<12; i++) {
        names[i] = name;
        authors[i] = author;
    }

    int argi;
    u64 titleid=0;
    for (argi=6; argi<argc; argi++) {
        if (strncmp(argv[argi], "--titleid=", 10)==0) sscanf(&argv[argi][10], "%016" SCNx64, &titleid);
    }

    for (i=0; i<12; i++) {//These are UTF-8.
        strncpy(nacp.lang[i].name, names[i], sizeof(nacp.lang[i].name)-1);
        strncpy(nacp.lang[i].author, authors[i], sizeof(nacp.lang[i].author)-1);
    }

    strncpy(nacp.version, argv[4], sizeof(nacp.version)-1);

    if (titleid) {
        nacp.titleid0 = titleid;
        nacp.titleid_dlcbase = titleid+0x1000;
        nacp.titleid1 = titleid;
        nacp.titleid2 = titleid;
    }

    u8 unk_data[0x20] = {0x0C, 0xFF, 0xFF, 0x0A, 0xFF, 0x0C, 0x0C, 0x0C, 0x0C, 0x0C, 0x0D, 0x0D, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

    nacp.x3024_unk = 0x100;
    nacp.x302C_unk = 0xbff;
    nacp.x3034_unk = 0x10000;

    memcpy(nacp.x3040_unk, unk_data, sizeof(unk_data));

    nacp.x3080_unk = 0x3e00000;
    nacp.x3088_unk = 0x180000;
    nacp.x30F0_unk = 0x102;

    FILE* out = fopen(argv[5], "wb");

    if (out == NULL) {
        fprintf(stderr, "Failed to open output file!\n");
        return EXIT_FAILURE;
    }

    fwrite(&nacp, sizeof(nacp), 1, out);

    fclose(out);

    return EXIT_SUCCESS;
}
