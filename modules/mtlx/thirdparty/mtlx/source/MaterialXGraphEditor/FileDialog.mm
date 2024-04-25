//
// Copyright Contributors to the MaterialX Project
// SPDX-License-Identifier: Apache-2.0
//

#include <MaterialXGraphEditor/FileDialog.h>

#include <MaterialXCore/Exception.h>

#import <Cocoa/Cocoa.h>

mx::StringVec launchFileDialog(const std::vector<std::pair<std::string, std::string>>& filetypes, bool save, bool multiple)
{
    if (save && multiple)
    {
        throw mx::Exception("launchFileDialog(): 'save' and 'multiple' must not both be true.");
    }

    mx::StringVec result;
    if (save)
    {
        NSSavePanel* saveDlg = [NSSavePanel savePanel];

        NSMutableArray* types = [NSMutableArray new];
        for (size_t idx = 0; idx < filetypes.size(); ++idx)
            [types addObject:[NSString stringWithUTF8String:filetypes[idx].first.c_str()]];

        [saveDlg setAllowedFileTypes:types];

        if ([saveDlg runModal] == NSModalResponseOK)
            result.emplace_back([[[saveDlg URL] path] UTF8String]);
    }
    else
    {
        NSOpenPanel* openDlg = [NSOpenPanel openPanel];

        [openDlg setCanChooseFiles:YES];
        [openDlg setCanChooseDirectories:NO];
        [openDlg setAllowsMultipleSelection:multiple];
        NSMutableArray* types = [NSMutableArray new];
        for (size_t idx = 0; idx < filetypes.size(); ++idx)
            [types addObject:[NSString stringWithUTF8String:filetypes[idx].first.c_str()]];

        [openDlg setAllowedFileTypes:types];

        if ([openDlg runModal] == NSModalResponseOK)
        {
            for (NSURL* url in [openDlg URLs])
            {
                result.emplace_back((char*) [[url path] UTF8String]);
            }
        }
    }
    return result;
}
