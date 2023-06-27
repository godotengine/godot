on run argv
    set vButtons to { "OK" }
    set vButtonCodes to { 0 }
    set vDbutton to "OK"
    set vText to ""
    set vTitle to ""
    set vTimeout to -1

    repeat with i from 1 to length of argv
        try
            set vArg to item i of argv
            if vArg = "-buttons" then
                set vButtonsAndCodes to my fSplit(item (i + 1) of argv, ",")
                set vButtons to {}
                set vButtonCodes to {}
                repeat with j from 1 to length of vButtonsAndCodes
                    set vBtn to my fSplit(item j of vButtonsAndCodes, ":")
                    copy (item 1 of vBtn) to the end of the vButtons
                    copy (item 2 of vBtn) to the end of the vButtonCodes
                end repeat
            else if vArg = "-title" then
                set vTitle to item (i + 1) of argv
            else if vArg = "-center" then
                -- not supported
            else if vArg = "-default" then
                set vDbutton to item (i + 1) of argv
            else if vArg = "-geometry" then
                -- not supported
            else if vArg = "-nearmouse" then
                -- not supported
            else if vArg = "-timeout" then
                set vTimeout to item (i + 1) of argv as integer
            else if vArg = "-file" then
                set vText to read (item (i + 1) of argv) as string
            else if vArg = "-text" then
                set vText to item (i + 1) of argv
            end if
        end try
    end repeat

    set vDlg to display dialog vText buttons vButtons default button vDbutton with title vTitle giving up after vTimeout with icon stop
    set vRet to button returned of vDlg
    repeat with i from 1 to length of vButtons
        set vBtn to item i of vButtons
        if vBtn = vRet
            return item i of vButtonCodes
        end if
    end repeat

    return 0
end run

on fSplit(vString, vDelimiter)
    set oldDelimiters to AppleScript's text item delimiters
    set AppleScript's text item delimiters to vDelimiter
    set vArray to every text item of vString
    set AppleScript's text item delimiters to oldDelimiters
    return vArray
end fSplit
