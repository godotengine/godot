#ifndef OS_3DS_H
#define OS_3DS_H

class OS_3DS {
public:
    static OS_3DS *get_singleton();

    void initialize();
    void finalize();

private:
    static OS_3DS *singleton;
};

#endif
