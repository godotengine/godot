#!/bin/bash

python loader_genvk.py -registry vk.xml -scripts . vk_layer_dispatch_table.h
python loader_genvk.py -registry vk.xml -scripts . vk_dispatch_table_helper.h
python loader_genvk.py -registry vk.xml -scripts . vk_object_types.h
python loader_genvk.py -registry vk.xml -scripts . vk_loader_extensions.h
python loader_genvk.py -registry vk.xml -scripts . vk_loader_extensions.c
mv ./*.c ../loader/
mv ./*.h ../loader/
