#!/bin/bash

URLS_FILE=$(pwd)/subcoco_val_img_urls.txt
IMGS_OUT_DIR=$(pwd)/imgs

mkdir -p $IMGS_OUT_DIR
cd $IMGS_OUT_DIR

while IFS= read -r url;
do
	curl -O $url;
done < "$URLS_FILE"
