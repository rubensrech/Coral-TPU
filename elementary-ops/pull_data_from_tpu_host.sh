rsync -vzP --exclude 'couscous.jpg' rasp-tpu:coral/elementary-ops/*.jpg ./
rsync -avzP rasp-tpu:coral/elementary-ops/models ./