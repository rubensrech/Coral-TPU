rsync -vzP --exclude 'couscous.jpg' rasp-tpu:coral/elementary-ops/*.jpg ./
rsync -vzP rasp-tpu:coral/elementary-ops/*.npy ./
rsync -avzP rasp-tpu:coral/elementary-ops/models ./