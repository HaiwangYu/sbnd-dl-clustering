rm -f upload.zip
cd bee
zip -r ../upload data
cd -
./upload-to-bee.sh upload.zip
