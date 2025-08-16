#/usr/bin/env bash
for i in {1..1000000}
do
  echo "#### $i ####"
  curl -X POST -F image_file=@$1 http://localhost:1234/post > /dev/null
done
