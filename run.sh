docker build -t mdrecognizer .

docker run \
  --rm \
  -it \
  --name mdr_container \
  -p 8888:8888 \
  mdrecognizer