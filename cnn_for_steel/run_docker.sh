docker run --runtime=nvidia --rm -it -p 8888:8888 --name root -v $(pwd):/workdir -w /workdir tensor-keras
