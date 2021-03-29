### Todo:
- Auto create a mp4 from generate images
    - Instead of using the command `ffmpeg -r 20 -i tmp/out_%d.png -vcodec libx265 -crf 25 -s 512x512 test.mp4` maybe use a rust library to do the same (more research needed)
- GPU compute 
    - Tried [ArrayFire-rust](https://github.com/arrayfire/arrayfire-rust) didn't work well, looking for another library