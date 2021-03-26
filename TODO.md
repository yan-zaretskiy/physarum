### Todo:
- Auto create a mp4 from generate images
    - Instead of using the command `ffmpeg -r 20 -i tmp/out_%d.png -vcodec  libx264 -crf 25 test.mp4` maybe use a rust library to do the same (more research needed)
- GPU compute via [ArrayFire-rust](https://github.com/arrayfire/arrayfire-rust)