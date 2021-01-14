## start
### image matting
python image_matting.py --input_path PATH_TO_TEST_IMAGES --output_path PATH_TO_SAVE_RESULT --ckpt_path PATH_TO_PRETRAINED_MODEL --image_type IMAGE_TYPE --input_size INPUT_IMAGE_SIZE

### image matting

![image](https://github.com/LianShuaiLong/CV_Applications/blob/master/matting/test_results/multi_combined.png)

![image](https://github.com/LianShuaiLong/CV_Applications/blob/master/matting/test_results/Obama_combined.png)

![image](https://github.com/LianShuaiLong/CV_Applications/blob/master/matting/test_results/beauty_combined.png)

From left to the right : Input image,predicted alpha matte,predicted foreground with black blackground,predicted foreground with white blackground
### Notice you can change your background to get a better result according to your input!

### video matting

![image]("https://github.com/LianShuaiLong/CV_Applications/blob/master/matting/test_videos/demo.gif")

![image]("https://github.com/LianShuaiLong/CV_Applications/blob/master/matting/video_results/demo.gif)


## Reference
1.[Deep Image Matting](https://arxiv.org/pdf/1703.03872.pdf)

2.[Is a Green Screen Really Necessary for Real-Time Portrait Matting?](https://arxiv.org/pdf/2011.11961.pdf)

3.[Modnet](https://github.com/ZHKKKe/MODNet)

4.[opencv读取图像格式-HWC](https://blog.csdn.net/qq_39938666/article/details/86701344)

5.[Tensor.repeat vs np.repeat](https://blog.csdn.net/qq_39938666/article/details/88412817?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-2.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-2.control)

6.[opencv写视频流的格式](https://blog.csdn.net/qq_34877350/article/details/89415672)
