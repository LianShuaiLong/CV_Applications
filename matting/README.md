## start
python modnet_infer.py --input_path PATH_TO_TEST_IMAGES --output_path PATH_TO_SAVE_RESULT --ckpt_path PATH_TO_PRETRAINED_MODEL --image_type IMAGE_TYPE --input_size INPUT_IMAGE_SIZE

## test results

![image](https://github.com/LianShuaiLong/CV_Applications/blob/master/matting/test_results/multi_combined.png)

![image](https://github.com/LianShuaiLong/CV_Applications/blob/master/matting/test_results/Obama_combined.png)

![image](https://github.com/LianShuaiLong/CV_Applications/blob/master/matting/test_results/beauty_combined.png)

From left to the right : Input image,predicted alpha matte,predicted foreground with black blackground,predicted foreground with white blackground
### Notice you can change your background to get a better result according to your input!
