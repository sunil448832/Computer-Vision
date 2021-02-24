File Structure:
1. input  	-this directory contain input images
2. output 	-this directory contain output images
3. img_pro.py 	-python file to test all features
4. utils.py    -python file which contain all functions.
5. README.txt
6. writeup.html



General Syntax to test particular feature is as bellow :
python program_name input_img output_dir --operation priority 

I have Implemented the following features and corresponding command to test these features are given.
1. Brightness      python img_pro.py './input/princeton_small.jpg' './output' --brightness 1 --factor 0.5 
2. Contrast        python img_pro.py './input/princeton_small.jpg' './output' --contrast 1 --factor 0.5 
3. Blur            python img_pro.py './input/princeton_small.jpg' './output' --blur 1  --sigma 2
4. Sharpen         python img_pro.py './input/princeton_small.jpg' './output' --sharpen 1
5. Edge detect     python img_pro.py './input/princeton_small.jpg' './output' --edge_detect 1
5  Scale           python img_pro.py './input/scaleinput.jpg' './output' --scale 1 --sx .3 --sy .3 --interpolation_method 'gaussian'
6. Composite       python img_pro.py './input/comp_background.jpg' './output' --foreground './input/comp_foreground.jpg' 
		    								 --mask './input/comp_mask.jpg' --composition 1
		    								 
To perform mutiple operation on single image sequentialy like "composition and blur" command is :
python img_pro.py './input/comp_background.jpg' './output' --foreground './input/comp_foreground.jpg' --mask './input/comp_mask.jpg' --   composition 1 --blur 2 --sigma 2.0

Here priorty for composition and blur are 1 and 2 respectively.
Lowest periority operation will be executed first. 
Lowest periority value is 1.


