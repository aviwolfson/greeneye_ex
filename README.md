### The code
the code is script like and is execute as the goals order
for running the code u just need the "data_path" changed to the real data dirs
the utils.py file has most of the complex functions. 
iou, calculating TP,FP for precision recall, ny ratio metric and comparing iou vs my metric with plotting the difference.  

### Metric
i used a matric that calculate the ratio of the center distance multiply the height/width ratio. 
the idea is that the detection rectangle has 2 main charcaristics that we want to compare. 
its size comparing the  GT and the distance it is from the GT. u can see that this metrica is more strict with the predictions than IOU (i belive i need to improve it by using the Pythagorean Theorem )  , the 4 parameters of the metric (width ratio, hight ratio , center width distance and center hight distance) can be calculated differently for getting diff metrics (like mean of the distance and not multiply it , etc...)
