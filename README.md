Project-2
=========

A Study in Parallel Algorithms : Stream Compaction

Part2&3
Scan comparison:
+![](http://xxx.jpg)

As shown in the plot, the serial version is faster when the array size is small, but it becomes slower than GPU version when the array size is large, as it is single thread and will spend more time on loops. 
The global memory version is a little slower than the shared memory version, as writing/reading global memory is slower than writing/reading shared memory.


Part4
Stream compaction comparison:
+![](http://xxx.jpg)

As shown in the plot, the thrust version is faster than GPU version, but their speed gets closer when the array size becomes larger. I think GPU version is slower because my code is not optimized and it uses some global memory.



References
http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
http://docs.nvidia.com/cuda/thrust/#axzz3EameA17V
