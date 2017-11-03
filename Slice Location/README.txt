Run RegressSlice_Localization.m

The output is something like the text file "out."
"RegressSlice_Localization.m" will make a five independent realizations. 
For each realization, part of training data is spared for validation for bandwidth selection.
With chosen bandwidth and all training data, the target values of testing data are estimated and evaluated to make NMSE.

In the file "out", the time for 5 realizations were 4661 seconds (=77.7 min), and the average NMSE reduced from 0.0066 to 0.0022 by using the proposed metric.
In real experiments, the average of 20 realizations are reported, and we used "KernelMetricregMultiplier = -2" for all benchmark datasets.

