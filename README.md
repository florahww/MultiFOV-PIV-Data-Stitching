# MultiFOV-PIV-Data-Stitching
PIV Data Stitching of 9 FOVs.

The folder includes 8 Python scripts.

Post_01_01_LineStitch.py and Post_01_02_RegionalStitch.py contain the required functions of two stitching methods.

Post_02_01_KeyLineMatchingStitch, Post_02_02_KeyRegionalStitch and Post_02_02_KeyRegionalStitchstereo, shows the way to implement the functions for detecting the shift/move distance of FOVs..

Post_03_BoundarySettings creates the boundary conditions using the previous results and the building geometry information.

Post_04_Reconstruction shows how to ensemble the data and reconstruct the coordination.

Post_05_AverageCalculation calculates the averaged flow.
