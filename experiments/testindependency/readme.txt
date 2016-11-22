the independency test is calculated in R, both for XMRF and PSPN.

however, to compare times properly, we learn a shallow PSPN and measure the time taken for the shallow PSPN.

R(CompareXMRFtoINDtest.R) produces the file gnspnoutfile3.json and then the PSPN (SPNLearnTime.py) creates a file with the right times gnspnoutfile3_withtime.txt

after that, plotTimeError.py plots both the time and error of both methods 
