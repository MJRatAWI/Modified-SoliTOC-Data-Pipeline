The Modified-SoliTOC-Data-Pipeline is an automated data analysis tool for the modified SoliTOC setup (Kattein et al., 2026) at the Alfred-Wegener-Institute in Bremerhaven, Germany. The automated analysis is currently available for pyrolysis mode only (oxidation will be added in the future).




Package Information

Authors: 	Manuel Ruben (manuel.ruben@awi.de)

Version: 	1.0.0

Python:  3.13.13

Release: 	1st October 2026

License: 	GNU GPL v3 (or greater)

URL:		https://github.com/MJRatAWI/Modified-SoliTOC-Data-Pipeline 

DOI:		10.5281/zenodo.23078601 




How to use

The application is built as a Kedro project (https://kedro.org/) embedded into a graphical user interface (GUI), allowing for batch analysis. To use the code successfully, the “name column” in the modified SoliTOC needs to be in a specific format, please use provided the excel template for this purpose. Further information are provided as comments in the excel sheet. Default parameters in the GUI should not be change, if not otherwise instructed by the modified-SoliTOC personal. Detailed information can be found in the my-pipeline readme under RUN.




Output

The output of the application will include:
1)	Batch summary files, in both csv and xlsx format, including all output and method parameters.
2)	Batch thermogram xslx file includes all raw and corrected thermograms, one page per sample. Individual pages are saved as CSV files in the prossed_csv subfolder.
3)	Summary folder inhibits txt files with further information to each processed sample.
4)	RampedPyrox_Input folder includes ready to use input files for the RampedPyrox application (Hemingway et al., 2017). 
Detailed information can be found in the my-pipeline readme.

References
1)	Kattein L, Höhn M, Grotheer H, Gentz T, Mollenhauer G. Introducing a ramped pyrolysis/oxidation‐14CO2 analysis system operated with mobile molecular sieve traps. Limnology and Oceanography: Methods. 2026:e70097. DOI: 10.1002/lom3.70097
2)	J.D. Hemingway et al. (2017) Technical note: An inverse model to realte organic carbon reactivity to isotope composition from serial oxidation. Biogeosciences, 22, 5099-5114. DOI: 10.5194/bg-14-5099-2017 
