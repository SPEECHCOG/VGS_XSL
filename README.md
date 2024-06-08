# Cross-situational learning in computational models of visually grounded speech (VGS_XSL)

Python and MATLAB scripts for the experiments reported in manuscript titled "*Can phones, syllables, and words emerge as side-products of cross-situational audiovisual learning? --- A computational investigation*" by Khazar Khorrami and Okko Räsänen. 


Feature extraction, model training and semantic retrieval evalution scripts were written in Python and are available from the respective folders. 
Analysis scripts of hidden layer activations were written mostly in MATLAB, and can be found under `selectivity_analyses/`  

Models and model activation data are available for download at Zenodo: https://doi.org/10.5281/zenodo.4564283   

Manuscript is available at: https://ldr.lps.library.cmu.edu/article/id/434/


# Model 

![VGSnetworkarchitecture](https://github.com/SPEECHCOG/VGS_XSL/assets/33454475/3a12ddb7-1058-4bee-85aa-85fb7613eeb8)



## Data used in the experiments

Brent-Siskind corpus is available at  
https://childes.talkbank.org/access/Eng-NA/Brent.html  

Places audio captions (English) are available at  
https://groups.csail.mit.edu/sls/downloads/placesaudio/downloads.cgi  

Places205 images: http://places.csail.mit.edu/downloadData.html

SPEECH-COCO audio captions are available at  
https://zenodo.org/record/4282267

MSCOCO images are available at  
https://cocodataset.org/#download  

The derived version of "Large-Brent" with utterance-level waveforms with their  
 phone, syllable and word-level transcripts (based on Rytting et al., 2010,  
 and Räsänen et al., 2018) is available from the second author upon request (okko.rasanen@tuni.fi).  
 The data cannot be shared publicly as it would require redistribution of modified  
 Brent-Siskind audio files. Annotations corresponding to the derived model activations  
are included in the model activation package shared through Zenodo (link above). 
