# EmotionMusic - Classification
Implementation of the data processing pipeline for the EmotionMusic project.


First create an Anaconda environment with MNE and pyFFTW installed:
`conda create --name=mne --channel=conda-forge mne`
`conda install -c conda-forge pyfftw`

then activate the evironment with:
`activate mne`

install meegkit:
`pip install git+https://github.com/nbara/python-meegkit.git`

and finally install the SignalProcessingToolbox and PyMensia previously downloaded in a separated folder:
`pip install ../toolboxes/SignalProcessing.Toolbox`
`pip install ../toolboxes/pymensia2`
`pip install ../toolboxes/python-mensia-sigpro2`

All the missing dependencies can be installed at once with a simple `pip install -r requirements.txt`

