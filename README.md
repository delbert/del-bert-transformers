# del-bert-transformers

based on code here:  https://www.analyticsvidhya.com/blog/2020/10/simple-text-multi-classification-task-using-keras-bert/

some notes

you can train the model from either cli v2 or pipelines and then you need to register the model using the Portal.  then in score.py (one in each directory), 
there is code to download the registered model and load it into memory for scoring

## /data contains the datasets for training and testing the Model


## /cli_v2 contains the .yml and scripts for running cli v2 (from your laptop) jobs for training and scoring the Model

the cli jobs are intended for you to quickly test out and prototype new training and scoring scenarios -- and then you can finalize in a pipeline

to run the cli v2 jobs, you need to open a cli prompt and navigate to the /cli_v2 directory and then enter:

`az login`
`az configure --defaults group="<your resource group>" workspace="<your workspace>"`           (you only need to set this once)

to run the training job:
`az ml job create -f job.yml --web`

to run the scoring job:
`az ml job create -f job_score.yml --web`

the score.py file in this directory is slightly different from the one in the pipelines directory -- this one hard-codes the input file name


## /pipelines contains the notebooks and scripts for setting up the pipeline to score the Model and scripts to train the Model

the train.ipynb does the Model training 

the datasets.ipynb converts json datasets into csv datasets and it creates a new Environment from a dockerfile (dockerfile.txt)

the batch_scoring_pipeline.ipnb is divided into sections which:
* builds the scoring pipeline and submits it as an Experiment
* publishes the scoring pipeline to a ReST endpoint
* calls the pipeline
* calls the pipeline using stand-alone code that can be translated into any language

the batch_scoring_pipeline also calls one of two Python scripts:
* score.py to score the example file
* hello-score.py which is a minimal hello-world script that reads the input parameter (file) and creates an output file (based on current date/time)

