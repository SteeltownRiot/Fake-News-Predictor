#!/usr/bin/env python
# coding: utf-8

# # Final Project
# 
# You have made it to the end of the course, and you have worked hard to develop your DSA perspectives and skills.  So far we have been internally focused on the operations of performing data science and analytics.  Now we will extend our work to the development of a data story that is externally focused.
# 
# In the Module8 labs, you saw simplified examples of constructing data stories. In module4 (Database) there also was an abbreviated example data story.  Throughout the course, there are components and parts useful to consider as a basis for developing a short, unique, focused data story.
# 
# 
# For this final project, you will 
# 
# - Step 0: Choose your Language for this Adventure
# 
# - Step 1: Find a Story
# 
# - Step 2: Remember your Audience
# 
# - Step 3: Find and Stage Your Data
# 
# - Step 4: Vet Data Sources
# 
# - Step 5: Filter Results and Build/Validate Models
# 
# - Step 6: Visualize Results
# 
# - Step 7: Communicate the Story to your intended audience using visualizations and narratives
# 
# - Final Step: Connect your workflow/process to the DSA-Project Life Cycle
# 
# ---
# Here are some recommendations for managing the scope and quality of this project:
# 
# - Narrow down the issue, problem, question, or hypothesis for you data story to a single, relatively simple perspective.
# 
# - Identify already available data that affords addressing your problem.  If using completely new data, know it well.
# 
# - Address the data relative to the statistical/machine learning model(s) chosen to minimize any issues.
# 
# - Internally document your code using comments that explain the purpose of the operation(s).
# 
# 
# Make your project unique by
# 
# - Comparing two or more different statistical/machine learning models using the same data.
# - Refrain from identically replicating any existing projects obtained from external sources.
# - Running a single model multiple times and changing a different single parameter each time for comparison.
# - Changing the sampling proportions for building the hold-out data and comparing the same model performance repeatedly.
# - Select something you find interesting or unique in the data and write a story around it.
# 
# 
# 

# ## Step 0: Choose your Language for this Adventure:
# 
# You can do this project in either *R* or *Python*.
# 
# To change the kernel of this notebook, do the following with the `Kernel` menu.
# 
#  * `Kernel > Change Kernel > Python 3`
#  * `Kernel > Change Kernel > R`
# 
# ![FP_Change_Kernel.png MISSING](../images/FP_Change_Kernel.png)
# 

# ---
# ## Step 1: Find a Story
# 
# Think back to any of the data files we have used in this class. 
# Alternatively, you can search online for potential data and story ideas.
# 
# In the cell below, please detail the source of your data (with link).
# Additionally, preview your story you hope to uncover.
Tool Box is a tech website that listed some project ideas. Disinformation has plagued the internet since the days of message boards long before the World Wide Web made the Internet widely accessible. But those were always on the fringe of even the eccentric computer community. YouTube really kicked the door open to conspiracy theories, starting with bringing "Flat Earth" nearly into the mainstream, with even superstar athletes publicly promoting it. Social media has supercharged the spread of disinformation in the last half a decade, and I spent way too much of my time from 2015 through 2017, when I mostly gave up social media, on Facebook correcting friends and family posting all kinds of crazy conspiracy theories and flat out fake news. All the while, these same people were denouncing news they disagreed with as fake news. Finding a data set that will allow me to try and predict fake news from real news made me want to jump in and give it a try.

https://www.toolbox.com/tech/big-data/guest-article/top-10-data-science-project-ideas-for-2020/
# ## Step 2: Remember your Audience
# 
# In the cell below, describe your audience!
#  * Who will the audience be?
#  * What value will they derive from your story?
I would like to think the audience for this kind of project would be everyone because they would want to know whether what they are readying is true or not. Unfortunately, I think too many people would assume nefarious intent if my model says a news source they feel like they can trust is mostly fake or a story they believe to be fake is actually true. So realisticly, the audience would be folks like I used to be (and sometimes still cannot help myself) looking to verify or refute a news story so they can try to inform their friends and family to its veracity. The value they would get from this project would be to spend less time researching the truth of a news story so they can spend more time on homework or with their familys.
# ## Step 3: Find and Stage Your Data
# 
# If you data is from another source, such as Kaggle, you must download it to your local computer, then upload the data to JuptyerHub.
# 
# #### If you are uploading files:
#  * Use folder navigation of your first JupyterTab to get to course's `/modules/module8/exercises/` folder.
# ![FP_Folder_Navigation.png MISSING](../images/FP_Folder_Navigation.png)
#  * Click the Upload Button and Choose File(s)
# ![FP_Upload_Button.png MISSING](../images/FP_Upload_Button.png)
#  * Activate the upload
# ![FP_UploadFile_2.png MISSING](../images/FP_UploadFile_2.png)
#  
# 
# ### In the cell below, please list the name(s) of the file(s) that is now accessible on the JupyterHub environment.
# 
# **Note**: 
# If you uploaded a file to your `module8/exercises` folder, the file name is all you need to load it into the a data frame in the usual manner.
# If you are using a file from another module of the course, you should be able to copy the full pathname and use it as is in this notebook.
I uploaded news.csv to my module8/exercises folder.


# ## Step 4: Vet Data Sources
# 
# Use the cells below to load the data, inspect it, conduct data carpentry and shaping; perform exploratory data analysis.  
# 
# Add more cells (`Insert > Insert Cell Below`) if you want additional cells.

# In[2]:


# Load packages and libraries
import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
# Checks number of times a word appears in text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Read the CSV as a Pandas dataframe
with open('news.csv') as file:
    df = pd.read_csv(file)

# Display the shape and first few records of data set
print(df.shape)
df.head()


# In[3]:


# Check for any NaNs
df.isnull().values.any()


# In[7]:


#Checking which columns have NaNs
print(df['id'].isnull().sum())
print(df['title'].isnull().sum())
print(df['text'].isnull().sum())
print(df['label'].isnull().sum())


# In[8]:


# There is no way to fill NaNs in this data set so I will remove all rows with NaNs
df = df.dropna()

# Check to make sure NaNs were removed
df.isnull().values.any()


# In[17]:


# Checking for duplicate articles by title
duplicateTitles = df[df.duplicated(['title'])]

duplicateTitles


# In[18]:


# Checking for duplicate articles by text
pd.concat(g for _, g in df.groupby("title") if len(g) > 1)


# In[ ]:





# In[ ]:





# In[ ]:





# ## Step 5: Filter Results and Build/Validate Models
# 
# 
# Perform any additional data carpentry and begin filtering results/data and then build, validate, and describe your model(s). 
# 
# Add more cells (`Insert > Insert Cell Below`) if you want additional cells.

# In[ ]:





def findlabel(newtext):
vec_newtest=tfidf_vectorizer.transform([newtext])
t_pred=pac.predict(vec_newtest)
t_pred.shape
return t_pred


# In[ ]:





# ## Step 6: Visualize Results
# 
# Build up your key visual story elements!
# 
# Add more cells (`Insert > Insert Cell Below`) if you want additional cells.

# In[ ]:






# In[ ]:





# ## Step 7: Communicate the Story to your intended audience using visualizations and narrative
# 
# 
# In a few paragraphs, describe the story the data tells. 
# 
# Additionally, post your most compelling visual and provide a brief description of what it conveys on to our mutual aid channel (the slack course channel). 
# 
# Feel free to post more examples for people to look at and provide feedback. Your classmates will be vital providers of feedback in this process. Utilize them.












# # Final Step: Connect your workflow/process to the DSA-Project Life Cycle
# - List and briefly discuss how important details from each stage of the [DSA-PLC](../../module1/resources/DSA-ProjectLifecycle-slidedeck.pdf) played a role in your story development.
# - Use markdown to provide this overview below:
# <hr/>
# 
# <h1 align="center"><u>DSA-Project Life Cycle Discussion</u></h1>
# 
# 

# # Save your notebook, then `File > Close and Halt`
