# ML-Data_Preprocessing
Also called as [ Wrangling(munging|transformation|manipulation) | Cleaning | Pre-processing | Feature Engineering ]

Data Preparation is most important part of a machine learning project which is least discussed and most time consuming
* machine learning algorithms have expectations regarding
  * data types 
  * scale 
  * probability distribution and 
  * relationships between input variables, data must be changed meet these expectations.
* Challenge of data processing is that each dataset is unique and different (data preprocessing is chalenging because of)
  * Datasets differ in number of variables (tens, hundreds, thousands, or more), 
  * Types of variables (numeric, nominal, ordinal, boolean), 
  * Scale of variables,
  * Drift in values over time etc...

---

Project can be different but steps on path to a good or even best result are generally same from project to project
  * Sometimes referred to as `applied machine learning process`, `data science process`
### `Applied ML Process`
 > `Step 1`: **`Define Problem`**
    
    * Gather data from problem domain
    * Discuss project with subject matter experts
    * Select those variables to be used as inputs and outputs for a predictive model   
    * Review data that has been collected
    * Summarize collected data using statistical methods
    * Visualize collected data using plots and charts
  
 > `Step 2`: **`Data Preparation [Tasks]`** 
Transform collected raw data as to make it more suitable for model <br> 
  2.1. Data Cleaning <br>
  2.2. Feature Selection <br>
  2.3. Data Transformation <br>
  2.4. Feature Engineering  <br>
  2.5. Dimensionality Reduction <br>

> `Step 3`: **`Evaluate Models`**

  * Select performance metric for model evalution
  * Select model evaluation procedure
  * Select algorithms to Evaluate
  * make a baseline to compare with other model
  * reshampling technique to split data 
    * k-fold is often used
  * get most out of well performing model by tuning
    * Hyperparameters 
    * Combine predictive models into ensembles

> `Step 4`: **`Finalize Model`**
  * slect best performing model
  * production and all

---
---

### `Step 2`: `Data Prepration Techniques with Common Task's`
  ##### 2.1.`Data Cleaning`: 
  - Identifying and correcting mistakes or errors in data
  - Data can be mistyped, corrupted, duplicated
  - Messy, noisy, corrupt, or erroneous values must be addressed
  - Might involve 
    - Removing a row or a column
    - Replacing observations with new values <br> 
  
  ##### 2.1.1`Common Data Cleaning Operations`:
  - Using statistics to 
    - Define Normal data and 
    - Identify Outliers
  - Identifying & Removing columns which have 
    - Same value or 
    - No variance
  - Identifying and Removing 
    - Duplicate rows of data
  - Marking empty values as missing
  - Imputing missing values using 
    - Statistics or 
    - A learned model
![Data Cleaning Overview](https://user-images.githubusercontent.com/26667491/128863018-c854c8d7-6344-4d57-9563-6d8a7369b85e.png) <br>
 
 
##### 2.2.`Feature Selection`:     
Identifying those input variables that are most relevant to target variable  <br>
Feature Selection technique is generally grouped into Supervised(having targets) and Unsupervised(not having targets) <br> 
* `Supervised Technique is further divided into Models that` 
  * `Intrinsic` : Automatically select features as part of model fitting [Trees]
  * `Wrapper Model` : Explicitly choose features which result in best performing model [Recursive Feature Elemination]
  * `Filter Model` : score each input feature and allow a subset to be selected [Feature Importance, Stats]
   * Statistical Methods such as 
      * Correlation is popular for Scoring Input features  <br>     
![image](https://user-images.githubusercontent.com/26667491/128865759-9dd31eeb-59c1-47d3-ba4b-3d62c3584fef.png) <br>

 ##### 2.2.1.`Common Feature Selection Use Cases`:
  1. `Categorical Inputs` for a `Classification Target Variable`
  2. `Numerical Inputs` for a `Classification Target Variable` 
  3. `Numerical Inputs` for a `Regression Target Variable` 


2.3. `Data Transformation`: 
  - Changing scale or distribution of variables

2.4. `Feature Engineering`: 
  - Deriving new variables from available data

2.5 `Dimensionality Reduction`: 
   - Creating compact projections of data

----
----

> I am trying to understand :
* Technique to `prepare data` so that it `avoids data leakage`, which lea to result of `incorrect model evaluation`
* Technique to `identify and handle problems with messy data`, such as `outliers and missing values`
* Technique to `identify and remove irrelevant and redundant input variables` with `feature selection methods`
* Technique to know which `feature selection method` to choose `based on data types of variables`
* Technique to `scale range of input variables` `using normalization and standardization` technique
* Technique to `encode categorical variables as numbers` and `numeric variables as categories`
* Technique to `transform probability distribution of input variables`
* Technique to `transform a dataset with different variable types` and how to `transform target variables`
* Technique to `project variables into a lower-dimensional space` to `captures salient data relationships`

----

`Part 1`: Basic
  * Importance of Data preparation and its techniques, best practices to use in order to avoid data leakage

`Part 2`: Data Cleaning 
  * Transform messy data into clean data by identifying outliers, handling missing values with statistical and modeling techniques
 
 `Part 3`: Feature Selection
  * Statistical and Modeling techniques for feature selection and feature importance with how to choose technique to use for different variable types.

`Part 4`: Data Transforms 
  * Transform variable types and variable probability distributions with a suite of standard data transform algorithms

`Part 5`: Advanced Transforms
  * Handle some of trickier aspects of data transforms, such as handling multiple variable types at once, transforming targets, and saving transforms after choosing a final model

`Part 6`: Dimensionality Reduction
  * Remove input variables by projecting data into a lower dimensional space with dimensionality-reduction algorithms

---




