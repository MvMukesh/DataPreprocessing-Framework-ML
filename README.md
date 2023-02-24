# ML-Data_Preprocessing
can say Wrangling(munging|transformation|manipulation) | Cleaning | Pre-processing | Feature Engineering

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

> `Step 1`: **`Define Problem`** <br>
 1.1. Gather data from problem domain <br>
 1.2. Discuss project with subject matter experts <br>
 1.3. Select those variables to be used as inputs and outputs for a predictive model <br>   
 1.4. Review data that has been collected <br>
 1.5. Summarize collected data using statistical methods <br>
 1.6. Visualize collected data using plots and charts <br>
  
> `Step 2`: **`Data Preparation [Tasks]`** <br>
Transform collected raw data as to make it more suitable for model  
 2.1. Data Cleaning <br>
 2.2. Feature Selection <br>
 2.3. Data Transformation <br>
 2.4. Feature Engineering  <br>
 2.5. Dimensionality Reduction <br>

> `Step 3`: **`Evaluate Models`** <br> 
 3.1. Select performance metric for model evalution<br>
 3.2. Select model evaluation procedure<br>
 3.3. Select algorithms to Evaluate<br>
 3.4. make a baseline to compare with other model<br>
 3.5. reshampling technique to split data <br>
    * k-fold is often used<br>
 3.6. get most out of well performing model by tuning<br>
    * Hyperparameters <br>
    * Combine predictive models into ensembles<br>

> `Step 4`: **`Finalize Model`**
  * slect best performing model
  * production and all

---

### `Data Prepration Techniques with Common Task's`
  
  ##### `2.1. Data Cleaning`: 
  - Identifying and correcting mistakes or errors in data | Data can be mistyped, corrupted, duplicated | Messy, noisy, corrupt, or erroneous values must be addressed
  - Might involve 
    - Removing a row or a column
    - Replacing observations with new values <br> 
  
  ##### `2.1.1. Common Data Cleaning Operations`:
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
 
 
##### `2.2. Feature Selection`:     
Identifying those input variables that are most relevant to target variable  <br>
Feature Selection technique is generally grouped into Supervised(having targets) and Unsupervised(not having targets) <br> 
* `Supervised Technique is further divided into Models that` 
  * `Intrinsic` : Automatically select features as part of model fitting [Trees]
  * `Wrapper Model` : Explicitly choose features which result in best performing model [Recursive Feature Elemination]
  * `Filter Model` : score each input feature and allow a subset to be selected [Feature Importance, Stats]
   * Statistical Methods such as 
      * Correlation is popular for Scoring Input features  <br>     
![image](https://user-images.githubusercontent.com/26667491/128865759-9dd31eeb-59c1-47d3-ba4b-3d62c3584fef.png) <br>

 ##### `2.2.1. Common Feature Selection Use Cases`:
  1. `Categorical Inputs` for a `Classification Target Variable`
  2. `Numerical Inputs` for a `Classification Target Variable` 
  3. `Numerical Inputs` for a `Regression Target Variable` 


`2.3. Data Transformation`: <br>
Changing scale,type or distribution of variables <br>
1. `Numeric Data Type`: Number values <br>
 * `Integer` : Integers with no fractional part <br>
 * `Float` : Floating point values <br>
2. `Categorical Data Type`: Label values <br>
 * `Ordinal` : Labels with a rank ordering <br>
 * `Nominal` : Labels with no rank ordering  <br>
 * `Boolean` : Values True and False <br>
![image](https://user-images.githubusercontent.com/26667491/128886257-de8213ea-43d3-4b2f-b0eb-37eadd67b59c.png) <br>

NOTE: Important consideration with data transforms is that operations are generally performed separately for each variable <br>
`Discretization Transform:` Encode (to convert) a numeric variable to an ordinal variable <br>
`Ordinal Transform:` Encode a categorical variable into an integer variable <br>
`One Hot Transform:` Encode a categorical variable into binary variables(boolean), required on most classification tasks <br>

If the data has a Gaussian probability distribution, it may be more useful to shift data to a standard Gaussian with a mean of zero and a standard deviation of one<br>
`Normalization Transform`: Scale a variable to range 0 and 1 <br>
`Standardization Transform`: Scale a variable to a standard Gaussian<br>
`Powe Transform:` Changes distribution of a variable which is nearly Gaussian, but is skewed or sifted can be made more Gaussian <br>
`Quantile Transform:` Force a probability distribution, such as Uniform or Gaussian on a variable with an unusual natural distribution <br>
![Overview of Data Transform Techniques](https://user-images.githubusercontent.com/26667491/128888984-288a0fa1-e1b8-40c0-9120-4871a41667f9.png) <br>



`2.4. Feature Engineering`: <br>
Deriving new variables from available data <br>
Technuques to reuse: <br>
1. Adding a boolean flag variable for some state
2. Adding a group or global summary statistic, such as a mean
3. Adding new variables for each component of a compound variable, such as a date-time <br>
`Polynomial Transform:`Create copies of numerical input variables that are raised to a power <br>


`2.5. Dimensionality Reduction`: <br>
Creating compact projections of data by creating projection of data in lower-dimensional that still preserves most important properties of original data  <br>
Common approach to dimensionality reduction is to use a `Matrix Factorization Technique` <br>
1. `Principal Component Analysis` (PCA)
2. `Singular Value Decomposition` (SVD)
`Model-based methods:` <br>
3. `Linear Discriminant Analysis`
4. `Autoencoders`
These techniques removes Linear Dependencies b/w input variables <br>
Sometimes `Manifold Learning Algorithms` can also be used  <br>
5. `Self-organizing maps` (SOME)
6. `t-Distributed Stochastic Neighbor Embedding` (t-SNE) <br>
![Dimensionality Reduction Techniques](https://user-images.githubusercontent.com/26667491/128892627-5b1b514e-ea6d-43e4-b01e-e7ca693b3757.png)

----

< ## `Data Processing without Leakage`
A naive approach to preparing data applying transform on entire dataset before evaluating performance of model <br> 
This results in a problem referred to as data leakage where knowledge of the hold-out test set leaks into dataset used to train model<br>
Careful application of data preprocessing is required depending on model evalution scheme used such as
1. `train-test-split`
2. `k-fold cross validation`
* Data processing must be done on Training set only in order to avoide data leakage <br>

**`Problem with Naive Data Processing`**
This could happen when test data is leaked into training set, or when data from future is leaked to past <br>
`For example` <br>
 * Consider case where we want to Normalize data, that is scale input variables to range 0-1 
 * When we normalize input variables, this requires that we first calculate minimum and maximum values for each variable before using these values to scale variables 
 * Dataset is then split into train and test datasets, but examples in training dataset know something about data in test dataset; they have been scaled by global minimum and maximum values, so they know more about the global distribution of variable then they should <br>

`Data preprocessing on train_test_split, processing must be fit on training dataset only` <br>
1. `Split Data`
2. `Fit Data Preparation on Training Dataset`
3. `Apply Data Preparation to Train and Test Datasets` 
4. `Evaluate Models` <br>

`Data preprocessing on k-Fold Cross Validation` <br>
Defines sequence(list) of Data Preparation Steps to apply by fitting model and evaluate it <br>
Each sequence(step) in list is a tuple having 2 element <br>
 1st element: name of step (string) <br>
 2nd element: configured object of step such as Transform or Model <br>

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




