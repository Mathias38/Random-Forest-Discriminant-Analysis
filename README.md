# Random-Forest-Discriminant-Analysis
Forest Discriminant Analysis can be use to find the cause of a problem in a large data set.

HOW TO USE :

The example.py file shows how to use the discrimant analysis function. Example.csv in the data folder shows the required data structure.

The example is from an imaginary coffe machine constructor. To build a coffee machine, three machines can be used. Each machine requires an operator. Three operator work in the company. The quality team found that some of the coffe machine are defectuous. We want to find which machine / operator is the cause of the defect. 

Running the example should print the variable importance ranking. The defect cause is MACHINE2;OPERATOR3. 

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Support function are aslo available :

in Functions file :

get_stat_df : Creates a statistical dataframe that can be use to understand the problem. 
drop_nan : Drop columns that have more than na_percent n/a

in RandomForestDA file :

get_ez_forest : Creates a summary data frame of the random forest model (for more details on the discriminant analysis result)
