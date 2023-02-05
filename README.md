# Rule extraction using Genetic Algorithm
 
Extracting rules from a dataset using Genetic Algorithm for classification. 

chromosome representation is as follows:

    - Each chromosome's length is equal to the number of features in the dataset
    
    - Each part of the chromosome corresponds to a feature, with the same order as features in the dataset

    - For discrete features, the part corresponding to it in the chromosome, has two sub-parts. first is the limit value for that feature and second is a binary string. this binary string has the same length of the feature's domain and 1 means that this feature can take the value and 0 means that it can't take that value in the domain. 

    - For continuous features, the part corresponding to it in the chromosome, has three sub-parts. first is the limit value for that feature, second is the lower bound for it and third value is the upper bound for this feature. The condition on this feature is as follows: if lower bound < feature value < upper bound

    - limit parameter controls the presence of each feature in the rule represented by each chromosome. if limit value for feature i is greater or equal to the value of LIMIT parameter specified for algorithm, then the condition on feature i will be in the rule, otherwise it won't be in the rule.

    - the algorithm will extract rules for each class.

Here is a pseudocode for GARule function.

<img src="/pseudocode.png" alt="pseudocode of algorithm" title="pseudocode">

Below is some rules extracted from iris dataset, fitness column show each rule's accuracy.

<img src="/rules.jpg" alt="rules extracted from iris dataset." title="iris dataset.">