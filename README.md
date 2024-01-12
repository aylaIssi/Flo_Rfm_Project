# Flo_Rfm_Project
# Data Understanding and Preparation:

The code imports necessary libraries reads the dataset, and performs initial data exploration.
It creates new columns to represent total purchases and total expenses for omnichannel customers.
There is a bonus section that analyzes total expenses for different product categories.

## Calculating RFM Metrics:

The code calculates the Recency, Frequency, and Monetary (RFM) metrics for each customer.
The RFM metrics are based on the last order date, total number of purchases, and total customer expenses.

## Calculating RFM Scores:

The code discretizes the RFM metrics into quintiles and assigns scores accordingly.
These scores are used to create an RFM score for each customer.

## Defining RFM Score as a Segment:

The code defines customer segments based on the RFM scores, using predefined rules.
It assigns segment names such as 'hibernating,' 'loyal_customers,' etc.

## Definition of variables
**master_id:** Unique Customer Number <br/>
**order_channel:**  Which channel of the shopping platform is used (Android, ios, Desktop, Mobile)) <br/>
**last_order_channel:**  The channel where the most recent purchase was made <br/>
**first_order_date:**  The customer's first purchase date <br/>
**last_order_date:**  The customer's last purchase date <br/>
**last_order_date_online:**   The customer's last purchase date in online shopping platform <br/>
**last_order_date_offline:** The customer's last purchase date in offline shopping platform <br/>
**order_num_total_ever_online:** The customer's total purchases in online shopping platform <br/>
**order_num_total_ever_offline:**  The customer's total purchases in offline shopping platform <br/>
**customer_value_total_ever_offline:** The total expenditure by customer in offline shopping platform <br/>
**customer_value_total_ever_online:** The total expenditure by customer in online shopping platform <br/>
**interested_in_categories_12:** List of categories the customer has shopped in the last 12 months <br/>

## Examples
Two specific cases are discussed for targeting customers: targeting women shoe shoppers among champions and loyal customers, and applying a discount for men and children categories for specific customer segments.

## Additional Content:

There is an additional section analyzing total expenses for different product categories.
Suggestions:

It's important to handle missing values appropriately and check for any potential data quality issues.
Consider adding comments and docstrings to explain the purpose of functions and code blocks.
Make sure to provide clear documentation on how to interpret the results of the analysis and use the generated segments.
Verify the accuracy of the predefined rules for segmenting customers based on RFM scores.
Further analysis and visualizations could be added to enhance the understanding of customer behavior.


