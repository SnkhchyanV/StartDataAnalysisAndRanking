This guide explains how to set up the project environment, install dependencies, and execute the main script with customizable parameters.

1. Create a Virtual Environment
To isolate dependencies, create a virtual environment by following these steps:
	1.	Open a terminal in the project directory.
	2.	Run the following command to create a virtual environment: `python -m venv venv`
	3.	Activate the virtual environment:
	-	On Windows: `venv\Scripts\activate`
	-	On macOS and Linux: `source venv/bin/activate`
2. Install Required Dependencies

With the virtual environment activated, install the necessary packages: `pip install -r requirements.txt`

3. Running the Script

To execute the script, use the following syntax in the terminal: `python main.py --data_path <your_data_path> --<parameter_name> <weight>`

Example Usage

`python main.py --data_path /Users/vladimir/Desktop/UW/Python\ Script\ Development\ for\ CSV\ Data\ Analysis/data.csv --Industry 1 --Mosaic 2 --Latest_Funding_Round 1`

In this example, data.csv is the data file to analyze, and weights are assigned to selected parameters. If a parameter weight is not specified, it will NOT considered in the composite score calculation.

4. Available Parameters

When specifying parameters in the command, replace spaces with underscores. Here is the full list of possible parameters:

PARAMETERS = {
    'Industry': 0,
    'Latest_Funding_Round': 0,
    'Latest_Funding_Date': 0,
    'Total_Funding': 0,
    'All_Investors': 0,
    'Latest_Valuation': 0,
    'Latest_Funding_Amount': 0,
    'Exit_Round': 0,
    'Exit_Date': 0,
    'Acquirers': 0,
    'Mosaic': 0,
    'Date_Added': 0,
    'Applications': 0,
}
