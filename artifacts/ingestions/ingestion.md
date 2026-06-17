### TP1
0. Add the 'ipykernel' kernel
  ```
  python 
  uv add ipykernel
  ```
1. Anonymize the information in dataset:
   - Replace the values in the "operator_name" column with "ANONYMOUS"
   - Replace the values in the "operator_badge" column with "ANONYMOUS" 
   -  1. Load the CSV file
   -  2. Anonymize the 'operator_name' and 'operator_badge' columns 
   -  3. Replace with a fixed value (e.g., "ANONYMOUS")
   -  4. Create a directory with the date in YYYYMMDDHHMM format
   -  5. Create the directory if it does not exist
   -  6. Save the anonymized file in the directory

2. Set an output directory with a date in the format "YYYYMMDDHHMM"
   - Add libraries: 
     - datetime
     - os
   - Create the output directory in the correct format. 
3. Possible evolution: Add random information using the uuid library
4. Add library "matplotlib"
```
python
uv add matplotlib
```
5. Ask to générate graph by day.
   - Add as input environment variable file location [INPUT_DATA_DIR] 
6. Keep Graphs in same folder as csv input file
   - Use number of week for graph per week
   - Keep only last two digit of year
7. Histograms of incidents 
   - per signal (type of issue)
   - per machine
8. Correlation graph of incidents by signal (type of issue)
   - Add library 'seaborn'
     ```
     python
     uv add seaborn
     ```
   -  1. Configure paths and environment variables
   -  2. Name of the file to search for
   -  3. Search for the file in the input directory
   -  4. Load the CSV file
   -  5. Configure column names (adjust according to your CSV)
   -  6. Column containing date/time (e.g., "incident_datetime")
   -  7. Column containing the shift (e.g., "work_shift", "shift_type")
   -  8. Check if the columns exist
   -  9. Convert the date column to datetime
   -  10. Create output directory for graphs
   -  11. Generate the graphs
   -  12. --- Graph 1: Distribution by Day --- ['incidents_by_day.png'] 
   -  13. --- Graph 2: Distribution by Week --- ['incidents_by_week.png']
   -  14. --- Graph 3: Distribution by Shift --- ['incidents_by_shift.png']
   -  15. --- Graph 4: Histogram of Incidents ---
   -  16. By Machine (only valid issue) ['incidents_by_machine.png']
   -  17. --- Graph 5: Histogram of Incidents ---
   -  18. By Signal (only valid issue, type_* = 1) ['incidents_by_signal']
   -  19. Count incidents for each type_* column where value = 1
   -  20. Extract severity and type_* columns for valid issue
   -  21. Calculate the correlation matrix
   -  22. --- Plot the heatmap --- ['severity_correlation_by_signal.png']
   -  23. Display graphs-  

9. **Refactoring** :
    1.  Anonymize data
       1.  Define location of dataset as parameter : 
            - define environment variable : INPUT_FILE_PATH (with a default value)
        2. Define colomn to be anonymized as parameter :
            - define environment variable as table of element with a specific separator : ANONYMIZE_COLUMNS (default: "operator_name|operator_badge")
        3. Define output location as parameter :
           - define environment variable as  OUTPUT_DIR  
        4. Load csv file
        5. Anonymize specified column
           1. Replace each colun value by "ANONYMOUS"
        6. Create a directory with specific format YYYYMMDDHHMM
        7. Save data in OUTPUT_DIR.
    2. Generate graphics
       1. Configure paths and environment variables (INPUT_DATA_DIR)
       2. Define column names (adjust according to your CSV files) 
       3. Define function to parse csv file
       4. Define funciton to generate graph
       5. Main program : Process all CSV files in the input directory that ends with "_anonymised.csv"
          - For each csv
             1. Load the CSV file
             2. Check if the required columns exist
             3. Convert the date column to datetime
             4. Identify all "type_*" columns (types of issues)
             5. Filter rows where at least one "type_*" column is 1 (valid issue)
             6. Create output directory for graphs
             7. Generate Graphics
           - Display graph.