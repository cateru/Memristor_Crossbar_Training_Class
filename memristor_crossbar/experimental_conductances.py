import pandas as pd

# Load experimental data from CSV file
#
# This section reads experimental data from a CSV file named 'SP291D1NR100.dat'.
# The data is read using pandas' `read_csv` function with the following parameters:
# - Skips the first 11 rows.
# - Uses the provided column names:
#   - 'Time stamp (s)'
#   - 'Voltage (V)'
#   - 'Current (A)'
#   - 'half_Voltage (V)'
#   - 'half_Current (A)'
#   - 'Resistance (Ohm)'
# - Considers empty values as NaNs.
#
# The data is stored in a DataFrame named `experimental_data`.
experimental_data = pd.read_csv(
    "SP291D1NR100.dat",
    skiprows=11,
    header=None,
    sep=",",
    names=[
        "Time stamp (s)",
        "Voltage (V)",
        "Current (A)",
        "half_Voltage (V)",
        "half_Current (A)",
        "Resistance (Ohm)",
    ],
    na_values=[""],
)
# Extract 'Voltage' and 'Resistance' columns as numpy arrays
#
# Converts the 'Voltage' and 'Resistance' columns from the DataFrame to numpy arrays for further processing.
Voltage = experimental_data["Voltage (V)"].to_numpy()
Resistance = experimental_data["Resistance (Ohm)"].to_numpy()
# Filter Resistance values based on Voltage
#
# Filters the `Resistance` array to include only those values where the corresponding `Voltage` is either 2 or -2.
Resistance = Resistance[(Voltage == 2) | (Voltage == -2)]
# Calculate conductance data
#
# Computes the conductance as the inverse of the filtered `Resistance` values. The resulting array `conductance_data`
# contains the conductance values corresponding to the filtered resistance measurements.
conductance_data = 1 / Resistance