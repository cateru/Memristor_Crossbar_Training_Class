import pandas as pd

def load_conductance_data(
    file_path, 
    skip_rows=1, 
    separator=',', 
    column_names=[
        "Conductance (S)"
    ]
):
    """
    Loads and processes conductance data from a specified CSV or .dat file.

    Args:
        file_path (str): The path to the CSV or .dat file containing the conductance data.
        skip_rows (int): The number of rows to skip when reading the file. Default is 0.
        separator (str): The delimiter used in the file. Default is ','.
        column_names (list): A list of column names to assign to the data. Default is ["Conductance (S)"].

    Returns:
        np.ndarray: An array containing the conductance values read from the file.
    """
    experimental_data = pd.read_csv(
        file_path,
        skiprows=skip_rows,
        header=None,
        sep=separator,
        names=column_names,
        na_values=[""]
    )
    
    conductance_data = experimental_data["Conductance (S)"].to_numpy()

    return conductance_data
