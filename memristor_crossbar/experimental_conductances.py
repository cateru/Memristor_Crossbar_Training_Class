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
    Load and process conductance data from a specified CSV or .dat file.

    Parameters
    ----------
    file_path : str
        The path to the CSV or .dat file containing the conductance data.
    skip_rows : int, optional
        The number of rows to skip when reading the file. Default is 1.
    separator : str, optional
        The delimiter used in the file. Default is ','.
    column_names : list of str, optional
        A list of column names to assign to the data. Default is ["Conductance (S)"].

    Returns
    -------
    np.ndarray
        An array containing the conductance values read from the file.
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
