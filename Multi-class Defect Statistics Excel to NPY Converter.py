import pandas as pd
import numpy as np
import os


def excel_to_onehot_npy(columns, excel_path, output_npy_path, num_images=359):
    """
    Convert defect labels from Excel to a (num_images, num_classes) one-hot encoded NumPy array and save as .npy.

    Parameters:
    - excel_path (str): Path to the Excel file (e.g., './same_zoom_selected_rename/缺陷统计.xlsx').
    - output_npy_path (str): Path to save the output .npy file (e.g., 'labels_6class.npy').
    - num_images (int): Expected number of images (default: 227).
    - num_classes (int): Number of defect classes (default: 6).

    Returns:
    - None: Saves the one-hot matrix to output_npy_path.
    """
    print(f"Reading Excel file: {excel_path}")

    if not os.path.exists(excel_path):
        print(f"Error: Excel file '{excel_path}' not found.")
        return

    # skip count row
    try:
        df = pd.read_excel(excel_path, sheet_name='Sheet1', header=1)
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    df = df.dropna(how='all').reset_index(drop=True)
    print(f"Excel data shape after removing count row: {df.shape}")
    print(f"Excel columns: {df.columns.tolist()}")
    print("First few rows of the Excel file:")
    print(df.head())

    if len(df) <= num_images:
        raise ValueError(f"Expected {num_images} images, but found {len(df)}-1 image rows in Excel.")
    elif len(df) > num_images:
        print(f"Warning: Found {len(df)}-1 image rows, using first {num_images} image rows.")
        df = df.iloc[:num_images]

    actual_columns = df.columns.tolist()
    required_columns = ['image id'] + columns
    if not all(col in actual_columns for col in required_columns):
        print(f"Actual columns: {actual_columns}")
        raise ValueError(f"Missing required columns in Excel. Expected: {required_columns}")

    image_ids = df['image id'].values
    expected_ids = np.arange(1, num_images + 1)
    if not np.array_equal(image_ids[:num_images], expected_ids):
        print(f"Expected IDs: {expected_ids[:5]}...{expected_ids[-5:]}")
        print(f"Actual IDs: {image_ids[:5]}...{image_ids[-5:]}")
        raise ValueError(f"Image IDs are not continuous or do not start from 1.")

    # one-hot
    onehot_labels = np.zeros((num_images, len(columns)), dtype=np.int32)
    for idx, col in enumerate(columns):
        # NaN = 0, else 1
        onehot_labels[:, idx] = df[col].notna().astype(np.int32)

    print(f"One-hot labels shape: {onehot_labels.shape}")
    print("one-hot labels:")
    for i in range(onehot_labels.shape[0]):
        print(f'image{i+1}:{onehot_labels[i]}')
    class_counts = np.sum(onehot_labels, axis=0)
    print(f"Class distribution: {class_counts}")

    # save as .npy file
    try:
        os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
        np.save(output_npy_path, onehot_labels)
    except Exception as e:
        raise RuntimeError(f"Error saving .npy file") from e
    print(f"Saved one-hot labels to: {output_npy_path}")


    try:
        loaded_labels = np.load(output_npy_path)
        if np.array_equal(loaded_labels, onehot_labels):
            print("Verification: Saved .npy file matches the generated labels.")
        else:
            raise ValueError(f"Saved .npy file does not match the generated labels.")
    except Exception as e:
        raise RuntimeError(f"Error verifying .npy file") from e


if __name__ == "__main__":
    columns = ['no_defect', 'dislocation', 'bridges', 'junction']
    excel_path = "./same_zoom_selected250812_rename/缺陷统计250903_no disclin no spot.xlsx"
    output_npy_path = "./same_zoom_selected250812_rename/labels_4class250903.npy"
    excel_to_onehot_npy(columns=columns, excel_path=excel_path, output_npy_path=output_npy_path)