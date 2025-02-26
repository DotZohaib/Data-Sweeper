import streamlit as st
import pandas as pd
import numpy as np
import io
import os

# Set page configuration
st.set_page_config(page_title="Data Sweeper", layout="wide")
st.title("Data Sweeper")
st.write("This app helps clean, analyze, and convert your data.")

# Initialize session state for storing dataframes
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'current_file' not in st.session_state:
    st.session_state.current_file = None

# Function to handle different file types
def load_data(file):
    file_ext = os.path.splitext(file.name)[-1].lower()
    
    try:
        if file_ext == ".csv":
            return pd.read_csv(file)
        elif file_ext == ".txt":
            return pd.read_csv(file, delimiter='\t')  # Adjust delimiter as needed
        elif file_ext == ".json":
            return pd.read_json(file)
        elif file_ext in [".xlsx", ".xls"]:
            return pd.read_excel(file)
        elif file_ext == ".parquet":
            return pd.read_parquet(file)
        elif file_ext == ".docx":
            try:
                doc = Document(file)
                text = [paragraph.text for paragraph in doc.paragraphs]
                return pd.DataFrame({'Text': text})
            except ImportError:
                st.error("python-docx package is required for .docx files. Install with 'pip install python-docx'")
                return None
        else:
            st.error(f"Unsupported file format: {file_ext}")
            return None
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

# File uploader
uploaded_files = st.file_uploader("Upload files", type=["csv", "txt", "json", "xlsx", "xls", "docx", "parquet"], accept_multiple_files=True)

# Process uploaded files
for file in uploaded_files:
    if file.name not in st.session_state.datasets:
        df = load_data(file)
        if df is not None:
            st.session_state.datasets[file.name] = df
            st.session_state.current_file = file.name

# Display uploaded files
if st.session_state.datasets:
    file_options = list(st.session_state.datasets.keys())
    selected_file = st.selectbox("Select a file to work with", file_options, index=file_options.index(st.session_state.current_file) if st.session_state.current_file in file_options else 0)
    st.session_state.current_file = selected_file
    df = st.session_state.datasets[selected_file]

    st.header(f"Working with: {selected_file}")
    
    # Find the file size for the selected file
    file_size = None
    for file in uploaded_files:
        if file.name == selected_file:
            file_size = file.size
            break
    
    if file_size:
        st.write(f"**File size**: {file_size} bytes")
    
    # Show data info
    with st.expander("Data Preview", expanded=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(df.head(10))
        with col2:
            st.write(f"Rows: {df.shape[0]}")
            st.write(f"Columns: {df.shape[1]}")
            st.write(f"Missing values: {df.isna().sum().sum()}")

    # Data cleaning section
    st.subheader("Data Cleaning Options")
    clean_col1, clean_col2, clean_col3 = st.columns(3)
    
    with clean_col1:
        if st.button("Remove Duplicates"):
            original_rows = len(df)
            df = df.drop_duplicates()
            removed = original_rows - len(df)
            st.session_state.datasets[selected_file] = df
            st.success(f"Removed {removed} duplicate rows!")
    
    with clean_col2:
        fill_method = st.selectbox("Fill Missing Values Method", ["Mean", "Median", "Mode", "Zero", "Custom"])
        if st.button("Fill Missing Values"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if fill_method == "Mean":
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                st.success("Filled numeric missing values with mean!")
            elif fill_method == "Median":
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                st.success("Filled numeric missing values with median!")
            elif fill_method == "Mode":
                for col in numeric_cols:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
                st.success("Filled numeric missing values with mode!")
            elif fill_method == "Zero":
                df[numeric_cols] = df[numeric_cols].fillna(0)
                st.success("Filled numeric missing values with zeros!")
            elif fill_method == "Custom":
                custom_value = st.number_input("Enter a custom value", value=0.0)
                df[numeric_cols] = df[numeric_cols].fillna(custom_value)
                st.success(f"Filled numeric missing values with {custom_value}!")
            
            st.session_state.datasets[selected_file] = df
    
    with clean_col3:
        threshold_percent = st.slider("Missing values threshold (%)", 0, 100, 50)
        if st.button("Drop Columns with High NaN"):
            threshold = len(df) * (threshold_percent / 100)
            original_cols = len(df.columns)
            df = df.dropna(thresh=threshold, axis=1)
            removed_cols = original_cols - len(df.columns)
            st.session_state.datasets[selected_file] = df
            st.success(f"Dropped {removed_cols} columns with >{threshold_percent}% missing values!")

    # Column operations
    st.subheader("Column Operations")
    col_ops_1, col_ops_2 = st.columns(2)
    
    with col_ops_1:
        selected_columns = st.multiselect("Select columns to keep", df.columns, default=list(df.columns))
        if st.button("Apply Column Selection") and selected_columns:
            df = df[selected_columns]
            st.session_state.datasets[selected_file] = df
            st.success(f"Kept {len(selected_columns)} columns!")
    
    with col_ops_2:
        rename_col = st.selectbox("Select column to rename", [""] + list(df.columns))
        new_name = st.text_input("New column name")
        if st.button("Rename Column") and rename_col and new_name:
            df = df.rename(columns={rename_col: new_name})
            st.session_state.datasets[selected_file] = df
            st.success(f"Renamed column '{rename_col}' to '{new_name}'!")

    # Data visualization
    st.subheader("Data Visualization")
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 1:
            selected_col = st.selectbox("Select column for histogram", [""] + list(numeric_cols))
            if selected_col:
                st.bar_chart(df[selected_col].value_counts())
        else:
            st.warning("No numeric columns for visualization")
    
    with viz_col2:
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        if len(categorical_cols) >= 1:
            selected_cat = st.selectbox("Select categorical column for pie chart", [""] + list(categorical_cols))
            if selected_cat and st.button("Generate Pie Chart"):
                value_counts = df[selected_cat].value_counts().head(10)  # Top 10 categories
                st.write(f"Top categories in {selected_cat}:")
                st.write(value_counts)
        else:
            st.warning("No categorical columns for visualization")

    # Data statistics
    with st.expander("Show Statistics"):
        st.write("Summary Statistics:")
        st.write(df.describe(include='all'))
        st.write("Data Types:")
        st.write(pd.DataFrame(df.dtypes, columns=['Data Type']))
        st.write("Missing Values by Column:")
        missing_data = pd.DataFrame({
            'Count': df.isna().sum(),
            'Percentage': (df.isna().sum() / len(df) * 100).round(2)
        })
        st.write(missing_data)

    # Data transformation
    st.subheader("Data Transformation")
    transform_col1, transform_col2 = st.columns(2)
    
    with transform_col1:
        transform_col = st.selectbox("Select numeric column to transform", [""] + list(numeric_cols))
        transform_method = st.selectbox("Transformation method", ["Log", "Square Root", "Z-Score", "Min-Max"])
        
        if st.button("Apply Transformation") and transform_col:
            try:
                if transform_method == "Log":
                    df[f"{transform_col}_log"] = np.log1p(df[transform_col])
                    st.success(f"Applied log transformation to {transform_col}")
                elif transform_method == "Square Root":
                    df[f"{transform_col}_sqrt"] = np.sqrt(df[transform_col])
                    st.success(f"Applied square root transformation to {transform_col}")
                elif transform_method == "Z-Score":
                    df[f"{transform_col}_zscore"] = (df[transform_col] - df[transform_col].mean()) / df[transform_col].std()
                    st.success(f"Applied Z-score normalization to {transform_col}")
                elif transform_method == "Min-Max":
                    min_val = df[transform_col].min()
                    max_val = df[transform_col].max()
                    df[f"{transform_col}_minmax"] = (df[transform_col] - min_val) / (max_val - min_val)
                    st.success(f"Applied min-max scaling to {transform_col}")
                
                st.session_state.datasets[selected_file] = df
            except Exception as e:
                st.error(f"Error during transformation: {str(e)}")
    
    with transform_col2:
        if len(categorical_cols) > 0:
            encode_col = st.selectbox("Select categorical column to encode", [""] + list(categorical_cols))
            encode_method = st.selectbox("Encoding method", ["One-Hot", "Label"])
            
            if st.button("Apply Encoding") and encode_col:
                try:
                    if encode_method == "One-Hot":
                        encoded = pd.get_dummies(df[encode_col], prefix=encode_col)
                        df = pd.concat([df, encoded], axis=1)
                        st.success(f"Applied one-hot encoding to {encode_col}")
                    elif encode_method == "Label":
                        df[f"{encode_col}_label"] = df[encode_col].astype('category').cat.codes
                        st.success(f"Applied label encoding to {encode_col}")
                    
                    st.session_state.datasets[selected_file] = df
                except Exception as e:
                    st.error(f"Error during encoding: {str(e)}")

    # File conversion
    st.subheader("File Conversion")
    conversion_type = st.selectbox("Select output format", 
                                 ["CSV", "Excel", "Parquet", "JSON", "TXT"],
                                 key=f"conv_{selected_file}")
    
    buffer = io.BytesIO()
    if conversion_type == "CSV":
        df.to_csv(buffer, index=False)
        ext = ".csv"
        mime = "text/csv"
    elif conversion_type == "Excel":
        df.to_excel(buffer, index=False)
        ext = ".xlsx"
        mime = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif conversion_type == "Parquet":
        df.to_parquet(buffer, index=False)
        ext = ".parquet"
        mime = "application/octet-stream"
    elif conversion_type == "JSON":
        df.to_json(buffer, orient='records')
        ext = ".json"
        mime = "application/json"
    elif conversion_type == "TXT":
        df.to_csv(buffer, sep='\t', index=False)
        ext = ".txt"
        mime = "text/plain"
    
    st.download_button(
        label=f"Download as {conversion_type}",
        data=buffer,
        file_name=f"{os.path.splitext(selected_file)[0]}{ext}",
        mime=mime
    )
else:
    st.info("Upload files to get started!")

# Add documentation section
with st.expander("App Documentation"):
    st.markdown("""
    ## Data Sweeper Features
    
    ### Supported File Types
    - CSV, TXT (tab-delimited), JSON
    - Excel (XLSX, XLS)
    - Word DOCX (text extraction)
    - Parquet
    
    ### Data Cleaning Options
    1. Remove duplicates
    2. Fill missing values (multiple methods)
    3. Drop columns with configurable threshold of missing values
    
    ### Data Transformation
    1. Numeric transformations (Log, Square Root, Z-Score, Min-Max)
    2. Categorical encoding (One-Hot, Label)
    
    ### Visualization
    - Histograms for numeric columns
    - Summary statistics for categorical columns
    
    ### Conversion Options
    - Convert between CSV, Excel, JSON, TXT, and Parquet
    
    ### Column Operations
    - Select/filter columns
    - Rename columns
    """)

# Add about section
st.sidebar.markdown("""
## About Data Sweeper
Version: 1.2  
Author: DotZohaib  
Features:
- Multi-file support
- Advanced data cleaning
- Data visualization
- Data transformation
- Statistical analysis
- File format conversion
- Missing value handling
- Categorical data encoding
""")
