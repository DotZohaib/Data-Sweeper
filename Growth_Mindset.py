import streamlit as st
import pandas as pd
import numpy as np
import io
import os
from docx import Document

# Set page configuration
st.set_page_config(page_title="Data Sweeper", layout="wide")
st.title("Data Sweeper")
st.write("This app helps clean, analyze, and convert your data.")

# Initialize session state for storing dataframes
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}

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
            doc = Document(file)
            text = [paragraph.text for paragraph in doc.paragraphs]
            return pd.DataFrame({'Text': text})
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

# Display uploaded files
if uploaded_files:
    selected_file = st.selectbox("Select a file to work with", list(st.session_state.datasets.keys()))
    df = st.session_state.datasets[selected_file]

    st.header(f"Working with: {selected_file}")
    st.write(f"**File size**: {file.size} bytes")
    
    # Show data preview
    with st.expander("Data Preview"):
        st.dataframe(df.head())

    # Data cleaning section
    st.subheader("Data Cleaning Options")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Remove Duplicates"):
            df = df.drop_duplicates()
            st.session_state.datasets[selected_file] = df
            st.success("Removed duplicates!")
    
    with col2:
        if st.button("Fill Missing Values"):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            st.session_state.datasets[selected_file] = df
            st.success("Filled numeric missing values with mean!")
    
    with col3:
        if st.button("Drop Columns with >50% NaN"):
            threshold = len(df) * 0.5
            df = df.dropna(thresh=threshold, axis=1)
            st.session_state.datasets[selected_file] = df
            st.success("Dropped columns with >50% missing values!")

    # Column selection
    st.subheader("Column Operations")
    selected_columns = st.multiselect("Select columns to keep", df.columns, default=list(df.columns))
    if selected_columns:
        df = df[selected_columns]
        st.session_state.datasets[selected_file] = df

    # Data visualization
    st.subheader("Data Visualization")
    if st.checkbox("Show Basic Visualizations"):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) >= 1:
            selected_col = st.selectbox("Select column for histogram", numeric_cols)
            st.bar_chart(df[selected_col].value_counts())
        else:
            st.warning("No numeric columns for visualization")

    # Data statistics
    with st.expander("Show Statistics"):
        st.write("Summary Statistics:")
        st.write(df.describe())
        st.write("Data Types:")
        st.write(df.dtypes.astype(str))

    # File conversion
    st.subheader("File Conversion")
    conversion_type = st.selectbox("Select output format", 
                                 ["CSV", "Excel", "Parquet", "JSON", "TXT"],
                                 key=f"conv_{selected_file}")
    
    buffer = io.BytesIO()
    if conversion_type == "CSV":
        df.to_csv(buffer, index=False)
        ext = ".csv"
    elif conversion_type == "Excel":
        df.to_excel(buffer, index=False)
        ext = ".xlsx"
    elif conversion_type == "Parquet":
        df.to_parquet(buffer, index=False)
        ext = ".parquet"
    elif conversion_type == "JSON":
        df.to_json(buffer, orient='records')
        ext = ".json"
    elif conversion_type == "TXT":
        df.to_csv(buffer, sep='\t', index=False)
        ext = ".txt"
    
    st.download_button(
        label=f"Download {conversion_type}",
        data=buffer,
        file_name=f"{os.path.splitext(selected_file)[0]}{ext}",
        mime="application/octet-stream"
    )

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
    2. Fill missing values (numeric columns)
    3. Drop columns with >50% missing values
    
    ### Visualization
    - Automatic histograms for numeric columns
    
    ### Conversion Options
    - Convert between CSV, Excel, JSON, TXT, and Parquet
    """)

# Add about section
st.sidebar.markdown("""
## About Data Sweeper
Version: 1.1  
Author: Data Analytics Team  
Features:
- Multi-file support
- Basic data cleaning
- Data visualization
- File format conversion
- Missing value handling
""")
