import streamlit as st
import pandas as pd
import numpy as np
import os
import io
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from datetime import datetime

# Set the working directory to the root of the app
st.set_page_config(page_title="Data Sweeper", layout="wide")
st.title("Data Sweeper")
st.write("This app is designed to help you clean, analyze, and transform your data files.")

# Create a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload & Preview", "Clean & Transform", "Visualize", "Export"])

# Initialize session state to store dataframes
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None

# Function to read various file formats
def read_file(file):
    file_ext = os.path.splitext(file.name)[-1].lower()
    try:
        if file_ext == ".csv":
            return pd.read_csv(file)
        elif file_ext in [".txt", ".tsv"]:
            return pd.read_csv(file, sep='\t')
        elif file_ext == ".xlsx" or file_ext == ".xls":
            return pd.read_excel(file)
        elif file_ext == ".json":
            return pd.read_json(file)
        elif file_ext == ".parquet":
            return pd.read_parquet(file)
        elif file_ext == ".html":
            return pd.read_html(file.read().decode('utf-8'))[0]
        elif file_ext == ".xml":
            return pd.read_xml(file)
        else:
            st.error(f"Unsupported file format: {file_ext}")
            return None
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

# Upload & Preview page
if page == "Upload & Preview":
    st.header("Upload and Preview Data")

    # File uploader
    uploaded_files = st.file_uploader("Upload data files",
                                      type=["csv", "txt", "xlsx", "xls", "json", "parquet", "html", "xml", "tsv"],
                                      accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.dataframes:
                df = read_file(file)
                if df is not None:
                    st.session_state.dataframes[file.name] = df

        # Display file selector and information
        if st.session_state.dataframes:
            st.session_state.selected_file = st.selectbox("Select a file to preview:",
                                                         list(st.session_state.dataframes.keys()),
                                                         index=0 if st.session_state.selected_file is None else
                                                         list(st.session_state.dataframes.keys()).index(st.session_state.selected_file))

            selected_df = st.session_state.dataframes[st.session_state.selected_file]

            # File info
            st.subheader("File Information")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{selected_df.shape[0]:,}")
            with col2:
                st.metric("Columns", f"{selected_df.shape[1]:,}")
            with col3:
                file_size = os.path.getsize(st.session_state.selected_file) if os.path.exists(st.session_state.selected_file) else "Unknown"
                if isinstance(file_size, int):
                    file_size_str = f"{file_size/1024/1024:.2f} MB" if file_size > 1024*1024 else f"{file_size/1024:.2f} KB"
                else:
                    file_size_str = file_size
                st.metric("File Size", file_size_str)

            # Preview with options
            st.subheader("Data Preview")
            preview_type = st.radio("Preview type:", ["Head", "Tail", "Sample", "Full"])
            num_rows = st.slider("Number of rows:", 5, 100, 5)

            if preview_type == "Head":
                st.dataframe(selected_df.head(num_rows))
            elif preview_type == "Tail":
                st.dataframe(selected_df.tail(num_rows))
            elif preview_type == "Sample":
                st.dataframe(selected_df.sample(min(num_rows, len(selected_df))))
            else:
                if len(selected_df) > 1000:
                    st.warning("The dataset is large. Displaying only 1000 rows.")
                    st.dataframe(selected_df.iloc[:1000])
                else:
                    st.dataframe(selected_df)

            # Data Summary
            st.subheader("Data Summary")
            summary_options = st.multiselect("Select summary options:",
                                             ["Data Types", "Missing Values", "Descriptive Statistics", "Unique Values"],
                                             default=["Data Types", "Missing Values"])

            if "Data Types" in summary_options:
                st.write("Data Types:")
                dtypes_df = pd.DataFrame(selected_df.dtypes, columns=['Data Type'])
                dtypes_df.index.name = 'Column'
                dtypes_df = dtypes_df.reset_index()
                st.dataframe(dtypes_df)

            if "Missing Values" in summary_options:
                st.write("Missing Values:")
                missing_df = pd.DataFrame({
                    'Missing Values': selected_df.isnull().sum(),
                    'Percentage': selected_df.isnull().sum() / len(selected_df) * 100
                })
                missing_df.index.name = 'Column'
                missing_df = missing_df.reset_index()
                st.dataframe(missing_df)

            if "Descriptive Statistics" in summary_options:
                st.write("Descriptive Statistics:")
                st.dataframe(selected_df.describe(include='all').transpose())

            if "Unique Values" in summary_options:
                st.write("Unique Values Count:")
                unique_df = pd.DataFrame(selected_df.nunique(), columns=['Unique Count'])
                unique_df.index.name = 'Column'
                unique_df = unique_df.reset_index()
                st.dataframe(unique_df)

# Clean & Transform page
elif page == "Clean & Transform":
    st.header("Clean and Transform Data")

    if not st.session_state.dataframes:
        st.warning("Please upload data files first.")
    else:
        st.session_state.selected_file = st.selectbox("Select a file to clean:",
                                                     list(st.session_state.dataframes.keys()),
                                                     index=0 if st.session_state.selected_file is None else
                                                     list(st.session_state.dataframes.keys()).index(st.session_state.selected_file))

        selected_df = st.session_state.dataframes[st.session_state.selected_file]

        st.subheader("Data Cleaning Options")

        # Handling missing values
        st.write("#### Handle Missing Values")
        missing_col1, missing_col2 = st.columns(2)

        with missing_col1:
            missing_strategy = st.selectbox("Strategy for numeric columns:",
                                          ["No action", "Fill with mean", "Fill with median", "Fill with mode", "Fill with zero", "Drop rows"])

            if st.button("Apply to numeric columns"):
                numeric_cols = selected_df.select_dtypes(include=[np.number]).columns.tolist()

                if missing_strategy == "Fill with mean":
                    for col in numeric_cols:
                        selected_df[col] = selected_df[col].fillna(selected_df[col].mean())
                    st.success(f"Filled missing values in numeric columns with mean")

                elif missing_strategy == "Fill with median":
                    for col in numeric_cols:
                        selected_df[col] = selected_df[col].fillna(selected_df[col].median())
                    st.success(f"Filled missing values in numeric columns with median")

                elif missing_strategy == "Fill with mode":
                    for col in numeric_cols:
                        selected_df[col] = selected_df[col].fillna(selected_df[col].mode()[0] if not selected_df[col].mode().empty else 0)
                    st.success(f"Filled missing values in numeric columns with mode")

                elif missing_strategy == "Fill with zero":
                    selected_df[numeric_cols] = selected_df[numeric_cols].fillna(0)
                    st.success(f"Filled missing values in numeric columns with zero")

                elif missing_strategy == "Drop rows":
                    orig_len = len(selected_df)
                    selected_df.dropna(subset=numeric_cols, inplace=True)
                    st.success(f"Dropped {orig_len - len(selected_df)} rows with missing values in numeric columns")

                st.session_state.dataframes[st.session_state.selected_file] = selected_df

        with missing_col2:
            cat_missing_strategy = st.selectbox("Strategy for categorical columns:",
                                              ["No action", "Fill with mode", "Fill with 'Unknown'", "Drop rows"])

            if st.button("Apply to categorical columns"):
                cat_cols = selected_df.select_dtypes(include=['object', 'category']).columns.tolist()

                if cat_missing_strategy == "Fill with mode":
                    for col in cat_cols:
                        mode_val = selected_df[col].mode()[0] if not selected_df[col].mode().empty else "Unknown"
                        selected_df[col] = selected_df[col].fillna(mode_val)
                    st.success(f"Filled missing values in categorical columns with mode")

                elif cat_missing_strategy == "Fill with 'Unknown'":
                    selected_df[cat_cols] = selected_df[cat_cols].fillna("Unknown")
                    st.success(f"Filled missing values in categorical columns with 'Unknown'")

                elif cat_missing_strategy == "Drop rows":
                    orig_len = len(selected_df)
                    selected_df.dropna(subset=cat_cols, inplace=True)
                    st.success(f"Dropped {orig_len - len(selected_df)} rows with missing values in categorical columns")

                st.session_state.dataframes[st.session_state.selected_file] = selected_df

        # Remove duplicates
        st.write("#### Remove Duplicates")
        dup_col1, dup_col2 = st.columns(2)

        with dup_col1:
            subset_cols = st.multiselect("Select columns to consider for duplicates (empty = all columns):",
                                         options=selected_df.columns.tolist())
            keep_option = st.radio("Which duplicate to keep:", ["first", "last", "none"])

        with dup_col2:
            if st.button("Remove Duplicates"):
                orig_len = len(selected_df)
                subset_cols = subset_cols if subset_cols else None
                selected_df.drop_duplicates(subset=subset_cols, keep=keep_option, inplace=True)
                removed = orig_len - len(selected_df)
                st.success(f"Removed {removed} duplicate rows")
                st.session_state.dataframes[st.session_state.selected_file] = selected_df

        # Filter data
        st.write("#### Filter Data")
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            filter_column = st.selectbox("Select column to filter:", selected_df.columns.tolist())

            # Dynamically adjust filter options based on column type
            if filter_column:
                col_type = selected_df[filter_column].dtype

                if np.issubdtype(col_type, np.number):
                    min_val = float(selected_df[filter_column].min())
                    max_val = float(selected_df[filter_column].max())

                    filter_min, filter_max = st.slider(f"Range for {filter_column}:",
                                                      min_val, max_val, (min_val, max_val))

                    filter_condition = (selected_df[filter_column] >= filter_min) & (selected_df[filter_column] <= filter_max)

                else:  # Categorical
                    unique_vals = selected_df[filter_column].dropna().unique().tolist()
                    selected_vals = st.multiselect(f"Select values for {filter_column}:", unique_vals, default=unique_vals[:5] if len(unique_vals) > 5 else unique_vals)

                    filter_condition = selected_df[filter_column].isin(selected_vals)

        with filter_col2:
            if filter_column and st.button("Apply Filter"):
                filtered_df = selected_df[filter_condition]
                st.success(f"Filtered from {len(selected_df)} to {len(filtered_df)} rows")
                st.session_state.dataframes[st.session_state.selected_file] = filtered_df

        # Column operations
        st.write("#### Column Operations")
        col_ops_col1, col_ops_col2 = st.columns(2)

        with col_ops_col1:
            col_operation = st.selectbox("Select operation:",
                                       ["Select columns", "Rename columns", "Change data type"])

            if col_operation == "Select columns":
                selected_columns = st.multiselect("Choose columns to keep:", selected_df.columns.tolist(), default=selected_df.columns.tolist())

                if st.button("Keep Selected Columns"):
                    selected_df = selected_df[selected_columns]
                    st.success(f"Kept {len(selected_columns)} columns")
                    st.session_state.dataframes[st.session_state.selected_file] = selected_df

            elif col_operation == "Rename columns":
                st.write("Enter new names for columns:")
                rename_dict = {}

                for i, col in enumerate(selected_df.columns[:10]):  # Limit to first 10 for UI clarity
                    new_name = st.text_input(f"New name for '{col}':", value=col, key=f"rename_{i}")
                    if new_name != col:
                        rename_dict[col] = new_name

                if st.button("Rename Columns"):
                    if rename_dict:
                        selected_df = selected_df.rename(columns=rename_dict)
                        st.success(f"Renamed {len(rename_dict)} columns")
                        st.session_state.dataframes[st.session_state.selected_file] = selected_df
                    else:
                        st.info("No columns renamed")

            elif col_operation == "Change data type":
                col_to_change = st.selectbox("Select column to change type:", selected_df.columns.tolist())
                current_type = str(selected_df[col_to_change].dtype)

                new_type = st.selectbox("New data type:",
                                      ["int", "float", "str", "datetime", "category"],
                                      index=["int", "float", "str", "datetime", "category"].index(current_type) if current_type in ["int", "float", "str", "datetime", "category"] else 0)

                if st.button("Change Data Type"):
                    try:
                        if new_type == "int":
                            selected_df[col_to_change] = selected_df[col_to_change].astype(int)
                        elif new_type == "float":
                            selected_df[col_to_change] = selected_df[col_to_change].astype(float)
                        elif new_type == "str":
                            selected_df[col_to_change] = selected_df[col_to_change].astype(str)
                        elif new_type == "datetime":
                            selected_df[col_to_change] = pd.to_datetime(selected_df[col_to_change])
                        elif new_type == "category":
                            selected_df[col_to_change] = selected_df[col_to_change].astype('category')

                        st.success(f"Changed {col_to_change} to {new_type} type")
                        st.session_state.dataframes[st.session_state.selected_file] = selected_df
                    except Exception as e:
                        st.error(f"Error changing data type: {str(e)}")

        with col_ops_col2:
            # Create derived columns
            st.write("Create new column:")
            new_col_name = st.text_input("New column name:")

            operation_type = st.selectbox("Operation type:",
                                        ["Simple formula", "Text manipulation", "Date extraction"])

            if operation_type == "Simple formula":
                formula_cols = st.multiselect("Select columns for formula:", selected_df.select_dtypes(include=[np.number]).columns.tolist())
                formula_op = st.selectbox("Operation:", ["+", "-", "*", "/", "mean", "sum", "min", "max"])

                if st.button("Create Column") and new_col_name and formula_cols:
                    try:
                        if formula_op == "+":
                            selected_df[new_col_name] = selected_df[formula_cols].sum(axis=1)
                        elif formula_op == "-":
                            if len(formula_cols) == 2:
                                selected_df[new_col_name] = selected_df[formula_cols[0]] - selected_df[formula_cols[1]]
                            else:
                                st.error("Subtraction requires exactly 2 columns")
                        elif formula_op == "*":
                            result = selected_df[formula_cols[0]].copy()
                            for col in formula_cols[1:]:
                                result *= selected_df[col]
                            selected_df[new_col_name] = result
                        elif formula_op == "/":
                            if len(formula_cols) == 2:
                                selected_df[new_col_name] = selected_df[formula_cols[0]] / selected_df[formula_cols[1]]
                            else:
                                st.error("Division requires exactly 2 columns")
                        elif formula_op == "mean":
                            selected_df[new_col_name] = selected_df[formula_cols].mean(axis=1)
                        elif formula_op == "sum":
                            selected_df[new_col_name] = selected_df[formula_cols].sum(axis=1)
                        elif formula_op == "min":
                            selected_df[new_col_name] = selected_df[formula_cols].min(axis=1)
                        elif formula_op == "max":
                            selected_df[new_col_name] = selected_df[formula_cols].max(axis=1)

                        st.success(f"Created new column '{new_col_name}'")
                        st.session_state.dataframes[st.session_state.selected_file] = selected_df
                    except Exception as e:
                        st.error(f"Error creating column: {str(e)}")

            elif operation_type == "Text manipulation":
                text_col = st.selectbox("Select text column:", selected_df.select_dtypes(include=['object']).columns.tolist())
                text_op = st.selectbox("Text operation:", ["uppercase", "lowercase", "title case", "extract substring", "replace"])

                if text_op == "extract substring":
                    start_pos = st.number_input("Start position:", min_value=0, value=0)
                    end_pos = st.number_input("End position (leave 0 for end of string):", min_value=0, value=0)

                elif text_op == "replace":
                    find_str = st.text_input("Find:")
                    replace_str = st.text_input("Replace with:")

                if st.button("Create Text Column") and new_col_name and text_col:
                    try:
                        if text_op == "uppercase":
                            selected_df[new_col_name] = selected_df[text_col].str.upper()
                        elif text_op == "lowercase":
                            selected_df[new_col_name] = selected_df[text_col].str.lower()
                        elif text_op == "title case":
                            selected_df[new_col_name] = selected_df[text_col].str.title()
                        elif text_op == "extract substring":
                            if end_pos > 0:
                                selected_df[new_col_name] = selected_df[text_col].str[start_pos:end_pos]
                            else:
                                selected_df[new_col_name] = selected_df[text_col].str[start_pos:]
                        elif text_op == "replace":
                            selected_df[new_col_name] = selected_df[text_col].str.replace(find_str, replace_str)

                        st.success(f"Created new column '{new_col_name}'")
                        st.session_state.dataframes[st.session_state.selected_file] = selected_df
                    except Exception as e:
                        st.error(f"Error creating column: {str(e)}")

            elif operation_type == "Date extraction":
                date_col = st.selectbox("Select date column:", selected_df.columns.tolist())
                date_component = st.selectbox("Extract component:", ["year", "month", "day", "dayofweek", "quarter"])

                if st.button("Create Date Column") and new_col_name and date_col:
                    try:
                        # Ensure column is datetime
                        date_series = pd.to_datetime(selected_df[date_col])

                        if date_component == "year":
                            selected_df[new_col_name] = date_series.dt.year
                        elif date_component == "month":
                            selected_df[new_col_name] = date_series.dt.month
                        elif date_component == "day":
                            selected_df[new_col_name] = date_series.dt.day
                        elif date_component == "dayofweek":
                            selected_df[new_col_name] = date_series.dt.dayofweek
                        elif date_component == "quarter":
                            selected_df[new_col_name] = date_series.dt.quarter

                        st.success(f"Created new column '{new_col_name}'")
                        st.session_state.dataframes[st.session_state.selected_file] = selected_df
                    except Exception as e:
                        st.error(f"Error creating date column: {str(e)}")

        # Preview updated data
        st.subheader("Updated Data Preview")
        st.dataframe(selected_df.head())

# Visualize page
elif page == "Visualize":
    st.header("Data Visualization")

    if not st.session_state.dataframes:
        st.warning("Please upload data files first.")
    else:
        st.session_state.selected_file = st.selectbox("Select a file to visualize:",
                                                     list(st.session_state.dataframes.keys()),
                                                     index=0 if st.session_state.selected_file is None else
                                                     list(st.session_state.dataframes.keys()).index(st.session_state.selected_file))

        selected_df = st.session_state.dataframes[st.session_state.selected_file]

        st.subheader("Create Visualizations")

        # Choose visualization type
        viz_type = st.selectbox("Select visualization type:",
                               ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram",
                                "Box Plot", "Heatmap", "Pie Chart", "Correlation Matrix"])

        # Common settings for multiple chart types
        if viz_type in ["Bar Chart", "Line Chart", "Scatter Plot", "Histogram", "Box Plot", "Pie Chart"]:
            col1, col2 = st.columns(2)

            with col1:
                if viz_type == "Scatter Plot":
                    x_col = st.selectbox("X-axis:", selected_df.select_dtypes(include=[np.number]).columns.tolist())
                    y_col = st.selectbox("Y-axis:", [c for c in selected_df.select_dtypes(include=[np.number]).columns.tolist() if c != x_col])
                    color_col = st.selectbox("Color by (optional):", ["None"] + selected_df.columns.tolist())
                elif viz_type == "Histogram":
                    x_col = st.selectbox("Select column:", selected_df.select_dtypes(include=[np.number]).columns.tolist())
                    bins = st.slider("Number of bins:", 5, 100, 20)
                elif viz_type == "Pie Chart":
                    labels_col = st.selectbox("Labels:", selected_df.columns.tolist())
                    values_col = st.selectbox("Values:", selected_df.select_dtypes(include=[np.number]).columns.tolist())
                    max_slices = st.slider("Max number of slices:", 2, 15, 8)
                elif viz_type == "Box Plot":
                    y_col = st.selectbox("Values:", selected_df.select_dtypes(include=[np.number]).columns.tolist())
                    x_col = st.selectbox("Group by (optional):", ["None"] + selected_df.select_dtypes(include=['object', 'category']).columns.tolist())
                else:  # Bar and Line charts
                    x_col = st.selectbox("X-axis:", selected_df.columns.tolist())
                    y_col = st.selectbox("Y-axis:", selected_df.select_dtypes(include=[np.number]).columns.tolist())
                    group_by = st.selectbox("Group by (optional):", ["None"] + selected_df.columns.tolist())

            with col2:
                # Chart title and dimensions
                chart_title = st.text_input("Chart title:", f"{viz_type} of {st.session_state.selected_file}")
                width = st.slider("Width:", 400, 1200, 800)
                height = st.slider("Height:", 300, 1000, 500)

                # Color options
                color_theme = st.selectbox("Color theme:", ["blues", "viridis", "plasma", "inferno", "magma", "cividis"])

        # Heatmap specific settings
        elif viz_type == "Heatmap":
            numeric_cols = selected_df.select_dtypes(include=[np.number]).columns.tolist()
            heatmap_cols = st.multiselect("Select columns for heatmap:", numeric_cols, default=numeric_cols[:min(10, len(numeric_cols))])

            chart_title = st.text_input("Chart title:", f"Correlation Heatmap - {st.session_state.selected_file}")
            annot = st.checkbox("Show values", value=True)

            # Color options
            color_theme = st.selectbox("Color theme:", ["viridis", "plasma", "inferno", "magma", "cividis", "blues", "reds"])

        # Correlation Matrix specific settings
        elif viz_type == "Correlation Matrix":
            numeric_cols = selected_df.select_dtypes(include=[np.number]).columns.tolist()
            corr_cols = st.multiselect("Select columns for correlation:", numeric_cols, default=numeric_cols[:min(10, len(numeric_cols))])

            chart_title = st.text_input("Chart title:", f"Correlation Matrix - {st.session_state.selected_file}")
            corr_method = st.selectbox("Correlation method:", ["pearson", "kendall", "spearman"])

            # Color options
            color_theme = st.selectbox("Color theme:", ["coolwarm", "viridis", "plasma", "inferno", "magma", "cividis"])

        # Generate visualization
        if st.button("Generate Visualization"):
            st.subheader(chart_title)

            try:
                if viz_type == "Bar Chart":
                    if group_by != "None":
                        pivot_df = selected_df.pivot_table(index=x_col, columns=group_by, values=y_col, aggfunc='mean')
                        fig = px.bar(pivot_df, x=pivot_df.index, y=pivot_df.columns,
                                    title=chart_title, width=width, height=height,
                                    color_discrete_sequence=px.colors.sequential.Blues)
                    else:
                        fig = px.bar(selected_df, x=x_col, y=y_col, title=chart_title,
                                    width=width, height=height, color_discrete_sequence=px.colors.sequential.Blues)
                    st.plotly_chart(fig)

                elif viz_type == "Line Chart":
                    if group_by != "None":
                        pivot_df = selected_df.pivot_table(index=x_col, columns=group_by, values=y_col, aggfunc='mean')
                        fig = px.line(pivot_df, x=pivot_df.index, y=pivot_df.columns,
                                     title=chart_title, width=width, height=height,
                                     color_discrete_sequence=px.colors.sequential.Blues)
                    else:
                        fig = px.line(selected_df, x=x_col, y=y_col, title=chart_title,
                                     width=width, height=height, color_discrete_sequence=px.colors.sequential.Blues)
                    st.plotly_chart(fig)

                elif viz_type == "Scatter Plot":
                    color_arg = color_col if color_col != "None" else None
                    fig = px.scatter(selected_df, x=x_col, y=y_col, color=color_arg,
                                    title=chart_title, width=width, height=height,
                                    color_continuous_scale=color_theme)
                    st.plotly_chart(fig)

                elif viz_type == "Histogram":
                    fig = px.histogram(selected_df, x=x_col, nbins=bins,
                                      title=chart_title, width=width, height=height,
                                      color_discrete_sequence=px.colors.sequential.Blues)
                    st.plotly_chart(fig)

                elif viz_type == "Box Plot":
                    x_arg = x_col if x_col != "None" else None
                    fig = px.box(selected_df, y=y_col, x=x_arg,
                                title=chart_title, width=width, height=height,
                                color_discrete_sequence=px.colors.sequential.Blues)
                    st.plotly_chart(fig)
                elif viz_type == "Pie Chart":
                    # Group by labels and sum values
                    pie_data = selected_df.groupby(labels_col)[values_col].sum().reset_index()
                    # Take only top slices if needed
                    if len(pie_data) > max_slices:
                        top_data = pie_data.nlargest(max_slices-1, values_col)
                        others = pd.DataFrame({
                            labels_col: ['Others'],
                            values_col: [pie_data.iloc[max_slices-1:][values_col].sum()]
                        })
                        pie_data = pd.concat([top_data, others])

                    fig = px.pie(pie_data, names=labels_col, values=values_col,
                                title=chart_title, width=width, height=height,
                                color_discrete_sequence=px.colors.sequential.Blues)
                    st.plotly_chart(fig)

                elif viz_type == "Heatmap":
                    if heatmap_cols:
                        # Create correlation matrix
                        corr_df = selected_df[heatmap_cols].corr()

                        # Create heatmap with seaborn
                        fig, ax = plt.subplots(figsize=(width/100, height/100))
                        sns.heatmap(corr_df, annot=annot, cmap=color_theme, ax=ax)
                        ax.set_title(chart_title)
                        st.pyplot(fig)
                    else:
                        st.warning("Please select at least one column for the heatmap.")

                elif viz_type == "Correlation Matrix":
                    if corr_cols:
                        # Create correlation matrix
                        corr_df = selected_df[corr_cols].corr(method=corr_method)

                        # Create heatmap
                        fig = px.imshow(corr_df, text_auto=True, color_continuous_scale=color_theme,
                                      title=chart_title, width=width, height=height)
                        st.plotly_chart(fig)
                    else:
                        st.warning("Please select at least one column for the correlation matrix.")

            except Exception as e:
                st.error(f"Error generating visualization: {str(e)}")
                st.write("Troubleshooting tips:")
                st.write("- Check if the selected columns are appropriate for the visualization type")
                st.write("- Ensure there's enough data in the selected columns")
                st.write("- Try a different combination of columns")

# Export page
elif page == "Export":
    st.header("Export Data")

    if not st.session_state.dataframes:
        st.warning("Please upload data files first.")
    else:
        st.session_state.selected_file = st.selectbox("Select a file to export:",
                                                   list(st.session_state.dataframes.keys()),
                                                   index=0 if st.session_state.selected_file is None else
                                                   list(st.session_state.dataframes.keys()).index(st.session_state.selected_file))

        selected_df = st.session_state.dataframes[st.session_state.selected_file]

        st.subheader("Export Options")

        # Export format selection
        export_format = st.radio("Export format:",
                                ["CSV", "Excel", "JSON", "Parquet", "HTML", "SQLite", "Text (Tab separated)"])

        # Format specific options
        if export_format == "CSV":
            delimiter = st.selectbox("Delimiter:", [",", ";", "|", "Tab"])
            if delimiter == "Tab":
                delimiter = "\t"
            include_index = st.checkbox("Include index column", value=False)

            if st.button("Export to CSV"):
                buffer = BytesIO()
                selected_df.to_csv(buffer, index=include_index, sep=delimiter)
                buffer.seek(0)

                file_name = os.path.splitext(st.session_state.selected_file)[0] + ".csv"
                st.download_button("Download CSV file", buffer, file_name=file_name, mime="text/csv")

        elif export_format == "Excel":
            include_index = st.checkbox("Include index column", value=False)
            sheet_name = st.text_input("Sheet name:", "Sheet1")

            if st.button("Export to Excel"):
                buffer = BytesIO()
                selected_df.to_excel(buffer, index=include_index, sheet_name=sheet_name)
                buffer.seek(0)

                file_name = os.path.splitext(st.session_state.selected_file)[0] + ".xlsx"
                st.download_button("Download Excel file", buffer, file_name=file_name, mime="application/vnd.ms-excel")

        elif export_format == "JSON":
            orient_option = st.selectbox("JSON orientation:",
                                       ["records", "columns", "index", "split", "table"],
                                       index=0,
                                       help="'records' is a list of objects, 'columns' is a dict of lists, etc.")

            if st.button("Export to JSON"):
                buffer = BytesIO()
                selected_df.to_json(buffer, orient=orient_option)
                buffer.seek(0)

                file_name = os.path.splitext(st.session_state.selected_file)[0] + ".json"
                st.download_button("Download JSON file", buffer, file_name=file_name, mime="application/json")

        elif export_format == "Parquet":
            compression = st.selectbox("Compression:", ["none", "snappy", "gzip", "brotli"])
            compression = None if compression == "none" else compression

            if st.button("Export to Parquet"):
                buffer = BytesIO()
                selected_df.to_parquet(buffer, compression=compression)
                buffer.seek(0)

                file_name = os.path.splitext(st.session_state.selected_file)[0] + ".parquet"
                st.download_button("Download Parquet file", buffer, file_name=file_name, mime="application/octet-stream")

        elif export_format == "HTML":
            include_index = st.checkbox("Include index column", value=True)
            html_table_id = st.text_input("HTML table ID:", "data_table")
            html_classes = st.text_input("HTML table classes:", "table table-striped")
            include_bootstrap = st.checkbox("Include Bootstrap CSS", value=True)

            if st.button("Export to HTML"):
                # Create HTML content
                html_buffer = BytesIO()

                # Create HTML string
                html_string = f"""
                <html>
                <head>
                    <title>Data Sweeper Export - {st.session_state.selected_file}</title>
                    {"<link href='https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css' rel='stylesheet'>" if include_bootstrap else ""}
                    <style>
                        body {{ padding: 20px; }}
                        h1 {{ margin-bottom: 20px; }}
                    </style>
                </head>
                <body>
                    <h1>Data Sweeper Export - {st.session_state.selected_file}</h1>
                    <p>Exported on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    {selected_df.to_html(index=include_index, table_id=html_table_id, classes=html_classes)}
                </body>
                </html>
                """

                html_buffer.write(html_string.encode())
                html_buffer.seek(0)

                file_name = os.path.splitext(st.session_state.selected_file)[0] + ".html"
                st.download_button("Download HTML file", html_buffer, file_name=file_name, mime="text/html")

        elif export_format == "SQLite":
            table_name = st.text_input("Table name:", "data_table")
            if_exists = st.radio("If table exists:", ["replace", "append", "fail"])

            if st.button("Export to SQLite"):
                try:
                    # Create in-memory SQLite database
                    import sqlite3
                    from sqlalchemy import create_engine

                    buffer = BytesIO()
                    conn = sqlite3.connect(buffer)

                    # Write DataFrame to SQLite
                    selected_df.to_sql(table_name, conn, if_exists=if_exists, index=False)
                    conn.commit()
                    conn.close()

                    buffer.seek(0)
                    file_name = os.path.splitext(st.session_state.selected_file)[0] + ".db"
                    st.download_button("Download SQLite Database", buffer, file_name=file_name, mime="application/x-sqlite3")

                except Exception as e:
                    st.error(f"Error exporting to SQLite: {str(e)}")

        elif export_format == "Text (Tab separated)":
            include_header = st.checkbox("Include header", value=True)
            include_index = st.checkbox("Include index column", value=False)

            if st.button("Export to Text"):
                buffer = BytesIO()
                selected_df.to_csv(buffer, sep='\t', header=include_header, index=include_index)
                buffer.seek(0)

                file_name = os.path.splitext(st.session_state.selected_file)[0] + ".txt"
                st.download_button("Download Text file", buffer, file_name=file_name, mime="text/plain")

        # Preview of export data
        st.subheader("Export Preview")
        st.dataframe(selected_df.head())

        # Export metadata
        st.subheader("Export Metadata")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", f"{selected_df.shape[0]:,}")
        with col2:
            st.metric("Columns", f"{selected_df.shape[1]:,}")
        with col3:
            # Calculate approximate file size
            size_estimate = selected_df.memory_usage(deep=True).sum()
            size_str = f"{size_estimate/1024/1024:.2f} MB" if size_estimate > 1024*1024 else f"{size_estimate/1024:.2f} KB"
            st.metric("Estimated Size", size_str)

# Add footer
st.markdown("---")
st.markdown("Data Sweeper - A comprehensive data cleaning and analysis tool")
