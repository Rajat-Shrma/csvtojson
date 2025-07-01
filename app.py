
import streamlit as st
import pandas as pd
import os
import json
import io
import zipfile
import tempfile
import shutil
import google.generativeai as genai

# Configure the Google Generative AI API with the provided key.
# In a production application, it's recommended to use Streamlit secrets or environment variables
# instead of hardcoding the API key directly in the code.
API_KEY = "AIzaSyAJeh9r_aWHBp2WxdtLdqcEr62DFpxfJSI"
genai.configure(api_key=API_KEY)

# Set up the Streamlit page configuration
st.set_page_config(page_title="CSV to JSON Converter", layout="wide")

st.title("Medical Report CSV to JSON Converter")
st.write(
    "Upload your CSV annotation files. This app will iterate through each row, "
    "extract the 'table' data, use the Gemini API to structure it into JSON, "
    "and then provide a ZIP file containing all the generated JSONs."
)

# Define the JSON template as provided by the user.
# This template guides the Gemini model on the desired output structure.
json_template = {
    "image_name": "",
    "tables": [
        {
            "title": "Table Title Here (if known or inferred)",
            "columns": ["Column 1", "Column 2", "Column 3", "..."],
            "rows": [
                {
                    "Column 1": "Value 1",
                    "Column 2": "Value 2",
                    "Column 3": "Value 3"
                },
                {
                    "Column 1": "Value 4",
                    "Column 2": "Value 5",
                    "Column 3": "Value 6"
                }
            ]
        }
    ]
}

# Streamlit file uploader to accept multiple CSV files.
uploaded_files = st.file_uploader(
    "Upload CSV annotation files (select multiple files)",
    type="csv",
    accept_multiple_files=True
)

# Process files only if they have been uploaded
if uploaded_files:
    if st.button("Process Files"):
        # Create a temporary directory to store the generated JSON files.
        # This directory will be zipped and offered for download.
        temp_dir = tempfile.mkdtemp()
        json_output_dir = os.path.join(temp_dir, "generated_jsons")
        os.makedirs(json_output_dir, exist_ok=True)

        st.info(f"Processing {len(uploaded_files)} CSV files...")

        # Calculate the total number of rows across all CSVs for an accurate progress bar.
        total_expected_rows = 0
        for uploaded_file_temp in uploaded_files:
            try:
                # Reset file pointer before reading to ensure correct data loading
                uploaded_file_temp.seek(0)
                df_temp = pd.read_csv(uploaded_file_temp)
                total_expected_rows += len(df_temp)
                # Reset file pointer again for the main processing loop
                uploaded_file_temp.seek(0)
            except Exception as e:
                st.warning(f"Could not pre-read {uploaded_file_temp.name} for row count: {e}")
                # If pre-reading fails, the progress bar might be less accurate.
                pass

        progress_bar = st.progress(0)
        total_rows_processed = 0
        all_json_files_for_zip = []

        # Initialize the Gemini model once outside the loop for efficiency
        model = genai.GenerativeModel('gemini-2.5-flash') # CORRECTED: Initialize the model here

        # Iterate through each uploaded CSV file
        for uploaded_file in uploaded_files:
            st.subheader(f"Processing: {uploaded_file.name}")
            try:
                # Ensure the file pointer is at the beginning before reading
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file)

                # Validate essential columns
                if 'image' not in df.columns:
                    st.warning(f"Skipping {uploaded_file.name}: 'image' column not found. "
                               "Please ensure your CSV has an 'image' column for naming JSON files.")
                    continue
                if 'table' not in df.columns:
                    st.warning(f"Skipping {uploaded_file.name}: 'table' column not found. "
                               "Please ensure your CSV has a 'table' column with the data to be structured.")
                    continue

                num_rows = len(df)
                st.write(f"Found {num_rows} entries in {uploaded_file.name}.")

                # Iterate through each row in the DataFrame
                for index, row in df.iterrows():
                    image_name = row['image']
                    # Ensure extracted_data is a string for the prompt
                    extracted_data = str(row['table'])

                    # Construct the prompt for the Gemini model
                    prompt = f"""
                    I have provided you with the data extracted from a medical report. You need to structure it as a JSON object.
                    According to the following JSON schema.
                    extracted_data: {extracted_data}
                    JSON Schema: {json.dumps(json_template, indent=2)}"""

                    try:
                        # Make the API call to the Gemini model using the initialized model object
                        response = model.generate_content(contents=prompt) # CORRECTED: Call generate_content on the model object

                        responsetxt = response.text
                        # Attempt to parse the Gemini response as JSON
                        try:
                            generated_json_data = json.loads(responsetxt)
                            # Add the original image name to the generated JSON data
                            generated_json_data["image_name"] = image_name

                            # Determine the output JSON filename.
                            # Use os.path.basename to handle cases where 'image_name' might be a path.
                            base_image_name = os.path.basename(image_name)
                            json_filename = os.path.join(json_output_dir, f"{os.path.splitext(base_image_name)[0]}.json")

                            # Save the generated JSON to the temporary directory
                            with open(json_filename, 'w', encoding='utf-8') as f:
                                json.dump(generated_json_data, f, indent=2)
                            all_json_files_for_zip.append(json_filename)
                            st.success(f"Successfully processed and saved: {os.path.basename(json_filename)}")

                        except json.JSONDecodeError:
                            st.error(f"Error: Gemini response for {image_name} was not valid JSON. "
                                     f"Response snippet: {responsetxt[:500]}...")
                        except Exception as e:
                            st.error(f"An unexpected error occurred while processing Gemini response for {image_name}: {e}")

                    except Exception as e:
                        st.error(f"Gemini API call failed for image {image_name}: {e}")

                    # Update the progress bar based on the total number of rows processed
                    total_rows_processed += 1
                    if total_expected_rows > 0:
                        progress_bar.progress(int((total_rows_processed / total_expected_rows) * 100))
                    else:
                        progress_bar.progress(0) # Fallback if total_expected_rows is 0 (e.g., empty CSVs)

            except Exception as e:
                st.error(f"Error reading or processing {uploaded_file.name}: {e}")

        st.success("All files processed!")

        # Create a zip file containing all the generated JSONs
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for json_file in all_json_files_for_zip:
                # Add each JSON file to the zip archive, maintaining its relative path
                # within the 'generated_jsons' folder inside the zip.
                arcname = os.path.relpath(json_file, temp_dir)
                zf.write(json_file, arcname)
        zip_buffer.seek(0) # Rewind the buffer to the beginning for downloading

        # Provide a download button for the generated ZIP file
        st.download_button(
            label="Download All JSONs as ZIP",
            data=zip_buffer,
            file_name="generated_medical_reports_jsons.zip",
            mime="application/zip"
        )

        # Clean up the temporary directory and its contents
        shutil.rmtree(temp_dir)
        st.info("Temporary files cleaned up.")

else:
    st.info("Please upload CSV files to begin the conversion process.")
