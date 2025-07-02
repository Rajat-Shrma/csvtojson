    import streamlit as st
    import pandas as pd
    import os
    import json
    import io
    import zipfile
    import tempfile
    import shutil
    import google.generativeai as genai
    import time
    import gc


    API_KEY = "AIzaSyAoC0cNyI_pHM793KDg-0NO6HEji1ZZIZc"
    genai.configure(api_key=API_KEY)
    st.set_page_config(page_title="CSV to JSON Converter", layout="wide")

    st.title("Medical Report CSV to JSON Converter")

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
        "Upload CSV annotation file",
        type="csv",
        accept_multiple_files=True
    )

    # Process files only if they have been uploaded
    if uploaded_files:
        if st.button("Process File"):
            # Create a temporary directory to store the generated JSON files.
            temp_dir = tempfile.mkdtemp()
            json_output_dir = os.path.join(temp_dir, "generated_jsons")
            os.makedirs(json_output_dir, exist_ok=True)

            st.info(f"Starting processing of {len(uploaded_files)} CSV files...")

            # Calculate the total number of rows across all CSVs for an accurate progress bar.
            total_expected_rows = 0
            for uploaded_file_temp in uploaded_files:
                try:
                    uploaded_file_temp.seek(0)
                    df_temp = pd.read_csv(uploaded_file_temp)
                    total_expected_rows += len(df_temp)
                    uploaded_file_temp.seek(0)
                except Exception as e:
                    st.warning(f"Could not pre-read {uploaded_file_temp.name} for row count: {e}. Skipping for total count.")
                    pass

            progress_bar = st.progress(0)
            total_rows_processed = 0
            all_json_files_for_zip = []

            # Initialize the Gemini model once outside the loop for efficiency
            model = genai.GenerativeModel('gemini-2.5-flash')

            # Iterate through each uploaded CSV file
            for uploaded_file in uploaded_files:
                st.subheader(f"Processing CSV: {uploaded_file.name}")
                try:
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
                        extracted_data = str(row['table']) # Ensure it's a string

                        st.info(f"Processing entry {index + 1}/{num_rows} for image: {image_name}")

                        prompt = f"""
                        I have provided you with the data extracted from a medical report. You need to structure it as a JSON object.
                        According to the following JSON schema.
                        extracted_data: {extracted_data}
                        JSON Schema: {json.dumps(json_template, indent=2)}"""

                        max_retries = 5
                        retry_delay = 2 # seconds
                        gemini_response_text = None

                        for attempt in range(max_retries):
                            try:
                                # Make the API call to the Gemini model
                                response = model.generate_content(contents=prompt)
                                gemini_response_text = response.text
                                st.success(f"Gemini API call successful for {image_name} (Attempt {attempt + 1}).")
                                break # Success, exit retry loop
                            except Exception as e:
                                st.warning(f"Attempt {attempt + 1}/{max_retries} for image {image_name}: Gemini API call failed: {e}")
                                if attempt < max_retries - 1:
                                    time.sleep(retry_delay * (2 ** attempt)) # Exponential backoff
                                else:
                                    st.error(f"Failed to get Gemini response for image {image_name} after {max_retries} attempts. Skipping this entry.")
                                    gemini_response_text = None # Ensure it's None if all retries fail

                        if gemini_response_text:
                            # Strip markdown code block fences
                            if gemini_response_text.startswith("```json") and gemini_response_text.endswith("```"):
                                gemini_response_text = gemini_response_text[len("```json"): -len("```")].strip()

                            try:
                                generated_json_data = json.loads(gemini_response_text)
                                generated_json_data["image_name"] = image_name

                                base_image_name = os.path.basename(image_name)
                                json_filename = os.path.join(json_output_dir, f"{os.path.splitext(base_image_name)[0]}.json")

                                with open(json_filename, 'w', encoding='utf-8') as f:
                                    json.dump(generated_json_data, f, indent=2)
                                all_json_files_for_zip.append(json_filename)
                                st.success(f"Successfully processed and saved: {os.path.basename(json_filename)}")

                            except json.JSONDecodeError:
                                st.error(f"Error: Gemini response for {image_name} was not valid JSON after stripping. "
                                         f"Response snippet: {gemini_response_text[:500]}...")
                            except Exception as e:
                                st.error(f"An unexpected error occurred while processing Gemini response for {image_name}: {e}")
                        else:
                            st.error(f"Skipping JSON generation for {image_name} due to unrecoverable API failures.")

                        # Explicitly delete variables to free up memory
                        del extracted_data
                        del gemini_response_text
                        if 'response' in locals():
                            del response # Delete the response object if it exists
                        gc.collect() # Trigger garbage collection

                        # Update the progress bar based on the total number of rows processed
                        total_rows_processed += 1
                        if total_expected_rows > 0:
                            progress_bar.progress(int((total_rows_processed / total_expected_rows) * 100))
                        else:
                            progress_bar.progress(0)

                except Exception as e:
                    st.error(f"Critical error reading or processing {uploaded_file.name}: {e}. Skipping this file.")

            st.success("All available files processed! Creating ZIP file...")

            # Create a zip file containing all the generated JSONs
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                for json_file in all_json_files_for_zip:
                    arcname = os.path.relpath(json_file, temp_dir)
                    zf.write(json_file, arcname)
            zip_buffer.seek(0)

            st.download_button(
                label="Download All JSONs as ZIP",
                data=zip_buffer,
                file_name="generated_medical_reports_jsons.zip",
                mime="application/zip"
            )

            shutil.rmtree(temp_dir)
            st.info("Temporary files cleaned up.")

    else:
        st.info("Please upload CSV files to begin the conversion process.")
    
