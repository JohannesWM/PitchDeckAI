import pandas
import re
from dotenv import load_dotenv
from google import genai
import os
from google.genai import types
import httpx
from google.genai.types import GenerateContentConfig

from constants import *
import json
import io
import pikepdf
import time
from collections import deque
from datetime import datetime, timedelta


def pdf_pages(doc_data) -> int:
    # Get PDF page count using pikepdf
    pdf_length = 0
    try:
        with io.BytesIO(doc_data) as pdf_stream:
            pdf = pikepdf.open(pdf_stream)
            pdf_length = len(pdf.pages)
            print(f"PDF Page Count: {pdf_length}")
    except Exception as e:
        print(f"Error getting PDF length: {e}")

    return pdf_length


def initialize_client() -> genai.Client:
    """
    This function gets the locally stored google api key, from '.env'.  It then in creates a gemini client
    and returns it.

    :return: genai.Client
    """
    load_dotenv()
    google_api = os.getenv("GOOGLE_API")  # take environment variables from .env.
    google_api_two = os.getenv("GOOGLE_API_TWO")
    google_api_three = os.getenv("GOOGLE_API_THREE")

    return genai.Client(api_key=google_api)


class RateLimiter:
    """
    Rate limiter for the Gemini API that enforces limits of:
    - 15 requests per minute
    - 1500 requests per day
    """
    def __init__(self, requests_per_minute=15, requests_per_day=1500):
        self.requests_per_minute = requests_per_minute
        self.requests_per_day = requests_per_day
        self.minute_requests = deque()
        self.day_requests = deque()
        print(f"Rate limiter initialized with {requests_per_minute} requests/minute and {requests_per_day} requests/day")

    def can_make_request(self):
        """Check if a request can be made based on rate limits"""
        current_time = datetime.now()

        # Clean up expired timestamps
        self._cleanup_timestamps(current_time)

        # Check if we're within the limits
        minute_ok = len(self.minute_requests) < self.requests_per_minute
        day_ok = len(self.day_requests) < self.requests_per_day

        if not minute_ok:
            print(f"Rate limit: Minute limit reached ({len(self.minute_requests)}/{self.requests_per_minute})")
        if not day_ok:
            print(f"Rate limit: Daily limit reached ({len(self.day_requests)}/{self.requests_per_day})")

        return minute_ok and day_ok

    def record_request(self):
        """Record a request timestamp"""
        current_time = datetime.now()
        self.minute_requests.append(current_time)
        self.day_requests.append(current_time)
        print(f"Request recorded. Minute: {len(self.minute_requests)}/{self.requests_per_minute}, Day: {len(self.day_requests)}/{self.requests_per_day}")

    def wait_until_allowed(self):
        """Wait until a request is allowed based on rate limits"""
        if self.can_make_request():
            return

        print("Rate limit reached. Waiting for next available slot...")
        while not self.can_make_request():
            # If we can't make a request, wait a bit and check again
            time.sleep(1)
        print("Rate limit cleared. Proceeding with request.")

    def _cleanup_timestamps(self, current_time):
        """Remove timestamps that are outside the time windows"""
        # Remove requests older than 1 minute
        minute_cutoff = current_time - timedelta(minutes=1)
        while self.minute_requests and self.minute_requests[0] < minute_cutoff:
            self.minute_requests.popleft()

        # Remove requests older than 1 day
        day_cutoff = current_time - timedelta(days=1)
        while self.day_requests and self.day_requests[0] < day_cutoff:
            self.day_requests.popleft()


def llm_to_df(llm_output: str, company_name: str) -> pandas.DataFrame | None:
    """
    Converts LLM output in JSON format to a pandas DataFrame

    :param llm_output: str
    :param company_name: str
    :return: pandas.DataFrame | None
    """
    if not llm_output:
        return None

    # Clean up the output by removing markdown code blocks
    clean_output = llm_output.replace("```json", "").replace("```", "").strip()

    try:
        # Parse the JSON
        json_data = json.loads(clean_output)
        company_dict = {"company_name": company_name}
        json_data = dict(company_dict, **json_data)
        # If json_data is a dictionary, wrap it in a list to create a single-row DataFrame
        if isinstance(json_data, dict):
            return pandas.DataFrame([json_data])
        # If json_data is already a list, create DataFrame directly
        elif isinstance(json_data, list):
            return pandas.DataFrame(json_data)
        else:
            print(f"Unexpected JSON structure: {type(json_data)}")
            return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        return None


def df_to_xlsx(df: pandas.DataFrame, file_path: str):
    """
    Appends a dataframe to an Excel file. If the file doesn't exist, it creates a new one.

    :param df: pandas DataFrame containing data to append
    :param file_path: Path to the Excel file
    :return: None
    """
    try:
        # Check if file exists
        if os.path.exists(file_path):
            # Read existing file
            existing_df = pandas.read_excel(file_path)
            # Append the new dataframe
            updated_df = pandas.concat([existing_df, df], ignore_index=True)
            # Write back to file
            updated_df.to_excel(file_path, index=False)
        else:
            # Create new file if it doesn't exist
            df.to_excel(file_path, index=False)

        print(f"Data successfully written to {file_path}")
    except Exception as e:
        print(f"Error writing to Excel file: {e}")


def main(model_name: str = "gemini-2.0-flash", num_companies: int = None, csv_file_path: str = "airtable-data/Deals-Pipeline View Clean.csv", company_name: str = None, file_path: str = "output.xlsx", system_instructions: str = ""):
        """
        This function runs the system. It opens the provided csv file and extracts each row from it.  It then
        loops through all the rows finding the pitch deck's links. Once these have been identified, it sends
        requests to them and passes them to a gemini model for further analysis.

        :param model_name: str
        :param num_companies: int
        :param csv_file_path: str
        :param company_name: str - If provided, processing starts from this company
        :param file_path: str
        :return: None
        """
        client = initialize_client()
        rate_limiter = RateLimiter()  # Initialize the rate limiter

        with open(csv_file_path) as airtable_f:
            df = pandas.read_csv(airtable_f)

        total_rows = df.shape[0]
        processed_count = 0

        # Determine starting index based on company_name
        start_index = 0
        if company_name is not None:
            company_rows = df[df["HiCap"] == company_name].index.tolist()
            if company_rows:
                start_index = company_rows[0]
                print(f"Starting processing from company '{company_name}' at index {start_index}")
            else:
                print(f"Company '{company_name}' not found in dataset. Starting from the beginning.")

        for row_i in range(start_index, total_rows):
            row = df.iloc[row_i]
            pitch = row["Attachments"]
            if pitch != "":
                pdf_string: str = str(pitch)  # select a pitch deck pdf string
                pattern = r"https?://[^\)]+"
                links = re.findall(pattern, pdf_string)  # parse through the string, find all links, if any
                HiCap = row["HiCap"]

                if links:
                    doc_url = links[len(links) - 1]
                    # Retrieve and encode the PDF byte
                    doc_data = httpx.get(doc_url).content

                    prompt = pdf_prompt

                    # Wait until we're allowed to make a request
                    print(f"Processing company {row_i+1}/{total_rows}: {HiCap}")
                    rate_limiter.wait_until_allowed()

                    try:
                        # Make the request
                        response = client.models.generate_content(
                            model=model_name,
                            contents=[
                                types.Part.from_bytes(
                                    data=doc_data,
                                    mime_type='application/pdf',
                                ),
                                prompt],
                            config=GenerateContentConfig(
                                system_instruction=[
                                    system_instructions,
                                ]
                            )
                        )
                    except Exception as e:
                        print(f"Error generating content for company {row_i+1}: {e}")
                        continue

                    # Record the request
                    rate_limiter.record_request()
                    processed_count += 1

                    response_df = llm_to_df(response.text, HiCap)
                    df_to_xlsx(response_df, file_path)

            if num_companies is not None:
                if processed_count >= num_companies:
                    break

        print(f"Completed processing {processed_count} companies with rate limiting applied.")